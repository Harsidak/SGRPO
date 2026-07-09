"""
trainer/base_trainer.py

Shared training loop for all five algorithms.
The loss function and rollout generator are selected based on algorithm name.
Everything else is identical — optimizer, gradient clipping, logging, eval.

This is what makes the comparison scientifically valid:
one trainer, five loss functions, same everything else.

Research-grade features:
- Full metric computation per wandb_tracking_spec.md
- Evaluation loop on held-out test split
- Checkpoint saving with full state
- Histogram logging of key distributions
- Sample generation tables for qualitative inspection
- Timing instrumentation for throughput analysis
- Entropy and KL divergence computation
- Reference model for KL tracking
"""

import os
import re
import copy
import time
import random
import logging

import torch
from torch.optim import AdamW

from rewards.gsm8k_reward import compute_group_rewards
from tracking.wandb_logger import (
    init_run, log_step, log_eval, log_histograms,
    log_sample_table, log_checkpoint, log_degenerate_group, finish,
)

# Import loss modules (not wildcard — each exports compute())
from losses import ppo_loss, grpo_loss, dapo_loss, bapo_loss, sgrpo_loss

# Import rollout generators
from rollouts.base_rollout import generate_rollouts
from rollouts.sgrpo_rollout import generate_rollouts_isolated

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Single trainer for all five algorithms.
    Algorithm-specific behavior is isolated to:
    1. Which rollout generator is used (base vs isolated)
    2. Which loss function is called
    Everything else is shared.
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset,
                 algorithm: str, config, arch: str = "ssm"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.algorithm = algorithm
        self.config = config

        # Reward comes from the dataset (BaseRewardDataset interface) so the
        # trainer works on any verifiable benchmark. Plain datasets without
        # the interface fall back to the GSM8K reward for backward compat.
        self._train_reward_fn = getattr(
            train_dataset, "compute_group_rewards", compute_group_rewards
        )
        self._eval_reward_fn = getattr(
            eval_dataset, "compute_group_rewards", compute_group_rewards
        )
        self.arch = arch
        self.device = config.device
        self.step = 0

        # Select rollout generator
        # SGRPO uses state-isolated rollouts — this is the novel contribution.
        # Isolation only applies to architectures that carry recurrent state
        # across generations (SSM and hybrid SSM+attention). Transformers are
        # stateless between generate() calls, so SGRPO degenerates to GRPO —
        # which is exactly the paper's control condition.
        if algorithm == "sgrpo" and arch in ("ssm", "hybrid"):
            self.rollout_fn = generate_rollouts_isolated
        elif algorithm == "sgrpo":
            print("WARNING: SGRPO state isolation is unnecessary for "
                  "transformers — using standard rollouts (control condition).")
            self.rollout_fn = generate_rollouts
        else:
            self.rollout_fn = generate_rollouts

        # Optimizer — same for all algorithms
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Frozen reference model for the KL penalty term.
        # Required by RLHF PPO (InstructGPT), GRPO (DeepSeekMath), and BAPO.
        # DAPO removes it by design; SGRPO replaces it with Future-KL
        # weighting computed from stored rollout log-probs (no second model).
        if algorithm in ("ppo", "grpo", "bapo"):
            self._ref_model = copy.deepcopy(model)
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
            print(f"Reference model initialized for KL penalty "
                  f"(kl_coef={config.kl_coef})")
        else:
            self._ref_model = None

        # PPO advantage baseline. PPO runs with group_size=1, so
        # group-relative statistics are undefined (std of one sample is
        # NaN). Without a value critic, the standard critic-free choice is
        # an exponential moving average of past rewards as the baseline:
        # A = r - EMA(r).
        self._reward_baseline: float | None = None

        # Tracking state
        self._all_rewards = []       # for histogram logging
        self._all_advantages = []
        self._all_ratios = []
        self._all_response_lengths = []
        self._sample_buffer = []     # for sample generation tables

        # W&B init
        if not config.no_wandb:
            init_run(
                algorithm=algorithm,
                run_name=config.run_name,
                config=config.to_dict(),
                project=config.wandb_project,
            )

        # Checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        print(f"Trainer initialized: {algorithm.upper()} | arch: {arch}")
        print(f"Rollout generator: "
              f"{'isolated (SGRPO)' if self.rollout_fn is generate_rollouts_isolated else 'standard'}")

    def _compute_new_log_probs_and_entropy(
        self,
        generated_ids: torch.Tensor,    # [G, full_len]
        prompt_len: int,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute log probs under the CURRENT policy for generated sequences,
        plus policy entropy for collapse detection.

        Returns:
            new_log_probs: [G, gen_len]
            entropy:       scalar — average entropy across all generated positions
        """
        gen_len = generated_ids.shape[1] - prompt_len

        # Forward pass through current policy
        outputs = self.model(input_ids=generated_ids)
        logits = outputs.logits  # [G, full_len, vocab_size]

        # Log probs for generated tokens only
        gen_logits = logits[:, prompt_len - 1:-1, :]  # [G, gen_len, vocab_size]
        gen_ids = generated_ids[:, prompt_len:]        # [G, gen_len]

        log_probs = torch.log_softmax(gen_logits, dim=-1)

        # Gather log prob of the actual token at each position
        new_log_probs = log_probs.gather(
            2, gen_ids.unsqueeze(2)
        ).squeeze(2)  # [G, gen_len]

        # Compute entropy: H = -Σ p(x) log p(x)
        probs = torch.softmax(gen_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        return new_log_probs, entropy

    def _compute_ref_log_probs(
        self,
        generated_ids: torch.Tensor,    # [G, full_len]
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Compute log probs of the generated tokens under the FROZEN reference
        model. Used only by algorithms with a KL penalty (ppo/grpo/bapo).

        Returns:
            ref_log_probs: [G, gen_len], detached
        """
        with torch.no_grad():
            outputs = self._ref_model(input_ids=generated_ids)
            logits = outputs.logits  # [G, full_len, vocab_size]

            gen_logits = logits[:, prompt_len - 1:-1, :]
            gen_ids = generated_ids[:, prompt_len:]

            log_probs = torch.log_softmax(gen_logits, dim=-1)
            ref_log_probs = log_probs.gather(
                2, gen_ids.unsqueeze(2)
            ).squeeze(2)  # [G, gen_len]

        return ref_log_probs

    def _run_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ):
        """
        Call the correct loss function for the current algorithm.
        Returns (loss, advantages, metrics_dict) or None if degenerate group.

        All loss functions now return a metrics dict for WandB logging.
        """
        if self.algorithm == "ppo":
            # Critic-free baseline: EMA of past rewards (group stats are
            # undefined at group_size=1 — std of one sample is NaN)
            reward_mean = rewards.mean().item()
            if self._reward_baseline is None:
                self._reward_baseline = reward_mean
            advantages = rewards - self._reward_baseline
            self._reward_baseline = (
                0.9 * self._reward_baseline + 0.1 * reward_mean
            )
            loss, advantages, metrics = ppo_loss.compute(
                new_log_probs, old_log_probs, advantages, self.config.clip_eps,
                ref_log_probs=ref_log_probs, kl_coef=self.config.kl_coef,
            )
            metrics["ppo/reward_baseline"] = self._reward_baseline
            return loss, advantages, metrics

        elif self.algorithm == "grpo":
            result = grpo_loss.compute(
                new_log_probs, old_log_probs, rewards, self.config.clip_eps,
                ref_log_probs=ref_log_probs, kl_coef=self.config.kl_coef,
            )
            return result  # (loss, advantages, metrics)

        elif self.algorithm == "dapo":
            result = dapo_loss.compute(
                new_log_probs, old_log_probs, rewards,
                attention_mask, self.config.clip_eps
            )
            return result  # (loss, advantages, metrics) or None

        elif self.algorithm == "bapo":
            result = bapo_loss.compute(
                new_log_probs, old_log_probs, rewards,
                attention_mask, self.config.clip_eps,
                ref_log_probs=ref_log_probs, kl_coef=self.config.kl_coef,
            )
            return result  # (loss, advantages, metrics) or None

        elif self.algorithm == "sgrpo":
            result = sgrpo_loss.compute(
                new_log_probs, old_log_probs, rewards,
                attention_mask, self.config.clip_eps,
                self.config.future_kl_decay,
                self.config.future_kl_clip_low,
                self.config.future_kl_clip_high,
            )
            return result  # (loss, advantages, metrics) or None

    def _compute_response_stats(
        self,
        generated_texts: list[str],
        generated_ids: torch.Tensor,
        prompt_len: int,
    ) -> dict:
        """Compute response length and diversity statistics."""
        lengths = [len(text) for text in generated_texts]
        token_lengths = [
            (generated_ids[i, prompt_len:] != self.tokenizer.pad_token_id).sum().item()
            for i in range(generated_ids.shape[0])
        ]

        # Unique token ratio: vocabulary diversity metric
        all_gen_tokens = generated_ids[:, prompt_len:].flatten()
        real_tokens = all_gen_tokens[all_gen_tokens != self.tokenizer.pad_token_id]
        unique_ratio = (
            len(real_tokens.unique()) / max(len(real_tokens), 1)
        )

        return {
            "response_length_mean": sum(token_lengths) / max(len(token_lengths), 1),
            "response_length_std": (
                torch.tensor(token_lengths, dtype=torch.float32).std().item()
                if len(token_lengths) > 1 else 0.0
            ),
            "response_length_max": max(token_lengths) if token_lengths else 0,
            "response_length_min": min(token_lengths) if token_lengths else 0,
            "unique_tokens_ratio": unique_ratio,
            "raw_lengths": token_lengths,
        }

    def _evaluate(self, step: int) -> None:
        """
        Run evaluation on held-out GSM8K test split.
        Computes accuracy, reward statistics, and response quality metrics.
        """
        self.model.eval()
        eval_data = list(self.eval_dataset)
        random.shuffle(eval_data)
        eval_data = eval_data[:self.config.eval_samples]

        correct = 0
        total = 0
        all_rewards = []
        all_lengths = []
        reflection_count = 0
        self_correction_count = 0
        eval_samples = []

        with torch.no_grad():
            for item in eval_data:
                prompt = item["prompt"]
                answer = item["answer"]

                # Generate single greedy response for eval
                prompt_inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(self.device)

                output = self.model.generate(
                    prompt_inputs.input_ids,
                    attention_mask=prompt_inputs.attention_mask,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=False,  # greedy for eval
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                gen_text = self.tokenizer.decode(
                    output[0, prompt_inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                reward = self._eval_reward_fn([gen_text], answer)[0]
                all_rewards.append(reward)
                correct += int(reward > 0.5)
                total += 1

                gen_len = output.shape[1] - prompt_inputs.input_ids.shape[1]
                all_lengths.append(gen_len)

                # Reasoning quality heuristics
                if re.search(r"\b(wait|alternatively|let me reconsider)\b",
                             gen_text, re.IGNORECASE):
                    reflection_count += 1
                if re.search(r"\b(actually|correction|I made an error)\b",
                             gen_text, re.IGNORECASE):
                    self_correction_count += 1

                # Buffer samples for table logging
                if len(eval_samples) < 10:
                    eval_samples.append({
                        "prompt": prompt,
                        "response": gen_text,
                        "reward": reward,
                        "response_length": gen_len,
                    })

        accuracy = correct / max(total, 1)
        avg_reward = sum(all_rewards) / max(len(all_rewards), 1)

        import numpy as np
        lengths_arr = np.array(all_lengths)

        if not self.config.no_wandb:
            log_eval(
                step=step,
                algorithm=self.algorithm,
                gsm8k_accuracy=accuracy,
                average_reward=avg_reward,
                correct_count=correct,
                total_count=total,
                response_length_mean=float(lengths_arr.mean()),
                response_length_median=float(np.median(lengths_arr)),
                reasoning_steps_mean=None,
                reflection_count=reflection_count / max(total, 1),
                self_correction_rate=self_correction_count / max(total, 1),
            )

            # Log sample table
            if step % self.config.sample_table_every == 0:
                log_sample_table(step, eval_samples)

        print(f"  EVAL @ step {step}: accuracy={accuracy:.3f}, "
              f"avg_reward={avg_reward:.3f}, "
              f"correct={correct}/{total}")

        self.model.train()

    def _save_checkpoint(self, step: int) -> str:
        """Save model checkpoint and return the path."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.algorithm}_step{step}.pt"
        )
        torch.save({
            "step": step,
            "algorithm": self.algorithm,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")
        return path

    def train(self):
        """Main training loop with full research-grade instrumentation."""
        self.model.train()
        data = list(self.train_dataset)
        random.shuffle(data)

        accumulation_count = 0
        self.optimizer.zero_grad()

        for step in range(self.config.steps):
            step_start_time = time.time()
            # Track progress even when a step is skipped (degenerate group,
            # non-finite loss) so the final report reflects steps attempted.
            self.step = step

            # Sample a prompt
            item = data[step % len(data)]
            prompt = item["prompt"]
            answer = item["answer"]

            # ── Rollout phase ────────────────────────────────────────────
            gen_start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                rollout_data = self.rollout_fn(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    group_size=self.config.group_size,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    device=self.device,
                )
            generation_time = time.time() - gen_start_time

            # Compute rewards (via the dataset's verifiable reward)
            rewards_list = self._train_reward_fn(
                rollout_data["generated_texts"], answer
            )
            rewards = torch.tensor(
                rewards_list, dtype=torch.float32, device=self.device
            )

            # ── Early degenerate-group check ─────────────────────────────
            # DAPO/BAPO/SGRPO skip groups where all rewards are identical
            # (advantages would be zero). Checking here — before the
            # gradient forward pass — matters for two reasons:
            #   1. It saves an entire wasted forward over [G, full_len].
            #   2. Skipping AFTER the forward leaks the autograd graph:
            #      with no backward() to free it, the graph stays alive
            #      through the next step's forward, doubling peak VRAM
            #      (observed OOM on 6 GB with Mamba's slow path).
            if (self.algorithm in ("dapo", "bapo", "sgrpo")
                    and dapo_loss.is_degenerate_group(rewards)):
                if not self.config.no_wandb:
                    log_degenerate_group(step)
                if step % 10 == 0:
                    print(f"Step {step}: degenerate group "
                          f"(all same reward), skipping.")
                continue

            # ── Response statistics ──────────────────────────────────────
            generated_ids = rollout_data["generated_ids"].to(self.device)
            prompt_len = rollout_data["prompt_len"]
            gen_len = generated_ids.shape[1] - prompt_len

            response_stats = self._compute_response_stats(
                rollout_data["generated_texts"], generated_ids, prompt_len
            )

            # ── Optimization phase ───────────────────────────────────────
            self.model.train()

            old_log_probs = rollout_data["old_log_probs"].to(self.device)

            # Build attention mask for generated tokens
            attention_mask = (
                generated_ids[:, prompt_len:] != self.tokenizer.pad_token_id
            ).float()

            # Align old_log_probs length with gen_len
            if old_log_probs.shape[1] > gen_len:
                old_log_probs = old_log_probs[:, :gen_len]
            elif old_log_probs.shape[1] < gen_len:
                pad = torch.zeros(
                    old_log_probs.shape[0],
                    gen_len - old_log_probs.shape[1],
                    device=self.device,
                )
                old_log_probs = torch.cat([old_log_probs, pad], dim=1)

            # Forward pass — compute new log probs + entropy
            new_log_probs, entropy = self._compute_new_log_probs_and_entropy(
                generated_ids, prompt_len
            )

            # Reference log probs for KL penalty (ppo/grpo/bapo only)
            ref_log_probs = (
                self._compute_ref_log_probs(generated_ids, prompt_len)
                if self._ref_model is not None else None
            )

            # Compute loss
            result = self._run_loss(
                new_log_probs, old_log_probs, rewards, attention_mask,
                ref_log_probs=ref_log_probs,
            )

            if result is None:
                # Degenerate group — skip this step. Drop the references to
                # the forward pass so its autograd graph is freed NOW; with
                # no backward() coming, holding it through the next step's
                # forward doubles peak VRAM.
                del new_log_probs, entropy
                if not self.config.no_wandb:
                    log_degenerate_group(step)
                if step % 10 == 0:
                    print(f"Step {step}: degenerate group (all same reward), skipping.")
                continue

            loss, advantages, algo_metrics = result

            # Defensive guard: never let a non-finite loss reach the
            # optimizer — one NaN step corrupts the weights permanently.
            if not torch.isfinite(loss):
                print(f"Step {step}: non-finite loss ({loss.item()}), "
                      f"skipping update.")
                del loss, new_log_probs, entropy  # free autograd graph
                self.optimizer.zero_grad()
                accumulation_count = 0
                continue

            # ── Accumulate tracking data for histograms ──────────────────
            with torch.no_grad():
                ratio = torch.exp(
                    (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
                )
                self._all_rewards.append(rewards.detach().cpu())
                self._all_advantages.append(advantages.detach().cpu())
                self._all_ratios.append(ratio.detach().cpu().flatten())
                self._all_response_lengths.extend(
                    response_stats["raw_lengths"]
                )

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.grad_accum
            scaled_loss.backward()

            accumulation_count += 1

            if accumulation_count % self.config.grad_accum == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.max_grad_norm
                ).item()

                grad_was_clipped = grad_norm > self.config.max_grad_norm

                self.optimizer.step()
                self.optimizer.zero_grad()

                step_time = time.time() - step_start_time

                # Throughput: tokens processed per second
                total_tokens = (
                    self.config.group_size * gen_len
                )
                throughput = total_tokens / max(step_time, 1e-6)

                # ── WandB logging ────────────────────────────────────────
                if not self.config.no_wandb:
                    log_step(
                        step=step,
                        algorithm=self.algorithm,
                        loss=loss.item(),
                        policy_loss=loss.item(),
                        entropy=entropy,
                        clip_fraction=algo_metrics.get("train/clip_fraction"),
                        approx_kl=algo_metrics.get("train/approx_kl"),
                        ratio_mean=algo_metrics.get("train/ratio_mean"),
                        ratio_std=algo_metrics.get("train/ratio_std"),
                        ratio_max=algo_metrics.get("train/ratio_max"),
                        ratio_min=algo_metrics.get("train/ratio_min"),
                        reward_mean=rewards.mean().item(),
                        reward_std=rewards.std().item(),
                        reward_max=rewards.max().item(),
                        reward_min=rewards.min().item(),
                        advantage_mean=advantages.mean().item(),
                        advantage_std=advantages.std().item(),
                        advantage_max=advantages.max().item(),
                        advantage_min=advantages.min().item(),
                        positive_advantage_ratio=algo_metrics.get(
                            "train/positive_advantage_ratio"
                        ),
                        grad_norm=grad_norm,
                        grad_was_clipped=grad_was_clipped,
                        response_length_mean=response_stats["response_length_mean"],
                        response_length_std=response_stats["response_length_std"],
                        response_length_max=response_stats["response_length_max"],
                        response_length_min=response_stats["response_length_min"],
                        unique_tokens_ratio=response_stats["unique_tokens_ratio"],
                        step_time=step_time,
                        generation_time=generation_time,
                        throughput=throughput,
                        algo_metrics={
                            k: v for k, v in algo_metrics.items()
                            if not k.startswith("train/")
                        },
                    )

                    # Histogram logging
                    if step % self.config.histogram_every == 0 and step > 0:
                        if self._all_advantages:
                            log_histograms(
                                step=step,
                                advantages=torch.cat(self._all_advantages),
                                rewards=torch.cat(self._all_rewards),
                                ratios=torch.cat(self._all_ratios),
                                response_lengths=self._all_response_lengths.copy(),
                            )
                            # Reset buffers
                            self._all_rewards.clear()
                            self._all_advantages.clear()
                            self._all_ratios.clear()
                            self._all_response_lengths.clear()

                # ── Console output ───────────────────────────────────────
                if step % 10 == 0:
                    print(
                        f"Step {step:4d} | Loss: {loss.item():.4f} "
                        f"| Reward: {rewards.mean().item():.3f} "
                        f"| Adv: {advantages.mean().item():.3f} "
                        f"| Entropy: {entropy:.2f} "
                        f"| GradNorm: {grad_norm:.3f} "
                        f"| {step_time:.1f}s"
                    )

            # ── Periodic evaluation ──────────────────────────────────────
            if (step > 0 and step % self.config.eval_every == 0):
                self._evaluate(step)

            # ── Periodic checkpoint ──────────────────────────────────────
            if (step > 0 and step % self.config.checkpoint_every == 0):
                ckpt_path = self._save_checkpoint(step)
                if not self.config.no_wandb:
                    log_checkpoint(step, self.algorithm, ckpt_path)

        # ── Final eval and checkpoint ────────────────────────────────────
        print(f"\nTraining complete. Final step: {self.step}")
        self._evaluate(self.step)
        ckpt_path = self._save_checkpoint(self.step)
        if not self.config.no_wandb:
            log_checkpoint(self.step, self.algorithm, ckpt_path)
            finish()
