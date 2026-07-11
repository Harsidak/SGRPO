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
    init_run, init_local_sink, log_rollout, log_step, log_eval,
    log_histograms, log_sample_table, log_checkpoint,
    log_degenerate_group, log_run_metadata, finish,
)

# Import loss modules (not wildcard — each exports compute())
from losses import ppo_loss, grpo_loss, dapo_loss, bapo_loss, sgrpo_loss

# Import rollout generators
from rollouts.base_rollout import generate_rollouts
from rollouts.sgrpo_rollout import generate_rollouts_isolated

# Architecture detection is authoritative for rollout routing (see __init__).
from models.load_model import detect_architecture

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
                 algorithm: str, config, arch: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.algorithm = algorithm
        self.config = config

        # ── Architecture detection (authoritative) ───────────────────────
        # Rollout routing MUST be driven by the model's actual architecture,
        # never by a caller-supplied string or the algorithm name. A stale or
        # missing `arch` would route a stateless transformer into the SSM-only
        # isolated generator, whose _snapshot_ssm_states() raises when it finds
        # no recurrent state. Detecting here (before accelerator.prepare(), so
        # the module is still bare and unwrapped) makes every entry point
        # correct even if it never passed `arch`. The optional `arch` argument
        # is kept for backward compatibility but the detected value wins.
        detected_arch = detect_architecture(model)
        if arch is not None and arch != detected_arch:
            print(f"Note: caller passed arch={arch!r} but the model is "
                  f"{detected_arch!r} — using detected value for routing.")

        # Reward comes from the dataset (BaseRewardDataset interface) so the
        # trainer works on any verifiable benchmark. Plain datasets without
        # the interface fall back to the GSM8K reward for backward compat.
        self._train_reward_fn = getattr(
            train_dataset, "compute_group_rewards", compute_group_rewards
        )
        self._eval_reward_fn = getattr(
            eval_dataset, "compute_group_rewards", compute_group_rewards
        )
        self.arch = detected_arch
        self.step = 0

        # ── Optional multi-GPU via HuggingFace Accelerate ────────────────
        # When enabled (accelerate launch ... --use_accelerate), Accelerate
        # owns device placement, DDP wrapping, and mixed precision. When
        # disabled, everything below reduces to plain single-device PyTorch
        # so local runs are bit-identical to the pre-Accelerate trainer.
        if config.use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator(
                mixed_precision=(
                    "bf16" if config.dtype == "bfloat16" else "no"
                ),
            )
            self.device = str(self.accelerator.device)
        else:
            self.accelerator = None
            self.device = config.device

        # Select rollout generator
        # SGRPO uses state-isolated rollouts — this is the novel contribution.
        # Isolation only applies to architectures that carry recurrent state
        # across generations (SSM and hybrid SSM+attention). Transformers are
        # stateless between generate() calls, so SGRPO degenerates to GRPO —
        # which is exactly the paper's control condition.
        if algorithm == "sgrpo" and self.arch in ("ssm", "hybrid"):
            self.rollout_fn = generate_rollouts_isolated
            self.arch_branch = "isolated"
            print(f"SGRPO: architecture={self.arch}, using isolated rollouts "
                  f"(SSM state isolation active).")
        elif algorithm == "sgrpo":
            self.rollout_fn = generate_rollouts
            self.arch_branch = "standard"
            print(f"SGRPO: architecture={self.arch}, using standard rollouts "
                  f"(state isolation not applicable — degenerates to GRPO, "
                  f"the paper's control condition).")
        else:
            self.rollout_fn = generate_rollouts
            self.arch_branch = "standard"

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
        # Deepcopy happens BEFORE accelerator.prepare() so the reference is
        # the bare (non-DDP-wrapped) module.
        if algorithm in ("ppo", "grpo", "bapo"):
            self._ref_model = copy.deepcopy(model).to(self.device)
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
            print(f"Reference model initialized for KL penalty "
                  f"(kl_coef={config.kl_coef})")
        else:
            self._ref_model = None

        # Under Accelerate the model comes back DDP-wrapped: gradient
        # forward/backward must go through self.model, but generation
        # (rollouts, eval) needs the bare module — DDP does not expose
        # generate() and adds no value inside torch.no_grad().
        if self.accelerator is not None:
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer
            )
            self._gen_model = self.accelerator.unwrap_model(self.model)
        else:
            self._gen_model = self.model

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

        # Inner optimization epochs (PPO-style rollout-batch reuse). mu == 1
        # is the legacy single-pass path; mu > 1 steps the optimizer after
        # every inner epoch so theta moves between passes. See train().
        self._inner_epochs = max(1, int(getattr(config, "inner_epochs", 1)))
        self._global_update = 0      # monotonic optimizer-step counter

        # W&B init — main process only under multi-GPU
        if not config.no_wandb and self._is_main:
            init_run(
                algorithm=algorithm,
                run_name=config.run_name,
                config=config.to_dict(),
                project=config.wandb_project,
            )
        # Local JSONL metrics sink — always on (main process), so every run
        # (including --no_wandb smoke tests) leaves a structured metrics
        # record that Experiments/local_benchmark.py can graph offline.
        if self._is_main:
            local_path = init_local_sink(algorithm, config.run_name)
            print(f"Local metrics log: {local_path}")
            # One-time: record architecture + rollout branch (control vs
            # treatment) so both sinks capture the condition for each run.
            log_run_metadata(algorithm, self.arch, self.arch_branch)

        # Checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        if self._is_main:
            print(f"Trainer initialized: {algorithm.upper()} | arch: {self.arch}")
            print(f"Rollout generator: "
                  f"{'isolated (SGRPO)' if self.rollout_fn is generate_rollouts_isolated else 'standard'}")
            print(f"Inner epochs (mu): {self._inner_epochs}"
                  + ("" if self._inner_epochs > 1 else "  (legacy single-pass;"
                     " ratio/Future-KL are trivial at mu=1)"))
            if self.accelerator is not None:
                print(f"Accelerate: {self.accelerator.num_processes} "
                      f"process(es), device {self.device}, "
                      f"mixed_precision={self.accelerator.mixed_precision}")

    @property
    def _is_main(self) -> bool:
        """True on the main process (always True without Accelerate)."""
        return self.accelerator.is_main_process if self.accelerator else True

    def _zero_backward(self, anchor: torch.Tensor) -> None:
        """
        Backward a zero-valued surrogate through the existing forward graph.

        Under DDP every rank must run backward every microbatch, or the
        gradient all-reduce collectives desynchronize (hang) and the
        grad-accum counters drift apart (silent weight divergence). A rank
        that skips a step (degenerate group, non-finite loss) therefore
        contributes an explicit zero gradient instead of skipping backward.

        nan_to_num guards the non-finite-loss case: 0 * inf = NaN, so the
        anchor must be sanitized before the zero-multiply.
        """
        self.accelerator.backward(anchor.nan_to_num().sum() * 0.0)

    def _clip_grads(self) -> float:
        """Clip gradients, via Accelerate when active (handles DDP/AMP)."""
        if self.accelerator is not None:
            return self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            ).item()
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.max_grad_norm
        ).item()

    def _step_if_boundary(self, accumulation_count: int) -> None:
        """Optimizer step at the grad-accum boundary (skip-path variant —
        the normal path steps inline where it also logs)."""
        if accumulation_count % self.config.grad_accum == 0:
            self._clip_grads()
            self.optimizer.step()
            self.optimizer.zero_grad()

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
                new_log_probs, old_log_probs, rewards, attention_mask,
                c_low_min=self.config.bapo_c_low_min,
                c_low_max=self.config.bapo_c_low_max,
                c_high_min=self.config.bapo_c_high_min,
                c_high_max=self.config.bapo_c_high_max,
                delta_high=self.config.bapo_delta_high,
                delta_low=self.config.bapo_delta_low,
                rho_target=self.config.bapo_rho_target,
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

    def _evaluate(self, step: int) -> dict:
        """
        Run evaluation on the held-out test split.
        Computes accuracy, reward statistics, and response quality metrics.
        Returns the headline metrics so callers (benchmark runner) can
        aggregate across runs.
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

                output = self._gen_model.generate(
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

        if self._is_main:
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

        return {
            "step": step,
            "accuracy": accuracy,
            "average_reward": avg_reward,
            "correct": correct,
            "total": total,
            "response_length_mean": float(lengths_arr.mean()),
        }

    def _save_checkpoint(self, step: int) -> str:
        """Save model checkpoint and return the path.

        The filename includes run_name so benchmark runs of the same
        algorithm with different seeds don't overwrite each other.
        Optimizer state is optional (config.save_optimizer_state) — the
        benchmark disables it because its checkpoints exist for test-set
        evaluation, not resuming, and AdamW state triples the file size.
        """
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.algorithm}_{self.config.run_name}_step{step}.pt"
        )
        checkpoint = {
            "step": step,
            "algorithm": self.algorithm,
            # Bare-module weights — loadable without DDP/Accelerate
            "model_state_dict": self._gen_model.state_dict(),
            "config": self.config.to_dict(),
        }
        if getattr(self.config, "save_optimizer_state", True):
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
        return path

    def train(self):
        """Main training loop with full research-grade instrumentation."""
        self.model.train()
        data = list(self.train_dataset)
        random.shuffle(data)

        accumulation_count = 0
        self.optimizer.zero_grad()

        # Under multi-GPU each process trains on a disjoint slice of the
        # shuffled prompt stream (stride = world size); DDP averages the
        # gradients, so one optimizer step consumes world_size prompts.
        rank = self.accelerator.process_index if self.accelerator else 0
        world = self.accelerator.num_processes if self.accelerator else 1

        for step in range(self.config.steps):
            step_start_time = time.time()
            # Track progress even when a step is skipped (degenerate group,
            # non-finite loss) so the final report reflects steps attempted.
            self.step = step

            # Sample a prompt
            item = data[(step * world + rank) % len(data)]
            prompt = item["prompt"]
            answer = item["answer"]

            # ── Rollout phase ────────────────────────────────────────────
            gen_start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                rollout_data = self.rollout_fn(
                    # Bare module: rollout generators call generate()/custom
                    # decode loops, which DDP wrappers don't expose. Shares
                    # parameters with self.model, so it's the current policy.
                    model=self._gen_model,
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

            # ── Response statistics ──────────────────────────────────────
            # Computed BEFORE the degenerate check (cheap token counting) so
            # rollout-phase metrics are logged for every step, including the
            # skipped ones.
            generated_ids = rollout_data["generated_ids"].to(self.device)
            prompt_len = rollout_data["prompt_len"]
            gen_len = generated_ids.shape[1] - prompt_len

            response_stats = self._compute_response_stats(
                rollout_data["generated_texts"], generated_ids, prompt_len
            )

            is_degenerate = (
                self.algorithm in ("dapo", "bapo", "sgrpo")
                and dapo_loss.is_degenerate_group(rewards)
            )

            # ── Rollout-phase RL metrics — logged EVERY step ─────────────
            # log_step only fires on optimizer updates; early in training
            # nearly every group is degenerate, so without this the W&B run
            # shows little beyond GPU/system charts. rollout/* keeps reward
            # signal, response lengths, and the degenerate flag visible
            # from step 0.
            if self._is_main:
                log_rollout(
                    step=step,
                    algorithm=self.algorithm,
                    reward_mean=rewards.mean().item(),
                    reward_std=rewards.std().item() if rewards.numel() > 1 else 0.0,
                    reward_max=rewards.max().item(),
                    reward_min=rewards.min().item(),
                    reward_positive_fraction=(rewards > 0).float().mean().item(),
                    response_length_mean=response_stats["response_length_mean"],
                    response_length_std=response_stats["response_length_std"],
                    response_length_max=response_stats["response_length_max"],
                    response_length_min=response_stats["response_length_min"],
                    unique_tokens_ratio=response_stats["unique_tokens_ratio"],
                    generation_time=generation_time,
                    group_size=self.config.group_size,
                    is_degenerate=is_degenerate,
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
            # Under Accelerate this early exit is disabled: skipping before
            # the forward would leave this rank with no backward to
            # contribute, desynchronizing DDP. The degenerate group instead
            # falls through to _run_loss -> None, where _zero_backward keeps
            # the collectives in lockstep. (Multi-GPU nodes have the VRAM
            # headroom the early exit exists to save.)
            if self.accelerator is None and is_degenerate:
                if self._is_main:
                    log_degenerate_group(step)
                if step % 10 == 0:
                    print(f"Step {step}: degenerate group "
                          f"(all same reward), skipping.")
                continue

            # ── Optimization phase ───────────────────────────────────────
            # Inner optimization epochs (PPO-style rollout-batch reuse):
            #
            #   rollouts are generated ONCE per outer step above (old_log_probs
            #   fixed at rollout-time theta). We then re-optimize that same
            #   batch `mu` times. old_log_probs stays fixed across all mu inner
            #   passes; only new_log_probs is recomputed each pass because
            #   theta has moved. At inner_epoch 0 theta == theta_old, so
            #   ratio == 1 and Future-KL == 0 (identical to the single-pass
            #   trainer); from inner_epoch 1 onward theta has been stepped, so
            #   the importance ratio and Future-KL become non-trivial — the
            #   entire point of Fix 3.
            #
            # Composition with grad_accum (they are orthogonal in intent but
            # both gate the optimizer step, so the reconciliation is explicit,
            # never silent):
            #   - mu == 1 (default): LEGACY path. Gradients accumulate across
            #     `grad_accum` consecutive PROMPTS; one optimizer step per
            #     grad_accum prompts. Bit-identical to the pre-Fix-3 trainer.
            #   - mu > 1: theta MUST move between inner epochs or the reuse is
            #     pointless (new_log_probs would equal old_log_probs), so the
            #     optimizer steps after EVERY inner epoch and grad_accum's
            #     cross-prompt accumulation is not applied (each pass divides by
            #     1 and steps). The step count is visible in the console/W&B via
            #     sys/global_update and sys/inner_epoch.
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

            # Reference log probs for KL penalty (ppo/grpo/bapo only). The
            # reference model is frozen, so ref_log_probs are theta-independent
            # — compute ONCE and reuse across all inner epochs.
            ref_log_probs = (
                self._compute_ref_log_probs(generated_ids, prompt_len)
                if self._ref_model is not None else None
            )

            mu = self._inner_epochs
            step_every_pass = mu > 1
            loss_divisor = 1.0 if step_every_pass else self.config.grad_accum

            for inner_epoch in range(mu):
                # Forward pass under the CURRENT policy (theta may have been
                # stepped by a previous inner epoch).
                new_log_probs, entropy = self._compute_new_log_probs_and_entropy(
                    generated_ids, prompt_len
                )

                result = self._run_loss(
                    new_log_probs, old_log_probs, rewards, attention_mask,
                    ref_log_probs=ref_log_probs,
                )

                # ── Skip paths: degenerate group or non-finite loss ──────
                # Single-GPU: dapo/bapo/sgrpo degenerate groups are already
                # filtered before this loop, so `result is None` here only
                # happens under Accelerate (early exit disabled to keep DDP
                # collectives aligned). Non-finite losses are guarded for all.
                bad_reason = None
                if result is None:
                    bad_reason = "degenerate"
                elif not torch.isfinite(result[0]):
                    bad_reason = "non-finite"

                if bad_reason is not None:
                    if self.accelerator is not None:
                        # Zero contribution keeps DDP in lockstep; every rank
                        # runs exactly mu passes with an identical step cadence.
                        self._zero_backward(new_log_probs)
                        accumulation_count += 1
                        if step_every_pass or (
                            accumulation_count % self.config.grad_accum == 0
                        ):
                            self._clip_grads()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self._global_update += 1
                    else:
                        # Single-GPU: free the graph, take no step. For mu == 1
                        # this reproduces the legacy non-finite reset exactly.
                        self.optimizer.zero_grad()
                        if not step_every_pass:
                            accumulation_count = 0
                    if bad_reason == "degenerate":
                        if self._is_main:
                            log_degenerate_group(step)
                        if step % 10 == 0 and self._is_main:
                            print(f"Step {step}: degenerate group "
                                  f"(all same reward), skipping.")
                    elif self._is_main:
                        print(f"Step {step} (inner {inner_epoch}): non-finite "
                              f"loss, skipping update.")
                    del new_log_probs, entropy
                    if result is not None:
                        del result
                    continue

                loss, advantages, algo_metrics = result

                # ── Accumulate tracking data for histograms ──────────────
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

                # Scale loss (grad_accum for the legacy mu==1 path; 1 when
                # stepping every inner pass).
                scaled_loss = loss / loss_divisor
                if self.accelerator is not None:
                    self.accelerator.backward(scaled_loss)
                else:
                    scaled_loss.backward()

                accumulation_count += 1
                do_step = step_every_pass or (
                    accumulation_count % self.config.grad_accum == 0
                )

                if do_step:
                    # Gradient clipping
                    grad_norm = self._clip_grads()
                    grad_was_clipped = grad_norm > self.config.max_grad_norm

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._global_update += 1

                    step_time = time.time() - step_start_time

                    # Throughput: tokens processed per second
                    total_tokens = (
                        self.config.group_size * gen_len
                    )
                    throughput = total_tokens / max(step_time, 1e-6)

                    # Per-inner-epoch diagnostics ride along in the algo-metric
                    # passthrough so Future-KL's signal (zero on inner_epoch 0,
                    # nonzero afterwards) is visible in W&B.
                    passthrough = {
                        k: v for k, v in algo_metrics.items()
                        if not k.startswith("train/")
                    }
                    passthrough["sys/inner_epoch"] = inner_epoch
                    passthrough["sys/global_update"] = self._global_update

                    # ── Metrics logging: W&B + local JSONL (main only) ───
                    if self._is_main:
                        log_step(
                            step=step,
                            algorithm=self.algorithm,
                            loss=loss.item(),
                            policy_loss=algo_metrics.get(
                                f"{self.algorithm}/policy_loss", loss.item()
                            ),
                            kl_loss=algo_metrics.get(
                                f"{self.algorithm}/kl_penalty"
                            ),
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
                            algo_metrics=passthrough,
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

                    # ── Console output ───────────────────────────────────
                    if step % 10 == 0 and self._is_main:
                        if mu > 1:
                            # Richer line surfaces the per-inner-epoch signal
                            # that Fix 3 activates.
                            clipf = algo_metrics.get("train/clip_fraction")
                            clip_tag = (f" | ClipFrac: {clipf:.3f}"
                                        if clipf is not None else "")
                            sgrpo_tag = ""
                            if self.algorithm == "sgrpo":
                                sgrpo_tag = (
                                    f" | w_std: "
                                    f"{algo_metrics.get('sgrpo/influence_weight_std', 0.0):.2e}"
                                )
                            print(
                                f"Step {step:4d} [inner {inner_epoch+1}/{mu}] "
                                f"| Loss: {loss.item():.4f} "
                                f"| Reward: {rewards.mean().item():.3f} "
                                f"| Adv: {advantages.mean().item():.3f} "
                                f"| Entropy: {entropy:.2f} "
                                f"| GradNorm: {grad_norm:.3f}"
                                f"{clip_tag}{sgrpo_tag} "
                                f"| {step_time:.1f}s"
                            )
                        else:
                            # Legacy single-pass console line (unchanged).
                            print(
                                f"Step {step:4d} | Loss: {loss.item():.4f} "
                                f"| Reward: {rewards.mean().item():.3f} "
                                f"| Adv: {advantages.mean().item():.3f} "
                                f"| Entropy: {entropy:.2f} "
                                f"| GradNorm: {grad_norm:.3f} "
                                f"| {step_time:.1f}s"
                            )

                # Free this inner pass's forward graph before the next pass /
                # next step so peak VRAM does not grow with mu.
                del new_log_probs, entropy, loss, result

            # ── Periodic evaluation (main process only) ──────────────────
            if (step > 0 and step % self.config.eval_every == 0
                    and self._is_main):
                self._evaluate(step)

            # ── Periodic checkpoint (main process only) ──────────────────
            if (step > 0 and step % self.config.checkpoint_every == 0
                    and self._is_main):
                ckpt_path = self._save_checkpoint(step)
                if not self.config.no_wandb:
                    log_checkpoint(step, self.algorithm, ckpt_path)

        # ── Final eval and checkpoint (main process only) ─────────────────
        final_eval = None
        final_ckpt_path = None
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        if self._is_main:
            print(f"\nTraining complete. Final step: {self.step}")
            final_eval = self._evaluate(self.step)
            final_ckpt_path = self._save_checkpoint(self.step)
            if not self.config.no_wandb:
                log_checkpoint(self.step, self.algorithm, final_ckpt_path)
            finish()

        return {
            "algorithm": self.algorithm,
            "steps_attempted": self.step + 1 if self.config.steps else 0,
            "final_eval": final_eval,   # None on non-main ranks
            "final_checkpoint": final_ckpt_path,   # None on non-main ranks
        }
