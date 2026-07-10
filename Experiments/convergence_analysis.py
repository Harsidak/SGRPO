"""
Experiments/convergence_analysis.py

Convergence analysis: does SSM state contamination measurably hurt the
OPTIMIZATION process, not just the pre-rollout states?

The contamination probe (contamination_probe.py) proves the states differ.
This experiment closes the loop to the training claim with three measures:

Part A — Gradient estimator quality (no parameter updates):
    1. Gradient variance: repeat the GRPO gradient estimate K times on the
       same prompt under (a) contaminated rollouts (shared cache, no reset —
       what naive GRPO-on-Mamba does) and (b) isolated rollouts (SGRPO).
       Contamination breaks the i.i.d. assumption, so the group baseline is
       mis-centered and the per-coordinate variance of the gradient estimate
       is expected to rise.
    2. Rollout-order bias: under i.i.d. sampling, E[reward | rollout index k]
       is constant in k. Under contamination rollout k starts from rollout
       k-1's terminal state, so reward becomes correlated with k. We report
       mean reward by k and the Pearson correlation across all (group, k)
       pairs — a direct test of the paper's "bias correlated with rollout
       order" claim.

Part B — Convergence trajectories (short training runs):
    Same model init, same seed, same masked GRPO loss and optimizer; the
    ONLY difference between arms is whether rollouts are isolated. Records
    per step: training reward (sample efficiency = reward vs rollouts
    consumed), policy entropy during generation (collapse detection), grad
    norm, and loss.

Usage:
    python -m Experiments.convergence_analysis --steps 100 --group_size 4
    python -m Experiments.convergence_analysis --device cpu --steps 20 \
        --max_new_tokens 32 --grad_repeats 4 --no_wandb   # smoke test

Output:
    Experiments/results/convergence_analysis.json
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import set_deterministic_seeds
from models.load_model import load_model
from data import DATASETS
from losses.grpo_loss import compute_group_advantages
from rollouts.sgrpo_rollout import _snapshot_ssm_states, _restore_ssm_states


# ─────────────────────────────────────────────────────────────────────────────
# Rollout generation — one function, isolation as the ONLY switch
# ─────────────────────────────────────────────────────────────────────────────

def generate_group(
    model, tokenizer, prompt: str, group_size: int, max_new_tokens: int,
    temperature: float, device: str, isolated: bool,
) -> dict:
    """
    Generate G rollouts against a SINGLE shared DynamicCache.

    isolated=True:  restore the clean post-prompt state h_0 before every
                    rollout (SGRPO's mechanism).
    isolated=False: let the state carry over — rollout k starts from
                    rollout k-1's terminal state, and the starting logits
                    chain from the previous rollout (what a naive
                    cache-reusing GRPO implementation does).

    Both arms share this exact token loop, so isolation is the only
    experimental difference.

    Returns:
        generated_ids:   [G, prompt_len + max_gen_len] (right-padded)
        old_log_probs:   [G, max_gen_len] (right-padded with 0)
        attention_mask:  [G, max_gen_len] (1 = real generated token)
        generated_texts: list of G strings (generated portion only)
        prompt_len:      int
        entropy_mean:    mean sampling-distribution entropy over all
                         generated positions in the group
    """
    model.eval()

    prompt_inputs = tokenizer(
        prompt, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(device)
    prompt_len = prompt_inputs.input_ids.shape[1]

    all_ids, all_log_probs, all_texts, gen_lens = [], [], [], []
    entropy_sum, entropy_count = 0.0, 0

    with torch.no_grad():
        prompt_out = model(input_ids=prompt_inputs.input_ids, use_cache=True)
        cache = prompt_out.cache_params
        first_token_logits = prompt_out.logits[:, -1, :].detach().clone()
        h0_clean = _snapshot_ssm_states(cache) if isolated else None

        logits = first_token_logits
        for k in range(group_size):
            if isolated:
                _restore_ssm_states(cache, h0_clean, prompt_len=prompt_len)
                logits = first_token_logits

            generated_ids = prompt_inputs.input_ids.clone()
            log_probs = []

            for _ in range(max_new_tokens):
                step_logits = logits / temperature if temperature != 1.0 else logits
                log_dist = torch.log_softmax(step_logits, dim=-1)
                probs = torch.exp(log_dist)

                entropy_sum += -(probs * log_dist).sum().item()
                entropy_count += 1

                next_token = torch.multinomial(probs, 1)  # [1, 1]
                log_probs.append(log_dist[0, next_token[0, 0]])
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token[0, 0].item() == tokenizer.eos_token_id:
                    break

                out = model(input_ids=next_token, cache_params=cache,
                            use_cache=True)
                logits = out.logits[:, -1, :]

            all_ids.append(generated_ids[0])
            all_log_probs.append(torch.stack(log_probs))
            gen_lens.append(len(log_probs))
            all_texts.append(tokenizer.decode(
                generated_ids[0, prompt_len:], skip_special_tokens=True))

    max_len = max(gen_lens)
    padded_lp, mask_rows = [], []
    for lp, n in zip(all_log_probs, gen_lens):
        pad = max_len - n
        if pad > 0:
            lp = torch.cat([lp, torch.zeros(pad, device=device, dtype=lp.dtype)])
        padded_lp.append(lp)
        row = torch.zeros(max_len, device=device)
        row[:n] = 1.0
        mask_rows.append(row)

    return {
        "generated_ids": torch.nn.utils.rnn.pad_sequence(
            all_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "old_log_probs": torch.stack(padded_lp),
        "attention_mask": torch.stack(mask_rows),
        "generated_texts": all_texts,
        "prompt_len": prompt_len,
        "entropy_mean": entropy_sum / max(entropy_count, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss / gradient helpers
# ─────────────────────────────────────────────────────────────────────────────

def masked_grpo_loss(
    new_log_probs, old_log_probs, rewards, attention_mask, clip_eps=0.2,
):
    """
    Token-normalized clipped GRPO surrogate (DAPO normalization) with the
    padding mask applied. Identical for both arms — the loss is NOT the
    experimental variable here.
    """
    advantages = compute_group_advantages(rewards)
    ratio = torch.exp(new_log_probs - old_log_probs)
    adv = advantages.unsqueeze(1).expand_as(ratio)
    surrogate = torch.min(
        ratio * adv,
        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
    )
    return -(surrogate * attention_mask).sum() / attention_mask.sum().clamp(min=1.0)


def compute_new_log_probs(model, generated_ids, prompt_len):
    """Forward pass over full sequences; gather generated-token log probs."""
    logits = model(input_ids=generated_ids).logits
    gen_logits = logits[:, prompt_len - 1:-1, :]
    gen_ids = generated_ids[:, prompt_len:]
    return torch.log_softmax(gen_logits, dim=-1).gather(
        2, gen_ids.unsqueeze(2)).squeeze(2)


def gradient_sample(model, rollout, rewards, clip_eps):
    """
    One GRPO gradient estimate for one rollout group.
    Returns (flat_grad_cpu_fp32, grad_norm). No optimizer step is taken.
    """
    model.zero_grad(set_to_none=True)
    new_lp = compute_new_log_probs(
        model, rollout["generated_ids"], rollout["prompt_len"])
    gen_len = rollout["old_log_probs"].shape[1]
    loss = masked_grpo_loss(
        new_lp[:, :gen_len], rollout["old_log_probs"], rewards,
        rollout["attention_mask"], clip_eps)
    loss.backward()

    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    flat = torch.cat(grads).float()
    norm = flat.norm().item()
    flat_cpu = flat.cpu()
    model.zero_grad(set_to_none=True)
    return flat_cpu, norm


# ─────────────────────────────────────────────────────────────────────────────
# Part A — gradient estimator quality
# ─────────────────────────────────────────────────────────────────────────────

def analyze_gradients(
    model, tokenizer, dataset_items, args, subset_idx: torch.Tensor,
) -> dict:
    """
    For each prompt and each arm, draw `grad_repeats` independent gradient
    estimates and measure:
      - per-coordinate variance across repeats (on a fixed random coordinate
        subset — an unbiased estimate of the full trace of the gradient
        covariance / #params)
      - grad-norm mean/std
      - reward by rollout index k (order-bias test)

    Degenerate groups (all rewards identical → zero advantage → zero
    gradient) are excluded from the variance statistics, exactly as DAPO's
    dynamic sampling filter excludes them from training; they still count
    toward the order-bias reward matrix.
    """
    results = {}
    for arm, isolated in [("contaminated", False), ("isolated", True)]:
        per_prompt_var, all_norms = [], []
        reward_by_k = []          # rows: one group's rewards indexed by k
        degenerate, used = 0, 0

        for p_idx, item in enumerate(dataset_items[:args.grad_prompts]):
            grad_subsets = []
            for rep in range(args.grad_repeats):
                rollout = generate_group(
                    model, tokenizer, item["prompt"], args.group_size,
                    args.max_new_tokens, args.temperature, args.device,
                    isolated=isolated)
                rewards = torch.tensor(
                    [item_reward(item, t)
                     for t in rollout["generated_texts"]],
                    device=args.device)
                reward_by_k.append(rewards.tolist())

                if rewards.std() < 1e-8:
                    degenerate += 1
                    continue
                flat, norm = gradient_sample(
                    model, rollout, rewards, args.clip_eps)
                grad_subsets.append(flat[subset_idx])
                all_norms.append(norm)
                used += 1

            if len(grad_subsets) >= 2:
                stacked = torch.stack(grad_subsets)          # [K, subset]
                per_prompt_var.append(
                    stacked.var(dim=0, unbiased=True).mean().item())
            print(f"  [{arm}] prompt {p_idx + 1}/{args.grad_prompts}: "
                  f"{len(grad_subsets)}/{args.grad_repeats} non-degenerate")

        rk = np.array(reward_by_k) if reward_by_k else np.zeros((0, args.group_size))
        # Pearson r between rollout index k and reward over all (group, k)
        if rk.size and rk.std() > 0:
            ks = np.tile(np.arange(rk.shape[1]), rk.shape[0])
            r_flat = rk.flatten()
            order_corr = float(np.corrcoef(ks, r_flat)[0, 1])
        else:
            order_corr = 0.0

        results[arm] = {
            "grad_variance_mean": (float(np.mean(per_prompt_var))
                                   if per_prompt_var else None),
            "grad_variance_per_prompt": per_prompt_var,
            "grad_norm_mean": float(np.mean(all_norms)) if all_norms else None,
            "grad_norm_std": float(np.std(all_norms)) if all_norms else None,
            "gradient_samples_used": used,
            "degenerate_groups_skipped": degenerate,
            "reward_mean_by_rollout_index": (rk.mean(axis=0).tolist()
                                             if rk.size else []),
            "reward_order_correlation": order_corr,
        }

    iso_v = results["isolated"]["grad_variance_mean"]
    con_v = results["contaminated"]["grad_variance_mean"]
    results["variance_ratio_contaminated_over_isolated"] = (
        con_v / iso_v if (iso_v and con_v and iso_v > 0) else None)
    return results


def item_reward(item, text: str) -> float:
    """Reward via the dataset object attached at load time."""
    return item["_dataset"].compute_reward(text, item["answer"])


# ─────────────────────────────────────────────────────────────────────────────
# Part B — convergence trajectories
# ─────────────────────────────────────────────────────────────────────────────

def run_training_arm(
    model, tokenizer, dataset_items, args, isolated: bool,
    init_state: dict,
) -> list[dict]:
    """
    Short training run with masked GRPO loss. Both arms start from the same
    weights (init_state) and the same seed; `isolated` is the only variable.
    Returns one record per step.
    """
    model.load_state_dict(init_state)
    set_deterministic_seeds(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    records = []
    rollouts_consumed = 0
    for step in range(args.steps):
        item = dataset_items[step % len(dataset_items)]

        rollout = generate_group(
            model, tokenizer, item["prompt"], args.group_size,
            args.max_new_tokens, args.temperature, args.device,
            isolated=isolated)
        rewards = torch.tensor(
            [item_reward(item, t)
             for t in rollout["generated_texts"]],
            device=args.device)
        rollouts_consumed += args.group_size

        record = {
            "step": step,
            "rollouts_consumed": rollouts_consumed,
            "reward_mean": rewards.mean().item(),
            "entropy": rollout["entropy_mean"],
            "loss": None,
            "grad_norm": None,
            "skipped_degenerate": False,
        }

        if rewards.std() < 1e-8:
            record["skipped_degenerate"] = True
            records.append(record)
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        new_lp = compute_new_log_probs(
            model, rollout["generated_ids"], rollout["prompt_len"])
        gen_len = rollout["old_log_probs"].shape[1]
        loss = masked_grpo_loss(
            new_lp[:, :gen_len], rollout["old_log_probs"], rewards,
            rollout["attention_mask"], args.clip_eps)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        optimizer.step()

        record["loss"] = loss.item()
        record["grad_norm"] = float(grad_norm)
        records.append(record)

        if (step + 1) % 10 == 0:
            recent = [r["reward_mean"] for r in records[-10:]]
            print(f"  [{'isolated' if isolated else 'contaminated'}] "
                  f"step {step + 1}/{args.steps} | "
                  f"reward(10)={np.mean(recent):.3f} | "
                  f"entropy={record['entropy']:.3f}")

    return records


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convergence analysis: contaminated vs isolated rollouts")
    parser.add_argument("--steps", type=int, default=100,
                        help="Training steps per arm in Part B")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--grad_prompts", type=int, default=3,
                        help="Prompts used for Part A gradient variance")
    parser.add_argument("--grad_repeats", type=int, default=8,
                        help="Independent gradient estimates per prompt/arm")
    parser.add_argument("--grad_subset", type=int, default=262144,
                        help="Random coordinate subset size for variance")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=sorted(DATASETS.keys()))
    parser.add_argument("--model_name", type=str,
                        default="state-spaces/mamba-130m-hf")
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Training prompt pool size for Part B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_gradient_analysis", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="Experiments/results")
    args = parser.parse_args()

    set_deterministic_seeds(args.seed)

    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS — contaminated vs isolated rollouts")
    print(f"Model: {args.model_name} | Dataset: {args.dataset}")
    print(f"Part A: {args.grad_prompts} prompts x {args.grad_repeats} repeats")
    print(f"Part B: {args.steps} steps x 2 arms | G={args.group_size}")
    print(f"{'='*60}\n")

    model, tokenizer = load_model(device=args.device,
                                  model_name=args.model_name)
    dataset = DATASETS[args.dataset](split="train")
    items = list(dataset)[:max(args.num_prompts, args.grad_prompts)]
    for it in items:
        it["_dataset"] = dataset

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in vars(args).items()},
    }

    # ── Part A ───────────────────────────────────────────────────────────
    if not args.skip_gradient_analysis:
        print("Part A: gradient estimator quality")
        n_params = sum(p.numel() for p in model.parameters())
        g = torch.Generator().manual_seed(args.seed)
        subset_idx = torch.randperm(n_params, generator=g)[
            :min(args.grad_subset, n_params)]
        results["gradient_analysis"] = analyze_gradients(
            model, tokenizer, items, args, subset_idx)

        ga = results["gradient_analysis"]
        print(f"\n  grad variance  contaminated: "
              f"{ga['contaminated']['grad_variance_mean']}")
        print(f"  grad variance  isolated:     "
              f"{ga['isolated']['grad_variance_mean']}")
        print(f"  variance ratio (cont/iso):   "
              f"{ga['variance_ratio_contaminated_over_isolated']}")
        print(f"  reward-order corr  contaminated: "
              f"{ga['contaminated']['reward_order_correlation']:.4f}")
        print(f"  reward-order corr  isolated:     "
              f"{ga['isolated']['reward_order_correlation']:.4f}\n")

    # ── Part B ───────────────────────────────────────────────────────────
    if not args.skip_training:
        print("Part B: convergence trajectories")
        init_state = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
        trajectories = {}
        for arm, isolated in [("contaminated", False), ("isolated", True)]:
            print(f"\n  --- arm: {arm} ---")
            trajectories[arm] = run_training_arm(
                model, tokenizer, items, args, isolated, init_state)
            if args.device == "cuda":
                torch.cuda.empty_cache()
        results["convergence"] = trajectories

        for arm in ("contaminated", "isolated"):
            recs = trajectories[arm]
            last = [r["reward_mean"] for r in recs[-20:]]
            ent = [r["entropy"] for r in recs[-20:]]
            results[f"{arm}_final_reward_mean"] = float(np.mean(last))
            results[f"{arm}_final_entropy_mean"] = float(np.mean(ent))
            print(f"  {arm}: final-20-step reward "
                  f"{np.mean(last):.4f}, entropy {np.mean(ent):.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "convergence_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if not args.no_wandb:
        import wandb
        wandb.init(
            project="rl-algo-comparison-2026",
            name=f"convergence_analysis_"
                 f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["experiment", "convergence-analysis"],
            config=results["config"],
        )
        flat = {}
        ga = results.get("gradient_analysis", {})
        for arm in ("contaminated", "isolated"):
            if arm in ga:
                for key in ("grad_variance_mean", "grad_norm_mean",
                            "reward_order_correlation"):
                    if ga[arm].get(key) is not None:
                        flat[f"convergence/{arm}_{key}"] = ga[arm][key]
        for key in ("contaminated_final_reward_mean",
                    "isolated_final_reward_mean",
                    "contaminated_final_entropy_mean",
                    "isolated_final_entropy_mean"):
            if key in results:
                flat[f"convergence/{key}"] = results[key]
        if flat:
            wandb.log(flat)
        if "convergence" in results:
            for arm, recs in results["convergence"].items():
                for r in recs:
                    wandb.log({
                        f"convergence/{arm}_reward": r["reward_mean"],
                        f"convergence/{arm}_entropy": r["entropy"],
                        "convergence/rollouts": r["rollouts_consumed"],
                    })
        wandb.finish()


if __name__ == "__main__":
    main()
