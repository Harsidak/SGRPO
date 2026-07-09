"""
Experiments/contamination_probe.py

Statistical experiment proving SSM state contamination exists
and SGRPO's state isolation fixes it.

Hypothesis:
    H0: Mamba SSM states after rollout k are independent of rollout k-1
    H1: SSM states carry contamination between sequential rollouts

Methodology:
1. Generate G rollouts WITHOUT state isolation (standard GRPO approach)
   - Capture SSM states before each rollout: {h_pre_1, h_pre_2, ..., h_pre_G}
2. Generate G rollouts WITH state isolation (SGRPO approach)
   - Capture SSM states before each rollout: {h_iso_1, h_iso_2, ..., h_iso_G}
3. Compute pairwise cosine similarity within each group
4. t-test: cos_sim(standard) vs cos_sim(isolated)

Expected result:
- Standard: cos_sim varies (contamination causes drift)
- Isolated: cos_sim ≈ 1.0 (all start from identical h_0)
- t-test: p < 0.001

Usage:
    python -m Experiments.contamination_probe --num_prompts 10 --group_size 6
    python -m Experiments.contamination_probe --device cpu --no_wandb

Output:
- Console: t-statistic, p-value, effect size
- WandB: contamination metrics, cosine similarity distributions
- File: Experiments/results/contamination_probe_results.json
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

from config import TrainingConfig, set_deterministic_seeds
from models.load_model import load_model
from data.gsm8k import GSM8KDataset


def _snapshot_ssm_states(cache) -> list[dict[str, torch.Tensor]]:
    """
    Clone all SSM recurrent states AND conv states from a DynamicCache.

    transformers >= 4.46 removed MambaCache; the model returns a
    DynamicCache in output.cache_params where per-layer state lives in
    cache.layers[i].recurrent_states ([batch, d_inner, d_state]) and
    cache.layers[i].conv_states ([batch, d_inner, conv_kernel]).
    """
    return [
        {
            "recurrent": layer.recurrent_states.detach().clone(),
            "conv": layer.conv_states.detach().clone(),
        }
        for layer in cache.layers
    ]


def _restore_ssm_states(cache, snapshot: list[dict[str, torch.Tensor]]) -> None:
    """Restore SSM + conv states from snapshot (the isolation operation)."""
    for i, state in enumerate(snapshot):
        cache.layers[i].recurrent_states.copy_(state["recurrent"])
        cache.layers[i].conv_states.copy_(state["conv"])


def _pairwise_cosine_sims(
    states_list: list[list[dict[str, torch.Tensor]]],
) -> list[float]:
    """
    Compute pairwise cosine similarity between SSM state snapshots.

    Measures the recurrent_states (h_t) — the tensor that carries
    contamination. NOT the residual-stream hidden states.

    Args:
        states_list: list of G snapshots, each a list of L per-layer dicts

    Returns:
        list of pairwise cosine similarities across all layer-level pairs
    """
    sims = []
    for (i, states_i), (j, states_j) in combinations(enumerate(states_list), 2):
        for layer_idx in range(len(states_i)):
            flat_i = states_i[layer_idx]["recurrent"].flatten().float()
            flat_j = states_j[layer_idx]["recurrent"].flatten().float()
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_i.unsqueeze(0), flat_j.unsqueeze(0)
            ).item()
            sims.append(cos_sim)
    return sims


def _drift_vs_h0(states_list: list[list[dict[str, torch.Tensor]]]) -> list[float]:
    """
    Cosine similarity of each pre-rollout state vs the FIRST pre-rollout
    state (= clean h_0), averaged over layers. Index k in the returned list
    is rollout k's starting-state similarity to h_0.

    Under contamination this decays monotonically with k (the paper's
    headline evidence, e.g. 0.490 → 0.451 → 0.424 → 0.380 at T=30).
    Under isolation it is exactly 1.0 for all k.
    """
    h0 = states_list[0]
    drift = []
    for states_k in states_list:
        layer_sims = []
        for layer_idx in range(len(h0)):
            flat_0 = h0[layer_idx]["recurrent"].flatten().float()
            flat_k = states_k[layer_idx]["recurrent"].flatten().float()
            layer_sims.append(torch.nn.functional.cosine_similarity(
                flat_0.unsqueeze(0), flat_k.unsqueeze(0)
            ).item())
        drift.append(sum(layer_sims) / len(layer_sims))
    return drift


def probe_contamination(
    model, tokenizer, prompt: str, group_size: int,
    max_new_tokens: int, device: str,
) -> dict:
    """
    Run one prompt through both standard and isolated rollout pipelines.
    Returns cosine similarity statistics for comparison.
    """
    model.eval()

    prompt_inputs = tokenizer(
        prompt, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(device)

    def _generate_one_rollout(cache, first_logits):
        """
        Sample one rollout token-by-token against `cache`, starting from
        `first_logits` (the next-token distribution at the rollout start).
        Identical code for both arms — the ONLY experimental difference
        between standard and isolated is whether h_0 was restored.
        Returns the last logits so the contaminated arm can chain rollouts.
        """
        logits = first_logits
        for _ in range(max_new_tokens):
            next_token = torch.multinomial(
                torch.softmax(logits, dim=-1), 1
            )
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break
            out = model(
                input_ids=next_token,
                cache_params=cache,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
        return logits

    # ── STANDARD rollouts (no isolation) ─────────────────────────────────
    # Use a single shared cache across all rollouts — this is what
    # naive GRPO does and what causes contamination
    standard_pre_states = []

    with torch.no_grad():
        # Process prompt once; the model auto-creates a DynamicCache
        prompt_out = model(
            input_ids=prompt_inputs.input_ids,
            use_cache=True,
        )
        shared_cache = prompt_out.cache_params
        logits = prompt_out.logits[:, -1, :]

        for k in range(group_size):
            # Snapshot state BEFORE this rollout
            standard_pre_states.append(_snapshot_ssm_states(shared_cache))
            # Generate rollout (state carries forward — contamination!)
            logits = _generate_one_rollout(shared_cache, logits)

    # ── ISOLATED rollouts (SGRPO) ────────────────────────────────────────
    isolated_pre_states = []

    with torch.no_grad():
        prompt_out = model(
            input_ids=prompt_inputs.input_ids,
            use_cache=True,
        )
        iso_cache = prompt_out.cache_params
        first_token_logits = prompt_out.logits[:, -1, :].detach().clone()
        h0_clean = _snapshot_ssm_states(iso_cache)

        for k in range(group_size):
            # Restore h_0 before EVERY rollout — state isolation
            _restore_ssm_states(iso_cache, h0_clean)
            isolated_pre_states.append(_snapshot_ssm_states(iso_cache))
            # Generate rollout from clean state
            _generate_one_rollout(iso_cache, first_token_logits)

    # ── Compute pairwise cosine similarities ─────────────────────────────
    standard_sims = _pairwise_cosine_sims(standard_pre_states)
    isolated_sims = _pairwise_cosine_sims(isolated_pre_states)

    return {
        "standard_cosine_sims": standard_sims,
        "isolated_cosine_sims": isolated_sims,
        "standard_drift_by_rollout": _drift_vs_h0(standard_pre_states),
        "isolated_drift_by_rollout": _drift_vs_h0(isolated_pre_states),
    }


def main():
    parser = argparse.ArgumentParser(
        description="SSM State Contamination Probe"
    )
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--group_size", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="Experiments/results",
                        help="Directory for structured JSON results")
    args = parser.parse_args()

    set_deterministic_seeds(args.seed)

    print(f"\n{'='*60}")
    print("SSM STATE CONTAMINATION PROBE")
    print(f"Prompts: {args.num_prompts} | Group size: {args.group_size}")
    print(f"{'='*60}\n")

    model, tokenizer = load_model(device=args.device)
    dataset = GSM8KDataset(split="test")
    data = list(dataset)[:args.num_prompts]

    all_standard_sims = []
    all_isolated_sims = []
    all_standard_drift = []   # [num_prompts, G] — drift vs h0 by rollout index
    all_isolated_drift = []

    for i, item in enumerate(data):
        print(f"Prompt {i+1}/{args.num_prompts}...", end=" ", flush=True)
        result = probe_contamination(
            model, tokenizer, item["prompt"],
            args.group_size, args.max_new_tokens, args.device,
        )
        all_standard_sims.extend(result["standard_cosine_sims"])
        all_isolated_sims.extend(result["isolated_cosine_sims"])
        all_standard_drift.append(result["standard_drift_by_rollout"])
        all_isolated_drift.append(result["isolated_drift_by_rollout"])
        print(f"std_sim={np.mean(result['standard_cosine_sims']):.6f}, "
              f"iso_sim={np.mean(result['isolated_cosine_sims']):.6f}")

    # ── Statistical test ─────────────────────────────────────────────────
    std_arr = np.array(all_standard_sims)
    iso_arr = np.array(all_isolated_sims)

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(std_arr, iso_arr, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std_arr.std()**2 + iso_arr.std()**2) / 2)
    cohens_d = (std_arr.mean() - iso_arr.mean()) / max(pooled_std, 1e-10)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Standard rollouts - cosine sim: {std_arr.mean():.6f} ± {std_arr.std():.6f}")
    print(f"Isolated rollouts - cosine sim: {iso_arr.mean():.6f} ± {iso_arr.std():.6f}")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_value:.2e}")
    print(f"Cohen's d:   {cohens_d:.4f}")
    print(f"\nContamination {'CONFIRMED' if p_value < 0.001 else 'NOT confirmed'}")
    print(f"State isolation {'EFFECTIVE' if iso_arr.std() < 1e-6 else 'has variance'}")

    # Drift-by-rollout-index curves (mean over prompts)
    std_drift_curve = np.array(all_standard_drift).mean(axis=0)
    iso_drift_curve = np.array(all_isolated_drift).mean(axis=0)
    print(f"\nDrift vs h0 by rollout index (contaminated): "
          f"{[round(x, 4) for x in std_drift_curve.tolist()]}")
    print(f"Drift vs h0 by rollout index (isolated):     "
          f"{[round(x, 4) for x in iso_drift_curve.tolist()]}")

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": args.num_prompts,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "standard_cosine_sim_mean": float(std_arr.mean()),
        "standard_cosine_sim_std": float(std_arr.std()),
        "isolated_cosine_sim_mean": float(iso_arr.mean()),
        "isolated_cosine_sim_std": float(iso_arr.std()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "contamination_confirmed": bool(p_value < 0.001),
        "n_standard_pairs": len(all_standard_sims),
        "n_isolated_pairs": len(all_isolated_sims),
        "standard_drift_by_rollout_index": std_drift_curve.tolist(),
        "isolated_drift_by_rollout_index": iso_drift_curve.tolist(),
    }

    out_path = os.path.join(args.output_dir, "contamination_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── WandB logging ────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb
        wandb.init(
            project="rl-algo-comparison-2026",
            name=f"contamination_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["experiment", "contamination-probe"],
            config=results["config"],
        )
        wandb.log({
            "probe/standard_cosine_sim_mean": std_arr.mean(),
            "probe/standard_cosine_sim_std": std_arr.std(),
            "probe/isolated_cosine_sim_mean": iso_arr.mean(),
            "probe/isolated_cosine_sim_std": iso_arr.std(),
            "probe/t_statistic": t_stat,
            "probe/p_value": p_value,
            "probe/cohens_d": cohens_d,
        })
        wandb.log({
            "probe/standard_sims_hist": wandb.Histogram(std_arr),
            "probe/isolated_sims_hist": wandb.Histogram(iso_arr),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
