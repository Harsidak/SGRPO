"""
Experiments/run_benchmark.py

Main benchmark runner for comparing all 5 RL algorithms.

Runs each algorithm with identical configuration across multiple seeds,
captures the final held-out accuracy of every run, then computes the
statistics reviewers will ask for:

- per-algorithm mean ± std and 95% confidence interval (t-based)
- pairwise paired t-test and Wilcoxon signed-rank test (paired by seed)
- Cohen's d effect size (paired: mean(diff) / std(diff))

Outputs:
- Experiments/results/benchmark_results.json  — raw numbers + stats
- Docs/results/main_benchmark_table.md        — paper-ready table

Usage:
    python -m Experiments.run_benchmark --steps 500
    python -m Experiments.run_benchmark --algorithms sgrpo grpo --seeds 42 123
    python -m Experiments.run_benchmark --model_name state-spaces/mamba-370m-hf --dataset math

Research design:
- Same model, dataset, hyperparameters across all algorithms
- Only the algorithm (loss function + rollout generator) changes
- 5 seeds by default — 3 is insufficient for AAAI
- All logged to the same WandB project for overlaid comparison
"""

import sys
import os
import json
import argparse
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats as scipy_stats

from config import TrainingConfig, set_deterministic_seeds
from models.load_model import load_model, detect_architecture
from data import DATASETS
from trainer.base_trainer import BaseTrainer

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]
ALGORITHMS = ["ppo", "grpo", "dapo", "bapo", "sgrpo"]


def run_single_experiment(
    algorithm: str,
    seed: int,
    steps: int,
    group_size: int,
    device: str,
    model_name: str,
    dataset_name: str,
    no_wandb: bool = False,
) -> dict:
    """
    Run a single algorithm with a single seed.
    Returns a summary dict with the run's headline results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = TrainingConfig(
        algorithm=algorithm,
        seed=seed,
        steps=steps,
        group_size=group_size,
        device=device,
        model_name=model_name,
        dataset=dataset_name,
        run_name=f"bench_{algorithm}_seed{seed}_{timestamp}",
        wandb_project="rl-algo-comparison-2026",
        no_wandb=no_wandb,
        eval_every=50,
        eval_samples=50,
        checkpoint_every=steps,  # checkpoint only at end
    )

    set_deterministic_seeds(seed, config.deterministic)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {algorithm.upper()} | Seed: {seed}")
    print(f"Model: {model_name} | Dataset: {dataset_name}")
    print(f"Steps: {steps} | Group size: {config.group_size}")
    print(f"{'='*60}")

    model, tokenizer = load_model(
        device=device, dtype=config.dtype, model_name=model_name
    )
    arch = detect_architecture(model)
    dataset_cls = DATASETS[dataset_name]
    train_dataset = dataset_cls(split="train")
    eval_dataset = dataset_cls(split="test")

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        algorithm=algorithm,
        config=config,
        arch=arch,
    )

    start_time = time.time()
    summary = trainer.train()
    total_time = time.time() - start_time

    final_eval = summary.get("final_eval") or {}

    # Clean up GPU memory before the next run
    del model, trainer
    torch.cuda.empty_cache()

    return {
        "algorithm": algorithm,
        "seed": seed,
        "steps": steps,
        "model_name": model_name,
        "dataset": dataset_name,
        "final_accuracy": final_eval.get("accuracy"),
        "final_average_reward": final_eval.get("average_reward"),
        "final_correct": final_eval.get("correct"),
        "final_total": final_eval.get("total"),
        "total_time_seconds": total_time,
        "error": None,
    }


def aggregate_by_algorithm(runs: list[dict]) -> dict:
    """Per-algorithm mean ± std and 95% CI of final accuracy over seeds."""
    aggregates = {}
    for algorithm in {r["algorithm"] for r in runs}:
        accs = [r["final_accuracy"] for r in runs
                if r["algorithm"] == algorithm
                and r["final_accuracy"] is not None]
        if not accs:
            continue
        accs_arr = np.array(accs, dtype=np.float64)
        n = len(accs_arr)
        mean = float(accs_arr.mean())
        std = float(accs_arr.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            sem = std / np.sqrt(n)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
            ci = (mean - t_crit * sem, mean + t_crit * sem)
        else:
            ci = (mean, mean)
        aggregates[algorithm] = {
            "n_seeds": n,
            "accuracy_mean": mean,
            "accuracy_std": std,
            "accuracy_ci95": [float(ci[0]), float(ci[1])],
            "accuracies": [float(a) for a in accs_arr],
        }
    return aggregates


def pairwise_statistics(runs: list[dict], seeds: list[int]) -> list[dict]:
    """
    For every algorithm pair, compute paired-by-seed statistics.
    Pairing by seed removes seed-level variance from the comparison —
    both runs of a pair saw the same data ordering and initialization.
    """
    by_algo_seed = {
        (r["algorithm"], r["seed"]): r["final_accuracy"]
        for r in runs if r["final_accuracy"] is not None
    }
    algorithms = sorted({r["algorithm"] for r in runs})

    comparisons = []
    for i, algo_a in enumerate(algorithms):
        for algo_b in algorithms[i + 1:]:
            paired = [
                (by_algo_seed[(algo_a, s)], by_algo_seed[(algo_b, s)])
                for s in seeds
                if (algo_a, s) in by_algo_seed and (algo_b, s) in by_algo_seed
            ]
            if len(paired) < 2:
                continue

            a = np.array([p[0] for p in paired], dtype=np.float64)
            b = np.array([p[1] for p in paired], dtype=np.float64)
            diff = a - b

            t_stat, t_p = scipy_stats.ttest_rel(a, b)

            # Wilcoxon requires at least one non-zero difference
            if np.any(diff != 0):
                try:
                    w_stat, w_p = scipy_stats.wilcoxon(a, b)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
            else:
                w_stat, w_p = float("nan"), 1.0

            # Cohen's d for paired samples: mean(diff) / std(diff)
            diff_std = diff.std(ddof=1)
            cohens_d = float(diff.mean() / diff_std) if diff_std > 0 else 0.0

            comparisons.append({
                "algorithm_a": algo_a,
                "algorithm_b": algo_b,
                "n_pairs": len(paired),
                "mean_diff_a_minus_b": float(diff.mean()),
                "paired_t_stat": float(t_stat),
                "paired_t_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "cohens_d_paired": cohens_d,
            })
    return comparisons


def write_markdown_table(
    aggregates: dict,
    comparisons: list[dict],
    meta: dict,
    path: str,
) -> None:
    """Paper-ready benchmark table with baseline deltas and significance."""
    baseline = "grpo" if "grpo" in aggregates else None
    baseline_mean = aggregates[baseline]["accuracy_mean"] if baseline else None

    # Index pairwise results for the Δ-vs-baseline significance column
    pair_index = {
        frozenset((c["algorithm_a"], c["algorithm_b"])): c
        for c in comparisons
    }

    lines = [
        "# Main Benchmark Results",
        "",
        f"- Model: `{meta['model_name']}` | Dataset: `{meta['dataset']}` | "
        f"Steps: {meta['steps']} | Group size: {meta['group_size']}",
        f"- Seeds: {meta['seeds']} | Generated: {meta['timestamp']}",
        "",
        "| Algorithm | Accuracy (mean ± std) | 95% CI | Δ vs GRPO | "
        "p (paired t) | Cohen's d |",
        "|---|---|---|---|---|---|",
    ]

    order = [a for a in ALGORITHMS if a in aggregates]
    for algo in order:
        agg = aggregates[algo]
        mean, std = agg["accuracy_mean"], agg["accuracy_std"]
        lo, hi = agg["accuracy_ci95"]
        if baseline and algo != baseline:
            delta = f"{mean - baseline_mean:+.3f}"
            pair = pair_index.get(frozenset((algo, baseline)))
            if pair:
                p = pair["paired_t_p"]
                d = pair["cohens_d_paired"]
                # d is signed a-minus-b in alphabetical order; re-sign so
                # positive means this algorithm beats the baseline
                if pair["algorithm_a"] == baseline:
                    d = -d
                p_str, d_str = f"{p:.4f}", f"{d:+.2f}"
            else:
                p_str, d_str = "—", "—"
        else:
            delta, p_str, d_str = ("0.000" if algo == baseline else "—"), "—", "—"
        lines.append(
            f"| {algo.upper()} | {mean:.3f} ± {std:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {delta} | {p_str} | {d_str} |"
        )

    lines += [
        "",
        "## All pairwise comparisons",
        "",
        "| A | B | mean(A−B) | paired t p | Wilcoxon p | Cohen's d |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['algorithm_a'].upper()} | {c['algorithm_b'].upper()} | "
            f"{c['mean_diff_a_minus_b']:+.3f} | {c['paired_t_p']:.4f} | "
            f"{c['wilcoxon_p']:.4f} | {c['cohens_d_paired']:+.2f} |"
        )
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown table written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for RL algorithm comparison"
    )
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS,
                        choices=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Use >= 5 seeds for publication-grade stats.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str,
                        default="state-spaces/mamba-130m-hf")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=sorted(DATASETS.keys()))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join("Experiments", "results"))
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if len(args.seeds) < 5:
        print(f"WARNING: {len(args.seeds)} seeds — statistical tests need "
              f">= 5 seeds to be publication-grade (defaults: {DEFAULT_SEEDS}).")

    print(f"\n{'#'*60}")
    print("# SGRPO RESEARCH BENCHMARK")
    print(f"# Algorithms: {', '.join(a.upper() for a in args.algorithms)}")
    print(f"# Model: {args.model_name} | Dataset: {args.dataset}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Steps per run: {args.steps}")
    print(f"# Total runs: {len(args.algorithms) * len(args.seeds)}")
    print(f"{'#'*60}\n")

    runs = []
    total_runs = len(args.algorithms) * len(args.seeds)
    run_idx = 0

    for algorithm in args.algorithms:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n>>> Run {run_idx}/{total_runs}: "
                  f"{algorithm.upper()} seed={seed}")
            try:
                result = run_single_experiment(
                    algorithm=algorithm,
                    seed=seed,
                    steps=args.steps,
                    group_size=args.group_size,
                    device=args.device,
                    model_name=args.model_name,
                    dataset_name=args.dataset,
                    no_wandb=args.no_wandb,
                )
            except Exception as exc:  # one bad run must not kill the sweep
                traceback.print_exc()
                torch.cuda.empty_cache()
                result = {
                    "algorithm": algorithm, "seed": seed,
                    "steps": args.steps, "model_name": args.model_name,
                    "dataset": args.dataset, "final_accuracy": None,
                    "final_average_reward": None, "final_correct": None,
                    "final_total": None, "total_time_seconds": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            runs.append(result)

    # ── Statistics ────────────────────────────────────────────────────────
    aggregates = aggregate_by_algorithm(runs)
    comparisons = pairwise_statistics(runs, args.seeds)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "dataset": args.dataset,
        "steps": args.steps,
        "group_size": args.group_size,
        "seeds": args.seeds,
        "algorithms": args.algorithms,
    }

    # ── Structured results ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta,
            "runs": runs,
            "aggregates": aggregates,
            "pairwise_comparisons": comparisons,
        }, f, indent=2)
    print(f"\nStructured results written: {results_path}")

    if aggregates:
        write_markdown_table(
            aggregates, comparisons, meta,
            path=os.path.join("Docs", "results", "main_benchmark_table.md"),
        )

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    for algo in [a for a in ALGORITHMS if a in aggregates]:
        agg = aggregates[algo]
        print(f"  {algo.upper():6s} | acc = {agg['accuracy_mean']:.3f} "
              f"± {agg['accuracy_std']:.3f} "
              f"(n={agg['n_seeds']})")
    failed = [r for r in runs if r["error"]]
    if failed:
        print(f"\n  {len(failed)} run(s) FAILED:")
        for r in failed:
            print(f"    {r['algorithm']} seed={r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()
