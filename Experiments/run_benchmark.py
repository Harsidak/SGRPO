"""
Experiments/run_benchmark.py

Main benchmark runner for comparing all 5 RL algorithms.

Runs each algorithm with identical configuration across multiple seeds,
logging everything to the same WandB project for direct comparison.

Usage:
    python -m Experiments.run_benchmark --algorithms ppo grpo dapo bapo sgrpo --seeds 42 123 456
    python -m Experiments.run_benchmark --algorithms sgrpo grpo --steps 200 --seeds 42

Research design:
- Same model, dataset, hyperparameters across all algorithms
- Only the algorithm (loss function + rollout generator) changes
- Multiple seeds for statistical significance
- All logged to WandB project "rl-algo-comparison-2026" for overlaid comparison
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import TrainingConfig, set_deterministic_seeds
from models.load_model import load_model
from data.gsm8k import GSM8KDataset
from trainer.base_trainer import BaseTrainer


def run_single_experiment(
    algorithm: str,
    seed: int,
    steps: int,
    group_size: int,
    device: str,
    no_wandb: bool = False,
) -> dict:
    """
    Run a single algorithm with a single seed.
    Returns a summary dict with key results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = TrainingConfig(
        algorithm=algorithm,
        seed=seed,
        steps=steps,
        group_size=group_size,
        device=device,
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
    print(f"Steps: {steps} | Group size: {config.group_size}")
    print(f"{'='*60}")

    model, tokenizer = load_model(device=device, dtype=config.dtype)
    train_dataset = GSM8KDataset(split="train")
    eval_dataset = GSM8KDataset(split="test")

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        algorithm=algorithm,
        config=config,
    )

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return {
        "algorithm": algorithm,
        "seed": seed,
        "steps": steps,
        "total_time_seconds": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for RL algorithm comparison"
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["ppo", "grpo", "dapo", "bapo", "sgrpo"],
        choices=["ppo", "grpo", "dapo", "bapo", "sgrpo"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# SGRPO RESEARCH BENCHMARK")
    print(f"# Algorithms: {', '.join(a.upper() for a in args.algorithms)}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Steps per run: {args.steps}")
    print(f"# Total runs: {len(args.algorithms) * len(args.seeds)}")
    print(f"{'#'*60}\n")

    results = []
    total_runs = len(args.algorithms) * len(args.seeds)
    run_idx = 0

    for algorithm in args.algorithms:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n>>> Run {run_idx}/{total_runs}: "
                  f"{algorithm.upper()} seed={seed}")

            result = run_single_experiment(
                algorithm=algorithm,
                seed=seed,
                steps=args.steps,
                group_size=args.group_size,
                device=args.device,
                no_wandb=args.no_wandb,
            )
            results.append(result)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['algorithm'].upper():6s} | "
              f"seed={r['seed']:3d} | "
              f"time={r['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
