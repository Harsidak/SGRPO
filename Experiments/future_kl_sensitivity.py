"""
Experiments/future_kl_sensitivity.py

Hyperparameter sensitivity analysis for SGRPO's Future-KL component.

Research question:
How sensitive is SGRPO's performance to Future-KL hyperparameters?

Parameters swept:
1. decay_rate ∈ {5, 10, 20, 30, 50, 100}
   - Controls temporal horizon of influence
   - Small = local token influence only
   - Large = long-range future token influence

2. clip_high ∈ {1.05, 1.1, 1.2, 1.5, 2.0}
   - Controls maximum influence weight
   - Too small = Future-KL has no effect (degenerates to DAPO)
   - Too large = training instability

Expected findings:
- Sweet spot around decay_rate=30, clip_high=1.2 (paper defaults)
- Performance degrades gracefully at extremes
- This demonstrates robustness of the Future-KL mechanism

Usage:
    python -m Experiments.future_kl_sensitivity --steps 200 --seed 42
    python -m Experiments.future_kl_sensitivity --device cpu --no_wandb
"""

import sys
import os
import json
import argparse
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import TrainingConfig, set_deterministic_seeds
from models.load_model import load_model
from data.gsm8k import GSM8KDataset
from trainer.base_trainer import BaseTrainer


# ── Sweep grid ───────────────────────────────────────────────────────────────
DECAY_RATES = [5, 10, 20, 30, 50, 100]
CLIP_HIGHS = [1.05, 1.1, 1.2, 1.5, 2.0]


def run_sensitivity_point(
    decay_rate: float,
    clip_high: float,
    seed: int,
    steps: int,
    group_size: int,
    device: str,
    no_wandb: bool = False,
) -> dict:
    """Run SGRPO with specific Future-KL hyperparameters."""
    set_deterministic_seeds(seed)

    config = TrainingConfig(
        algorithm="sgrpo",
        seed=seed,
        steps=steps,
        group_size=group_size,
        device=device,
        future_kl_decay=decay_rate,
        future_kl_clip_low=1.0,
        future_kl_clip_high=clip_high,
        run_name=f"fkl_decay{decay_rate}_clip{clip_high}_seed{seed}",
        wandb_project="rl-algo-comparison-2026",
        no_wandb=no_wandb,
        eval_every=50,
        eval_samples=30,
        checkpoint_every=steps,
    )

    print(f"\n  decay_rate={decay_rate}, clip_high={clip_high}")

    model, tokenizer = load_model(device=device, dtype=config.dtype)
    train_dataset = GSM8KDataset(split="train")
    eval_dataset = GSM8KDataset(split="test")

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        algorithm="sgrpo",
        config=config,
    )

    trainer.train()

    del model, trainer
    torch.cuda.empty_cache()

    return {
        "decay_rate": decay_rate,
        "clip_high": clip_high,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Future-KL Sensitivity Analysis"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--sweep_type", type=str, default="decay",
        choices=["decay", "clip", "grid"],
        help="'decay': sweep decay_rate only; 'clip': sweep clip_high only; "
             "'grid': full grid sweep"
    )
    args = parser.parse_args()

    if args.sweep_type == "decay":
        sweep_points = [(d, 1.2) for d in DECAY_RATES]
    elif args.sweep_type == "clip":
        sweep_points = [(30.0, c) for c in CLIP_HIGHS]
    else:
        sweep_points = list(product(DECAY_RATES, CLIP_HIGHS))

    print(f"\n{'#'*60}")
    print(f"# FUTURE-KL SENSITIVITY ANALYSIS")
    print(f"# Sweep type: {args.sweep_type}")
    print(f"# Points: {len(sweep_points)}")
    print(f"# Steps per point: {args.steps}")
    print(f"{'#'*60}")

    results = []
    for i, (decay_rate, clip_high) in enumerate(sweep_points):
        print(f"\n>>> Point {i+1}/{len(sweep_points)}")
        result = run_sensitivity_point(
            decay_rate=decay_rate,
            clip_high=clip_high,
            seed=args.seed,
            steps=args.steps,
            group_size=args.group_size,
            device=args.device,
            no_wandb=args.no_wandb,
        )
        results.append(result)

    # Save
    os.makedirs("Experiments/results", exist_ok=True)
    out_path = f"Experiments/results/future_kl_sensitivity_{args.sweep_type}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "steps": args.steps, "seed": args.seed,
                "sweep_type": args.sweep_type
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
