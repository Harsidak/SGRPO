"""
Experiments/ablation_study.py

Ablation study for SGRPO components.

Research question: What is the marginal contribution of each SGRPO component?

Conditions:
1. SGRPO (full)        = state isolation + Future-KL weighting + DAPO token normalization
2. SGRPO - Future-KL   = state isolation + DAPO token normalization (no Future-KL)
3. SGRPO - Isolation    = Future-KL weighting + DAPO token normalization (no state isolation)
4. DAPO (baseline)      = DAPO token normalization only

If state isolation is the key contribution:
    Condition 1 ≈ Condition 2 >> Condition 3 ≈ Condition 4

If Future-KL is the key contribution:
    Condition 1 ≈ Condition 3 >> Condition 2 ≈ Condition 4

If both matter:
    Condition 1 > Condition 2 > Condition 4, Condition 1 > Condition 3 > Condition 4

Usage:
    python -m Experiments.ablation_study --steps 300 --seeds 42 123
    python -m Experiments.ablation_study --device cpu --no_wandb
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import TrainingConfig, set_deterministic_seeds
from models.load_model import load_model
from data.gsm8k import GSM8KDataset
from trainer.base_trainer import BaseTrainer


# ── Ablation configurations ─────────────────────────────────────────────────

ABLATION_CONDITIONS = {
    "sgrpo_full": {
        "algorithm": "sgrpo",
        "description": "Full SGRPO: state isolation + Future-KL + DAPO normalization",
        "future_kl_clip_low": 1.0,
        "future_kl_clip_high": 1.2,
    },
    "sgrpo_no_future_kl": {
        "algorithm": "sgrpo",
        "description": "SGRPO without Future-KL (clip_low=clip_high=1.0 → weight=1.0 everywhere)",
        "future_kl_clip_low": 1.0,
        "future_kl_clip_high": 1.0,  # This makes all weights = 1.0
    },
    "dapo_with_future_kl": {
        "algorithm": "dapo",
        "description": "DAPO with Future-KL but NO state isolation (tests if isolation matters)",
        "future_kl_clip_low": 1.0,
        "future_kl_clip_high": 1.2,
        # Note: DAPO uses base_rollout (no isolation) by default
    },
    "dapo_baseline": {
        "algorithm": "dapo",
        "description": "Pure DAPO baseline — no SGRPO components",
        "future_kl_clip_low": 1.0,
        "future_kl_clip_high": 1.0,
    },
}


def run_ablation_condition(
    condition_name: str,
    condition_config: dict,
    seed: int,
    steps: int,
    group_size: int,
    device: str,
    no_wandb: bool = False,
) -> dict:
    """Run a single ablation condition."""
    set_deterministic_seeds(seed)

    config = TrainingConfig(
        algorithm=condition_config["algorithm"],
        seed=seed,
        steps=steps,
        group_size=group_size,
        device=device,
        future_kl_clip_low=condition_config.get("future_kl_clip_low", 1.0),
        future_kl_clip_high=condition_config.get("future_kl_clip_high", 1.2),
        run_name=f"ablation_{condition_name}_seed{seed}",
        wandb_project="rl-algo-comparison-2026",
        no_wandb=no_wandb,
        eval_every=50,
        eval_samples=50,
        checkpoint_every=steps,
    )

    print(f"\n{'─'*60}")
    print(f"ABLATION: {condition_name}")
    print(f"  {condition_config['description']}")
    print(f"  Algorithm: {config.algorithm} | Seed: {seed}")
    print(f"  Future-KL clip: [{config.future_kl_clip_low}, {config.future_kl_clip_high}]")
    print(f"{'─'*60}")

    model, tokenizer = load_model(device=device, dtype=config.dtype)
    train_dataset = GSM8KDataset(split="train")
    eval_dataset = GSM8KDataset(split="test")

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        algorithm=config.algorithm,
        config=config,
    )

    trainer.train()

    del model, trainer
    torch.cuda.empty_cache()

    return {
        "condition": condition_name,
        "algorithm": condition_config["algorithm"],
        "description": condition_config["description"],
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="SGRPO Ablation Study")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+",
        default=list(ABLATION_CONDITIONS.keys()),
        choices=list(ABLATION_CONDITIONS.keys()),
    )
    args = parser.parse_args()

    total = len(args.conditions) * len(args.seeds)
    print(f"\n{'#'*60}")
    print(f"# SGRPO ABLATION STUDY")
    print(f"# Conditions: {len(args.conditions)}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Total runs: {total}")
    print(f"{'#'*60}")

    results = []
    run_idx = 0

    for condition_name in args.conditions:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n>>> Run {run_idx}/{total}")

            result = run_ablation_condition(
                condition_name=condition_name,
                condition_config=ABLATION_CONDITIONS[condition_name],
                seed=seed,
                steps=args.steps,
                group_size=args.group_size,
                device=args.device,
                no_wandb=args.no_wandb,
            )
            results.append(result)

    # Save summary
    os.makedirs("Experiments/results", exist_ok=True)
    out_path = "Experiments/results/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {"steps": args.steps, "group_size": args.group_size, "seeds": args.seeds},
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
