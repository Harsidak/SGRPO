"""
main.py

Entry point for the SGRPO research project.
One command runs any algorithm with identical infrastructure.

Usage:
    python main.py --algorithm ppo   --steps 500
    python main.py --algorithm grpo  --steps 500 --group_size 4
    python main.py --algorithm dapo  --steps 500 --group_size 4
    python main.py --algorithm bapo  --steps 500 --group_size 4
    python main.py --algorithm sgrpo --steps 500 --group_size 4

    # Disable W&B for local testing
    python main.py --algorithm sgrpo --steps 50 --no_wandb --device cpu

Swapping --algorithm is the ONLY change needed to run a different algorithm.
Everything else — model, dataset, trainer, logging — is identical.
This guarantees a fair comparison.
"""

import argparse
import torch

# Config module handles .env loading at import time
from config import TrainingConfig, set_deterministic_seeds, config_from_args
from models.load_model import load_model, detect_architecture
from data.gsm8k import GSM8KDataset
from trainer.base_trainer import BaseTrainer

# Benchmark registry — every entry implements BaseRewardDataset, so the
# trainer needs no changes when a new benchmark is added here.
DATASETS = {
    "gsm8k": GSM8KDataset,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="SGRPO Research: Custom RL Algorithm Trainer"
    )

    # ── Algorithm ────────────────────────────────────────────────────────
    parser.add_argument(
        "--algorithm", type=str, required=True,
        choices=["ppo", "grpo", "dapo", "bapo", "sgrpo"],
        help="Which algorithm to run"
    )

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str,
                        default="state-spaces/mamba-130m-hf",
                        help="HuggingFace model name or path. SSM/hybrid "
                             "models get state isolation under sgrpo; "
                             "transformers fall back to standard rollouts "
                             "(control condition).")

    # ── Dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=sorted(DATASETS.keys()),
                        help="Benchmark to train/eval on. All benchmarks "
                             "use verifiable rewards (no reward model).")

    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--group_size", type=int,   default=4,
                        help="G — rollouts per prompt. Use 1 for PPO.")
    parser.add_argument("--lr",         type=float, default=1e-6)
    parser.add_argument("--max_tokens", type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=1,
                        help="Prompts per step. Keep at 1 for 6GB VRAM.")
    parser.add_argument("--grad_accum", type=int,   default=8,
                        help="Gradient accumulation steps.")
    parser.add_argument("--clip_eps",   type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)

    # ── SGRPO-specific ───────────────────────────────────────────────────
    parser.add_argument("--future_kl_decay",     type=float, default=30.0)
    parser.add_argument("--future_kl_clip_low",  type=float, default=1.0)
    parser.add_argument("--future_kl_clip_high", type=float, default=1.2)

    # ── Reproducibility ──────────────────────────────────────────────────
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     type=str,   default="cuda")

    # ── Evaluation ───────────────────────────────────────────────────────
    parser.add_argument("--eval_every",  type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=50)

    # ── WandB ────────────────────────────────────────────────────────────
    parser.add_argument("--run_name",      type=str, default="trial-1")
    parser.add_argument("--wandb_project", type=str, default="rl-algo-comparison-2026")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Disable W&B logging (for quick local tests)")

    return parser.parse_args()


def main():
    args = get_args()
    config = config_from_args(args)

    # ── Reproducibility ──────────────────────────────────────────────────
    set_deterministic_seeds(config.seed, config.deterministic)

    print(f"\n{'='*60}")
    print(f"SGRPO Research | Algorithm: {config.algorithm.upper()}")
    print(f"Model: {config.model_name} | Dataset: {config.dataset}")
    print(f"Steps: {config.steps} | Group size: {config.group_size}")
    print(f"LR: {config.lr} | Clip eps: {config.clip_eps}")
    print(f"Seed: {config.seed} | Device: {config.device}")
    print(f"Eval every: {config.eval_every} steps")
    if config.algorithm == "sgrpo":
        print(f"Future-KL: decay={config.future_kl_decay}, "
              f"clip=[{config.future_kl_clip_low}, {config.future_kl_clip_high}]")
    print(f"{'='*60}\n")

    # ── Load model and data ──────────────────────────────────────────────
    model, tokenizer = load_model(
        device=config.device, dtype=config.dtype,
        model_name=config.model_name,
    )
    arch = detect_architecture(model)
    dataset_cls = DATASETS[config.dataset]
    train_dataset = dataset_cls(split="train")
    eval_dataset = dataset_cls(split="test")

    # ── Build trainer ────────────────────────────────────────────────────
    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        algorithm=config.algorithm,
        config=config,
        arch=arch,
    )

    trainer.train()


if __name__ == "__main__":
    main()
