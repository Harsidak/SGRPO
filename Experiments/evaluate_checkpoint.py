"""
Experiments/evaluate_checkpoint.py

Standalone test-set evaluation of saved model checkpoints.

The benchmark (Experiments/run_benchmark.py) saves one final checkpoint per
algorithm x seed run and records its path in benchmark_results.json. This
script loads those checkpoints and measures accuracy on the held-out test
split — with a sample budget you choose, independent of the small in-training
eval (eval_samples=50).

Usage:
    # Evaluate every checkpoint recorded by a benchmark sweep
    python -m Experiments.evaluate_checkpoint \
        --results Experiments/results/benchmark_results.json --eval_samples 200

    # Evaluate a single checkpoint
    python -m Experiments.evaluate_checkpoint \
        --checkpoint Experiments/results/checkpoints/sgrpo_bench_seed42_..._step499.pt

Outputs a per-checkpoint accuracy report to the console and a JSON file
(default: Experiments/results/checkpoint_eval.json).
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime

# Allow `python Experiments/evaluate_checkpoint.py` as well as -m invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import set_deterministic_seeds
from models.load_model import load_model
from data import DATASETS


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_name: str | None,
    eval_samples: int,
    max_tokens: int | None,
    device: str,
    seed: int,
) -> dict:
    """Load one checkpoint and evaluate greedy-decoding accuracy on the
    test split. dataset/max_tokens default to the values stored in the
    checkpoint's training config."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = ckpt.get("config", {})
    algorithm = ckpt.get("algorithm", "unknown")
    model_name = train_config.get("model_name", "state-spaces/mamba-130m-hf")
    dataset_name = dataset_name or train_config.get("dataset", "gsm8k")
    max_tokens = max_tokens or train_config.get("max_tokens", 256)
    dtype = train_config.get("dtype", "bfloat16")

    print(f"  Algorithm: {algorithm} | trained seed: {train_config.get('seed')}"
          f" | step: {ckpt.get('step')}")
    print(f"  Model: {model_name} | Dataset: {dataset_name} (test split)")

    # Seed the eval itself so the sampled test subset is identical across
    # checkpoints — accuracy differences then reflect the models, not the
    # question draw.
    set_deterministic_seeds(seed, deterministic=True)

    model, tokenizer = load_model(
        device=device, dtype=dtype, model_name=model_name
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eval_dataset = DATASETS[dataset_name](split="test")
    reward_fn = eval_dataset.compute_group_rewards
    eval_data = list(eval_dataset)
    random.Random(seed).shuffle(eval_data)
    eval_data = eval_data[:eval_samples]

    correct = 0
    rewards = []
    lengths = []

    with torch.no_grad():
        for i, item in enumerate(eval_data):
            inputs = tokenizer(
                item["prompt"], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)

            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy, same as in-training eval
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_text = tokenizer.decode(
                output[0, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            reward = reward_fn([gen_text], item["answer"])[0]
            rewards.append(reward)
            correct += int(reward > 0.5)
            lengths.append(output.shape[1] - inputs.input_ids.shape[1])

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(eval_data)} | "
                      f"running accuracy: {correct / (i + 1):.3f}")

    total = len(eval_data)
    accuracy = correct / max(total, 1)
    avg_reward = sum(rewards) / max(len(rewards), 1)

    print(f"  RESULT: accuracy={accuracy:.3f} ({correct}/{total}) | "
          f"avg_reward={avg_reward:.3f}")

    # Free GPU memory before the next checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_path,
        "algorithm": algorithm,
        "trained_seed": train_config.get("seed"),
        "trained_steps": ckpt.get("step"),
        "model_name": model_name,
        "dataset": dataset_name,
        "eval_samples": total,
        "eval_seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "average_reward": avg_reward,
        "response_length_mean": sum(lengths) / max(len(lengths), 1),
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints on the held-out test set"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str,
                     help="Path to a single .pt checkpoint")
    src.add_argument("--results", type=str,
                     help="benchmark_results.json — evaluates every "
                          "checkpoint recorded in it")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=sorted(DATASETS.keys()),
                        help="Override the dataset stored in the checkpoint")
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override the max_tokens stored in the checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the test-subset draw (identical "
                             "across checkpoints)")
    parser.add_argument("--output", type=str,
                        default=os.path.join("Experiments", "results",
                                             "checkpoint_eval.json"))
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        with open(args.results, encoding="utf-8") as f:
            bench = json.load(f)
        checkpoint_paths = [
            r["final_checkpoint"] for r in bench.get("runs", [])
            if r.get("final_checkpoint")
        ]
        if not checkpoint_paths:
            print(f"No checkpoints recorded in {args.results} — was the "
                  f"benchmark run before checkpoint tracking was added?")
            return

    print(f"Evaluating {len(checkpoint_paths)} checkpoint(s), "
          f"{args.eval_samples} test samples each.")

    results = []
    for path in checkpoint_paths:
        try:
            results.append(evaluate_checkpoint(
                checkpoint_path=path,
                dataset_name=args.dataset,
                eval_samples=args.eval_samples,
                max_tokens=args.max_tokens,
                device=args.device,
                seed=args.seed,
            ))
        except Exception as exc:  # one bad checkpoint must not kill the sweep
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append({
                "checkpoint": path,
                "error": f"{type(exc).__name__}: {exc}",
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_samples": args.eval_samples,
            "eval_seed": args.seed,
            "results": results,
        }, f, indent=2)
    print(f"\nEvaluation results written: {args.output}")

    # Console summary grouped by algorithm
    ok = [r for r in results if not r.get("error")]
    if ok:
        print(f"\n{'='*60}")
        print("CHECKPOINT EVALUATION SUMMARY")
        print(f"{'='*60}")
        by_algo: dict[str, list] = {}
        for r in ok:
            by_algo.setdefault(r["algorithm"], []).append(r["accuracy"])
        for algo, accs in sorted(by_algo.items()):
            mean = sum(accs) / len(accs)
            print(f"  {algo.upper():6s} | acc = {mean:.3f} "
                  f"(n={len(accs)}: {', '.join(f'{a:.3f}' for a in accs)})")
    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"\n  {len(failed)} checkpoint(s) FAILED:")
        for r in failed:
            print(f"    {r['checkpoint']}: {r['error']}")


if __name__ == "__main__":
    main()
