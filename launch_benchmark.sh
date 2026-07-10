#!/bin/bash
# launch_benchmark.sh — full SGRPO benchmark sweep on a cloud GPU instance
# (Lambda Labs / RunPod / any Ubuntu box with NVIDIA drivers).
#
# Usage:
#   export WANDB_API_KEY=<key>          # or put it in .env
#   bash launch_benchmark.sh
#
# Overridable knobs:
#   MODELS  — space-separated HF model names (default: 130m; add 1.4b/2.8b
#             on >= 24GB GPUs)
#   STEPS   — training steps per run       (default 500)
#   SEEDS   — seeds per algorithm          (default "42 123 456 789 1337")
#   DATASET — gsm8k | math                 (default gsm8k)
#
# Example full sweep on an H100:
#   MODELS="state-spaces/mamba-130m-hf state-spaces/mamba-1.4b-hf" \
#       STEPS=500 bash launch_benchmark.sh

set -euo pipefail

MODELS=${MODELS:-"state-spaces/mamba-130m-hf"}
STEPS=${STEPS:-500}
SEEDS=${SEEDS:-"42 123 456 789 1337"}
DATASET=${DATASET:-gsm8k}
GROUP_SIZE=${GROUP_SIZE:-4}

# ── Sanity checks ──────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — this script expects a GPU instance." >&2
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs detected: ${NUM_GPUS}"
nvidia-smi -L

if [[ -z "${WANDB_API_KEY:-}" && ! -f .env ]]; then
    echo "WARNING: no WANDB_API_KEY and no .env — runs will use --no_wandb." >&2
    NO_WANDB="--no_wandb"
else
    NO_WANDB=""
fi

# ── Dependencies ───────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Two-phase sync: mamba-ssm / causal-conv1d import torch in setup.py and
# cannot build in an isolated build env — install torch first, then build
# the CUDA extensions against it (same pattern as the Dockerfile).
uv sync --frozen --no-install-project \
    --no-install-package mamba-ssm \
    --no-install-package causal-conv1d
uv sync --frozen --no-install-project --no-build-isolation

# ── Benchmark sweep ────────────────────────────────────────────────────────
# run_benchmark handles the algorithm x seed grid, per-run failure
# isolation, statistics, and result files itself. One invocation per model
# keeps result files separated.
for MODEL in ${MODELS}; do
    MODEL_TAG=$(echo "${MODEL}" | tr '/' '_')
    echo ""
    echo "############################################################"
    echo "# Benchmark: ${MODEL} | steps=${STEPS} | dataset=${DATASET}"
    echo "############################################################"
    uv run python -m Experiments.run_benchmark \
        --algorithms ppo grpo dapo bapo sgrpo \
        --seeds ${SEEDS} \
        --steps "${STEPS}" \
        --group_size "${GROUP_SIZE}" \
        --dataset "${DATASET}" \
        --model_name "${MODEL}" \
        --output_dir "Experiments/results/${MODEL_TAG}" \
        ${NO_WANDB}
done

# ── Supporting experiments (single-GPU, fast relative to the sweep) ───────
uv run python -m Experiments.contamination_probe \
    --num_prompts 10 --group_size 6 ${NO_WANDB}
uv run python -m Experiments.convergence_analysis \
    --steps 100 --group_size "${GROUP_SIZE}" ${NO_WANDB}

echo ""
echo "All done. Results:"
echo "  Experiments/results/<model>/benchmark_results.json"
echo "  Experiments/results/contamination_probe_results.json"
echo "  Experiments/results/convergence_analysis.json"
echo "  Docs/results/main_benchmark_table.md"

# ── Multi-GPU note ─────────────────────────────────────────────────────────
# The sweep above runs each training job on one GPU (the jobs are small and
# the grid parallelism is across runs). For a single LARGE run that needs
# data-parallel training across all GPUs, use Accelerate directly:
#
#   uv run accelerate launch --num_processes ${NUM_GPUS} \
#       main.py --algorithm sgrpo --steps 500 --use_accelerate
