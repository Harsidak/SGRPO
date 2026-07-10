# Dockerfile — reproducible cloud environment for the SGRPO benchmark
#
# Build:
#   docker build -t sgrpo-bench .
#
# Run the full benchmark (mount /data to persist HF cache + results):
#   docker run --gpus all -v $PWD/data:/data -e WANDB_API_KEY=<key> \
#       sgrpo-bench python -m Experiments.run_benchmark --steps 500
#
# Run a single training job:
#   docker run --gpus all -v $PWD/data:/data -e WANDB_API_KEY=<key> \
#       sgrpo-bench python main.py --algorithm sgrpo --steps 500
#
# The -devel base image is required: mamba-ssm and causal-conv1d compile
# CUDA kernels at install time and need nvcc.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# uv — same resolver the project uses locally, pinned lockfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # HF model + dataset cache on the mounted volume so pulls survive
    # container restarts
    HF_HOME=/data/hf \
    # results also live on the volume by default
    PYTHONUNBUFFERED=1

# ── Dependencies (cached layer — only re-runs when the lock changes) ──────
COPY pyproject.toml uv.lock ./

# Two-phase install: mamba-ssm and causal-conv1d import torch in their
# setup.py, so they cannot build inside uv's isolated build env. Phase 1
# installs everything else (including torch); phase 2 builds the two CUDA
# packages against the already-installed torch.
RUN uv sync --frozen --no-install-project \
        --no-install-package mamba-ssm \
        --no-install-package causal-conv1d \
 && uv sync --frozen --no-install-project --no-build-isolation

# ── Project code ──────────────────────────────────────────────────────────
COPY . .

VOLUME /data

# `uv run <cmd>` executes inside the locked venv
ENTRYPOINT ["uv", "run"]
CMD ["python", "-m", "Experiments.run_benchmark", "--help"]
