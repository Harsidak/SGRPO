"""
config.py

Centralized configuration for the SGRPO research project.

Responsibilities:
1. Typed hyperparameter defaults in a dataclass
2. Deterministic seed propagation (torch, numpy, random, CUDA)
3. .env loading for WandB API key
4. Hardware and environment info capture for reproducibility
5. Serialization to dict for WandB config logging
6. Git commit hash capture when available

Every experiment must be reproducible by another researcher
without additional instructions. This module enforces that.
"""

import os
import sys
import random
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv


# ── Load .env at import time ────────────────────────────────────────────────
# This ensures WANDB_API_KEY is available before any wandb import
load_dotenv()

# Map the .env key name to what wandb expects
_wandb_key = os.environ.get("wandb_api")
if _wandb_key and not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = _wandb_key


@dataclass
class TrainingConfig:
    """
    All hyperparameters for a single training run.
    Designed so that two runs with identical configs produce identical results
    (given deterministic seeds and hardware).
    """

    # ── Algorithm ────────────────────────────────────────────────────────────
    algorithm: str = "sgrpo"  # ppo | grpo | dapo | bapo | sgrpo

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "state-spaces/mamba-130m-hf"
    dtype: str = "bfloat16"  # bfloat16 | float32

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "gsm8k"    # any key in main.py's DATASETS registry

    # ── Training ─────────────────────────────────────────────────────────────
    steps: int = 500
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    batch_size: int = 1       # prompts per step (keep 1 for 6GB VRAM)
    grad_accum: int = 8       # gradient accumulation steps
    warmup_steps: int = 0
    # PPO-style inner optimization epochs: reuse each rollout batch mu times.
    # mu == 1 (default) is the legacy single-pass behavior (bit-identical).
    # mu > 1 re-optimizes the same rollouts, stepping the optimizer after each
    # inner epoch so theta moves — this is what makes the importance ratio and
    # Future-KL non-trivial (with mu == 1 they are identically 1.0 / 0.0).
    # Default kept at 1 for backward compatibility; 2-4 recommended for real
    # runs (see sgrpo_research_roadmap.md Fix 3).
    inner_epochs: int = 1

    # ── RL ────────────────────────────────────────────────────────────────────
    group_size: int = 4       # G — rollouts per prompt (1 for PPO)
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    temperature: float = 1.0
    max_tokens: int = 256     # max generated tokens per rollout

    # ── SGRPO-specific ───────────────────────────────────────────────────────
    future_kl_decay: float = 30.0
    future_kl_clip_low: float = 1.0
    future_kl_clip_high: float = 1.2

    # ── DAPO-specific ────────────────────────────────────────────────────────
    dapo_epsilon_high: float = 0.28

    # ── BAPO-specific (paper notation in comments) ───────────────────────────
    bapo_rho_target: float = 0.5    # rho_0 — positive token contribution threshold
    bapo_c_low_min: float = 0.8     # a- — initial (tightest) lower clipping bound
    bapo_c_low_max: float = 0.9     # b- — maximum the lower bound may be raised to
    bapo_c_high_min: float = 1.2    # a+ — initial (tightest) upper clipping bound
    bapo_c_high_max: float = 1.32   # b+ — maximum the upper bound may be widened to
    bapo_delta_high: float = 0.01   # d1 — step size of the upper bound
    bapo_delta_low: float = 0.01    # d2 — step size of the lower bound

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_every: int = 50      # evaluate every N steps
    eval_samples: int = 50    # number of test prompts per eval
    histogram_every: int = 50
    sample_table_every: int = 100
    checkpoint_every: int = 250

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42
    deterministic: bool = True  # torch.use_deterministic_algorithms

    # ── System ───────────────────────────────────────────────────────────────
    device: str = "cpu"
    # Multi-GPU / cloud training via HuggingFace Accelerate. Launch with
    # `accelerate launch main.py --use_accelerate ...`. When False (default)
    # the trainer is plain single-device PyTorch — local behavior unchanged.
    use_accelerate: bool = False

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_project: str = "rl-algo-comparison-2026"
    run_name: str = "trial-1"
    no_wandb: bool = False

    # ── Paths ────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        """Enforce algorithm-specific constraints."""
        if self.algorithm == "ppo" and self.group_size != 1:
            print("Warning: PPO uses group_size=1. Overriding.")
            self.group_size = 1

    def to_dict(self) -> dict:
        """Serialize to dict for WandB config, including environment info."""
        d = asdict(self)
        d.update(self._get_environment_info())
        return d

    def _get_environment_info(self) -> dict:
        """Capture full environment for reproducibility."""
        info = {
            "env/python_version": sys.version,
            "env/platform": platform.platform(),
            "env/torch_version": torch.__version__,
            "env/cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["env/gpu_name"] = torch.cuda.get_device_name(0)
            info["env/gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            info["env/cuda_version"] = torch.version.cuda or "N/A"

        # Git commit hash
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).decode().strip()
            info["env/git_commit"] = commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            info["env/git_commit"] = "unknown"

        try:
            import transformers
            info["env/transformers_version"] = transformers.__version__
        except ImportError:
            pass

        return info


def set_deterministic_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU and CUDA generators
    - CUDA deterministic algorithms (when deterministic=True)
    - CUBLAS workspace config for determinism
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS deterministic — required for full reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions
            torch.use_deterministic_algorithms(True)


def config_from_args(args) -> TrainingConfig:
    """Convert argparse Namespace to TrainingConfig."""
    return TrainingConfig(**{
        k: v for k, v in vars(args).items()
        if k in TrainingConfig.__dataclass_fields__
    })
