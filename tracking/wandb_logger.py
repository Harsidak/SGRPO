"""
tracking/wandb_logger.py

Research-grade Weights & Biases logging for the SGRPO project.

Implements EVERY metric from Docs/wandb_tracking_spec.md:
- Universal metrics (every step): loss, entropy, KL, ratios, rewards, advantages
- Algorithm-specific metrics (every step): per-algorithm diagnostics
- Gradient health (every 10 steps): grad norms, clip counts
- Histograms (every 50 steps): distributions of key quantities
- Evaluation metrics (every eval_every steps): accuracy, response quality
- System metrics (every step): GPU memory, throughput, timing
- Artifacts: model checkpoints, code snapshots, sample generation tables

RULE: Never call wandb.log() directly from any other module.
All logging goes through this module's functions.
"""

import os
import json
import time
import logging
from typing import Optional
from dataclasses import asdict

import torch
import wandb

logger = logging.getLogger(__name__)


# ── Module state ─────────────────────────────────────────────────────────────
_run_active = False
_grad_clip_count = 0
_degenerate_group_count = 0
_total_groups = 0

# Local JSONL metrics sink. Independent of W&B: active for --no_wandb runs
# (where it is the only record) and alongside W&B runs. One JSON object per
# line, tagged with a "kind" field (metadata / step / degenerate / eval).
# Experiments/local_benchmark.py consumes these files to build offline
# comparison graphs, so every algorithm writes the same schema here that it
# sends to W&B.
_local_sink = None
_local_sink_path = None


def init_local_sink(
    algorithm: str,
    run_name: str,
    out_dir: Optional[str] = None,
) -> str:
    """
    Open a local JSONL metrics file for this run.

    out_dir precedence: explicit arg > SGRPO_LOCAL_LOG_DIR env var (set by
    Experiments/local_benchmark.py to group one benchmark's runs together) >
    Experiments/results/local_logs.
    """
    global _local_sink, _local_sink_path
    if out_dir is None:
        out_dir = os.environ.get(
            "SGRPO_LOCAL_LOG_DIR",
            os.path.join("Experiments", "results", "local_logs"),
        )
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _local_sink_path = os.path.join(
        out_dir, f"{algorithm}_{run_name}_{stamp}.jsonl"
    )
    _local_sink = open(_local_sink_path, "a", encoding="utf-8")
    return _local_sink_path


def _local_log(kind: str, payload: dict) -> None:
    """Append one record to the local sink, keeping JSON-safe scalars only."""
    if _local_sink is None:
        return
    record = {"kind": kind, "wall_time": time.time()}
    for k, v in payload.items():
        if v is None or isinstance(v, (int, float, str, bool)):
            record[k] = v
    _local_sink.write(json.dumps(record) + "\n")
    _local_sink.flush()


def init_run(
    algorithm: str,
    run_name: str,
    config: dict,
    project: str = "SGRPO",
) -> None:
    """
    Initialize a W&B run with the full config from wandb_tracking_spec.md.

    Config includes all hyperparameters, environment info, hardware info,
    and git commit hash for full reproducibility.
    """
    global _run_active, _grad_clip_count, _degenerate_group_count, _total_groups
    _grad_clip_count = 0
    _degenerate_group_count = 0
    _total_groups = 0

    wandb.init(
        project=project,
        name=f"{algorithm}_{run_name}",
        tags=[algorithm, "mamba-130m", "gsm8k", "math-rl", "comparison"],
        config=config,
        save_code=True,
    )

    # Define custom x-axis for clarity
    wandb.define_metric("train/*", step_metric="sys/step")
    wandb.define_metric("rollout/*", step_metric="sys/step")
    wandb.define_metric("eval/*", step_metric="sys/step")
    wandb.define_metric("system/*", step_metric="sys/step")
    wandb.define_metric("hist/*", step_metric="sys/step")

    # Save all Python source files as code artifact
    try:
        _save_code_snapshot()
    except Exception as e:
        logger.warning(f"Failed to save code snapshot: {e}")

    _run_active = True
    logger.info(f"W&B run initialized: {algorithm}_{run_name} in project {project}")


def _save_code_snapshot():
    """Save all .py files as a W&B artifact for reproducibility."""
    artifact = wandb.Artifact("source_code", type="code")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root, dirs, files in os.walk(project_root):
        # Skip hidden dirs, __pycache__, .venv, etc.
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))
                   and d not in ('venv', '.venv', 'node_modules', '.git')]
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, project_root)
                artifact.add_file(filepath, name=relpath)
    wandb.log_artifact(artifact)


def log_rollout(
    step: int,
    algorithm: str,
    # ── Reward statistics (from the rollout group) ───────────────────────
    reward_mean: float,
    reward_std: float,
    reward_max: float,
    reward_min: float,
    reward_positive_fraction: Optional[float] = None,
    # ── Response statistics ──────────────────────────────────────────────
    response_length_mean: Optional[float] = None,
    response_length_std: Optional[float] = None,
    response_length_max: Optional[float] = None,
    response_length_min: Optional[float] = None,
    unique_tokens_ratio: Optional[float] = None,
    # ── Rollout phase timing / meta ──────────────────────────────────────
    generation_time: Optional[float] = None,
    group_size: Optional[int] = None,
    is_degenerate: bool = False,
) -> None:
    """
    Log rollout-phase RL metrics for EVERY training step — including steps
    that are subsequently skipped as degenerate groups.

    log_step() only fires when an optimizer update happens; early in
    training (untrained model, all-zero rewards) nearly every dapo/bapo/
    sgrpo step is degenerate, so without this function the W&B run shows
    almost nothing but system/GPU metrics. rollout/* charts fill that gap:
    reward signal, response lengths, and degenerate rate are visible from
    step 0 regardless of whether the optimization phase ran.
    """
    if not _run_active and _local_sink is None:
        return

    payload = {
        "sys/step": step,
        "sys/algorithm": algorithm,
        "rollout/reward_mean": reward_mean,
        "rollout/reward_std": reward_std,
        "rollout/reward_max": reward_max,
        "rollout/reward_min": reward_min,
        "rollout/reward_positive_fraction": reward_positive_fraction,
        "rollout/response_length_mean": response_length_mean,
        "rollout/response_length_std": response_length_std,
        "rollout/response_length_max": response_length_max,
        "rollout/response_length_min": response_length_min,
        "rollout/unique_tokens_ratio": unique_tokens_ratio,
        "rollout/generation_time": generation_time,
        "rollout/group_size": group_size,
        "rollout/is_degenerate": int(is_degenerate),
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    _local_log("rollout", payload)
    if _run_active:
        wandb.log(payload)


def log_step(
    step: int,
    algorithm: str,
    # ── Core training metrics ────────────────────────────────────────────
    loss: float,
    policy_loss: Optional[float] = None,
    kl_loss: Optional[float] = None,
    entropy: Optional[float] = None,
    # ── Clipping / ratio statistics ──────────────────────────────────────
    clip_fraction: Optional[float] = None,
    approx_kl: Optional[float] = None,
    ratio_mean: Optional[float] = None,
    ratio_std: Optional[float] = None,
    ratio_max: Optional[float] = None,
    ratio_min: Optional[float] = None,
    # ── Reward statistics ────────────────────────────────────────────────
    reward_mean: float = 0.0,
    reward_std: float = 0.0,
    reward_max: Optional[float] = None,
    reward_min: Optional[float] = None,
    # ── Advantage statistics ─────────────────────────────────────────────
    advantage_mean: float = 0.0,
    advantage_std: float = 0.0,
    advantage_max: Optional[float] = None,
    advantage_min: Optional[float] = None,
    positive_advantage_ratio: Optional[float] = None,
    # ── Gradient health ──────────────────────────────────────────────────
    grad_norm: Optional[float] = None,
    grad_norm_policy: Optional[float] = None,
    grad_was_clipped: bool = False,
    # ── Response statistics ──────────────────────────────────────────────
    response_length_mean: Optional[float] = None,
    response_length_std: Optional[float] = None,
    response_length_max: Optional[float] = None,
    response_length_min: Optional[float] = None,
    unique_tokens_ratio: Optional[float] = None,
    # ── Timing ───────────────────────────────────────────────────────────
    step_time: Optional[float] = None,
    generation_time: Optional[float] = None,
    throughput: Optional[float] = None,
    # ── Algorithm-specific extras ────────────────────────────────────────
    algo_metrics: Optional[dict] = None,
) -> None:
    """
    Log one training step. Every algorithm calls this with the same schema.

    Optional fields are None for algorithms that don't compute them.
    The WandB column still exists for cross-run comparison alignment.
    """
    global _grad_clip_count, _total_groups
    if not _run_active and _local_sink is None:
        return

    _total_groups += 1
    if grad_was_clipped:
        _grad_clip_count += 1

    # ── Build payload ────────────────────────────────────────────────────
    payload = {
        "sys/step": step,
        "sys/algorithm": algorithm,
        # Training dynamics
        "train/loss": loss,
        "train/policy_loss": policy_loss,
        "train/kl_loss": kl_loss,
        "train/entropy": entropy,
        "train/clip_fraction": clip_fraction,
        "train/approx_kl": approx_kl,
        "train/ratio_mean": ratio_mean,
        "train/ratio_std": ratio_std,
        "train/ratio_max": ratio_max,
        "train/ratio_min": ratio_min,
        # Rewards
        "train/reward_mean": reward_mean,
        "train/reward_std": reward_std,
        "train/reward_max": reward_max,
        "train/reward_min": reward_min,
        # Advantages
        "train/advantage_mean": advantage_mean,
        "train/advantage_std": advantage_std,
        "train/advantage_max": advantage_max,
        "train/advantage_min": advantage_min,
        "train/positive_advantage_ratio": positive_advantage_ratio,
        # Gradients
        "train/grad_norm": grad_norm,
        "train/grad_norm_policy": grad_norm_policy,
        "train/grad_clip_count": _grad_clip_count,
        # Response statistics
        "train/response_length_mean": response_length_mean,
        "train/response_length_std": response_length_std,
        "train/response_length_max": response_length_max,
        "train/response_length_min": response_length_min,
        "train/unique_tokens_ratio": unique_tokens_ratio,
        # Timing
        "system/step_time": step_time,
        "system/generation_time": generation_time,
        "system/throughput": throughput,
    }

    # ── System metrics ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        payload["system/gpu_memory_allocated"] = (
            torch.cuda.memory_allocated() / 1e9
        )
        payload["system/gpu_memory_reserved"] = (
            torch.cuda.memory_reserved() / 1e9
        )

    # ── Algorithm-specific metrics ───────────────────────────────────────
    if algo_metrics:
        payload.update(algo_metrics)

    # Remove None values to keep WandB clean but preserve schema
    payload = {k: v for k, v in payload.items() if v is not None}

    _local_log("step", payload)
    if _run_active:
        # No explicit step= — every chart uses sys/step via define_metric.
        # Mixing explicit steps with step-less calls (log_run_metadata) makes
        # wandb silently drop any log whose step is behind its internal
        # counter, which is how early-run RL metrics went missing.
        wandb.log(payload)


def log_run_metadata(algorithm: str, arch: str, arch_branch: str) -> None:
    """
    Log one-time run metadata at trainer start.

    Records the detected model architecture and which rollout branch the run
    took, so W&B captures the control-vs-treatment condition per run:
      - arch_branch == "isolated": SGRPO on an SSM/hybrid model — state
        isolation active (the treatment).
      - arch_branch == "standard": SGRPO on a stateless transformer —
        isolation is a no-op, degenerates to GRPO (the paper's control), OR
        any non-SGRPO algorithm.

    Logged once (not per step), so it uses W&B's default step counter.
    """
    payload = {
        "sys/architecture": arch,
        "sys/architecture_branch": arch_branch,
        "sys/algorithm": algorithm,
    }
    _local_log("metadata", payload)
    if not _run_active:
        return
    wandb.log({
        "sys/architecture": arch,
        "sys/architecture_branch": arch_branch,
    })


def log_degenerate_group(step: int):
    """Track when a group is skipped due to degenerate rewards (all same)."""
    global _degenerate_group_count, _total_groups
    _degenerate_group_count += 1
    _total_groups += 1

    payload = {
        "train/degenerate_group_rate": _degenerate_group_count / max(_total_groups, 1),
        "sys/step": step,
    }
    _local_log("degenerate", payload)
    if _run_active:
        wandb.log(payload)


def log_histograms(
    step: int,
    advantages: Optional[torch.Tensor] = None,
    rewards: Optional[torch.Tensor] = None,
    ratios: Optional[torch.Tensor] = None,
    response_lengths: Optional[list] = None,
    token_probs: Optional[torch.Tensor] = None,
    gradients: Optional[dict] = None,
) -> None:
    """
    Log distribution histograms every histogram_every steps.
    These are critical for detecting:
    - advantage collapse (all near zero → degenerate training)
    - ratio explosion (extreme importance sampling weights)
    - response length mode collapse
    """
    if not _run_active:
        return

    payload = {"sys/step": step}

    if advantages is not None:
        payload["hist/advantages"] = wandb.Histogram(
            advantages.detach().cpu().float().numpy()
        )
    if rewards is not None:
        payload["hist/rewards"] = wandb.Histogram(
            rewards.detach().cpu().float().numpy()
        )
    if ratios is not None:
        payload["hist/ratios"] = wandb.Histogram(
            ratios.detach().cpu().float().numpy().clip(-10, 10)
        )
    if response_lengths is not None:
        payload["hist/response_lengths"] = wandb.Histogram(response_lengths)
    if token_probs is not None:
        payload["hist/token_probs"] = wandb.Histogram(
            token_probs.detach().cpu().float().numpy()
        )

    wandb.log(payload)


def log_eval(
    step: int,
    algorithm: str,
    # ── Task performance ─────────────────────────────────────────────────
    gsm8k_accuracy: Optional[float] = None,
    average_reward: Optional[float] = None,
    correct_count: Optional[int] = None,
    total_count: Optional[int] = None,
    # ── Reasoning quality ────────────────────────────────────────────────
    response_length_mean: Optional[float] = None,
    response_length_median: Optional[float] = None,
    reasoning_steps_mean: Optional[float] = None,
    reflection_count: Optional[float] = None,
    self_correction_rate: Optional[float] = None,
    # ── Distribution analysis ────────────────────────────────────────────
    entropy: Optional[float] = None,
    kl_from_ref: Optional[float] = None,
) -> None:
    """Log evaluation metrics on held-out test set."""
    if not _run_active and _local_sink is None:
        return

    payload = {
        "sys/step": step,
        "sys/algorithm": algorithm,
        "eval/gsm8k_accuracy": gsm8k_accuracy,
        "eval/average_reward": average_reward,
        "eval/correct_count": correct_count,
        "eval/total_count": total_count,
        "eval/response_length_mean": response_length_mean,
        "eval/response_length_median": response_length_median,
        "eval/reasoning_steps_mean": reasoning_steps_mean,
        "eval/reflection_count": reflection_count,
        "eval/self_correction_rate": self_correction_rate,
        "eval/entropy": entropy,
        "eval/kl_from_ref": kl_from_ref,
    }

    payload = {k: v for k, v in payload.items() if v is not None}
    _local_log("eval", payload)
    if _run_active:
        wandb.log(payload)


def log_sample_table(
    step: int,
    samples: list[dict],
) -> None:
    """
    Log a W&B table of sample generations for qualitative inspection.

    Each sample dict should have: prompt, response, reward, response_length
    """
    if not _run_active:
        return

    columns = ["step", "prompt", "response", "reward", "response_length"]
    table = wandb.Table(columns=columns)

    for s in samples:
        table.add_data(
            step,
            s.get("prompt", "")[:500],   # truncate for readability
            s.get("response", "")[:1000],
            s.get("reward", 0.0),
            s.get("response_length", 0),
        )

    wandb.log({"eval/samples": table, "sys/step": step})


def log_checkpoint(
    step: int,
    algorithm: str,
    checkpoint_path: str,
) -> None:
    """Log a model checkpoint as a W&B artifact."""
    if not _run_active:
        return

    artifact = wandb.Artifact(
        f"{algorithm}_checkpoint_step{step}",
        type="model",
        metadata={"step": step, "algorithm": algorithm},
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    logger.info(f"Checkpoint artifact logged: step {step}")


def finish() -> None:
    """Finish the current W&B run and close the local metrics sink."""
    global _run_active, _local_sink
    if _local_sink is not None:
        _local_sink.close()
        _local_sink = None
        logger.info(f"Local metrics written: {_local_sink_path}")
    if _run_active:
        wandb.finish()
        _run_active = False
        logger.info("W&B run finished.")
