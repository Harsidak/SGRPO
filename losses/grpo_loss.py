"""
losses/grpo_loss.py

GRPO: Group Relative Policy Optimization loss.
Adds group-relative advantage estimation on top of PPO's clipped surrogate.

Key difference from PPO:
- No value critic needed
- Advantages computed from reward differences within the group
- Baseline is the group mean reward, not a learned value function

KL penalty: the original GRPO (DeepSeekMath, Shao et al. 2024) includes
beta * KL(pi_theta || pi_ref) against a frozen reference model, using the
unbiased non-negative k3 estimator:

    D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
         = exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1

Omitting it (as this code previously did) makes the baseline collapse on
long runs — which would invalidate any comparison against SGRPO.

Returns:
    loss:       scalar loss
    advantages: [G] advantages for logging
    metrics:    dict of GRPO-specific metrics for WandB
"""

import torch


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute group-relative advantages from a group of rewards.

    A_i = (r_i - mean(r)) / (std(r) + eps)

    Args:
        rewards: [G] tensor of scalar rewards, one per rollout

    Returns:
        advantages: [G] tensor, zero-mean, unit-variance within group

    Note: if std is zero (all rewards identical), advantages are all zero.
    This is the degenerate case DAPO's dynamic sampling filter prevents.
    """
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    return (rewards - mean_r) / std_r


def compute_kl_penalty_k3(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    ref_log_probs: torch.Tensor,    # [G, gen_len]
) -> torch.Tensor:
    """
    Per-token KL(pi_theta || pi_ref) via the k3 estimator (Schulman 2020):

        k3 = exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1

    Unbiased and always >= 0. This is the exact form used in the GRPO
    paper (DeepSeekMath eq. 4). Shared by GRPO and BAPO.
    """
    log_ratio = (ref_log_probs - new_log_probs).clamp(-20.0, 20.0)
    return torch.exp(log_ratio) - log_ratio - 1.0


def compute(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    old_log_probs: torch.Tensor,    # [G, gen_len]
    rewards: torch.Tensor,          # [G] — scalar reward per rollout
    clip_epsilon: float = 0.2,
    ref_log_probs: torch.Tensor | None = None,  # [G, gen_len] — frozen reference
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    GRPO loss with group-relative advantage estimation and reference-model
    KL penalty (k3 estimator, per the DeepSeekMath paper).

    Returns:
        (loss, advantages, metrics_dict)
    """
    advantages = compute_group_advantages(rewards)

    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)

    unclipped = ratio * advantages_expanded
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_expanded

    # Sequence-level normalization (divide by G, average over tokens per sequence)
    policy_loss = -torch.min(unclipped, clipped).mean()

    if ref_log_probs is not None:
        kl_penalty = compute_kl_penalty_k3(new_log_probs, ref_log_probs).mean()
        loss = policy_loss + kl_coef * kl_penalty
    else:
        kl_penalty = torch.tensor(0.0, device=new_log_probs.device)
        loss = policy_loss

    # ── Metrics for WandB ────────────────────────────────────────────────
    with torch.no_grad():
        G = rewards.shape[0]
        clip_fraction = (
            (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
        ).float().mean().item()

        metrics = {
            "train/clip_fraction": clip_fraction,
            "train/ratio_mean": ratio.mean().item(),
            "train/ratio_std": ratio.std().item(),
            "train/ratio_max": ratio.max().item(),
            "train/ratio_min": ratio.min().item(),
            "train/approx_kl": (0.5 * (new_log_probs - old_log_probs).pow(2)).mean().item(),
            "grpo/group_size": G,
            "grpo/group_reward_mean": rewards.mean().item(),
            "grpo/group_reward_std": rewards.std().item(),
            "grpo/group_reward_range": (rewards.max() - rewards.min()).item(),
            "grpo/num_groups": 1,  # per-step, always 1 group
            "grpo/policy_loss": policy_loss.item(),
            "grpo/kl_penalty": kl_penalty.item(),
            "grpo/kl_coef": kl_coef,
            "grpo/has_ref_model": ref_log_probs is not None,
        }

    return loss, advantages, metrics
