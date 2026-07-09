"""
losses/dapo_loss.py

DAPO: Direct Advantage Policy Optimization loss.

Two changes from GRPO:
1. Token-level normalization instead of sequence-level
   (longer responses get proportionally more gradient)
2. Dynamic sampling filter: skip groups where all rewards
   are identical (std=0 → degenerate advantage estimates)

No KL penalty — deliberately removed in the DAPO paper.

Returns:
    loss:       scalar loss
    advantages: [G] advantages for logging
    metrics:    dict of DAPO-specific metrics for WandB
    (or None if degenerate group)
"""

import torch
from losses.grpo_loss import compute_group_advantages


def is_degenerate_group(rewards: torch.Tensor) -> bool:
    """
    Returns True if all rewards in the group are identical.
    In this case, group-relative advantages are all zero — skip this group.
    This is DAPO's dynamic sampling filter.
    """
    return rewards.std().item() < 1e-8


def compute(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    old_log_probs: torch.Tensor,    # [G, gen_len]
    rewards: torch.Tensor,          # [G]
    attention_mask: torch.Tensor,   # [G, gen_len] — 1 for real tokens, 0 for padding
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """
    DAPO loss with token-level normalization and dynamic sampling filter.

    Returns None if group is degenerate (all same reward) — caller skips this batch.
    Returns (loss, advantages, metrics_dict) otherwise.

    Token-level normalization:
    Instead of mean() over [G, gen_len], divide by total number of real tokens.
    This gives longer correct responses more gradient weight — they contributed
    more tokens to the correct reasoning path.
    """
    if is_degenerate_group(rewards):
        return None

    advantages = compute_group_advantages(rewards)
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)

    unclipped = ratio * advantages_expanded
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_expanded

    per_token_loss = -torch.min(unclipped, clipped)

    # Token-level normalization: sum over real tokens, divide by total real token count
    total_real_tokens = attention_mask.sum().clamp(min=1)
    loss = (per_token_loss * attention_mask).sum() / total_real_tokens

    # ── Metrics for WandB ────────────────────────────────────────────────
    with torch.no_grad():
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
            "dapo/epsilon_low": clip_epsilon,
            "dapo/epsilon_high": clip_epsilon,  # symmetric in standard DAPO
            "dapo/token_level_loss": loss.item(),
            "dapo/kl_removed": True,
        }

    return loss, advantages, metrics
