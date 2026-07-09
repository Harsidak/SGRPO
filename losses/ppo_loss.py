"""
losses/ppo_loss.py

RLHF PPO clipped surrogate loss.

Role in this project: infrastructure validator.
If this trains and reward goes up, the shared trainer is confirmed working.
Every bug found here is a bug not found later inside SGRPO's math.

KL penalty: since this project is RL-for-LLMs, we use RLHF PPO
(InstructGPT / Ouyang et al. 2022), which adds beta * KL(pi_theta || pi_ref)
against a frozen copy of the pretrained model to prevent the policy from
drifting away from the pretrained distribution. Vanilla PPO (Schulman 2017)
has no such penalty; using it as an LLM baseline would let it collapse on
long runs and invalidate the comparison.

Returns:
    loss:       scalar loss
    advantages: [G] advantages for logging
    metrics:    dict of PPO-specific metrics for WandB
"""

import torch


def compute(
    new_log_probs: torch.Tensor,    # [G, gen_len] — log probs under current policy
    old_log_probs: torch.Tensor,    # [G, gen_len] — log probs stored during rollout
    advantages: torch.Tensor,       # [G] — one advantage per rollout
    clip_epsilon: float = 0.2,
    ref_log_probs: torch.Tensor | None = None,  # [G, gen_len] — frozen reference model
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    RLHF PPO clipped surrogate objective.

    r_t = pi_theta(a_t) / pi_theta_old(a_t) = exp(new_log_prob - old_log_prob)
    L = -E[min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)] + beta * KL(pi_theta || pi_ref)

    KL uses the k1 estimator E[log pi_theta - log pi_ref] as in InstructGPT.

    Returns:
        (loss, advantages, metrics_dict)
    """
    # Importance sampling ratio: [G, gen_len]
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Expand advantages from [G] to [G, gen_len] for token-level multiplication
    advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)

    # Clipped surrogate
    unclipped = ratio * advantages_expanded
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_expanded

    policy_loss = -torch.min(unclipped, clipped).mean()

    # RLHF KL penalty against frozen reference model (k1 estimator)
    if ref_log_probs is not None:
        kl_penalty = (new_log_probs - ref_log_probs).mean()
        loss = policy_loss + kl_coef * kl_penalty
    else:
        kl_penalty = torch.tensor(0.0, device=new_log_probs.device)
        loss = policy_loss

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
            "ppo/policy_loss": policy_loss.item(),
            "ppo/kl_penalty": kl_penalty.item(),
            "ppo/kl_coef": kl_coef,
            "ppo/has_ref_model": ref_log_probs is not None,
        }

    return loss, advantages, metrics
