"""
losses/bapo_loss.py

BAPO: Balanced Advantage Policy Optimization loss.

One change from DAPO:
Adaptive clipping bounds based on the ratio of positive to negative
advantage tokens in the current batch.

Why: fixed clipping treats positive and negative advantage updates
symmetrically. But negative advantage tokens (suppressing bad behavior)
tend to dominate early in training, producing unstable gradients.
Adaptive bounds rebalance this.

eps_high = eps * (N_neg / (N_pos + N_neg))
eps_low  = eps * (N_pos / (N_pos + N_neg))

Note: BAPO is a standalone baseline in this project.
Its adaptive clipping is NOT composed with SGRPO's Future-KL weighting
because the adaptive bound derivation assumes uniform advantage weighting
per token — Future-KL changes this and the correct bounds under weighted
advantages have not been re-derived. Including them without justification
would be mathematically unjustified.

Returns:
    loss:       scalar loss
    advantages: [G] advantages for logging
    metrics:    dict of BAPO-specific metrics for WandB
    (or None if degenerate group)
"""

"""
Algorithm Working as per Research Paper: BAPO
Input: Initialized LLM policy 𝜋𝜃, training dataset D, reward function 𝑅, staleness 𝐸,
    movable range of clipping bounds [𝑎−, 𝑏−] and [𝑎+, 𝑏+], step size of upper bound 𝛿1,
    step size of lower bound 𝛿2, positive token contribution threshold 𝜌0
1 for step 𝑠 = 1...𝑆 do
2   Procedure Sample and filter out responses
3       Update the old LLM policy 𝜋𝜃rollout ← 𝜋𝜃 ;
4       Sample the 𝑠-th batch D𝑠 from D ;
5       Sample 𝐺 responses {𝒚𝑖}𝐺𝑖=1 ∼ 𝜋𝜃rollout (·|𝒙), where 𝒙 ∈ D𝑠 ;
6       Compute reward and advantage for each 𝒚𝑖 based on reward function 𝑅 ;
7   for staleness = 0...𝐸 do
8   Procedure Dynamically adjusting the clipping bounds 𝑐high and 𝑐low
9       Initialize clipping bounds 𝑐low = 𝑎− and 𝑐high = 𝑎+ ;
10      while the positive token contribution 𝜌 < 𝜌0 and 𝑐low + 𝛿2 ≤ 𝑏−
11      do
12          if 𝑐high + 𝛿1 ≤ 𝑏+ then
13              𝑐high ← 𝑐high + 𝛿1
14          else
15              𝑐low ← 𝑐low + 𝛿2
16          end
17      end
18      Procedure Update the LLM policy 𝜋𝜃
19          Update the LLM policy 𝜋𝜃 by maximizing the following objective:
20              𝐽BAPO(𝜃) = 𝔼𝒚∼𝜋𝜃rollout ( · |𝒙) Summation from t=1 to T ( min(𝑟𝑡 · 𝐴𝑡 , clip(𝑟𝑡 , 𝑐low, 𝑐high) · 𝐴𝑡) ) 
21     end
22  end
"""

import torch
from losses.dapo_loss import is_degenerate_group
from losses.grpo_loss import compute_group_advantages, compute_kl_penalty_k3


def compute_adaptive_bounds(
    advantages_expanded: torch.Tensor,  # [G, gen_len]
    base_epsilon: float,
) -> tuple[float, float]:
    """
    Compute adaptive clipping bounds based on positive/negative advantage split.

    Args:
        advantages_expanded: [G, gen_len] advantage values per token
        base_epsilon:        base clipping bound (e.g. 0.2)

    Returns:
        (eps_low, eps_high) — asymmetric bounds
    """
    n_pos = (advantages_expanded > 0).float().sum().item()
    n_neg = (advantages_expanded < 0).float().sum().item()
    total = n_pos + n_neg + 1e-8

    eps_high = base_epsilon * (n_neg / total)
    eps_low = base_epsilon * (n_pos / total)

    # Clamp to prevent degenerate bounds
    eps_high = max(eps_high, 0.01)
    eps_low = max(eps_low, 0.01)

    return eps_low, eps_high


def compute(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    old_log_probs: torch.Tensor,    # [G, gen_len]
    rewards: torch.Tensor,          # [G]
    attention_mask: torch.Tensor,   # [G, gen_len]
    base_epsilon: float = 0.2,
    ref_log_probs: torch.Tensor | None = None,  # [G, gen_len] — frozen reference
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """
    BAPO loss with adaptive clipping bounds and reference-model KL penalty
    (inherited from GRPO's foundation — the original BAPO paper keeps it).
    Returns None if group is degenerate.
    Returns (loss, advantages, metrics_dict) otherwise.
    """
    if is_degenerate_group(rewards):
        return None

    advantages = compute_group_advantages(rewards)
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)

    eps_low, eps_high = compute_adaptive_bounds(advantages_expanded, base_epsilon)

    unclipped = ratio * advantages_expanded
    clipped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    clipped = clipped_ratio * advantages_expanded

    per_token_loss = -torch.min(unclipped, clipped)

    total_real_tokens = attention_mask.sum().clamp(min=1)
    policy_loss = (per_token_loss * attention_mask).sum() / total_real_tokens

    # KL penalty (k3), masked and token-normalized like the policy loss
    if ref_log_probs is not None:
        kl_per_token = compute_kl_penalty_k3(new_log_probs, ref_log_probs)
        kl_penalty = (kl_per_token * attention_mask).sum() / total_real_tokens
        loss = policy_loss + kl_coef * kl_penalty
    else:
        kl_penalty = torch.tensor(0.0, device=new_log_probs.device)
        loss = policy_loss

    # ── Metrics for WandB ────────────────────────────────────────────────
    with torch.no_grad():
        n_pos = (advantages_expanded > 0).float().sum().item()
        n_neg = (advantages_expanded < 0).float().sum().item()
        total = n_pos + n_neg + 1e-8
        rho = n_pos / total  # positive contribution ratio

        clip_fraction = (
            (ratio < 1 - eps_low) | (ratio > 1 + eps_high)
        ).float().mean().item()

        metrics = {
            "train/clip_fraction": clip_fraction,
            "train/ratio_mean": ratio.mean().item(),
            "train/ratio_std": ratio.std().item(),
            "train/ratio_max": ratio.max().item(),
            "train/ratio_min": ratio.min().item(),
            "train/approx_kl": (0.5 * (new_log_probs - old_log_probs).pow(2)).mean().item(),
            "bapo/c_low": eps_low,
            "bapo/c_high": eps_high,
            "bapo/positive_contribution_ratio": rho,
            "bapo/positive_contribution_target": 0.5,
            "bapo/contribution_gap": abs(rho - 0.5),
            "bapo/policy_loss": policy_loss.item(),
            "bapo/kl_penalty": kl_penalty.item(),
            "bapo/kl_coef": kl_coef,
            "bapo/has_ref_model": ref_log_probs is not None,
        }

    return loss, advantages, metrics
