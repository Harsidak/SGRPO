"""
losses/bapo_loss.py

BAPO: Balanced Policy Optimization with Adaptive Clipping.

One change from DAPO: the clipping bounds (c_low, c_high) are not fixed —
they are adjusted per batch so that positive-advantage tokens contribute
at least a target fraction rho_0 of the policy-gradient signal.

Why: fixed clipping treats positive and negative advantage updates
symmetrically. But negative-advantage tokens (suppressing bad behavior)
tend to dominate early in training, producing unstable, entropy-collapsing
gradients. BAPO rebalances this by widening c_high (letting more
positive-advantage tokens through) and, when that is exhausted, raising
c_low (clipping away more negative-advantage tokens).

Algorithm (verbatim from the BAPO paper):

    Input: movable range of clipping bounds [a-, b-] and [a+, b+],
           step size of upper bound d1, step size of lower bound d2,
           positive token contribution threshold rho_0

    Procedure Dynamically adjusting the clipping bounds c_high and c_low
        Initialize clipping bounds c_low = a- and c_high = a+
        while the positive token contribution rho < rho_0
              and c_low + d2 <= b- do
            if c_high + d1 <= b+ then
                c_high <- c_high + d1
            else
                c_low <- c_low + d2
        end

    Update the policy by maximizing:
        J_BAPO = E_y Sum_t min(r_t * A_t, clip(r_t, c_low, c_high) * A_t)

The positive token contribution rho is the share of the gradient-carrying
surrogate magnitude coming from positive-advantage tokens:

    rho = S+ / (S+ + S-)
    S+  = Sum over {A_t > 0, r_t < c_high} of  r_t * A_t        (unclipped)
    S-  = Sum over {A_t < 0, r_t > c_low}  of  r_t * |A_t|      (unclipped)

Tokens outside their bound are clipped: their surrogate is a constant and
carries no gradient, so they are excluded from rho. Widening c_high
un-clips more positive tokens (raises S+); raising c_low clips away more
negative tokens (lowers S-). Both moves push rho toward rho_0.

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

import torch
from losses.dapo_loss import is_degenerate_group
from losses.grpo_loss import compute_group_advantages, compute_kl_penalty_k3

# Safety cap on the bound-adjustment loop. With default ranges and step
# sizes the loop runs at most (b+ - a+)/d1 + (b- - a-)/d2 ~ 22 iterations;
# this cap only guards against misconfigured deltas (e.g. 0).
_MAX_BOUND_ITERS = 1000


def _positive_contribution(
    ratio: torch.Tensor,                 # [G, gen_len]
    advantages_expanded: torch.Tensor,   # [G, gen_len]
    attention_mask: torch.Tensor,        # [G, gen_len]
    c_low: float,
    c_high: float,
) -> float:
    """
    rho(c_low, c_high): share of gradient-carrying surrogate magnitude from
    positive-advantage tokens, given candidate clipping bounds.

    A token carries gradient only while its ratio is inside its clipping
    bound: positive-advantage tokens are clipped (gradient-dead) at
    r >= c_high, negative-advantage tokens at r <= c_low.
    """
    pos = (advantages_expanded > 0) & (ratio < c_high)
    neg = (advantages_expanded < 0) & (ratio > c_low)

    contrib = (ratio * advantages_expanded.abs()) * attention_mask
    s_pos = contrib[pos].sum().item() if pos.any() else 0.0
    s_neg = contrib[neg].sum().item() if neg.any() else 0.0

    return s_pos / (s_pos + s_neg + 1e-8)


def compute_adaptive_bounds(
    ratio: torch.Tensor,                 # [G, gen_len]
    advantages_expanded: torch.Tensor,   # [G, gen_len]
    attention_mask: torch.Tensor,        # [G, gen_len]
    c_low_min: float = 0.8,      # a-  (paper: start of lower-bound range)
    c_low_max: float = 0.9,      # b-  (paper: end of lower-bound range)
    c_high_min: float = 1.2,     # a+  (paper: start of upper-bound range)
    c_high_max: float = 1.32,    # b+  (paper: end of upper-bound range)
    delta_high: float = 0.01,    # d1  (paper: step size of upper bound)
    delta_low: float = 0.01,     # d2  (paper: step size of lower bound)
    rho_target: float = 0.5,     # rho_0 (paper: contribution threshold)
) -> tuple[float, float, float, int]:
    """
    The BAPO paper's "Dynamically adjusting the clipping bounds" procedure.

    Starts from the tightest bounds (a-, a+) and, while the positive token
    contribution rho is below rho_0, first widens c_high in steps of d1 up
    to b+, then raises c_low in steps of d2 up to b-. rho is recomputed
    after every adjustment because moving a bound changes which tokens are
    clipped.

    Returns:
        (c_low, c_high, rho, n_iterations)
    """
    c_low, c_high = c_low_min, c_high_min
    rho = _positive_contribution(
        ratio, advantages_expanded, attention_mask, c_low, c_high
    )

    iters = 0
    while (rho < rho_target
           and c_low + delta_low <= c_low_max
           and iters < _MAX_BOUND_ITERS):
        if c_high + delta_high <= c_high_max:
            c_high += delta_high
        else:
            c_low += delta_low
        rho = _positive_contribution(
            ratio, advantages_expanded, attention_mask, c_low, c_high
        )
        iters += 1

    return c_low, c_high, rho, iters


def compute(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    old_log_probs: torch.Tensor,    # [G, gen_len]
    rewards: torch.Tensor,          # [G]
    attention_mask: torch.Tensor,   # [G, gen_len]
    c_low_min: float = 0.8,         # a-
    c_low_max: float = 0.9,         # b-
    c_high_min: float = 1.2,        # a+
    c_high_max: float = 1.32,       # b+
    delta_high: float = 0.01,       # d1
    delta_low: float = 0.01,        # d2
    rho_target: float = 0.5,        # rho_0
    ref_log_probs: torch.Tensor | None = None,  # [G, gen_len] — frozen reference
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """
    BAPO loss: DAPO's token-normalized clipped surrogate with the paper's
    dynamically adjusted clipping bounds, plus the reference-model KL
    penalty (inherited from GRPO's foundation — the original BAPO paper
    keeps it).

    Note the bounds are ABSOLUTE ratio bounds clip(r, c_low, c_high),
    not the symmetric 1 +/- eps form — this is how the paper states the
    objective. Defaults (c_low in [0.8, 0.9], c_high in [1.2, 1.32])
    reduce to standard eps=0.2 clipping when no adjustment fires, with
    the upper range extending past DAPO's clip-higher value of 1.28.

    Returns None if group is degenerate.
    Returns (loss, advantages, metrics_dict) otherwise.
    """
    if is_degenerate_group(rewards):
        return None

    advantages = compute_group_advantages(rewards)
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)

    # Bound search is a batch statistic, not part of the computation graph
    with torch.no_grad():
        c_low, c_high, rho, bound_iters = compute_adaptive_bounds(
            ratio, advantages_expanded, attention_mask,
            c_low_min=c_low_min, c_low_max=c_low_max,
            c_high_min=c_high_min, c_high_max=c_high_max,
            delta_high=delta_high, delta_low=delta_low,
            rho_target=rho_target,
        )

    unclipped = ratio * advantages_expanded
    clipped = torch.clamp(ratio, c_low, c_high) * advantages_expanded

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
        clip_fraction = (
            (ratio < c_low) | (ratio > c_high)
        ).float().mean().item()

        metrics = {
            "train/clip_fraction": clip_fraction,
            "train/ratio_mean": ratio.mean().item(),
            "train/ratio_std": ratio.std().item(),
            "train/ratio_max": ratio.max().item(),
            "train/ratio_min": ratio.min().item(),
            "train/approx_kl": (0.5 * (new_log_probs - old_log_probs).pow(2)).mean().item(),
            "bapo/c_low": c_low,
            "bapo/c_high": c_high,
            "bapo/bound_adjust_iterations": bound_iters,
            "bapo/positive_contribution_ratio": rho,
            "bapo/positive_contribution_target": rho_target,
            "bapo/contribution_gap": abs(rho - rho_target),
            "bapo/policy_loss": policy_loss.item(),
            "bapo/kl_penalty": kl_penalty.item(),
            "bapo/kl_coef": kl_coef,
            "bapo/has_ref_model": ref_log_probs is not None,
        }

    return loss, advantages, metrics
