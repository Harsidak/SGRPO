"""
losses/sgrpo_loss.py

SGRPO: State-aware Group Relative Policy Optimization loss.

Components:
1. Group-relative advantage estimation (from GRPO)
   — computed from contamination-free rollouts (via sgrpo_rollout.py)
2. Token-level normalization (from DAPO)
3. Dynamic sampling filter (from DAPO)
4. Future-KL token weighting (from FIPO, cited)
   — operates at optimization phase, orthogonal to state isolation

The complete SGRPO objective:

    L^SGRPO = -(1 / Σ|o_i|) Σ_i Σ_t min(r_t · Â_t^ISO, clip(r_t, 1-ε, 1+ε) · Â_t^ISO)

where:
    r_t = π_θ(o_{i,t} | q, o_{i,<t}) / π_{θ_old}(o_{i,t} | q, o_{i,<t})
    Â_t^ISO = A_i^ISO · w(FutureKL_t)
    A_i^ISO = (r_i - mean(r_j)) / std(r_j)

The superscript ISO denotes that advantages were estimated under the i.i.d.
guarantee restored by state isolation in rollouts/sgrpo_rollout.py.

What is NOT included and why:
- BAPO's adaptive clipping: mathematically incompatible with Future-KL weighting
  without re-deriving optimal bounds under weighted advantages (future work)
- Frozen reference model KL penalty: not needed — Future-KL uses stored
  rollout log_probs (pi_theta_old), not a separate frozen model

Returns:
    loss:              scalar loss
    advantages:        [G] advantages for logging
    metrics:           dict with SGRPO-specific metrics for WandB
    (or None if degenerate group)
"""

import torch
from losses.dapo_loss import is_degenerate_group
from losses.grpo_loss import compute_group_advantages


def _reverse_discounted_cumsum(
    delta: torch.Tensor,    # [G, gen_len]
    gamma: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    out[t] = Σ_{k=t}^{T-1} γ^{k-t} · delta[k]  — a right-to-left discounted
    scan, computed without a per-token Python loop.

    Within a chunk of length L the scan collapses to one cumsum:
        out[t] = (Σ_{k=t}^{e-1} γ^k · delta[k]) / γ^t  +  γ^{e-t} · carry
    Only the carry between chunks is sequential, so the Python loop runs
    gen_len / chunk_size times (4 iterations for a 2048-token generation)
    instead of gen_len times.

    Chunking is also what makes the γ-power trick numerically safe: the
    powers γ^j inside one chunk span at most chunk_size exponents
    (2^(512/30) ≈ 2^17 at decay=30), far from fp32 overflow — whereas a
    single global cumsum over thousands of tokens would need γ^{-t} up to
    2^(t/30) and overflow.
    """
    G, T = delta.shape
    orig_dtype = delta.dtype
    delta = delta.float()

    out = torch.empty_like(delta)
    carry = torch.zeros(G, 1, dtype=delta.dtype, device=delta.device)

    first_start = ((T - 1) // chunk_size) * chunk_size
    for start in range(first_start, -1, -chunk_size):
        end = min(start + chunk_size, T)
        L = end - start
        j = torch.arange(L, device=delta.device, dtype=delta.dtype)
        gpow = gamma ** j                                       # [L]
        weighted = delta[:, start:end] * gpow                   # [G, L]
        # c[j] = Σ_{m=j}^{L-1} γ^m · delta[start+m]
        c = torch.flip(torch.cumsum(torch.flip(weighted, dims=[1]), dim=1),
                       dims=[1])
        out[:, start:end] = c / gpow + carry * gamma ** (L - j)
        carry = out[:, start:start + 1]

    return out.to(orig_dtype)


def compute_future_kl(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    old_log_probs: torch.Tensor,    # [G, gen_len]
    attention_mask: torch.Tensor,   # [G, gen_len]
    decay_rate: float = 30.0,
) -> torch.Tensor:
    """
    Compute Future-KL influence weights per token (vectorized).

    From FIPO (Ma et al. 2026):
    FutureKL_t = Σ_{k=t}^{T} γ^{k-t} · M_k · δlog_p_k

    where:
    - δlog_p_k = log π_θ(y_k) - log π_{θ_old}(y_k)
    - γ = 2^{-1/decay_rate}
    - M_k = attention_mask (1 for real tokens, 0 for padding)

    Padding is zeroed BEFORE the scan (the "tensor trap"): a padded token
    must contribute nothing to any earlier token's sum, and masking after
    the scan would not undo its contribution to the running discount.

    Implementation: chunked reverse discounted cumsum — see
    _reverse_discounted_cumsum. O(gen_len / chunk_size) sequential steps,
    everything else parallel on GPU.

    Args:
        new_log_probs:  [G, gen_len] — current policy
        old_log_probs:  [G, gen_len] — stored rollout policy
        attention_mask: [G, gen_len]
        decay_rate:     controls how far future tokens influence current token
                        larger = more future context, smaller = more local

    Returns:
        future_kl: [G, gen_len] — influence weight per token
    """
    gamma = 2.0 ** (-1.0 / decay_rate)

    # Clamp log-prob differences to prevent exp() overflow
    # |δ| > 20 means ratio > 5e8 — numerically meaningless
    delta_log_p = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
    delta_log_p = delta_log_p * attention_mask

    return _reverse_discounted_cumsum(delta_log_p, gamma)


def compute_influence_weights(
    future_kl: torch.Tensor,       # [G, gen_len]
    clip_low: float = 1.0,
    clip_high: float = 1.2,
) -> torch.Tensor:
    """
    Convert Future-KL signal to influence weights via clipping.

    From FIPO: weights are clipped to [clip_low, clip_high] to prevent
    extreme weighting of any single token from destabilizing training.

    w_t = clip(exp(future_kl_t), clip_low, clip_high)

    Numerical stability: clamp future_kl before exp() to prevent overflow.
    exp(20) ≈ 5e8 which is already well beyond clip_high.

    Args:
        future_kl:  [G, gen_len]
        clip_low:   minimum weight (default 1.0 from FIPO paper)
        clip_high:  maximum weight (default 1.2 from FIPO paper)

    Returns:
        weights: [G, gen_len], each in [clip_low, clip_high]
    """
    # Clamp input to prevent exp() overflow — any value > log(clip_high)
    # would be clipped anyway, so no information loss
    safe_kl = future_kl.clamp(-10.0, 10.0)
    weights = torch.exp(safe_kl)
    weights = torch.clamp(weights, clip_low, clip_high)
    return weights


def compute(
    new_log_probs: torch.Tensor,    # [G, gen_len]
    old_log_probs: torch.Tensor,    # [G, gen_len]
    rewards: torch.Tensor,          # [G]
    attention_mask: torch.Tensor,   # [G, gen_len]
    clip_epsilon: float = 0.2,
    future_kl_decay: float = 30.0,
    future_kl_clip_low: float = 1.0,
    future_kl_clip_high: float = 1.2,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """
    SGRPO loss.

    The complete objective:
    L = -(1 / Σ|o_i|) · Σ_{i,t} [
        min(r_t · A_i^ISO · w_t, clip(r_t, 1-ε, 1+ε) · A_i^ISO · w_t)
    ]

    where:
    - A_i^ISO = group-relative advantage from contamination-free rollouts
    - w_t = Future-KL influence weight at token t
    - r_t = importance sampling ratio at token t

    Returns None if group is degenerate.
    Returns (loss, advantages, metrics_dict) otherwise.
    """
    if is_degenerate_group(rewards):
        return None

    # Group-relative advantages from isolated rollouts
    # (the "ISO" superscript — isolation happened in the rollout generator)
    advantages = compute_group_advantages(rewards)

    # Importance sampling ratio with numerical stability.
    # This DOES require grad: d r_t / d theta is the intended policy gradient.
    log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)

    # Future-KL influence weights.
    # w_t scales the advantage, so it is an advantage-like coefficient and
    # MUST be treated as a constant w.r.t. theta — exactly as `advantages`
    # itself is. Computing it under no_grad both (a) prevents an unintended
    # second gradient term  r_t · A_i · ∇_θ w_t  from leaking into the policy
    # gradient wherever the clip is not saturating, and (b) avoids building
    # the autograd graph for the reverse-cumsum scan at all (nothing to free
    # later). `ratio` above is built from `new_log_probs` OUTSIDE this block,
    # so the intended gradient  ∇_θ r_t · A_i · w_t  is fully preserved.
    with torch.no_grad():
        future_kl = compute_future_kl(
            new_log_probs, old_log_probs, attention_mask, future_kl_decay
        )
        influence_weights = compute_influence_weights(
            future_kl, future_kl_clip_low, future_kl_clip_high
        )

    # Weighted advantages: A_i^ISO * w_t
    advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)
    weighted_advantages = advantages_expanded * influence_weights

    # Clipped surrogate with weighted advantages
    unclipped = ratio * weighted_advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * weighted_advantages

    per_token_loss = -torch.min(unclipped, clipped)

    # Token-level normalization (from DAPO)
    total_real_tokens = attention_mask.sum().clamp(min=1)
    loss = (per_token_loss * attention_mask).sum() / total_real_tokens

    # ── Metrics for WandB ────────────────────────────────────────────────
    with torch.no_grad():
        clip_fraction = (
            (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
        ).float().mean().item()

        pos_adv = (advantages > 0).float().sum().item()
        total_adv = advantages.numel()

        metrics = {
            # Universal
            "train/clip_fraction": clip_fraction,
            "train/ratio_mean": ratio.mean().item(),
            "train/ratio_std": ratio.std().item(),
            "train/ratio_max": ratio.max().item(),
            "train/ratio_min": ratio.min().item(),
            "train/approx_kl": (0.5 * log_ratio.pow(2)).mean().item(),
            "train/positive_advantage_ratio": pos_adv / max(total_adv, 1),
            # SGRPO-specific
            "sgrpo/future_kl_mean": future_kl.mean().item(),
            "sgrpo/future_kl_std": future_kl.std().item(),
            "sgrpo/future_kl_max": future_kl.max().item(),
            "sgrpo/future_kl_min": future_kl.min().item(),
            "sgrpo/influence_weight_mean": influence_weights.mean().item(),
            "sgrpo/influence_weight_std": influence_weights.std().item(),
            "sgrpo/weighted_advantage_mean": weighted_advantages.mean().item(),
            "sgrpo/weighted_advantage_std": weighted_advantages.std().item(),
            "sgrpo/decay_rate": future_kl_decay,
            "sgrpo/clip_low": future_kl_clip_low,
            "sgrpo/clip_high": future_kl_clip_high,
        }

    return loss, advantages, metrics
