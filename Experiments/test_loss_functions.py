"""
Experiments/test_loss_functions.py

Synthetic-tensor smoke tests for all five loss functions.

Why this exists: the 10-step GPU validation runs exercise the trainer
scaffolding, but an untrained model scores 0.0 on essentially every GSM8K
group — so DAPO/BAPO/SGRPO skip every step as degenerate and their loss
math never executes. These tests feed hand-built tensors with reward
variance so every branch of every loss actually runs.

Checks per loss:
  1. Loss is finite and backward() produces finite gradients.
  2. Degenerate groups (all-identical rewards) return None where the
     algorithm specifies skipping (DAPO/BAPO/SGRPO).
  3. Padding correctness: perturbing values at masked positions must not
     change the loss (DAPO/BAPO/SGRPO — the token-normalized losses).
  4. SGRPO Future-KL weights stay inside [clip_low, clip_high].
  5. k3 KL estimator is non-negative (it is a proper divergence estimate).

Run:  python -m Experiments.test_loss_functions   (CPU, no model needed)
"""

import sys

import torch

from losses import ppo_loss, grpo_loss, dapo_loss, bapo_loss, sgrpo_loss

G, T = 4, 12
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_inputs(seed: int = 0, degenerate: bool = False):
    """Build a realistic fake rollout batch with padding in the last rows."""
    g = torch.Generator().manual_seed(seed)
    old_lp = -torch.rand(G, T, generator=g) * 3.0        # log-probs in (-3, 0)
    # new policy = old + small perturbation (requires_grad → tests backward)
    new_lp = (old_lp + 0.1 * torch.randn(G, T, generator=g)).requires_grad_(True)
    ref_lp = old_lp + 0.05 * torch.randn(G, T, generator=g)
    rewards = (torch.zeros(G) if degenerate
               else torch.tensor([1.0, 0.0, 0.0, 1.0]))
    mask = torch.ones(G, T)
    mask[1, 8:] = 0.0   # rollout 1 stopped early
    mask[3, 5:] = 0.0   # rollout 3 stopped even earlier
    return new_lp, old_lp, ref_lp, rewards, mask


def masked_invariance(loss_fn, **kwargs):
    """Return True if corrupting padded positions leaves the loss unchanged."""
    new_lp, old_lp, ref_lp, rewards, mask = make_inputs(seed=1)
    base = loss_fn(new_lp, old_lp, rewards, mask, **kwargs)
    if base is None:
        return False
    base_loss = base[0].item()

    # Corrupt old_log_probs at padded positions only
    old_corrupt = old_lp.clone()
    old_corrupt[mask == 0] = -50.0
    corrupt = loss_fn(new_lp, old_corrupt, rewards, mask, **kwargs)
    return abs(corrupt[0].item() - base_loss) < 1e-6


def main():
    torch.manual_seed(0)

    # ── PPO ──────────────────────────────────────────────────────────────
    print("\nPPO:")
    new_lp, old_lp, ref_lp, rewards, mask = make_inputs()
    advantages = rewards - rewards.mean()          # EMA-baseline stand-in
    loss, adv, metrics = ppo_loss.compute(new_lp, old_lp, advantages,
                                          ref_log_probs=ref_lp)
    loss.backward()
    check("finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("finite grads", torch.isfinite(new_lp.grad).all().item())
    check("kl penalty in metrics", "ppo/kl_penalty" in metrics)

    # ── GRPO ─────────────────────────────────────────────────────────────
    print("\nGRPO:")
    new_lp, old_lp, ref_lp, rewards, mask = make_inputs()
    loss, adv, metrics = grpo_loss.compute(new_lp, old_lp, rewards,
                                           ref_log_probs=ref_lp)
    loss.backward()
    check("finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("finite grads", torch.isfinite(new_lp.grad).all().item())
    check("advantages zero-mean", abs(adv.mean().item()) < 1e-5)
    k3 = grpo_loss.compute_kl_penalty_k3(new_lp.detach(), ref_lp)
    check("k3 estimator non-negative", (k3 >= 0).all().item(),
          f"min={k3.min().item():.2e}")

    # ── DAPO ─────────────────────────────────────────────────────────────
    print("\nDAPO:")
    new_lp, old_lp, ref_lp, rewards, mask = make_inputs()
    result = dapo_loss.compute(new_lp, old_lp, rewards, mask)
    check("non-degenerate group returns result", result is not None)
    loss, adv, metrics = result
    loss.backward()
    check("finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("finite grads", torch.isfinite(new_lp.grad).all().item())
    deg = make_inputs(degenerate=True)
    check("degenerate group returns None",
          dapo_loss.compute(deg[0], deg[1], deg[3], deg[4]) is None)
    check("padded positions do not affect loss",
          masked_invariance(dapo_loss.compute))

    # ── BAPO ─────────────────────────────────────────────────────────────
    print("\nBAPO:")
    new_lp, old_lp, ref_lp, rewards, mask = make_inputs()
    result = bapo_loss.compute(new_lp, old_lp, rewards, mask,
                               ref_log_probs=ref_lp)
    check("non-degenerate group returns result", result is not None)
    loss, adv, metrics = result
    loss.backward()
    check("finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("finite grads", torch.isfinite(new_lp.grad).all().item())
    deg = make_inputs(degenerate=True)
    check("degenerate group returns None",
          bapo_loss.compute(deg[0], deg[1], deg[3], deg[4]) is None)
    check("padded positions do not affect loss",
          masked_invariance(bapo_loss.compute, ref_log_probs=None))

    # ── SGRPO ────────────────────────────────────────────────────────────
    print("\nSGRPO:")
    new_lp, old_lp, ref_lp, rewards, mask = make_inputs()
    result = sgrpo_loss.compute(new_lp, old_lp, rewards, mask)
    check("non-degenerate group returns result", result is not None)
    loss, adv, metrics = result
    loss.backward()
    check("finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("finite grads", torch.isfinite(new_lp.grad).all().item())
    deg = make_inputs(degenerate=True)
    check("degenerate group returns None",
          sgrpo_loss.compute(deg[0], deg[1], deg[3], deg[4]) is None)
    check("padded positions do not affect loss",
          masked_invariance(sgrpo_loss.compute))
    # Future-KL influence weights must respect the clip bounds.
    fkl = sgrpo_loss.compute_future_kl(new_lp.detach(), old_lp, mask)
    w = sgrpo_loss.compute_influence_weights(fkl)
    check("influence weights within [1.0, 1.2]",
          bool((w >= 1.0 - 1e-6).all() and (w <= 1.2 + 1e-6).all()),
          f"range=[{w.min().item():.4f}, {w.max().item():.4f}]")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("All loss-function smoke tests passed.")


if __name__ == "__main__":
    main()
