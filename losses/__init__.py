"""
losses/__init__.py

Explicit module imports — no wildcards.
Every module exports a compute() function; wildcard imports would collide.
The trainer imports these as modules: `from losses import ppo_loss`.
"""
from losses import ppo_loss, grpo_loss, dapo_loss, bapo_loss, sgrpo_loss

__all__ = ["ppo_loss", "grpo_loss", "dapo_loss", "bapo_loss", "sgrpo_loss"]
