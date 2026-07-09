"""
data/base_dataset.py

One job: define the interface every benchmark dataset must implement so the
trainer can run on ANY benchmark without modification.

Design contract (roadmap §1.3.2):
    - Every item is a dict with at least:
          "prompt":  formatted string ready for tokenization
          "answer":  ground-truth string used by compute_reward
    - compute_reward(generated_text, ground_truth) -> float must be
      VERIFIABLE (deterministic, no learned reward model). This keeps
      reward variance out of the algorithm comparison — the whole point
      of using verifiable benchmarks.

The trainer discovers the reward function via the dataset object
(`dataset.compute_group_rewards`), so adding a new benchmark means adding
one file that subclasses BaseRewardDataset — no trainer edits.
"""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseRewardDataset(Dataset, ABC):
    """Interface for any benchmark dataset with a verifiable reward."""

    #: short identifier used in logs / run names (e.g. "gsm8k", "math")
    name: str = "base"

    def __init__(self, split: str = "train"):
        self.data = self._load(split)

    # ── Required per-benchmark implementations ──────────────────────────

    @abstractmethod
    def _load(self, split: str) -> list[dict]:
        """Return a list of item dicts with 'prompt' and 'answer' keys."""

    @abstractmethod
    def compute_reward(self, generated_text: str, ground_truth: str) -> float:
        """Binary (or bounded) verifiable reward for one generation."""

    # ── Shared behavior ──────────────────────────────────────────────────

    def get_prompt(self, item: dict) -> str:
        return item["prompt"]

    def get_ground_truth(self, item: dict) -> str:
        return item["answer"]

    def compute_group_rewards(
        self, generated_texts: list[str], ground_truth: str
    ) -> list[float]:
        """Reward each of the G rollouts for one prompt."""
        return [self.compute_reward(t, ground_truth) for t in generated_texts]

    # ── PyTorch Dataset protocol ─────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
