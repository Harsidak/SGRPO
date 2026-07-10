"""
data/math_dataset.py

One job: load the MATH competition benchmark (Hendrycks et al. 2021) and
return formatted prompts with verifiable boxed-answer ground truths.

Why MATH as the second benchmark:
- Harder than GSM8K (competition problems, 5 difficulty levels) —
  shows the algorithm ranking is not a GSM8K artifact
- Same verifiable-reward property: ground truth is the final \\boxed{}
  expression in the reference solution, so the reward stays binary and
  deterministic (no reward-model confound)

Source: EleutherAI/hendrycks_math — the maintained public mirror,
organized as one config per subject.
"""

from datasets import load_dataset

from data.base_dataset import BaseRewardDataset
from rewards.math_reward import compute_reward, extract_boxed


SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "At the end of your solution, write your final answer inside "
    "\\boxed{}."
)

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_math(split: str = "train", subjects: list[str] | None = None):
    """
    Load the MATH benchmark.

    Args:
        split:    "train" (~7,500 problems) or "test" (~5,000 problems)
        subjects: optional subset of MATH_SUBJECTS (default: all)

    Returns:
        list of dicts with keys:
            "prompt":  formatted string ready for tokenization
            "answer":  ground-truth boxed expression (e.g. "\\frac{1}{2}")
            "level":   difficulty string (e.g. "Level 3")
            "subject": subject config name

    Problems whose reference solution has no parseable \\boxed{} answer
    are dropped — without a ground truth the binary reward is undefined.
    """
    formatted = []
    for subject in (subjects or MATH_SUBJECTS):
        raw = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        for item in raw:
            answer = extract_boxed(item["solution"])
            if answer is None:
                continue
            prompt = (f"{SYSTEM_PROMPT}\n\n"
                      f"Problem: {item['problem']}\n\nSolution:")
            formatted.append({
                "prompt": prompt,
                "answer": answer,
                "level": item.get("level", ""),
                "subject": subject,
            })

    return formatted


class MATHDataset(BaseRewardDataset):
    """
    MATH benchmark implementing the BaseRewardDataset interface.
    Reward: binary verifiable — extract last \\boxed{}, normalize, compare.
    """

    name = "math"

    def _load(self, split: str) -> list[dict]:
        return load_math(split)

    def compute_reward(self, generated_text: str, ground_truth: str) -> float:
        return compute_reward(generated_text, ground_truth)
