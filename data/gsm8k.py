"""
data/gsm8k.py

One job: load GSM8K and return formatted prompts ready for rollout generation.

Why GSM8K:
- Verifiable binary reward (correct numeric answer = 1.0, wrong = 0.0)
- No learned reward model needed — eliminates reward model variance
  as a confound in algorithm comparisons
- Short problems = manageable generation lengths on 6GB VRAM
- Standard benchmark — reviewers know it
"""

from datasets import load_dataset

from data.base_dataset import BaseRewardDataset
from rewards.gsm8k_reward import compute_reward


SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "At the end of your solution, write your final answer as: "
    "#### <number>"
)


def load_gsm8k(split: str = "train"):
    """
    Load GSM8K dataset.

    Args:
        split: "train" (7,473 problems) or "test" (1,319 problems)

    Returns:
        list of dicts, each with keys:
            "prompt":   formatted string ready for tokenization
            "answer":   ground truth numeric answer string (e.g. "42")
            "question": raw question text
    """
    raw = load_dataset("openai/gsm8k", "main", split=split)

    formatted = []
    for item in raw:
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {item['question']}\n\nSolution:"
        # GSM8K answers are formatted as "...\n#### 42"
        # Extract just the numeric part after ####
        answer = item["answer"].split("####")[-1].strip()
        formatted.append({
            "prompt": prompt,
            "answer": answer,
            "question": item["question"],
        })

    return formatted


class GSM8KDataset(BaseRewardDataset):
    """
    GSM8K benchmark implementing the BaseRewardDataset interface.
    Reward: binary verifiable — extract final number, compare, 1.0/0.0.
    """

    name = "gsm8k"

    def _load(self, split: str) -> list[dict]:
        return load_gsm8k(split)

    def compute_reward(self, generated_text: str, ground_truth: str) -> float:
        return compute_reward(generated_text, ground_truth)


def get_prompt(item: dict) -> str:
    return item["prompt"]


def get_answer(item: dict) -> str:
    return item["answer"]
