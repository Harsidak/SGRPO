"""
rewards/math_reward.py

One job: binary verifiable reward for the MATH (Hendrycks et al. 2021)
competition benchmark.

Ground truth on MATH is the content of the final \\boxed{...} in the
reference solution. The model is instructed to end its solution with
\\boxed{<answer>}; we extract the last boxed expression from the
generation, normalize both sides, and compare exactly.

Same design contract as gsm8k_reward: deterministic, no learned reward
model, 1.0/0.0 only — reward variance must not confound the algorithm
comparison.
"""

import re


def extract_boxed(text: str) -> str | None:
    """
    Return the content of the LAST \\boxed{...} in the text, handling
    nested braces (e.g. \\boxed{\\frac{1}{2}}), or None if absent.
    """
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1

    return "".join(out) if depth == 0 else None


def normalize_answer(answer: str) -> str:
    """
    Canonicalize a MATH answer string for exact comparison.

    Covers the common notational variance between model output and the
    reference solutions: spacing commands, \\left/\\right, \\dfrac vs
    \\frac, \\text{} wrappers, surrounding $ signs, trailing units-like
    text, and thousands separators in plain numbers. This is the standard
    lightweight normalizer used by MATH evaluation harnesses — not a CAS;
    symbolically equal but textually different answers score 0, identically
    for every algorithm, so the comparison stays fair.
    """
    s = answer.strip()

    # Strip surrounding $ ... $ and trailing period
    s = s.strip("$").strip()
    s = s.rstrip(".")

    # LaTeX cosmetic commands that don't change the value
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    # \text{...} wrappers (units, words)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)

    # \frac{a}{b} with single-char args sometimes appears as \frac ab
    s = re.sub(r"\\frac\s*([0-9a-z])\s*([0-9a-z])", r"\\frac{\1}{\2}", s)

    # Remove all whitespace
    s = re.sub(r"\s+", "", s)

    # Thousands separators in plain numbers: 1,000 -> 1000
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")

    return s


def compute_reward(generated_text: str, ground_truth_answer: str) -> float:
    """
    1.0 if the last \\boxed{} in the generation matches the ground truth
    after normalization, else 0.0. Numeric answers also match across
    representation (7 == 7.0).
    """
    extracted = extract_boxed(generated_text)
    if extracted is None:
        return 0.0

    a = normalize_answer(extracted)
    b = normalize_answer(ground_truth_answer)

    if a == b:
        return 1.0

    # Numeric fallback: 7 vs 7.0 vs 7.00
    try:
        return 1.0 if abs(float(a) - float(b)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def compute_group_rewards(
    generated_texts: list[str],
    ground_truth_answer: str,
) -> list[float]:
    """Reward each of the G rollouts for one prompt."""
    return [compute_reward(t, ground_truth_answer) for t in generated_texts]
