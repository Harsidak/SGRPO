"""
data/__init__.py

Benchmark registry — every entry implements BaseRewardDataset, so the
trainer and every experiment script need no changes when a new benchmark
is added here. main.py and Experiments/run_benchmark.py both read this
registry.
"""

from data.base_dataset import BaseRewardDataset
from data.gsm8k import GSM8KDataset
from data.math_dataset import MATHDataset

DATASETS: dict[str, type[BaseRewardDataset]] = {
    "gsm8k": GSM8KDataset,
    "math": MATHDataset,
}

__all__ = ["BaseRewardDataset", "GSM8KDataset", "MATHDataset", "DATASETS"]
