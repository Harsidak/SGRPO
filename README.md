# SGRPO: State-Isolated Group Relative Policy Optimization

This repository contains the research implementation for **SGRPO (State-aware Group Relative Policy Optimization)**, along with a baseline comparison against four other RL-for-LLM algorithms (PPO, GRPO, DAPO, and BAPO).

The primary focus is on applying these algorithms to `state-spaces/mamba-130m-hf` on the GSM8K dataset. The novel contribution of this work is isolating SSM states during rollouts to fix the i.i.d. assumption violation in GRPO-style training for Mamba/SSM models, combined with a vectorized Future-KL token weighting mechanism.

## Environment Setup

This project uses `uv` for dependency management with Python 3.12+.

```bash
uv sync
```

Alternatively, using `pip` (based on `pyproject.toml`):
```bash
pip install -e .
```

**Note:** Core requirements `mamba-ssm` and `causal-conv1d` require CUDA and must be installed separately:
```bash
pip install mamba-ssm causal-conv1d
```

For other requirements please check requirements.txt

## Running Training

The unified entry point for all algorithms is `main.py`. The trainer, dataset, model, and logging infrastructure are identical across all runs to ensure scientifically valid comparisons. 

```bash
# Run the novel SGRPO algorithm
python main.py --algorithm sgrpo --steps 500 --group_size 4

# Run baselines.0
python main.py --algorithm grpo  --steps 500 --group_size 4
python main.py --algorithm ppo   --steps 500 --group_size 1

# Optimize for 6GB VRAM (e.g. RTX 4060)
python main.py --algorithm sgrpo --batch_size 1 --grad_accum 8

# Disable WandB for quick local tests
python main.py --algorithm sgrpo --steps 50 --no_wandb --device cpu
```

## Experiment Suite

The `Experiments/` directory contains executable scripts to reproduce the core claims of the research:

```bash
# 1. Full Benchmark (Compare all 5 algorithms across 3 seeds)
python -m Experiments.run_benchmark --steps 500 --seeds 42 123 456

# 2. Contamination Probe (Statistically prove state isolation fixes SSM drift)
python -m Experiments.contamination_probe --num_prompts 10 --group_size 6

# 3. Ablation Study (Isolate State Isolation vs. Future-KL)
python -m Experiments.ablation_study --steps 300 --seeds 42 123

# 4. Hyperparameter Sensitivity
python -m Experiments.future_kl_sensitivity --sweep_type decay --steps 200
```

## Tracking

All logging goes through `tracking/wandb_logger.py`, adhering strictly to `Docs/wandb_tracking_spec.md` with over 80 tracked metrics natively. Ensure you set your W&B API key in a local `.env` file at the repository root.

```env
WANDB_API_KEY=your_key_here
```