# Distributionally Robust Neural Posterior Estimation (DRNPE)

A PyTorch implementation of Distributionally Robust Neural Posterior Estimation, which provides robust posterior inference under model misspecification.

## Overview

This repository implements:
- **NPE (Neural Posterior Estimation)**: Standard amortized posterior inference
- **DRNPE (Distributionally Robust NPE)**: Robust variant that accounts for potential model misspecification

Both methods support two variational distribution families:
- **Gaussian**: Simple location-scale family
- **Neural Spline Flow**: Flexible normalizing flow using rational quadratic splines

## Installation

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

To install pre-commit hooks:
```bash
uv run pre-commit install
```

## Training

Train encoders by pointing `--config-path` at an experiment's config directory:

```bash
# Gaussian experiment
uv run python drnpe/train.py --config-path=../experiments/gaussian/conf -cn config_npe
uv run python drnpe/train.py --config-path=../experiments/gaussian/conf -cn config_drnpe
uv run python drnpe/train.py --config-path=../experiments/gaussian/conf -cn config_npe_flow
uv run python drnpe/train.py --config-path=../experiments/gaussian/conf -cn config_drnpe_flow

# SIR experiment
uv run python drnpe/train.py --config-path=../experiments/sir/conf -cn config_sir_npe
uv run python drnpe/train.py --config-path=../experiments/sir/conf -cn config_sir_drnpe
uv run python drnpe/train.py --config-path=../experiments/sir/conf -cn config_sir_npe_flow
uv run python drnpe/train.py --config-path=../experiments/sir/conf -cn config_sir_drnpe_flow
```

Monitor training with TensorBoard:
```bash
uv run tensorboard --logdir=logs
```

## Experiments

- `experiments/gaussian/` — Gaussian inference benchmark with analytic posterior
- `experiments/sir/` — Stochastic SIR epidemic model (Ward et al., 2022)

Each experiment folder contains its own data module, Hydra configs, and evaluation notebook.

## Project Structure

```
drnpe/
├── drnpe/
│   ├── data.py         # Base data module class
│   ├── encoder.py      # NPE and DRNPE encoder classes
│   ├── networks.py     # Neural network architectures
│   └── train.py        # Training script
├── experiments/
│   ├── gaussian/
│   │   ├── conf/       # Hydra configs
│   │   ├── data_gaussian.py  # GaussianDataModule
│   │   └── gaussian.ipynb
│   └── sir/
│       ├── conf/             # Hydra configs
│       ├── data_sir.py       # SIRDataModule
│       └── sir.ipynb
├── trained_ckpts/      # Pre-trained model checkpoints
└── logs/               # Training logs and checkpoints
```
