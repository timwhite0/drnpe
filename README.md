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

Train encoders using Hydra configs in `drnpe/conf/`:

```bash
# NPE with Gaussian variational distribution
uv run python drnpe/train.py -cn config_npe

# DRNPE with Gaussian variational distribution
uv run python drnpe/train.py -cn config_drnpe

# NPE with Neural Spline Flow
uv run python drnpe/train.py -cn config_npe_flow

# DRNPE with Neural Spline Flow
uv run python drnpe/train.py -cn config_drnpe_flow
```

Monitor training with TensorBoard:
```bash
uv run tensorboard --logdir=logs
```

## Example

See `examples/gaussian.ipynb` for a demonstration on a Gaussian inference problem, comparing coverage probabilities under model misspecification.

## Project Structure

```
drnpe/
├── drnpe/
│   ├── conf/           # Hydra configuration files
│   ├── data.py         # Data modules
│   ├── encoder.py      # NPE and DRNPE encoder classes
│   ├── networks.py     # Neural network architectures
│   └── train.py        # Training script
├── examples/
│   └── gaussian.ipynb  # Example notebook
├── trained_ckpts/      # Pre-trained model checkpoints
└── logs/               # Training logs and checkpoints
```
