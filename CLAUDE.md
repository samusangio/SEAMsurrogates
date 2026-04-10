# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Course materials for "Surrogate Modeling and Design Optimization" from LLNL's SEAM program. The `surmod` package implements core machine learning techniques for surrogate modeling: Feedforward Neural Networks (FFNNs), Gaussian Processes (GPs), Bayesian Optimization (BO), and Sensitivity Analysis (SA).

**CRITICAL**: Code changes should be made ONLY in the `finalproject/` directory unless explicitly requested otherwise. The rest of the repository contains course materials that should remain stable.

## Development Commands

This project uses **Poetry** for dependency management. Always run Python through Poetry, even for one-liners:

```bash
# Install dependencies (run this first)
poetry install

# Run Python scripts
poetry run python finalproject/script.py

# Quick checks
poetry run python -c "import surmod; print(surmod.__version__)"
poetry run python -m py_compile finalproject/script.py

# Tests (if present)
poetry run pytest
```

**Never use** bare `python` or `python3` commands - the system interpreter may lack dependencies or point to the wrong environment.

## Architecture

### Package Structure

- **`surmod/`** - Core package modules:
  - `neural_network.py` - PyTorch-based feedforward NNs with device detection (MPS for Mac, CPU fallback)
  - `gaussian_process_regression.py` - Scikit-learn GP surrogates
  - `bayesian_optimization.py` - BoTorch-based BO for design optimization
  - `sensitivity_analysis.py` - SA utilities using SALib
  - `data_processing.py` - Dataset loaders and preprocessing for JAG, borehole, and HST datasets
  - `test_functions.py` - Synthetic test functions for experimentation

- **`scripts/`** - Weekly course workflows organized by technique:
  - `neural_network/` - NN sandbox and data examples
  - `gaussian_process_regression/` - GP sandbox and data examples
  - `bayesian_optimization/` - BO sandbox and data examples
  - `sensitivity_analysis/` - SA sandbox and data examples

- **`finalproject/`** - Active development directory for final project work
  - Contains NN, GP, and BO implementations for HST drag data
  - Output logs and plots are generated here

- **`data/`** - Example datasets:
  - `JAG_10k.csv` - JAG ICF dataset (5 inputs, 1 output)
  - `borehole_10k.csv` - Borehole function dataset (8 inputs, 1 output)
  - `hst_*.csv` - Hubble Space Telescope datasets (8 inputs, 1 output)
  - `HST-drag/` - Final project data (train/validation/test splits)

### Key Design Patterns

- **NeuralNet class**: Customizable feedforward architecture with configurable hidden layers, optional weight initialization
- **Device management**: Automatic MPS (Metal Performance Shaders) detection for Mac, CPU fallback
- **Data loading**: Standardized interface via `DATASET_CONFIG` in `data_processing.py` - supports JAG, borehole, HST datasets
- **Train/test splitting**: Uses scikit-learn's `train_test_split` with configurable ratios
- **Plotting utilities**: Matplotlib-based functions in each module for loss curves, predictions, and optimization results

## Git Workflow

- **Commit style**: Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.)
- **Commit messages**: Keep concise - title only, 1-2 lines max. Avoid lengthy multi-paragraph descriptions.
- **Staging files**: Always stage specific files with `git add <file>` - NEVER use `git add -A`, `git add .`, or `git add -u`
- **Unrelated changes**: Leave unstaged unless explicitly requested
- **Git config**: Never modify persistent git config; use existing identity
- **Artifacts**: Don't commit `.DS_Store`, `.idea/`, or other editor/OS artifacts (see `.gitignore`)

## Code Style

- Python 3.10+, PEP 8, 4-space indentation
- Type hints where practical
- Concise docstrings for functions and classes
- Device-aware PyTorch code (check for MPS/CPU)

## Common Data Workflows

When working with datasets:
1. Load via `surmod.data_processing.load_data(dataset_name)`
2. Split using `train_test_split` or custom logic
3. Normalize inputs/outputs with `StandardScaler` from scikit-learn
4. Convert to PyTorch tensors for NN models
5. Train/evaluate with appropriate metrics (MSE, R², etc.)

## Documentation

Full package documentation available at: https://seamsurrogates.readthedocs.io/en/latest/
