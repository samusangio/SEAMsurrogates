# Repository Guidelines

This repository contains materials for an introductory course on "Surrogate Modeling and Design Optimization," originally developed for the Shared Education in Artificial intelligence and Machine learning (SEAM) professional development program at Lawrence Livermore National Laboratory (LLNL). It also includes the current user final project in `finalproject` directory.

## Project Structure & Module Organization
-  `surmod` Python package, which implements core techniques including Feedforward Neural Networks (FFNNs) and Gaussian Processes (GPs) as surrogates, basic Bayesian Optimization (BO) for design optimization, and Sensitivity Analysis (SA).
- Weekly workflow scripts in the `scripts` directory demonstrate these methods in practice.
- Example datasets included in the `data` directory are used to support hands-on learning and experimentation. The data for the final project is in `data/HST-drag`
- Final project development folder: `finalproject`

IMPORTANT: Keep code changes only within the `finalproject` folder unless explicitly prompted otherwise

## Build, Test, and Development Commands
- Install dependencies (and the `ctdp` package): `poetry install`
- Always run Python inside Poetry, even for quick one-liners: use `poetry run python ...` (not `python`, not `python3`). The system interpreter may be missing or point to a different environment without repo dependencies.
  - Examples:
    - Import/smoke check: `poetry run python -c "import ctdp; print(ctdp.__version__)"`
    - Compile check: `poetry run python -m py_compile tools/upload_folder.py`
    - Tests (if present): `poetry run pytest`
  - If `poetry run ...` fails because deps aren’t installed yet, run `poetry install` first, then retry.

## Coding Style & Naming Conventions
- Python (3.10+), PEP 8, 4‑space indent, type hints where practical; concise docstrings.

## Commit Guidelines
- Adopt Conventional Commits (e.g., `feat:`, `fix:`) with concise messages.
- Note: If automated commits fail with "Unable to create `.git/index.lock`", re-run the commit with escalated permissions.
- Never change persistent git config; use existing git identity.
- Stage specific files only with `git add <file>` (never `git add -A` or `git add .` or `git add -u`)
- Leave unrelated changes unstaged unless explicitly requested
- Do not commit editor or OS artifacts (`.idea/`, `.DS_Store`). Respect `.gitignore`.
