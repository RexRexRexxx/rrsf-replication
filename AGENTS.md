# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project for replicating reinforcement-learning models from Fang, Gao, Xia, Cheng et al. (2026). Core code lives at the top level in [env.py](/Users/rex/Documents/SR project/replication/env.py) and [main.py](/Users/rex/Documents/SR project/replication/main.py). Model implementations live in [models/](/Users/rex/Documents/SR project/replication/models), with one file per model (`mf.py`, `sf.py`, `rrsf.py`) plus shared utilities in `base.py`. Exploratory and validation work belongs in [notebooks/](/Users/rex/Documents/SR project/replication/notebooks), using numbered names such as `01_env.ipynb`. Reference material and workflow notes live in `model.md`, `model_explained.md`, `setup.md`, and `CLAUDE.md`.

## Build, Test, and Development Commands
Use `uv` with Python 3.12.

- `uv sync`: install runtime and dev dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py`: run the current package entrypoint.
- `uv run jupyter notebook`: open the notebook workflow used for model development.
- `uv run python -m compileall env.py models main.py`: quick syntax check before committing.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module docstrings, and small, single-purpose classes. Use `snake_case` for functions and variables, `UPPER_CASE` for constants such as `TRAINING_GOALS`, and preserve paper-aligned class names like `MF`, `MB`, and `RRSF`. Keep NumPy-heavy logic explicit rather than overly compact. No formatter or linter is configured yet, so match the surrounding file style and keep imports minimal and ordered.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with focused notebook checks and lightweight script execution. For model changes, confirm both `simulate(...)` and `log_likelihood(...)` still run and that probability calculations remain numerically stable. When adding tests later, place them under `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
The repository currently has no commit history, so use clear imperative commit subjects such as `Add RRSF likelihood guard`. Keep the subject under 72 characters and explain parameter or equation changes in the body when needed. Pull requests should include: the purpose of the change, affected modules or notebooks, commands run for verification, and screenshots or exported figures when notebook visuals change.
