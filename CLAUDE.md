# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Replication of computational models from:
> "Imbalanced learning efficiency and cognitive effort in individuals with substance use disorder" — Fang, Gao, Xia, Cheng et al. (2026)

The paper compares 8 reinforcement learning models to explain multi-task learning behavior in substance use disorder (SUD) vs. healthy controls (HC). The best-fitting model is RRSF (Resource-Rational Successor Features).

See [model.md](model.md) for full model specifications, task environment details, equations, and key findings. See [setup.md](setup.md) for the intended implementation workflow.

## Environment & Commands

This project uses `uv` for dependency management with Python 3.12.

```bash
uv sync          # install/update dependencies
uv run python main.py
uv run jupyter notebook
```

Primary development happens in Jupyter notebooks (`.ipynb`). Select kernel: `.venv/bin/python`.

## Architecture

### Task Environment ([env.py](env.py))
`Env` is a static-method class defining the shared task world:
- 10-state deterministic tree: s=0 (root) → s=1–3 (intermediate) → s=4–9 (terminal)
- `Env.PHI[s]` — 3D feature/resource vector φ(s) = d(s); zero for non-terminal states
- `Env.GOALS` — price vectors w_g for 4 training tasks + 1 test task
- `Env.TRANSITIONS[s][a]` — deterministic next state
- `Env.make_trial_sequence()` — 80-trial training sequence (4 goals × 20 each)
- Reward: `R = w_g · φ(s)`, zero for non-terminal states

### Models ([models/](models/))
All models live in `models/` and expose a uniform interface:
```python
model.simulate(trial_sequence, params, pi0_init, rng) -> list[int]
model.log_likelihood(actions, trial_sequence, params, pi0_init) -> float
```

**Framework 1 — Classical RL** (optimize expected reward):
- `mb.py` — Model-Based: learns T̂ and d̂, recomputes Q each trial via value iteration
- `mf.py` — Model-Free: goal-conditioned Q-table updated via TD
- `mfp.py` — MF + Perseveration: adds task-agnostic M(a) memory
- `sf.py` — Successor Features: learns Ψ(s,a) once; `Q_g = Ψ · w_g`
- `sfp.py` — SF + Perseveration

**Framework 2 — Resource-Rational RL** (optimize reward minus KL cost):
- `rrmf.py` — RRMF: MF Q-values + RR policy
- `rrsf.py` — **RRSF (best model)**: SF values + RR policy + learnable π_0
- `rrmb.py` — RRMB: MB planning with soft value iteration (excluded from recovery; slow)

**Shared utilities** ([models/base.py](models/base.py)):
- `softmax(q, tau)` — standard softmax policy
- `rr_policy(q, pi0, beta)` — resource-rational policy: `π* ∝ π_0 · exp(Q/β)`
- `pack(params, spec)` / `unpack(x, spec)` — logit/log transforms for unconstrained BFGS optimization
- `DEFAULT_PI0` — uniform default policy per state

### Notebooks ([notebooks/](notebooks/))
Sequential implementation pipeline:
1. `01_env.ipynb` — verify task environment and reward matrix
2. `02_models.ipynb` — implement and sanity-check each model
3. `03_fitting.ipynb` — MAP fitting on synthetic data
4. `04_recovery.ipynb` — parameter and model recovery
5. `05_analysis.ipynb` — reproduce paper results

### Model Fitting
- **MAP estimation** via `scipy.optimize.minimize(method='BFGS')` with 40 random restarts
- Parameters optimized in unconstrained space (logit for [0,1], log for [0,∞))
- Model comparison: BIC, AIC, PXP (Protected Exceedance Probability)

## Key Implementation Details

**Parameter transforms** (`[0,1]` params use logit/sigmoid; `[0,∞)` params like β and τ use log/exp):
```python
# [0,1]: logit → sigmoid
# (0,∞): log → exp
```

**π_0 initialization**: Must be computed from the **group-level histogram of participants' first actions** at each state — before fitting any individual. Never initialize to uniform for fitting (only `DEFAULT_PI0` in `base.py` is uniform, for reference).

**Ψ initialization**: All zeros. At terminal states, drop the `γΨ(s',a')` term:
```python
Ψ[s][a] += α_SF * (phi(s_next) - Ψ[s][a])
```

**β interpretation**: Larger β = more goal-directed (counterintuitive). SUD's higher β coexists with more default-policy reliance because lower `α_SF` produces flatter Q-values — not enough signal to steer away from π_0.

**Trial structure**: 2 actions per trial (t=0: 3 choices from root; t=1: 2 choices from intermediate). Reward received and Q/Ψ updated when arriving at terminal room (t=2). No action at t=2.

**RR policy unit tests**:
- Q all-zeros → π* == π_0 exactly
- π_0 uniform → π* == softmax(Q, β)
