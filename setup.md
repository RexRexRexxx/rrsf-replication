# Replication Setup & Workflow

Replicating the computational models from:
> "Imbalanced learning efficiency and cognitive effort in individuals with substance use disorder"
> Fang, Gao, Xia, Cheng et al. (2026)

See `model.md` for full model specifications, equations, and task environment details.

---

## Environment Setup

```bash
# Create project and initialize uv
uv init replication
cd replication

# Add dependencies
uv add numpy scipy pandas plotly kaleido

# Jupyter (primary tool)
uv add jupyter ipykernel
```

After installing, create the venv and register the kernel with VSCode:
```bash
uv sync
```

Then in VSCode:
1. Open any `.ipynb` file
2. Click **Select Kernel** → **Python Environments**
3. Choose `.venv/bin/python` (Python 3.12)

If it doesn't appear, use **Enter interpreter path** and type `.venv/bin/python`. VSCode remembers the selection per project — you only need to do this once.

---

## Project Structure

```
replication/
├── model.md              # model specifications (reference)
├── setup.md              # this file
│
├── env.py                # task environment: states, transitions, rewards, features
├── models/
│   ├── __init__.py
│   ├── base.py           # shared utilities (softmax, logit transforms, etc.)
│   ├── mf.py             # Model-Free
│   ├── mfp.py            # Model-Free + Perseveration
│   ├── sf.py             # Successor Features
│   ├── sfp.py            # Successor Features + Perseveration
│   ├── mb.py             # Model-Based
│   ├── rrmf.py           # Resource-Rational Model-Free
│   ├── rrsf.py           # Resource-Rational Successor Features (main model)
│   └── rrmb.py           # Resource-Rational Model-Based (optional)
│
├── fitting.py            # MAP estimation, BIC/AIC/PXP
│
└── notebooks/
    ├── 01_env.ipynb          # Step 1: build & verify task environment
    ├── 02_models.ipynb       # Step 2: implement & sanity-check each model
    ├── 03_fitting.ipynb      # Step 3: MAP fitting on synthetic data
    ├── 04_recovery.ipynb     # Step 4: parameter & model recovery
    └── 05_analysis.ipynb     # Step 5: reproduce paper results
```

The `.py` files under `models/` hold the reusable implementations. Notebooks import from them — this keeps notebooks clean and makes it easy to reuse code across notebooks.

---

## Implementation Order

Work from the simplest building block outward. Each notebook produces something verifiable before moving on.

### Notebook 1: Task Environment (`01_env.ipynb`)

Define everything about the world before touching any model. Implement in `env.py`, then import and verify here.

```python
from env import Env

# Class variables (access directly, no instantiation needed)
Env.STATES        # list(range(10))
Env.NON_TERMINAL  # [0, 1, 2, 3]
Env.TERMINAL      # [4, 5, 6, 7, 8, 9]
Env.TRANSITIONS   # {s: {a: s'}, ...}
Env.N_ACTIONS     # {s: n_actions, ...}
Env.PHI           # {s: np.array([...]), ...} — φ(s) = d(s), zeros for non-terminal
Env.GOALS         # {'A_easy': w_g, 'B_easy': ..., 'A_hard': ..., 'B_hard': ..., 'test': ...}
Env.TRAINING_GOALS  # ['A_easy', 'B_easy', 'A_hard', 'B_hard']

# Static methods
Env.step(s, a)              # deterministic transition → s'
Env.phi(s)                  # returns PHI[s].copy()
Env.reward(s, w_g)          # R = w_g · φ(s); 0 for non-terminal
Env.is_terminal(s)
Env.actions(s)              # list of valid action indices
Env.make_trial_sequence(n_per_goal=20, seed=None)  # 80-trial shuffled list of goal names
Env.reward_matrix()         # {goal_name: {s: reward}} for all goals × terminal states
Env.optimal_room(goal_name) # terminal state with highest reward for a given goal
```

**Sanity check in notebook:** Reproduce the full reward matrix from `model.md` and confirm optimal rooms for each task type.

---

### Notebook 2: Models (`02_models.ipynb`)

Implement each model in its own `.py` file. Import and test interactively here.

Each model is a class with class-level metadata and two public methods:

```python
class SomeModel:
    PARAM_SPEC = [('gamma', 'logit'), ('alpha', 'logit'), ('tau', 'log'), ...]
    N_PARAMS   = len(PARAM_SPEC)

    def simulate(self, trial_sequence, params, pi0_init, rng) -> list[list[int]]:
        """
        Run the model forward through trials, return list of [a0, a1] per trial.
        trial_sequence: list of goal name strings
        params: dict of model parameters (natural/bounded values)
        pi0_init: initial default policy {s: np.array} (for RR models; ignored by classical models)
        rng: np.random.Generator
        """

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init) -> float:
        """
        Compute sum of log P(a_t | s_t, g_t, params) over all trials.
        actions_per_trial: list of [a0, a1] per trial (observed)
        """
```

Shared utilities in `models/base.py`:

```python
# Policy functions
softmax(q_vals, tau)       # standard softmax → probability array
rr_policy(q_vals, pi0, beta)  # π* ∝ π_0 · exp(Q/β) → probability array

# Parameter transforms (for BFGS optimization in unconstrained space)
logit(x)    # [0,1] → ℝ
sigmoid(u)  # ℝ → [0,1]
log_t(x)    # (0,∞) → ℝ
exp_t(u)    # ℝ → (0,∞)

# Pack/unpack param dicts using a model's PARAM_SPEC
pack(params, spec)    # dict → 1D np.array (unconstrained)
unpack(x, spec)       # 1D np.array → dict (natural/bounded)

# Uniform default policy per state (reference; not for fitting)
DEFAULT_PI0   # {s: uniform array} for s in Env.NON_TERMINAL
```

**Implementation order:**
1. `mf.py` — simplest; Q-table + TD update + softmax
2. `sf.py` — swap Q-table for Ψ-table; `Q_g = Ψ · w_g`
3. `mfp.py` — add perseveration on top of MF
4. `sfp.py` — add perseveration on top of SF
5. `mb.py` — transition/reward learning + value iteration
6. `rrmf.py` — replace softmax with `rr_policy`; add π_0 update
7. `rrsf.py` — combine SF + RR policy (main model)
8. `rrmb.py` — optional; computationally expensive

For each model in the notebook: simulate 1 agent × 80 trials with known params, confirm log-likelihood is higher for the true params than for random params.

---

### Notebook 3: Fitting (`03_fitting.ipynb`)

Implement MAP fitting in `fitting.py`, test it here on synthetic data.

```python
from scipy.optimize import minimize
from models import base

def fit_map(model, actions, trial_sequence, pi0_init, n_restarts=40, seed=0):
    rng = np.random.default_rng(seed)
    best_result = None
    for _ in range(n_restarts):
        x0 = sample_random_init(model.PARAM_SPEC, rng)
        result = minimize(
            fun=lambda x: -model.log_likelihood(
                actions, trial_sequence, base.unpack(x, model.PARAM_SPEC), pi0_init
            ),
            x0=x0,
            method='BFGS',
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    return base.unpack(best_result.x, model.PARAM_SPEC), -best_result.fun

def bic(log_lik, n_params, n_trials):
    return n_params * np.log(n_trials) - 2 * log_lik

def aic(log_lik, n_params):
    return 2 * n_params - 2 * log_lik
```

**Key details:**
- Optimize in unconstrained space (logit/log transforms); unpack to natural space inside `log_likelihood`
- Use `np.clip` to guard against `log(0)`
- Initialize `π_0^active` from the **group-level histogram of first actions**, computed once before fitting any individual

---

### Notebook 4: Recovery (`04_recovery.ipynb`)

Mirrors the paper's validation procedure.

**Parameter recovery** (per model):
1. Sample "true" params from reasonable ranges
2. Simulate 1 participant × 80 trials
3. Fit the same model back
4. Scatter plot true vs. recovered — check correlation ≥ 0.8 for key params (`β`, `α_SF`)

**Model recovery** (across all models):
1. For each model: generate 10 synthetic datasets per participant using fitted params
2. Fit all 7 models to each synthetic dataset
3. Plot confusion matrix: rows = generating model, cols = best-fit model (by BIC)
4. Diagonal should be dominant — RRSF should be highly self-recovering

---

### Notebook 5: Analysis (`05_analysis.ipynb`)

Reproduce paper results in order of difficulty:

1. **Behavioral** (no model needed): group-level reward curves, Type A vs. B performance, SUD's Room 4 bias
2. **Model comparison**: BIC/AIC/PXP across all 7 models, confirm RRSF wins
3. **Group parameter differences**: t-tests on RRSF params (β, α_SF, α_p) between HC and SUD
4. **Correlations**: β vs. cognitive effort, β vs. impulsivity scores

---

## Key Implementation Notes

### Default policy initialization
The group-level first-action histogram must be computed **before** fitting any participant. Collect each participant's first action at each state (s=0,1,2,3), normalize, and use as `π_0^active` initialization for everyone.

### Ψ initialization
Initialize all Ψ(s,a) to zero. At terminal states, the TD target has no `γΨ` term:
```python
# Terminal state: no further step, so γΨ(s',a') = 0
Ψ[s][a] += α_SF * (phi(s_next) - Ψ[s][a])
```

### Within-trial step structure
Each trial has 3 steps (t=0,1,2). Actions are taken at t=0 (s=0, 3 choices) and t=1 (intermediate state, 2 choices). At t=2 the agent arrives at the terminal room — reward is received and Ψ/Q is updated. No action is taken at t=2.

### β interpretation
In `π* ∝ π_0 · exp(Q/β)`, **larger β = more goal-directed**. SUD's higher β coexists with more π_0 reliance because their lower `α_SF` produces flatter, less informative Q values — the signal is too weak to steer away from the default even at high β.

### Unit tests for the RR policy
- When `Q_g` is all-zeros: `π* = π_0` exactly
- When `π_0` is uniform: `π*` reduces to standard softmax with temperature β
