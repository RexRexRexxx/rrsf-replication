"""
fitting.py — MAP estimation and model comparison metrics.

MAP objective (per participant):
  θ̂ = argmax_θ  [ Σ_t log P(a_t | s_t, g_t, θ)  +  log P(θ) ]

Priors (from paper):
  - (0,∞) params (log transform):  HalfNormal(σ=20) — very flat, keeps estimates finite
  - [0,1] params (logit transform): Uniform          — zero contribution to gradient

Optimization:
  - BFGS via scipy.optimize.minimize, 40 random restarts (paper default)
  - Final estimate: median of all restarts that achieved the minimum MAP objective
    (within tolerance 1e-4), following the paper's tie-breaking procedure

Model comparison:
  - BIC = k·log(n) − 2·log L   where n = n_trials × 2 (one term per action)
  - AIC = 2k − 2·log L

pi0_init (group-level first-action histogram):
  - Computed once from all participants before any individual fitting
  - Separate histogram per state (s=0,1,2,3), normalized
  - Use compute_pi0_init() then pass the result to fit_map for every participant
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import halfnorm

from models.base import pack, unpack


# ── Priors ────────────────────────────────────────────────────────────────────

_HALFNORM_SCALE = 20.0


def log_prior(params, spec):
    """
    Log-prior for MAP estimation.
      - 'log'   params → HalfNormal(0, 20):  lp += halfnorm.logpdf(θ, scale=20)
      - 'logit' params → Uniform on [0,1]:   zero contribution (constant)
    """
    lp = 0.0
    for name, tr in spec:
        if tr == 'log':
            lp += halfnorm.logpdf(params[name], scale=_HALFNORM_SCALE)
    return lp


# ── Random initialization ─────────────────────────────────────────────────────

def _sample_init(spec, rng):
    """
    Sample a random starting point in unconstrained space.
    Draws from sensible ranges in natural space, then packs to unconstrained.
    """
    p = {}
    for name, tr in spec:
        if   tr == 'logit': p[name] = float(rng.uniform(0.05, 0.95))
        elif tr == 'log':   p[name] = float(rng.uniform(0.1, 10.0))
        else:               p[name] = float(rng.uniform(0.0,  1.0))
    return pack(p, spec)


# ── MAP estimation ────────────────────────────────────────────────────────────

def fit_map(model, actions, trial_sequence, pi0_init, n_restarts=40, seed=0):
    """
    MAP estimation via BFGS with n_restarts random initializations.

    Parameters
    ----------
    model         : model instance (has .log_likelihood and .PARAM_SPEC)
    actions       : list of [a0, a1] per trial (observed behaviour)
    trial_sequence: list of goal name strings (length n_trials)
    pi0_init      : {s: np.array} — initial default policy (group-level histogram)
    n_restarts    : number of random BFGS restarts (paper uses 40)
    seed          : RNG seed for reproducibility

    Returns
    -------
    best_params : dict  — MAP parameter estimates in natural/bounded space
    best_ll     : float — log-likelihood at MAP params (not the MAP objective)
    """
    rng = np.random.default_rng(seed)

    def neg_map(x):
        params = unpack(x, model.PARAM_SPEC)
        ll = model.log_likelihood(actions, trial_sequence, params, pi0_init)
        lp = log_prior(params, model.PARAM_SPEC)
        return -(ll + lp)

    results = []
    for _ in range(n_restarts):
        x0 = _sample_init(model.PARAM_SPEC, rng)
        try:
            res = minimize(neg_map, x0, method='BFGS')
            if np.isfinite(res.fun):
                results.append(res)
        except Exception:
            continue

    if not results:
        raise RuntimeError('All optimization restarts failed for this model/participant.')

    # Paper procedure: median of all restarts that achieved the minimum objective
    best_fun = min(r.fun for r in results)
    tol      = 1e-4
    best_xs  = np.array([r.x for r in results if r.fun <= best_fun + tol])
    median_x = np.median(best_xs, axis=0)

    best_params = unpack(median_x, model.PARAM_SPEC)
    best_ll     = model.log_likelihood(actions, trial_sequence, best_params, pi0_init)
    return best_params, best_ll


# ── Model comparison metrics ──────────────────────────────────────────────────

def bic(log_lik, n_params, n_obs):
    """
    BIC = k·log(n) − 2·log L

    n_obs should be the total number of observed choices:
      n_obs = n_trials × 2  (two actions per trial)
    """
    return n_params * np.log(n_obs) - 2 * log_lik


def aic(log_lik, n_params):
    """AIC = 2k − 2·log L"""
    return 2 * n_params - 2 * log_lik


# ── Group-level π_0 initialization ───────────────────────────────────────────

def compute_pi0_init(first_actions_by_state):
    """
    Compute the group-level π_0 initialization from participants' first actions.

    Called once before fitting any individual — the result is passed unchanged
    as `pi0_init` to every fit_map call.

    Parameters
    ----------
    first_actions_by_state : dict {s: list[int]}
        For each non-terminal state s, the list of first action indices observed
        across all participants (typically the first action each participant took
        at that state across all their trials, or just trial 1).

    Returns
    -------
    pi0 : dict {s: np.array}
        Normalised histogram per state.  Falls back to uniform if no data for s.

    Example
    -------
    >>> # Collect first actions from raw data before fitting:
    >>> first_actions = {s: [] for s in Env.NON_TERMINAL}
    >>> for participant_actions in all_participants:
    ...     for a0, a1 in participant_actions:          # loop over all trials
    ...         first_actions[0].append(a0)
    ...         s1 = Env.step(0, a0)                   # intermediate state actually visited
    ...         first_actions[s1].append(a1)
    >>> pi0_init = compute_pi0_init(first_actions)
    """
    from env import Env

    pi0 = {}
    for s in Env.NON_TERMINAL:
        acts = first_actions_by_state.get(s, [])
        if len(acts) == 0:
            pi0[s] = np.ones(Env.N_ACTIONS[s]) / Env.N_ACTIONS[s]
        else:
            counts = np.zeros(Env.N_ACTIONS[s])
            for a in acts:
                counts[a] += 1
            pi0[s] = counts / counts.sum()
    return pi0
