"""
base.py — Shared utilities for all models.
"""

import numpy as np


# ── Policy functions ──────────────────────────────────────────────────────────

def softmax(q_vals, tau):
    """Softmax policy. Returns probability array."""
    q = np.asarray(q_vals, dtype=float)
    q = q - q.max()                          # numerical stability
    e = np.exp(q / tau)
    return e / e.sum()


def rr_policy(q_vals, pi0, beta):
    """
    Resource-rational policy: π*(a) ∝ π_0(a) · exp(Q(a) / β).
    Unit tests:
      - Q all-zeros → π* == π_0 (exactly)
      - π_0 uniform  → π* == softmax(Q, β)
    """
    q  = np.asarray(q_vals, dtype=float)
    p0 = np.asarray(pi0,    dtype=float)
    unnorm = p0 * np.exp((q - q.max()) / beta)
    return unnorm / unnorm.sum()


# ── Parameter transforms for unconstrained optimization ──────────────────────

def logit(x):
    """[0, 1] → ℝ"""
    x = np.clip(x, 1e-8, 1 - 1e-8)
    return np.log(x / (1 - x))


def sigmoid(u):
    """ℝ → [0, 1]"""
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def log_t(x):
    """(0, ∞) → ℝ"""
    return np.log(np.clip(x, 1e-8, None))


def exp_t(u):
    """ℝ → (0, ∞)"""
    return np.exp(np.clip(u, -500, 500))


# ── Pack / unpack parameter vectors ──────────────────────────────────────────
# PARAM_SPEC in each model is a list of (name, transform) in order.
# transform ∈ {'logit', 'log', 'none'}

def pack(params, spec):
    """Transform bounded params → unconstrained 1D array."""
    out = []
    for name, tr in spec:
        v = params[name]
        if   tr == 'logit': out.append(logit(v))
        elif tr == 'log':   out.append(log_t(v))
        else:               out.append(float(v))
    return np.array(out)


def unpack(x, spec):
    """Transform unconstrained 1D array → bounded param dict."""
    result = {}
    for i, (name, tr) in enumerate(spec):
        if   tr == 'logit': result[name] = float(sigmoid(x[i]))
        elif tr == 'log':   result[name] = float(exp_t(x[i]))
        else:               result[name] = float(x[i])
    return result


# ── Default π_0 (uniform) ─────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env import Env

DEFAULT_PI0 = {s: np.ones(Env.N_ACTIONS[s]) / Env.N_ACTIONS[s]
               for s in Env.NON_TERMINAL}
