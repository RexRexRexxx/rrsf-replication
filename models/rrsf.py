"""
rrsf.py — Resource-Rational Successor Features (RRSF). ★ Best model ★

Combines SF (Ψ learning) with the resource-rational policy framework.
Since φ(s) = d(s) is goal-independent, Ψ can be learned without knowing
the current goal — the only information cost comes from the policy step.

  Q_g(s,a) = Ψ(s,a) · w_g
  π*(a|s,g) ∝ π_0(a|s) · exp(Q_g(s,a) / β)

Default policy update (same as RRMF):
  π_0^active(a|s) ← π_0^active(a|s) + α_p·(𝟙[a_t=a] − π_0^active(a|s))

Key finding (SUD vs HC):
  β↑ (more effort-averse) + α_SF↓ (weaker SF signal) + α_p↑ (habit lock-in)
  → SUD agents over-rely on default policy even though β is large,
    because flat Ψ means Q_g ≈ 0 and the policy defaults back to π_0.

Parameters (5): beta, gamma, alpha_sf, alpha_p, lam
"""

import numpy as np
from .base import rr_policy
from env import Env


class RRSF:

    PARAM_SPEC = [
        ('beta',     'log'),
        ('gamma',    'logit'),
        ('alpha_sf', 'logit'),
        ('alpha_p',  'logit'),
        ('lam',      'logit'),
    ]
    N_PARAMS = len(PARAM_SPEC)
    PHI_DIM  = 3

    @staticmethod
    def _make_pi0(pi0_active, lam, s):
        n = Env.N_ACTIONS[s]
        return (1 - lam) * pi0_active[s] + lam / n

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        beta     = params['beta']
        gamma    = params['gamma']
        alpha_sf = params['alpha_sf']
        alpha_p  = params['alpha_p']
        lam      = params['lam']

        Psi        = {s: np.zeros((Env.N_ACTIONS[s], self.PHI_DIM)) for s in Env.NON_TERMINAL}
        pi0_active = {s: pi0_init[s].copy() for s in Env.NON_TERMINAL}

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            q0  = Psi[0] @ w_g
            p0  = self._make_pi0(pi0_active, lam, 0)
            pi0 = rr_policy(q0, p0, beta)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            q1  = Psi[s1] @ w_g
            p1  = self._make_pi0(pi0_active, lam, s1)
            pi1 = rr_policy(q1, p1, beta)
            if actions_in is None:
                a1 = int(rng.choice(Env.N_ACTIONS[s1], p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)

            # ── SF updates (SARSA-style, online forward) ──────────────────────
            Psi[0][a0]  += alpha_sf * (gamma * Psi[s1][a1] - Psi[0][a0])
            Psi[s1][a1] += alpha_sf * (Env.phi(s2) - Psi[s1][a1])

            # ── π_0^active updates ────────────────────────────────────────────
            one_hot_a0 = np.zeros(3);                   one_hot_a0[a0] = 1.0
            one_hot_a1 = np.zeros(Env.N_ACTIONS[s1]);   one_hot_a1[a1] = 1.0
            pi0_active[0]  += alpha_p * (one_hot_a0 - pi0_active[0])
            pi0_active[s1] += alpha_p * (one_hot_a1 - pi0_active[s1])

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        actions, _ = self._run(trial_sequence, params, pi0_init, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, pi0_init, actions_per_trial, None)
        return ll
