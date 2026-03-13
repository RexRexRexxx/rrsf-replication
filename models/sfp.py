"""
sfp.py — Successor Features with Perseveration (SFP).

Extends SF with the same additive perseveration policy as MFP (paper Eq. 20).
Q-values are computed via Q_g(s,a) = Ψ(s,a) · w_g, then:
  π(a | g, s) ∝ exp( Q_g(s,a)/τ  +  h·(1−λ)·p0(a) )

p0(a) is state-agnostic (see mfp.py for full description).

Parameters (6): gamma, alpha_sf, tau, alpha_m, h, lam
"""

import numpy as np
from env import Env


class SFP:

    PARAM_SPEC = [
        ('gamma',    'logit'),
        ('alpha_sf', 'logit'),
        ('tau',      'log'),
        ('alpha_m',  'logit'),
        ('h',        'log'),
        ('lam',      'logit'),
    ]
    N_PARAMS = len(PARAM_SPEC)
    PHI_DIM  = 3

    @staticmethod
    def _persev_policy(q_vals, p0, n_actions, tau, h, lam):
        """Perseveration policy at a single state (paper Eq. 20)."""
        p0_s   = p0[:n_actions]
        logits = q_vals / tau + h * (1 - lam) * p0_s
        logits = logits - logits.max()
        unnorm = np.exp(logits)
        return unnorm / unnorm.sum()

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        gamma    = params['gamma']
        alpha_sf = params['alpha_sf']
        tau      = params['tau']
        alpha_m  = params['alpha_m']
        h        = params['h']
        lam      = params['lam']

        Psi = {s: np.zeros((Env.N_ACTIONS[s], self.PHI_DIM)) for s in Env.NON_TERMINAL}
        # Single state-agnostic perseveration memory (length 3 = max actions)
        p0  = pi0_init[0].copy()

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            q0  = Psi[0] @ w_g
            pi0 = self._persev_policy(q0, p0, 3, tau, h, lam)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            n1  = Env.N_ACTIONS[s1]
            q1  = Psi[s1] @ w_g
            pi1 = self._persev_policy(q1, p0, n1, tau, h, lam)
            if actions_in is None:
                a1 = int(rng.choice(n1, p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)

            # ── SF updates (SARSA-style) ──────────────────────────────────────
            Psi[0][a0]  += alpha_sf * (gamma * Psi[s1][a1] - Psi[0][a0])
            Psi[s1][a1] += alpha_sf * (Env.phi(s2) - Psi[s1][a1])

            # ── Perseveration update (state-agnostic, Eq. 21) ─────────────────
            one_hot = np.zeros(3)
            one_hot[a0] = 1.0
            p0 += alpha_m * (one_hot - p0)

            one_hot = np.zeros(3)
            one_hot[a1] = 1.0
            p0 += alpha_m * (one_hot - p0)

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        actions, _ = self._run(trial_sequence, params, pi0_init, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, pi0_init, actions_per_trial, None)
        return ll
