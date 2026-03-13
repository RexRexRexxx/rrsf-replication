"""
sf.py — Successor Features (SF) reinforcement learning.

Learns Ψ(s,a) ∈ ℝ³ (discounted expected future features) via SARSA-style TD.
Q-values are reconstructed on-the-fly for each goal: Q_g(s,a) = Ψ(s,a) · w_g.
Enables zero-shot transfer to new goals by swapping w_g only.

Parameters (3): gamma, alpha_sf, tau
"""

import numpy as np
from .base import softmax
from env import Env


class SF:

    PARAM_SPEC = [
        ('gamma',    'logit'),
        ('alpha_sf', 'logit'),
        ('tau',      'log'),
    ]
    N_PARAMS  = len(PARAM_SPEC)
    PHI_DIM   = 3

    def _init_Psi(self):
        """Ψ[s] = array of shape (N_ACTIONS[s], PHI_DIM), all zeros."""
        return {s: np.zeros((Env.N_ACTIONS[s], self.PHI_DIM))
                for s in Env.NON_TERMINAL}

    def _run(self, trial_sequence, params, actions_in, rng):
        gamma    = params['gamma']
        alpha_sf = params['alpha_sf']
        tau      = params['tau']

        Psi = self._init_Psi()
        ll  = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]

            # ── Stage 1: root s=0 ─────────────────────────────────────────────
            q0  = Psi[0] @ w_g
            pi0 = softmax(q0, tau)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2: intermediate state s1 ───────────────────────────────
            q1  = Psi[s1] @ w_g
            pi1 = softmax(q1, tau)
            if actions_in is None:
                a1 = int(rng.choice(Env.N_ACTIONS[s1], p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)

            # ── SF updates (SARSA-style, online forward) ──────────────────────
            # s0→s1: φ(s1)=0 for non-terminal s1
            Psi[0][a0]  += alpha_sf * (gamma * Psi[s1][a1] - Psi[0][a0])
            # s1→s2: s2 is terminal, γ·Ψ(s2,·)=0
            Psi[s1][a1] += alpha_sf * (Env.phi(s2) - Psi[s1][a1])

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        """Simulate SF agent. pi0_init unused."""
        actions, _ = self._run(trial_sequence, params, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, actions_per_trial, None)
        return ll
