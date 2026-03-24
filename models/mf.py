"""
mf.py — Model-Free (MF) reinforcement learning.

Directly learns action values via SARSA (on-policy TD).
No world model; no transfer across goals.

Parameters (3): gamma, alpha_Q, lam
"""

import numpy as np
from .base import softmax
from env import Env


class MF:

    PARAM_SPEC = [
        ('gamma',   'logit'), # \gamma
        ('alpha_Q', 'logit'), # \alpha_Q (Q-value learning rate)
        ('lam',     'log'),   # \lambda (Softmax temperature)
    ]
    N_PARAMS = len(PARAM_SPEC)

    def _init_Q(self):
        """Goal-conditioned Q-table: one zero-initialised table per goal."""
        return {g: {s: np.zeros(Env.N_ACTIONS[s]) for s in Env.NON_TERMINAL}
                for g in Env.TRAINING_GOALS}

    def _run(self, trial_sequence, params, actions_in, rng):
        """
        Shared forward pass for simulate and log_likelihood.
        actions_in=None → sample actions (simulate mode).
        actions_in=list → use observed actions (likelihood mode).
        """
        gamma   = params['gamma']
        alpha_Q = params['alpha_Q']
        lam     = params['lam']

        Q  = self._init_Q()
        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]
            Qg  = Q[goal_name]

            # ── Stage 1: root s=0 ─────────────────────────────────────────────
            pi0 = softmax(Qg[0], lam)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2: intermediate state s1 ───────────────────────────────
            pi1 = softmax(Qg[s1], lam)
            if actions_in is None:
                a1 = int(rng.choice(Env.N_ACTIONS[s1], p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)
            R  = Env.reward(s2, w_g)

            # ── TD updates (online, forward order) ───────────────────────────
            # s0→s1: no reward at s1; SARSA bootstrap with Q[s1][a1] (action taken)
            Qg[0][a0]  += alpha_Q * (gamma * Qg[s1][a1] - Qg[0][a0])
            # s1→s2: reward R; s2 is terminal so V(s2)=0
            Qg[s1][a1] += alpha_Q * (R - Qg[s1][a1])

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        """Simulate MF agent. pi0_init unused (present for API uniformity)."""
        actions, _ = self._run(trial_sequence, params, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        """Compute sum of log P(observed actions | params) under MF."""
        _, ll = self._run(trial_sequence, params, actions_per_trial, None)
        return ll
