"""
mb.py — Model-Based (MB) reinforcement learning.

Learns a world model (transition distribution T̂ and resource quantities d̂)
then runs exact value iteration each trial to compute Q(g,s,a).

Since the tree is only 2 levels deep, value iteration is solved analytically
in 2 steps (no iterative loop needed):
  1. Q(g, s1, a1) = Σ_s2 T̂(s2|s1,a1) · R̂_g(s2)
  2. Q(g, s0, a0) = Σ_s1 T̂(s1|s0,a0) · γ · max_{a1} Q(g, s1, a1)

T̂ and d̂ are goal-independent; Q is recomputed fresh each trial using w_g.

Parameters (4): gamma, lam, alpha_t, alpha_phi
"""

import numpy as np
from .base import softmax
from env import Env


class MB:

    PARAM_SPEC = [
        ('gamma',     'logit'), # \gamma
        ('lam',       'log'),   # \lambda (Softmax temperature)
        ('alpha_t',   'logit'), # \alpha_T (Transition learning rate)
        ('alpha_phi', 'logit'), # \alpha_\varphi (Resource learning rate)
    ]
    N_PARAMS = len(PARAM_SPEC)

    # Successor state sets for compact T̂ representation
    SUCCESSORS = {
        0: [1, 2, 3],
        1: [4, 5],
        2: [6, 7],
        3: [8, 9],
    }

    def _init_world_model(self):
        """
        T̂[s][a] = probability array over SUCCESSORS[s], initialised to zeros.
        d̂[s]    = learned resource vector for terminal s, initialised to zeros.
        """
        T_hat = {
            s: {a: np.zeros(len(self.SUCCESSORS[s]))
                for a in range(Env.N_ACTIONS[s])}
            for s in Env.NON_TERMINAL
        }
        d_hat = {s: np.zeros(3) for s in Env.TERMINAL}
        return T_hat, d_hat

    def _value_iteration(self, T_hat, d_hat, w_g, gamma):
        """
        Exact 2-step value iteration. Returns Q[s] arrays for all non-terminal s.
        Q is inherently goal-conditioned because w_g is plugged in here.
        """
        Q = {}
        # Stage-2: Q(g, s1, a1) = Σ_s2 T̂(s2|s1,a1) · R̂_g(s2)
        for s1 in [1, 2, 3]:
            succs = self.SUCCESSORS[s1]
            Q[s1] = np.array([
                sum(T_hat[s1][a1][i] * np.dot(d_hat[s2], w_g)
                    for i, s2 in enumerate(succs))
                for a1 in range(Env.N_ACTIONS[s1])
            ])
        # Stage-1: Q(g, 0, a0) = Σ_s1 T̂(s1|0,a0) · γ · max Q(g, s1, ·)
        Q[0] = np.array([
            sum(T_hat[0][a0][i] * gamma * Q[s1].max()
                for i, s1 in enumerate(self.SUCCESSORS[0]))
            for a0 in range(Env.N_ACTIONS[0])
        ])
        return Q

    def _run(self, trial_sequence, params, actions_in, rng):
        gamma     = params['gamma']
        lam       = params['lam']
        alpha_t   = params['alpha_t']
        alpha_phi = params['alpha_phi']

        T_hat, d_hat = self._init_world_model()
        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]
            Q   = self._value_iteration(T_hat, d_hat, w_g, gamma)

            # ── Stage 1 ───────────────────────────────────────────────────────
            pi0 = softmax(Q[0], lam)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            pi1 = softmax(Q[s1], lam)
            if actions_in is None:
                a1 = int(rng.choice(Env.N_ACTIONS[s1], p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)

            # ── World-model updates ───────────────────────────────────────────
            def _update_T(s, a, s_obs):
                succs  = self.SUCCESSORS[s]
                target = np.array([1.0 if s_ == s_obs else 0.0 for s_ in succs])
                T_hat[s][a] += alpha_t * (target - T_hat[s][a])

            _update_T(0,  a0, s1)
            _update_T(s1, a1, s2)
            d_hat[s2] += alpha_phi * (Env.phi(s2) - d_hat[s2])

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        """Simulate MB agent. pi0_init unused."""
        actions, _ = self._run(trial_sequence, params, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, actions_per_trial, None)
        return ll
