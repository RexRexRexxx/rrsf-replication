"""
rrmb.py — Resource-Rational Model-Based (RRMB).

Incorporates cognitive cost directly into MB planning via the resource-rational
Bellman equation. The optimal policy at each state has the analytical form:

  π*(a|s,g) ∝ π_0(a|s) · exp(Q(g,s,a) / β)

where Q(g,s,a) uses the standard soft value function:

  V*(s,g) = β · log Σ_a π_0(a|s) · exp(Q(g,s,a) / β)

π_0(a|s) is self-consistent: found via the alternating scheme from SI Note 6:
  Step 1: π(a|g,s) ∝ π_0(a|s) · exp(Q(g,s,a)/β)   [for all training goals g]
  Step 2: π_0(a|s) = Σ_g p(g) · π(a|g,s)           [uniform p(g)]
  Repeat until convergence (max 50 iterations, tol=1e-6).

For this 2-level tree, Q is computed analytically in 2 steps per goal per
iteration (no inner value-iteration loop). The outer alternating loop runs
up to 50 iterations until π_0 converges.

T̂ and d̂ are learned identically to the classical MB model.

Note: this model showed the worst fit and was excluded from model recovery
in the original paper due to computational cost at scale.

Parameters (4): beta, gamma, alpha_t, alpha_r
"""

import numpy as np
from .base import rr_policy
from env import Env


class RRMB:

    PARAM_SPEC = [
        ('beta',    'log'),
        ('gamma',   'logit'),
        ('alpha_t', 'logit'),
        ('alpha_r', 'logit'),
    ]
    N_PARAMS = len(PARAM_SPEC)

    SUCCESSORS = {
        0: [1, 2, 3],
        1: [4, 5],
        2: [6, 7],
        3: [8, 9],
    }

    def _init_world_model(self):
        T_hat = {
            s: {a: np.zeros(len(self.SUCCESSORS[s]))
                for a in range(Env.N_ACTIONS[s])}
            for s in Env.NON_TERMINAL
        }
        d_hat = {s: np.zeros(3) for s in Env.TERMINAL}
        return T_hat, d_hat

    def _compute_Q(self, T_hat, d_hat, w_g, gamma, beta, pi0):
        """
        Exact 2-step soft value iteration for one goal given a fixed π_0.
        Returns Q[s] arrays for all non-terminal states.
        """
        Q = {}
        # Stage-2: Q(g, s1, a1) = Σ_s2 T̂(s2|s1,a1) · d̂(s2)·w_g  [terminal V*=0]
        for s1 in [1, 2, 3]:
            succs = self.SUCCESSORS[s1]
            Q[s1] = np.array([
                sum(T_hat[s1][a1][i] * np.dot(d_hat[s2], w_g)
                    for i, s2 in enumerate(succs))
                for a1 in range(Env.N_ACTIONS[s1])
            ])

        # Soft V*(s1,g) = β·log Σ_{a1} π_0(a1|s1)·exp(Q(g,s1,a1)/β)
        V_soft = {}
        for s1 in [1, 2, 3]:
            q         = Q[s1]
            q_shifted = q - q.max()
            V_soft[s1] = q.max() + beta * np.log(
                np.sum(pi0[s1] * np.exp(q_shifted / beta))
            )

        # Stage-1: Q(g, 0, a0) = Σ_s1 T̂(s1|0,a0)·γ·V*(s1,g)
        Q[0] = np.array([
            sum(T_hat[0][a0][i] * gamma * V_soft[s1]
                for i, s1 in enumerate(self.SUCCESSORS[0]))
            for a0 in range(Env.N_ACTIONS[0])
        ])
        return Q

    def _solve_pi0(self, T_hat, d_hat, gamma, beta, pi0_warm, max_iter=50, tol=1e-6):
        """
        Find self-consistent π_0 via alternating updates (SI Note 6).
        π_0(a|s) = Σ_g p(g)·π(a|g,s), with uniform p(g) over training goals.
        Uses pi0_warm as the starting point (warm-start across trials).
        """
        n_goals = len(Env.TRAINING_GOALS)
        pi0 = {s: pi0_warm[s].copy() for s in Env.NON_TERMINAL}

        for _ in range(max_iter):
            pi0_new = {s: np.zeros(Env.N_ACTIONS[s]) for s in Env.NON_TERMINAL}
            for goal_name in Env.TRAINING_GOALS:
                w_g = Env.GOALS[goal_name]
                Q   = self._compute_Q(T_hat, d_hat, w_g, gamma, beta, pi0)
                for s in Env.NON_TERMINAL:
                    pi0_new[s] += rr_policy(Q[s], pi0[s], beta) / n_goals

            if all(np.max(np.abs(pi0_new[s] - pi0[s])) < tol
                   for s in Env.NON_TERMINAL):
                break
            pi0 = pi0_new

        return pi0

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        beta    = params['beta']
        gamma   = params['gamma']
        alpha_t = params['alpha_t']
        alpha_r = params['alpha_r']

        T_hat, d_hat = self._init_world_model()
        # π_0 warm-started from group-level histogram; re-solved each trial
        pi0_active = {s: pi0_init[s].copy() for s in Env.NON_TERMINAL}

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]

            # Solve self-consistent π_0 for the current world model (SI Note 6)
            pi0_active = self._solve_pi0(T_hat, d_hat, gamma, beta, pi0_active)
            Q = self._compute_Q(T_hat, d_hat, w_g, gamma, beta, pi0_active)

            # ── Stage 1 ───────────────────────────────────────────────────────
            pi0 = rr_policy(Q[0], pi0_active[0], beta)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            pi1 = rr_policy(Q[s1], pi0_active[s1], beta)
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
            d_hat[s2] += alpha_r * (Env.phi(s2) - d_hat[s2])

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        actions, _ = self._run(trial_sequence, params, pi0_init, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, pi0_init, actions_per_trial, None)
        return ll
