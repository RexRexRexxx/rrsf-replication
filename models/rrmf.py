"""
rrmf.py — Resource-Rational Model-Free (RRMF).

Combines goal-conditioned MF SARSA with a resource-rational policy:

  π*(a|s,g) ∝ π_0(a|s) · exp(Q(g,s,a) / β)

and a Bellman target that penalises deviation from the default policy:

  r - β·log(π(a|s,g) / π_0(a|s)) + γ·Q(g,s',a')

β (cost sensitivity) subsumes the role of τ — no separate temperature.

Default policy:
  π_0(a|s) = (1−λ)·π_0^active(a|s) + λ·Uniform(a)

π_0^active is updated toward the chosen action each step:
  π_0^active(a|s) ← π_0^active(a|s) + α_p·(𝟙[a_t=a] − π_0^active(a|s))

Parameters (5): beta, gamma, alpha, alpha_p, lam
"""

import numpy as np
from .base import rr_policy
from env import Env


class RRMF:

    PARAM_SPEC = [
        ('beta',    'log'),
        ('gamma',   'logit'),
        ('alpha',   'logit'),
        ('alpha_p', 'logit'),
        ('lam',     'logit'),
    ]
    N_PARAMS = len(PARAM_SPEC)

    @staticmethod
    def _make_pi0(pi0_active, lam, s):
        """Mix active policy with uniform lapse: π_0 = (1−λ)·π_0^active + λ·U."""
        n = Env.N_ACTIONS[s]
        return (1 - lam) * pi0_active[s] + lam / n

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        beta    = params['beta']
        gamma   = params['gamma']
        alpha   = params['alpha']
        alpha_p = params['alpha_p']
        lam     = params['lam']

        def _info_cost(policy, default_policy, action):
            eps = 1e-300
            return beta * np.log((policy[action] + eps) / (default_policy[action] + eps))

        Q          = {g: {s: np.zeros(Env.N_ACTIONS[s]) for s in Env.NON_TERMINAL}
                      for g in Env.TRAINING_GOALS}
        pi0_active = {s: pi0_init[s].copy() for s in Env.NON_TERMINAL}

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]
            Qg  = Q[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            p0  = self._make_pi0(pi0_active, lam, 0)
            pi0 = rr_policy(Qg[0], p0, beta)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            cost0 = _info_cost(pi0, p0, a0)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            p1  = self._make_pi0(pi0_active, lam, s1)
            pi1 = rr_policy(Qg[s1], p1, beta)
            if actions_in is None:
                a1 = int(rng.choice(Env.N_ACTIONS[s1], p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            cost1 = _info_cost(pi1, p1, a1)
            s2 = Env.step(s1, a1)
            R  = Env.reward(s2, w_g)

            # ── Q updates (resource-rational SARSA, goal-specific) ────────────
            delta0 = -cost0 + gamma * Qg[s1][a1] - Qg[0][a0]
            delta1 = R - cost1 - Qg[s1][a1]
            Qg[0][a0]  += alpha * delta0
            Qg[s1][a1] += alpha * delta1

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
