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
  π_0^active(a|s) ← π_0^active(a|s) + α_0·(𝟙[a_t=a] − π_0^active(a|s))

Parameters (5): lam, gamma, alpha, alpha_0, epsilon
"""

import numpy as np
from .base import rr_policy
from env import Env


class RRMF:

    PARAM_SPEC = [
        ('lam',    'log'),
        ('gamma',   'logit'),
        ('alpha',   'logit'),
        ('alpha_0', 'logit'),
        ('epsilon',     'logit'),
    ]
    N_PARAMS = len(PARAM_SPEC)

    @staticmethod
    def _make_pi0(pi0_active, epsilon, s):
        """Mix active policy with uniform lapse: π_0 = (1−λ)·π_0^active + λ·U."""
        n = Env.N_ACTIONS[s]
        return (1 - epsilon) * pi0_active[s] + epsilon / n
    
    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        lam     = params['lam']      # Cost sensitivity (\lambda)
        gamma   = params['gamma']
        alpha   = params['alpha']
        alpha_0 = params['alpha_0']
        epsilon = params['epsilon']  # Lapse rate (\epsilon)

        def _info_cost(policy, default_policy, action):
            eps = 1e-300
            # FIXED: Now correctly using lam (\lambda)
            return lam * np.log((policy[action] + eps) / (default_policy[action] + eps))

        Q          = {g: {s: np.zeros(Env.N_ACTIONS[s]) for s in Env.NON_TERMINAL}
                      for g in Env.TRAINING_GOALS}
        pi0_active = {s: pi0_init[s].copy() for s in Env.NON_TERMINAL}

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]
            Qg  = Q[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            p0  = self._make_pi0(pi0_active, epsilon, 0)
            # FIXED: Now correctly using lam (\lambda)
            pi0 = rr_policy(Qg[0], p0, lam)  
            
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            cost0 = _info_cost(pi0, p0, a0)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            p1  = self._make_pi0(pi0_active, epsilon, s1)
            # FIXED: Now correctly using lam (\lambda)
            pi1 = rr_policy(Qg[s1], p1, lam)  
            
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
            
            # This part is correct from your update! (Eq 26)
            pi0_active[0]  += (1 - epsilon) * alpha_0 * (one_hot_a0 - pi0_active[0])
            pi0_active[s1] += (1 - epsilon) * alpha_0 * (one_hot_a1 - pi0_active[s1])

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        actions, _ = self._run(trial_sequence, params, pi0_init, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, pi0_init, actions_per_trial, None)
        return ll
