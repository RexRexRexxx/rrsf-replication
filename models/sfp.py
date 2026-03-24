"""
sfp.py — Successor Features with Perseveration (SFP).

Extends SF with the same additive perseveration policy as MFP (paper Eq. 20).
Q-values are computed via Q_g(s,a) = Ψ(s,a) · w_g, then:
  π(a | g, s) ∝ exp( Q_g(s,a)/τ  +  h·(1−λ)·p0(a) )

p0(a) is state-agnostic (see mfp.py for full description).

Parameters (6): gamma, alpha_psi, lam, alpha_0, h, epsilon
"""

import numpy as np
from env import Env

class SFP:

    # FIXED: Parameters exactly match Lines 798-801 in the paper
    PARAM_SPEC = [
        ('gamma',     'logit'), # \gamma
        ('alpha_psi', 'logit'), # \alpha_\psi (SF learning rate)
        ('lam',       'log'),   # \lambda (Softmax temperature)
        ('alpha_0',   'logit'), # \alpha_0 (Perseveration learning rate)
        ('h',         'log'),   # h (Tendency to perseverate)
        ('epsilon',   'logit'), # \epsilon (Lapse rate)
    ]
    N_PARAMS = len(PARAM_SPEC)
    PHI_DIM  = 3

    @staticmethod
    def _persev_policy(q_vals, p0, n_actions, lam, h, epsilon):
        """Perseveration policy at a single state using the exact MFP formulation."""
        p0_s   = p0[:n_actions]
        
        # FIXED: Variable names updated. 
        # (Optional: You could leave out the `+ (epsilon / n_actions)` for computational 
        # efficiency, but keeping it makes it literally faithful to Eq 20).
        persev_term = h * ((1 - epsilon) * p0_s + (epsilon / n_actions))
        
        logits = (q_vals / lam) + persev_term
        logits = logits - logits.max()
        unnorm = np.exp(logits)
        return unnorm / unnorm.sum()

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        # FIXED: Unpacking matching parameters
        gamma     = params['gamma']
        alpha_psi = params['alpha_psi']
        lam       = params['lam']
        alpha_0   = params['alpha_0']
        h         = params['h']
        epsilon   = params['epsilon']

        Psi = {s: np.zeros((Env.N_ACTIONS[s], self.PHI_DIM)) for s in Env.NON_TERMINAL}
        p0  = pi0_init[0].copy()

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            q0  = Psi[0] @ w_g
            pi0 = self._persev_policy(q0, p0, 3, lam, h, epsilon)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            n1  = Env.N_ACTIONS[s1]
            q1  = Psi[s1] @ w_g
            pi1 = self._persev_policy(q1, p0, n1, lam, h, epsilon)
            if actions_in is None:
                a1 = int(rng.choice(n1, p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)

            # ── SF updates (SARSA-style) ──────────────────────────────────────
            # FIXED: Using alpha_psi
            Psi[0][a0]  += alpha_psi * (gamma * Psi[s1][a1] - Psi[0][a0])
            Psi[s1][a1] += alpha_psi * (Env.phi(s2) - Psi[s1][a1])

            # ── Perseveration update (state-agnostic, Eq. 21 fixed) ───────────
            # FIXED: Using alpha_0
            one_hot = np.zeros(3)
            one_hot[a0] = 1.0
            p0 += alpha_0 * (one_hot - p0)

            one_hot = np.zeros(3)
            one_hot[a1] = 1.0
            p0 += alpha_0 * (one_hot - p0)

            actions_out.append([a0, a1])

        return actions_out, ll

    def simulate(self, trial_sequence, params, pi0_init, rng):
        actions, _ = self._run(trial_sequence, params, pi0_init, None, rng)
        return actions

    def log_likelihood(self, actions_per_trial, trial_sequence, params, pi0_init):
        _, ll = self._run(trial_sequence, params, pi0_init, actions_per_trial, None)
        return ll
