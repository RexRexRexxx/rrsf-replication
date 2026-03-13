"""
mfp.py — Model-Free with Perseveration (MFP).

Extends MF with an additive perseveration bias in the log-policy (paper Eq. 20):
  π(a | g, s) ∝ exp( Q(g,s,a)/τ  +  h·[(1−λ)·p0(a) + λ/|A|] )

Since λ/|A| is constant across actions it cancels in normalisation, leaving:
  π(a | g, s) ∝ exp( Q(g,s,a)/τ  +  h·(1−λ)·p0(a) )

p0(a) is task-agnostic and state-agnostic: a single running average of
recently chosen action indices, updated after every action (paper Eq. 21):
  p0(a) ← p0(a) + α_M·(𝟙[a_t=a] − p0(a))

Stored as a length-3 vector (max action count). At stage-2 states
(2 available actions) only p0[:2] is used.

Initialised from pi0_init[0] (group-level first-action histogram).

Parameters (6): gamma, alpha, tau, alpha_m, h, lam
"""

import numpy as np
from env import Env


class MFP:

    PARAM_SPEC = [
        ('gamma',   'logit'),
        ('alpha',   'logit'),
        ('tau',     'log'),
        ('alpha_m', 'logit'),
        ('h',       'log'),
        ('lam',     'logit'),
    ]
    N_PARAMS = len(PARAM_SPEC)

    @staticmethod
    def _persev_policy(Q_s, p0, n_actions, tau, h, lam):
        """Perseveration policy at a single state (paper Eq. 20)."""
        p0_s   = p0[:n_actions]
        logits = Q_s / tau + h * (1 - lam) * p0_s
        logits = logits - logits.max()
        unnorm = np.exp(logits)
        return unnorm / unnorm.sum()

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        gamma   = params['gamma']
        alpha   = params['alpha']
        tau     = params['tau']
        alpha_m = params['alpha_m']
        h       = params['h']
        lam     = params['lam']

        # Goal-conditioned Q-table
        Q = {g: {s: np.zeros(Env.N_ACTIONS[s]) for s in Env.NON_TERMINAL}
             for g in Env.TRAINING_GOALS}
        # Single state-agnostic perseveration memory (length 3 = max actions)
        p0 = pi0_init[0].copy()

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]
            Qg  = Q[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            pi0 = self._persev_policy(Qg[0], p0, 3, tau, h, lam)
            if actions_in is None:
                a0 = int(rng.choice(3, p=pi0))
            else:
                a0 = actions_in[t][0]
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            n1  = Env.N_ACTIONS[s1]
            pi1 = self._persev_policy(Qg[s1], p0, n1, tau, h, lam)
            if actions_in is None:
                a1 = int(rng.choice(n1, p=pi1))
            else:
                a1 = actions_in[t][1]
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)
            R  = Env.reward(s2, w_g)

            # ── Q updates (SARSA, goal-specific) ──────────────────────────────
            Qg[0][a0]  += alpha * (gamma * Qg[s1][a1] - Qg[0][a0])
            Qg[s1][a1] += alpha * (R - Qg[s1][a1])

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
