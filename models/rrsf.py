"""
rrsf.py — Resource-Rational Successor Features (RRSF). ★ Best model ★

Combines SF (Ψ learning) with the resource-rational policy framework.
Since φ(s) = d(s) is goal-independent, Ψ can be learned without knowing
the current goal — the only information cost comes from the policy step.

  Q_g(s,a) = Ψ(s,a) · w_g
  π*(a|s,g) ∝ π_0(a|s) · exp(Q_g(s,a) / λ)

Default policy update (same as RRMF):
  π_0^active(a|s) ← π_0^active(a|s) + α_0·(𝟙[a_t=a] − π_0^active(a|s))

Key finding (SUD vs HC):
  λ↑ (more effort-averse) + α_ψ↓ (weaker SF signal) + α_0↑ (habit lock-in)
  → SUD agents over-rely on default policy even though λ is large,
    because flat Ψ means Q_g ≈ 0 and the policy defaults back to π_0.

Parameters (5): lam, gamma, alpha_psi, alpha_0, epsilon
"""

import numpy as np
from .base import rr_policy
from env import Env

# ────────────────────────────────────────────────────────────────────────────

class RRSF:
    """
    Faithful implementation of the Resource-Rational Successor Features (RRSF) 
    model exactly as written in the article equations.
    """

    # Renamed to strictly match the paper's terminology
    PARAM_SPEC = [
        ('lam',       'log'),   # \lambda: Cost sensitivity
        ('gamma',     'logit'), # \gamma: Future discount factor
        ('alpha_psi', 'logit'), # \alpha_\psi: SF learning rate
        ('alpha_0',   'logit'), # \alpha_0: Prior/default policy learning rate
        ('epsilon',   'logit'), # \epsilon: Lapse rate
    ]
    
    N_PARAMS = len(PARAM_SPEC)
    PHI_DIM  = 3

    @staticmethod
    def _make_pi0(pi0_active, epsilon, s):
        """
        Equation 25: \pi_0(a|s_t) = (1 - \epsilon)\tilde{\pi}_0(a|s_t) + \epsilon / |A|
        """
        n = Env.N_ACTIONS[s]
        return (1 - epsilon) * pi0_active[s] + epsilon / n

    def _run(self, trial_sequence, params, pi0_init, actions_in, rng):
        # Unpack parameters
        lam       = params['lam']
        gamma     = params['gamma']
        alpha_psi = params['alpha_psi']
        alpha_0   = params['alpha_0']
        epsilon   = params['epsilon']

        Psi        = {s: np.zeros((Env.N_ACTIONS[s], self.PHI_DIM)) for s in Env.NON_TERMINAL}
        pi0_active = {s: pi0_init[s].copy() for s in Env.NON_TERMINAL}

        ll = 0.0
        actions_out = []

        for t, goal_name in enumerate(trial_sequence):
            w_g = Env.GOALS[goal_name]

            # ── Stage 1 ───────────────────────────────────────────────────────
            # Eq 30: Q(g,s,a) = \psi(s,a)^T w_g
            q0  = Psi[0] @ w_g
            p0  = self._make_pi0(pi0_active, epsilon, 0)
            
            # Eq 34: \pi(a|g,s) \propto \pi_0(a|s) * exp(1/\lambda * Q)
            pi0 = rr_policy(q0, p0, lam)
            
            if actions_in is None:
                a0 = int(rng.choice(Env.N_ACTIONS[0], p=pi0))
            else:
                a0 = actions_in[t][0]
                
            ll += np.log(pi0[a0] + 1e-300)
            s1 = Env.step(0, a0)

            # ── Stage 2 ───────────────────────────────────────────────────────
            q1  = Psi[s1] @ w_g
            p1  = self._make_pi0(pi0_active, epsilon, s1)
            pi1 = rr_policy(q1, p1, lam)
            
            if actions_in is None:
                a1 = int(rng.choice(Env.N_ACTIONS[s1], p=pi1))
            else:
                a1 = actions_in[t][1]
                
            ll += np.log(pi1[a1] + 1e-300)
            s2 = Env.step(s1, a1)

            # ── Calculating Cognitive Effort Penalty ──────────────────────────
            # Penalty = \lambda * log(\pi / \pi_0)
            effort_0 = lam * np.log((pi0[a0] + 1e-300) / (p0[a0] + 1e-300))
            effort_1 = lam * np.log((pi1[a1] + 1e-300) / (p1[a1] + 1e-300))

            # ── SF updates (Equation 33 literal implementation) ───────────────
            # \psi(s_t,a_t) = \psi(s_t,a_t) + \alpha_\psi ( \phi(s_{t+1}) - \lambda*log(\pi/\pi_0) + \gamma\psi_{t+1} - \psi_t )
            
            # Stage 1 Update: Transition from room 0 to intermediate room s1. 
            # Note: No resources (\phi=0) in non-final rooms.
            td_err_0 = np.zeros(self.PHI_DIM) - effort_0 + (gamma * Psi[s1][a1]) - Psi[0][a0]
            Psi[0][a0] += alpha_psi * td_err_0

            # Stage 2 Update: Transition from intermediate room s1 to final room s2 (terminal). 
            # Note: Because s2 is terminal, there is no future \psi (gamma * \psi_{t+1} = 0).
            phi_s2 = Env.phi(s2) # This returns an array like [wood, stone, iron]
            td_err_1 = phi_s2 - effort_1 + 0.0 - Psi[s1][a1]
            Psi[s1][a1] += alpha_psi * td_err_1

            # ── π_0^active updates (Equation 26 literal implementation) ───────
            # \tilde{\pi}_0(a|s) += (1-\epsilon)\alpha_0 (I(a=a_t) - \tilde{\pi}_0(a|s))
            one_hot_a0 = np.zeros(Env.N_ACTIONS[0]);   one_hot_a0[a0] = 1.0
            one_hot_a1 = np.zeros(Env.N_ACTIONS[s1]);  one_hot_a1[a1] = 1.0
            
            # Note the addition of (1 - epsilon) as prescribed by Eq 26
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
