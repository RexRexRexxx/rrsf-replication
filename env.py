"""
env.py — Task environment as a class.

Tree structure (deterministic, two-stage):

                    s=0  (root)
                   /     |     \
                s=1      s=2     s=3        ← intermediate rooms
               /   \    /   \    /   \
             s=4   s=5 s=6  s=7 s=8  s=9   ← terminal rooms

Actions:
  At s=0:      0=left→s=1,  1=mid→s=2,   2=right→s=3
  At s=1,2,3:  0=left,      1=right

Source: Fang, Gao, Xia, Cheng et al. (2026)
"""

import numpy as np


class Env:

    # ── State space ───────────────────────────────────────────────────────────

    STATES       = list(range(10))
    NON_TERMINAL = [0, 1, 2, 3]
    TERMINAL     = [4, 5, 6, 7, 8, 9]

    # ── Transitions: TRANSITIONS[s][a] = s' ──────────────────────────────────

    TRANSITIONS = {
        0: {0: 1, 1: 2, 2: 3},
        1: {0: 4, 1: 5},
        2: {0: 6, 1: 7},
        3: {0: 8, 1: 9},
    }

    N_ACTIONS = {0: 3, 1: 2, 2: 2, 3: 2}

    # ── Feature / resource vectors φ(s) = d(s) ───────────────────────────────

    PHI = {
        0: np.zeros(3),
        1: np.zeros(3),
        2: np.zeros(3),
        3: np.zeros(3),
        4: np.array([10.,  0.,  0.]),
        5: np.array([ 4.,  5., 10.]),
        6: np.array([ 9.,  9., 10.]),
        7: np.array([ 0.,  8.,  2.]),
        8: np.array([ 0., 10.,  6.]),
        9: np.array([ 8.,  0.,  0.]),
    }

    # ── Goal / price vectors w_g ──────────────────────────────────────────────

    GOALS = {
        'A_easy': np.array([-1.,  1.,  0.]),
        'B_easy': np.array([ 1., -1.,  0.]),
        'A_hard': np.array([-2.,  1.,  0.]),
        'B_hard': np.array([ 1., -2.,  0.]),
        'test':   np.array([ 1.,  1.,  1.]),
    }

    TRAINING_GOALS = ['A_easy', 'B_easy', 'A_hard', 'B_hard']

    # ── Core methods ──────────────────────────────────────────────────────────

    @staticmethod
    def step(s, a):
        """Deterministic transition. Returns next state s'."""
        return Env.TRANSITIONS[s][a]

    @staticmethod
    def phi(s):
        """Feature vector at state s. Zero vector for non-terminal states."""
        return Env.PHI[s].copy()

    @staticmethod
    def reward(s, w_g):
        """Reward R = w_g · φ(s). Zero for non-terminal states."""
        return float(np.dot(Env.PHI[s], w_g))

    @staticmethod
    def is_terminal(s):
        return s in Env.TERMINAL

    @staticmethod
    def actions(s):
        """List of valid action indices at state s. Empty for terminal states."""
        if s in Env.NON_TERMINAL:
            return list(Env.TRANSITIONS[s].keys())
        return []

    @staticmethod
    def make_trial_sequence(n_per_goal=20, seed=None):
        """
        Generate an 80-trial training sequence: 4 goals × n_per_goal each,
        randomly shuffled. Returns a list of goal name strings.
        """
        seq = Env.TRAINING_GOALS * n_per_goal
        rng = np.random.default_rng(seed)
        rng.shuffle(seq)
        return seq

    @staticmethod
    def reward_matrix():
        """Compute full reward matrix R[goal][terminal_state] = w_g · φ(s)."""
        return {
            goal_name: {s: Env.reward(s, w_g) for s in Env.TERMINAL}
            for goal_name, w_g in Env.GOALS.items()
        }

    @staticmethod
    def optimal_room(goal_name):
        """Return the terminal state with the highest reward for a given goal."""
        w_g = Env.GOALS[goal_name]
        return max(Env.TERMINAL, key=lambda s: Env.reward(s, w_g))
