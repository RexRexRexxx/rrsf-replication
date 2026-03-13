# Model-Free (MF) RL — Detailed Explanation

## Overview

The Model-Free model is the simplest and most canonical reinforcement learning agent: it learns how good each action is at each state — purely from experience, with no representation of *why* an action is good or what the world looks like internally. It has no world model, no transition table, and no feature decomposition. It just tracks numbers (`Q-values`) that get nudged after every trial.

---

## What problem does it solve?

The agent faces a two-stage decision tree. On every trial, a goal `g` is shown (a price vector `w_g`). The agent must navigate from the root `s=0` to a terminal room `s∈{4,...,9}`, collecting reward `R = w_g · φ(s)`. Since different goals make different terminal rooms rewarding, the agent needs separate value estimates per goal — hence a **goal-conditioned Q-table**.

---

## The Q-Table

```
Q[g][s][a]  →  estimated return of taking action a at state s under goal g
```

- Indexed by: **goal** `g`, **state** `s`, **action** `a`
- One table per training goal (`A_easy`, `B_easy`, `A_hard`, `B_hard`)
- All entries initialized to **0**

### In code ([models/mf.py:24-27](models/mf.py#L24-L27)):
```python
def _init_Q(self):
    return {g: {s: np.zeros(Env.N_ACTIONS[s]) for s in Env.NON_TERMINAL}
            for g in Env.TRAINING_GOALS}
```

Only non-terminal states need Q-values — there are no actions at terminal rooms. The root `s=0` has 3 actions; intermediate states `s=1,2,3` each have 2 actions.

---

## Parameters (3)

| Symbol | Range | Transform | Meaning |
|--------|-------|-----------|---------|
| `γ` (gamma) | [0, 1] | logit/sigmoid | **Discount factor** — how much future rewards are down-weighted relative to immediate ones |
| `α` (alpha) | [0, 1] | logit/sigmoid | **Learning rate** — how fast Q-values shift toward each new experience |
| `τ` (tau)   | (0, ∞) | log/exp | **Softmax temperature** — how deterministic the policy is; lower τ → greedier |

Parameters are stored in their natural (bounded) space but optimized in unconstrained space using logit/log transforms (see [models/base.py](models/base.py)).

---

## The Decision Policy

At each state, the agent picks an action probabilistically via **softmax**:

```
π(a | s, g) = exp(Q[g][s][a] / τ) / Σ_{a'} exp(Q[g][s][a'] / τ)
```

- τ → 0: nearly deterministic (pick the highest-Q action almost always)
- τ → ∞: nearly uniform (random choice)

### In code ([models/base.py:10-15](models/base.py#L10-L15)):
```python
def softmax(q_vals, tau):
    q = np.asarray(q_vals, dtype=float)
    q = q - q.max()        # subtract max for numerical stability
    e = np.exp(q / tau)
    return e / e.sum()
```

The `q - q.max()` shift prevents overflow in `exp()` without changing the output probabilities.

---

## The TD Update (SARSA — on-policy TD)

After each trial, two Q-values are updated — one for each decision step.

### Step 1: s=0 → s1 (no immediate reward)

```
Q[g][0][a0] ← Q[g][0][a0] + α · (γ · Q[g][s1][a1] − Q[g][0][a0])
```

The target bootstraps from `Q[g][s1][a1]` — the Q-value of the **action actually taken** at the next step (a1), not the maximum over all actions. This is the defining feature of SARSA (on-policy TD).

### Step 2: s1 → s2 (terminal; reward R received)

```
Q[g][s1][a1] ← Q[g][s1][a1] + α · (R − Q[g][s1][a1])
```

The target is just `R`. Because `s2` is terminal, `V(s2) = 0` — there is no future to bootstrap from.

### General form (SARSA / on-policy TD):

```
Q(s,a) ← Q(s,a) + α · [R + γ · Q(s', a') − Q(s,a)]
```

where `a'` is the action *taken* at `s'`, and the bracketed term `δ = R + γ · Q(s',a') − Q(s,a)` is the **temporal difference error** (Eq. 18 in the paper).

### In code ([models/mf.py:67-70](models/mf.py#L67-L70)):
```python
# s0→s1: no reward at s1; SARSA bootstrap with Q[s1][a1] (action taken)
Qg[0][a0]  += alpha * (gamma * Qg[s1][a1] - Qg[0][a0])
# s1→s2: reward R; s2 is terminal so V(s2)=0
Qg[s1][a1] += alpha * (R - Qg[s1][a1])
```

The paper (Eqs. 16–18) uses `Q(g, s_{t+1}, a_{t+1})` throughout — the Q-value of the next state evaluated at the action *taken* there, confirming SARSA.

---

## Full Forward Pass

The `_run` method ([models/mf.py:29-74](models/mf.py#L29-L74)) implements one complete pass over a trial sequence. It serves double duty for both simulation and likelihood evaluation, controlled by whether `actions_in` is `None`.

### Per-trial flow:

```
1. Look up current goal g → get w_g and Qg = Q[g]
2. Stage 1 (s=0):
     - Compute policy: π0 = softmax(Qg[0], τ)
     - Sample or observe action a0
     - Accumulate log P(a0) for likelihood
     - Transition: s1 = Env.step(0, a0)
3. Stage 2 (s1):
     - Compute policy: π1 = softmax(Qg[s1], τ)
     - Sample or observe action a1
     - Accumulate log P(a1) for likelihood
     - Transition: s2 = Env.step(s1, a1)
     - Compute reward: R = w_g · φ(s2)
4. TD updates:
     - Q[g][0][a0]  += α * (γ * Q[g][s1][a1] − Q[g][0][a0])
     - Q[g][s1][a1] += α * (R − Q[g][s1][a1])
5. Record actions
```

Updates happen **after** acting (online, forward order). The Q-table carries state across trials.

---

## Simulate vs. Log-Likelihood

Two public methods share the same `_run` core:

**`simulate`** ([models/mf.py:76-78](models/mf.py#L76-L78)):
- Passes `actions_in=None` → actions are **sampled** from the policy
- Returns a list of `[a0, a1]` pairs
- `pi0_init` is unused (accepted for API uniformity with resource-rational models)

**`log_likelihood`** ([models/mf.py:81-84](models/mf.py#L81-L84)):
- Passes the observed `actions_per_trial` as `actions_in` → actions are **fixed**
- Accumulates `log π(a_t | s_t, g_t, θ)` over all trials
- Returns total log-likelihood (a negative number; less negative = better fit)

The `1e-300` floor added before taking the log prevents `log(0)` when a policy assigns essentially zero probability to the observed action:
```python
ll += np.log(pi0[a0] + 1e-300)
```

---

## What MF Cannot Do

- **No transfer across goals.** Each goal's Q-table is updated independently. Learning "room 8 is good" under `A_easy` gives no boost to `A_hard`, even though the optimal path is identical. This is the key weakness exploited by the Successor Features models.
- **No world model.** The agent does not know which room it will end up in when choosing an action. It only knows that action `a0` at `s=0` led to some outcome worth `Q[g][0][a0]` on average.
- **Slow to adapt.** With many goals and sparse experience per goal, Q-values converge slowly relative to a model-based agent that can reason about transitions explicitly.

---

## Role in the Paper

MF is one of five classical RL baselines (Framework 1). It is expected to perform worse than:
- **MB**: can reason about the world explicitly, thus faster to converge
- **SF**: learns the environment structure once and transfers across goals
- **RRSF**: adds a default-policy regularization that captures the habitual bias seen in SUD

MF's three parameters (`γ`, `α`, `τ`) are the core building blocks shared or extended by every other model in the paper.

---

---

# Model-Based (MB) RL — Detailed Explanation

## Overview

The Model-Based model is the most cognitively sophisticated classical RL agent. Instead of directly memorizing action values, it builds an internal model of the world — estimating transition probabilities and resource quantities — and then *plans* through that model each trial to derive fresh Q-values. It is "model-based" because decisions are guided by a mental simulation, not by cached experience.

---

## What problem does it solve?

The same two-stage navigation task, but the agent approaches it differently. Rather than asking "how good was action `a0` last time goal `g` was shown?", the MB agent asks: "given where I think action `a0` leads, and how good those rooms seem to be, what should I do?" This planning step means the agent can immediately generalize a newly-learned room value to all goals, without needing to re-experience each goal separately.

---

## The World Model

MB maintains two learned structures, both **goal-independent**:

### T̂ — Transition model

```
T̂[s][a]  →  probability distribution over successor states of s under action a
```

- `T̂[0][a0]` is a 3-element vector over `{s=1, s=2, s=3}`
- `T̂[s1][a1]` is a 2-element vector over `{s=4,s=5}`, `{s=6,s=7}`, or `{s=8,s=9}` depending on `s1`

### d̂ — Resource quantity model (called φ̂ in the paper)

```
d̂[s]  →  estimated 3D resource vector at terminal room s
```

Only terminal states `s∈{4,...,9}` have non-zero resources. Non-terminal rooms always yield zero reward, so they are not estimated.

**Note on naming:** The paper calls this learned quantity **φ̂(s')** with learning rate **α_φ**. The codebase uses `d_hat` and `alpha_r`. They refer to the same thing.

### Initialization

The paper states: *"Both T and φ are initialized to be 0."*

```python
T_hat = {s: {a: np.zeros(len(SUCCESSORS[s])) for a in ...} for s in NON_TERMINAL}
d_hat = {s: np.zeros(3) for s in TERMINAL}
```

With all-zero T̂, `Q = 0` everywhere on trial 1, so the agent chooses uniformly via softmax. After the first visit to each (s, a) pair, T̂ becomes non-zero and planning starts producing meaningful Q-values.

---

## Parameters (4)

| Symbol | Code name | Range | Transform | Meaning |
|--------|-----------|-------|-----------|---------|
| `γ` | `gamma` | [0, 1] | logit/sigmoid | Discount factor — weight of future rewards |
| `τ` (paper: λ_β) | `tau` | (0, ∞) | log/exp | Softmax temperature — higher = more random |
| `α_T` | `alpha_t` | [0, 1] | logit/sigmoid | Learning rate for transition model T̂ |
| `α_φ` | `alpha_r` | [0, 1] | logit/sigmoid | Learning rate for resource quantity model d̂ |

**On τ vs. λ_β:** The paper writes the softmax temperature as λ_β and notes it deliberately uses "temperature" (higher = more stochastic) rather than the conventional "inverse temperature" (β). The code uses `tau` for the same quantity, in the same direction.

---

## Planning: Value Iteration

Before acting on each trial, the agent runs **value iteration** through its current world model to compute Q-values for the current goal `w_g`. Because the tree is only 2 levels deep, this resolves analytically in exactly 2 backward steps — no iterative loop needed.

### Stage 2 Q-values (Eq. 9 in paper)

For each intermediate state `s1 ∈ {1, 2, 3}` and each action `a1`:
```
Q(g, s1, a1) = Σ_{s2} T̂(s2|s1,a1) · [d̂(s2) · w_g]
```
No future term because `s2` is terminal — `V(s2) = 0`.

### Stage 1 Q-values (Eq. 9 applied to s=0)

For each action `a0` from the root:
```
Q(g, 0, a0) = Σ_{s1} T̂(s1|0,a0) · γ · max_{a1} Q(g, s1, a1)
```
Bootstraps from the best value achievable at each reachable intermediate state.

### In code ([models/mb.py:53-73](models/mb.py#L53-L73)):
```python
def _value_iteration(self, T_hat, d_hat, w_g, gamma):
    Q = {}
    # Stage-2
    for s1 in [1, 2, 3]:
        succs = self.SUCCESSORS[s1]
        Q[s1] = np.array([
            sum(T_hat[s1][a1][i] * np.dot(d_hat[s2], w_g)
                for i, s2 in enumerate(succs))
            for a1 in range(Env.N_ACTIONS[s1])
        ])
    # Stage-1
    Q[0] = np.array([
        sum(T_hat[0][a0][i] * gamma * Q[s1].max()
            for i, s1 in enumerate(self.SUCCESSORS[0]))
        for a0 in range(Env.N_ACTIONS[0])
    ])
    return Q
```

Note: the Stage-1 backup uses `Q[s1].max()` — the best action at the next state. This is valid because MB is a *planning* agent that assumes it will act optimally in the future. This is not a SARSA vs. Q-learning distinction; it is exact Bellman optimality applied to the internal model (Eq. 8 in the paper).

---

## Decision Policy

Same softmax as MF, applied to the planned Q-values:

```
π(a | s, g) = exp(Q(g, s, a) / τ) / Σ_{a'} exp(Q(g, s, a') / τ)
```

Q-values are recomputed fresh from T̂ and d̂ before each trial. There is no stored Q-table that persists across trials — only the world model (T̂, d̂) persists.

---

## Learning: World-Model Updates (Eqs. 11–12 in paper)

After observing each transition, both T̂ and d̂ are updated via Rescorla-Wagner / exponential moving average:

### Transition update (Eq. 11)

```
T̂(s'|s, a) ← T̂(s'|s, a) + α_T · (𝟙[s' = s_observed] − T̂(s'|s, a))   ∀s'
```

This is applied to both transitions in the trial: `(0, a0) → s1` and `(s1, a1) → s2`.

### Resource update (Eq. 12)

```
d̂(s2) ← d̂(s2) + α_φ · (φ(s2) − d̂(s2))
```

Only the terminal room actually visited (`s2`) is updated. The true resource vector `φ(s2)` is observed from the environment.

### In code ([models/mb.py:107-115](models/mb.py#L107-L115)):
```python
def _update_T(s, a, s_obs):
    succs  = self.SUCCESSORS[s]
    target = np.array([1.0 if s_ == s_obs else 0.0 for s_ in succs])
    T_hat[s][a] += alpha_t * (target - T_hat[s][a])

_update_T(0,  a0, s1)
_update_T(s1, a1, s2)
d_hat[s2] += alpha_r * (Env.phi(s2) - d_hat[s2])
```

Both the world-model updates are **goal-independent** — T̂ and d̂ do not depend on `w_g`. This is the key transfer advantage: a room visited under one goal teaches d̂ something useful for all other goals.

---

## Full Per-Trial Flow

```
1. Look up current goal g → get w_g
2. Plan: run value iteration through T̂, d̂, w_g, γ → compute Q[s] for all s
3. Stage 1 (s=0):
     - π0 = softmax(Q[0], τ)
     - Sample or observe a0; accumulate log P(a0)
     - Transition: s1 = Env.step(0, a0)
4. Stage 2 (s1):
     - π1 = softmax(Q[s1], τ)
     - Sample or observe a1; accumulate log P(a1)
     - Transition: s2 = Env.step(s1, a1)
5. Update world model:
     - T̂[0][a0]  ← update toward s1
     - T̂[s1][a1] ← update toward s2
     - d̂[s2]     ← update toward φ(s2)
6. Record actions
```

The planning step (2) uses the world model *before* it is updated for this trial — the agent acts on its prior beliefs, then learns from the outcome.

---

## MB vs. MF: The Key Contrast

| | MB | MF |
|---|---|---|
| **Stores** | T̂(s'\|s,a), d̂(s) | Q(g,s,a) |
| **Plans** | Yes — value iteration each trial | No |
| **Transfers across goals** | Yes — d̂ is goal-independent | No — separate Q-table per goal |
| **Transfers across states** | Yes — T̂ is shared | No |
| **Parameters** | 4 (γ, τ, α_T, α_φ) | 3 (γ, α, τ) |
| **Sample efficiency** | High (can generalize immediately) | Low (needs many trials per goal) |
| **Computational cost** | Higher (planning per trial) | Lower |

In this deterministic 2-level tree, T̂ converges quickly — after seeing each (s, a) pair once, the transition is known with certainty if α_T = 1. The MB agent's main advantage over MF is this cross-goal generalization via d̂.

---

## Role in the Paper

MB is the strongest classical RL baseline. Its limitation is that it does not account for the habitual/default-policy bias observed empirically — it purely maximizes expected reward. The resource-rational models (RRMB, RRSF) extend this framework by adding a cognitive cost term that penalizes deviation from a default policy, which better explains the SUD group's behavior.

---

---

# Successor Features (SF) RL — Detailed Explanation

## Overview

Successor Features occupy a middle ground between MF and MB. Like MF, SF learns from experience without an explicit world model. Like MB, SF achieves cross-goal transfer — but through a different mechanism. Instead of learning scalar Q-values, SF learns **vector-valued state occupancy expectations** (Ψ) that are goal-independent. Q-values for any goal are then recovered on-the-fly by a single dot product.

The paper describes SF as:
> *"SF learning neither models full state transitions as in MB learning nor learns action-reward associations as in MF learning. As such, SF learning balances computational cost and flexibility to multiple tasks. The key of SF learning is to learn task structure by extracting the successor features of each state, which can serve as shared components across tasks and supports generalization to novel goals."*

A key design choice in this paper: **a single shared Ψ is learned across all four training tasks**, rather than a separate set per goal. The paper notes this "simplified approach provided a better fit to human behavioral data."

---

## The Core Idea: What Are Successor Features?

In standard MF, `Q(g, s, a)` conflates two things:
1. **Where** action `a` tends to lead (task structure)
2. **How valuable** those destinations are under goal `g` (reward)

Successor features decouple these. Ψ(s, a) is a **3D vector** encoding:
> *the discounted expected future accumulation of resource quantities, starting from (s, a)*

More precisely (Eq. 28 in the paper):
```
Ψ(s, a) = E[ Σ_{k=0}^∞ γ^k φ(s_{t+k+1}) | s_t=s, a_t=a ]
```

where φ(s) = d(s) is the goal-independent resource vector at state s. Because φ is goal-independent, Ψ captures **task structure** — not reward. The reward signal enters only when computing Q:

```
Q_g(s, a) = Ψ(s, a) · w_g        (Eq. 30)
```

**Concrete example:** If Ψ(s=0, a=2) ≈ [0, 10, 6] (the right door leads mostly to room 8, which has resources [0,10,6]), then:
- Under `A_easy` (w=[-1,1,0]): Q = -0 + 10 - 0 = 10 → good choice
- Under `B_easy` (w=[1,-1,0]): Q = 0 - 10 - 0 = -10 → bad choice

The same Ψ serves all goals.

---

## Data Structure

```python
Psi[s]  →  np.array of shape (N_ACTIONS[s], PHI_DIM=3)
```

`Psi[s][a]` is the 3D successor feature vector for taking action `a` at state `s`.

- `Psi[0]` has shape `(3, 3)` — 3 actions × 3 resource dimensions
- `Psi[1]`, `Psi[2]`, `Psi[3]` each have shape `(2, 3)` — 2 actions × 3 dimensions
- **Shared across all goals** — there is no per-goal indexing

### Initialization ([models/sf.py:26-29](models/sf.py#L26-L29))

```python
def _init_Psi(self):
    return {s: np.zeros((Env.N_ACTIONS[s], self.PHI_DIM))
            for s in Env.NON_TERMINAL}
```

All zeros. The paper does not state this explicitly for SF, but zero-initialization is consistent with the MF and MB conventions stated in the paper.

---

## Parameters (3)

| Symbol | Code name | Range | Transform | Meaning |
|--------|-----------|-------|-----------|---------|
| `γ` | `gamma` | [0, 1] | logit/sigmoid | Discount factor — how far ahead Ψ integrates |
| `α_ψ` (paper) | `alpha_sf` | [0, 1] | logit/sigmoid | Learning rate for Ψ updates |
| `τ` (paper: λ_β) | `tau` | (0, ∞) | log/exp | Softmax temperature |

Same count as MF (3 parameters). The subscript `_sf` distinguishes the SF learning rate from the MF Q-learning rate `alpha` even though they play the same structural role.

---

## Q-Value Reconstruction

At the start of each trial, Q-values are computed from the current Ψ and the current goal's price vector:

```
Q_g(s, ·) = Ψ[s] @ w_g
```

This is a matrix-vector product: `Psi[s]` has shape `(n_actions, 3)` and `w_g` has shape `(3,)`, yielding a `(n_actions,)` Q-value vector. There is no learned Q-table — Q is always derived fresh.

### In code ([models/sf.py:44, 54](models/sf.py#L44)):
```python
q0  = Psi[0] @ w_g    # shape (3,) — Q-values at root
q1  = Psi[s1] @ w_g   # shape (2,) — Q-values at intermediate state
```

---

## The Ψ Update (SARSA-Style TD, Eq. 29)

Ψ is updated after each trial via SARSA-style TD. The target for each (s, a) pair is the resource features actually encountered, plus a discounted bootstrap from the *next (s, a) pair actually visited*.

### General form (Eq. 29):
```
Ψ(s_t, a_t) ← Ψ(s_t, a_t) + α_ψ · [φ(s_{t+1}) + γ · Ψ(s_{t+1}, a_{t+1}) − Ψ(s_t, a_t)]
```

The update is SARSA-style: `a_{t+1}` is the action actually taken at the next state — not an argmax. This mirrors the MF update structure.

### Two updates per trial:

**Step 1 — s=0 → s1 (non-terminal):**
```
Ψ[0][a0] ← Ψ[0][a0] + α_ψ · (φ(s1) + γ · Ψ[s1][a1] − Ψ[0][a0])
```
Since `s1` is non-terminal, `φ(s1) = 0`. The update simplifies to:
```
Ψ[0][a0] ← Ψ[0][a0] + α_ψ · (γ · Ψ[s1][a1] − Ψ[0][a0])
```

**Step 2 — s1 → s2 (terminal):**
```
Ψ[s1][a1] ← Ψ[s1][a1] + α_ψ · (φ(s2) − Ψ[s1][a1])
```
Since `s2` is terminal, there is no next action — the `γ · Ψ(s2, ·)` bootstrap term drops out entirely. The target is just the observed resource vector at the terminal room.

### In code ([models/sf.py:63-67](models/sf.py#L63-L67)):
```python
# s0→s1: φ(s1)=0 for non-terminal s1
Psi[0][a0]  += alpha_sf * (gamma * Psi[s1][a1] - Psi[0][a0])
# s1→s2: s2 is terminal, γ·Ψ(s2,·)=0
Psi[s1][a1] += alpha_sf * (Env.phi(s2) - Psi[s1][a1])
```

### What Ψ converges to:

At convergence, `Psi[s1][a1]` equals the true resource vector of the terminal room reachable from `(s1, a1)`. Since transitions are deterministic:
- `Psi[1][0]` → `φ(s=4) = [10, 0, 0]`
- `Psi[1][1]` → `φ(s=5) = [4, 5, 10]`
- `Psi[3][0]` → `φ(s=8) = [0, 10, 6]`
- etc.

And `Psi[0][a0]` → `γ · Psi[s1][a1]`, the discounted resource expectation one step ahead. Plugging any `w_g` into `Psi[s1] @ w_g` then recovers the true Q-value for that goal immediately.

---

## The Key Insight: No Goal Information in the Update

The paper highlights (after Eq. 29):
> *"Note that we are not adding the information term when updating the successor feature, because these features represent the participant's understanding of the task structure, which is independent from the task context."*

Crucially, **`w_g` never appears in the Ψ update** — only `φ(s2)` (the raw resource vector) does. This means every trial, regardless of goal, contributes to the same shared Ψ — all 80 trials are informative. Contrast with MF, where `Q[g][s][a]` is indexed by goal: a trial under `A_easy` only updates `Q['A_easy']`, so each of the 4 goal-specific tables receives only ~20 updates across training. SF effectively gets 4× the data for learning task structure.

---

## Full Per-Trial Flow

```
1. Look up current goal g → get w_g
2. Stage 1 (s=0):
     - q0 = Psi[0] @ w_g    (Q-values via dot product)
     - π0 = softmax(q0, τ)
     - Sample or observe a0; accumulate log P(a0)
     - Transition: s1 = Env.step(0, a0)
3. Stage 2 (s1):
     - q1 = Psi[s1] @ w_g
     - π1 = softmax(q1, τ)
     - Sample or observe a1; accumulate log P(a1)
     - Transition: s2 = Env.step(s1, a1)
4. Ψ updates:
     - Psi[0][a0]  += α_ψ * (γ * Psi[s1][a1] − Psi[0][a0])
     - Psi[s1][a1] += α_ψ * (φ(s2)           − Psi[s1][a1])
5. Record actions
```

---

## SF vs. MF vs. MB

| | SF | MF | MB |
|---|---|---|---|
| **Stores** | Ψ(s,a) ∈ ℝ³ | Q(g,s,a) ∈ ℝ | T̂(s'\|s,a), d̂(s) |
| **Q-value source** | Ψ · w_g (dot product) | Looked up directly | Value iteration |
| **Transfer across goals** | Yes — Ψ is goal-independent | No | Yes — via d̂ |
| **World model** | No | No | Yes |
| **Update uses goal?** | No | Yes | No (T̂, d̂ updates) |
| **Parameters** | 3 (γ, α_ψ, τ) | 3 (γ, α, τ) | 4 (γ, τ, α_T, α_φ) |

SF and MB achieve cross-goal transfer through different routes: SF via learned feature expectations (Ψ), MB via a learned world model (T̂, d̂). SF is more sample-efficient than MF on multi-goal problems, and cheaper than MB (no planning step per trial), but unlike MB it cannot reason about states it has never visited.

---

## Role in the Paper

SF is the foundation for the best-fitting model (RRSF). The classical SF model itself is expected to outperform MF (due to cross-goal transfer) and potentially match MB in this simple environment. RRSF extends SF by replacing the softmax policy with a resource-rational policy that penalizes deviation from a default π_0 — this addition is what captures the SUD group's effort-sensitive behavior.
