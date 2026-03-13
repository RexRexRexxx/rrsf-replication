# RL Models in "Imbalanced Learning Efficiency and Cognitive Effort in Individuals with Substance Use Disorder"

**Authors:** Zeming Fang, Yu-Yan Gao, Yu-Feng Xia, Zijian Cheng, Lingxiao Yang, Wei Li, Lei Guo, Ru-Yuan Zhang

---

## Overview

The paper tests 8 computational models across two frameworks to explain multi-task reinforcement learning behavior in substance use disorder (SUD) vs. healthy controls (HC).

The central thesis: addiction is not simply a shift from Model-Based to Model-Free learning, but rather reflects heightened **cognitive effort sensitivity** that drives reliance on simplified "default" policies — captured by the **Resource-Rational Successor Features (RRSF)** model, which best fits human behavior.

---

## Task Environment

### Tree Structure

The task is a **deterministic, two-stage navigation** through a tree of states. Each trial has **3 time steps** (t = 0, 1, 2):

```
                        s=0 (root)
                       /    |    \
                    s=1    s=2    s=3        ← intermediate rooms (Stage 1)
                   / \    / \    / \
                 s=4 s=5 s=6 s=7 s=8 s=9   ← terminal rooms (Stage 2)
```

- **Stage 1 (t=0→1):** From root s=0, participant chooses one of 3 doors → arrives at an intermediate room (s=1, s=2, or s=3).
- **Stage 2 (t=1→2):** From an intermediate room, participant chooses one of 2 doors (color-matched to Stage 1 choice) → arrives at a terminal room.
- **Total states:** 10 (s=0 through s=9). Rooms 0–3 are non-terminal; rooms 4–9 are terminal.
- All transitions are **deterministic**.

**Known terminal transitions (from SI Note 5):**
- s=2, right → s=7 (deterministic)
- s=2, left  → s=6
- s=3, left  → s=8 (deterministic; optimal for Type A tasks)
- s=3, right → s=9 (large negative reward in Type A tasks)
- s=1, left  → s=4 (inferred; s=4 is most rewarding for Type B)
- s=1, right → s=5 (inferred)

### Reward Function

Only terminal rooms (s=4–9) deliver rewards. Reward is computed as:
```
R = w_g · d(s)
```
where:
- `d(s)` = resource quantity vector at terminal room s (the "goods" received)
- `w_g` = goal-specific price vector (the "prices" shown to participant at trial start)
- `g` = current goal/task, which defines the price set

Non-terminal rooms deliver no reward: `d(s) = 0` for s ∈ {0, 1, 2, 3}.

### Feature Vectors φ(s)

In the Successor Features framework, `φ(s) = d(s)` — the resource quantity vector at state s. This is **goal-independent**, so it can be learned once and reused across all tasks.

- `φ(s)` = `d(s)` for terminal rooms (nonzero resource quantities)
- `φ(s)` = `0` for non-terminal rooms

Each `φ(s)` is a **3-dimensional vector** (3 resource types).

### Goal / Task Types

There are **4 training tasks** in a 2×2 design, plus 1 test-only task:

| Task | `w_g` | Optimal room | Optimal path | Max reward |
|------|--------|-------------|-------------|------------|
| Type A easy | `[−1, 1, 0]` | s=8 (R=10) | right → left | 10 |
| Type B easy | `[1, −1, 0]` | s=4 (R=10) | left → left | 10 |
| Type A hard | `[−2, 1, 0]` | s=8 (R=10) | right → left | 10 |
| Type B hard | `[1, −2, 0]` | s=4 (R=10) | left → left | 10 |
| Test only   | `[1, 1, 1]`  | s=6 (R=28) | mid → left  | 28 |

- **Type A** ("right-left", complex): Optimal path is s=0→s=3 (right door), then s=3→s=8 (left door). Room s=9 is catastrophic (R=−8 easy, R=−16 hard).
- **Type B** ("left-left", simple): Optimal path is s=0→s=1 (left door), then s=1→s=4 (left door).
- Easy/hard share the same optimal room; hard tasks amplify penalties for suboptimal rooms.

**Full reward matrix** (`R = w_g · d(s)`, computed):

| Room | d(s) | A easy | B easy | A hard | B hard | Test |
|------|------|--------|--------|--------|--------|------|
| s=4  | [10,0,0] | −10 | **10** | −20 | **10** | 10 |
| s=5  | [4,5,10] |   1 |   −1 |   −3 |   −6 | 19 |
| s=6  | [9,9,10] |   0 |    0 |   −9 |   −9 | **28** |
| s=7  | [0,8,2]  |   8 |   −8 |    8 |  −16 | 10 |
| s=8  | [0,10,6] | **10** | −10 | **10** | −20 | 16 |
| s=9  | [8,0,0]  |  −8 |    8 |  −16 |    8 |  8 |

**Trial structure:**
- **80 training trials:** 4 price sets × 20 trials each, randomized order
- **Test trials:** Use `w=[1,1,1]` exclusively — not included in model fitting/analysis
- All Q-values / Ψ are updated across all 80 training trials

**Goal distribution:** Uniform over 4 training tasks — `ρ(g) = 0.25` for each.

**Goal stability within trial:** The goal `g` remains constant throughout all steps (t=0,1,2) of a single trial.

### Participants & Data

- 126 total participants: 43 HC, 83 SUD (all male)
  - *(Note: paper cites 128 in the model recovery description, but demographics table shows 43 HC + 83 SUD = 126.)*
- ~30-minute multi-task learning phase + resource quantity memory test
- All training and testing trials used jointly for model fitting (no cross-validation)

---

## Framework 1: Classical RL (5 models)

**Objective:** Maximize expected cumulative reward:
```
max_π ∑_g ρ(g) ∑_t γ^t R_t
```

---

### Model 1: Model-Based (MB)

Learns a **goal-independent** world model (transition dynamics `T̂` and resource quantities `d̂`), then recomputes `Q(g,s,a)` fresh each trial via exact value iteration using the current goal's price vector `w_g`.

**Bellman optimality (per goal g):**
```
V*(g,s) = max_a ∑_s' T̂(s'|s,a) [R̂_g(s') + γ V*(g,s')]
Q(g,s,a) = ∑_s' T̂(s'|s,a) [R̂_g(s') + γ V*(g,s')]
```
where `R̂_g(s') = w_g · d̂(s')`. Since the tree is 2 levels deep, this solves analytically in 2 steps — no iterative loop needed.

**Policy (softmax):**
```
π(a|s,g) = exp(Q(g,s,a)/τ) / ∑_a' exp(Q(g,s,a')/τ)
```

**Learning rules (trial-by-trial, goal-independent):**
- Transition model: `T̂(s'|s,a) ← T̂(s'|s,a) + α_T · [𝟙(s_t+1=s') − T̂(s'|s,a)]`
- Resource quantities: `d̂(s) ← d̂(s) + α_R · [d_observed − d̂(s)]`

`T̂` and `d̂` are shared across goals — only `w_g` changes per trial.

**Parameters (4):** `γ`, `τ`, `α_T`, `α_R`

---

### Model 2: Model-Free (MF)

Directly learns action-value functions via **temporal-difference (TD) learning**, no world model.

The Q-table is **goal-conditioned**: `Q(g, s, a)` — a separate Q-table is maintained for each goal `g`. This allows the agent to learn different action values for different reward structures while still relying on pure TD without a world model. Updates only affect the Q-table for the current trial's goal.

**TD update:**
```
Q(g, s, a) ← Q(g, s, a) + α [R + γ max_a' Q(g, s', a') − Q(g, s, a)]
```
where `δ = R + γ max_a' Q(g, s', a') − Q(g, s, a)` is the prediction error, and `R = w_g · φ(s')` is the goal-specific reward.

**Policy (softmax):** Same form as MB, applied to `Q(g, s, ·)` for the current goal `g`.

All Q-values initialized to 0.

**Parameters (3):** `γ`, `α`, `τ`

---

### Model 3: Model-Free with Perseveration (MFP)

Extends MF with a **perseveration** mechanism. Q-table remains goal-conditioned `Q(g, s, a)`; the perseveration memory `M(a|s)` is **task-agnostic** (shared across goals) since it tracks habitual action tendencies independent of reward structure.

**Policy:**
```
π(a|s,g) = (1−λ) · [exp(Q(s,a)/τ) · (ρ + (1−ρ)·M(a))] / Z  +  λ·ε
```
where:
- `M(a)` = distribution of recently chosen actions (perseveration memory)
- `ρ` = tendency to perseverate
- `λ` = lapse rate (probability of random/inattentive choice)
- `Z` = normalization constant

**Perseveration update:**
```
M(a) ← M(a) + α_M · 𝟙[a_t = a]
```

**Initialization:** `M(a)` is initialized from the **normalized group-level histogram of all participants' first actions** (not zero).

**Parameters (6):** `γ`, `α`, `τ`, `α_M`, `ρ`, `λ`

---

### Model 4: Successor Features (SF)

Instead of learning Q-values directly, learns **successor features** Ψ(s,a) — discounted expected cumulative future state-feature vectors. These decouple task structure from task-specific reward, enabling transfer across goals.

**Definition:**
```
Ψ(s,a) = E[∑_{k=0}^∞ γ^k φ(s_{t+k+1}) | s_t=s, a_t=a]
```

**TD update:**
```
Ψ(s,a) ← Ψ(s,a) + α_SF [φ(s') + γ Ψ(s',a') − Ψ(s,a)]
```
where `φ(s') = d(s')` is the resource quantity vector (goal-independent).

**Q-value reconstruction for goal g:**
```
Q_g(s,a) = Ψ(s,a) · w_g
```
where `w_g` is the current goal's price vector.

**Policy:** Same softmax as MF/MB.

**Parameters (3):** `γ`, `α_SF`, `τ`

**Key property:** Ψ is learned once and immediately transfers to any new goal — only `w_g` changes per task.

---

### Model 5: Successor Features with Perseveration (SFP)

Extends SF with the same perseveration mechanism as MFP. Q-values are computed via `Q_g = Ψ · w_g`, then fed into the perseveration policy.

**Parameters (6):** `γ`, `α_SF`, `τ`, `α_M`, `ρ`, `λ`

---

## Framework 2: Resource-Rational RL (3 models)

### Core Idea

Agents do not purely maximize reward — they also minimize **cognitive effort**, defined as the KL divergence between the goal-specific policy and a task-agnostic **default policy** `π_0`:

```
H(a|g,s) = KL( π(·|s,g) || π_0(·|s) )
           = ∑_a π(a|s,g) · log[ π(a|s,g) / π_0(a|s) ]
```

**Resource-rational objective:**
```
Maximize ∑_g ρ(g) ∑_t [ γ^t R_t  −  β · H(π(·|s_t,g) || π_0(·|s_t)) ]
```

- `β` = cost sensitivity (higher → more effort-averse → more reliance on `π_0`)
- `π_0(a|s)` = default policy (learned, task-agnostic baseline)

**Optimal policy** (analytical solution, from KL regularization):
```
π*(a|s,g) = π_0(a|s) · exp(Q_g(s,a) / β) / Z(s,g)

Z(s,g) = ∑_a' π_0(a'|s) · exp(Q_g(s,a') / β)
```

Note: `β` plays the role of temperature in this framework. There is **no separate τ** in resource-rational models (β subsumes it). Low β → stays close to `π_0`; high β → more goal-directed.

**Default policy structure:**
```
π_0(a|s) = (1−λ) · π_0^active(a|s)  +  λ · Uniform(a)
```
**Default policy update (trial-by-trial):**
```
π_0^active(a|s) ← π_0^active(a|s) + α_p · 𝟙[a_t = a, s_t = s]
```
(then renormalize)

**Initialization:** `π_0^active` is initialized from the **normalized group-level histogram of participants' first actions at each state**.

---

### Model 6: Resource-Rational Model-Free (RRMF)

Uses goal-conditioned MF Q-learning `Q(g, s, a)` to estimate values, then plugs into the resource-rational optimal policy. The default policy `π_0` remains task-agnostic.

**Q-value update (same as MF):**
```
Q(s,a) ← Q(s,a) + α [R + γ Q(s',a) − Q(s,a)]
```

**Policy:** Resource-rational optimal policy (see above) with Q from MF.

**Parameters (5):** `β`, `γ`, `α`, `α_p`, `λ`

---

### Model 7: Resource-Rational Model-Based (RRMB)

Incorporates cognitive effort directly into MB planning. Uses the same goal-independent world model (`T̂`, `d̂`) as MB, but replaces hard-max planning with soft value iteration using the resource-rational Bellman equation.

**Resource-rational Bellman equation (per goal g, per state s):**
```
V*(g,s) = max_π [ ∑_a π(a|s) · ∑_s' T̂(s'|s,a)(R̂_g(s') + γ V*(g,s'))
                  − β · KL(π(·|s) || π_0(·|s)) ]
```

The inner optimization has the analytical solution:
```
π*(a|s,g) = π_0(a|s) · exp(Q(g,s,a) / β) / Z(s,g)
```
where `Q(g,s,a) = ∑_s' T̂(s'|s,a)[R̂_g(s') + γ V*(g,s')]` is the MB Q-value, and the soft value function is:
```
V*(g,s) = β · log ∑_a π_0(a|s) · exp(Q(g,s,a) / β)
```

Like MB, `T̂` and `d̂` are shared across goals; `Q(g,s,a)` is recomputed each trial using `w_g`. Value iteration converges in exactly 2 steps for this tree.

**Learning rules:** Same as MB (`α_T` for transitions, `α_R` for resource quantities).

**Parameters (4):** `β`, `γ`, `α_T`, `α_R`

> **Note:** RRMB showed the worst fit and was excluded from model recovery due to computational cost (convergence per participant would take months at scale).

---

### Model 8: Resource-Rational Successor Features (RRSF) — **Best Model**

Combines SF learning with the resource-rational framework. Since `φ(s) = d(s)` is goal-independent, the successor feature update itself incurs no information cost — only the policy step does.

**Successor feature update (same as SF):**
```
Ψ(s,a) ← Ψ(s,a) + α_SF [φ(s') + γ Ψ(s',a') − Ψ(s,a)]
```

**Q-value:**
```
Q_g(s,a) = Ψ(s,a) · w_g
```

**Policy:** Resource-rational optimal policy with Q from SF:
```
π*(a|s,g) = π_0(a|s) · exp(Q_g(s,a) / β) / Z(s,g)
```

**Default policy:** Same update rule as RRMF (initialized from group-level first-action histogram).

**Parameters (5):** `β`, `γ`, `α_SF`, `α_p`, `λ`

---

## Model Fitting

### Estimation

**Method:** Maximum A Posteriori (MAP):
```
θ̂ = argmax_θ  ∑_t log P(a_t | s_t, g_t, θ)  +  log P(θ)
```
where `g_t` = presented goal, `s_t` = current state, `a_t` = action taken.

**Optimizer:** BFGS via `scipy.optimize.minimize` with **40 random initializations** per participant. Take the lowest-loss solution across all runs.

### Parameter Reparameterization (for unconstrained optimization)

| Range | Transform | Inverse |
|-------|-----------|---------|
| [0, 1] (rates, probabilities) | `logit(θ) = log(θ/(1−θ))` | `sigmoid` |
| [0, ∞) (β, τ) | `log(θ)` | `exp` |

**Priors:** Flat/uninformative within bounded ranges.

### Model Comparison

| Metric | Description |
|--------|-------------|
| BIC | Penalizes model complexity: `BIC = k·log(n) − 2·log L` |
| AIC | `AIC = 2k − 2·log L` |
| PXP | Protected Exceedance Probability — accounts for uncertainty in model comparison |

### Model & Parameter Recovery

- For each model: simulate **10 behavioral datasets** per participant using fitted parameters
- Fit all 7 models to each synthetic dataset
- Check that the generating model is most frequently recovered
- RRSF is highly recoverable; RRMB was excluded (computational cost)

---

## Parameter Summary

| Parameter | Symbol | Range | Transform | Role | Models |
|-----------|--------|-------|-----------|------|--------|
| Discount factor | `γ` | [0,1] | logit | Weight of future rewards | All |
| Softmax temperature | `τ` | [0,∞) | log | Policy stochasticity | MB, MF, MFP, SF, SFP |
| Q learning rate | `α` | [0,1] | logit | Speed of Q-value updates | MF, MFP, RRMF |
| SF learning rate | `α_SF` | [0,1] | logit | Speed of Ψ updates | SF, SFP, RRSF |
| Transition learning rate | `α_T` | [0,1] | logit | Speed of T(s'\|s,a) learning | MB, RRMB |
| Reward learning rate | `α_R` | [0,1] | logit | Speed of d(s) learning | MB, RRMB |
| Perseveration learning rate | `α_M` | [0,1] | logit | Speed of M(a) updating | MFP, SFP |
| Perseveration tendency | `ρ` | [0,1] | logit | Strength of repetition bias | MFP, SFP |
| Lapse rate | `λ` | [0,1] | logit | Probability of random choice | MFP, SFP, RRMF, RRSF |
| **Cost sensitivity** | **`β`** | [0,∞) | log | **Sensitivity to cognitive effort** | RRMB, RRMF, RRSF |
| Default policy learning rate | `α_p` | [0,1] | logit | Speed of π_0 updating | RRMF, RRSF |

---

## Key Findings (RRSF Parameters, SUD vs HC)

| Parameter | Direction | t-stat | p-value | Cohen's d | 95% CI |
|-----------|-----------|--------|---------|-----------|--------|
| `β` (cost sensitivity) | SUD **higher** | t=3.694 | p<0.001 | d=0.571 | [1.43, 4.74] |
| `α_SF` (SF learning rate) | SUD **lower** | t=−2.579 | p=0.011 | d=0.433 | [0.54, 4.11] |
| `α_p` (default policy LR) | SUD **higher** | t=2.469 | p=0.017 | d=0.591 | [−4.62, −0.47] |

**Interpretation:** SUD individuals are more sensitive to cognitive costs (`β`↑), learn task structure less efficiently (`α_SF`↓), and over-update their default/habitual policies (`α_p`↑). This produces a bias toward familiar, low-effort rooms (Room 4) even when switching would yield higher rewards.

**Correlations (all participants, RRSF):**
- Cognitive effort ↔ `β`: r=−0.824, p<0.001
- Cognitive effort ↔ `α_SF`: r=0.473, p<0.001
- `β` ↔ cognitive impulsivity: r=0.209, p=0.030

---

## Resolved Implementation Details

All key unknowns are now resolved:

1. **Reward matrix:** `w_g` and `d(s)` fully specified — see Task Environment section above.
2. **Feature dimensionality:** `dim(φ) = dim(w_g) = 3`.
3. **Trial count:** 80 training trials (4 tasks × 20 each, randomized). Test trials use `w=[1,1,1]` and are excluded from analysis.
4. **Tree topology:** s=1→{s=4,s=5}, s=2→{s=6,s=7}, s=3→{s=8,s=9}. Left action from s=3 → s=8; right → s=9.
5. **`w_g` availability:** Provided to the agent at the start of each trial (not learned).
