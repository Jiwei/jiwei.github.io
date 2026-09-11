---
layout: post
title: "Energy-Based Models, JEPA, and the Trajectory View"
date: 2026-07-03 15:45:07 +0800
categories: paper-notes
---

## 1. Energy-Based Models (EBMs)

### The Core Idea

An EBM is a function `E(x, y)` that outputs a single scalar — the **energy** — for any pair of inputs `x` (observation) and `y` (prediction). The only rule:

```
Low energy  = compatible (good prediction)
High energy = incompatible (bad prediction)
```

At inference, given an observation `x`, you find the `y` that minimizes the energy:

```
y* = argmin_y E(x, y)
```

### A Concrete Example

Suppose `x` is a partially occluded image and `y` is a completed image:

```
E(occluded_cat, completed_cat)    = 0.3   ← low energy, makes sense
E(occluded_cat, completed_dog)    = 8.7   ← high energy, nonsense
E(occluded_cat, completed_car)    = 15.2  ← very high energy, absurd
```

The EBM doesn't generate anything — it just scores. If you have a way to search over possible `y` values, the EBM guides you toward good ones. For images, you might run gradient descent in pixel space, starting from noise, following `−∇_y E(x, y)` until you land in a low-energy region.

### The Training Problem

Training an EBM is hard because the loss function has two terms pulling in opposite directions:

```
L = E(x, y_real) − E(x, y_fake)
     ↑ push down      ↑ push up
```

You want to **lower** the energy of real (observed) pairs and **raise** the energy of generated (bad) pairs. But sampling `y_fake` requires finding configurations the model currently thinks are low-energy — which is computationally expensive. This is why EBMs were powerful in theory but difficult to scale for decades.

---

## 2. What Makes Something an EBM

A proper EBM has two essential properties:

1. It assigns a scalar score to any configuration (low = good)
2. It's trained with a **contrastive objective** — push down energy on observed (good) configurations AND push up energy on generated (bad) configurations

The second property is what makes training hard — you need to sample from the model's current belief to find those "bad" configurations.

---

## 3. Is a Cosine Similarity Regularizer an EBM?

**Not really, but it sits on the same spectrum.**

`L_STP = 1 − cos(...)` satisfies property 1: it assigns a low value when the trajectory is collinear (good) and a high value when it deviates (bad). So it **measures** energy. But STP only ever pushes **down** — it never explicitly pushes up. There's no sampling of bad configurations.

### How STP Actually Trains

```
L = L_NTP + λ · L_STP
```

STP is a regularizer — it adds a penalty for non-collinear trajectories. The NTP loss pulls the hidden states toward configurations that predict the right token; STP pulls them toward locally straight paths. Together they shape the trajectory, but neither explicitly says "this configuration has high energy."

### The Spectrum

```
                      scores configurations    trained contrastively
                           │                      │
Pure loss function    ✓    │                      │   (cross-entropy, MSE)
Cosine regularizer    ✓    │                      │   (STP)
JEPA-style loss       ✓    │                 ✓ish │   (prediction error)
Proper EBM            ✓    │                   ✓  │   (explicit energy contrast)
```

---

## 4. Where JEPA Fits In

EBMs treat energy as a function of the **state** — a single configuration `(x, y)`. JEPA shifts this from **state energy** to **prediction consistency**. Instead of asking "is (x, y) a good pair?", JEPA asks "can I predict the representation of one view from another?"

```
I-JEPA:    L = ∥P(z_view1) − z_view2∥²
```

If I can predict `z_view2` from `z_view1`, the two views are consistent — and if they're consistent, the energy of that configuration should be low. The prediction error IS the energy signal.

JEPA is closer to an EBM than STP because the prediction error `∥P(z) − z_target∥²` is computed against a **stopped-gradient target**. The target encoder provides a fixed reference that the predictor must match, creating an implicit contrast between the prediction and a stable reference. There's no explicit `y_fake` sampling, but the stop-grad mechanism creates a similar push-pull dynamic.

STP doesn't even have that. It's a pure self-consistency constraint: "your own trajectory should be collinear." It's closer in spirit to a **smoothness prior** — like L2 weight decay, but applied to the geometry of hidden state trajectories rather than parameter magnitudes.

---

## 5. STP Extends the Principle to Trajectories

Both EBMs and JEPA minimize a **local** quantity:

```
EBM:   minimize E(state)          ← energy at a point
JEPA:  minimize ∥pred − target∥²   ← consistency between two views
STP:   minimize ∫ L(trajectory) dt  ← action along the whole path
```

The Principle of Least Action says that physical systems take the path of minimum action — not minimum energy at each point, but minimum **integrated** energy along the entire trajectory. STP enforces this by constraining hidden state trajectories to be geodesics (locally straight), which are precisely the paths of least action on a manifold.

The paper's claim that STP "generalizes EBM from state-wise energy to trajectory-wise action" is a **conceptual framing**, not a literal implementation claim. STP doesn't implement EBM training. Rather, it occupies the same philosophical position — learn by enforcing consistency constraints — but applies those constraints at the trajectory level rather than the configuration level, and uses a simple regularizer rather than explicit contrastive training.

### The Hierarchy

```
Energy-Based Models (1980s–2000s)
    │
    │  "Score every configuration with a scalar. Low energy = good."
    │  Problem: sampling bad configurations is expensive.
    │
    ▼
JEPA (2022)
    │
    │  "Don't score states. Instead, predict one view's representation
    │   from another. Prediction error IS the energy signal."
    │  Avoids sampling: the negative signal comes from the predictor's
    │  failure, not from explicit contrastive negatives.
    │  Uses stop-gradient on target to create implicit contrast.
    │
    ▼
STP (2026)
    │
    │  "Don't just check consistency between two views. Enforce that
    │   the whole trajectory is a geodesic — a path of least action."
    │  Generalizes point-wise energy to trajectory-wise action.
    │  The predictor collapses to identity because of local linearity.
    │  Pure self-consistency regularizer — no contrastive mechanism.
```

---

## 6. Key Takeaway

The through-line from EBMs → JEPA → STP is a progressive relaxation of the training mechanism while preserving the underlying philosophy:

- **EBMs** require explicit contrastive sampling (hardest training, most general)
- **JEPA** uses stop-grad + prediction error as an implicit contrast signal (easier training)
- **STP** uses a pure self-consistency regularizer — the geodesic hypothesis makes this sufficient (simplest training)

Each step simplifies the training while making a stronger structural assumption about the data. EBM: no assumptions. JEPA: the world has consistent views. STP: consistent views form straight trajectories in semantic space.
