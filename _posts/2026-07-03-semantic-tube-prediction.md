---
layout: post
title: "Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA"
date: 2026-07-03 15:24:02 +0800
categories: paper-notes
---

**Paper:** [arXiv:2602.22617](https://arxiv.org/abs/2602.22617)
**Authors:** Hai Huang (Atlassian), Yann LeCun (NYU), Randall Balestriero (Brown)
**Date:** February 2026
**Code:** github.com/galilai-group/llm-jepa#stp

---

## 1. Core Problem: Scaling Laws Are Descriptive, Not Prescriptive

LLM scaling laws (Chinchilla, Kaplan) accurately predict that loss decreases as a power law with compute, data, and parameters. But these are **descriptive** — they describe what happens with standard training, not what's **optimal**. Very few works have successfully beaten the data-efficiency bounds implied by these laws.

The paper argues this is because **next-token prediction (NTP) alone is insufficient** — it conflates surface statistical noise with the global semantic trajectory. NTP only cares about getting the next token right; it doesn't enforce any structure on _how_ the hidden state evolves across the sequence. This leaves the model under-constrained, requiring massive data to disambiguate signal from noise.

---

## 2. The Geodesic Hypothesis

The paper proposes a geometric theory of token sequences:

1. **Training as an ODE:** Token sequence dynamics can be modeled as `dx/dt = u(f(x))` — an ODE in the sequence embedding space. The Picard-Lindelöf theorem guarantees unique solutions from distinct initial conditions, theoretically **ruling out mode collapse**.

2. **Principle of Least Action:** Error-free token sequences follow **geodesics** (shortest paths) on a smooth semantic manifold. These geodesics are **locally linear almost everywhere** — any short segment can be approximated by a straight line.

3. **Noise decomposition:** The deviation of actual hidden states from the geodesic can be decomposed into a **parallel component** (signal — useful semantic evolution) and a **perpendicular component** (noise — harmful drift).

Concretely, for indices `s < r < t`:
```
Noise  = (h_r − h_s)_⊥(h_t − h_s)     ← perpendicular to trajectory
Signal = (h_r − h_s)_∥(h_t − h_s)     ← parallel to trajectory
```

NTP only optimizes the parallel component; the perpendicular noise accumulates and causes mode collapse at inference time.

### Five Predictions (P1–P5)

- **(P1)** NTP alone is insufficient for high-quality generation — L_NTP plateaus while L_STP should continue decreasing
- **(P2)** Semantic Tube improves SNR, resulting in superior data efficiency and accuracy
- **(P3)** Semantic Tube preserves diversity by preventing trajectory collisions
- **(P4)** λ ≪ 1 is preferred to accommodate instances where the geodesic deviates from a straight line
- **(P5)** The identity function serves as a superior predictor compared to learned projections

---

## 3. The Solution: Semantic Tube Prediction (STP)

STP is a **JEPA-style auxiliary loss** that constrains hidden state trajectories to stay within a tubular neighborhood of the geodesic. The loss is remarkably simple:

```
L_STP = 1 − cos(h_t − h_r, h_r − h_s)
```

where `s < r < t` are three randomly selected token positions. This is just the cosine distance between two consecutive trajectory segments. Minimizing it encourages collinearity — forcing the trajectory to be locally straight.

The full training objective:

```
L = L_NTP + λ · L_STP
```

where `λ` is typically small (0.01–0.08).

### Why This Generalizes JEPA to Language

JEPA works in vision by learning to predict the representation of one view from another, but requires explicit two-view augmentations. LLM-JEPA tried this for language but was bottlenecked by manual two-view scaffolding and extra forward passes.

STP eliminates both problems:
- **No two-view scaffolding needed** — `s`, `r`, `t` are randomly sampled from any sequence. The "views" are different segments of the same trajectory.
- **Identity predictor** — since the geodesic is locally linear, `h_r − h_s` already points toward `h_t`, so the optimal predictor is identity (no extra network needed).
- **Negligible compute overhead** — just computing cosine similarity from already-available hidden states.

### Implementation (HuggingFace)

```python
# Get per-token hidden states from last layer
h = outputs.hidden_states[-1]  # [batch, seq_len, d_model]

# Pick random indices s < r < t
s, r, t = sample_indices(seq_len)

# Compute STP loss
stp_loss = 1 - F.cosine_similarity(h[:,t] - h[:,r], h[:,r] - h[:,s])
loss = ntp_loss + lambda * stp_loss
```

Zero extra forward passes. STP is applied only during training — not needed at inference.

---

## 4. Key Theoretical Results

**Theorem 3.3 (Semantic Tube):** If `h*` is locally linear and `L_STP → 0` for all `s < r < t`, then `∥h_r − h*∥ ≲ ε` — the hidden state trajectory is confined within a tube centered on the optimal geodesic.

**Inference Cone (Appendix F):** Without STP, the accumulated noise forms a Brownian motion that causes trajectories to diverge into a **cone** whose radius grows as `σ√t`. Different sequences' cones can collide → mode collapse. STP reduces `σ` (the noise magnitude), narrowing the cone and preventing collisions — thus **preserving diversity**.

**SNR → Data Efficiency (Appendix H):** Under a Gaussian channel model, mutual information `I(X;Y) = ½log(1+SNR)`. The required training data `m` is inversely proportional to `log(1+SNR)` — doubling SNR directly reduces data requirements. The error probability similarly decreases with SNR.

---

## 5. Key Results

### 5.1 NTP Alone Doesn't Minimize STP Loss (P1)

In standard fine-tuning, `L_NTP` plateaus while `L_STP` remains high (~1.4, indicating near-perpendicular trajectory segments — essentially Brownian motion). With STP, `L_STP` drops to 0.6 even after `L_NTP` plateaus. Minimizing NTP does not automatically produce coherent trajectories.

### 5.2 Massive Data Efficiency Gains (P2)

On the NL-RX-SYNTH dataset (natural language → regular expression):

| Training data | NTP only | NTP + STP |
|---|---|---|
| 1/1 (full) | Baseline | Baseline+ |
| 1/2 | Significant drop | **Negligible drop** |
| 1/16 | Severe degradation | **Matches full-dataset NTP accuracy** |

STP matches the full-dataset baseline using only **1/16** of the training data — directly violating Chinchilla-style scaling laws. This holds across Llama 1B, 3B, and 8B.

### 5.3 Better Accuracy Across the Board

STP outperforms both regular fine-tuning and LLM-JEPA across 6 datasets (NL-RX-SYNTH, NL-RX-TURK, GSM8K, Spider, NQ-Open, HellaSwag), 6 model families (Llama, Gemma, OpenELM, Qwen, DeepSeek-R1-Distill, OLMo), and 3 model sizes (1B, 3B, 8B).

### 5.4 Diversity Preservation (P3)

STP preserves nuanced diversity that other methods collapse. On a regular expression dataset where ".*" appears 35× more often than ".*.*" (functionally equivalent suffixes):

| Suffix | STP | Regular FT | LLM-JEPA |
|---|---|---|---|
| `.*` (majority) | 88.5% | 29.9% | 68.9% |
| `.*.*` (minority) | 68.0% | 28.0% | 32.0% |

STP learns both; others collapse toward the majority or fail on both. SVD analysis reveals STP exhibits **polymorphism**: on normalized vectors it aligns with LLM-JEPA (simple directional structure), but on unnormalized vectors it resembles regular fine-tuning (tolerating magnitude complexity).

### 5.5 Small λ Works Best (P4)

Optimal λ consistently falls in the 0.01–0.08 range across all configurations. The accuracy-vs-λ curve is concave — too little STP is ineffective, too much over-constrains the trajectory.

### 5.6 Identity Predictor Outperforms Learned Projections (P5)

A variant that trains a learned projector `P` to predict `h_t − h_s` from `h_r − h_s` consistently underperforms the identity formulation. The geodesic is straight enough that the identity function is the optimal predictor.

---

## 6. What Is JEPA? (Joint Embedding Predictive Architecture)

### The Problem JEPA Solves

Given two different views of the same thing (e.g., front and side photos of a cat), how do you learn representations where both views are close in representation space? Three approaches:

| | Reconstruction | Contrastive | JEPA |
|---|---|---|---|
| **Method** | Reconstruct pixels from masked input | Push different images apart, pull similar together | Predict representation of one view from another |
| **Problem** | Wastes capacity on pixel-level trivia | Needs negatives; can collapse | Needs stop-grad to prevent collapse |
| **Examples** | VAE, MAE | SimCLR, CLIP | I-JEPA, V-JEPA, data2vec |

### How JEPA Works (I-JEPA example)

```
Step 1: Encode the context
┌─────────────────────────────┐
│  [context encoder]           │
│  Input: top half of cat img  │
│  Output: z_context (vector)  │
└─────────────────────────────┘

Step 2: Encode the target (STOP GRADIENT!)
┌─────────────────────────────┐
│  [target encoder] —— STOP GRADIENT
│  Input: bottom half of cat   │
│  Output: z_target (vector)   │
└─────────────────────────────┘

Step 3: Predict across views
┌─────────────────────────────┐
│  [predictor network]         │
│  Input: z_context            │
│  Output: z_pred              │
│  Loss: ∥z_pred − z_target∥² │
└─────────────────────────────┘
```

The predictor is a small network that says: "given what I see in the top half, what should the representation of the bottom half look like?" The target encoder uses **stop-gradient** — its output is a fixed target. This is crucial: without stop-grad, both encoders would collapse to outputting zero and the predictor would learn to output zero, trivially satisfying the loss.

### Why This Works

The predictor must learn **semantic relationships** between views — "cat ear shape in the top" implies "cat body in the bottom." This can't be done from memorized pixels since the views share no pixels. The representation must capture what makes a cat a cat across perspectives.

### The Challenge for Language

Images naturally have spatial views (different crops). Language is sequential — you can't "crop" a sentence. LLM-JEPA (Huang et al., 2025) tried (query, answer) pairs as two views but required manual scaffolding and extra forward passes.

### STP's Insight

For token sequences, **the two views are different time windows of the same sequence.** View 1 = segment `[s, r]`. View 2 = segment `[r, t]`. The predictor asks: "given the trajectory from s to r, where should it go from r to t?" And because of the Geodesic Hypothesis (trajectories are locally straight), the optimal predictor is just identity — `h_r − h_s` already points toward `h_t − h_r`. The loss simplifies to the cosine distance between consecutive segments.

| | Reconstruction | Contrastive | JEPA | STP |
|---|---|---|---|---|
| **What to learn** | Reconstruct pixels | Separate different images | Predict another view's representation | Predict next segment's direction |
| **What's the view?** | Same image, masked parts | Different images | Different crops of same image | Different time windows of same sequence |
| **Predictor?** | Full decoder network | None (matching) | Learned predictor P | **Identity** (geodesic is straight) |
| **Why it works** | Information bottleneck | Push-pull dynamics | Semantic invariance across views | Collinearity constraint |

### DINO vs. JEPA

DINO/DINOv2 is **not** JEPA — it's self-distillation. Two views go through student and teacher (EMA of student), and the loss matches their outputs. No explicit predictor. The "prediction" is implicit in the student-teacher dynamics. JEPA has an explicit learned predictor `P(z_context) → z_target` with stop-gradient on the target.

### The Lineage

```
Energy-Based Models (LeCun 2006)
    │
    ▼
JEPA framework (LeCun 2022) ─── "predict one view from another"
    │
    ├── DINO/DINOv2: implicit prediction via self-distillation
    ├── I-JEPA/V-JEPA: explicit predictor for images/video
    ├── LLM-JEPA (Huang et al. 2025): JEPA for language, needs two-view scaffolding
    └── STP (this paper): JEPA for language, identity predictor, no scaffolding
```

---

## 7. Connection to Broader Frameworks

- **Energy-Based Models:** While EBMs minimize energy at specific states, STP minimizes the **action** — the integral of the Lagrangian along the trajectory. This generalizes state-wise energy minimization to trajectory-wise action minimization.

- **Linear Representation Hypothesis:** If token sequences follow locally linear geodesics, vector arithmetic (`v_Paris − v_France + v_Italy ≈ v_Rome`) emerges naturally from path linearity.

- **Curvature Straightening:** Prior work observes that training straightens curvature between consecutive tokens — interpreted here as the geodesic approximating a straight line.

- **Exposure Bias:** STP addresses the classic problem where teacher forcing causes drift at inference time by explicitly constraining the hidden state trajectory.

---

## 8. Key Takeaways

1. **Scaling laws can be beaten.** The Chinchilla data-efficiency bound is an artifact of NTP-only training, not a fundamental limit. Adding geometric structure achieves 16× data efficiency.

2. **NTP alone under-constrains the trajectory.** Getting the next token right doesn't guarantee coherent hidden state paths. Perpendicular noise accumulates, causing mode collapse.

3. **A simple cosine loss is sufficient.** No extra networks, no two-view scaffolding, no additional forward passes. Random triplets of positions provide enough signal.

4. **The geodesic is a self-consistency condition.** It bridges JEPA, EBMs, and the Linear Representation Hypothesis within a single geometric framework grounded in the Principle of Least Action.
