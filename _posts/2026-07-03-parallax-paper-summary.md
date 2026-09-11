---
layout: post
title: "Parallax: Parameterized Local Linear Attention for Language Modeling"
date: 2026-07-03 10:43:46 +0800
categories: paper-notes
---

**Paper:** arXiv:2605.29157v1 (May 2026)
**Authors:** Yifei Zuo (Northwestern), Dhruv Pai (Tilde Research), Zhichen Zeng (UW), Alec Dewulf, Shuming Hu (Tilde Research), Zhaoran Wang (Northwestern)
**Code:** github.com/yifei-zuo/Parallax

---

## Softmax Attention — The Basics

### The Forward Pass

Given a sequence of tokens, each attention layer projects the input into three matrices:

- **Q (Query)** — what each token is "looking for"
- **K (Key)** — what each token "offers" as context
- **V (Value)** — the actual content each token contributes

For a single query at position `i`, softmax attention computes:

```
o_i = Σ p_ij · v_j    where    p_ij = softmax(q_i^T k_j / √d)
```

The softmax does two things simultaneously:
1. **Similarity scoring** — `q_i^T k_j` measures how relevant token `j` is to token `i`
2. **Normalization** — converts raw scores into a probability distribution (non-negative, sum to 1)

The temperature `√d` (where `d` is head dimension) prevents the dot products from growing too large as dimensionality increases, which would push the softmax into near-one-hot behavior.

### The Test-Time Regression Interpretation

At each position `i`, the model faces a **nonparametric regression problem**:

- **Training data**: the preceding KV pairs `D_i = {(k_j, v_j)}_{j≤i}`
- **Test point**: the current query `q_i`
- **Goal**: predict the best output for position `i`

Softmax attention solves this using the **Nadaraya-Watson (NW) estimator** — a kernel-weighted local average:

```
f_NW(q_i) = Σ [ K_h(q_i, k_j) / Σ K_h(q_i, k_j') ] · v_j
```

where the kernel is `K_h(q, k) = exp(q^T k / h)`, making it an **exponential kernel smoother** with bandwidth `h = √d`. The NW estimator fits a **local constant** — it predicts a weighted average of nearby values, treating the underlying function as roughly flat within the kernel's neighborhood.

### What This Means for Associative Memory

This is why Transformers excel at **in-context recall**. When a query token needs to retrieve information from earlier in the context, the softmax kernel concentrates probability mass on the most similar keys, effectively implementing a differentiable key-value lookup. The exponential form of the kernel means that small differences in similarity produce large differences in attention weight — sharp selectivity.

### The Critical Limitation

The NW estimator has a known flaw: **boundary bias**. When the query lies near the boundary of the key distribution, the local constant fit becomes systematically biased because it can only average values on one side. Upgrading from a local **constant** to a local **linear** estimator fixes this — and that's what LLA and Parallax do.

### Quick Summary Table

| Aspect | Softmax Attention |
|--------|------------------|
| Hypothesis space | Local constant |
| Estimator type | Nadaraya-Watson (nonparametric) |
| Complexity | O(L²) compute, O(L) I/O |
| Strength | Sharp associative recall via exponential kernel |
| Weakness | Boundary bias, attention sink on first token, quadratic scaling |
| Regression view | Weighted average of nearby values |

---

## 1. Motivation & Background

### The Problem with Standard Attention
Softmax Attention has remained structurally unchanged since the original Transformer, despite extensive research into efficient alternatives (Linear Attention, SSMs like Mamba, etc.). Those alternatives consistently **underperform on in-context information retrieval** — they lose the associative recall capability that makes Transformers powerful.

### The Test-Time Regression Lens
The paper builds on the **test-time regression framework** (Wang et al., 2025): attention is reinterpreted as solving a regression problem at each token position, where keys are training points, values are labels, and the query is the test point. Under this view:

- **Softmax Attention** = Nadaraya-Watson (local **constant**) estimator — nonparametric, kernel-weighted averaging
- **Linear Attention** = global parametric linear model — suffers from irreducible misspecification error
- **Local Linear Attention (LLA)** (Zuo et al., 2026) = local **linear** estimator — upgrades the constant fit to a linear fit around each query, achieving **provably superior bias-variance tradeoffs**

### Why LLA Hasn't Scaled
LLA requires solving a linear system Σ_i x = μ_i for **every query** via conjugate gradient (CG), causing:
1. **Intensive I/O** — CG iterations dominate memory access
2. **Regularization-expressiveness tradeoff** — large λ makes LLA collapse to Softmax Attention; small λ risks numerical instability
3. **Low-precision incompatibility** — CG is sensitive to FP8/BF16

---

## 2. The Parallax Mechanism

### Core Idea: Parameterize the Probe
Parallax eliminates the per-query CG solve by **learning** the probe vector: `ρ_i = W_R x_i` where `W_R` is a learnable projection matrix (like a second query). It also removes the "boundary amplification" factor η_i for stability.

The Parallax forward is an **additive correction to Softmax Attention**:

```
o_PLX = o_SA − Σ_KV ρ
```

where `Σ_KV` is the KV covariance matrix (computed from softmax-weighted K,V statistics), and `ρ` is the learned probe. Intuitively: Softmax Attention gives a baseline prediction (intercept), and the covariance-probe term adds a **directional correction** based on how keys and values co-vary around the query.

### A Unified Family of Attention Mechanisms
The paper positions Parallax within a 3×3 family defined by:
- **Rows**: softmax-weighted, uniform+intercept, uniform w/o intercept
- **Columns**: probe = zero (⇒ Softmax/ValueAvg/LinearAttn), probe = learned (⇒ Parallax/AffineLinearAttn), probe = solved (⇒ LLA/AffineMesaNet/MesaNet)

This cleanly shows that Linear Attention and MesaNet are the **intercept-removed** versions of Affine Linear Attention and Affine MesaNet, and that Parallax is the learned-probe counterpart of LLA.

### The "Magnitude Tension" Problem
Because `ρ_i = W_R x_i` is learned rather than optimally solved, its alignment with the true optimal probe `ρ*` and its norm are not guaranteed. The correction branch can become **functionally inert** if the optimizer produces poorly-aligned or norm-suppressed probes — and this turns out to depend critically on optimizer choice.

---

## 3. Hardware-Aware Streaming Algorithm

### Increased Arithmetic Intensity
Parallax adds one extra matrix multiply (`R K^T`) and one extra fused operation (`P₂ = P₁ ⊙ S₂`) per KV tile, reusing the same KV stream. The key insight: in the regime `num_row_blocks × L_kv ≫ L_q`, Parallax **roughly doubles arithmetic intensity** (FLOPs per byte of HBM traffic) over FlashAttention.

### Decode Kernel Optimizations (CuTeDSL, Hopper GPUs)
Three clever optimizations:
1. **WGMMA sharing** — Q and R share one shared-memory tile; both `QK^T` and `RK^T` accumulate in the same Tensor Core instruction, since Hopper's WGMMA minimum tile size (64 rows) is otherwise mostly idle during decode
2. **Persistent split over KV loop** — partitions the KV tile loop across S CTAs for the same (batch, head) pair
3. **In-kernel reduction** — final rescale and combine `(1 + d₂/d₁)O₁/d₁ − O₂/d₁` happens in-kernel, no separate reduction launch

**Result**: Parallax decode matches or outperforms FlashAttention 2/3 across all tested batch sizes (1–2048) and context lengths (128–32K), in both compute-matched (dh=64) and I/O-matched (dh=128) settings.

---

## 4. Experimental Results

### Synthetic Benchmarks (MAD-Benchmark)
Parallax achieves top average accuracy (0.716 vs 0.672 for Attention), with strong gains on recall tasks (ICR: 0.951 vs 0.803, SC: 0.988 vs 0.950). On harder scaled-up versions (vocab 512, context 2048), Parallax retains accuracy while all baselines degrade dramatically, especially on Selective Copying.

### Language Model Pretraining (0.6B & 1.7B)

**Setup**: Qwen-3 architecture, Ultra-FineWeb dataset, context 4096, trained with Muon optimizer + WSD schedule.

**Key results at 0.6B scale** (Muon, with RoPE on ρ):

| Metric | Transformer | Parallax |
|--------|------------|----------|
| LAMBADA ppl | 22.15 | **18.56** |
| WikiText ppl | 23.43 | **22.25** |
| Avg downstream | 54.54 | **55.99** |

**Parameter-matched control** (Transformer†): Parallax still wins (55.99 vs 54.90), confirming gains aren't from extra parameters.

**Compute-matched control** (Parallax†): Parallax† still significantly beats Transformer (55.79 vs 54.54), confirming gains aren't from extra FLOPs.

**At 1.7B scale**: The advantage persists (62.45 vs 61.43 avg downstream), suggesting the gain scales.

### The Optimizer-Architecture Interaction (Critical Finding)

This is perhaps the paper's most surprising result:

- **Under Muon**: Parallax shows substantial and consistent improvement over Transformer
- **Under AdamW**: The advantage **shrinks markedly or disappears entirely** (e.g., AdamW WSD: PLX 52.68 vs Attn 52.61)

**Why?** Through detailed mechanism analysis:
1. **Correction-to-Output Ratio (COR)** reaches >8 in deep layers under Muon, but <4 under AdamW
2. **Probe norm ∥ρ∥** shows the largest optimizer gap — AdamW produces much smaller ρ vectors
3. **Stable rank of W_R** collapses under AdamW (134 → ~10 vs 134 → 134 under Muon), bottlenecking the entire correction pathway
4. **Gating experiment**: A learnable sigmoid gate on ρ converges to ~1 under Muon (correction active) but ~0.26 under AdamW (model learns to **suppress** the correction)

The root cause: AdamW's spectral geometry (steepest descent under ℓ₁→ℓ∞ norm) leads to rank collapse in weight matrices, while Muon's spectral norm geometry (∥·∥_{ℓ₂→ℓ₂}) preserves high stable rank through orthogonalized updates (all singular values = 1). W_R is disproportionately sensitive to this effect.

### Attention Score Distribution Patterns
- **Negative weights**: Parallax scores routinely go negative (range ~±40 in deep layers), allowing active **subtraction** of value components from irrelevant tokens
- **Reduced attention sink**: Parallax substantially reduces the "first token" concentration phenomenon
- **Higher entropy**: Parallax's base softmax is more diffuse — it offloads fine-grained discrimination to the correction branch

---

## 5. Limitations & Future Directions

1. **Scaling**: Needs validation at larger scales, longer contexts, with MoE
2. **Efficiency**: Compatible with sparse attention patterns (sliding window, block sparse) and MLA — kernel work pending
3. **Post-training adaptation**: Since Parallax with W_R=0 ≡ Softmax Attention, pretrained Transformer checkpoints can be **converted** by adding W_R and fine-tuning — unique advantage over Linear Attention family
4. **Theoretical understanding**: Precise characterization of the optimizer-architecture interaction remains open
5. **Broader implications**: Affine variants of DeltaNet, Linear Attention (adding back the intercept) may also benefit from this framework

---

## Key Takeaways

1. **Parallax is a principled upgrade to attention** — it adds a learned covariance correction term to Softmax Attention, rooted in nonparametric statistics (local linear regression), and costs minimal extra compute by reusing the KV stream.

2. **The optimizer matters more than previously recognized for architecture design** — Muon's spectral-norm geometry is essential for Parallax's correction branch to function. This is the first empirical demonstration of strong architecture-optimizer codesign for attention mechanisms.

3. **Parallax is a Pareto improvement** — better perplexity and downstream accuracy without sacrificing throughput (in fact, decode is faster due to higher arithmetic intensity).

4. **It's backward-compatible with Transformers** — W_R = 0 recovers exact Softmax Attention, enabling conversion of pretrained checkpoints, unlike the Linear Attention family which requires retraining.

---

## Muon Optimizer — Quick Reference

Muon is a recently popular alternative to AdamW for matrix parameters in hidden layers. For a weight matrix W with gradient G:

1. Maintain momentum buffer B_t = βB_{t-1} + G_t
2. Compute the polar factor polar(B_t) = U_t V_t^T (via Newton-Schulz iteration, not full SVD)
3. Update: W_{t+1} = W_t − η_t · U_t V_t^T

Key property: all singular values of the update are exactly 1 (condition number = 1). This prevents the spectral collapse that AdamW suffers from, where weight matrices lose effective rank over training. The polar factor is the nearest semi-orthogonal matrix to the momentum buffer in Frobenius norm — it's steepest descent under the spectral norm (∥·∥_{ℓ₂→ℓ₂}).
