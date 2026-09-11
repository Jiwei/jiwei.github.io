---
layout: post
title: "Parallax: Parameterized Local Linear Attention for Language Modeling — Fresh Summary"
date: 2026-07-03 10:44:29 +0800
categories: paper-notes
---

**Paper:** [arXiv:2605.29157](https://arxiv.org/abs/2605.29157)
**Authors:** Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, Zhaoran Wang (Northwestern University, Tilde Research, University of Washington)
**Date:** May 2026
**Code:** github.com/yifei-zuo/Parallax

---

## 1. Core Problem: Softmax Attention Has Stagnated, But Efficient Alternatives Underperform

Softmax Attention has remained largely unchanged since the original Transformer (2017), despite a flood of proposed alternatives. Linear Attention and State Space Models (Mamba, DeltaNet) offer subquadratic complexity but **consistently underperform on in-context recall**. The underlying reason is theoretical: from a nonparametric statistics perspective (the test-time regression framework), Softmax Attention is a **local constant estimator** (Nadaraya-Watson), and constant functions suffer from **boundary bias** — poor predictions near the edges of the key distribution.

**Local Linear Attention (LLA)** (Zuo et al., 2026) solves this by upgrading the estimator to a **local linear** fit, which provably reduces integrated MSE. The bias-variance hierarchy is:

```
R(f̂_global_linear) ≫ R(f̂_NW_softmax) ≫ R(f̂_local_linear)
```

But LLA has never been scaled to LLM pretraining because it requires solving a **per-query linear system** `ρ* = Σ⁻¹μ` via conjugate gradient — which is I/O-intensive, numerically unstable at low precision, and requires careful regularization tuning.

Parallax asks: can we get the statistical advantages of LLA **without the per-query solver**?

---

## 2. The Solution: Parallax = Softmax Attention + Learned Covariance Correction

### 2.1 Key Reformulation of LLA

Parallax starts by rewriting LLA as a correction to the standard Softmax Attention output:

```
o_LLA = o_SA − (1 + η) · Σ_KV · ρ*
```

where:
- `o_SA` is the standard softmax attention output
- `Σ_KV` is the **softmax-weighted KV covariance** — how values and keys co-vary under the attention distribution
- `ρ* = Σ⁻¹µ` is the optimal probe (the local linear slope), solved per-query via CG
- `η` is a non-negative **boundary amplification** factor measuring the Mahalanobis distance from the query to the key center

### 2.2 The Parametrization Trick

Parallax eliminates the per-query CG solve by **learning to predict ρ directly from the layer input**:

```
ρ_i = W_R · x_i      (one learned projection, no solver)
```

and drops the boundary amplification (`η = 0`) since the parameterized ρ no longer has the geometric properties of the exact solve that made η well-defined. The Parallax forward is:

```
o_PLX = o_SA − Σ_KV · ρ      where ρ = W_R · x
```

In other words: **standard softmax attention output, minus a covariance-weighted correction learned from the input**. The correction branch subtracts value components aligned with irrelevant keys — allowing the mechanism to actively **reject** tokens rather than merely de-emphasize them with near-zero softmax weights. Parallax weights can be **negative** and unbounded (unlike softmax weights which are always positive and sum to 1).

### 2.3 Connection to the Attention Family

Under different limits, Parallax connects to other mechanisms:

| Limit | Parallax becomes |
|---|---|
| `λ → ∞` (strong regularization, `∥ρ∥ → 0`) | Standard Softmax Attention |
| `h → ∞` (wide bandwidth) | Affine Linear Attention (`v̄ − S̃·ρ`) |
| Full solve (`ρ = Σ⁻¹µ`) | LLA (the original, with CG) |

The bandwidth `h` controls the softmax temperature; the probe construction (zero, learned, or solved) determines which family member you get.

### 2.4 The Common Affine Template

All mechanisms in this family share the same **affine output structure**:

```
o = intercept + correction
o = v̄ (value average) + Σ_KV · ρ (covariance-probe product)
```

| Mechanism | Intercept | Probe ρ |
|---|---|---|
| Softmax Attention | v̄ (softmax-weighted) | 0 |
| Parallax | v̄ (softmax-weighted) | W_R·x (learned) |
| LLA | v̄ (softmax-weighted) | Σ⁻¹µ (solved per query) |
| Affine Linear Attn | v̄ (uniform average) | W_R·x (learned) |
| Affine MesaNet | v̄ (uniform average) | H̃⁻¹q̃ (solved per query) |

**Key insight:** What conventional Linear Attention calls the "query" is actually the **probe** ρ — a directional readout from the recurrent state that can be completely determined by other statistics (as in MesaNet or LLA).

### 2.5 Magnitude Tension in the Affine Structure

Since `ρ = W_R·x` is parametric rather than an optimal solve, its alignment and norm relative to the exact solve `ρ*` are not guaranteed. Only the component of ρ aligned with `ρ*` contributes functionally; the orthogonal component is unidentifiable. A poorly aligned or norm-suppressed probe makes the correction branch functionally inert, and Parallax collapses toward Softmax Attention regardless of the affine structure nominally available. Both alignment and norm **depend heavily on optimizer choice**.

---

## 3. Hardware-Aware Streaming Algorithm

### 3.1 Doubling Arithmetic Intensity

Parallax's compute pattern is `o = O₁/d₁ · (1 + d₂/d₁) − O₂/d₁`, which requires two parallel scoring branches (QK and RK) but reuses the **same KV stream**. The arithmetic intensity roughly doubles over FlashAttention:

```
AI_FA  ≈ 2L_q·L_kv / (L_q + 2·n_r·L_kv)
AI_PLX ≈ 2L_q·L_kv / (L_q + n_r·L_kv)   ← roughly 2× in compute-bound regime
```

### 3.2 Decode Kernel Prototype (CuTeDSL on H200)

1. **WGMMA sharing:** Q and R tiles stacked in shared memory; both S₁ (QK) and S₂ (RK) computed in the same tensor core WGMMA accumulator. P₁V and P₂V also computed jointly. Cost: one extra row of register accumulators per CTA, zero extra HBM traffic.

2. **Persistent split over the KV loop:** For decode (few query rows), CTAs split the KV tile loop, with cross-split reduction performed in-kernel.

3. **In-kernel reduction:** The merger CTA atomically collects partials, runs log-sum-exp rescaling in fp32, and writes the final output — all in one kernel launch.

**Result:** The prototype matches or outperforms FlashAttention 2/3 across all tested batch sizes (1–2048) and context lengths (128–32,768), in both I/O-matched (`dh=128`) and compute-matched (`dh=64`) settings.

---

## 4. Pretraining Results

### 4.1 Setup

- **Backbone:** Qwen-3 architecture, RMSNorm on Q/K/ρ, RoPE
- **Data:** Ultra-FineWeb, context length 4096
- **Scales:** 0.6B (28 layers) and 1.7B
- **Controls:**
  - **Parameter-matched Transformer†:** extra Q heads to match Parallax parameter count
  - **Compute-matched Parallax†:** halved head dimension, same attention FLOPs, extended FFN to match total params

### 4.2 0.6B Scale (Muon optimizer, WSD schedule)

| Model | LAMBADA ppl↓ | Avg downstream acc↑ |
|---|---|---|
| Transformer | 22.15 | 54.54 |
| Transformer† (param-matched) | 22.35 | 54.90 |
| Kimi DeltaAttn | 25.16 | 52.73 |
| Gated DeltaNet | 24.63 | 53.67 |
| **Parallax** (RoPE on ρ) | **18.56** | **55.99** |
| Parallax† (compute-matched) | 20.29 | 55.79 |
| Parallax (no RoPE on ρ) | 19.77 | 55.54 |

**Key finding:** Parallax† (compute-matched) significantly outperforms both Transformer and Transformer†, confirming the gain is from the **mechanism itself**, not from extra compute or parameters. The advantage persists across 8 downstream benchmarks (BoolQ, HellaSwag, PIQA, ARC-easy, ARC-challenge, WinoGrande, OpenBookQA, SciQ).

### 4.3 1.7B Scale (Muon)

| Model | LAMBADA ppl↓ | Avg downstream acc↑ |
|---|---|---|
| Transformer | 13.07 | 61.43 |
| **Parallax** (RoPE on ρ) | **10.80** | **62.45** |

The gain holds at 1.7B, confirming it persists with scale.

### 4.4 Synthetic Benchmarks (MAD-Benchmark)

Parallax attains the highest overall accuracy, with particular strength on recall-oriented tasks (ICR, FCR, NCR, Selective Copying). On harder challenge variants with vocabulary sizes up to 512 and context lengths up to 2048, Parallax retains accuracy while all other baselines degrade dramatically — most visibly on Selective Copying at long contexts.

---

## 5. The Optimizer-Architecture Interaction (Central Finding)

Under **Muon**, Parallax shows a large and consistent advantage over Softmax Attention. Under **AdamW**, the advantage **shrinks markedly or disappears entirely**. This is not incidental — it reflects a fundamental interaction between the optimizer geometry and the affine attention structure.

### 5.1 Correction-to-Output Ratio (COR)

```
COR = ∥Σ_KV·ρ∥ / ∥o_SA∥
```

| Optimizer | COR in deep layers | Behavior |
|---|---|---|
| Muon | > 8 | Correction branch is highly active |
| AdamW (Cosine) | < 4 | Barely exceeds random init level |
| AdamW (WSD) | < 4 | Similar suppression |

### 5.2 Decomposition: Why AdamW Struggles

| Diagnostic | Muon | AdamW |
|---|---|---|
| **Probe norm ∥ρ∥** | Large (grows with depth) | Small (stays suppressed) |
| **KV correlation ∥Corr∥** | Higher (richer KV associations) | Lower |
| **Covariance-Probe Alignment (CPA)** | Higher (ρ better aligned with leading covariance directions) | Lower |

### 5.3 Gating Experiment: AdamW Actively Suppresses the Correction

A learnable sigmoid gate `g = σ(w_g·x)` modulates the probe as `ρ = g · W_R·x`. Under Muon, the model **learns to open the gate** and converges to the same loss as the ungated baseline. Under AdamW, the gate **stabilizes around 0.26** — the model actively chooses to suppress the correction and achieves final performance comparable to plain Softmax Attention. This is not a scaling convention; AdamW intrinsically fails to utilize the correction branch.

### 5.4 The Stable Rank Story (Root Cause)

| Projection / Circuit | Muon Parallax | AdamW-Cos Parallax | Ratio |
|---|---|---|---|
| `W_Q` | 97.4 | 20.9 | 4.7× |
| `W_K` | 106.4 | 18.0 | 5.9× |
| **W_R** | **134.0** | **9.3** | **14.4×** |
| `W_QK` circuit | 25.5 | 4.9 | 5.2× |
| **W_RK circuit** | **29.1** | **9.4** | **3.1×** |
| `W_OV` circuit (arch. effect) | 34.1 | 30.4 | 1.1× |

Under AdamW, projection matrices suffer **spectral collapse** — stable rank drops dramatically. `W_R` is disproportionately affected, with the largest optimizer sensitivity (14.4× gap). The `W_RK` bilinear circuit has ~3× higher stable rank under Muon, enabling ρ to align effectively with leading covariance directions.

Muon's gradient orthogonalization (polar factor, all singular values = 1) preserves matrix conditioning throughout training. AdamW's sign-descent geometry causes weight matrices to collapse toward low-rank, which disproportionately harms the correction branch — a low-rank probe cannot meaningfully interact with a high-dimensional covariance structure.

**This is the first empirical demonstration of strong architecture-optimizer codesign for attention mechanisms.**

### 5.5 Architectural Effect (Independent of Optimizer)

Beyond the optimizer effect, there is a consistent **architectural effect**: `W_V`, `W_O`, and `W_OV` circuits have higher stable rank under Parallax than under Softmax Attention for **all optimizers**. The affine structure enriches the value pathway, giving the output projection a richer set of directions to read from.

---

## 6. Score Distribution Patterns

Parallax produces qualitatively different attention patterns than Softmax Attention:

1. **Negative weights:** Parallax weights routinely take negative values (−20 to −40 in deep layers under Muon), allowing the model to actively **subtract** value components from irrelevant tokens rather than merely de-emphasize them.

2. **Reduced attention sink:** Parallax substantially reduces the concentration of probability mass on the first token (the "attention sink" phenomenon), in both the base softmax and the combined weights. The correction branch absorbs the routing role that softmax typically discharges onto the sink token.

3. **Higher entropy:** Parallax's base softmax distribution is more diffuse than the Transformer baseline. The model uses softmax for broad contextual aggregation and offloads fine-grained token discrimination to the correction branch.

---

## 7. Practical Benefits

### 7.1 Post-Training Adaptation from Pretrained Transformers

When `W_R = 0`, Parallax behaves identically to Softmax Attention. A pretrained Transformer checkpoint can be converted to Parallax by adding the zero-initialized `W_R` and fine-tuning — no need to retrain from scratch. This contrasts sharply with Linear Attention, where no parameter setting recovers Softmax Attention exactly.

### 7.2 Weight Decay Annealing (WDA)

The Parallax advantage shrinks during the final linear decay phase of the WSD schedule because weight norms shrink. Weight decay annealing gradually reduces `λ_wd` during decay:

```
λ(t) = λ · (1 − t)^γ
```

WDA with `γ=2` (quadratic annealing) gives the largest gain, confirming that weight norm shrinkage is a real, mechanistic contributor to advantage erosion. However, WDA only partially mitigates the issue.

### 7.3 Contextual Sparsity Compatibility

Parallax inherits the streaming structure of Softmax Attention. Any contextual sparsity pattern (sliding window, dilated, block sparse) extends directly to Parallax. It is also structurally compatible with MLA and other optimization techniques.

---

## 8. Limitations and Future Directions

1. **Scale validation:** Largest model is 1.7B; validation at frontier scale with MoE and longer context is future work
2. **Theoretical understanding:** The precise characterization behind the Muon/AdamW gap remains an open question
3. **Affine variants:** Reintroducing the dropped intercept in Linear Attention, DeltaNet, and MesaNet could similarly benefit from Muon
4. **Training recipe:** Muon with WSD is not optimal for Parallax in its current form; better recipes likely exist
5. **Nonparametric DeltaNet:** Deriving the nonparametric counterpart of DeltaNet would complete the family diagram (Figure 1)

---

## 9. Key Takeaways

1. **A practical, scalable upgrade to Softmax Attention.** Parallax preserves the streaming structure of FlashAttention, doubles arithmetic intensity without extra HBM traffic, and improves perplexity/downstream performance at matched compute and parameters — all while allowing zero-init adaptation from pretrained Transformer checkpoints.

2. **Architecture-optimizer codesign is real and important.** The additive affine structure (intercept + correction) depends critically on optimizer geometry. Muon's spectral conditioning unlocks the correction branch; AdamW's spectral collapse suppresses it. This has implications beyond Parallax — the choice of optimizer should be part of architecture design, not an afterthought.

3. **Softmax attention is a local constant estimator, and upgrading to local linear works at scale.** The nonparametric statistics perspective (test-time regression) provides a principled framework for attention design, and Parallax demonstrates that its theoretical advantages translate to practical pretraining gains.

4. **The correction branch fundamentally changes attention behavior.** Negative weights, reduced attention sink, and higher entropy distributions suggest Parallax delegates different functions to the softmax base (broad aggregation) and correction branch (fine-grained discrimination), rather than forcing softmax to do both.

---

## Quick Reference: Muon Optimizer

Muon is a recently popular alternative to AdamW for matrix parameters in hidden layers. For a weight matrix W with gradient G:

1. Maintain momentum buffer `B_t = βB_{t-1} + G_t`
2. Compute the polar factor `polar(B_t) = U_t V_t^T` (via Newton-Schulz iteration, not full SVD)
3. Update: `W_{t+1} = W_t − η_t · U_t V_t^T`

Key property: all singular values of the update are exactly 1 (condition number = 1). This prevents the spectral collapse that AdamW suffers from, where weight matrices lose effective rank over training. The polar factor is the nearest semi-orthogonal matrix to the momentum buffer in Frobenius norm — it's steepest descent under the spectral norm.
