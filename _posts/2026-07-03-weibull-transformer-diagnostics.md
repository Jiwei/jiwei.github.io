---
layout: post
title: "A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions"
date: 2026-07-03 10:22:00 +0800
categories: paper-notes
---

**Paper:** [arXiv:2605.18898](https://arxiv.org/abs/2605.18898)
**Author:** Tiexin Ding (Independent Researcher)
**Date:** May 2026 (27 pages, 14 figures)
**Code:** github.com/tiexinding/NPM-Weibull-public

---

## 1. Core Problem: We Lack Granular Tools for Understanding Weight Distributions

Existing tools for analyzing transformer weights operate on the wrong level of abstraction:

| Tool | What it measures | Limitation |
|---|---|---|
| **WeightWatcher / HT-SR** | Eigenvalue spectra | Compresses entire matrix into singular value distribution — averages across component types |
| **AlphaDecay** | Drift in top singular values | Singular-value focus, no element-level granularity |
| **Massive activation analysis** | Activation magnitudes | Measures activations, not weights |

None of these directly characterizes the most fundamental representation: the **element-wise distribution of weight magnitudes |Wᵢⱼ|** — the raw numbers that make up the model. This gap matters because different functional components within the same layer (e.g., FFN gates vs. attention query projections) follow systematically different distributions. Aggregate spectral statistics **average across** these differences, obscuring them.

---

## 2. The Framework: Weibull Distribution as a Diagnostic Lens

The paper applies the **Weibull distribution** — a two-parameter family from extreme-value theory — to element-wise |W| distributions. The Weibull PDF is:

```
f(x; k, λ) = (k/λ)(x/λ)^{k-1} · exp(-(x/λ)^k)
```

### The Two Parameters

| Parameter | What it measures | Behavior |
|---|---|---|
| **k** (shape) | Tailedness of |W| — smaller k = heavier right tail, larger k = narrower body | Dimensionless, architecture-independent, comparable across everything |
| **λ** (scale) | Overall magnitude of weights | Grows substantially during training |

### The Initialization Anchor

This is the framework's key insight. At initialization, weights are i.i.d. Gaussian: `w ~ N(0, σ²)`. The absolute values `|w|` follow a **Half-Normal** distribution. Fitting this to a Weibull distribution via the paper's middle-80% probability-plot protocol yields:

```
k₀ ≈ 1.20   (universal — independent of σ, vendor, architecture)
λ₀ ≈ 0.8875 × σ_init   (recipe-specific)
```

The shape `k₀ ≈ 1.20` is the **universal anchor**. Any departure from it in a trained model is attributable entirely to training dynamics — it's a principled, dimensionless measuring stick. The constants are derived analytically (not empirically) from deterministic integrals of special functions over the trim interval.

### The Middle 80% Trim Protocol

Before fitting, the paper discards the smallest and largest 10% of |W| values. This is **not empirical tinkering** — it's derived from the sampling-noise properties of Weibull probability coordinates:

In Weibull probability coordinates `Y = ln(-ln(1 - F))`, the empirical rank `Y_i` of the i-th order statistic has sampling-noise variance:

```
Var(Y_i) ∝ p / [(1-p) · (ln(1-p))²]     where p = i/N
```

This variance **diverges at both tails** (p → 0 and p → 1) and reaches its minimum at p ≈ 0.80. The bottom 10% carries approximately **6.5× more measurement noise** than the optimal central region. Fitting on full data (no trim) systematically underestimates k by 3-7% — enough to pull 5 of 7 models outside the paper's main finding (the Transmission band).

The trim makes the framework **anti-interference**: robust to outliers, numerical precision artifacts, and heavy-tail contamination from super-weights.

### Γ Closure Consistency Check

All fits are validated through an internal self-consistency check: for any Weibull(k, λ), the theoretical second moment is `E[W²] = λ²Γ(1 + 2/k)`, which must match the empirical sample variance. All 837 FFN fits in the cohort pass this check with relative error below 2%, confirming the fits are genuine rather than numerical artifacts.

---

## 3. Core Finding 1: The Transmission Class — A Universal k Band

FFN modules and the attention output projection `W_o` (the OV circuit) form what the paper calls the **Transmission Class** — components whose role is information transmission without gating. Their shape parameter `k` stays remarkably close to initialization throughout training.

Across **12 model entries** spanning **7 architectural families** (Pythia, OLMo-1/2, LLaMA-3, Mistral, Qwen2.5/3), covering:
- **Model sizes:** 70M to 14B (200× range)
- **Activation patterns:** GeLU (2-matrix) and SwiGLU (3-matrix)
- **Normalization placements:** Pre-LN, Pre-LN + RMSNorm, Peri-LN, QK-Norm

```
Median terminal k ∈ [1.186, 1.204]
Cross-family CV = 0.51% (0.57% on depth ≥ 12 subset)
```

Every fit achieves R² ≥ 0.99. The band holds **regardless of activation, normalization, model size, or initialization scheme**. The scale λ varies by more than an order of magnitude across families, yet k stays locked in a ~0.5% band.

### Body–Tail Ablation

Re-fitting with different trim protocols confirms the band is a body property:

| Protocol | Median k | In band [1.186, 1.204] |
|---|---|---|
| `k_80%` (body only) | 1.195 | 10/12 |
| `k_90%` | 1.182 | 0/12 |
| `k_100%` (full data) | 1.143 | 0/12 |

The body–tail gap `k_80% − k_100% = 0.0519 ± 0.0017` is stable across all 12 entries (CV = 3.3%), quantifying exactly how much the heavy tail drags full-data fits away from the body.

### Why? The OV Circuit Explanation

The QK/OV circuit decomposition (Elhage et al., 2021) provides the functional explanation. The OV circuit and FFN modules **transmit** information without gating — their job is to move information from one representation to another. Optimization pressure favors **uniform weight distributions** that maximize aggregate conductance, preserving the initialization Weibull body. Only the extreme tail develops outliers (super-weights), which the middle-80% trim explicitly excludes — the body remains intact.

---

## 4. Core Finding 2: The Selection Class — Systematic Departure from Weibull

`W_q` and `W_k` — the attention input projections that determine **which tokens to attend to** — systematically depart from the initialization anchor. They constitute the **Selection Class**.

### Five Driving Mechanisms (Appendix A.4)

1. **Functional necessity (D1):** Sparse attention patterns require `W_q` and `W_k` to selectively amplify some embedding directions and suppress others → heavy tails
2. **AdamW sign-descent dynamics (D2):** Adam's gradient-sign updates push individual weight elements away from zero, compounding into heavy-tailed accumulation (Kunstner et al., 2023; causally verified by Kaul et al., 2025)
3. **Softmax saturation feedback (D3):** "No-op" attention heads push logits toward ±∞, backpropagating extreme values into `W_k` (Bondarenko et al., 2023)
4. **Residual-stream coupling (D4):** As `W_o` scales up during training, gradient flow propagates adjustment pressure back to `W_q`/`W_k`
5. **Training budget (D5):** The cumulative signal `T/τ = T · η · λ_wd` determines how long D1–D4 operate

### Architecture-Dependent Severity

The **magnitude** of Selection departure depends on attention storage architecture:

| Architecture | Models | Median terminal k | Severity |
|---|---|---|---|
| Separately-stored MHA | OLMo-1, OLMo-2 | [0.76, 0.99] | **Deep Selection** — far from 1.20 |
| GQA (grouped-query) | LLaMA-3, Mistral, Qwen | [1.10, 1.16] | **Mild Selection** — below band but close |
| Merged W_qkv | Pythia (all sizes) | [1.05, 1.18] | **Transitional** — tracks T/τ |

The MHA/GQA dichotomy has a plausible mechanical explanation: in GQA, a single K head serves 4–7 query heads simultaneously, mechanically limiting how selectively `W_k` can specialize for any single query direction. Pythia's merged `W_qkv` (all three projections in one matrix) further constrains specialization.

### QK-Norm as a Mitigator

Qwen3-8B (with QK-Norm applied to the QK product before softmax) shows **visibly tighter** Q/K distributions than Qwen2.5-14B (no QK-Norm), with the specialization tail compressed and confined to fewer blocks. This is consistent with QK-Norm dampening the extreme-logit feedback that drives `W_k` toward heavier tails — though the paper is careful to note this is observational, not causally isolated (a controlled ablation with otherwise-identical training would be required).

### Spatial Localization

Selection drift is **not uniform across depth**. The heavy-tail signature concentrates in **mid-to-deep layers** (roughly layers 8–20 of 32 in OLMo-1-7B), consistent with where induction heads and specialized attention patterns form. Early layers and the deepest layers show much less specialization.

### Temporal Evolution: The Dimensionless Training Budget

Within the Pythia family (5 sizes, identical recipe), Selection drift severity tracks `T/τ` monotonically:

```
T/τ = T · η · λ_wd
```

| Pythia size | T/τ | Physical State | Selection behavior |
|---|---|---|---|
| 70M | 1.43 | Saturated | Strongest drift |
| 160M | 0.86 | Near-saturated | Moderate drift |
| 410M / 1B | 0.43 | Approaching | Mild drift |
| 6.9B | 0.17 | Transition | Minimal drift (still developing) |

The 6.9B uses a lower `η_peak` (1.2 × 10⁻⁴ vs. 3.0 × 10⁻⁴–1.0 × 10⁻³ for others), yielding lower T/τ despite the same 143k training steps. This confirms that Selection drift is driven by **cumulative training signal**, not model size per se.

---

## 5. Core Finding 3: λ Scales with Training Signal

The scale parameter λ grows substantially during training. Within the Pythia family, terminal λ across Transmission Class components scales with:

```
λ ∝ √(η_peak / λ_wd)    (Pearson r = 0.94, n = 5, linear fit through origin)
```

This shows **directional consistency** with the AdamW steady-state scaling analysis of Fan et al. (2025) within their validated regime (d ≤ 2048). Per-size deviations of 7–36% indicate directional match rather than quantitative law.

### Per-Component Paired Growth

`λ_O` (attention output) and `λ_FFN_out` (FFN output) track each other with **Pearson r = 0.9967** across 25 size–step combinations. Both grow in lockstep because they write into the same residual stream. The λ trajectory is non-monotonic: it rises through learning-rate warmup, peaks near warmup completion (~step 10k in Pythia), then retreats under cosine LR decay. The degree of post-peak retreat depends on the Physical State (T/τ).

### k and λ Carry Independent Information

- **k** labels the **functional class** (Transmission ≈ 1.20 vs. Selection varies by architecture)
- **λ** labels the **training progress** (how much optimization signal has been applied)

The optimizer scales the magnitude (λ) while preserving the distributional shape (k) within the Transmission Class. Selection pressure (when present, as in `W_q`/`W_k`) modifies the tail shape without destroying the Weibull body.

### Cross-Family λ Scaling Is Observational Only

Across non-Pythia 7B–14B entries, the ratio `λ / √(η/λ_wd)` spans ~6.6× (Mistral-7B at 0.048 to Qwen3-8B at 0.319), far larger than the ~1.9× Pythia-internal range. This scatter is attributed to confounding factors: different initialization recipes set different σ_init baselines, and different normalization placements alter residual-stream dynamics. The cross-family λ relationship is reported as a qualitative trend, not a quantitative law.

---

## 6. Super-Weights: A Universal Phenomenon

Every architectural family in the cohort contains isolated **dragon-king outliers** — individual weight elements that detach from the Weibull body. The body of the distribution remains Weibull-conforming throughout; only one or a handful of extreme elements per matrix detach.

| Family | Per-block max/q99 (median) | max/q99 (extreme) | Kurtosis extreme |
|---|---|---|---|
| Pythia-70m | 7.2× | 15.7× | 196.9 |
| Pythia-160m | 6.6× | 13.4× | 104.2 |
| Pythia-410m | 8.0× | 19.6× | 257.1 |
| Pythia-1B | 11.3× | 21.5× | 21.8 |
| Pythia-6.9B | 8.6× | 31.4× | 27.7 |
| OLMo-1-7B | 27.7× | 107.2× | 445.9 |
| Qwen2.5-14B | 18.4× | 22.9× | 46.9 |
| Qwen3-8B | 17.3× | 40.4× | 14.5 |

These super-weights are the element-wise counterparts of massive activations (Sun et al., 2024) and are universally present across architectures. The middle-80% protocol is robust to them by design — they affect only the trimmed tails.

---

## 7. The Eight Diagnostic Functions (npm-weibull-py v0.4)

| Function | Purpose |
|---|---|
| F1_extract_weights | Extract all weight matrices from a specified layer |
| F2_fit_weibull | Fit Weibull(k, λ) via least-squares on the Weibull probability plot (middle-80% trim) |
| F3_gamma_closure | Verify Γ closure consistency (E[W²] = λ²Γ(1+2/k)) |
| F4_cross_family_band | Compute per-entry median k, aggregate to cross-family CV and Transmission band |
| F5_lambda_scaling | Fit λ ~ √(η/λ_wd) within the Pythia family |
| F6_k_drift | Compute k drift magnitude: Δk = k_terminal − k_init |
| F7_attention_arch_classify | Classify attention architecture: MHA / GQA / MQA |
| F8_lambda_paired_correlation | Pearson correlation of λ_O vs. λ_FFN_out |

The companion **DATABASE_v9_1** contains per-component Weibull fits for all 12 model entries: Pythia (70M/160M/410M/1B/6.9B), OLMo-1-7B, OLMo-2-7B, LLaMA-3-8B, Mistral-7B, Qwen2.5-7B/14B, Qwen3-8B.

---

## 8. Relationship to Existing Tools

| Tool | Measurement space | Relationship to this framework |
|---|---|---|
| **WeightWatcher / HT-SR** | Eigenvalue spectra (quadratic form) | Orthogonal — eigenvalue spectrum and element-wise |W| measure different structures; carry independent information |
| **AlphaDecay** | Top singular value drift | Consistent in finding systematic structural change during training; Weibull adds per-component granularity |
| **OrthoAdam** (Kaul et al., 2025) | Kurtosis of trained weights | Kurtosis and Weibull k are both tail-sensitive but non-equivalent; framework serves as independent verification channel |

---

## 9. A Measurement-First Paper

This paper's stance is notable for what it **doesn't** do:

- **No new architecture or training method.** It's purely diagnostic.
- **No overclaimed causal claims.** The MHA/GQA dichotomy, QK-Norm mitigation, and five Selection mechanisms are presented as observations — controlled ablations are explicitly left to future work.
- **Negative results reported.** An attempted similarity-based pattern search for F/S layers (Appendix C of IndexCache is from a different paper — here, the paper is open about cross-family λ scaling being observational only).
- **Careful distinction between body and tail.** The middle-80% trim is theoretically justified, not empirically hacked. The body–tail gap is quantified and stable.
- **Limitations explicitly acknowledged.** Qwen2.5-14B coverage is partial (27/48 layers), downstream performance correlation is not established, per-family λ differences are confounded.

---

## 10. Limitations (Acknowledged by Author)

1. **Selection mechanism:** Five candidate forces identified, but no controlled ablation isolating each contribution
2. **Cross-family λ scaling:** Observational only due to confounding factors (initialization recipes, normalization placements)
3. **Downstream performance:** No correlation established between (k, λ) signature and task performance — potential for early-stopping proxies left to future work
4. **Qwen2.5-14B data coverage:** Only first 27 of 48 layers extracted due to GPU memory constraints
5. **MHA/GQA dichotomy:** Observational correlation; causal derivation requires controlled ablation

---

## 11. Key Takeaways

1. **A universal measuring stick exists.** `k₀ ≈ 1.20` at initialization, derived analytically, gives a dimensionless reference point for quantifying training-induced distributional change.

2. **Transformers have two functional classes baked into their weight distributions.** Transmission (FFN + `W_o`) preserves the initialization shape across architectures; Selection (`W_q`/`W_k`) departs from it, with severity modulated by attention architecture.

3. **Per-component, per-layer diagnostics reveal structure invisible to aggregate statistics.** The Transmission band (CV=0.51%), the depth-localized Selection signatures, and the architecture-dependent drift severity all become visible only when each weight matrix is examined independently.

4. **The two Weibull parameters are informationally independent.** k = functional class, λ = training progress. This clean separation makes the framework useful as a monitoring tool during training.
