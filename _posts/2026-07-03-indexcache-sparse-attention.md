---
layout: post
title: "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"
date: 2026-07-03 16:23:08 +0800
categories: paper-notes
---

**Paper:** [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)
**Authors:** Yushi Bai, Qian Dong, Ting Jiang, Xin Lv, Zhengxiao Du, Aohan Zeng, Jie Tang, Juanzi Li (Tsinghua University, Z.ai)
**Date:** March 2026

---

## 1. Core Problem: The Indexer Is Still O(L²)

DeepSeek Sparse Attention (DSA) is a production-grade sparse attention mechanism used in frontier models like DeepSeek-V3.2 and GLM-5. It works in two stages per layer:

1. A **lightning indexer** scores all L preceding tokens against each query and selects the **top-k** (k=2048) most relevant
2. **Core attention** is computed only over those k tokens — reducing the main attention from O(L²) to O(Lk)

The indexer is designed to be cheap: few heads, low-rank projections, FP8 arithmetic. **But it still scores all L tokens at every layer**, so its total cost across N layers is **O(NL²)**. At long context (200K tokens), profiling shows the indexer consumes **68-81% of total attention time** during prefill and ~40% during decode. The indexer, not core attention, becomes the bottleneck.

### The Key Observation

Adjacent layers' top-k index sets are **70-100% identical**. The pairwise overlap heatmap reveals distinct clusters of layers with mutually high overlap — the model organizes into functional blocks where token selection is internally consistent. This means most indexer computations are redundant: layer ℓ+1 is largely recalculating what layer ℓ already figured out.

This is a known phenomenon in full-attention models (Kascade, HySparse), but those methods require **full attention as the oracle** to identify important tokens. DSA **removes full attention entirely** — there is no oracle to fall back on. The question becomes: can the indexer's own output serve as the oracle for cross-layer sharing?

---

## 2. IndexCache: The Solution

IndexCache partitions the N layers into:

- **F (Full) layers:** retain their indexer, compute fresh top-k indices, **cache them**
- **S (Shared) layers:** skip the indexer entirely, **reuse the cached indices** from the nearest preceding F layer

The only code change at inference is a single conditional branch:

```
if layer is F:
    indices = indexer(x); cache = indices
else:  # layer is S
    indices = cache          # literally just reuse
x = sparse_attention(x, indices)
```

No KV cache sharing, no head remapping, no additional memory — the cache holds only the current index tensor and is overwritten at each F layer.

---

## 3. Approach 1: Training-Free IndexCache (Greedy Search)

**No weight updates needed.** Works on any off-the-shelf DSA model.

### 3.1 Why Uniform Interleaving Fails

The simplest strategy — keep every r-th indexer (e.g., `FSSS FSSS...` for r=4) — ignores that **indexer importance varies dramatically across layers**. Early and transitional layers are far more sensitive to indexer removal. Uniform interleaving may delete a critical indexer while keeping a redundant one.

### 3.2 Greedy Layer Selection Algorithm

Starting from all-F (all indexers retained), the algorithm incrementally flips layers to S, each time picking the flip that **minimizes LM loss** on a small calibration set:

```
c = all F
for step in 1..K:
    for each currently-F layer ℓ:
        tentatively flip ℓ to S, evaluate LM loss
    commit the flip with lowest loss
```

The search cost is O(N²) forward passes, but pipeline parallelism reduces this by splitting layers into P blocks searched concurrently.

### 3.3 Properties of the Greedy Solution

1. **Searched patterns substantially outperform uniform interleaving** at the same retention ratio. At 1/4 retention with uniform: Long Avg drops from 50.2 to 43.0. With search: recovers to **49.9** — nearly indistinguishable from the full-indexer baseline.

2. **The per-step loss curve reveals a natural ordering.** The first ~20 flips cause negligible loss increase; the last ~10 flips cause sharp degradation. This suggests a clear "expendable vs. critical" boundary among indexers.

3. **Results are stable across calibration sets** — the importance ranking is an intrinsic model property. LM loss serves as a valid proxy for downstream task performance.

---

## 4. Approach 2: Training-Aware IndexCache (Multi-Layer Distillation)

When you can retrain the model, you can explicitly train each retained indexer to serve **multiple layers** at once.

### 4.1 Multi-Layer Distillation Loss

In standard DSA training, each indexer at layer ℓ is distilled via KL divergence against **its own layer's** full attention distribution. IndexCache generalizes this: if layer ℓ serves m subsequent S layers, the loss becomes:

```
L_multi = Σ_t Σ_{j=0}^{m} D_KL(p_t^{(ℓ+j)} ∥ q_t^{(ℓ)})
```

where `q_t^{(ℓ)}` is the indexer's output distribution, and `p_t^{(ℓ+j)}` is the aggregated attention at each served layer.

### 4.2 Key Theoretical Result (Proposition 1)

The multi-layer KL loss has exactly the same gradient as distilling against the **averaged** attention distribution:

```
∇L_multi = ∇L_avg    where    L_avg = Σ_t D_KL(p̄_t ∥ q_t)
```

with `p̄_t` being the centroid of all served layers' attention distributions. This means the indexer learns a **consensus top-k** that jointly covers important tokens across all layers it serves — not overfitting to its own layer.

### 4.3 What Training-Aware Enables

With multi-layer distillation, **even simple uniform interleaving** matches the full-indexer baseline. The pattern sensitivity observed in training-free IndexCache **vanishes** because:

- **S layers adapt** their core attention to work with inherited indices
- **F layer indexers adapt** to produce selections that generalize across their served block

At 1/4 retention with uniform interleaving: Long Avg **50.6 vs. baseline 51.0** — within 0.4 points without any greedy search.

---

## 5. Key Results

### 5.1 End-to-End Speedup (30B DSA model, H100)

| Context Length | Metric | DSA | IndexCache (1/4) | Speedup |
|---|---|---|---|---|
| 10K | Prefill time | 0.57s | 0.45s | **1.27×** |
| 200K | Prefill time | 19.5s | 10.7s | **1.82×** |
| 200K | Decode (per-req) | 58 tok/s | 86 tok/s | **1.48×** |
| 200K | Decode (full KV) | 197 tok/s | 297 tok/s | **1.51×** |

The speedup **grows with context length** because the indexer's O(L²) share of total compute increases — IndexCache removes a larger and larger fraction of the bottleneck.

### 5.2 Training-Free Quality

| Retention | Pattern | Long Avg | G&R Avg |
|---|---|---|---|
| 1/1 (baseline) | All F | 50.2 | 74.6 |
| 1/2 | Uniform | 47.4 | 74.3 |
| 1/2 | **Searched** | **50.3** | **74.4** |
| 1/4 | Uniform | 43.0 | 73.8 |
| 1/4 | **Searched** | **49.9** | **74.9** |
| 1/8 | Uniform | 35.3 | 70.0 |
| 1/8 | Searched | 46.1 | 73.7 |

At 1/4 retention (75% of indexers removed), searched patterns match baseline on both long-context and general reasoning. At 1/8, degradation becomes non-negligible but searched still substantially outperforms uniform.

General & Reasoning benchmarks (AIME 2025, GPQA, LiveCodeBench, IFBench) are essentially unaffected — IndexCache preserves reasoning ability.

### 5.3 Training-Aware Quality

| Retention | Pattern | Long Avg | G&R Avg |
|---|---|---|---|
| 1/1 (baseline) | All F | 51.0 | 74.2 |
| 1/2 | **Uniform** | **51.6** | **74.5** |
| 1/2 | Searched | 50.6 | 73.6 |
| 1/4 | **Uniform** | **50.6** | **74.1** |

Uniform interleaving with training matches or exceeds baseline — pattern sensitivity is gone. Removing the cross-layer loss drops Long Avg from 51.6 to 49.8 (at 1/2), confirming the multi-layer distillation objective is practically necessary.

### 5.4 Production-Scale: GLM-5 (744B)

Training-free IndexCache at 1/2 retention on the 744B GLM-5: searched pattern achieves Long Avg **78.7 vs. 78.4 baseline**. At 1/4: **78.0 vs. 78.4**. End-to-end speedup of ~1.2× on the full Artificial Analysis benchmark suite.

---

## 6. Interesting Negative Result: Similarity-Based Search Fails

Before the greedy loss-based search, the authors tried a seemingly natural approach: construct a similarity matrix where `S[i,j]` = cosine similarity between layer i's attention output when using its own indexer vs. reusing layer j's index. Then use **dynamic programming** to find the pattern that maximizes total similarity.

**Result:** similarity-optimal patterns perform no better than uniform interleaving. Both substantially underperform the loss-based search.

**Why:** Per-layer output similarity is a **local metric** — it measures how well a single layer's attention output is preserved in isolation, without accounting for how small perturbations **cascade** through downstream layers. Two layers may have nearly identical attention outputs (S ≈ 1) yet differ in ways that matter: the reused index may miss a small number of critical tokens whose importance only becomes apparent in later layers' reasoning steps. The greedy loss-based search avoids this by directly optimizing a **global metric** (LM loss) that captures the end-to-end effect.

---

## 7. Significance

IndexCache is a clean, high-impact systems paper. Its contributions:

1. **Identifies a new bottleneck.** In production sparse attention (DSA), the indexer — not core attention — dominates at long context. This was not obvious before the paper's profiling analysis (the indexer is "lightweight" per-FLOP, but O(NL²) adds up).

2. **Extends cross-layer sharing to the sparse regime.** Prior work required full attention as oracle. IndexCache shows the principle works when the oracle is itself the lightweight indexer — and in fact is simpler because you only cache indices, not KV tensors.

3. **Provides both a plug-and-play and a train-time solution.** Training-free IndexCache can be applied to **any existing DSA model** with zero weight changes. Training-aware IndexCache shows how to make the sharing even more robust.

4. **Demonstrates scale.** Results on 30B and 744B models confirm the approach is production-viable. The paper notes that both DeepSeek-V3.2 and GLM-5 already use DSA by default — IndexCache is a drop-in acceleration.

The core insight: adjacent transformer layers largely agree on which tokens matter, and once one layer has done the work of identifying them, subsequent layers can simply reuse the answer.
