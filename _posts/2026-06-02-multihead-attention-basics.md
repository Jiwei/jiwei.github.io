---
layout: post
title: "Multi-Head Attention in the Vanilla Transformer"
date: 2026-06-02 15:19:41 +0800
categories: paper-notes
---

**Reference:** Vaswani et al. (2017), "Attention Is All You Need"

---

## Multi-Head Attention — How Heads Are Combined

### Parallel Projections, Then Concatenate

Instead of computing one attention with full-dimensional Q, K, V, the input is **split across `h` heads**, each operating in a lower-dimensional subspace:

```
For each head i (1 to h):
    Q_i = X · W_Q^i     (d_model → d_k)
    K_i = X · W_K^i     (d_model → d_k)
    V_i = X · W_V^i     (d_model → d_v)

    head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / √d_k) · V_i
```

Each head produces an output of dimension `d_v`. The original paper uses `d_k = d_v = d_model / h` (e.g., 512 / 8 = 64).

### Concatenation + Output Projection

The heads are combined by **concatenation followed by a linear projection**:

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · W_O
```

Where:
- `Concat(...)` stacks the `h` head outputs into a vector of dimension `h × d_v` (= `d_model`)
- `W_O` is a learned projection matrix of shape `(h × d_v) × d_model` (= `d_model × d_model`)

### Why This Design?

1. **Each head learns a different attention pattern.** With separate `W_Q`, `W_K`, `W_V` per head, heads specialize — one might attend to syntactic dependencies, another to long-range semantic links, another to positional neighbors. The paper's ablation showed heads do in fact diversify.
2. **Concatenation preserves per-head information.** Unlike averaging, which would collapse distinct head outputs into one blurry representation, concatenation keeps each head's signal intact. `W_O` then learns how to **mix and weight** these separate signals into the final output.
3. **Computational efficiency.** The per-head dimension is `d_model / h`, so the total FLOPs of all heads combined is roughly the same as a single full-dimensional attention — but with the representational benefit of multiple distinct attention patterns.

### The Concatenation is Positional, Not Semantic

There's no learned decision about "which head goes where." The concatenation is purely **structural** — head 1 occupies dimensions 0 through `d_v-1`, head 2 occupies `d_v` through `2d_v-1`, and so on:

```
Concat(head_1, head_2, ..., head_h) = [h1_dim0, h1_dim1, ..., h1_dim63, h2_dim0, ..., h2_dim63, ..., h8_dim63]
                                       |── head 1 (d_v=64) ──|── head 2 (d_v=64) ──|     |── head 8 ──|
```

This is just stacking. There's no routing or gating that decides permuting the order.

### W_O Does the Cross-Head Mixing

This is the critical detail that answers "how do heads interact." `W_O` is a **full dense matrix** of shape `d_model × d_model` — it is **not block-diagonal**. Every output dimension of the final result can attend to every dimension from every head:

```
output[j] = Σ W_O[j, k] · concat[k]
```

This means `W_O` learningly mixes information from all heads. A single output neuron can pull from head 3's dimension 12, head 7's dimension 41, and head 1's dimension 5 simultaneously. The heads don't stay separate after `W_O` — they're fully blended.

Key insights:

1. **Heads are independent during attention computation but fully mixed by W_O.** The attention patterns themselves are computed in isolated subspaces (no cross-head communication in the `QK^T` or `PV` steps). All cross-head interaction happens after concatenation, through `W_O`. This is a deliberate design: heads can develop specialized attention patterns without interference, but their outputs are combinatorially combined.

2. **If W_O were block-diagonal, heads would never interact.** Each head's output would only affect a fixed subset of the final representation. The full dense `W_O` is what makes multi-head attention more expressive than simply averaging heads — it learns a weighted, cross-dimensional synthesis.

3. **The residual addition is across the same vector space after W_O.** `W_O` projects the concatenated head outputs back into `d_model`-dimensional space — the same space as the input `x`. So the residual `x + Attn(x)` adds vectors that live in the same space. It's not adding vectors from different spaces.

### The Residual Connection — Adding Back the Input

Each Transformer layer has two sublayers, both wrapped in residual connections:

```
x_out = LayerNorm(x_in + MultiHeadAttention(x_in))
x_out = LayerNorm(x_out + FFN(x_out))
```

The attention output is **added element-wise** to the original input, then layer-normalized. This is why `W_O` must project back to `d_model` — the shapes must match for the residual addition to work.

Why residuals are essential:
1. **Identity-preserving gradient highway.** Without residuals, gradients must flow through the attention and FFN transformations. With residuals, there's a direct additive path `∂x_out/∂x_in = I + ...` — the identity term ensures gradients don't vanish even in 100+ layer networks.
2. **Each sublayer refines rather than replaces.** The attention/FFN learns the **delta** from the current representation — what to add or correct. The original signal always passes through. Early-layer information is never fully overwritten.
3. **Pre-LN vs Post-LN.** The original paper applied `x + Sublayer(LayerNorm(x))` (Post-LN). Modern Transformers (including the Parallax paper's Qwen-3 backbone) use Pre-LN: `x + Sublayer(LayerNorm(x))` — normalize first, then transform, then add. Pre-LN is more stable at initialization.

### Visualizing the Full Flow

```
Input x: (d_model = 512)
     │
     ├──> W_Q (512→512) ──> reshape ──> [head_1 Q | head_2 Q | ... | head_8 Q]
     ├──> W_K (512→512) ──> reshape ──> [head_1 K | head_2 K | ... | head_8 K]
     └──> W_V (512→512) ──> reshape ──> [head_1 V | head_2 V | ... | head_8 V]
                                              │
              Each head computes independently: head_i = softmax(Q_i K_i^T / √d) · V_i
              Each head outputs 64-dim vector
                                              │
     Concat: [64-dim | 64-dim | ... | 64-dim] = 512-dim vector
                                              │
     W_O (512×512): full dense projection — every output dim mixes across all heads
                                              │
     output: 512-dim vector ← SAME SPACE as input x
                                              │
     Residual: x + output  ← element-wise addition, same space
                                              │
     LayerNorm
                                              │
     FeedForward (typically 4× expansion then projection)
                                              │
     Residual: norm_out + FFN_out
```

---

### In Pseudocode

```python
# x: (batch, seq_len, d_model)
# h = num_heads, d_k = d_v = d_model // h

# Project to Q, K, V for all heads at once (batched matmul)
Q = x @ W_Q  # (batch, seq_len, d_model)
K = x @ W_K
V = x @ W_V

# Reshape to separate heads
Q = Q.reshape(batch, seq_len, h, d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
K = K.reshape(batch, seq_len, h, d_k).transpose(1, 2)
V = V.reshape(batch, seq_len, h, d_v).transpose(1, 2)

# Attention per head (parallelized as batched matmul)
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # (batch, h, seq_len, seq_len)
attn_weights = softmax(scores, dim=-1)
head_outputs = attn_weights @ V                # (batch, h, seq_len, d_v)

# Combine: concat + project
combined = head_outputs.transpose(1, 2).reshape(batch, seq_len, d_model)
output = combined @ W_O                        # (batch, seq_len, d_model)
```

---

### Shape Walkthrough (Original Paper: d_model=512, h=8)

| Step | Tensor | Shape |
|------|--------|-------|
| Input | X | (batch, seq, 512) |
| Project to Q/K/V | Q, K, V | (batch, seq, 512) |
| Split into heads | Q, K, V per head | (batch, seq, 8, 64) |
| Transpose for batching | Q, K, V | (batch, 8, seq, 64) |
| Attention scores | Q @ K^T | (batch, 8, seq, seq) |
| Weighted values | attn @ V | (batch, 8, seq, 64) |
| Merge heads + project | Concat · W_O | (batch, seq, 512) |

---

### GQA (Grouped-Query Attention) — A Practical Variant

Introduced in GQA (Ainslie et al., 2023) and used in Llama, Qwen, and the Parallax paper's experiments:

- Instead of each query head having its own KV head, **multiple query heads share one KV head**
- Example: 16 query heads grouped into 8 KV heads (2 query heads share each KV)
- Reduces KV-cache memory during inference without much quality loss
- The combination (concat + W_O) remains identical

### Relation to Parallax

In the Parallax paper's context, each head independently computes both:
- The softmax attention (intercept term)
- The covariance correction term `Σ_KV · ρ`

Both per-head outputs go through the same concat + `W_O` projection. The paper's parameter-matched control (Transformer†) increases the query head count via GQA grouping to match Parallax's extra `W_R` parameters — the combination mechanism itself remains the same.
