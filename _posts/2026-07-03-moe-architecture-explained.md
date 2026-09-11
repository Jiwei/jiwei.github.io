---
layout: post
title: "Mixture of Experts (MoE) for Transformers"
date: 2026-07-03 15:11:49 +0800
categories: paper-notes
---

## 1. The Basic Idea

A standard transformer FFN layer looks like this:

```
x → FFN(x) → output
```

Every token goes through the same FFN parameters. With MoE, you have **multiple FFNs (experts)** and a **router** that decides which ones each token uses:

```
x → Router(x) → "token should go to experts 3, 7, 12"
              → Expert_3(x) + Expert_7(x) + Expert_12(x) → output
```

The key insight: **total model capacity grows, but compute per token stays roughly the same** — each token only activates a small subset of experts.

## 2. The Components

**Router (Gate):** A learned linear layer `W_r` that takes the token's hidden state and outputs a distribution over experts:

```
scores = softmax(W_r · x)     # shape: [num_experts]
```

**Top-k selection:** Only the k highest-scoring experts are activated (typically k=2, or k=8–16 in fine-grained MoE):

```
selected_experts = topk(scores, k)
```

**Experts:** Each expert is a full FFN (typically with the same hidden dimension as the original FFN, or smaller in fine-grained designs). Different experts specialize in different kinds of tokens or patterns.

**Combined output:**

```
output = Σ_i g_i · Expert_i(x)    for i in top-k experts
```

where `g_i` is the router score (softmax weight) for expert i.

### Soft vs. Hard Routing

**Soft routing (weighted combination):**
```
output = Σ scores_i · Expert_i(x)    for i in top-k
```
Each expert's contribution is weighted by its router score. This is what most MoE implementations use.

**Hard routing (equal weight after selection):**
All selected experts contribute equally regardless of router score. Simpler but less expressive.

## 3. The Full Per-Layer Flow

```
Token hidden state x (from attention + residual)
        │
        ▼
    ┌─────────────┐
    │   Router     │  ← linear layer W_r
    │  scores =    │
    │  softmax(W_r·x)│
    └──────┬──────┘
           │
           ▼
    Pick top-k experts (e.g., experts 3, 7, 12 with scores 0.5, 0.3, 0.2)
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  FFN_3  FFN_7  FFN_12     ← each expert is an independent FFN
    │      │      │
    ▼      ▼      ▼
   y₃     y₇     y₁₂       ← each produces a vector of same dimension d_model
    │      │      │
    └──────┼──────┘
           │
           ▼
    y = 0.5·y₃ + 0.3·y₇ + 0.2·y₁₂    ← weighted sum
           │
           ▼
    Add back residual, LayerNorm → goes to next layer
```

The output `y` is a single vector — a weighted combination of the experts' outputs, where the **router scores are the weights**. No voting, no ensembling — it's just a gated linear combination. The next layer sees one hidden state, not k of them.

## 4. Per-Token Routing

Every token **independently** goes through this flow. Two adjacent tokens in the same sentence can be routed to completely different experts:

```
Token₁: "cat"  → Router → experts {3, 7, 12}  → 0.5·FFN₃ + 0.3·FFN₇ + 0.2·FFN₁₂
Token₂: "the"  → Router → experts {1, 3, 19}  → 0.6·FFN₁ + 0.25·FFN₃ + 0.15·FFN₁₉
Token₃: "sat"  → Router → experts {7, 9, 14}  → 0.4·FFN₇ + 0.35·FFN₉ + 0.25·FFN₁₄
```

This is the whole point — **conditional computation per token**. A content word like "cat" might need different expertise than a function word like "the," and the router learns this.

In practice, it's batched: the entire sequence (shape: `[seq_len, d_model]`) goes through the router in one matrix multiply:

```
scores = softmax(X · W_r^T)     # shape: [seq_len, num_experts]
```

Each row is one token's expert distribution. Then, for each expert, you gather the tokens that selected it, run them through that expert's FFN, and scatter the results back to compute the weighted sum at each position.

## 5. Where Token Prediction Actually Happens

The experts don't each predict a token. They collectively shape a single hidden state, and only the final layer's output goes through the LM head:

```
Position:     0        1        2        3        4
Input:       [BOS]    "The"    "cat"    "sat"    "on"
              │        │        │        │        │
              ▼        ▼        ▼        ▼        ▼
           Hidden₀  Hidden₁  Hidden₂  Hidden₃  Hidden₄
              │        │        │        │        │
              ▼        ▼        ▼        ▼        ▼
           LM Head  LM Head  LM Head  LM Head  LM Head
              │        │        │        │        │
              ▼        ▼        ▼        ▼        ▼
Predict:    "The"    "cat"    "sat"    "on"     "the"
```

Each position's hidden state goes through the **same LM head** (a linear projection of shape `[vocab_size, d_model]`) to produce logits over the entire vocabulary for the **next** token:

```
logits_i = W_lm · Hidden_i     # Hidden_i contains all context up to and including token i
next_token_i = argmax(logits_i)
```

### At Training Time

The loss is computed at every position simultaneously:

```
Loss = Σ_i CrossEntropy(logits_i, actual_token_{i+1})
```

All in one forward pass — the causal mask ensures position i can only see tokens ≤ i.

### At Inference Time

Only the **last** position matters:

```
Input:     "The cat sat on"     ← positions 0-4
                │
                ▼
           Hidden₄ (at position "on")
                │
                ▼
           LM Head → logits → "the" (position 5)

Now append "the":
Input:     "The cat sat on the"
                │
                ▼
           Hidden₅ (at position "the")
                │
                ▼
           LM Head → logits → "mat" (position 6)
```

Each new token's hidden state encodes the entire prefix up to that point (via causal attention). So Hidden₄ already "knows" about "The cat sat on" — no need to combine multiple hidden states. The LM head just reads that single enriched vector and predicts what comes next.

## 6. Key Design Parameters

| Parameter | What it means | Typical values |
|---|---|---|
| **E** (total experts) | How many FFNs exist in total | 8–256+ |
| **k** (top-k) | How many experts each token activates | 2 (classic MoE), 8–16 (fine-grained) |
| **G** (granularity) | Expert size relative to model dim: G = 4·d_model / d_expert | 1 (few large experts) to 64 (many tiny experts) |
| **Active ratio** | k / E — fraction of total capacity used per token | 50% down to 1.6% |
| **Sparsity** | E / k — total capacity vs. active compute | 2× to 64×+ |

## 7. Classic vs. Fine-Grained MoE

**Classic MoE** (Switch Transformer, GShard): k=1 or 2, experts are the same size as the original FFN (~d_ff = 4·d_model). Few large experts.

**Fine-grained MoE** (DeepSeek-V2/V3): Many small experts. For example, G=16 means each expert has dimension d_model/4 instead of 4·d_model, but there are 256 of them, and each token routes to 16. This gives the router more granular control — a token can mix a larger number of specialized capabilities.

## 8. Load Balancing

Without constraints, the router can collapse — sending all tokens to the same few experts, leaving others unused. Solutions:

- **Auxiliary loss:** Penalize the router if expert utilization is too uneven (the most common approach)
- **Expert capacity:** Cap how many tokens each expert can process per batch; overflow tokens are dropped or routed to a residual path
- **Load-balanced routing:** Use techniques like Sinkhorn normalization to enforce balanced assignment

## 9. Why MoE Matters for Scaling

It decouples **total capacity** from **compute per token**. You can have a 50B-parameter model that only costs as much to run as a 1.5B dense model, because each token only sees 1.5B parameters. This is why DeepSeek-V2/V3, Mixtral, and increasingly multimodal models use MoE.

## 10. MoE in Multimodal Models

From the "Beyond Language Modeling" paper, key findings about MoE in multimodal pretraining:

1. **Experts naturally specialize by modality** without being told — some become text experts, some vision experts, some multimodal
2. **Early layers are text-dominated**; later layers have more vision/multimodal experts (separate-then-integrate processing strategy)
3. **Vision experts are general-purpose** — the same experts handle both image understanding and image generation (r ≥ 0.90 correlation)
4. **Higher granularity helps** up to G=16, with language benefiting more from fine-grained routing than vision
5. **Per-modality shared experts** (always-active experts dedicated to text or vision) outperform a single global shared expert

## 11. The Core Intuition

MoE is not like an ensemble where each expert makes an independent prediction and you vote. It's more like having a **toolbox of specialized subroutines** — for each token, you pick a few tools and combine their results. The routing and the expert parameters are trained jointly, so the experts evolve to complement each other rather than compete. This is why load-balancing losses are needed — without them, the router would collapse to using one expert for everything, which defeats the purpose.
