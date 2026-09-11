---
layout: post
title: "FFN and Next-Token Prediction in Transformers"
date: 2026-07-13 10:56:07 +0800
categories: paper-notes
---

## Part 1: FFN — Per-Token Operation with Full Parallelism

### 1. The Basic Computation

FFN is a **position-wise** operation — it applies the same weights to each token independently:

```
For position i:  y_i = W₂ · σ(W₁ · x_i + b₁) + b₂
```

Where `x_i` is the hidden vector at position i, and `W₁`, `W₂` are shared across all positions.

### 2. All Tokens Are Computed in Parallel

On GPU, this is one matrix multiplication — no for-loop needed:

```
X = [x₁, x₂, ..., x_L]^T          shape: [L, d_model]
Y = σ(X @ W₁ + b₁) @ W₂ + b₂      shape: [L, d_model]
```

A single GEMM (general matrix multiply) processes the entire batch of tokens simultaneously.

### 3. FFN vs. RNN — The Fundamental Difference

RNNs process tokens **sequentially**:

```
Token₁ → compute → state₁ → Token₂ → compute → state₂ → Token₃ → ...
```

Must wait for Token₁ to finish before starting Token₂. This is why RNN inference is slow.

In a Transformer, **all tokens enter simultaneously and complete simultaneously**. The FFN doesn't need to wait for preceding tokens — this is the foundation of parallelism.

### 4. The Division of Labor: Attention vs. FFN

FFN does NOT model relationships between tokens. This is not a bug — it's a deliberate design:

```
A single transformer layer:

Attention ─── "Which tokens are related to each other?"
                    └── Sequence-level information (inter-token relationships)

FFN ─────── "Now that I have the full context, what should I understand/process about this token?"
                    └── Token-level information (independent depth processing)
```

#### What Attention Does: Aggregate Sequence Information

Attention is an **information routing** mechanism. It lets each token **collect** relevant context from the entire sequence:

```
"The cat sat on the mat"
   │   ↑   │   │
   └───┼───┘───┘   ← Attention: every word sees every other word

After Attention, the representation of "cat" = "cat" itself + info from "The" + info from "sat" + ...
```

#### What FFN Does: Deep Processing on Context-Rich Representations

Attention has already **compressed the entire sequence's information into each token's vector**. The FFN then:

- **Stores knowledge.** FFNs have been shown to function as key-value memory banks — `W₁` matches input patterns, `W₂` retrieves corresponding knowledge.
- **Adds non-linearity.** Without FFNs, the entire network would be a linear transformation of the input (since attention is also linear in values).
- **Processes independently.** Each token already has full context — independent processing at this stage is exactly what's needed. The FFN performs deep transformation on the already-contextualized representation.

#### Why No Inter-Token Interaction in FFN?

Because **Attention already did it**. The flow is:

```
Entering the layer:  token has only its own information
         │
         ▼  Attention (inter-token interaction)
Each token now contains the entire sequence's information
         │
         ▼  FFN (per-token independent processing)
Deep transformation on the already-contextualized representation
```

| | Attention | FFN |
|---|---|---|
| **Token interaction** | Yes — each token can attend to others | No — each token fully independent |
| **Computational complexity** | O(L² · d) — pairwise computation | O(L · d²) — per-token computation |
| **Parallelism** | All query-key pairs computed simultaneously | All tokens transformed simultaneously |
| **Role** | Information routing (sequence space) | Deep processing (representation space) |

### 5. The FFN Updates Each Token's Hidden State

Yes — that's the cleanest way to think about it. The entire transformer layer can be understood as each token's hidden state being updated through two complementary steps:

```
Token hidden state h_i (shape: d_model)
        │
        ▼
   [Attention]  ─── "Gather relevant information from other tokens"
        │            Updates h_i with context from the sequence
        │            h_i ← h_i + Attention(h_i, all other tokens)
        ▼
   [FFN]        ─── "Process and refine what this token now represents"
        │            Updates h_i with deeper computation on itself
        │            h_i ← h_i + FFN(h_i)
        ▼
Updated hidden state h_i'  (same shape: d_model)
```

Both steps are **additive** (via residual connections), so each one refines rather than replaces the representation.

The difference is where the new information comes from:

| | Attention | FFN |
|---|---|---|
| **Where does new information come from?** | Other tokens in the sequence | This token's own representation |
| **What does it do to h_i?** | Injects context from relevant tokens | Applies stored knowledge and non-linear transformation |
| **Analogy** | "Let me look around and see what's relevant" | "Let me think about what I've gathered" |

### 6. Input and Output Shape Are Identical

The FFN takes `[L, d_model]` and outputs `[L, d_model]` — same shape in both directions.

Internally, it expands then contracts:

```
[L, d_model]  →  [L, d_ff]  →  [L, d_model]
   input        hidden (wider)    output
```

The hidden dimension `d_ff` is typically **4× larger** than `d_model` (e.g., d_model=4096 → d_ff=16384). This expansion-then-contraction pattern is why the FFN contains most of the model's parameters:

```
Parameters in FFN  = d_model × d_ff + d_ff × d_model
                   = 2 × d_model × d_ff
                   ≈ 8 × d_model² (when d_ff = 4 × d_model)
```

This is typically **~2/3 of all parameters** in a transformer layer (e.g., LLaMA-8B: ~5.3B in FFN vs. ~2.1B in attention).

The same-shape in/out is also what makes **residual connections** possible — you can only add `x + F(x)` if `F(x)` has the same shape as `x`.

---

## Part 2: Next-Token Prediction — How the Final Token Is Generated

### 1. The Inference Flow

At inference time, the **entire sequence** is input, but only the **last position's output** is used for prediction:

```
Input sequence:    "The  cat  sat  on"
                      │    │    │    │
                      ▼    ▼    ▼    ▼
              [Transformer with causal attention:
               each position can only see itself and preceding tokens]
                      │    │    │    │
                      ▼    ▼    ▼    ▼
Hidden states:       h₀   h₁   h₂   h₃
                                      │
                                      ▼
                                 LM Head (linear projection)
                                      │
                                      ▼
                              Vocabulary distribution
                                      │
                                      ▼
                              Prediction: "the"
```

### 2. h₃ Already Contains the Entire Sequence

Only h₃ enters the LM Head, but h₃ is **not isolated** — through causal attention, it has seen all preceding tokens:

```
h₃'s information chain:
  "The" ──→ "cat" ──→ "sat" ──→ "on" ──→ h₃
    ↑         ↑         ↑         ↑        ↑
  included   included  included  direct    final
                                 input   representation
```

This is why there's no need to merge hidden states from all positions — **causal attention has already made the last position a summary of the entire sequence**.

### 3. At Training Time: All Positions Predict Simultaneously

During training, all positions generate predictions in one forward pass:

```
Input:     "The  cat  sat  on"
             │    │    │    │
             ▼    ▼    ▼    ▼
Hidden:     h₀   h₁   h₂   h₃
             │    │    │    │
             ▼    ▼    ▼    ▼
Predict:   "cat" "sat" "on"  "the"    ← each position predicts the next token
             │    │    │    │
             ▼    ▼    ▼    ▼
Labels:    "cat" "sat" "on"  "the"    ← compared with ground truth
```

Because the causal mask ensures:
- h₀ can only see "The" → predicts "cat"
- h₁ can only see "The cat" → predicts "sat"
- h₂ can only see "The cat sat" → predicts "on"
- h₃ can see "The cat sat on" → predicts "the"

All positions compute their losses simultaneously in one forward pass.

### 4. Summary

| | Input | Which hidden states are used for prediction |
|---|---|---|
| **Inference** | Entire sequence | **Only the last** position's h |
| **Training** | Entire sequence | **All positions'** h (single forward pass) |

The key insight: **the last position's hidden vector h is already a complete contextual representation of the entire sequence** — it aggregates all preceding information through causal attention, requiring no additional merging step.
