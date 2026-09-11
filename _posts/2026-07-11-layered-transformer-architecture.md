---
layout: post
title: "The Layered Transformer Architecture — Why Stacking Matters"
date: 2026-07-11 00:40:20 +0800
categories: paper-notes
---

## 1. The Basic Building Block

A single transformer layer contains two main sub-layers:

```
Input (hidden states, shape: L × d_model)
   │
   ├─ [Multi-Head Self-Attention] ─── "which tokens should I pay attention to?"
   │      │
   │      └─ Add & LayerNorm (residual connection)
   │
   ├─ [Feed-Forward Network (FFN)] ─── "what should I do with that information?"
   │      │
   │      └─ Add & LayerNorm (residual connection)
   │
   ▼
Output (same shape: L × d_model)
```

Each layer takes a sequence of vectors and produces a sequence of the same shape. This is why layers can be **stacked** — the output of layer 1 is the input to layer 2, and so on.

## 2. Why Stack Layers At All?

A single layer can only do so much. Here's what stacking buys you:

### 2.1 Compositional Abstraction

Think of it like a pipeline. Each layer transforms the representation into something slightly more abstract:

```
Layer 1-6 (early):   Local syntax, word morphology, part-of-speech
Layer 7-18 (middle): Semantic roles, entity references, coreference
Layer 19-30 (late):  Discourse structure, world knowledge integration
Layer 31-36 (final): Task-specific prediction (what comes next)
```

A single layer cannot do all of this. The depth creates a **hierarchy of representations** — each layer builds on the abstractions of the previous ones. This is the same principle that makes deep ConvNets work for vision (edges → textures → parts → objects).

### 2.2 Path Length for Information Flow

In a single-layer transformer, every token can attend to every other token — one hop. In a stacked transformer, a token at position 1 can influence a token at position 100 through multiple intermediate representations across layers. Even though attention is global within each layer, meaning propagates through the depth.

Consider: "The cat that chased the mouse that ate the cheese is black."

A single layer sees all tokens at once, but understanding that "is black" refers to "the cat" (not the mouse, not the cheese) requires compositional reasoning that depth enables — each layer refines the subject-verb agreement signal.

### 2.3 Capacity Without Width Explosion

You could theoretically use one very wide layer instead of many narrow ones. But the parameter count would be enormous. Stacking is parameter-efficient:

```
One wide layer:  d_model = 16384, FFN = 65536  →  ~1.1B params
12 layers:       d_model = 768,   FFN = 3072    →  ~85M params
```

The deep version uses far fewer parameters for similar expressive power, because each layer can **reuse** the same parameters across different levels of abstraction. It's like the difference between one giant function and a composition of smaller functions.

### 2.4 Gradient Flow via Residual Connections

Each layer has a residual connection (`y = x + F(x)`). In a deep stack, this creates **gradient highways** — the gradient can flow directly from the final layer back to the first layer without attenuation through the residual path. Without residual connections, deep transformers would suffer from vanishing gradients just like deep plain networks.

This is why all the architectural innovations we've discussed (mHC, IndexCache cross-layer sharing, the Weibull paper's layer-wise diagnostics) matter — the depth enables rich representation learning, but it also creates new engineering challenges:

| Challenge | Solution |
|---|---|
| Signal explosion in deep models | mHC (constrain mixing to doubly stochastic) |
| Redundant computation across layers | IndexCache (reuse attention indices) |
| Understanding what each layer learns | Weibull diagnostics (per-layer weight analysis) |
| Parameter inefficiency | MoE (sparse activation per layer) |

## 3. How Many Layers Do Real Models Use?

| Model | Layers | Notes |
|---|---|---|
| Original Transformer (2017) | 6 enc + 6 dec | The starting point |
| BERT-base | 12 | Encoder-only |
| GPT-3 175B | 96 | The scaling era begins |
| LLaMA-3 8B | 32 | Modern dense model |
| LLaMA-3 70B | 80 | Wider and deeper |
| DeepSeek-V3 | 61 | MoE |
| DeepSeek-V4 | 61 | MoE + mHC |
| The DSA model in IndexCache | 47 | Sparse attention |

## 4. What Changes Across Layers?

Not all layers do the same thing. Evidence from the papers we've read:

**Attention patterns change.** Early layers attend broadly (local syntax). Middle layers develop specialized attention heads (induction heads for copying patterns). Late layers focus on task-relevant tokens.

**Weight distributions differ.** The Weibull paper showed that FFN weights stay near initialization (`k ≈ 1.20`) across all layers, but attention query/key weights in mid-to-deep layers develop heavy tails — they become more selective.

**Expert specialization in MoE.** In multimodal MoE models, early layers are dominated by text experts, while later layers have progressively more vision and multimodal experts — the model learns a **separate-then-integrate** strategy across depth.

**Indexer redundancy.** Adjacent layers select 70-100% the same tokens in sparse attention. The similarity forms blocks (layers 17-30 share one pattern, layers 31-36 another), which is what IndexCache exploits.

## 5. The Key Insight

The layered architecture is useful not because each layer does something completely different, but because stacking creates a **continuous transformation of the representation space**. Each layer applies a relatively small, learnable change. Cumulatively, these small changes can produce dramatically different representations — moving from raw token embeddings to rich, context-aware semantic vectors over the course of 30-60 small steps.

This is fundamentally the same idea as an ODE solver taking many small steps to solve a complex differential equation, or gradient descent taking many small steps to find a minimum. The power comes from the **composition** of simple operations, not from any individual layer's complexity.
