---
layout: post
title: "Scaling Latent Reasoning via Looped Language Models (Ouro)"
date: 2026-08-17 17:06:01 +0800
categories: paper-notes
---

**Paper:** [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)
**Authors:** Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, et al. (ByteDance Seed, UC Santa Cruz, Princeton, Mila, and others)
**Date:** November 2025
**Project:** ouro-llm.github.io
**Model:** Ouro (1.4B and 2.6B)

---

## 1. Core Idea: Looped Language Models (LoopLM)

Standard LLMs reason through **explicit text generation** (Chain-of-Thought), which defers reasoning to post-training and under-leverages pre-training data. This paper proposes **Looped Language Models (LoopLM)** — a third scaling axis beyond model size and data:

> **Reuse the same transformer layers iteratively** within a single forward pass, letting the model "think" in latent space instead of emitting tokens.

The design (named Ouro, after the recursive Ouroboros serpent):

```
F^(t)(·) = lmhead ∘ (M_L ∘ M_L ∘ ... ∘ M_L) ∘ emb(·)
                     └──── t iterations ────┘
```

A stack of L layers is applied **t times**. At t=1 it's a normal transformer; at t=4 the same 24/48 layers run four times.

### Why This Helps
1. **Adaptive computation:** A learned exit gate lets simple inputs stop early, hard inputs run deeper — decoupling compute depth from parameter count.
2. **No context-length bloat:** Unlike CoT (which extends the output sequence), LoopLM deepens the internal computation graph.
3. **Capacity per parameter:** Same parameters, more computation, better capability.

---

## 2. Key Technical Innovations

### 2.1 Entropy-Regularized Adaptive Computation

At each loop step t, an **exit gate** predicts probability λ_t of exiting. The model learns a distribution over exit steps.

The training objective combines next-token prediction with entropy regularization:

```
L = Σ p_ϕ(t|x) · L^(t)  +  β · H(p_ϕ(·|x))
    └─ expected task loss ─┘   └─ entropy regularization ─┘
```

**The insight:** Naive gradient descent collapses the exit distribution onto the deepest step (t = T_max). The entropy term (equivalently a KL divergence to a **uniform prior**) prevents collapse. This is framed as variational inference — an ELBO where the exit step is a latent variable.

The uniform prior is **depth-unbiased** (unlike geometric priors that favor early exit), letting input difficulty — not global compute preference — drive exit decisions.

### 2.2 Focused Adaptive Gate Training (Stage II)

A second stage freezes the LM and trains only the gate. It computes a **loss-improvement signal** I^(t) = L^(t-1) - L^(t), and teaches the gate to continue when improvement is large and exit when gains stall. This penalizes both:
- **Underthinking** (exiting when it should continue)
- **Overthinking** (continuing when it should exit)

### 2.3 KV Cache Sharing (Inference Efficiency)

Naively, 4 recurrent steps need 4× KV cache memory. Key finding:
- **Prefilling:** All steps need their own cache (reuse degrades >10 points)
- **Decoding:** Reusing **only the last step's** cache (or averaging) achieves near-identical performance with 4× memory reduction

This makes LoopLM deployment memory-efficient, comparable to standard transformers.

---

## 3. Training Pipeline (7.7T Tokens)

```
Warmup → Stable Training (3T) → CT Annealing (1.4T) → LongCT (20B) → Mid-training (300B) → Reasoning SFT
                                   └───────────── forks ─────────────┘
                                    1.4B (24 layers) + 2.6B (48 layers)
```

- Recurrent steps reduced from 8 → 4 during training for stability (8 caused loss spikes from compounded gradient flow)
- Batch size scaled 4M → 8M tokens for stability
- β reduced 0.1 → 0.05 to reduce conflict between task loss and KL penalty
- Recurrent architectures need **smaller learning rates** than parameter-matched transformers

---

## 4. Results

### 4.1 Parameter Efficiency (Base Models)

| Model | Key Result |
|---|---|
| Ouro 1.4B (R4) | Matches **4B Qwen3** on most benchmarks; beats it on GSM8K (78.9 vs 72.9), MATH500 (82.4 vs 59.6) |
| Ouro 2.6B (R4) | Outperforms **8B Qwen3** on MMLU-Pro (55.7 vs 53.7), BBH (80.5 vs 77.7), MATH500 (90.9 vs 62.3) |

**2-3× parameter efficiency gain** — a 2.6B model matching 8B models.

### 4.2 Reasoning Models (Ouro-Thinking)

On advanced benchmarks (AIME, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE):
- Ouro-1.4B-Thinking R4 is competitive with 4B models
- Ouro-2.6B-Thinking R4 matches or exceeds 8B models

### 4.3 Recurrent Depth Scaling and Extrapolation

- Performance generally peaks at the trained depth (T=4), then degrades at T=5-8 (extrapolation)
- Reasoning models peak slightly differently: 1.4B peaks at T=4/5, 2.6B at T=3/4
- **Safety keeps improving** even at extrapolated depths (T>4), unlike task performance

### 4.4 Early Exit Efficiency

The adaptive gate with specialized training achieves the best accuracy at every compute budget. A key finding: the jump from 1→2 rounds gives massive gains (40%→60%), while 3→4 rounds gives marginal gains — explaining why adaptive computation is so effective (most examples need only intermediate depth).

---

## 5. The Crucial Mechanistic Finding: Knowledge Manipulation, Not Knowledge Capacity

Using "Physics of Language Models" controlled experiments, the paper isolates **why** LoopLM works:

### 5.1 Knowledge Capacity: UNCHANGED
Using the "Capo" task (synthetic biographies), both looped and non-looped models store ~**2 bits/parameter** of factual knowledge. Looping does NOT increase knowledge storage.

### 5.2 Knowledge Manipulation: DRAMATICALLY IMPROVED
- **Mano task** (modular arithmetic with tree structure): Looped models beat iso-parameter models, and often beat iso-FLOP models
- **Multi-hop QA** (composing facts): Looped models learn with **fewer samples** and faster than non-looped models

**The conclusion:** LoopLM's advantage comes from **better knowledge manipulation** (composing, multi-hop reasoning, applying procedures), not from storing more facts.

### 5.3 Theoretical Explanation (Theorem 1)

For graph reachability (a proxy for multi-hop reasoning), a looped transformer can solve the task in **O(log D)** sequential steps, compared to:

| Latent reasoning method | Sequential computation steps |
|---|---|
| Discrete CoT | O(n²) |
| Continuous CoT | O(D) |
| Universal Transformer (LoopLM) | O(log D) |

Looping maximizes parallelism in exploring all-pair connectivity, exponentially reducing sequential computation steps. This is the likely source of LoopLM's superior knowledge manipulation.

---

## 6. Safety and Faithfulness

### 6.1 Safety Improves with Recurrence
On HEx-PHI (harmfulness benchmark), **safety improves monotonically with recurrent steps — even into extrapolated steps (T>4)**. This is the opposite of task performance, which degrades beyond T=4.

PCA analysis shows that as steps increase, the model better **separates benign from harmful prompts** — difficult-to-distinguish prompts (near the cluster boundary) are where unsafe responses occur, and more recurrence helps resolve them.

### 6.2 Faithfulness: LoopLM Beats CoT's Post-hoc Rationalization

A well-documented problem: CoT models often **decide the answer first, then rationalize** — the reasoning text is not causally coupled to the output.

On Quora Question Pairs (ambiguous semantic equivalence):
- Qwen3-4B-Thinking: a linear probe on early representations predicts the final answer with 0.99 ROC AUC — the "thinking" almost doesn't affect the result (pure rationalization)
- Ouro 1.4B: probes on earlier representations **do NOT** reliably predict later-step decisions. Adjacent steps show real disagreement (e.g., only 36.1% of step-2 answers match step-4). The model genuinely **updates its decision** as recurrence deepens.

This means LoopLM's latent trajectory is **causally faithful** — intermediate states genuinely mediate the final answer, unlike CoT's frozen rationalization.

### 6.3 Bonus: Built-in Speculative Decoding

The recurrent structure naturally provides a draft-verify decomposition: `Text(R_s)` (early step) proposes, `Text(R_T)` (final step) verifies — no external draft model needed.

---

## 7. Limitations

1. **RL attempts failed:** RLVR (DAPO/GRPO) didn't improve over SFT, because vLLM/SGLang's fixed execution path breaks under variable-depth computation
2. **Extrapolation degrades:** Beyond T=4, task performance drops (only safety keeps improving)
3. **Smaller learning rates needed:** Recurrent architectures require conservative LR schedules (not exhaustively swept)
4. **Chinese removed after Stage 1:** The tokenizer lacks Chinese vocabulary, so Chinese data was dropped from Stage 2 onward

---

## 8. Key Takeaways

1. **LoopLM is a third scaling axis.** Beyond parameters and data, iterative latent computation provides 2-3× parameter efficiency at scale (7.7T tokens).

2. **The gain is knowledge manipulation, not storage.** Controlled experiments prove looping doesn't store more facts (~2 bits/param both ways) but dramatically improves the ability to compose and manipulate knowledge — with a theoretical basis in graph reachability (O(log D) vs O(n²) steps).

3. **Adaptive depth is trainable.** Entropy regularization (uniform prior) prevents collapse, and focused gate training achieves the best accuracy-efficiency trade-off.

4. **Faithfulness is a unique advantage.** Unlike CoT's post-hoc rationalization, LoopLM's latent reasoning genuinely mediates decisions — intermediate steps update the answer.

5. **Safety improves with more thinking.** More recurrent steps monotonically improve safety alignment, even extrapolated beyond training depth.
