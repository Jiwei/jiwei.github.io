---
layout: post
title: "PolyAlign: Conditional Human-Distribution Alignment"
date: 2026-07-03 11:25:54 +0800
categories: paper-notes
---

**Paper:** [arXiv:2606.13227](https://arxiv.org/abs/2606.13227)
**Authors:** L. D. M. S. Sai Teja, Ufaq Khan, Sathira Silva, Xiao Wu, Muhammad Haris Khan (NIT Silchar, MBZUAI)
**Date:** June 2026
**Code:** github.com/saitejalekkala33/PolyAlign.git

---

## 1. Core Problem: Standard Alignment Makes Every Response Sound the Same

Standard post-training (SFT + DPO/RLHF) aligns LLMs toward a **single global assistant behavior** — polite, helpful, uniformly formatted. While this improves average helpfulness, it **suppresses the natural variation of human responses** across different contexts. Humans write differently in different situations: a short factual QA answer, a long-form explanation, an open-ended chat, a Chinese vs. English response. Standard alignment **collapses** these diverse distributions into one generic style.

The paper formulates this as **conditional human-distribution alignment**: the model should match the human response distribution appropriate to the **current interaction context** (language, task type, response family, length), not a universal template.

---

## 2. The Framework: PolyAlign

### 2.1 Bucket-Based Organization

Training data is partitioned into **buckets** based on metadata:

| Dimension | Values |
|---|---|
| **Language** | 2 (English, Chinese) |
| **Situation category** | 5 (assistant_like, longform_qa, open_chat, qa_search, task_dialogue) |
| **Interaction track** | 2 (single, multi) |
| **Response family** | Dataset-specific groupings |
| **Length bin** | Binned response lengths |

Each bucket has its own **human reference distribution** `Λ_b` over linguistic features — capturing how humans naturally vary their responses within that specific interaction regime.

The exact number of buckets isn't explicitly reported — it's the cross product of metadata dimensions after filtering out buckets with <20 held-out examples. It's on the order of dozens to low hundreds across the 14 datasets (~694K English + ~104K Chinese examples).

### 2.2 Bucket-Aware SFT (Bucket-SFT)

Standard SFT lets frequent buckets dominate. Bucket-SFT assigns **equal optimization mass to each bucket** regardless of size:

```
L_Bucket-SFT(θ) = (1/|B|) · Σ_b [(1/n_b) · Σ_{i: b_i=b} ℓ_i(θ)]
```

**Theorem 1** proves this is exactly **macro bucket risk minimization** — every bucket contributes identical total weight to the gradient. A QA bucket with 100K examples gets the same total optimization mass as an open_chat bucket with 4K examples.

### 2.3 Human-Distribution Preference Optimization (HDPO)

HDPO extends standard DPO with a **critic-based regularizer** that favors responses closer to their bucket's human support:

**Critic:** Trained to estimate bucket-support distance `D_b(z)` — how far a generated response's linguistic features deviate from the human feature range for its bucket. Zero when all features fall inside the human support intervals; grows as the response moves outside. This is a continuous relaxation of human membership: `D_b(z) ≥ 0` and `D_b(z) = 0` iff all features are within bucket bounds (Theorem 2).

**HDPO Regularizer:**

```
R_HDPO(θ) = p_θ · s_ϕ(y⁺, b) + (1 − p_θ) · s_ϕ(y⁻, b)
```

where `p_θ = σ(β·Δπ)` is the policy's probability of preferring the chosen response, and `s_ϕ` is the critic score (lower = closer to human distribution). This penalizes the policy when it puts mass on distributionally-deviant responses.

**HDPO Loss:**

```
L_HDPO(θ) = (1/K) · Σ [w_i · L_DPO(θ) + λ_hd · R_HDPO(θ)]
```

**Theorem 3:** When the critic judges the chosen response as closer to human support (`s_ϕ(y⁺) < s_ϕ(y⁻)`), both the DPO term and the HDPO regularizer push the policy in the same direction — they are **distributionally aligned**.

### 2.4 Weighting: Scarce Buckets Get More Gradient Mass

HDPO up-weights preference pairs from scarce buckets, rare languages, and critic-hard pairs — directly reallocating gradient mass where distributional alignment is most needed (Theorem 5).

---

## 3. Example: Why Bucket-SFT Outperforms DPO

**Scenario:** User asks "What's the weather like today?" in two interaction contexts.

### Setting A: `open_chat` (casual multi-turn conversation)

Human distribution:
> "ugh it's absolutely miserable out there lol. raining nonstop since morning. staying in with coffee today 😴"

### Setting B: `qa_search` (single-turn factual query)

Human distribution:
> "The current temperature is 12°C with overcast skies and intermittent rain. Humidity at 87%."

### What Each Method Generates

| Method | open_chat | qa_search |
|---|---|---|
| **BaseLM** | "The weather conditions vary by location. Please specify your city." | "The weather conditions vary by location. Please specify your city." |
| **Full-SFT** | "Today's weather is cloudy with rain expected." | "Today's weather is cloudy with rain expected." |
| **DPO** | "I'd be happy to help! Based on current conditions, the weather today is cloudy with a high chance of precipitation." | "I'd be happy to help! Based on current conditions, the weather today is cloudy with a high chance of precipitation." |
| **Bucket-SFT** | "oh man it's pretty gloomy here too. rain since 6am, thinking about just ordering in and watching netflix all day tbh 😅" | "Current: 12°C, overcast, rain. Humidity 87%. Wind NE 15 km/h." |

DPO learned that politeness markers ("I'd be happy to help!") and hedging are preferred by raters, but applies this **identically across all contexts** — the polite assistant voice colonizes every interaction. Bucket-SFT preserves distributional diversity: casual + personal for chat, terse + factual for QA.

---

## 4. Key Results

### 4.1 Main Benchmark (Table 2 — 4 models × 2 languages)

Aggregate scores (↑): combines QA-F1, BNG-macro, G-MAUVE, and NUF.

| Model | Method | English Agg | Chinese Agg |
|---|---|---|---|
| Qwen2.5-1.5B | BaseLM | 0.235 | 0.293 |
| | Full-SFT | 0.371 | 0.529 |
| | DPO | 0.527 | 0.184 |
| | **Bucket-SFT** | **0.643** | **0.684** |
| | **HDPO** | **0.639** | **0.694** |
| Qwen2.5-3B | Full-SFT | 0.137 | 0.498 |
| | DPO | 0.381 | 0.148 |
| | **Bucket-SFT** | **0.688** | **0.669** |
| Llama-3.2-3B | Full-SFT | 0.550 | 0.433 |
| | DPO | 0.288 | 0.088 |
| | **Bucket-SFT** | **0.566** | **0.636** |
| | **HDPO** | **0.604** | **0.665** |
| Gemma-2-2B | Full-SFT | 0.308 | 0.352 |
| | DPO | 0.310 | 0.081 |
| | **Bucket-SFT** | **0.387** | **0.324** |
| | **HDPO** | **0.541** | **0.555** |

**Key finding:** Bucket-SFT is the most reliable component — it consistently improves over both Full-SFT and DPO. DPO alone often **worsens** distributional alignment (Qwen2.5-1.5B Chinese: DPO = 0.184 vs. Full-SFT = 0.529).

### 4.2 Distributional Metrics

| Model | Method | BNG-macro(↓) | G-MAUVE(↑) | NUF(↑) |
|---|---|---|---|---|
| Qwen2.5-1.5B en | Full-SFT | 5.012 | 0.947 | 0.547 |
| | DPO | 1.259 | 0.847 | 0.580 |
| | **Bucket-SFT** | **0.427** | **0.939** | **0.585** |
| | **HDPO** | **0.465** | **0.846** | **0.623** |
| Qwen2.5-1.5B zh | Full-SFT | 0.917 | 0.918 | 0.458 |
| | DPO | 16.056 | 0.511 | 0.163 |
| | **Bucket-SFT** | **0.346** | **0.968** | **0.636** |
| | **HDPO** | **0.375** | **0.851** | **0.832** |

Bucket-SFT dramatically reduces BNG-macro (Bucketed Naturalness Gap — how far generated feature distributions drift from human distributions within each bucket). HDPO further improves NUF (Naturalness-Utility Frontier), especially in Chinese (0.832).

### 4.3 LLM-as-a-Judge (Table 3)

Across two judges (Qwen3-8B, Qwen2.5-7B), Bucket-SFT matches or exceeds Full-SFT on utility while improving Conditional Naturalness and Distribution Faithfulness. Qwen2.5-1.5B English: Bucket-SFT improves Overall from 76.9 (Full-SFT) to 78.9, with Distribution Faithfulness rising from 71.7 to 73.3.

### 4.4 Human-Likeness & AI Detection (Table 4)

Three AI-text detectors (Binoculars, Fast-DetectGPT, RADAR) confirm PolyAlign models produce text harder to distinguish from human writing. Joint score (harmonic mean of benchmark quality and human-likeness): Qwen2.5-1.5B English Bucket-SFT = 57.0 vs. Full-SFT/DPO at 45.7/46.0.

### 4.5 Human Evaluation (Table 8)

Chinese human evaluation (1–5 scale, n=20): Bucket-SFT = **4.85/5** average, vs. Full-SFT (4.55), DPO (3.20), BaseLM (3.74).

### 4.6 Why DPO Alone Fails

DPO often degrades distributional metrics compared to Full-SFT, especially in Chinese (Qwen2.5-1.5B zh: DPO BNG = 16.056 vs. Full-SFT = 0.917). DPO's preference signal isn't grounded in conditional human distributions — it optimizes toward whatever raters label "better," drifting away from situation-appropriate human response patterns. HDPO fixes this by grounding preferences in bucket-specific critic scores.

---

## 5. The Two-Stage Design Pattern

1. **Bucket-SFT first** — anchors the model to bucket-specific human response regions via macro bucket-risk optimization. Most reliable component across all settings.

2. **HDPO second** — refines alignment using critic-derived distributional preferences. More sensitive to critic quality, but can further improve naturalness when well-calibrated.

---

## 6. Evaluation Metrics

| Metric | What it measures |
|---|---|
| **QA-F1** | Token-level task utility |
| **BNG-macro** (↓) | How far generated feature distributions drift from human distributions within each bucket (Bucket Naturalness Gap) |
| **G-MAUVE** (↑) | Distributional similarity between generated and human responses |
| **NUF** (↑) | Naturalness-Utility Frontier — hypervolume of Pareto frontier in (utility, naturalness) space |
| **Joint** (↑) | Harmonic mean of benchmark quality and AI-detector human-likeness score |

---

## 7. The Bucket Structure

The fine-grained buckets are defined from `(language, track, family, length_bin)`. The 14 datasets span:

| Situation | English Datasets | Chinese Datasets |
|---|---|---|
| assistant_like | Dolly | COIG-CQIA |
| longform_qa | ELI5 | HC3-Chinese |
| open_chat | DailyDialog | OASST2-zh |
| qa_search | MS MARCO, CoQA, SQuAD v2, Natural Questions | CMRC2018, DRCD, DuReader |
| task_dialogue | MultiWOZ | — (not instantiated in Chinese) |

The exact number of buckets isn't explicitly stated — it's the cross product of metadata dimensions after filtering buckets with <20 held-out examples.

---

## 8. Limitations

1. Bilingual only (English + Chinese); broader multilingual evaluation is future work
2. Compact models only (1.5B–3B); larger-scale validation needed
3. Bucket construction depends on linguistic feature profiles — richer metadata would help
4. No task_dialogue bucket in Chinese (partially crossed language–situation design)
5. Human evaluation is initial (20 samples per method)

---

## 9. Key Takeaways

1. **The alignment question needs reframing.** From "how do we make a model better on average?" to "how do we make a model produce the right kind of answer for the right kind of interaction?"

2. **Bucket-SFT alone provides substantial, reliable gains.** Just rebalancing SFT across interaction buckets outperforms both standard Full-SFT and DPO, without any additional model complexity or preference data.

3. **DPO without distributional grounding is harmful.** Global preference signals collapse diverse interaction styles into a single generic assistant voice — DPO often underperforms even Full-SFT on distributional metrics.

4. **HDPO adds distributional awareness to preference optimization.** The critic-based regularizer grounds DPO's preference signal in bucket-specific human support, preventing drift away from situation-appropriate responses.
