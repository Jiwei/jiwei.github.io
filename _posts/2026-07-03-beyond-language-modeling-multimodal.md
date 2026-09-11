---
layout: post
title: "Beyond Language Modeling: An Exploration of Multimodal Pretraining"
date: 2026-07-03 14:44:12 +0800
categories: paper-notes
---

**Paper:** [arXiv:2603.03276](https://arxiv.org/abs/2603.03276)
**Authors:** Shengbang Tong, David Fan, John Nguyen, Ellis Brown, Gaoyue Zhou, Shengyi Qian, Boyang Zheng, Théophane Vallaeys, Junlin Han, Rob Fergus, Naila Murray, Marjan Ghazvininejad, Mike Lewis, Nicolas Ballas, Amir Bar, Michael Rabbat, Jakob Verbeek, Luke Zettlemoyer, Koustuv Sinha, Yann LeCun, Saining Xie
**Affiliations:** FAIR (Meta), New York University
**Date:** March 2026
**Project:** beyond-llms.github.io

---

## 1. Core Problem: The Multimodal Pretraining Design Space Is Opaque

The foundation model era has been defined by language pretraining. But text is a **lossy compression of reality** — the visual world contains physics, geometry, and causality that language misses. Moreover, high-quality text is approaching exhaustion, while visual data is virtually unlimited.

The problem: almost all multimodal models today (LLaMA-V, Qwen-VL, etc.) initialize from **pretrained language models** and then bolt on vision. This confounds everything — you can't tell what was learned from multimodal training vs. inherited from language pretraining. The fundamental dynamics between vision and language remain poorly understood.

**This paper's approach:** Train everything **from scratch**, controlling one variable at a time, to isolate the factors that actually govern multimodal pretraining. No pretrained LLM initialization anywhere.

---

## 2. The Framework: Transfusion

The model uses the **Transfusion** architecture: a single decoder-only Transformer that does **next-token prediction** for text and **diffusion (flow matching)** for vision. Text gets autoregressive cross-entropy loss; image/video tokens get flow matching loss with `v-pred` or `x-pred` parameterization.

A **hybrid attention mask** handles the mixed sequence: text uses standard causal masking; visual tokens within the same image/frame attend bidirectionally to each other, but causally to everything before.

**Data sources:** text (DCLM), raw video (YouTube, Kinetics, SSv2 at 1 FPS), image-text pairs (MetaCLIP, Shutterstock), and action-conditioned navigation trajectories.

**Default model:** 2.3B params (1.5B active per token due to modality-specific FFNs), using SigLIP 2 So400M as the frozen vision encoder.

---

## 3. Insight 1: A Single Visual Representation (RAE) Suffices (Section 3)

The paper evaluates a spectrum of visual representations:
- **VAEs:** SD-VAE, FLUX.1 VAE — low-dimensional latents, traditionally used for generation
- **Semantic encoders:** SigLIP 2, DINOv2, WebSSL — high-dimensional latents, traditionally used for understanding
- **Raw pixels**

**Key finding:** Semantic encoders (especially SigLIP 2) using RAE (Representation Autoencoder) **outperform VAEs on BOTH visual understanding AND generation**. SigLIP 2 beats FLUX.1 on DPGBench, GenEval, AND VQA — while matching text perplexity. This challenges the widespread assumption that you need dual encoders (one for understanding, one for generation, as in Janus, BAGEL). A single high-dimensional semantic representation excels at both.

Raw pixels underperform on generation but are competitive on VQA, suggesting they remain a promising direction at larger scale.

Text perplexity is nearly identical across all visual representations — multimodal pretraining does not significantly affect language capability regardless of which vision encoder is used.

**Suggestion 1:** A single RAE-based encoder (e.g., SigLIP 2) simplifies the architecture by excelling at both visual understanding and generation.

---

## 4. Insight 2: Vision and Language Are Complementary, Not Competitive (Section 4)

### 4.1 Data Composition

**Vision data minimally impacts text.** Text + Video even **outperforms** the text-only baseline on DCLM perplexity — raw visual data is compatible with language modeling. The small text degradation observed with I/T data comes from **distributional shift in captions** (cosine distance from DCLM), not from vision itself.

**I/T data enables visual capabilities.** Image-text pairs are essential for both visual understanding and generation. But different I/T sources have complementary trade-offs: MetaCLIP improves VQA, Shutterstock (high-aesthetic) improves generation. Using both captures the strengths of each.

### 4.2 Synergy in Multimodal Pretraining

Adding text tokens to a fixed vision budget consistently improves generation quality (text-conditioned benchmarks like GenEval benefit from better language modeling).

More strikingly: supplementing 20B VQA tokens with general-purpose data (text, video, or I/T) outperforms training on **100B VQA tokens alone** — 5× less domain-specific data, better results. Even out-of-domain data such as unlabeled video improves VQA. General pretraining provides a stronger foundation than specialized scaling.

Multimodal pretraining consistently outperforms text-only pretraining on VQA after finetuning across all visual representations — semantic encoders like SigLIP 2 lead by a wide margin.

**Suggestion 2:** Train with multimodal data. Visual data does not degrade language modeling, and diverse pretraining yields synergy for downstream tasks.

---

## 5. Insight 3: World Modeling Emerges from Multimodal Pretraining (Section 5)

The paper extends to the **Navigation World Model (NWM)** setting — predicting future visual frames conditioned on context and actions. Unlike NWM which encodes actions as specialized vectors, they represent actions as **plain text tokens** (e.g., `"action: dx=+1.338, dy=-0.659"`).

### 5.1 Key Findings

- **General video data drives world modeling, not domain-specific navigation data.** Adding 50B pure video tokens outperforms doubling the NWM-specific data from 50B to 100B tokens. Even text and image-text data help. World modeling relies more on capabilities acquired from general multimodal pretraining than on domain-specific trajectories.

- **World modeling transfers with minimal in-domain data.** Performance saturates at **1%** NWM data — the core capability is mostly acquired from general pretraining. The model reaches competitive planning performance with as little as 1% in-domain data.

- **Zero-shot natural language control:** Because actions are text tokens, the model can be controlled with **free-form language** (e.g., "get out of the shadow!", "go on the road") — an emergent capability from multimodal pretraining, never explicitly trained. WASD-style controls and complex natural language commands both work.

**Suggestion 3:** Unified multimodal pretraining unlocks world modeling. Represent actions as text without architectural changes; capabilities emerge via general training with minimal domain-specific data.

---

## 6. Insight 4: MoE Enables Efficient Multimodal Scaling (Section 6)

### 6.1 Design Space

**Granularity (G):** Higher granularity (many small experts) is critical — increasing from G=1 to G=16 substantially improves both language and vision. Interestingly, modalities saturate at different granularities (vision at G=4, language at G=16), suggesting different capacity needs.

**Sparsity:** Both modalities benefit consistently from increased sparsity (more total experts at fixed active compute). Going from 32 to 1008 total experts (active ratio dropping from 50% to 1.6%) improves both text PPL and visual generation. For RAE (SigLIP 2), both text and diffusion loss continue improving with more experts; for VAE (FLUX.1), diffusion loss saturates — semantic representations benefit more from sparsity.

**Prediction target depends on representation:** For RAE (SigLIP 2), x-pred outperforms v-pred. For VAE, v-pred is better and x-pred causes instability at high granularity.

**Per-modality shared experts** outperform global shared experts, suggesting modalities have distinct capacity needs that benefit from dedicated computation.

### 6.2 Emergent Specialization

Analyzing a 13.5B MoE model (256 experts, G=16, SigLIP 2, x-pred):

- **Modality specialization forms naturally** without human priors. Most experts are text-focused, but later layers contain progressively more vision and multimodal experts — the model learns a **separate-then-integrate** processing strategy.

- **No timestep specialization:** Vision experts are time-invariant throughout the diffusion process (CV ≈ 0.15), unlike architectures that explicitly enforce timestep-specific experts.

- **Understanding and generation share experts:** Pearson correlation r ≥ 0.90 between expert selection for image understanding vs. generation. The model converges to a truly unified visual representation.

### 6.3 Stacking Design Choices

Starting from a Transfusion baseline and progressively stacking improvements:

| Step | Design Choice | PPL↓ | DPG↑ |
|---|---|---|---|
| Baseline | Transfusion (shared FFN) | 15.93 | 0.45 |
| +FFN | Modality-specific FFN | 15.13 | 0.47 |
| +Encoder | SigLIP 2 (vs. SD-VAE) | 15.06 | 0.57 |
| +Separation | MoE (vs. dense/MoT) | 12.49 | 0.63 |
| +Pred | x-pred (vs. v-pred) | — | 0.65 |

MoE outperforms hand-crafted separation (MoT, modality-specific FFNs) because learned routing adapts per token rather than following a fixed schema.

**Suggestion 4:** Use MoE in unified models — it outperforms hand-crafted separation strategies and naturally learns modality-specific specialization from data.

---

## 7. Insight 5: Scaling Asymmetry — Vision Is More Data-Hungry (Section 7)

Chinchilla-style IsoFLOP analysis for both modalities simultaneously:

### 7.1 Dense Models

| Modality | Parameter exponent (a) | Data exponent (b) | Nature |
|---|---|---|---|
| **Language** | 0.47 | 0.53 | Chinchilla-like (nearly balanced) |
| **Vision** | 0.37 | 0.63 | **Significantly more data-hungry** |

The ratio of required vision data to language data grows as `O(N^{0.57})`. From a 1B-parameter baseline, vision's relative data demand increases by **14× at 100B parameters** and **51× at 1T parameters**. This creates a practical dilemma: you cannot simultaneously optimize both modalities at scale.

### 7.2 MoE Models

MoE **halves the scaling asymmetry gap** (exponent gap from 0.10 to 0.05). Under MoE, the language data exponent shifts from 0.53 to 0.59 — moving toward vision's data-intensive regime. Sparsity provides a structural mechanism for harmonizing modalities with fundamentally different scaling behaviors.

### 7.3 Compute Efficiency

Dense Multimodal matches or exceeds unimodal baselines across all metrics. At 10^21 FLOPs, it reaches Notes PPL 23.7 vs. 23.8 (Text-Only) and DPG 0.622 vs. 0.598 (T2I-Only). MoE Multimodal closely tracks unimodal MoE baselines — at 10^21 FLOPs: DCLM PPL 12.3 vs. 12.0 (Text-Only) and FID 39.2 vs. 39.8 (T2I-Only). A single unified model can match unimodal performance on both modalities.

---

## 8. Semantic Encoders vs. VAEs

### VAE (Variational Autoencoder)

**Training:** Self-supervised reconstruction. Encode an image into a low-dimensional latent, then decode back to pixels. The loss is "how well can I reconstruct the original image?" plus a regularization term that keeps the latent distribution close to a Gaussian.

**What you get:** A **low-dimensional latent vector** (e.g., SD-VAE: 4 channels, 32×32 grid) that's optimized for **pixel-level fidelity**. Every bit of the latent space is pressured to carry information needed for reconstruction. Great for generation (small, efficient latents → fast diffusion), poor for semantic understanding (the latent doesn't naturally separate "dog" from "cat" as a concept).

**Key property:** Compression bottleneck forces the latent to be efficient, but there's no incentive to learn semantically meaningful structure.

### DINOv2 (Self-Supervised Semantic Encoder)

**Training:** Self-distillation with no labels. Two augmented views of the same image go through a student and teacher network. The student is trained to match the teacher's output, with the teacher being an exponential moving average of the student. The loss encourages the model to recognize that two different crops/views of the same image represent the same thing.

**What you get:** A **high-dimensional feature vector** (e.g., DINOv2-L: 1024 channels, 16×16 grid) where semantically similar images are close in the representation space. The features naturally cluster by object category, even though the model was never given a single label. You can literally run k-NN classification on DINOv2 features and get strong ImageNet accuracy.

**Key property:** No reconstruction objective. The model learns what stays invariant across views of the same object — which turns out to be semantic identity. It discards pixel-level details that don't help with invariance.

### The Core Trade-off

| | VAE (SD-VAE, FLUX.1) | Semantic Encoder (DINOv2, SigLIP 2) |
|---|---|---|
| **Training objective** | Reconstruct pixels | Invariance across views (self-distillation) |
| **Latent dimensionality** | Low (4–16 channels) | High (768–1024 channels) |
| **What's preserved** | Pixel-level details, textures, exact spatial layout | Semantic identity, object categories, high-level structure |
| **Generation quality** | Excellent (purpose-built for it) | Surprisingly competitive (via RAE) |
| **Understanding quality** | Poor (latent not semantically organized) | Excellent (features cluster by meaning) |
| **Analogy** | JPEG compression — keeps what the eye needs | Conceptual sketch — keeps what the mind needs |

### Where This Paper Lands

RAE (Representation Autoencoder) enables high-dimensional semantic latents (SigLIP 2) to be used for **both** understanding AND generation — matching or beating VAEs on image generation benchmarks while dominating on VQA. One encoder, both tasks. No dual-encoder setup needed.

---

## 9. Key Takeaways

1. **Modality competition is largely solvable.** It stems from (a) distributional shift in captions, not vision itself, and (b) architectural rigidity. Modality-specific FFNs and MoE largely resolve it.

2. **A single visual representation works.** RAE-based semantic encoders (SigLIP 2) excel at both understanding and generation — no need for dual encoders.

3. **Diverse data beats specialized data.** General multimodal pretraining provides a stronger foundation for VQA and world modeling than scaling domain-specific data alone.

4. **MoE naturally learns modality specialization** and harmonizes the scaling asymmetry between vision (data-hungry) and language (parameter-hungry).

5. **World modeling emerges from general pretraining.** The boundary between multimodal models and world models blurs as capabilities emerge from broad training with minimal in-domain data.
