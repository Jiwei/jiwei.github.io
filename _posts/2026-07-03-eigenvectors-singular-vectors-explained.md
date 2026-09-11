---
layout: post
title: "Eigenvectors, Singular Vectors, and the Edge Spectrum"
date: 2026-07-03 11:16:54 +0800
categories: paper-notes
---

---

## 1. Eigenvectors and Eigenvalues

For a square matrix **A** (n × n), an **eigenvector v** is a non-zero vector that, when multiplied by A, **doesn't change direction** — it only gets scaled:

```
A·v = λ·v
```

The scalar **λ** is the **eigenvalue** associated with that eigenvector. It tells you how much the vector is stretched or compressed:

- λ > 1: vector stretches
- 0 < λ < 1: vector compresses
- λ = 0: vector collapses to zero (A is singular in that direction)
- λ < 0: vector flips direction and scales

### Geometric Intuition

If you think of A as a transformation of space (rotation, scaling, shearing), eigenvectors are the **special directions** that remain on their own line after the transformation. Everything else gets twisted; eigenvectors just get scaled.

### The Spectral Decomposition

If A is symmetric (or diagonalizable), you can decompose it in terms of its eigenvectors and eigenvalues:

```
A = V · Λ · V^T
```

where:
- **V** = matrix whose columns are the eigenvectors (orthogonal for symmetric A)
- **Λ** = diagonal matrix of eigenvalues λ₁, λ₂, ..., λ_n

Each eigenvalue-eigenvector pair (λ_i, v_i) represents an **independent mode** of the transformation. The transformation A can be understood as: project onto eigenvector directions → scale by eigenvalues → project back.

---

## 2. Eigenvectors vs. Singular Vectors

| | Eigen decomposition | Singular Value Decomposition (SVD) |
|---|---|---|
| **Works on** | Square matrices only (n × n) | Any rectangular matrix (m × n) |
| **Equation** | `A·v = λ·v` | `A = U · Σ · V^T` |
| **What you get** | Eigenvectors `v` + eigenvalues `λ` | Left singular vectors `U`, singular values `Σ`, right singular vectors `V` |

### The Core Difference

An **eigenvector** `v` of `A` satisfies `A·v = λ·v` — applying A to `v` gives you back `v` (just scaled). This only makes sense if `A` is square, because `v` must have the same dimension going in and coming out.

For a **rectangular** matrix (like the unembedding matrix `W_U`: vocab_size × hidden_dim), you can't ask "what vector stays on its own line after multiplication" because the input space and output space have different dimensions. A hidden state vector (d-dim) goes in, a vocabulary logit vector (V-dim) comes out. They live in different spaces.

So instead, SVD asks: can I find an **orthonormal basis** for the input space (`V`) and an orthonormal basis for the output space (`U`) such that `A` simply scales along corresponding pairs?

```
A = U · Σ · V^T
```

Each pair `(u_i, v_i)` forms a channel: input component along direction `v_i` gets scaled by `σ_i` and mapped to output direction `u_i`.

### The Relationship

The right singular vectors `V` are the **eigenvectors of A^T·A**:

```
(A^T·A) · v_i = σ_i² · v_i
```

And the left singular vectors `U` are the **eigenvectors of A·A^T**:

```
(A·A^T) · u_i = σ_i² · u_i
```

`A^T·A` (d × d) and `A·A^T` (V × V) are both square, so eigen decomposition applies. The singular values `σ_i` are the square roots of those eigenvalues.

**Singular vectors ARE eigenvectors — just of the squared Gram matrices `A^T·A` and `A·A^T`, rather than of A itself.** This is why SVD works for rectangular matrices: you convert the problem into two separate square eigen problems.

### The Signal Channel Analogy

A useful mental model: think of a matrix as a signal channel. The singular value is the **gain** — how strongly that channel transmits. High gain = dominant modes (the "loud" signals). Low gain = weak modes. But "weak" doesn't mean unimportant — the secondary edge spectrum in `W_U` has small singular values yet actively encodes the frequency bias that pollutes embeddings. Counterintuitively, the smallest singular components can have disproportionate semantic impact.

---

## 3. Why the Edge Spectrum Amplifies High-Frequency Tokens

In the EmbedFilter paper, both the **largest** and **smallest** singular vectors amplify high-frequency tokens — but through **different mechanisms**:

### Large Singular Values: Forward-Direction Amplification

The forward mapping is `logits = h · W_U^T`. The component of `h` along right singular vector `v_i` gets scaled by `σ_i` before reaching the vocabulary:

```
contribution to logits from direction v_i ∝ σ_i
```

Large `σ_i` → large gain. The directions with the largest singular values are the model's **"loudest" channels** — they dominate the output. During pretraining, the model learns that high-frequency tokens are the safest default prediction in most contexts (they're always somewhat relevant), so these dominant channels naturally encode them. Filtering out these large-σ directions **quiets the default-frequency signal**.

### Small Singular Values: Reverse-Direction Amplification (Pseudo-Inverse)

This is the less obvious mechanism. The pseudo-inverse `W_U^+` maps in the **opposite direction** — from vocabulary logits back to hidden states. Its singular values are `1/σ_i`:

```
W_U^+ = V · Σ^+ · U^T     where Σ^+ has entries 1/σ_i
```

So the **smallest** σ_i in `W_U` become the **largest** gains in `W_U^+`. When the paper reverse-engineers the average token:

```
ĥ = log(p̂) · W_U^+
```

The resulting `ĥ` is **dominated by the small singular vector directions** of `W_U`, because the pseudo-inverse amplifies them by `1/σ_i`. The average token vector lives mostly in this low-σ subspace. Filtering out these directions therefore **removes the centroid toward which the average token pulls all embeddings**.

### The Two-Edge Picture

| Edge | Gain in `W_U` (forward) | Gain in `W_U^+` (reverse) | How it encodes high-freq tokens |
|---|---|---|---|
| Largest σ | High | Low | Forward: dominant prediction channels for common tokens |
| Smallest σ | Low | High | Reverse: the average token's hidden state lives here, pulling all embeddings toward the centroid |

The **bulk** (middle of the spectrum) is where neither forward nor reverse gain dominates — these directions are balanced, encoding genuine semantic variation rather than frequency artifacts.

### Evidence from the EmbedFilter Paper

The ablation in Table 5 supports the asymmetry: filtering only the **secondary** subspace (smallest σ, score: **67.74**) gives a much bigger gain than filtering only the **dominant** subspace (largest σ, score: **60.34**, which is actually **worse** than the 63.04 baseline). The small-σ edge is the bigger culprit — the centroid removal via pseudo-inverse amplification has a larger effect on cleaning up the embedding space than the forward-channel attenuation.

---

## 4. Why Use SVD vs. Eigen Decomposition

- **Eigen decomposition** — when the matrix represents a transformation of a space onto itself (covariance matrices, dynamical systems, diffusion operators)
- **SVD** — when the matrix maps between different spaces (any linear layer, embedding/unembedding matrices, data matrices) or when you need the best low-rank approximation

In the EmbedFilter paper, `W_U` maps from hidden space (d dims) to vocabulary space (V dims) — two different spaces — so SVD is the natural tool.
