---
layout: post
title: "Elasticity Meets Continuous-Time: Map-Centric Dense 3D LiDAR SLAM"
date: 2026-07-03 16:10:04 +0800
categories: paper-notes
---

**Paper:** [arXiv:2008.02274](https://arxiv.org/abs/2008.02274)
**Authors:** Chanoh Park, Peyman Moghadam, Jason Williams, Soohwan Kim, Sridha Sridharan, Clinton Fookes  
**Affiliations:** CSIRO, Queensland University of Technology, Sun Moon University  
**Date:** August 2020

---

## 1. Core Problem: Bring Map-Centric SLAM to Real 3D LiDAR Systems

The paper starts from a strong systems-level observation:

- **Trajectory-centric SLAM** is the dominant paradigm for LiDAR SLAM: estimate a trajectory, then optimize it globally (pose graph / batch optimization), and build the map from that trajectory.
- **Map-centric SLAM** is an alternative: directly maintain a fused map and perform loop closure by **deforming the map elastically**, instead of re-optimizing a full trajectory.

Map-centric SLAM has two attractive properties:

1. **Dense fusion is natural** — redundant scans can be merged directly into a high-quality map.
2. **Loop closure cost does not grow over time the same way** as trajectory optimization — deformation is more space-dependent than time-dependent.

But previous map-centric methods were mostly designed for RGB-D or specific LiDAR setups, and had several weaknesses when applied to real 3D LiDAR systems:

- poor handling of **LiDAR motion distortion**
- weak support for **asynchronous multi-modal fusion** (LiDAR + IMU + camera)
- dependence on **projective sensor models** that do not work well for 360° spinning LiDAR
- fragile **loop closure** behavior: once a wrong fusion happens, the map can be hard to recover

The paper's goal is to preserve the benefits of map-centric SLAM while making it actually work for modern 3D LiDAR systems with realistic sensor payloads.

---

## 2. Main Idea

The core contribution is to combine:

- **map-centric SLAM**
- with a **local continuous-time trajectory representation**

This gives the system a way to:

1. locally model the platform motion continuously over the LiDAR scan,
2. correct motion distortion,
3. fuse scans into dense surfel maps,
4. keep the global map consistent through elastic deformation.

So the paper is really about merging **continuous-time local trajectory estimation** with **global map deformation**.

---

## 3. High-Level Architecture

The system has three main components:

### 1. Local mapping
- takes LiDAR, IMU, and visual measurements
- estimates a **local continuous-time trajectory**
- corrects LiDAR motion distortion
- builds a local sparse and dense surfel map

### 2. Global mapping
- fuses local maps into:
  - a **sparse surfel map** (for robust registration / deformation)
  - a **dense surfel map** (for high-quality reconstruction)

### 3. Loop closure
- detects loop closure candidates using visual place recognition
- estimates metric misalignment with a robust geometric localization method
- deforms the global map elastically to maintain consistency

A key design choice is that the system keeps **two surfel maps**:
- a **sparse multi-resolution surfel map** for localization / optimization
- a **dense surfel map** for reconstruction and rendering

---

## 4. Contribution 1: Local Continuous-Time Trajectory for a Map-Centric System

### Why continuous-time?
For spinning LiDARs, a single scan is not captured instantaneously — the platform is moving while the scan is being acquired. This creates **motion distortion**.

A continuous-time trajectory allows every LiDAR point to be associated with the robot pose at the exact time it was measured.

### How they represent it
They use a **local continuous-time trajectory** `T(τ)`, with:
- rotational component `R(τ)`
- translational component `t(τ)`

For interpolation between discrete poses:

```
T(τ) = T_k · exp(α[ξ]×)
```

They then optimize trajectory corrections using **B-spline control points**, but apply corrections in **SE(3)** rather than the older SO(3)+R³ update scheme. This is more accurate for their local-window setup.

### Key local constraints
They define four types of local constraints:

1. **Surfel-to-surfel local consistency**  
   Matches local surfels formed at different timestamps within the same local scan window.

2. **Surfel-to-map-prior constraint**  
   Forces the local trajectory to align with the existing global map, which is essential for a map-centric formulation.

3. **IMU acceleration constraint**

4. **IMU angular velocity constraint**

Together these allow the local trajectory to be estimated while staying compatible with the already-built global map.

### Result
In simulation, their local trajectory optimization is:
- more accurate than CT-SLAM [3]
- more accurate than spline approximation methods [9]
- achieved with significantly fewer optimized states than the approximation model

This local CT formulation is what lets the map-centric system handle:
- asynchronous fusion
- severe LiDAR motion distortion
- non-projective sensor geometries

---

## 5. Contribution 2: Surface-Resolution-Preserving Surfel Matching

A major difficulty in dense LiDAR surfel fusion is **data association**:
- nearest-neighbor matching can oversmooth and overfuse
- projective association works only for RGB-D / limited FoV LiDAR
- voxel grids control resolution but discretize the world and lose smoothness

The paper proposes a **two-stage surfel matching strategy** that preserves desired surface resolution without requiring a voxel grid.

### Matching logic
For each local dense surfel `ϕ_l`, it first finds candidate surfels in the global map via octree nearest neighbor search.

Then it applies two tests:

1. **Resolutional distance r**  
   Compared in Euclidean space against a threshold `θ_r`  
   → this preserves surface resolution

2. **Depth distance d**  
   Compared in 1D Mahalanobis space along the surfel normal direction  
   → this lets the matcher search deeper along the beam direction while respecting uncertainty

This matching strategy preserves desired map density while still being robust to LiDAR beam-direction uncertainty.

### Why it matters
This is one of the paper's key engineering insights:
- non-redundant dense fusion
- no dependence on a pinhole/projective sensor model
- works for 360° spinning LiDAR

---

## 6. Contribution 3: Wishart-Based Probabilistic Surfel Fusion

Instead of treating a surfel as just a point + normal, they model a surfel as a **Gaussian distribution of points** with:
- mean (centroid)
- covariance / scatter
- uncertainty in the centroid
- orientation inferred from the covariance structure

### Core probabilistic model
They use a **normal-inverse-Wishart-style formulation** (practically inspired by random matrix methods for noisy observations) to recursively fuse surfels.

The important idea is:
- LiDAR points are noisy, especially along the beam direction
- the uncertainty is anisotropic and depends on:
  - range noise
  - incidence angle
  - beam geometry

So each surfel update tracks not just a centroid but also a covariance/scatter structure.

### Why Wishart helps
This lets them:
- estimate both **position** and **orientation** jointly
- propagate surfel uncertainty correctly
- deal better with **degeneracy** in surfel geometry
- extract normals via eigen decomposition of the updated scatter matrix

The paper claims this is the **first use of a Wishart model for surfel fusion**.

### Practical outcome
Their fused surfel maps are:
- **denser**
- **less noisy**
- still non-redundant

On planar patches, they report the proposed method is up to **3× less noisy** than the baseline CT-SLAM point cloud.

---

## 7. Contribution 4: Robust Metric Loop Closure for Map-Centric SLAM

Loop closure is especially risky in a map-centric system:
- in trajectory-centric systems, bad loop closures can sometimes be corrected by later trajectory optimization
- in map-centric systems, if you fuse wrongly, you may irreversibly damage the map

So the paper spends real effort making loop closure robust.

### Their solution
They use:
- **visual place recognition** for loop candidate detection
- then a **sequential metric localization** procedure using LiDAR geometry to estimate the misalignment

The misalignment estimation combines:

1. **surfel-based point-to-plane constraints**
2. **3D feature-based point-to-point constraints** (e.g., FPFH/SHOT-like features)

These are complementary:
- surfel constraints are good for geometric refinement
- feature constraints give coarse alignment when surfel ICP alone would get stuck in local minima

### Sequential fusion
Instead of one large global registration step, they estimate misalignment **sequentially at multiple places** and fuse those pose estimates on SE(3) using a Bayesian fusion framework.

This gives:
- cross-validation across locations
- uncertainty-aware stopping
- reduced failure rate

### Quantitative result
Compared to Sparse ICP, Open3D global registration, and SHOT-based registration, their method is the most stable:
- translation error stays within about **6 cm**
- rotation error stays around **0.004 rad**

even under hard initial perturbations where others fail.

---

## 8. Contribution 5: Elastic Global Map Deformation

For global consistency, they use a **deformation graph**, inspired by ElasticFusion.

### How it works
- graph nodes are sampled from the sparse surfel map
- each node has:
  - position
  - local rotation
  - local translation
- deformation is blended over neighboring nodes

A point `p_i` is deformed as a weighted combination of neighboring node transforms:

```
p_i' = Σ_j w_j(p_i) · φ_j(p_i)
```

where each node applies a local rigid transform, and weights depend on distance.

They also deform:
- surfel normals
- centroid uncertainty
- scatter/covariance

So the deformation affects not just geometry but also the uncertainty representation.

### Active vs inactive maps
To reduce destructive fusion before loop closure, they divide the map into:
- **active map**: recent area, where fusion is allowed
- **inactive map**: older area

New surfels are fused only in the active region. If overlap and misalignment with inactive areas become large enough, deformation is triggered.

This is an important stabilization mechanism for map-centric fusion.

---

## 9. Experimental Results

The system is tested on:
- multiple sensor payloads
- both hand-held and robot-mounted setups
- indoor, outdoor, structured, and unstructured scenes
- single-beam spinning LiDAR
- multi-beam LiDAR
- visual + IMU + LiDAR combinations

### 9.1 Trajectory estimation
Compared to globally optimized CT-SLAM trajectories:
- deformation-based recovered trajectories differ by about **0.17–0.55 m RMSE**
- error correlates with map size
- local structure is well preserved, even when global offsets grow

### 9.2 Dense map quality
On planar patches:
- position and normal errors are substantially lower than the CT-SLAM baseline
- fusion improves significantly after about **4 observations**
- errors can be reduced to roughly **±10 mm** and **0.2 rad**

### 9.3 Loop closure localization
Their sequential metric localization is:
- more stable than sparse ICP
- more reliable than Open3D global registration and SHOT pipelines
- robust under poor initialization

---

## 10. Strengths of the Paper

### 1. Strong systems integration
This paper is not about just one trick — it is a carefully engineered full SLAM system where:
- continuous-time estimation
- map-centric fusion
- uncertainty-aware surfels
- loop closure localization
- elastic map deformation
all fit together coherently.

### 2. Solves a real mismatch
Map-centric dense fusion is attractive, but old map-centric approaches were too RGB-D / pinhole-camera-centric. This paper makes the paradigm much more realistic for LiDAR robotics.

### 3. Good probabilistic treatment
The Wishart/random-matrix-based surfel fusion is a serious probabilistic improvement over many simpler fusion heuristics.

### 4. Good robustness focus
The sequential loop closure design is clearly motivated by the destructive failure mode of map-centric systems.

---

## 11. Limitations

The paper is also very honest about its weaknesses.

### 1. Map distortion still grows with scale
At around **60 m scale**, map distortion can reach **10 cm**, and likely grows further at larger scale.

### 2. Partial observation weakness
Map-centric fusion can struggle when revisits are only partial and don't trigger local loop closure strongly enough.

### 3. Surface resolution trade-off
If surfel resolution is too coarse:
- small objects are ignored

If resolution is too fine:
- computational cost rises sharply
- reconstruction quality doesn't improve much

They empirically found **0.02 m** surfel resolution to be a good compromise.

### 4. Repeated non-Gaussian noise
Their probabilistic fusion still struggles with repeated non-Gaussian artifacts, e.g. mixed-pixel LiDAR noise.

---

## 12. Key Takeaway

This paper's core message is:

> **Map-centric dense SLAM can be made practical for real 3D LiDAR systems if you combine it with local continuous-time trajectory estimation, uncertainty-aware surfel fusion, and robust sequential metric loop closure.**

More concretely:

- **continuous-time local optimization** fixes LiDAR motion distortion and asynchronous fusion,
- **Wishart-based surfel fusion** gives non-redundant dense maps with principled uncertainty handling,
- **sequential metric localization + elastic deformation** makes loop closure much safer in a map-centric setting.

So the paper is a strong argument that **map-centric SLAM is a viable alternative to trajectory-centric SLAM for dense LiDAR mapping**, provided you solve the right engineering and probabilistic problems.
