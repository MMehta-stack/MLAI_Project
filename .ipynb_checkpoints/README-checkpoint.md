# ML AI Capstone Project - Black Box Optimisation (BBO)

## Project Overview

This capstone project builds and iteratively refines an optimisation model within a simulated black-box environment. The internal workings of the functions are hidden — only inputs and their corresponding outputs are observable. The focus is on making evidence-based improvements to the optimisation strategy across successive rounds, without access to exact evaluation metrics during the process.

Understanding how to optimise a black-box use case helps in building strong intuition for decision-making under uncertainty, balancing exploration and exploitation, and iteratively refining models, which is a key skill for a data scientist when faced with incomplete information. It reinforces probabilistic thinking, disciplined experimentation, and model / strategy refinement as evidence accumulates.

---

## Problem Statement

The goal is to find the input configuration that maximises the output across eight synthetic black-box functions (F1–F8). Every function is a maximisation problem. The internal structure of each function is unknown — the only information available is the set of inputs tried so far and the outputs they returned.

---

## Data — Inputs and Outputs

Eight synthetic functions of increasing dimensionality are used:

| Function | Input dimensions | Input format |
|----------|-----------------|--------------|
| F1, F2   | 2D              | x1-x2 |
| F3       | 3D              | x1-x2-x3 |
| F4, F5   | 4D              | x1-x2-x3-x4 |
| F6       | 5D              | x1-x2-x3-x4-x5 |
| F7       | 6D              | x1-x2-x3-x4-x5-x6 |
| F8       | 8D              | x1-x2-x3-x4-x5-x6-x7-x8 |

Each input value begins with 0 and is specified to six decimal places (e.g. `0.123456-0.654321` for a 2D function). All outputs are 1D scalar values.

At each iteration, the new query point and its corresponding output are appended to the respective function's data file and used in the subsequent round.

---
## Objective

Find the input vector that produces the highest possible output for each of the eight functions, using as few queries as possible. Each query has a cost, so every new data point must be chosen to maximise the information it contributes about where the optimum might lie. 

The project works in an iterative manner, with every iteration providing new query points to generate outputs.

---

## Overall Approach

The core framework is a **Gaussian Process (GP) surrogate model** combined with an **Upper Confidence Bound (UCB) acquisition function**. The GP builds a probabilistic map of the landscape — estimating both the expected output and the uncertainty at every point in the input space. UCB then guides query selection by balancing:

- **Exploration** — querying regions the model is uncertain about
- **Exploitation** — querying regions the model believes are already promising

Three structural layers sit on top of this foundation:
1. **Automatic kernel selection** via cross-validation (RBF, Matérn-2.5, Matérn-1.5)
2. **Cluster-aware acquisition** — silhouette-gated KMeans clustering directs search toward the most promising observation groups
3. **PCA-analogy enhancements** — orient search along the axes of maximum information

---

## Model and Hyperparameter optimisation - Iteration History

### BBO1–5 — Baseline GP with EI and UCB
Initial query points were selected using Expected Improvement (EI), which balances exploitation of high-predicted regions with exploration of uncertain ones. A consensus-based approach was also tested — selecting points where both EI and UCB agreed, indicating high-confidence regions where the surrogate model's mean and uncertainty were aligned. Thompson Sampling (TS) was assessed alongside UCB to encourage broader exploration in diverse, uncertain regions.

**Outcome:** Established that UCB with `kappa=4.0` was consistently reliable across functions. Provided baseline coverage of the input space for all eight functions.

---

### BBO6, 7 and 10 — GP with CNN Augmentation and Optuna HPO
A CNN gradient predictor and residual corrector were added on the assumption that more model complexity would improve fit. Optuna hyperparameter optimisation was applied to tune GP noise, kernel parameters and acquisition weights.

**Outcome:** HPO improved the cross-validated GP score by less than 0.001 across all eight functions. The CNN stack consumed more than ten times the runtime of a plain GP without measurable gain in output quality. Both components were identified for removal in later stages.

---

### BBO8 and 9 — LLM-Based Approach
An LLM-based strategy was explored in parallel, using language model reasoning to suggest candidate query points targeting high-uncertainty and high-potential regions of each function's landscape.

**Outcome:** Generated plausible candidates but lacked the structured uncertainty quantification of a GP surrogate. 

---

### BBO11 — Simplified GP with Silhouette-Gated Clustering
Optuna HPO and the CNN stack were removed based on the evidence from prior rounds. Silhouette-gated KMeans clustering was introduced — clustering is only activated when observations form genuinely separable groups (silhouette score ≥ 0.22). Gap injection between promising clusters was added to direct queries toward under-sampled regions.

**Outcome:** Faster pipeline with cleaner acquisition surfaces. F5 produced its strongest result — a query placed in a gap between two moderate clusters returned an output of 8,662 against a previous best of 4,414, nearly doubling the known best.

---

### BBO12 — PCA-Analogy Enhancements (Final Version used for Week 12 and Week 13)
Five enhancements inspired by PCA principles were added:

| Enhancement | Description |
|-------------|-------------|
| **E1** Uncertainty gradient direction | Post-L-BFGS-B step toward steepest GP variance increase |
| **E2** ARD length-scale extraction | Dimension-importance weights from fitted kernel |
| **E3** PCA pre-projection | Projects inputs to top PCs before clustering for d ≥ 5 |
| **E4** PCA-stratified candidates | 30% of candidates drawn along leading PCs of observations |
| **E5** Cluster-local PCA gap direction | Gap midpoints projected onto joint first PC of bridging clusters |

**Outcome:** F5 confirmed 96% improvement over initial best. F8 clustering became viable through PCA pre-projection. ARD weight extraction revealed F2 as effectively 1-dimensional (d0 carries 100% of sensitivity).

---

## Key Results

| Fn | Dim | n | Kernel | Silhouette | ARD top dim | Best output |
|----|-----|---|--------|------------|-------------|-------------|
| F1 | 2D  | 22 | Mat15 | 0.495 | d1 (89%) | 0.0000 |
| F2 | 2D  | 22 | Mat25 | 0.456 | d0 (100%) | 0.6112 |
| F3 | 3D  | 27 | RBF   | 0.323 | d2 (86%) | −0.0264 |
| F4 | 4D  | 42 | Mat25 | 0.246 | d2 (26%) | 0.3648 |
| F5 | 4D  | 32 | Mat25 | 0.329 | d3 (32%) | 8662.48 |
| F6 | 5D  | 32 | Mat15 | 0.245 | d1 (26%) | −0.5904 |
| F7 | 6D  | 42 | Mat15 | 0.259 | d4 (40%) | 1.4618 |
| F8 | 8D  | 52 | Mat25 | — (skipped) | d2 (21%) | 9.9758 |

---

## Key Lessons

- **Simplicity supported by evidence outperforms complexity built on assumption.** HPO and CNN were removed because the data showed no gain — not because they seemed wrong in theory.
- **Conditional activation works better than fixed rules.** Clustering, gradient steps and PCA projection are each gated on whether the data justifies them.
- **The gap between promising clusters is often the highest-leverage query location** in landscapes where the optimum is spatially localised (demonstrated by F5).
- **ARD kernels automatically identify irrelevant dimensions** — F2's second dimension received maximum length scale, functionally reducing it to a 1D problem without manual intervention.

---

## Technical Stack

```
Python 3.10
scikit-learn    — GaussianProcessRegressor, KMeans, PCA, silhouette_score
scipy           — minimize (L-BFGS-B)
numpy           — array operations
```

---

## Repository Structure

```
├── src       # Final optimisation notebook
├── data      # Initial data files and weekly new query point submissions (inputs) with results (outputs)
├── docs      # Model card, Data Sheet and presentation summary
├── version   # Weekly enhanced version of BBO notebooks
└── README.md # This file
```

---

## Links to Model Card and Data Sheet

- [BBO Model Card](https://github.com/MMehta-stack/MLAI_Project/blob/main/docs/BBO%20Model%20Card.md)
- [BBO Data Sheet](https://github.com/MMehta-stack/MLAI_Project/blob/main/docs/BBO%20Data%20Sheet.md)

