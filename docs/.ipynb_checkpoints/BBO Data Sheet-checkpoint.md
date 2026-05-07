# ML AI Capstone Project : Black Box Optimisation - Data Sheet

## Overview

This datasheet was created to support the BBO black-box optimisation pipeline, which uses **Gaussian Process (GP) surrogate model** combined with an **Upper Confidence Bound (UCB) acquisition function**

The eight datasets (F1–F8) represent the initial observations available to the optimiser at the start of a new round — they are the empirical foundation from which the surrogate model is built.

The primary task is sequential, single-objective optimisation over continuous bounded domains. Each F1-F8 input dataset provides the starting conditions for a single optimisation round: given these observations, the BBO pipeline recommends exactly one new point to evaluate next. The datasets span a range of dimensionalities (2D to 8D) and output characteristics, allowing the pipeline to be tested and benchmarked across varied problem structures.

## Data — Inputs and Outputs

Eight synthetic functions of increasing dimensionality are used:

| Function | Initial data points | Input dimensions | Input format |
|----------|---------------------|-----------------|--------------|
| F1, F2   | 10                  |2D              | x1-x2 |
| F3       | 15                  |3D              | x1-x2-x3 |
| F4       | 30                  |4D              | x1-x2-x3-x4 |
| F5       | 20                  |4D              | x1-x2-x3-x4 |
| F6       | 20                  |5D              | x1-x2-x3-x4-x5 |
| F7       | 30                  |6D              | x1-x2-x3-x4-x5-x6 |
| F8       | 40                  |8D              | x1-x2-x3-x4-x5-x6-x7-x8 |

Each input value begins with 0 and is specified to six decimal places (e.g. `0.123456-0.654321` for a 2D function). All outputs are 1D scalar values.

At each iteration, the new query point and its corresponding output are appended to the respective function's data file and used in the subsequent round.


## Nature of the Data

### Initial Input and Output shape
F1  (10, 2) (10,)
F2  (10, 2) (10,)
F3  (15, 3) (15,)
F4  (30, 4) (30,)
F5  (20, 4) (20,)
F6  (20, 5) (20,)
F7  (30, 6) (30,)
F8  (40, 8) (40,)

### Dataset evolution over the weeks
The dataset was accumulated incrementally across sequential optimisation rounds. Each round adds one observation per function. The exact calendar time frame is not recorded in the data files; the observation count serves as the temporal index.

Final input and output shapes after 12 iterations is as below:
F1  (22, 2) (22,)
F2  (22, 2) (22,)
F3  (27, 3) (27,)
F4  (42, 4) (42,)
F5  (32, 4) (32,)
F6  (32, 5) (32,)
F7  (42, 6) (42,)
F8  (52, 8) (52,)

## Optimisation strategy

### Final strategy
The core framework is a **Gaussian Process (GP) surrogate model** combined with an **Upper Confidence Bound (UCB) acquisition function**. The GP builds a probabilistic map of the landscape — estimating both the expected output and the uncertainty at every point in the input space. UCB then guides query selection by balancing:

- **Exploration** — querying regions the model is uncertain about
- **Exploitation** — querying regions the model believes are already promising

Three structural layers sit on top of this foundation:
1. **Automatic kernel selection** via cross-validation (RBF, Matérn-2.5, Matérn-1.5)
2. **Cluster-aware acquisition** — silhouette-gated KMeans clustering directs search toward the most promising observation groups
3. **PCA-analogy enhancements** — orient search along the axes of maximum information

### Balancing exploration and exploitation
The UCB acquisition function controls the exploration-exploitation balance through its kappa parameter, fixed at 4.0. Empirical analysis showed that optimising kappa produced less than 0.003 improvement in leave-one-out rank score across F1–F8 — not worth the overhead of another optimisation loop. A fixed kappa = 4.0 provides sufficient exploration weight to avoid premature convergence while still directing queries toward high-mean regions.
Where clustering is active, the acquisition blend weight (cluster_acq_weight = 0.25) means 75% of the signal comes from GP uncertainty and 25% from cluster promise scores. This ensures the cluster signal informs but does not dominate — a deliberate safeguard after early versions over-weighted cluster promise and generated queries in already-dense regions.

### Key strategic decisions
**Removing the CNN components** 
Earlier versions included a CNN to help predict and correct the GP's output. The assumption was that a more complex model would produce better results. In practice it did not — the CNN struggled with functions whose outputs vary at very different scales, and the corrections it made were inconsistent. It also added ten times the runtime of the plain GP with no measurable improvement in result quality. Removing it made the pipeline faster and the acquisition recommendations more reliable.

**F5 query - Using what worked** 
BBO11's cluster-aware acquisition identified a gap between two groups of observations in F5's input space — a region that had not yet been sampled. Placing the next query in that gap, between two moderately promising clusters, returned an output of 8,662 against a previous maximum output of 4,414 — nearly doubling the known best result. The lesson is straightforward: when the best outcome is concentrated in a small region of the input space, the most valuable query is often in the unexplored area between promising groups, not near the known best point.


## Data handling and preprocessing

**Input normalisation:** all inputs are already in [0, 1]^d — no further scaling is applied by the pipeline.
**Output normalisation:** the GP normalises outputs internally to zero mean and unit variance during fitting (normalize_y=True). Raw y values are stored and passed to the pipeline unnormalised.
**No transformations** have been applied to the stored .npy files themselves — they contain raw observed values as returned by the black-box function evaluations.


## Weekly iteration, learnings and outcomes

### BBO1–5 — Baseline GP with EI and UCB
Initial query points were selected using Expected Improvement (EI), which balances exploitation of high-predicted regions with exploration of uncertain ones. A consensus-based approach was also tested — selecting points where both EI and UCB agreed, indicating high-confidence regions where the surrogate model's mean and uncertainty were aligned. Thompson Sampling (TS) was assessed alongside UCB to encourage broader exploration in diverse, uncertain regions.

**Outcome:** Established that UCB with `kappa=4.0` was consistently reliable across functions. Provided baseline coverage of the input space for all eight functions.

---

### BBO6, 7 and 10 — GP with CNN Augmentation and Optuna HPO
A CNN gradient predictor and residual corrector were added on the assumption that more model complexity would improve fit. Optuna hyperparameter optimisation was applied to tune GP noise, kernel parameters and acquisition weights.

**Outcome:** HPO improved the cross-validated GP score by less than 0.001 across all eight functions. The CNN stack consumed more than ten times the runtime of a plain GP without measurable gain in output quality. Both components were identified for removal.

---

### BBO8 and 9 — LLM-Based Approach
An LLM-based strategy was explored in parallel, using language model reasoning to suggest candidate query points targeting high-uncertainty and high-potential regions of each function's landscape.

**Outcome:** Generated plausible candidates but lacked the structured uncertainty quantification of a GP surrogate. Confirmed that a principled probabilistic model was more appropriate for this task.

---

### BBO11 — Simplified GP with Silhouette-Gated Clustering
Optuna HPO and the CNN stack were removed based on the evidence from prior rounds. Silhouette-gated KMeans clustering was introduced — clustering is only activated when observations form genuinely separable groups (silhouette score ≥ 0.22). Gap injection between promising clusters was added to direct queries toward under-sampled regions.

**Outcome:** Faster pipeline with cleaner acquisition surfaces. F5 produced its strongest result — a query placed in a gap between two moderate clusters returned an output of 8,662 against a previous best of 4,414, nearly doubling the known best.

---

### BBO12 — PCA-Analogy Enhancements (Current Version)
Five enhancements inspired by PCA principles were added:

| Enhancement | Description |
|-------------|-------------|
| **E1** Uncertainty gradient direction | Post-L-BFGS-B step toward steepest GP variance increase |
| **E2** ARD length-scale extraction | Dimension-importance weights from fitted kernel |
| **E3** PCA pre-projection | Projects inputs to top PCs before clustering for d ≥ 5 |
| **E4** PCA-stratified candidates | 30% of candidates drawn along leading PCs of observations |
| **E5** Cluster-local PCA gap direction | Gap midpoints projected onto joint first PC of bridging clusters |

**Outcome:** F5 confirmed 96% improvement over initial best. F8 clustering became viable through PCA pre-projection. ARD weight extraction revealed F2 as effectively 1-dimensional (d0 carries 100% of sensitivity).


## Performance and results

**Key Metrics and Results**

| Fn | Dim | n  | Silhouette        | ARD top   | |PC1-y| | Landscape notes                          |
|----|-----|----|-------------------|-----------|--------|-------------------------------------------|
| F1 | 2D  | 21 | 0.53 — Strong     | d1 (89%)  | 0.14   | Compact; 95% near best                    |
| F2 | 2D  | 21 | 0.47 — Strong     | d0 (100%) | 0.28   | Effectively 1-dimensional                 |
| F3 | 3D  | 26 | 0.36 — Moderate   | d2 (91%)  | 0.60   | Single axis drives output                 |
| F4 | 4D  | 41 | 0.27 — Moderate   | d2 (25%)  | 0.75   | Balanced; all dims matter                 |
| F5 | 4D  | 31 | 0.30 — Moderate   | d3 (34%)  | 0.58   | Sharp peak — 96% gain found               |
| F6 | 5D  | 31 | 0.21 — Skipped    | d1 (27%)  | 0.75   | Diffuse; pure UCB used                   |
| F7 | 6D  | 41 | 0.23 — Weak       | d4 (53%)  | 0.32   | Sparse; d4 is key signal                 |
| F8 | 8D  | 51 | 0.13 — Skipped    | d2 (20%)  | 0.93   | Isotropic — PCA pre-projection applied   |

**Most influential findings**
**F5** — This contains the most sharply localised peak across all eight functions.The BBO11 suggested query point returned an output of 8,662. F5 has only 3% of observations within 10% of the optimum, confirming a sharply localised peak that uniform sampling was unlikely to find.
**F8** — PC1 of F8’s input space correlates at −0.93 with the output — a single dominant direction. Yet KMeans silhouette is only 0.13 because the raw input variance is spread nearly uniformly across all 8 dimensions. The predictive structure only becomes visible after PCA pre-projection.
**F2** — The ARD kernel assigns a length scale of 20 (the upper bound) to dimension d1, functionally removing it. F2 is a 2D function that behaves as 1D. This is directly analogous to PCA discarding a near-zero eigenvalue — the GP learns it automatically.
Across the full set, functions divide into three regimes that warrant different search strategies. Low-dimensional with strong cluster structure (F1, F2, F3) favour tight exploitation. Moderate-dimensional with partial structure (F4, F5, F7) require a balanced exploration-exploitation blend. High-dimensional near-isotropic functions (F6, F8) rely primarily on GP uncertainty rather than any geometric structure.

## Ethical, practical and general considerations

### Practical and general considerations
Bayesian optimisation is increasingly central to applied machine learning, most visibly in neural architecture search and large-scale hyperparameter tuning where each evaluation is expensive. The GP-UCB framework used here is a direct precursor to the acquisition strategies now embedded in tools like Google Vizier. The shift from HPO-heavy to evidence-driven, lightweight pipelines mirrors a broader trend in the field: the most robust models tend to be those that correctly identify what the data justifies, not those with the most parameters. The PCA-analogy enhancements developed in BBO12 — particularly PCA pre-projection for high-dimensional spaces and ARD weight extraction — reflect ideas that underpin active learning and experimental design more broadly.

### Limitations and risks
The most significant limitation of the approach is that the validity of the recommendations depends heavily on the quality and number of the initial observations, and for functions where those observations are statistically problematic, the pipeline's outputs may be unreliable regardless of how well the modell currently performs or the selected approach.
