# ML AI Capstone Project : Black-box Optimisation - Model Card

**Gaussian Process (GP) surrogate model** combined with an **Upper Confidence Bound (UCB) acquisition function**.

---

## Model Description 

**Summary**

The core framework is a **Gaussian Process (GP) surrogate model** combined with an **Upper Confidence Bound (UCB) acquisition function**. The GP builds a probabilistic map of the landscape — estimating both the expected output and the uncertainty at every point in the input space. UCB then guides query selection by balancing:

- **Exploration** — querying regions the model is uncertain about
- **Exploitation** — querying regions the model believes are already promising

Three structural layers sit on top of this foundation:
1. **Automatic kernel selection** via cross-validation (RBF, Matérn-2.5, Matérn-1.5)
2. **Cluster-aware acquisition** — silhouette-gated KMeans clustering directs search toward the most promising observation groups
3. **PCA-analogy enhancements** — orient search along the axes of maximum information

---

**Data — Inputs and Outputs:**

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

**Model Architecture:**

**Model architecture as 
used by the final version - Simplified GP with Silhouette-Gated Clustering with PCA-Analogy Enhancements**

Silhouette-gated KMeans clustering where clustering is only activated when observations form genuinely separable groups (silhouette score ≥ 0.22). Gap injection between promising clusters was added to direct queries toward under-sampled regions.

Enhancements added based on PCA principles:

| Enhancement | Description |
|-------------|-------------|
| **E1** Uncertainty gradient direction | Post-L-BFGS-B step toward steepest GP variance increase |
| **E2** ARD length-scale extraction | Dimension-importance weights from fitted kernel |
| **E3** PCA pre-projection | Projects inputs to top PCs before clustering for d ≥ 5 |
| **E4** PCA-stratified candidates | 30% of candidates drawn along leading PCs of observations |
| **E5** Cluster-local PCA gap direction | Gap midpoints projected onto joint first PC of bridging clusters |

---

## Performance

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

**F5** confirmed 96% improvement over initial best. **F8** clustering became viable through PCA pre-projection. ARD weight extraction revealed **F2** as effectively 1-dimensional (d0 carries 100% of sensitivity).

---

## Assumptions and limitations

**Assumptions:**
the underlying function changes smoothly (no sudden jumps), stays within a fixed input range (between 0 and 1 for all variables), and isn’t too noisy (observations are mostly reliable).
RBF and Matérn kernels assume stationary correlation structure. Non-stationary functions (e.g., smoothness that varies across the domain) may be mis-specified.
it needs a reasonable amount of initial data to learn useful patterns—otherwise it falls back to a simpler prediction method.

**Limitations**
The most significant limitation of the approach is that the validity of the recommendations depends heavily on the quality and number of the initial observations, and for high dimensional functions where those observations are statistically problematic, the pipeline's outputs may be unreliable regardless of how well the model performs or the selected approach.

---

## Trade-offs

The most significant trade-off was between model complexity and reliability. Adding the CNN felt like a logical improvement but introduced instability across functions with very different output scales. Removing it was the right decision but cost several rounds. The exploration-exploitation balance was handled differently per function — strong cluster structure (F1, F2) justified earlier exploitation; sparse high-dimensional functions (F7, F8) required sustained exploration throughout. The hardest decision was setting the silhouette threshold at 0.22 — too high and useful clustering is discarded; too low and false structure misleads the acquisition. The fixed threshold worked well overall but a data-driven adaptive threshold would have been more principled.


