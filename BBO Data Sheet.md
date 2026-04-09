# ML AI Capstone Project

**BBO Observation Dataset: F1–F8**
Initial input–output observations for eight black-box benchmark functions

**Motivation**

This dataset was created to support the BBO black-box optimisation pipeline, which uses Gaussian Process surrogates with CNN augmentation and Optuna-driven hyperparameter optimisation to identify the next most informative query point for each function. The eight datasets (F1–F8) represent the initial observations available to the optimiser at the start of a new round — they are the empirical foundation from which the surrogate model is built.

The primary task is sequential, single-objective optimisation over continuous bounded domains. Each dataset provides the starting conditions for a single optimisation round: given these observations, the BBO pipeline recommends exactly one new point to evaluate next. The datasets span a range of dimensionalities (2D to 8D) and output characteristics, allowing the pipeline to be tested and benchmarked across varied problem structures.

**Composition**

Format and structure
Each function produces two NumPy binary files: f{n}initial_inputs.npy (shape: observations × dimensions) and f{n}initial_outputs.npy (shape: observations).
All input values are bounded within [0, 1]^d. Outputs are unbounded and vary significantly in scale across functions.
No missing values, NaN entries, or non-numeric records were found in any file.

Identified gaps and data quality concerns
F1 - Total y range of 0.004; 94% of observations within top 10% of range. Skewness of −4.01. Near-zero variance makes GP normalisation and gradient estimation unreliable.
F3 - Negative skew of −2.27 with one outlier. Most observations clustered near the lower output bound.
F4 - y spans −55.8 to +0.4 with heavy negative skew. Only 7% of observations in the high-value region, leaving the GP with very few examples near the optimum.
F5 - y spans 0.1 to 4,414 — a factor of ~40,000. A single high outlier dominates. Positive skew of 2.26. Log-transformation is strongly recommended before fitting.
F6 - Lowest n/d ratio at 5.8 (29 observations in 5D). High-dimensional space significantly undersampled relative to other functions.
F7 - Minimum pairwise distance of 0.013, indicating at least one near-redundant query. Limited informational gain from these two observations.

F2 and F7 show no significant statistical concerns. F8 has the best overall coverage given its dimensionality. (F8 - Best spacing overall; good 8D coverage)

**Collection process**

Query generation strategy
The initial observations for each function were collected using a sequential black-box querying process across multiple optimisation rounds. 
Across rounds BBO1 through BBO8, one new query point was selected and evaluated per round per function, with the selection strategy evolving as follows:
Early rounds (BBO1–BBO3): random or quasi-random sampling to establish initial coverage of the input domain.
Mid rounds (BBO4–BBO6): GP-guided acquisition using UCB, with CNN gradient amortisation and residual correction progressively introduced.
Later rounds (BBO7-BBO10): full HPO-enhanced pipeline with Optuna-driven tuning of kernel, acquisition, and CNN architecture parameters. BBO8 and BBO9 were further optimised and generated using LLM.

Observation count by function and timeframe
Observation counts reflect the number of completed rounds prior to the current round: F1 and F2 have 19 observations (2D problems, lighter sampling needed); F3 has 24 (3D); F4, F7 have 39 (4D and 6D respectively); F5, F6 have 29 (4D and 5D); F8 has 49 (8D, densest sampling to support higher dimensionality).
The dataset was accumulated incrementally across sequential optimisation rounds. Each round adds one observation per function. The exact calendar time frame is not recorded in the data files; the observation count serves as the temporal index.

**Preprocessing and Uses**
Transformations applied
Input normalisation: all inputs are already in [0, 1]^d — no further scaling is applied by the pipeline.
Output normalisation: the GP normalises outputs internally to zero mean and unit variance during fitting (normalize_y=True). Raw y values are stored and passed to the pipeline unnormalised.
No transformations have been applied to the stored .npy files themselves — they contain raw observed values as returned by the black-box function evaluations.

Intended Use
Serving as the initial training set for the BBO surrogate model at the start of each new optimisation round.

**Distribution and Maintenance**
The datasets are stored as NumPy binary files (.npy) and distributed alongside the BBO notebook (BBO10_HPO_enhanced.ipynb). They are loaded directly by the run_all_functions() entry point using the naming convention f{n}initial_inputs.npy and f{n}initial_outputs.npy in the working directory.
No explicit licence is attached to the data files. As the outputs of a computational optimisation process rather than collected human data, standard academic reuse norms apply. If used in published research, the BBO pipeline and the sequential collection methodology should be cited and described.
The datasets contain no personal data, sensitive information, or human-derived measurements. All values are the results of evaluating mathematical or computational black-box functions. There are no privacy, consent, or ethical review obligations associated with these files.
Maintenance
Each new optimisation round appends one observation per function to the respective input/output arrays. The files are updated in-place by saving the augmented arrays back to the same filenames.
No automated validation or integrity checking is currently applied when files are updated. It is recommended to add a checksum or observation-count log to detect accidental overwrites.
The maintainer is the practitioner running the BBO pipeline. There is no central repository or versioned release process.