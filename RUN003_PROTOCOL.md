# Run 003 protocol: bounded multi-dataset deletion audit

## Question and scope

Does reusing an external StandardScaler fitted before deletion produce different TabPFN predictions from rebuilding that scaler after deletion, with all other model settings held fixed?

This is a controlled pipeline-state audit, not a new unlearning algorithm. It does not test deletion from pretraining weights, demonstrate privacy-law compliance, or establish internal memorization.

## Planned matrix

- Datasets: Iris; Adult version 2 from OpenML, numeric features only; Covertype from sklearn's loader, all 54 features.
- Seeds: 42, 123, 7.
- Deletions: one row, or ceil(5% of the training context) rows at once.
- 50 trials per dataset/seed/deletion-type cell, 18 cells and 900 paired deletion trials in total.
- Each trial runs stale versus fresh preprocessing on true labels and on a fixed shuffled-label negative control, with the same deleted row IDs in both.
- Free CPU scope: at most 256 total rows per dataset/seed, then a stratified 75/25 train/test split. Iris uses its full 150 rows. Adult and Covertype are small-subset experiments, not full-dataset performance benchmarks.
- Subsamples preserve approximate class proportions with a minimum of four rows per class to allow rare Covertype classes in both splits. Group deletions are sampled without replacement within a group and rejected if a class vanishes from either the real or shuffled-label retained context. Group sets are distinct within a cell but may overlap.

## Model and intervention

- tabpfn 9.2.0; CPU; four estimators; fit_mode='fit_preprocessors'; fixed random_state matching the seed; two Torch threads.
- Path A removes the requested rows and refits a fresh classifier using the pre-deletion external scaler.
- Path B removes the same rows, fits a fresh external scaler on retained rows, and fits a fresh classifier.
- Model construction and prediction are reseeded for each path. Execution order alternates by trial to reduce timing order bias.
- One full-data warmup per dataset/seed precedes measurement so initial weight downloads and first-use costs are not charged to a deletion.
- An identical-input clean-path repeat is recorded for the first trial in every cell. This directly checks repeatability; it is separate from the shuffled-label control.

## Outcomes and statistical reporting

- Per trial: maximum absolute probability difference (MAD), mean L2 distance, mean absolute difference, prediction-label disagreement, and fit/fit-plus-predict/pipeline wall times.
- Pipeline timings include external transforms and the fresh scaler fit for Path B. Shared row selection and file I/O are excluded. They are measurements of this implementation and hardware, not universal unlearning costs.
- Preserve the historical contrast, excess MAD = real-label MAD minus shuffled-label MAD. The shuffled-label experiment is a negative control for label-dependent sensitivity, not a numerical-noise floor. A nonpositive contrast does not prove no scaler-state effect, and a positive contrast does not prove privacy leakage.
- Report full raw distributions, means, quantiles and 95% percentile-bootstrap intervals using 3,000 resamples within each cell.
- Two-sided paired Wilcoxon signed-rank tests compare real and shuffled-label MAD on matched deletion sets. Effect sizes are paired rank-biserial correlation and a paired standardized mean difference when defined.
- Holm correction uses the 18 planned cells as the family; any not-yet-observed cells are conservatively treated as p=1 in interim summaries.
- These intervals and tests describe deletion-set variation conditional on fixed dataset samples and splits. Trials share training and test data; they are not independent population datasets. Three seeds are separate replications, not 150 independent datasets. Report seed-specific results rather than claiming population-level significance from pooled trials.

## Reproducibility and completion

The runner stores environment versions, script SHA-256, feature names, exact sample/train/test/deletion IDs, data and label hashes, per-trial JSONL checkpoints, and a regenerated summary after each completed trial. It resumes completed trial keys from the JSONL file. Only a COMPLETE.json marker with 900 completed trials establishes the full planned battery is finished. An incomplete subset must be labeled with its exact completed counts.

The API key is supplied only through a hidden notebook prompt and inherited process environment. Never save a key in source, outputs, notebooks, commits, or screenshots.

## Dataset-loader references

- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html

This document records the design, not results. Numerical claims require the actual Run 003 outputs.

## Measurement clarification, 22 November 2026

The runner now persists external scaler-fit and per-path transform times explicitly in each trial, in addition to complete pipeline times. This is additive recording only: no change to the 900-trial matrix or model fits. The old Run 002 timing difference cannot be decomposed retrospectively into scaler cost. Review the first planned identical-input repeat before causal attribution or scaling. Material disagreement requires investigation and a protocol decision; no additional repeat fits are authorized by this clarification.

Use free non-commercial research licensing and free compute only. Do not start paid API use, enroll in billing, or upgrade compute. A paid requirement blocks that route and needs a free alternative without silently changing models.
