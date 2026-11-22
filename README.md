# TabPFN Deletion-Consistency Audit

Preliminary pipeline-consistency case study. This is not a new unlearning algorithm, a privacy guarantee, or an established novelty claim.

## Question

After deleting rows from a TabPFN context, do predictions differ when an external StandardScaler is reused versus rebuilt without those rows? What do these complete implementation paths cost?

| Path | Intervention |
| --- | --- |
| A | Remove rows, reuse the pre-deletion external scaler, fit a fresh classifier |
| B | Remove the same rows, rebuild the external scaler on retained rows, fit a fresh classifier |

The stale external scaler is deliberately constructed by this harness. This is not evidence of a defective TabPFN deletion API or a shortcut used in production. Prediction disagreement alone does not prove privacy leakage or internal memorization.

## Status, 22 November 2026

- Run 001: completed RandomForest plumbing check, not TabPFN evidence.
- Run 002: completed Iris pilot, 20 real-label deletion trials, 5 separate shuffled-label controls, seed 42. Local tabpfn 9.2.0 package, CPU, four estimators. See results/audit_results.json.
- Historical cache-mode comparison: completed on Iris; both modes use clean refits. This is not deletion from a retained cache. See results/fit_cache_results.json.
- Run 003: prepared, NOT executed. Planned matrix: Iris, Adult numeric features, Covertype; seeds 42/123/7; 50 single-row and 50 group-deletion trials per dataset/seed; 900 trials total. Adult/Covertype use bounded 256-row samples, not full-dataset benchmarks. See RUN003_PROTOCOL.md.

## What Run 002 supports

The mean maximum probability difference was 0.05553 between the stale and rebuilt paths. The separate shuffled-label mean was 0.01823; their descriptive contrast was 0.03730. The mean L2 distance was 0.00713. The clean-minus-stale mean path timing difference was about 0.05 seconds on this pilot setup.

The timing difference includes complete, separate model fit/prediction paths. It is NOT an isolated measurement of scaler fitting and is not a universal unlearning cost. The pilot did not record an identical-input repeatability baseline. The observed difference should be described as pipeline disagreement pending stronger component and repeatability controls.

## Interpretation corrections added 22 November 2026

Historical source comments and saved outputs may retain earlier interpretations. Raw results and the original executed code are preserved for provenance; these corrections supersede their interpretation, not their measured values.

1. Shuffled labels are a negative control for label-dependent sensitivity, not a numerical-noise floor. A near-zero real-minus-shuffled contrast does not rule out a stale-state effect. A positive contrast does not prove privacy leakage. Report raw A/B differences as well as the contrast.
2. High confidence on an unseen deleted row can be ordinary generalization. It is not a membership or memorization test.
3. Comparing clean refits across fit modes is not a retained-cache erasure test.
4. Whole-pipeline unlearning is established prior work. The novelty of this exact comparison is unverified. More datasets alone do not establish novelty.
5. The model package version does not identify the exact weight checkpoint. Record resolved checkpoint identity in future execution evidence before making checkpoint-specific claims.

## Files and reproduction

- tabpfn_deletion_audit.py: historical RandomForest plumbing prototype.
- tabpfn_deletion_audit_colab.ipynb: historical Run 002 code, preserved with a later correction notice.
- fit_cache_audit.py: historical clean-refit fit-mode comparison; interpret with the corrections above.
- run003.py and Run003_Battery.ipynb: planned bounded multi-dataset runner and notebook.
- RUN003_PROTOCOL.md: intervention, paired controls, conditional inference and completion rules.
- results/: historical numerical outputs and lab log, unchanged by the interpretation correction.

Use the pinned setup in Run003_Battery.ipynb for Run 003. The local package uses a hidden token to unlock licensed model downloads; it does not send local predictions to the hosted prediction API. Do not save credentials in notebooks, outputs or commits. Use free non-commercial research access and free compute only; stop if any step requires payment. Existing code preparation is not evidence of a completed run.

Run 003 records identical-input repeats for the first trial in each cell and separates component timings from total pipeline timings. Review the initial repeatability evidence before causal attribution or scaling; material disagreement requires investigation, not a claim that the scaler explains everything. No extra fits or larger matrix are implied by this documentation.

## Sources

- Iris loader used by Run 002: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- StandardScaler mechanics: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Earlier pipeline-wide unlearning: https://www.cs.columbia.edu/~junfeng/papers/unlearning/
- Overlapping TabPFN row-erasure/refit implementation: https://github.com/jrachid/tabcustody
- Local model download and authentication: https://docs.priorlabs.ai/models/accessing-model-weights
- Academic/non-commercial pricing: https://priorlabs.ai/pricing

Related literature requires its own source verification before citation; this repository does not certify absence of overlapping work.
