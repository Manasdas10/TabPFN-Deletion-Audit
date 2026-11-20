# TabPFN Deletion-Consistency Audit

M.Tech thesis project (CSE, Data Science, Integrated Programme, expected Oct 2027).

**Core question:** when a training row is "deleted" from a TabPFN pipeline, does the
resulting predictor behave identically to a clean model that never saw that row?

TabPFN is an in-context tabular foundation model: there is no gradient training at
fit time, so "deleting" a row should be as simple as removing it from the context.
This project audits whether that is actually true in the surrounding pipeline
(preprocessing state, caching), and measures the cost of doing deletion correctly.

## The two deletion paths

| Path | Name | What happens |
| ---- | ---- | ------------ |
| A | Naive delete | Row removed, but the `StandardScaler` fitted on the full data is reused (stale state) |
| B | Clean rebuild | Row removed, scaler refit from scratch on the remaining data |

If A and B disagree in their predictions, the pipeline retains a trace of the
deleted row. A blind baseline (shuffled labels) separates real signal from
natural prediction variance.

A second experiment audits `fit_mode="fit_with_cache"` (tabpfn >= 9.1): the KV
cache is precomputed from the training rows, so we measure whether a deleted row
leaves a larger footprint when the cache is used, and probe the deleted row's
predicted confidence after deletion.

## Repository contents

| File | Purpose |
| ---- | ------- |
| `tabpfn_deletion_audit.py` | Local prototype: Path A vs Path B with a RandomForest stand-in (plumbing check only, not thesis evidence) |
| `tabpfn_deletion_audit_colab.ipynb` | Main experiment: real TabPFN on Colab (Path A vs B, blind baseline, plots, results JSON) |
| `fit_cache_audit.py` | KV-cache experiment: deletion footprint and deleted-row probe, normal fit vs `fit_with_cache` |
| `requirements.txt` | Pinned dependencies |
| `results/lab_notebook.txt` | Run log (every run gets a dated entry with versions, seeds, dataset hash) |

## Current status

- [x] Run 001: local plumbing check (RandomForest stand-in, Iris, 20 trials, seed 42)
- [ ] Run 002: real TabPFN pilot on Colab (Iris, 20 trials, blind baseline)
- [ ] Run 003: KV-cache (`fit_with_cache`) experiment
- [ ] Scale up: 50+ trials x 3 seeds (42, 123, 7) on 2-3 real tabular datasets
- [ ] MIA-style probes (TabPATE/IclMem methodology) if time allows

## How to run

**Local plumbing check** (no GPU, but needs Python 3.9-3.12; system Python 3.14
is too new for tabpfn, so use the venv):

```
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\python.exe tabpfn_deletion_audit.py
```

**Real TabPFN run** (Google Colab, free T4 GPU):

1. Upload `tabpfn_deletion_audit_colab.ipynb` to Colab.
2. Runtime -> Change runtime type -> T4 GPU.
3. Run all cells top to bottom. First `fit()` downloads ~100 MB of weights.
4. Copy the version block and results into `results/lab_notebook.txt` (Run 002 template is waiting).

**KV-cache experiment**: same Colab session, or locally with a 3.11 venv:

```
python fit_cache_audit.py
```

## Methodology rules (do not break these)

1. Never claim absolute novelty. The correct phrasing is: "we did not find a
   published study doing this exact audit."
2. Differentiate from [tabcustody](https://github.com/jrachid/tabcustody), which
   demos row erasure on TabPFN at the file level. Our edge is a systematic
   empirical audit with baselines and statistics.
3. Stand-in-model numbers (RandomForest) are plumbing checks, never evidence.
4. Only claim excess over the blind baseline, never raw A-vs-B differences alone.
5. A-vs-B with naive delete isolates the broken mechanism (stale scaler). It does
   not prove TabPFN internally memorises rows.
6. Pilot runs (20 trials, 1 seed) are not thesis-level evidence. Thesis claims
   need 50+ trials x 3 seeds on real datasets.
7. Record everything: versions, seed, dataset hash, wall-clock time (the
   unlearning cost), in the lab notebook for every run.

## References

- Wahdany et al., Dataset Inference for Tabular Foundation Models, arXiv:2610.24278
- Capano & Bohler, IclMem, arXiv:2606.31208
- Wahdany et al., TabPATE, arXiv:2606.31474
- Muresanu et al., ICML 2025 (exact unlearning definition and cost reporting)
- tabcustody: https://github.com/jrachid/tabcustody
- TabPFN KV cache docs: https://docs.priorlabs.ai/capabilities/kv-cache
