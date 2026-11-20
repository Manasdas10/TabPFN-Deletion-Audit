"""
fit_cache_audit.py
==================
Experiment: Does TabPFN's KV cache change what a deleted row leaves behind?
Author   : [Your Name]
Thesis   : M.Tech CSE (Data Science), Integrated Programme, Oct 2027
Date     : November 2026

REQUIRES: tabpfn >= 9.1.0  (fit_mode="fit_with_cache" does not exist in 2.x)
          Python 3.9-3.12  (tabpfn does not support 3.14 yet)

BACKGROUND
----------
TabPFN >= 9.1 offers fit_mode="fit_with_cache": during fit(), it precomputes
the training set's attention state (a "KV cache") and reuses it at predict
time. That cache is computed FROM the training rows. So if a row is deleted
and the pipeline is rebuilt, does the cache-built model "remember" the row
more than the normal mode? That is what this script measures.
(docs: https://docs.priorlabs.ai/capabilities/kv-cache)

THREE MEASUREMENTS
------------------
1. SANITY CHECK
   Normal fit vs cache fit on IDENTICAL data should produce (near-)identical
   predictions. If they do not, the fit mode itself adds noise, and we must
   know that before interpreting anything else.

2. DELETION FOOTPRINT (per fit mode)
   Fit on the FULL training set, record test predictions P_full.
   For each deleted row: refit on the reduced data (fresh scaler, clean
   rebuild) and record P_del. Footprint = MAD(P_full, P_del).
   A bigger footprint means the deleted row influenced predictions more,
   i.e. it leaves a larger trace while present. We compare the footprint
   distribution of cache mode vs normal mode.

3. DELETED-ROW PROBE
   Right after each clean refit, we predict the deleted row itself and
   record the probability the model assigns to its TRUE class.
   High residual confidence on a row the model supposedly never saw is a
   memorization signal (compare IclMem / TabPATE style probes).

Also recorded: refit wall-clock time per mode (the "unlearning cost",
per Muresanu et al.'s recommendation to always report it).

WHAT THIS SCRIPT DOES NOT DO
  - MIA with shadow models (needs more compute; future work).
  - Claim anything about tabpfn internals beyond prediction behaviour.
"""

# ===========================================================
# SECTION 0: Imports
# ===========================================================

import datetime
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

# ===========================================================
# SECTION 1: Configuration — change settings here, not in code
# ===========================================================

RANDOM_SEED: int = 42          # Record in thesis Methods section
N_DELETION_TRIALS: int = 20    # Pilot; use 50+ with 3 seeds for the thesis
TEST_FRACTION: float = 0.25
TABPFN_N_ESTIMATORS: int = 4   # Keep low for speed on CPU/Colab T4

# Device: use the GPU if PyTorch can see one, else CPU
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# The two fit modes we compare
MODE_NORMAL = "auto"            # Standard fit (recomputes attention per predict)
MODE_CACHE  = "fit_with_cache"  # Precomputes KV cache during fit()

# ===========================================================
# SECTION 2: Reproducibility helpers
# ===========================================================

def set_global_seed(seed: int) -> None:
    """Lock all random number generators to a fixed seed."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def hash_dataframe(df: pd.DataFrame) -> str:
    """Short MD5 fingerprint of a DataFrame. Record it in the lab notebook."""
    df_bytes: bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(df_bytes).hexdigest()


def print_section(title: str) -> None:
    """Print a visible separator — makes the console output readable."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ===========================================================
# SECTION 3: Metrics (same definitions as the main audit)
# ===========================================================

def max_abs_diff(probs_A: np.ndarray, probs_B: np.ndarray) -> float:
    """Worst-case disagreement between two probability matrices."""
    return float(np.max(np.abs(probs_A - probs_B)))


def mean_l2_distance(probs_A: np.ndarray, probs_B: np.ndarray) -> float:
    """Average per-row Euclidean distance between probability vectors."""
    return float(np.mean(np.linalg.norm(probs_A - probs_B, axis=1)))

# ===========================================================
# SECTION 4: Model builder
# ===========================================================

def build_tabpfn(fit_mode: str) -> TabPFNClassifier:
    """
    Build a fresh TabPFN classifier in the given fit mode.

    fit_mode="auto"            -> normal behaviour
    fit_mode="fit_with_cache"  -> precomputes the KV cache during fit()
    A fresh instance per fit guarantees no state leaks between trials.
    """
    return TabPFNClassifier(
        n_estimators=TABPFN_N_ESTIMATORS,
        device=DEVICE,
        seed=RANDOM_SEED,
        fit_mode=fit_mode,
    )

# ===========================================================
# SECTION 5: Main experiment
# ===========================================================

def run_cache_audit() -> None:
    set_global_seed(RANDOM_SEED)
    print_section("SETUP")

    # --- Load Iris (same dataset and split as the main audit) ---
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    dataset_hash = hash_dataframe(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    import sklearn, tabpfn
    print(f"Dataset       : Iris (sklearn built-in)")
    print(f"Dataset hash  : {dataset_hash}  <- must match the main audit")
    print(f"Train/Test    : {len(X_train)}/{len(X_test)}")
    print(f"Seed          : {RANDOM_SEED}  |  Trials: {N_DELETION_TRIALS}")
    print(f"Device        : {DEVICE}")
    print(f"tabpfn        : {tabpfn.__version__}  (must be >= 9.1.0)")
    print(f"sklearn       : {sklearn.__version__}")

    # -------------------------------------------------------
    # MEASUREMENT 1: sanity check — fit modes on identical data
    # -------------------------------------------------------
    print_section("MEASUREMENT 1: SANITY CHECK (same data, both modes)")

    full_scaler = StandardScaler().fit(X_train)
    X_train_sc = full_scaler.transform(X_train)
    X_test_sc  = full_scaler.transform(X_test)

    t0 = time.perf_counter()
    m_normal_full = build_tabpfn(MODE_NORMAL)
    m_normal_full.fit(X_train_sc, y_train)
    t_fit_normal_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    m_cache_full = build_tabpfn(MODE_CACHE)
    m_cache_full.fit(X_train_sc, y_train)
    t_fit_cache_full = time.perf_counter() - t0

    P_full_normal = m_normal_full.predict_proba(X_test_sc)
    P_full_cache  = m_cache_full.predict_proba(X_test_sc)

    sanity_mad = max_abs_diff(P_full_normal, P_full_cache)
    print(f"MAD(normal fit, cache fit) on identical data = {sanity_mad:.6f}")
    print(f"Fit time: normal {t_fit_normal_full:.2f}s | cache {t_fit_cache_full:.2f}s")
    if sanity_mad > 1e-6:
        print("NOTE: modes disagree slightly on identical data. This is")
        print("numerical/implementation noise - subtract it mentally when")
        print("reading the footprints below.")
    else:
        print("OK: modes are behaviourally identical on identical data.")

    # -------------------------------------------------------
    # MEASUREMENTS 2 + 3: deletion footprint and deleted-row probe
    # -------------------------------------------------------
    print_section(f"MEASUREMENTS 2+3: {N_DELETION_TRIALS} DELETION TRIALS")

    rng = np.random.default_rng(seed=RANDOM_SEED)
    delete_indices = rng.choice(
        X_train.index.tolist(), size=N_DELETION_TRIALS, replace=False
    )
    print(f"Rows selected: {delete_indices.tolist()}")
    print("(Same seed and selection logic as the main audit.)\n")

    footprint_normal: list[float] = []   # MAD(P_full, P_deleted) per mode
    footprint_cache:  list[float] = []
    probe_normal:     list[float] = []   # P(true class) of deleted row after refit
    probe_cache:      list[float] = []
    refit_t_normal:   list[float] = []   # unlearning cost per mode
    refit_t_cache:    list[float] = []

    for trial_num, del_idx in enumerate(delete_indices, start=1):

        # Keep a copy of the row we are about to delete (for the probe)
        deleted_row = X_train.iloc[[del_idx]]   # 1-row DataFrame
        deleted_label = int(y_train[del_idx])

        # Clean rebuild data: row removed, FRESH scaler on the rest
        X_minus = X_train.drop(index=del_idx).reset_index(drop=True)
        y_minus = np.delete(y_train, del_idx)

        fresh_sc = StandardScaler().fit(X_minus)
        X_minus_sc = fresh_sc.transform(X_minus)
        X_test_sc_fresh = fresh_sc.transform(X_test)
        deleted_row_sc = fresh_sc.transform(deleted_row)

        # --- Normal mode refit ---
        t0 = time.perf_counter()
        m_n = build_tabpfn(MODE_NORMAL)
        m_n.fit(X_minus_sc, y_minus)
        refit_t_normal.append(time.perf_counter() - t0)
        P_del_normal = m_n.predict_proba(X_test_sc_fresh)
        probe_normal.append(
            float(m_n.predict_proba(deleted_row_sc)[0, deleted_label])
        )

        # --- Cache mode refit ---
        t0 = time.perf_counter()
        m_c = build_tabpfn(MODE_CACHE)
        m_c.fit(X_minus_sc, y_minus)
        refit_t_cache.append(time.perf_counter() - t0)
        P_del_cache = m_c.predict_proba(X_test_sc_fresh)
        probe_cache.append(
            float(m_c.predict_proba(deleted_row_sc)[0, deleted_label])
        )

        # --- Footprints vs the full-data fits (Measurement 2) ---
        # NOTE: full-data fits used full_scaler; deleted fits use fresh_sc.
        # The tiny scaler difference is part of the row's total footprint
        # (same convention as Path B in the main audit).
        fp_n = max_abs_diff(P_full_normal, P_del_normal)
        fp_c = max_abs_diff(P_full_cache,  P_del_cache)
        footprint_normal.append(fp_n)
        footprint_cache.append(fp_c)

        print(f"  Trial {trial_num:2d} | row={del_idx:3d} | "
              f"footprint N={fp_n:.5f} C={fp_c:.5f} | "
              f"probe N={probe_normal[-1]:.3f} C={probe_cache[-1]:.3f} | "
              f"refit N={refit_t_normal[-1]:.2f}s C={refit_t_cache[-1]:.2f}s")

    # -------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------
    print_section("AGGREGATE RESULTS")

    fp_n_arr = np.array(footprint_normal)
    fp_c_arr = np.array(footprint_cache)
    pb_n_arr = np.array(probe_normal)
    pb_c_arr = np.array(probe_cache)

    print("Deletion footprint (MAD vs full-data fit):")
    print(f"  Normal mode : mean={fp_n_arr.mean():.5f} std={fp_n_arr.std():.5f} max={fp_n_arr.max():.5f}")
    print(f"  Cache  mode : mean={fp_c_arr.mean():.5f} std={fp_c_arr.std():.5f} max={fp_c_arr.max():.5f}")

    print("\nDeleted-row probe (P(true class) after clean refit):")
    print(f"  Normal mode : mean={pb_n_arr.mean():.5f} std={pb_n_arr.std():.5f}")
    print(f"  Cache  mode : mean={pb_c_arr.mean():.5f} std={pb_c_arr.std():.5f}")
    print(f"  (Chance level on Iris = 1/3 = 0.333)")

    print("\nUnlearning cost (clean refit wall-clock):")
    print(f"  Normal mode : mean={np.mean(refit_t_normal):.2f}s")
    print(f"  Cache  mode : mean={np.mean(refit_t_cache):.2f}s")

    # -------------------------------------------------------
    # Statistical tests (paired, one-sided)
    # -------------------------------------------------------
    print_section("STATISTICAL TESTS")

    diff = fp_c_arr - fp_n_arr
    if np.all(diff == 0.0):
        print("Footprints identical across modes — Wilcoxon skipped.")
        p_modes = 1.0
    else:
        stat, p_modes = stats.wilcoxon(diff, alternative="greater")
        print(f"Wilcoxon (H1: cache footprint > normal footprint)")
        print(f"  Statistic = {stat:.4f}   p = {p_modes:.4f}")
        if p_modes < 0.05:
            print("  -> Cache mode leaves a SIGNIFICANTLY larger footprint.")
        else:
            print("  -> No significant difference between fit modes.")

    if np.all(fp_c_arr == 0.0):
        print("Cache footprint all zero — vs-zero test skipped.")
        p_cache_zero = 1.0
    else:
        stat2, p_cache_zero = stats.wilcoxon(fp_c_arr, alternative="greater")
        print(f"\nWilcoxon (H1: cache footprint > 0)")
        print(f"  Statistic = {stat2:.4f}   p = {p_cache_zero:.4f}")

    # -------------------------------------------------------
    # Save results JSON
    # -------------------------------------------------------
    results = {
        "run_metadata": {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "script": "fit_cache_audit.py",
            "dataset": "Iris (sklearn built-in)",
            "dataset_hash": dataset_hash,
            "random_seed": RANDOM_SEED,
            "n_deletion_trials": N_DELETION_TRIALS,
            "deleted_row_indices": delete_indices.tolist(),
            "tabpfn_version": tabpfn.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "device": DEVICE,
            "tabpfn_n_estimators": TABPFN_N_ESTIMATORS,
        },
        "sanity_check": {"mad_normal_vs_cache_same_data": sanity_mad},
        "per_trial": {
            "footprint_normal": footprint_normal,
            "footprint_cache": footprint_cache,
            "probe_normal_true_class_prob": probe_normal,
            "probe_cache_true_class_prob": probe_cache,
            "refit_time_normal_s": refit_t_normal,
            "refit_time_cache_s": refit_t_cache,
        },
        "aggregate": {
            "footprint_normal_mean": float(fp_n_arr.mean()),
            "footprint_cache_mean": float(fp_c_arr.mean()),
            "probe_normal_mean": float(pb_n_arr.mean()),
            "probe_cache_mean": float(pb_c_arr.mean()),
            "wilcoxon_cache_vs_normal_p": float(p_modes),
            "wilcoxon_cache_vs_zero_p": float(p_cache_zero),
        },
        "caveats": [
            "Footprint includes the fresh-scaler difference (Path B convention).",
            "Probe is exploratory; high P(true class) is a signal, not proof of memorization.",
            "Pilot run: 20 trials, 1 seed, Iris. Thesis needs 50+ x 3 seeds, real datasets.",
            "Novelty phrasing: 'we did not find a published study doing this exact audit'.",
            "Differentiate from tabcustody (github.com/jrachid/tabcustody).",
        ],
    }
    with open("fit_cache_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to fit_cache_results.json — commit it to results/.")

    # -------------------------------------------------------
    # Lab-notebook-ready summary
    # -------------------------------------------------------
    print_section("COPY INTO results/lab_notebook.txt")
    print(f"Date           : {datetime.date.today().isoformat()}")
    print(f"Script         : fit_cache_audit.py")
    print(f"tabpfn         : {tabpfn.__version__} | sklearn {sklearn.__version__} | device {DEVICE}")
    print(f"Dataset hash   : {dataset_hash} | seed {RANDOM_SEED} | trials {N_DELETION_TRIALS}")
    print(f"Sanity MAD     : {sanity_mad:.6f}")
    print(f"Footprint      : normal {fp_n_arr.mean():.5f} | cache {fp_c_arr.mean():.5f} (p={p_modes:.4f})")
    print(f"Probe P(true)  : normal {pb_n_arr.mean():.3f} | cache {pb_c_arr.mean():.3f} (chance 0.333)")
    print(f"Refit time     : normal {np.mean(refit_t_normal):.2f}s | cache {np.mean(refit_t_cache):.2f}s")


if __name__ == "__main__":
    run_cache_audit()
