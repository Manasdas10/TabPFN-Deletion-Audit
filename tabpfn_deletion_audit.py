"""
tabpfn_deletion_audit.py
========================
Prototype: Deletion-Consistency Audit of TabPFN (in-context tabular model)
Author   : [Your Name]
Thesis   : M.Tech CSE (Data Science), Integrated Programme, Oct 2027
Date     : November 2026

PURPOSE
-------
This script answers one core question:
  When a training row is "deleted" from a TabPFN pipeline, does the
  resulting predictor behave identically to a clean model that never
  saw that row?

TWO DELETION PATHS compared here:
  (A) NAIVE DELETE  — remove the row BUT keep the StandardScaler that
                       was fitted on the full dataset (stale scaler).
  (B) CLEAN REBUILD — remove the row AND refit the scaler from scratch
                       on the remaining data (truly fresh pipeline).

If (A) ≠ (B) in predictions, the pipeline still "remembers" the deleted
row through its stale preprocessing state.

WHAT THIS PROTOTYPE DOES:
  1. Loads a small dataset (Iris, 150 rows) and splits it.
  2. For each of 20 random "deleted" rows, runs both deletion paths.
  3. Compares the prediction probability vectors from each path.
  4. Reports: Max Absolute Difference, Mean L2 Distance, and Wilcoxon
     signed-rank test p-value across the 20 trials.

WHAT THIS PROTOTYPE DOES NOT DO (yet):
  - Use real TabPFN (uses RandomForest as a stand-in; see NOTE below).
  - Membership Inference Attack (MIA) probes.
  - fit_mode="fit_with_cache" audit.
  - Blind baseline (model that never saw any data).

NOTE ON THE STAND-IN MODEL
  Real TabPFN requires PyTorch (~2 GB download) and does not yet
  support Python 3.14. This prototype uses RandomForest as a plumbing
  check only. Output numbers from this script are NOT thesis evidence.
  Replace `USE_REAL_TABPFN = False` with True once you have installed
  TabPFN in a Python 3.11 environment (see setup guide).

HOW TO READ THIS CODE (for Java developers)
  - Python has no types by default; we add type hints (x: int) as comments.
  - `import X` = like `import com.example.X` in Java.
  - `def f(a, b):` = like `public void f(TypeA a, TypeB b)`.
  - Indentation (4 spaces) replaces curly braces {}.
  - `print(...)` = like `System.out.println(...)`.
"""

# ===========================================================
# SECTION 0: Imports — load all the libraries we need
# ===========================================================

import hashlib           # For hashing the dataset (reproducibility check)
import time              # For measuring wall-clock time (unlearning cost)
import warnings          # For suppressing noisy library warnings

import numpy as np       # Numerical arrays (like double[] in Java)
import pandas as pd      # DataFrames (like a 2D table with named columns)
from scipy import stats  # Statistical tests

# sklearn = scikit-learn, the standard Python ML library
from sklearn.datasets import load_iris          # Small 150-row dataset
from sklearn.ensemble import RandomForestClassifier  # Stand-in for TabPFN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")  # Hide sklearn deprecation noise

# ===========================================================
# SECTION 1: Configuration — change settings here, not in code
# ===========================================================

# Set to True once you have TabPFN installed in a Python 3.11 venv.
# When False, we use RandomForest as a plumbing stand-in.
USE_REAL_TABPFN: bool = False

# Random seed — record this in your thesis Methods section!
# Using a fixed seed means anyone can reproduce your run exactly.
RANDOM_SEED: int = 42

# How many rows to "delete" and test, one at a time.
N_DELETION_TRIALS: int = 20

# Fraction of data to hold out as test set (never touched during training).
TEST_FRACTION: float = 0.25

# ===========================================================
# SECTION 2: Reproducibility helpers
# ===========================================================

def set_global_seed(seed: int) -> None:
    """
    Lock all random number generators to a fixed seed.
    Think of this like setting a fixed starting point for a random
    number table — same seed → same sequence every run.
    """
    np.random.seed(seed)
    # Note: if USE_REAL_TABPFN=True, also set torch.manual_seed(seed)


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Compute a short fingerprint (MD5 hash) of the entire DataFrame.
    Record this hash in your lab notebook and thesis so a reader can
    verify they have the exact same data as you.

    Example output: 'a3f9b2...' (32 hex characters)
    """
    # Convert the DataFrame to bytes, then hash it
    df_bytes: bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(df_bytes).hexdigest()


def print_section(title: str) -> None:
    """Print a visible separator — makes the console output readable."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ===========================================================
# SECTION 3: Dataset loading and preparation
# ===========================================================

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load the Iris dataset and return it as a DataFrame.

    Returns:
        X        : DataFrame of features (sepal/petal length, width)
        y        : Array of class labels (0, 1, 2)
        feature_names : List of column names

    Why Iris? It is tiny (150 rows), well-known, public domain,
    and has 3 classes — good for testing prediction probabilities.
    Replace this with your real dataset later.
    """
    iris = load_iris()

    # Create a DataFrame (like a Java 2D array, but with named columns)
    X: pd.DataFrame = pd.DataFrame(
        iris.data,
        columns=iris.feature_names   # e.g. ["sepal length (cm)", ...]
    )
    y: np.ndarray = iris.target      # Integer class labels: 0, 1, 2

    return X, y, list(iris.feature_names)


# ===========================================================
# SECTION 4: The two deletion paths
# ===========================================================

def path_A_naive_delete(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    delete_idx: int,
    scaler_fitted_on_full: StandardScaler
) -> np.ndarray:
    """
    PATH A: NAIVE DELETE
    --------------------
    Simulates what happens when a practitioner removes a row but
    forgets to refit the preprocessing pipeline.

    Steps:
      1. Use the scaler that was fitted on the FULL training set
         (including the row being deleted). <- This is the "bug"!
      2. Remove the target row from X_train.
      3. Scale both train and test with the STALE scaler.
      4. Train and predict.

    The stale scaler's mean and std still encode the deleted row's
    values. This is the mechanism we are isolating.

    Args:
        X_train               : Full training features (before deletion)
        y_train               : Full training labels
        X_test                : Hold-out test set
        delete_idx            : Index of the row to delete (in X_train)
        scaler_fitted_on_full : StandardScaler fitted on the FULL X_train

    Returns:
        prob_matrix : shape (n_test, n_classes) — predicted probabilities
    """
    # Step 1: Remove the row (like ArrayList.remove(idx) in Java)
    X_minus = X_train.drop(index=delete_idx).reset_index(drop=True)
    y_minus = np.delete(y_train, X_train.index.get_loc(delete_idx))

    # Step 2: Scale using the STALE scaler (fitted before deletion)
    X_minus_scaled: np.ndarray = scaler_fitted_on_full.transform(X_minus)
    X_test_scaled:  np.ndarray = scaler_fitted_on_full.transform(X_test)

    # Step 3: Train the model and get prediction probabilities
    model = _build_model()
    model.fit(X_minus_scaled, y_minus)
    prob_matrix: np.ndarray = model.predict_proba(X_test_scaled)

    return prob_matrix


def path_B_clean_rebuild(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    delete_idx: int
) -> np.ndarray:
    """
    PATH B: CLEAN REBUILD
    ----------------------
    The correct deletion procedure. Everything is recomputed from
    scratch as if the deleted row never existed.

    Steps:
      1. Remove the target row from X_train.
      2. Fit a FRESH scaler on the reduced training set.
      3. Scale both train and test with the FRESH scaler.
      4. Train and predict.

    This is the ground-truth "deletion was done correctly" path.
    If Path A ≈ Path B, deletion consistency holds for this mechanism.

    Args:
        X_train    : Full training features (before deletion)
        y_train    : Full training labels
        X_test     : Hold-out test set
        delete_idx : Index of the row to delete (in X_train)

    Returns:
        prob_matrix : shape (n_test, n_classes) — predicted probabilities
    """
    # Step 1: Remove the row
    X_minus = X_train.drop(index=delete_idx).reset_index(drop=True)
    y_minus = np.delete(y_train, X_train.index.get_loc(delete_idx))

    # Step 2: Fit a FRESH scaler on the remaining data only
    fresh_scaler = StandardScaler()
    fresh_scaler.fit(X_minus)   # <-- This is the key difference from Path A

    # Step 3: Scale with the fresh scaler
    X_minus_scaled: np.ndarray = fresh_scaler.transform(X_minus)
    X_test_scaled:  np.ndarray = fresh_scaler.transform(X_test)

    # Step 4: Train and predict
    model = _build_model()
    model.fit(X_minus_scaled, y_minus)
    prob_matrix: np.ndarray = model.predict_proba(X_test_scaled)

    return prob_matrix


def _build_model():
    """
    Internal helper: return whichever model we are using.

    When USE_REAL_TABPFN=True, this returns a TabPFNClassifier.
    When False, it returns a RandomForest stand-in.

    Leading underscore (_) is Python convention for "private method"
    (like private in Java, but not enforced by the language).
    """
    if USE_REAL_TABPFN:
        # --- REAL TabPFN path (requires Python 3.11 + tabpfn installed) ---
        # Uncomment and use once your environment is ready:
        #
        # from tabpfn import TabPFNClassifier
        # return TabPFNClassifier(
        #     n_estimators=4,     # Keep low for speed on CPU
        #     device="cpu",
        #     seed=RANDOM_SEED
        # )
        raise NotImplementedError(
            "Set USE_REAL_TABPFN=False or install tabpfn in Python 3.11 venv."
        )
    else:
        # --- Stand-in: RandomForest (plumbing check only) ---
        # This lets us verify the pipeline structure is correct
        # WITHOUT TabPFN. Output is NOT thesis evidence.
        return RandomForestClassifier(
            n_estimators=50,
            random_state=RANDOM_SEED
        )

# ===========================================================
# SECTION 5: Metrics — how we measure the difference
# ===========================================================

def max_abs_diff(probs_A: np.ndarray, probs_B: np.ndarray) -> float:
    """
    Maximum Absolute Difference (MAD) between two probability matrices.

    Think of this as: "what is the worst-case disagreement between
    Path A and Path B on any single (test sample, class) pair?"

    If MAD = 0.0, Path A and Path B are identical.
    If MAD > 0.05, there is a noticeable disagreement.

    Args:
        probs_A : shape (n_test, n_classes) — probabilities from Path A
        probs_B : shape (n_test, n_classes) — probabilities from Path B

    Returns:
        float — the maximum absolute pointwise difference
    """
    return float(np.max(np.abs(probs_A - probs_B)))


def mean_l2_distance(probs_A: np.ndarray, probs_B: np.ndarray) -> float:
    """
    Mean L2 Distance between probability vectors, averaged over test rows.

    For each test sample, compute the Euclidean distance between
    Path A's probability vector and Path B's probability vector.
    Then average over all test samples.

    Example: if Path A says [0.7, 0.2, 0.1] and
             Path B says [0.6, 0.3, 0.1],
             L2 = sqrt((0.7-0.6)^2 + (0.2-0.3)^2 + (0.1-0.1)^2) ≈ 0.141

    Args:
        probs_A : shape (n_test, n_classes)
        probs_B : shape (n_test, n_classes)

    Returns:
        float — average L2 distance (lower = more consistent)
    """
    # np.linalg.norm with axis=1 computes L2 per row
    per_row_l2: np.ndarray = np.linalg.norm(probs_A - probs_B, axis=1)
    return float(np.mean(per_row_l2))


# ===========================================================
# SECTION 6: Main audit loop
# ===========================================================

def run_audit() -> None:
    """
    Main function: runs the deletion-consistency audit end-to-end.

    Flow:
      1. Set random seed
      2. Load dataset, split into train/test
      3. Fit a stale scaler on the full training set
      4. For each of N_DELETION_TRIALS random rows:
           a. Run Path A (naive delete, stale scaler)
           b. Run Path B (clean rebuild, fresh scaler)
           c. Record the difference metrics
      5. Report aggregate statistics and statistical test
    """
    # -------------------------------------------------------
    # 6.1  Setup
    # -------------------------------------------------------
    set_global_seed(RANDOM_SEED)
    print_section("SETUP")

    # Load data
    X, y, feature_names = load_dataset()
    dataset_hash: str = hash_dataframe(X)

    print(f"Dataset       : Iris (sklearn built-in)")
    print(f"Dataset hash  : {dataset_hash}  <- record this!")
    print(f"Total rows    : {len(X)}")
    print(f"Features      : {feature_names}")
    print(f"Classes       : {np.unique(y).tolist()}")
    print(f"Random seed   : {RANDOM_SEED}  <- record this!")
    print(f"TabPFN mode   : {'REAL TabPFN' if USE_REAL_TABPFN else 'RandomForest STAND-IN (not thesis evidence)'}")

    # -------------------------------------------------------
    # 6.2  Train / test split
    # -------------------------------------------------------
    # We split BEFORE the deletion loop so the test set is fixed.
    # The test set is NEVER used to fit any scaler or model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y          # Keep class balance in both splits
    )

    # Reset integer indices after splitting (makes row removal easier)
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    print(f"\nTrain rows    : {len(X_train)}")
    print(f"Test rows     : {len(X_test)}")

    # -------------------------------------------------------
    # 6.3  Fit the "stale" scaler on the FULL training set
    # -------------------------------------------------------
    # This scaler represents the state of the pipeline BEFORE any
    # deletion. Path A reuses it; Path B ignores it and builds fresh.
    stale_scaler = StandardScaler()
    stale_scaler.fit(X_train)   # Sees all training rows, including targets

    print(f"\nStale scaler fitted on {len(X_train)} rows (before any deletion).")

    # -------------------------------------------------------
    # 6.4  Choose which rows to "delete" for the trials
    # -------------------------------------------------------
    # We pick N_DELETION_TRIALS rows at random from the training set.
    # Using a fixed seed ensures the same rows are chosen every run.
    rng = np.random.default_rng(seed=RANDOM_SEED)
    delete_indices = rng.choice(
        X_train.index.tolist(),
        size=N_DELETION_TRIALS,
        replace=False           # Each row deleted at most once
    )

    print(f"Rows selected for deletion trials: {delete_indices.tolist()}")

    # -------------------------------------------------------
    # 6.5  Deletion loop
    # -------------------------------------------------------
    print_section(f"RUNNING {N_DELETION_TRIALS} DELETION TRIALS")

    # Lists to collect per-trial metrics
    mad_scores:  list[float] = []
    l2_scores:   list[float] = []
    times_A:     list[float] = []   # Wall-clock time for Path A
    times_B:     list[float] = []   # Wall-clock time for Path B

    for trial_num, del_idx in enumerate(delete_indices, start=1):

        # --- PATH A: Naive delete (stale scaler) ---
        t0 = time.perf_counter()
        probs_A = path_A_naive_delete(
            X_train, y_train, X_test,
            del_idx, stale_scaler
        )
        times_A.append(time.perf_counter() - t0)

        # --- PATH B: Clean rebuild (fresh scaler) ---
        t0 = time.perf_counter()
        probs_B = path_B_clean_rebuild(
            X_train, y_train, X_test, del_idx
        )
        times_B.append(time.perf_counter() - t0)

        # --- Compute difference metrics ---
        mad = max_abs_diff(probs_A, probs_B)
        l2  = mean_l2_distance(probs_A, probs_B)
        mad_scores.append(mad)
        l2_scores.append(l2)

        print(f"  Trial {trial_num:2d} | row={del_idx} | MAD={mad:.5f} | L2={l2:.5f} | "
              f"t_A={times_A[-1]:.3f}s | t_B={times_B[-1]:.3f}s")

    # -------------------------------------------------------
    # 6.6  Aggregate results
    # -------------------------------------------------------
    print_section("AGGREGATE RESULTS")

    mad_arr = np.array(mad_scores)
    l2_arr  = np.array(l2_scores)

    print(f"Max Absolute Difference (MAD):")
    print(f"  Mean  = {mad_arr.mean():.5f}")
    print(f"  Std   = {mad_arr.std():.5f}")
    print(f"  Max   = {mad_arr.max():.5f}")
    print(f"  Min   = {mad_arr.min():.5f}")

    print(f"\nMean L2 Distance:")
    print(f"  Mean  = {l2_arr.mean():.5f}")
    print(f"  Std   = {l2_arr.std():.5f}")

    print(f"\nWall-clock time (seconds):")
    print(f"  Path A mean = {np.mean(times_A):.4f}s")
    print(f"  Path B mean = {np.mean(times_B):.4f}s")

    # -------------------------------------------------------
    # 6.7  Statistical test
    # -------------------------------------------------------
    # Wilcoxon signed-rank test: are the MAD scores significantly > 0?
    # H0 (null): Path A and Path B produce identical predictions (MAD = 0)
    # H1 (alternative): they differ (MAD > 0, one-sided)
    #
    # We need variance in the data for this test to work.
    # If all MADs are 0.0, Wilcoxon will return p=1.0 (no difference found).
    print_section("STATISTICAL TEST")

    if np.all(mad_arr == 0.0):
        print("All MAD scores are 0.0 — Path A and Path B are identical.")
        print("Wilcoxon test skipped (no variance to test).")
        print("\nINTERPRETATION: For this stand-in model, stale vs. fresh scaler")
        print("makes no difference. This is EXPECTED for RandomForest because")
        print("it uses tree splits (not distance-based), so scaling has no effect.")
        print("With real TabPFN (a transformer), scaling DOES affect attention")
        print("and predictions — the difference will be non-zero.")
    else:
        # Compare MAD scores against zero (a vector of zeros = "no difference")
        stat, p_value = stats.wilcoxon(mad_arr, alternative="greater")
        print(f"Wilcoxon signed-rank test (H1: MAD > 0)")
        print(f"  Statistic = {stat:.4f}")
        print(f"  p-value   = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → SIGNIFICANT at α=0.05: stale preprocessing causes measurable")
            print(f"    prediction drift. The pipeline retains a trace of the deleted row.")
        else:
            print(f"  → NOT SIGNIFICANT at α=0.05: no detectable prediction drift.")

    # -------------------------------------------------------
    # 6.8  Mandatory caveats (from your accuracy rules)
    # -------------------------------------------------------
    print_section("CAVEATS — READ BEFORE INTERPRETING")
    print("""
1. STAND-IN MODEL WARNING:
   This run used RandomForest, not real TabPFN. These numbers are
   plumbing checks only. Do NOT cite them as thesis evidence.

2. MECHANISM ISOLATED:
   Any A-vs-B difference here isolates ONLY the effect of reusing
   stale StandardScaler statistics. It does NOT show that TabPFN
   internally memorizes rows.

3. MISSING BLIND BASELINE:
   We did not run a model that never saw the data at all.
   Without a blind baseline, distribution shift cannot be ruled out.

4. SAMPLE SIZE:
   {n} deletion trials is a pilot run. For thesis-level evidence,
   use ≥50 trials with 3+ random seeds (seeds: 42, 123, 7).

5. NOVELTY STATEMENT:
   We did not find a published study doing this exact audit. This
   does NOT mean it is the first; it means our search did not find one.
   Always differentiate from tabcustody (github.com/jrachid/tabcustody).
""".format(n=N_DELETION_TRIALS))

    print_section("RUN COMPLETE")
    print("Next steps:")
    print("  1. Install TabPFN in a Python 3.11 venv (see setup guide).")
    print("  2. Set USE_REAL_TABPFN = True at the top of this file.")
    print("  3. Re-run and compare output.")
    print("  4. Add blind baseline (a model never shown training data).")


# ===========================================================
# SECTION 7: Entry point
# ===========================================================

# In Python, `if __name__ == "__main__":` means:
# "Only run this block when this file is run directly, not when
#  imported by another script." It is like Java's main() method.
if __name__ == "__main__":
    run_audit()
