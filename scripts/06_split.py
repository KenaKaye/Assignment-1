"""scripts/06_split.py: Q2 Step 1.

Splits data/GPAC_cleaned.txt (by sentence index) into train and held-out test.
Produces canonical v2 splits (D2, D4):
- data/split_indices_seed42_v2.json
- data/split_indices_seed7_v2.json
Each contains: train_full (90%), tune (5%), train_core (85%), test (10%).
Guarantees test in v2 matches the legacy v1 test split exactly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    SEED,
    SEED_VAL,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)

STAGE_NAME = "06_split"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
SPLIT_42_FILE = DATA_DIR / "split_indices_seed42.json"
SPLIT_7_FILE = DATA_DIR / "split_indices_seed7.json"
SPLIT_42_V2_FILE = DATA_DIR / "split_indices_seed42_v2.json"
SPLIT_7_V2_FILE = DATA_DIR / "split_indices_seed7_v2.json"


def generate_split_v2(
    total_sentences: int,
    seed: int,
    train_ratio: float = 0.9,
    tune_ratio: float = 0.05,
    legacy_split: dict | None = None,
) -> dict[str, int | list[int]]:
    """Deterministically partition indices into train_full, tune, train_core, and test."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total_sentences)

    n_train = int(round(total_sentences * train_ratio))
    train_full = np.sort(shuffled[:n_train])
    test = np.sort(shuffled[n_train:])

    # If legacy split exists, verify exact match of test partition
    if legacy_split is not None:
        legacy_test = np.asarray(legacy_split["test"], dtype=np.int64)
        if not np.array_equal(test, legacy_test):
            raise AssertionError(f"Seed {seed} generated test split differs from saved legacy test split!")

    # Carve out tuning set (D2) from train_full using independent RNG
    rng_tune = np.random.default_rng([seed, 1])
    tune_perm = rng_tune.permutation(train_full)
    n_tune = int(round(total_sentences * tune_ratio))
    tune = np.sort(tune_perm[:n_tune])
    train_core = np.sort(tune_perm[n_tune:])

    # NumPy disjointness and coverage check (item 6-5)
    combined = np.sort(np.concatenate([train_core, tune, test]))
    expected_arange = np.arange(total_sentences, dtype=np.int64)
    if not np.array_equal(combined, expected_arange):
        raise AssertionError("Split partitions do not cleanly partition [0, total_sentences)!")

    train_combined = np.sort(np.concatenate([train_core, tune]))
    if not np.array_equal(train_combined, train_full):
        raise AssertionError("train_core and tune do not cleanly partition train_full!")

    return {
        "seed": seed,
        "total_sentences": total_sentences,
        "train_full_count": len(train_full),
        "tune_count": len(tune),
        "train_core_count": len(train_core),
        "test_count": len(test),
        "train_full": train_full.tolist(),
        "tune": tune.tolist(),
        "train_core": train_core.tolist(),
        "test": test.tolist(),
    }


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    inputs = [STARTS_FILE]
    outputs = [SPLIT_42_V2_FILE, SPLIT_7_V2_FILE]

    if stage_is_done(STAGE_NAME, inputs, outputs):
        return

    if not STARTS_FILE.exists():
        print(f"[ERROR] Sentence starts file not found: {STARTS_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    print(f"NumPy Version: {np.__version__}")
    print(f"RNG BitGenerator: PCG64")
    start_time = time.time()

    starts = np.load(STARTS_FILE)
    n_sentences = len(starts)
    del starts
    print(f"Total sentences to split: {n_sentences:,}")

    # Load legacy splits for verification if they exist
    legacy_42 = None
    if SPLIT_42_FILE.exists():
        with open(SPLIT_42_FILE, "r", encoding="utf-8") as f:
            legacy_42 = json.load(f)

    legacy_7 = None
    if SPLIT_7_FILE.exists():
        with open(SPLIT_7_FILE, "r", encoding="utf-8") as f:
            legacy_7 = json.load(f)

    # Seed 42
    print(f"Generating primary v2 split (seed={SEED})...")
    split_42_v2 = generate_split_v2(n_sentences, SEED, legacy_split=legacy_42)
    tmp_42 = SPLIT_42_V2_FILE.with_suffix(".json.tmp")
    with open(tmp_42, "w", encoding="utf-8") as f:
        json.dump(split_42_v2, f)
    tmp_42.replace(SPLIT_42_V2_FILE)
    print(
        f"Saved {SPLIT_42_V2_FILE}: train_core={split_42_v2['train_core_count']:,}, "
        f"tune={split_42_v2['tune_count']:,}, test={split_42_v2['test_count']:,}"
    )

    # Seed 7
    print(f"Generating replication v2 split (seed={SEED_VAL})...")
    split_7_v2 = generate_split_v2(n_sentences, SEED_VAL, legacy_split=legacy_7)
    tmp_7 = SPLIT_7_V2_FILE.with_suffix(".json.tmp")
    with open(tmp_7, "w", encoding="utf-8") as f:
        json.dump(split_7_v2, f)
    tmp_7.replace(SPLIT_7_V2_FILE)
    print(
        f"Saved {SPLIT_7_V2_FILE}: train_core={split_7_v2['train_core_count']:,}, "
        f"tune={split_7_v2['tune_count']:,}, test={split_7_v2['test_count']:,}"
    )

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s")
    record_stage(STAGE_NAME, inputs, outputs)


if __name__ == "__main__":
    main()
