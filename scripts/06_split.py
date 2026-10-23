"""scripts/06_split.py: Q2 Step 1.

Splits data/GPAC_cleaned.txt (by sentence index) into 90% train and 10% held-out test.
Produces:
- data/split_indices_seed42.json
- data/split_indices_seed7.json (for seed stability check)
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
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)

STAGE_NAME = "06_split"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
SPLIT_42_FILE = DATA_DIR / "split_indices_seed42.json"
SPLIT_7_FILE = DATA_DIR / "split_indices_seed7.json"


def generate_split(total_sentences: int, seed: int, train_ratio: float = 0.9) -> dict[str, list[int]]:
    """Deterministically partition indices into train and test sets."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total_sentences)

    n_train = int(total_sentences * train_ratio)
    train_indices = np.sort(shuffled[:n_train])
    test_indices = np.sort(shuffled[n_train:])

    # Verification: disjoint and full coverage
    assert len(train_indices) + len(test_indices) == total_sentences, "Count mismatch!"
    train_set = set(train_indices)
    test_set = set(test_indices)
    assert train_set.isdisjoint(test_set), "Train and test index sets intersect!"
    assert len(train_set | test_set) == total_sentences, "Union does not cover all sentences!"

    return {
        "seed": seed,
        "total_sentences": total_sentences,
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "train": train_indices.tolist(),
        "test": test_indices.tolist(),
    }


def main() -> None:
    require_colab(STAGE_NAME)

    expected_files = [SPLIT_42_FILE, SPLIT_7_FILE]
    if is_checkpoint_done(STAGE_NAME, expected_files):
        return

    if not STARTS_FILE.exists():
        print(f"[ERROR] Sentence starts file not found: {STARTS_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    starts = np.load(STARTS_FILE)
    n_sentences = len(starts)
    del starts
    print(f"Total sentences to split: {n_sentences:,}")

    # Seed 42 (Primary)
    print(f"Generating primary split (seed={SEED})...")
    split_42 = generate_split(n_sentences, SEED)
    tmp_42 = SPLIT_42_FILE.with_suffix(".json.tmp")
    with open(tmp_42, "w", encoding="utf-8") as f:
        json.dump(split_42, f)
    tmp_42.replace(SPLIT_42_FILE)
    print(f"Saved {SPLIT_42_FILE}: train={split_42['train_count']:,}, test={split_42['test_count']:,}")

    # Seed 7 (Replication / Stability)
    print(f"Generating replication split (seed={SEED_VAL})...")
    split_7 = generate_split(n_sentences, SEED_VAL)
    tmp_7 = SPLIT_7_FILE.with_suffix(".json.tmp")
    with open(tmp_7, "w", encoding="utf-8") as f:
        json.dump(split_7, f)
    tmp_7.replace(SPLIT_7_FILE)
    print(f"Saved {SPLIT_7_FILE}: train={split_7['train_count']:,}, test={split_7['test_count']:,}")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s")
    record_stage_checkpoint(STAGE_NAME, expected_files)


if __name__ == "__main__":
    main()
