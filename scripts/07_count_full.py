"""scripts/07_count_full.py: Q1.2.

Builds and saves the full-corpus sorted n-gram arrays and counts for orders 1 to 4.
Produces:
- data/counts/full/order{n}_ngrams.npy (int32, sorted)
- data/counts/full/order{n}_counts.npy (int64)
"""

from __future__ import annotations

import gc
import importlib
import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    FULL_COUNTS_DIR,
    get_peak_memory_mb,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)

# Load helper functions from 05_profile_counts
scripts_05 = importlib.import_module("scripts.05_profile_counts")
build_ngram_matrix = scripts_05.build_ngram_matrix
compute_pos_in_sent = scripts_05.compute_pos_in_sent
count_ngrams_direct = scripts_05.count_ngrams_direct
count_ngrams_partitioned = scripts_05.count_ngrams_partitioned

STAGE_NAME = "07_count_full"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
MEMORY_BUDGET_MB = 8192.0


def main() -> None:
    require_colab(STAGE_NAME)

    expected_files = []
    for n in range(1, 5):
        expected_files.append(FULL_COUNTS_DIR / f"order{n}_ngrams.npy")
        expected_files.append(FULL_COUNTS_DIR / f"order{n}_counts.npy")

    if is_checkpoint_done(STAGE_NAME, expected_files):
        return

    if not TOKENS_FILE.exists() or not STARTS_FILE.exists():
        print(f"[ERROR] Required inputs not found ({TOKENS_FILE}, {STARTS_FILE})")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()
    FULL_COUNTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {TOKENS_FILE} and {STARTS_FILE}...")
    tokens = np.load(TOKENS_FILE)
    starts = np.load(STARTS_FILE)
    n_tokens = len(tokens)
    print(f"Loaded {n_tokens:,} tokens.")

    print("Computing sentence token positions...")
    pos_in_sent = compute_pos_in_sent(n_tokens, starts)
    del starts
    gc.collect()

    for n in range(1, 5):
        t0 = time.time()
        print(f"\nBuilding counts for order n = {n}...")
        matrix = build_ngram_matrix(tokens, pos_in_sent, n)

        mem = get_peak_memory_mb()
        if mem > MEMORY_BUDGET_MB * 0.7 and n >= 4:
            print("  Applying partition fallback...")
            unique_ngrams, counts = count_ngrams_partitioned(matrix, n_partitions=4)
        else:
            try:
                unique_ngrams, counts = count_ngrams_direct(matrix)
            except MemoryError:
                print("  MemoryError: switching to partition fallback...")
                unique_ngrams, counts = count_ngrams_partitioned(matrix, n_partitions=4)

        del matrix
        gc.collect()

        ngrams_file = FULL_COUNTS_DIR / f"order{n}_ngrams.npy"
        counts_file = FULL_COUNTS_DIR / f"order{n}_counts.npy"

        print(f"  Saving {ngrams_file.name} (shape {unique_ngrams.shape}) and {counts_file.name}...")
        np.save(ngrams_file, unique_ngrams)
        np.save(counts_file, counts)

        elapsed_n = time.time() - t0
        print(
            f"  Finished order {n}: {len(unique_ngrams):,} distinct n-grams, "
            f"{counts.sum():,} total tokens in {elapsed_n:.2f} s"
        )
        del unique_ngrams, counts
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} complete in {elapsed:.2f} s")
    record_stage_checkpoint(STAGE_NAME, expected_files)


if __name__ == "__main__":
    main()
