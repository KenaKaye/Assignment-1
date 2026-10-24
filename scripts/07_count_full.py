"""scripts/07_count_full.py: Q1.2.

Builds and saves the full-corpus sorted n-gram arrays and counts for orders 1 to 4.
Produces:
- data/counts/full/order{n}_ngrams.npy (int32, sorted)
- data/counts/full/order{n}_counts.npy (int64)
- data/profile_counts.json (profiling metrics per order, replacing stage 05)
"""

from __future__ import annotations

import gc
import json
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
    memory_budget_mb,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)
from scripts.ngram_lib import (
    context_matrix,
    group_rows,
    group_rows_partitioned,
    sentence_positions,
    use_partition,
)

STAGE_NAME = "07_count_full"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
PROFILE_FILE = DATA_DIR / "profile_counts.json"


def load_profile_records() -> list[dict]:
    if not PROFILE_FILE.exists():
        return []
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_profile_record(record: dict) -> None:
    records = [r for r in load_profile_records() if r.get("order") != record.get("order")]
    records.append(record)
    records.sort(key=lambda x: x.get("order", 0))
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_f = PROFILE_FILE.with_suffix(".json.tmp")
    with open(tmp_f, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    tmp_f.replace(PROFILE_FILE)


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    all_expected = []
    for n in range(1, 5):
        all_expected.append(FULL_COUNTS_DIR / f"order{n}_ngrams.npy")
        all_expected.append(FULL_COUNTS_DIR / f"order{n}_counts.npy")

    if stage_is_done(STAGE_NAME, [TOKENS_FILE, STARTS_FILE], all_expected):
        return

    if not TOKENS_FILE.exists() or not STARTS_FILE.exists():
        print(f"[ERROR] Required inputs not found ({TOKENS_FILE}, {STARTS_FILE})")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    budget_mb = memory_budget_mb()
    print(f"Operational Memory Budget: {budget_mb:.1f} MB")
    start_time = time.time()
    FULL_COUNTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {TOKENS_FILE} and {STARTS_FILE}...")
    tokens = np.load(TOKENS_FILE)
    starts = np.load(STARTS_FILE)
    n_tokens = len(tokens)
    print(f"Loaded {n_tokens:,} tokens across {len(starts):,} sentences.")

    print("Computing sentence token positions...")
    pos_in_sent = sentence_positions(starts, n_tokens)
    del starts
    gc.collect()

    for n in range(1, 5):
        order_stage = f"07_count_full_order{n}"
        ngrams_file = FULL_COUNTS_DIR / f"order{n}_ngrams.npy"
        counts_file = FULL_COUNTS_DIR / f"order{n}_counts.npy"
        order_outputs = [ngrams_file, counts_file]

        # Per-order checkpoint check (item 7-4)
        if stage_is_done(order_stage, [TOKENS_FILE, STARTS_FILE], order_outputs):
            print(f"Order n={n} already verified complete. Skipping.")
            continue

        t0 = time.time()
        print(f"\nBuilding counts for order n = {n}...")
        matrix = context_matrix(tokens, pos_in_sent, n)

        cur_mem = get_peak_memory_mb()
        partitioned = use_partition(n_tokens, n, budget_mb, current_usage_mb=cur_mem)

        if partitioned:
            print(f"  Estimated memory exceeds budget ({budget_mb:.1f} MB) -> Using group_rows_partitioned(parts=4)...")
            unique_ngrams, counts = group_rows_partitioned(matrix, parts=4)
        else:
            print(f"  Direct lexsort group_rows fits within budget ({budget_mb:.1f} MB)...")
            unique_ngrams, counts = group_rows(matrix)

        del matrix
        gc.collect()

        assert counts.sum() == n_tokens, f"Order {n} count sum ({counts.sum()}) != N ({n_tokens})!"
        assert unique_ngrams.dtype == np.int32, f"Expected int32 ngrams array, got {unique_ngrams.dtype}"
        assert counts.dtype == np.int64, f"Expected int64 counts array, got {counts.dtype}"

        tmp_ngrams = ngrams_file.with_name(f"{ngrams_file.stem}.tmp.npy")
        tmp_counts = counts_file.with_name(f"{counts_file.stem}.tmp.npy")

        print(f"  Saving {ngrams_file.name} (shape {unique_ngrams.shape}) and {counts_file.name}...")
        with open(tmp_ngrams, "wb") as f:
            np.save(f, unique_ngrams)
        with open(tmp_counts, "wb") as f:
            np.save(f, counts)
        tmp_ngrams.replace(ngrams_file)
        tmp_counts.replace(counts_file)

        elapsed_n = round(time.time() - t0, 2)
        peak_rss = round(get_peak_memory_mb(), 2)

        print(
            f"  Finished order {n}: {len(unique_ngrams):,} distinct n-grams, "
            f"{counts.sum():,} total tokens in {elapsed_n} s (Peak RSS: {peak_rss} MB)"
        )

        # Profile output (item 7-5)
        save_profile_record({
            "order": n,
            "N": int(n_tokens),
            "distinct": int(len(unique_ngrams)),
            "seconds": elapsed_n,
            "peak_rss_mb": peak_rss,
            "budget_mb": round(budget_mb, 1),
            "partitioned": partitioned,
        })

        record_stage(order_stage, [TOKENS_FILE, STARTS_FILE], order_outputs)
        del unique_ngrams, counts
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} complete in {elapsed:.2f} s")
    record_stage(STAGE_NAME, [TOKENS_FILE, STARTS_FILE], all_expected)


if __name__ == "__main__":
    main()
