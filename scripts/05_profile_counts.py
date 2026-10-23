"""scripts/05_profile_counts.py: Step 0, Item 12.

Profiles the vectorized n-gram counting pipeline for orders 1 to 4 on full data/tokens.npy.
Measures:
- Elapsed time per order
- Peak memory (RSS) after gc.collect()
- Number of unique n-grams and total n-gram tokens
- Validates 8 GB memory budget; applies partition fallback if memory exceeds limit.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    PAD_ID,
    get_peak_memory_mb,
    require_colab,
)

STAGE_NAME = "05_profile_counts"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
MEMORY_BUDGET_MB = 8192.0  # 8 GB


def compute_pos_in_sent(n_tokens: int, starts: np.ndarray) -> np.ndarray:
    """Compute token position (0-indexed) within its respective sentence."""
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = n_tokens
    lengths = ends - starts
    starts_repeated = np.repeat(starts, lengths)
    pos_in_sent = np.arange(n_tokens, dtype=np.int64) - starts_repeated
    return pos_in_sent.astype(np.int32)


def build_ngram_matrix(tokens: np.ndarray, pos_in_sent: np.ndarray, n: int) -> np.ndarray:
    """Build the int32 n-gram matrix of shape (N, n) with <s> padding for preceding context."""
    n_tokens = len(tokens)
    matrix = np.empty((n_tokens, n), dtype=np.int32)

    for k in range(n):
        b = n - 1 - k  # backward offset
        if b == 0:
            matrix[:, k] = tokens
        else:
            matrix[:b, k] = PAD_ID
            matrix[b:, k] = tokens[:-b]
            matrix[pos_in_sent < b, k] = PAD_ID

    return matrix


def count_ngrams_direct(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort n-gram matrix with np.lexsort and collapse identical rows."""
    n_rows, n_cols = matrix.shape
    if n_cols == 1:
        unique_ngrams, counts = np.unique(matrix[:, 0], return_counts=True)
        return unique_ngrams.reshape(-1, 1), counts

    # np.lexsort sorts by keys in reverse order (primary key is column 0)
    # So we pass columns in order: [col n-1, col n-2, ..., col 0]
    keys = [matrix[:, c] for c in reversed(range(n_cols))]
    sort_idx = np.lexsort(keys)

    sorted_matrix = matrix[sort_idx]
    del sort_idx
    gc.collect()

    # Find row boundaries where any column changes
    diff_mask = np.any(sorted_matrix[1:] != sorted_matrix[:-1], axis=1)
    boundary_indices = np.concatenate(([0], np.where(diff_mask)[0] + 1, [n_rows]))

    unique_ngrams = sorted_matrix[boundary_indices[:-1]]
    counts = np.diff(boundary_indices).astype(np.int64)

    return unique_ngrams, counts


def count_ngrams_partitioned(
    matrix: np.ndarray, n_partitions: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Partition fallback: partition rows by column 0 ID range to cap memory usage."""
    col0 = matrix[:, 0]
    min_val, max_val = col0.min(), col0.max()
    boundaries = np.linspace(min_val, max_val + 1, n_partitions + 1, dtype=np.int32)

    unique_list = []
    counts_list = []

    for i in range(n_partitions):
        low, high = boundaries[i], boundaries[i + 1]
        mask = (col0 >= low) & (col0 < high)
        sub_matrix = matrix[mask]
        if len(sub_matrix) == 0:
            continue

        u_part, c_part = count_ngrams_direct(sub_matrix)
        unique_list.append(u_part)
        counts_list.append(c_part)
        del sub_matrix, u_part, c_part
        gc.collect()

    all_unique = np.vstack(unique_list)
    all_counts = np.concatenate(counts_list)
    return all_unique, all_counts


def main() -> None:
    require_colab(STAGE_NAME)

    if not TOKENS_FILE.exists() or not STARTS_FILE.exists():
        print(f"[ERROR] Required inputs not found ({TOKENS_FILE}, {STARTS_FILE})")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    total_start = time.time()

    print(f"Loading {TOKENS_FILE} and {STARTS_FILE}...")
    tokens = np.load(TOKENS_FILE)
    starts = np.load(STARTS_FILE)
    n_tokens = len(tokens)
    print(f"Loaded {n_tokens:,} tokens and {len(starts):,} sentences.")

    print("Computing token sentence positions...")
    pos_in_sent = compute_pos_in_sent(n_tokens, starts)
    del starts
    gc.collect()

    results_summary = []

    for n in range(1, 5):
        print(f"\n================ Order n = {n} ================")
        t0 = time.time()

        print(f"Building int32 matrix for n={n} ({n_tokens:,} x {n})...")
        matrix = build_ngram_matrix(tokens, pos_in_sent, n)
        mem_after_matrix = get_peak_memory_mb()
        print(f"  Matrix memory: {matrix.nbytes / (1024 * 1024):.1f} MB (Peak RSS: {mem_after_matrix:.1f} MB)")

        method_used = "direct_lexsort"
        if mem_after_matrix > MEMORY_BUDGET_MB * 0.7 and n >= 4:
            print("  [INFO] Memory threshold reached; using partitioned counting fallback.")
            method_used = "partition_fallback"
            unique_ngrams, counts = count_ngrams_partitioned(matrix, n_partitions=4)
        else:
            try:
                unique_ngrams, counts = count_ngrams_direct(matrix)
            except MemoryError:
                print("  [WARN] MemoryError encountered in direct lexsort. Falling back to partitioned count.")
                method_used = "partition_fallback"
                unique_ngrams, counts = count_ngrams_partitioned(matrix, n_partitions=4)

        del matrix
        gc.collect()

        elapsed_n = time.time() - t0
        peak_rss = get_peak_memory_mb()
        total_count = counts.sum()
        distinct_count = len(unique_ngrams)

        print(f"Completed n={n} in {elapsed_n:.2f} s | Method: {method_used}")
        print(f"  Total n-grams: {total_count:,} (Expected: {n_tokens:,})")
        print(f"  Distinct n-grams: {distinct_count:,}")
        print(f"  Peak Memory: {peak_rss:.1f} MB (Budget: {MEMORY_BUDGET_MB:.0f} MB)")

        assert total_count == n_tokens, f"Total count {total_count} != prediction positions {n_tokens}"

        results_summary.append({
            "order": n,
            "method": method_used,
            "time_sec": elapsed_n,
            "peak_mem_mb": peak_rss,
            "distinct_ngrams": distinct_count,
            "total_tokens": total_count,
        })

        del unique_ngrams, counts
        gc.collect()

    print("\n================ Profiling Summary ================")
    print("Order | Method            | Time (s) | Peak Mem (MB) | Distinct N-grams | Total Tokens")
    print("--------------------------------------------------------------------------------------")
    for r in results_summary:
        print(
            f"  {r['order']}   | {r['method']:<17s} | {r['time_sec']:8.2f} | {r['peak_mem_mb']:13.1f} | "
            f"{r['distinct_ngrams']:16,} | {r['total_tokens']:,}"
        )

    print(f"\nTotal Profiling Time: {time.time() - total_start:.2f} s")
    print("[SUCCESS] Vectorized n-gram counting verification passed across all orders 1 to 4.")


if __name__ == "__main__":
    main()
