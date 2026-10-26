"""scripts/06_split.py: Step 1 of Q2, split by sentence text group (approved amendment).

Identical sentences (same text in GPAC_cleaned.txt) form one group. Every group goes
entirely to one side, so no test sentence text appears in the training counts.

Produces (for seed s in {42, 7}):
- data/split_indices_seed{s}.json with keys:
  seed, numpy_version, total_sentences, n_groups, rule,
  train_full, tune, train_core, test   (sorted line indices into GPAC_cleaned.txt)

Rules for seed s (N = total sentences, G = number of distinct sentence texts):
1. Test: groups are ordered by a random priority from default_rng(s). Walk the order and
   take each group whose size still fits within round(0.10 * N). Stop when the target is hit.
2. Train_full: every sentence not in test.
3. Tune: from train groups only, ordered by default_rng([s, 1]), greedy within round(0.05 * N).
4. Train_core: train_full minus tune. Q2, Q3, Q4 counts use train_core only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    SEED,
    SEED_VAL,
    record_stage,
    require_heavy_runtime,
    stage_is_done,
)

STAGE_NAME = "06_split"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
SPLIT_FILES = {
    SEED: DATA_DIR / f"split_indices_seed{SEED}.json",
    SEED_VAL: DATA_DIR / f"split_indices_seed{SEED_VAL}.json",
}
TEST_FRACTION = 0.10
TUNE_FRACTION = 0.05


def line_group_ids(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (group id per line, size per group). Lines with identical text share an id."""
    hashes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            hashes.append(int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little"))
    h = np.array(hashes, dtype=np.uint64)
    del hashes
    _, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)
    return inverse.astype(np.int64), counts.astype(np.int64)


def greedy_pick(order: np.ndarray, sizes: np.ndarray, target: int, allowed: np.ndarray) -> np.ndarray:
    """Walk groups in priority order, take each allowed group that still fits in target."""
    chosen = np.zeros(len(sizes), dtype=bool)
    total = 0
    size_list = sizes.tolist()
    allowed_list = allowed.tolist()
    for g in order.tolist():
        if not allowed_list[g]:
            continue
        sz = size_list[g]
        if total + sz <= target:
            chosen[g] = True
            total += sz
            if total == target:
                break
    return chosen


def build_split(seed: int, inv: np.ndarray, sizes: np.ndarray, n: int) -> dict:
    g_count = len(sizes)
    all_groups = np.ones(g_count, dtype=bool)

    test_order = np.random.default_rng(seed).permutation(g_count)
    test_groups = greedy_pick(test_order, sizes, round(TEST_FRACTION * n), all_groups)
    test_line = test_groups[inv]
    train_full_line = ~test_line

    tune_order = np.random.default_rng([seed, 1]).permutation(g_count)
    tune_groups = greedy_pick(tune_order, sizes, round(TUNE_FRACTION * n), ~test_groups)
    tune_line = tune_groups[inv]
    train_core_line = train_full_line & ~tune_line

    test_idx = np.flatnonzero(test_line)
    train_full_idx = np.flatnonzero(train_full_line)
    tune_idx = np.flatnonzero(tune_line)
    train_core_idx = np.flatnonzero(train_core_line)

    assert len(test_idx) + len(train_full_idx) == n, "Test and train do not cover all sentences"
    assert not np.any(test_line & train_full_line), "Test and train overlap"
    assert not np.any(tune_line & test_line), "Tune overlaps test"
    assert np.array_equal(np.sort(np.concatenate([train_core_idx, tune_idx])), train_full_idx), "Tune and train_core do not cover train_full"
    assert not np.any(test_groups & tune_groups), "A tune group is also a test group"

    # No test sentence text may appear in training
    assert not np.any(np.isin(inv[test_idx], inv[train_full_idx])), "A sentence text appears on both sides"

    return {
        "seed": seed,
        "numpy_version": np.__version__,
        "total_sentences": int(n),
        "n_groups": int(g_count),
        "rule": "text-group split; test 10% and tune 5% by greedy group fill in default_rng priority order",
        "test_count": int(len(test_idx)),
        "train_full_count": int(len(train_full_idx)),
        "tune_count": int(len(tune_idx)),
        "train_core_count": int(len(train_core_idx)),
        "train_full": train_full_idx.tolist(),
        "tune": tune_idx.tolist(),
        "train_core": train_core_idx.tolist(),
        "test": test_idx.tolist(),
    }


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    inputs = [CLEANED_FILE, STARTS_FILE]
    v2_outputs = [path.with_name(f"split_indices_seed{s}_v2.json") for s, path in SPLIT_FILES.items()]
    outputs = list(SPLIT_FILES.values()) + v2_outputs
    if stage_is_done(STAGE_NAME, inputs, outputs):
        return

    for p in inputs:
        if not p.exists():
            print(f"[ERROR] Missing input: {p}")
            sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    print(f"numpy {np.__version__} (pinned for reproducibility: 2.1.3)")
    start = time.time()

    starts = np.load(STARTS_FILE)
    n = len(starts)
    print(f"Sentences (from sentence_starts.npy): {n:,}")

    inv, sizes = line_group_ids(CLEANED_FILE)
    assert len(inv) == n, f"Line count {len(inv):,} differs from sentence count {n:,}"
    print(f"Distinct sentence texts: {len(sizes):,}")
    print(f"Sentences that share their text with another sentence: {int(sizes[sizes > 1].sum()):,}")

    for seed, path in SPLIT_FILES.items():
        split = build_split(seed, inv, sizes, n)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(split, f)
        tmp.replace(path)

        v2_path = path.with_name(f"split_indices_seed{seed}_v2.json")
        v2_tmp = v2_path.with_suffix(".json.tmp")
        with open(v2_tmp, "w", encoding="utf-8") as f:
            json.dump(split, f)
        v2_tmp.replace(v2_path)

        print(
            f"Seed {seed}: test={split['test_count']:,}, train_full={split['train_full_count']:,}, "
            f"tune={split['tune_count']:,}, train_core={split['train_core_count']:,}, "
            f"groups={split['n_groups']:,}  -> {path.name}"
        )

    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {time.time() - start:.2f} s")
    record_stage(STAGE_NAME, inputs, outputs)


if __name__ == "__main__":
    main()