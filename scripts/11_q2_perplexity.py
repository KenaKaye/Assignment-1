"""scripts/11_q2_perplexity.py: Q2 Held-out Perplexity Evaluation.

Computes test split perplexity for n = 1, 2, 3, 4 under Kneser-Ney smoothing:
- Primary split: seed 42 (evaluated with seed 42 training counts & LUT)
- Replication split: seed 7 (evaluated with seed 7 training counts & LUT)
- Full held-out test splits without sampling
- Vectorized scoring via kn_lib
- Produces: results/q2_perplexity.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    EOS_ID,
    RESULTS_DIR,
    UNK_ID,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)
from scripts.kn_lib import KneserNeyModel

STAGE_NAME = "11_q2_perplexity"
SPLIT_42_FILE = DATA_DIR / "split_indices_seed42.json"
if not SPLIT_42_FILE.exists() and (DATA_DIR / "split_indices_seed42_v2.json").exists():
    SPLIT_42_FILE = DATA_DIR / "split_indices_seed42_v2.json"

SPLIT_7_FILE = DATA_DIR / "split_indices_seed7.json"
if not SPLIT_7_FILE.exists() and (DATA_DIR / "split_indices_seed7_v2.json").exists():
    SPLIT_7_FILE = DATA_DIR / "split_indices_seed7_v2.json"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
RESULTS_FILE = RESULTS_DIR / "q2_perplexity.json"


def evaluate_seed_split(
    seed: int,
    split_file: Path,
    tokens: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> dict:
    """Evaluate held-out test split for a given seed with its own model & LUT."""
    train_dir = DATA_DIR / "counts" / f"train_seed{seed}"
    lut_file = train_dir / f"lut_train_seed{seed}.npy"
    if not lut_file.exists():
        lut_file = DATA_DIR / f"lut_train_seed{seed}.npy"

    print(f"\n--- Evaluating Seed {seed} ---")
    print(f"Loading LUT: {lut_file}")
    lut = np.load(lut_file)

    print(f"Loading KneserNeyModel from {train_dir}...")
    kn_model = KneserNeyModel.load_from_dir(train_dir, max_order=4)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_indices = np.array(split_data["test"], dtype=np.int64)

    test_starts = starts[test_indices]
    test_lengths = (ends[test_indices] - test_starts).astype(np.int64)
    valid_mask = test_lengths > 0
    test_indices = test_indices[valid_mask]
    test_starts = test_starts[valid_mask]
    test_lengths = test_lengths[valid_mask]

    total_test_tokens = int(test_lengths.sum())
    print(f"Extracting {len(test_indices):,} test sentences ({total_test_tokens:,} tokens)...")

    test_tokens = np.empty(total_test_tokens, dtype=np.int32)
    test_pos = np.empty(total_test_tokens, dtype=np.int64)

    curr = 0
    for s_start, s_len in zip(test_starts, test_lengths):
        test_tokens[curr : curr + s_len] = tokens[s_start : s_start + s_len]
        test_pos[curr : curr + s_len] = np.arange(s_len, dtype=np.int64)
        curr += s_len

    # Apply training LUT
    print("Applying training LUT to held-out test tokens...")
    test_tokens = lut[test_tokens]

    unk_count = int(np.count_nonzero(test_tokens == UNK_ID))
    unk_fraction = unk_count / total_test_tokens
    print(f"Held-out tokens: {total_test_tokens:,} | UNK count: {unk_count:,} ({unk_fraction * 100:.2f}%)")

    orders_result = {}
    unigram_ppl = None

    for n in range(1, 5):
        t0 = time.time()
        print(f"Scoring test sequence for order {n}...")
        logp_arr, _ = kn_model.score_sequence(test_tokens, test_pos, order=n)
        sum_logp = float(logp_arr.sum())
        ppl = math.exp(-sum_logp / total_test_tokens)
        elapsed_n = round(time.time() - t0, 2)

        reduction_pct = 0.0
        if n == 1:
            unigram_ppl = ppl
        elif unigram_ppl is not None and unigram_ppl > 0:
            reduction_pct = ((unigram_ppl - ppl) / unigram_ppl) * 100.0

        orders_result[f"n={n}"] = {
            "order": n,
            "perplexity": round(ppl, 2),
            "logp_sum": round(sum_logp, 2),
            "reduction_vs_unigram_pct": round(reduction_pct, 2),
            "time_sec": elapsed_n,
        }
        print(f"  [Seed {seed} n={n}] PP: {ppl:8.2f} (Reduction: {reduction_pct:5.2f}%) in {elapsed_n} s")

    return {
        "total_sentences": len(test_indices),
        "total_positions": total_test_tokens,
        "unk_positions": unk_count,
        "unk_fraction": round(unk_fraction, 6),
        "orders": orders_result,
    }


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_inputs = [
        SPLIT_42_FILE,
        SPLIT_7_FILE,
        TOKENS_FILE,
        STARTS_FILE,
    ]
    for s in [42, 7]:
        td = DATA_DIR / "counts" / f"train_seed{s}"
        expected_inputs.append(td / f"lut_train_seed{s}.npy")
        for n in range(1, 5):
            expected_inputs.append(td / f"order{n}_ngrams.npy")
            expected_inputs.append(td / f"order{n}_counts.npy")

    if stage_is_done(STAGE_NAME, expected_inputs, [RESULTS_FILE]):
        print(f"[{STAGE_NAME}] Checkpoint valid. Skipping.")
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    print(f"Loading {TOKENS_FILE.name} and {STARTS_FILE.name}...")
    tokens = np.load(TOKENS_FILE, mmap_mode="r")
    starts = np.load(STARTS_FILE)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(tokens)

    # Primary run: seed 42
    res_42 = evaluate_seed_split(42, SPLIT_42_FILE, tokens, starts, ends)

    # Replication run: seed 7
    res_7 = evaluate_seed_split(7, SPLIT_7_FILE, tokens, starts, ends)

    bigram_red_42 = res_42["orders"]["n=2"]["reduction_vs_unigram_pct"]
    meets_target = bigram_red_42 >= 20.0

    output = {
        "benchmark_targets": {
            "bigram_reduction_target_pct": 20.0,
            "label": "proposed target set by the group, not a published standard",
            "status": "PASS" if meets_target else "FAIL",
            "observed_seed42_bigram_reduction_pct": bigram_red_42,
        },
        "seed_42": res_42,
        "seed_7": res_7,
    }

    tmp_file = RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_file.replace(RESULTS_FILE)

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed} s")
    record_stage(STAGE_NAME, expected_inputs, [RESULTS_FILE])


if __name__ == "__main__":
    main()
