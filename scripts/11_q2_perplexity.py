"""scripts/11_q2_perplexity.py: Q2 Held-out Perplexity Evaluation.

Computes test split perplexity for n = 1, 2, 3, 4 under Kneser-Ney smoothing:
- Primary split: seed 42
- Replication split: seed 7
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
    TRAIN_COUNTS_DIR,
    UNK_ID,
    WORD_RE,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)
from scripts.09_kn import KneserNeyModel

STAGE_NAME = "11_q2_perplexity"
SPLIT_42_FILE = DATA_DIR / "split_indices_seed42.json"
SPLIT_7_FILE = DATA_DIR / "split_indices_seed7.json"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
TRAIN_VOCAB_FILE = TRAIN_COUNTS_DIR / "vocab_train.tsv"
RESULTS_FILE = RESULTS_DIR / "q2_perplexity.json"


def load_train_vocab() -> dict[str, int]:
    """Load word to ID mapping from data/counts/train_seed42/vocab_train.tsv."""
    w2id = {}
    with open(TRAIN_VOCAB_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                w2id[parts[1]] = int(parts[0])
    return w2id


def evaluate_split(
    split_file: Path,
    kn_model: KneserNeyModel,
    w2id: dict[str, int],
    max_eval_sentences: int | None = None,
) -> dict[str, dict]:
    """Evaluate perplexity across all orders for a given test split."""
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_indices = set(split_data["test"])
    if max_eval_sentences:
        test_indices = set(list(test_indices)[:max_eval_sentences])

    print(f"Loading {len(test_indices):,} test sentences from {CLEANED_FILE.name}...")
    test_token_sequences: list[list[int]] = []
    total_unk_count = 0
    total_tokens = 0

    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in test_indices:
                words = WORD_RE.findall(line)
                if len(words) < 2:
                    continue
                s_ids = []
                for w in words:
                    tid = w2id.get(w, UNK_ID)
                    if tid == UNK_ID:
                        total_unk_count += 1
                    s_ids.append(tid)
                s_ids.append(EOS_ID)
                total_tokens += len(s_ids)
                test_token_sequences.append(s_ids)

    print(f"Test Set: {len(test_token_sequences):,} sentences, {total_tokens:,} tokens ({total_unk_count:,} UNK)")

    results_by_order = {}
    baseline_ppl = None

    for n in range(1, 5):
        t0 = time.time()
        print(f"Computing test perplexity for n={n}...")
        ppl, evaluated_tokens = kn_model.perplexity(test_token_sequences, order=n)
        elapsed_n = time.time() - t0

        reduction_pct = 0.0
        if n == 1:
            baseline_ppl = ppl
        elif baseline_ppl is not None and baseline_ppl > 0:
            reduction_pct = ((baseline_ppl - ppl) / baseline_ppl) * 100.0

        results_by_order[f"n={n}"] = {
            "order": n,
            "perplexity": round(ppl, 2),
            "tokens": evaluated_tokens,
            "unk_count": total_unk_count,
            "reduction_vs_unigram_pct": round(reduction_pct, 2),
            "time_sec": round(elapsed_n, 2),
        }
        print(f"  n={n} | PP: {ppl:8.2f} (Reduction vs unigram: {reduction_pct:5.2f}%) in {elapsed_n:.2f} s")

    return results_by_order


def main() -> None:
    require_colab(STAGE_NAME)

    if is_checkpoint_done(STAGE_NAME, [RESULTS_FILE]):
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training vocabulary and Kneser-Ney model...")
    w2id = load_train_vocab()
    kn_model = KneserNeyModel.load_from_dir(TRAIN_COUNTS_DIR, max_order=4)

    # Primary Evaluation (Seed 42)
    print("\n[Primary Run: Seed 42]")
    eval_seed42 = evaluate_split(SPLIT_42_FILE, kn_model, w2id)

    # Seed Stability Run (Seed 7)
    print("\n[Stability Run: Seed 7]")
    eval_seed7 = evaluate_split(SPLIT_7_FILE, kn_model, w2id)

    output = {
        "benchmark_targets": {
            "bigram_reduction_target_pct": 20.0,
            "status": "PASS" if eval_seed42["n=2"]["reduction_vs_unigram_pct"] >= 20.0 else "FAIL",
        },
        "seed_42": eval_seed42,
        "seed_7": eval_seed7,
    }

    tmp_file = RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    tmp_file.replace(RESULTS_FILE)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} complete in {elapsed:.2f} s. Wrote {RESULTS_FILE}")
    record_stage_checkpoint(STAGE_NAME, [RESULTS_FILE])


if __name__ == "__main__":
    main()
