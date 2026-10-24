"""scripts/08_count_train.py: Q2 Step 2.

Builds training-split vocabulary, lookup table (LUT), and n-gram counts for Q2, Q3, and Q4.
Supports --seed 42 and --seed 7 over the train_core partition (D2, D4).
Produces:
- data/counts/train_seed{seed}/vocab_train.tsv (preserving original global IDs, D8)
- data/counts/train_seed{seed}/lut_train_seed{seed}.npy (int32 array mapping <=1 to UNK_ID)
- data/counts/train_seed{seed}/order{n}_ngrams.npy and order{n}_counts.npy (n=1 to 4)
"""

from __future__ import annotations

import argparse
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
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    V_LIMIT,
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

TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
VOCAB_FULL_FILE = DATA_DIR / "vocab.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training-split counts and LUT.")
    parser.add_argument("--seed", type=int, choices=[42, 7], default=42, help="Split seed (42 or 7)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    stage_name = f"08_count_train_seed{seed}"
    require_heavy_runtime(stage_name)

    target_dir = DATA_DIR / "counts" / f"train_seed{seed}"
    split_file = DATA_DIR / f"split_indices_seed{seed}.json"
    if not split_file.exists() and (DATA_DIR / f"split_indices_seed{seed}_v2.json").exists():
        split_file = DATA_DIR / f"split_indices_seed{seed}_v2.json"
    lut_file = target_dir / f"lut_train_seed{seed}.npy"
    vocab_train_file = target_dir / "vocab_train.tsv"

    expected_files = [lut_file, vocab_train_file]
    for n in range(1, 5):
        expected_files.append(target_dir / f"order{n}_ngrams.npy")
        expected_files.append(target_dir / f"order{n}_counts.npy")

    inputs = [TOKENS_FILE, STARTS_FILE, split_file, VOCAB_FULL_FILE]
    if stage_is_done(stage_name, inputs, expected_files):
        return

    if not TOKENS_FILE.exists() or not STARTS_FILE.exists() or not split_file.exists():
        print(f"[ERROR] Required inputs not found ({TOKENS_FILE}, {STARTS_FILE}, {split_file})")
        sys.exit(1)

    print(f"--- Starting Stage: {stage_name} (Seed {seed}) ---")
    budget_mb = memory_budget_mb()
    print(f"Operational Memory Budget: {budget_mb:.1f} MB")
    start_time = time.time()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading split partition from {split_file.name}...")
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_indices = np.array(split_data["train_core"], dtype=np.int64)

    print(f"Loading tokens and sentence starts...")
    tokens = np.load(TOKENS_FILE)
    starts = np.load(STARTS_FILE)
    n_sentences = len(starts)

    # Compute sentence lengths vectorized (item 8-4)
    ends = np.empty(n_sentences, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = len(tokens)

    train_lengths = ends[train_indices] - starts[train_indices]
    total_train_tokens = int(train_lengths.sum())
    n_train_sents = len(train_indices)

    print(f"train_core partition: {n_train_sents:,} sentences, {total_train_tokens:,} tokens (including EOS).")

    # Extract train_tokens sequence
    print("Extracting train_core token sequence...")
    train_tokens = np.empty(total_train_tokens, dtype=np.int32)
    train_starts = np.empty(n_train_sents, dtype=np.int64)

    curr_dst = 0
    for i, s_idx in enumerate(train_indices):
        train_starts[i] = curr_dst
        s_len = train_lengths[i]
        s_start = starts[s_idx]
        train_tokens[curr_dst : curr_dst + s_len] = tokens[s_start : s_start + s_len]
        curr_dst += s_len

    assert curr_dst == total_train_tokens
    del tokens, starts, ends, train_indices, train_lengths
    gc.collect()

    # Pass 1: Vectorized bincount to identify singletons (item 8-1)
    print("Computing word frequencies via np.bincount...")
    count_train = np.bincount(train_tokens, minlength=V_LIMIT)

    # Build Lookup Table (LUT, item 8-2)
    print(f"Constructing Lookup Table (LUT) of size {V_LIMIT:,}...")
    lut = np.full(V_LIMIT, UNK_ID, dtype=np.int32)
    non_singletons = np.flatnonzero(count_train > 1)
    lut[non_singletons] = non_singletons.astype(np.int32)

    lut[PAD_ID] = PAD_ID
    lut[EOS_ID] = EOS_ID
    lut[UNK_ID] = UNK_ID

    # Save LUT
    tmp_lut = lut_file.with_suffix(".npy.tmp")
    np.save(tmp_lut, lut)
    tmp_lut.replace(lut_file)
    print(f"Saved {lut_file.name}")

    # Map singletons in train_tokens using LUT
    print("Applying LUT to train_tokens...")
    train_tokens = lut[train_tokens]

    # Write vocab_train.tsv preserving original global IDs (item 8-3)
    print("Writing vocab_train.tsv with preserved global IDs...")
    id_to_word: dict[int, str] = {}
    with open(VOCAB_FULL_FILE, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                id_to_word[int(parts[0])] = parts[1]

    tmp_vocab = vocab_train_file.with_suffix(".tsv.tmp")
    with open(tmp_vocab, "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tword\tfrequency\n")
        f.write(f"{PAD_ID}\t{PAD_TOKEN}\t0\n")
        f.write(f"{EOS_ID}\t{EOS_TOKEN}\t{n_train_sents}\n")
        unk_count = int(np.count_nonzero(train_tokens == UNK_ID))
        f.write(f"{UNK_ID}\t{UNK_TOKEN}\t{unk_count}\n")

        valid_words = [w_id for w_id in non_singletons if w_id >= 3]
        valid_words.sort(key=lambda w: count_train[w], reverse=True)
        for w_id in valid_words:
            w_str = id_to_word.get(w_id, f"<ID_{w_id}>")
            f.write(f"{w_id}\t{w_str}\t{count_train[w_id]}\n")

    tmp_vocab.replace(vocab_train_file)
    print(f"Saved {vocab_train_file.name} ({len(valid_words) + 3:,} vocabulary entries)")
    del count_train, non_singletons, valid_words, id_to_word
    gc.collect()

    # Vectorized counting for orders 1 to 4 with upfront partitioning & per-order checkpoints
    print("\nComputing sentence positions for train_tokens...")
    pos_in_sent = sentence_positions(train_starts, total_train_tokens)
    del train_starts
    gc.collect()

    for n in range(1, 5):
        order_stage = f"{stage_name}_order{n}"
        ngrams_file = target_dir / f"order{n}_ngrams.npy"
        counts_file = target_dir / f"order{n}_counts.npy"
        order_outputs = [ngrams_file, counts_file]

        if stage_is_done(order_stage, inputs, order_outputs):
            print(f"Order n={n} verified complete. Skipping.")
            continue

        t0 = time.time()
        print(f"\nCounting training order n = {n}...")
        matrix = context_matrix(train_tokens, pos_in_sent, n)

        cur_mem = get_peak_memory_mb()
        partitioned = use_partition(total_train_tokens, n, budget_mb, current_usage_mb=cur_mem)

        if partitioned:
            print(f"  Estimated memory exceeds budget ({budget_mb:.1f} MB) -> Using group_rows_partitioned(parts=4)...")
            unique_ngrams, counts = group_rows_partitioned(matrix, parts=4)
        else:
            print(f"  Direct lexsort group_rows fits within budget ({budget_mb:.1f} MB)...")
            unique_ngrams, counts = group_rows(matrix)

        del matrix
        gc.collect()

        assert counts.sum() == total_train_tokens, f"Order {n} sum mismatch: {counts.sum()} != {total_train_tokens}"

        tmp_ngrams = ngrams_file.with_suffix(".npy.tmp")
        tmp_counts = counts_file.with_suffix(".npy.tmp")
        np.save(tmp_ngrams, unique_ngrams)
        np.save(tmp_counts, counts)
        tmp_ngrams.replace(ngrams_file)
        tmp_counts.replace(counts_file)

        elapsed_n = round(time.time() - t0, 2)
        print(
            f"  Finished order {n}: {len(unique_ngrams):,} distinct n-grams, "
            f"{counts.sum():,} tokens in {elapsed_n} s"
        )

        record_stage(order_stage, inputs, order_outputs)
        del unique_ngrams, counts
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {stage_name} completed in {elapsed:.2f} s")
    record_stage(stage_name, inputs, expected_files)


if __name__ == "__main__":
    main()
