"""scripts/08_count_train.py: Q2 Step 2.

Builds training-split vocabulary and n-gram counts:
- Maps singleton words in training set to <UNK>
- Produces data/counts/train_seed42/vocab_train.tsv
- Produces data/counts/train_seed42/order{n}_ngrams.npy and order{n}_counts.npy (n=1 to 4)
"""

from __future__ import annotations

from collections import Counter
import gc
import importlib
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
    TRAIN_COUNTS_DIR,
    UNK_ID,
    UNK_TOKEN,
    get_peak_memory_mb,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)

scripts_05 = importlib.import_module("scripts.05_profile_counts")
build_ngram_matrix = scripts_05.build_ngram_matrix
compute_pos_in_sent = scripts_05.compute_pos_in_sent
count_ngrams_direct = scripts_05.count_ngrams_direct
count_ngrams_partitioned = scripts_05.count_ngrams_partitioned

STAGE_NAME = "08_count_train"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
SPLIT_FILE = DATA_DIR / "split_indices_seed42.json"
VOCAB_FULL_FILE = DATA_DIR / "vocab.tsv"


def main() -> None:
    require_colab(STAGE_NAME)

    expected_files = [TRAIN_COUNTS_DIR / "vocab_train.tsv"]
    for n in range(1, 5):
        expected_files.append(TRAIN_COUNTS_DIR / f"order{n}_ngrams.npy")
        expected_files.append(TRAIN_COUNTS_DIR / f"order{n}_counts.npy")

    if is_checkpoint_done(STAGE_NAME, expected_files):
        return

    if not TOKENS_FILE.exists() or not STARTS_FILE.exists() or not SPLIT_FILE.exists():
        print(f"[ERROR] Required inputs not found ({TOKENS_FILE}, {STARTS_FILE}, {SPLIT_FILE})")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()
    TRAIN_COUNTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data split and sentence starts...")
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_indices = np.array(split_data["train"], dtype=np.int64)

    tokens = np.load(TOKENS_FILE)
    starts = np.load(STARTS_FILE)
    n_sentences = len(starts)

    ends = np.empty(n_sentences, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = len(tokens)

    # Calculate total training tokens
    train_lengths = ends[train_indices] - starts[train_indices]
    total_train_tokens = int(train_lengths.sum())
    n_train_sents = len(train_indices)

    print(f"Training split: {n_train_sents:,} sentences, {total_train_tokens:,} tokens (including EOS).")

    # Assemble contiguous train_tokens array and train_starts array
    print("Extracting training token sequence...")
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

    # Free full tokens and starts from memory
    del tokens, starts, ends, train_indices, train_lengths
    gc.collect()

    # Pass 1: Count word frequencies in train_tokens
    print("Counting training word frequencies to identify singletons...")
    train_counts = Counter()
    for token_id in train_tokens:
        if token_id >= 3:
            train_counts[token_id] += 1

    singletons = {t_id for t_id, cnt in train_counts.items() if cnt == 1}
    print(f"Total vocabulary seen in train: {len(train_counts):,}")
    print(f"Singletons (frequency == 1) mapped to <UNK>: {len(singletons):,}")

    # Map singletons to UNK_ID in train_tokens
    print("Remapping singletons to <UNK> in train_tokens...")
    if singletons:
        # Create lookup table for fast vector remap
        max_id = max(train_counts.keys())
        remap_lut = np.arange(max_id + 1, dtype=np.int32)
        for s_id in singletons:
            remap_lut[s_id] = UNK_ID
        train_tokens = remap_lut[train_tokens]
        del remap_lut
        gc.collect()

    # Rebuild vocab_train.tsv
    print("Writing data/counts/train_seed42/vocab_train.tsv...")
    # Load original word mappings from full vocab.tsv
    id_to_word = {}
    with open(VOCAB_FULL_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                id_to_word[int(parts[0])] = parts[1]

    train_vocab_file = TRAIN_COUNTS_DIR / "vocab_train.tsv"
    tmp_vocab = train_vocab_file.with_suffix(".tsv.tmp")
    with open(tmp_vocab, "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tword\tfrequency\n")
        f.write(f"{PAD_ID}\t{PAD_TOKEN}\t0\n")
        f.write(f"{EOS_ID}\t{EOS_TOKEN}\t{n_train_sents}\n")
        # Count total UNK in train_tokens
        unk_total = len(singletons)
        f.write(f"{UNK_ID}\t{UNK_TOKEN}\t{unk_total}\n")

        # Non-singletons
        kept_words = [(id_to_word[t_id], cnt) for t_id, cnt in train_counts.items() if cnt > 1]
        kept_words.sort(key=lambda x: x[1], reverse=True)
        for rank, (w, cnt) in enumerate(kept_words, start=3):
            f.write(f"{rank}\t{w}\t{cnt}\n")

    tmp_vocab.replace(train_vocab_file)
    print(f"Saved {train_vocab_file} ({len(kept_words) + 3:,} words).")
    del train_counts, singletons, id_to_word
    gc.collect()

    # Vectorized counting for training split
    print("\nComputing training token sentence positions...")
    train_pos_in_sent = compute_pos_in_sent(len(train_tokens), train_starts)
    del train_starts
    gc.collect()

    for n in range(1, 5):
        t0 = time.time()
        print(f"\nBuilding training counts for order n = {n}...")
        matrix = build_ngram_matrix(train_tokens, train_pos_in_sent, n)

        unique_ngrams, counts = count_ngrams_direct(matrix)
        del matrix
        gc.collect()

        ngrams_file = TRAIN_COUNTS_DIR / f"order{n}_ngrams.npy"
        counts_file = TRAIN_COUNTS_DIR / f"order{n}_counts.npy"

        print(f"  Saving {ngrams_file.name} (shape {unique_ngrams.shape}) and {counts_file.name}...")
        np.save(ngrams_file, unique_ngrams)
        np.save(counts_file, counts)

        elapsed_n = time.time() - t0
        print(f"  Finished order {n}: {len(unique_ngrams):,} unique, {counts.sum():,} tokens in {elapsed_n:.2f} s")
        del unique_ngrams, counts
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s")
    record_stage_checkpoint(STAGE_NAME, expected_files)


if __name__ == "__main__":
    main()
