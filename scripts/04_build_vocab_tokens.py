"""scripts/04_build_vocab_tokens.py: Step 0, Item 11.

Builds from data/GPAC_cleaned.txt:
- data/vocab.tsv: ID, word, frequency. IDs 0=<s>, 1=</s>, 2=<UNK>, 3+ sorted by frequency desc.
- data/tokens.npy: int32 array containing all token IDs with </s> (ID 1) appended to each sentence.
- data/sentence_starts.npy: int64 array containing start index of each sentence in tokens.npy.
"""

from __future__ import annotations

from collections import Counter
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
    WORD_RE,
    get_peak_memory_mb,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)

STAGE_NAME = "04_build_vocab_tokens"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
VOCAB_FILE = DATA_DIR / "vocab.tsv"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"


def main() -> None:
    require_colab(STAGE_NAME)

    expected_files = [VOCAB_FILE, TOKENS_FILE, STARTS_FILE]
    if is_checkpoint_done(STAGE_NAME, expected_files):
        return

    if not CLEANED_FILE.exists():
        print(f"[ERROR] Cleaned file not found: {CLEANED_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    # Pass 1: Build vocabulary frequencies and count total elements
    print("Pass 1: Accumulating word frequencies and sentence structure...")
    freqs = Counter()
    total_sentences = 0
    total_words = 0

    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            words = WORD_RE.findall(line)
            if len(words) < 2:
                continue
            total_sentences += 1
            total_words += len(words)
            freqs.update(words)
            if line_num % 1000000 == 0:
                print(f"  Counted {line_num:,} lines...")

    total_token_elements = total_words + total_sentences  # Each sentence ends with </s> (EOS)
    vocab_size = len(freqs) + 3  # + <s>, </s>, <UNK>

    print(f"\nTotal Sentences: {total_sentences:,}")
    print(f"Total Words: {total_words:,}")
    print(f"Total Elements in tokens.npy (including EOS): {total_token_elements:,}")
    print(f"Vocabulary Size (including reserved): {vocab_size:,}")

    # Check that vocabulary fits within 2^21 (2,097,152)
    assert vocab_size < 2097152, f"Vocab size {vocab_size} exceeds 2^21 benchmark!"

    # Write vocab.tsv
    print(f"Writing vocabulary to {VOCAB_FILE}...")
    word_to_id: dict[str, int] = {}
    tmp_vocab = VOCAB_FILE.with_suffix(".tsv.tmp")
    VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_vocab, "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tword\tfrequency\n")
        f.write(f"{PAD_ID}\t{PAD_TOKEN}\t0\n")
        f.write(f"{EOS_ID}\t{EOS_TOKEN}\t{total_sentences}\n")
        f.write(f"{UNK_ID}\t{UNK_TOKEN}\t0\n")

        curr_id = 3
        for word, count in freqs.most_common():
            word_to_id[word] = curr_id
            f.write(f"{curr_id}\t{word}\t{count}\n")
            curr_id += 1

    tmp_vocab.replace(VOCAB_FILE)
    del freqs
    print(f"Saved {VOCAB_FILE}")

    # Pass 2: Allocate and populate tokens.npy and sentence_starts.npy
    print("\nPass 2: Populating int32 tokens array and int64 sentence starts...")
    tokens = np.empty(total_token_elements, dtype=np.int32)
    sentence_starts = np.empty(total_sentences, dtype=np.int64)

    curr_token_idx = 0
    curr_sent_idx = 0

    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            words = WORD_RE.findall(line)
            if len(words) < 2:
                continue

            sentence_starts[curr_sent_idx] = curr_token_idx
            curr_sent_idx += 1

            for w in words:
                tokens[curr_token_idx] = word_to_id.get(w, UNK_ID)
                curr_token_idx += 1

            # Append EOS
            tokens[curr_token_idx] = EOS_ID
            curr_token_idx += 1

            if line_num % 1000000 == 0:
                print(f"  Tokenized {line_num:,} lines...")

    assert curr_token_idx == total_token_elements, "Token index mismatch!"
    assert curr_sent_idx == total_sentences, "Sentence index mismatch!"

    print(f"Saving {TOKENS_FILE} ({tokens.nbytes / (1024 * 1024):.1f} MB)...")
    np.save(TOKENS_FILE, tokens)
    print(f"Saving {STARTS_FILE} ({sentence_starts.nbytes / (1024 * 1024):.1f} MB)...")
    np.save(STARTS_FILE, sentence_starts)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()

    print("\n--- Vocabulary and Tokenization Completed ---")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Tokens Shape: {tokens.shape}, Dtype: {tokens.dtype}")
    print(f"Sentence Starts Shape: {sentence_starts.shape}, Dtype: {sentence_starts.dtype}")

    record_stage_checkpoint(STAGE_NAME, expected_files)


if __name__ == "__main__":
    main()
