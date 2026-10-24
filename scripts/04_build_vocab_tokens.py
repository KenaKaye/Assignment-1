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
    V_LIMIT,
    WORD_RE,
    get_peak_memory_mb,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)

STAGE_NAME = "04_build_vocab_tokens"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
VOCAB_FILE = DATA_DIR / "vocab.tsv"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    inputs = [CLEANED_FILE]
    outputs = [VOCAB_FILE, TOKENS_FILE, STARTS_FILE]

    if stage_is_done(STAGE_NAME, inputs, outputs):
        return

    if not CLEANED_FILE.exists():
        print(f"[ERROR] Cleaned file not found: {CLEANED_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    # Pass 1: Build vocabulary frequencies and record sentence lengths
    print("Pass 1: Accumulating word frequencies and sentence structure...")
    freqs = Counter()
    total_sentences = 0
    total_words = 0
    sentence_lengths_list: list[int] = []

    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            words = WORD_RE.findall(line)
            n_words = len(words)
            if n_words < 2:
                continue
            total_sentences += 1
            total_words += n_words
            freqs.update(words)
            sentence_lengths_list.append(n_words + 1)  # +1 for EOS_ID

            if line_num % 1000000 == 0:
                print(f"  Counted {line_num:,} lines...")

    total_token_elements = total_words + total_sentences
    vocab_size = len(freqs) + 3

    print(f"\nTotal Sentences: {total_sentences:,}")
    print(f"Total Words: {total_words:,}")
    print(f"Total Elements in tokens.npy (including EOS): {total_token_elements:,}")
    print(f"Vocabulary Size (including reserved): {vocab_size:,}")

    # Check V_LIMIT packing bound
    assert vocab_size < V_LIMIT, f"Vocab size {vocab_size} exceeds V_LIMIT ({V_LIMIT})!"

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

    # Build sentence_starts using vectorized cumsum
    sentence_lengths = np.array(sentence_lengths_list, dtype=np.int64)
    del sentence_lengths_list

    sentence_starts = np.empty(total_sentences, dtype=np.int64)
    if total_sentences > 0:
        sentence_starts[0] = 0
        if total_sentences > 1:
            np.cumsum(sentence_lengths[:-1], out=sentence_starts[1:])
    del sentence_lengths

    # Pass 2: Vectorized chunked token array generation (item 4-1)
    print("\nPass 2: Streaming tokens into contiguous int32 array via np.fromiter...")

    def token_generator():
        with open(CLEANED_FILE, "r", encoding="utf-8") as in_f:
            for l_num, l_str in enumerate(in_f, start=1):
                w_list = WORD_RE.findall(l_str)
                if len(w_list) < 2:
                    continue
                for w in w_list:
                    yield word_to_id.get(w, UNK_ID)
                yield EOS_ID
                if l_num % 1000000 == 0:
                    print(f"  Tokenized {l_num:,} lines...")

    tokens = np.fromiter(token_generator(), dtype=np.int32, count=total_token_elements)

    assert len(tokens) == total_token_elements, "Token array length mismatch!"
    assert len(sentence_starts) == total_sentences, "Sentence starts length mismatch!"

    print(f"Saving {TOKENS_FILE} ({tokens.nbytes / (1024 * 1024):.1f} MB)...")
    np.save(TOKENS_FILE, tokens)
    print(f"Saving {STARTS_FILE} ({sentence_starts.nbytes / (1024 * 1024):.1f} MB)...")
    np.save(STARTS_FILE, sentence_starts)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s (Peak Memory: {peak_mem:.2f} MB)")

    record_stage(STAGE_NAME, inputs, outputs)


if __name__ == "__main__":
    main()
