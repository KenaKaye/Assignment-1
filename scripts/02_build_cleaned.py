"""scripts/02_build_cleaned.py: Step 0, Item 9.

Build data/GPAC_cleaned.txt through a streaming normalization pass over data/GPAC.txt.
Applies:
- NFC Unicode normalization
- Character filter: Keep Ethiopic letters (ሀ-ፚ), whitespace, and punctuation (።, ፡, ፣, ?, !)
- Sentence splitting on [።?!]|፡፡+
- Whitespace collapse & strip
- Validity filter: minimum 2 words matching [ሀ-ፚ]+
- Output: 1 sentence per line, UTF-8, LF line endings
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    ETHIOPIC_FILTER_REGEX,
    SENTENCE_SPLIT_REGEX,
    WORD_RE,
    get_peak_memory_mb,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)

STAGE_NAME = "02_build_cleaned"
RAW_FILE = DATA_DIR / "GPAC.txt"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"


def main() -> None:
    require_colab(STAGE_NAME)

    if is_checkpoint_done(STAGE_NAME, [CLEANED_FILE]):
        return

    if not RAW_FILE.exists():
        print(f"[ERROR] Raw corpus file not found: {RAW_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    chunk_size = 2 * 1024 * 1024  # 2 MB chunks
    remainder = ""
    valid_sentences = 0
    total_tokens = 0
    nfc_changed_chars = 0

    tmp_cleaned_file = CLEANED_FILE.with_suffix(".txt.tmp")
    CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_FILE, "r", encoding="utf-8", errors="replace") as in_f, open(
        tmp_cleaned_file, "w", encoding="utf-8", newline="\n"
    ) as out_f:
        while True:
            chunk = in_f.read(chunk_size)
            if not chunk:
                # Handle end of stream
                if remainder:
                    nfc_chunk = unicodedata.normalize("NFC", remainder)
                    filtered = ETHIOPIC_FILTER_REGEX.sub(" ", nfc_chunk)
                    sentence = re.sub(r"\s+", " ", filtered).strip()
                    words = WORD_RE.findall(sentence)
                    if len(words) >= 2:
                        out_f.write(sentence + "\n")
                        valid_sentences += 1
                        total_tokens += len(words)
                break

            # Carry remainder across chunk boundary
            text = remainder + chunk

            # Find split boundaries
            matches = list(SENTENCE_SPLIT_REGEX.finditer(text))
            if not matches:
                remainder = text
                continue

            last_end = 0
            for m in matches:
                segment_raw = text[last_end : m.start()]
                last_end = m.end()

                # NFC normalization
                nfc_seg = unicodedata.normalize("NFC", segment_raw)
                if nfc_seg != segment_raw:
                    nfc_changed_chars += sum(1 for c1, c2 in zip(segment_raw, nfc_seg) if c1 != c2) + abs(
                        len(segment_raw) - len(nfc_seg)
                    )

                # Character filtering
                filtered = ETHIOPIC_FILTER_REGEX.sub(" ", nfc_seg)

                # Whitespace collapse & trimming
                sentence = re.sub(r"\s+", " ", filtered).strip()

                # Validity check: at least 2 Ethiopic words
                words = WORD_RE.findall(sentence)
                if len(words) >= 2:
                    out_f.write(sentence + "\n")
                    valid_sentences += 1
                    total_tokens += len(words)

                    if valid_sentences % 500000 == 0:
                        print(f"  Written {valid_sentences:,} cleaned sentences... ({total_tokens:,} tokens)")

            remainder = text[last_end:]

    # Finalize atomically
    tmp_cleaned_file.replace(CLEANED_FILE)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()
    avg_length = total_tokens / valid_sentences if valid_sentences > 0 else 0

    print("\n--- Cleaned Corpus Build Completed ---")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Valid Cleaned Sentences (Lines): {valid_sentences:,}")
    print(f"Total Word Tokens: {total_tokens:,}")
    print(f"Average Sentence Length: {avg_length:.2f} tokens")
    print(f"NFC Changed Characters: {nfc_changed_chars:,}")

    record_stage_checkpoint(STAGE_NAME, [CLEANED_FILE])


if __name__ == "__main__":
    main()
