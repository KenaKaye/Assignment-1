"""scripts/03_verify_cleaned.py: Step 0, Item 10.

Verify the integrity and quality of data/GPAC_cleaned.txt against benchmarks:
- Total lines match valid sentence count from Step 0, Item 8
- Every line contains >= 2 words matching [ሀ-ፚ]+
- No Latin letters or digits present
- Valid UTF-8 decoding without errors
- Ethiopic character share > 80%
- Report top 20 tokens and average sentence length
"""

from __future__ import annotations

from collections import Counter
import re
import sys
import time
from pathlib import Path

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    WORD_RE,
    compute_sha256,
    get_peak_memory_mb,
    require_colab,
)

STAGE_NAME = "03_verify_cleaned"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
ASSESSMENT_FILE = DATA_DIR / "sentence_assessment.tsv"

FORBIDDEN_CHARS_RE = re.compile(r"[a-zA-Z0-9\u1369-\u137C]")  # Latin, ASCII digits, Ethiopic numerals


def main() -> None:
    require_colab(STAGE_NAME)

    if not CLEANED_FILE.exists():
        print(f"[ERROR] Cleaned file not found: {CLEANED_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    # If assessment file exists, get valid count to cross-verify
    expected_valid_lines = None
    if ASSESSMENT_FILE.exists():
        with open(ASSESSMENT_FILE, "r", encoding="utf-8") as f:
            header = f.readline()
            valid_count = 0
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] == "valid":
                    valid_count += 1
            expected_valid_lines = valid_count
            print(f"Cross-check target from sentence_assessment.tsv: {expected_valid_lines:,} valid sentences")

    line_count = 0
    token_count = 0
    total_non_space_chars = 0
    total_ethiopic_chars = 0
    forbidden_violations = 0
    too_short_lines = 0

    word_freq = Counter()

    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_count += 1
            s = line.strip()

            # Character stats
            non_space = [c for c in s if not c.isspace()]
            total_non_space_chars += len(non_space)
            ethiopic_chars = [c for c in non_space if "\u1200" <= c <= "\u137F"]
            total_ethiopic_chars += len(ethiopic_chars)

            # Check forbidden Latin or digits
            if FORBIDDEN_CHARS_RE.search(s):
                forbidden_violations += 1

            # Word extraction
            words = WORD_RE.findall(s)
            n_words = len(words)
            if n_words < 2:
                too_short_lines += 1

            token_count += n_words
            word_freq.update(words)

            if line_num % 1000000 == 0:
                print(f"  Verified {line_num:,} lines...")

    sha256_hash = compute_sha256(CLEANED_FILE)
    ethiopic_share = (total_ethiopic_chars / total_non_space_chars * 100.0) if total_non_space_chars > 0 else 0.0
    avg_len = token_count / line_count if line_count > 0 else 0.0

    print("\n--- Verification Report for GPAC_cleaned.txt ---")
    print(f"File Size: {CLEANED_FILE.stat().st_size:,} bytes")
    print(f"SHA-256 Checksum: {sha256_hash}")
    print(f"Total Lines (Sentences): {line_count:,}")
    print(f"Total Word Tokens: {token_count:,}")
    print(f"Average Sentence Length: {avg_len:.2f} tokens")
    print(f"Vocabulary Size (Unique Types): {len(word_freq):,}")
    print(f"Ethiopic Non-Space Share: {ethiopic_share:.2f}% (Benchmark: > 80%)")
    print(f"Lines with < 2 words: {too_short_lines} (Benchmark: 0)")
    print(f"Lines with Latin/digits: {forbidden_violations} (Benchmark: 0)")

    print("\nTop 20 Frequent Words:")
    for rank, (w, count) in enumerate(word_freq.most_common(20), start=1):
        print(f"  {rank:2d}. {w:15s} : {count:,} ({count / token_count * 100:.2f}%)")

    # Assertions for pass criteria
    assert too_short_lines == 0, f"Failed: found {too_short_lines} lines with fewer than 2 words"
    assert forbidden_violations == 0, f"Failed: found {forbidden_violations} lines with forbidden characters"
    assert ethiopic_share > 80.0, f"Failed: Ethiopic share {ethiopic_share:.2f}% is below 80%"
    if expected_valid_lines is not None:
        assert line_count == expected_valid_lines, (
            f"Failed: Line count {line_count} != expected {expected_valid_lines}"
        )

    print("\n[SUCCESS] All Step 0, Item 10 verification criteria PASSED!")


if __name__ == "__main__":
    main()
