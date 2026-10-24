"""scripts/03_verify_cleaned.py: Step 0, Item 10.

Verify the integrity and quality of data/GPAC_cleaned.txt against benchmarks:
- Total lines match valid sentence count from Step 0, Item 8
- Every line contains >= 2 words matching [ሀ-ፚ]+
- No Latin letters or digits present
- Valid UTF-8 decoding without errors
- Ethiopic letters-only share > 80% (benchmark applies to letters only)
- Report both letters-only share and letters-plus-punctuation share
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
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)

STAGE_NAME = "03_verify_cleaned"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
ASSESSMENT_FILE = DATA_DIR / "sentence_assessment.tsv"
RECORD_FILE = DATA_DIR / "cleaned_verification_record.json"

FORBIDDEN_CHARS_RE = re.compile(r"[a-zA-Z0-9\u1369-\u137C]")
ETHIOPIC_LETTERS_RE = re.compile(r"[\u1200-\u135A]")
NON_SPACE_RE = re.compile(r"\S")
ETHIOPIC_PUNCT_RE = re.compile(r"[\u1200-\u135A።፡፣?!]")


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    inputs = [CLEANED_FILE]
    outputs = [RECORD_FILE]
    if ASSESSMENT_FILE.exists():
        inputs.append(ASSESSMENT_FILE)

    if stage_is_done(STAGE_NAME, inputs, outputs):
        return

    if not CLEANED_FILE.exists():
        print(f"[ERROR] Cleaned file not found: {CLEANED_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    expected_valid_lines = None
    if ASSESSMENT_FILE.exists():
        with open(ASSESSMENT_FILE, "r", encoding="utf-8") as f:
            f.readline()  # header
            valid_count = sum(1 for line in f if "\tvalid\t" in line)
            expected_valid_lines = valid_count
            print(f"Cross-check target from sentence_assessment.tsv: {expected_valid_lines:,} valid sentences")

    line_count = 0
    token_count = 0
    total_non_space_chars = 0
    total_ethiopic_letters = 0
    total_ethiopic_with_punct = 0
    forbidden_violations = 0
    too_short_lines = 0

    word_freq = Counter()

    # Pass 1: Chunked character regex counts (item 3-1)
    chunk_size = 2 * 1024 * 1024
    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_non_space_chars += len(NON_SPACE_RE.findall(chunk))
            total_ethiopic_letters += len(ETHIOPIC_LETTERS_RE.findall(chunk))
            total_ethiopic_with_punct += len(ETHIOPIC_PUNCT_RE.findall(chunk))

    # Pass 2: Line validation and word stats
    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_count += 1
            s = line.strip()

            if FORBIDDEN_CHARS_RE.search(s):
                forbidden_violations += 1

            words = WORD_RE.findall(s)
            n_words = len(words)
            if n_words < 2:
                too_short_lines += 1

            token_count += n_words
            if line_num <= 100000:
                word_freq.update(words)

            if line_num % 1000000 == 0:
                print(f"  Verified {line_num:,} lines...")

    sha256_hash = compute_sha256(CLEANED_FILE)
    letters_only_share = (
        (total_ethiopic_letters / total_non_space_chars * 100.0) if total_non_space_chars > 0 else 0.0
    )
    letters_with_punct_share = (
        (total_ethiopic_with_punct / total_non_space_chars * 100.0) if total_non_space_chars > 0 else 0.0
    )
    avg_len = token_count / line_count if line_count > 0 else 0.0

    print("\n--- Verification Report for GPAC_cleaned.txt ---")
    print(f"File Size: {CLEANED_FILE.stat().st_size:,} bytes")
    print(f"SHA-256 Checksum: {sha256_hash}")
    print(f"Total Lines (Sentences): {line_count:,}")
    print(f"Total Word Tokens: {token_count:,}")
    print(f"Average Sentence Length: {avg_len:.2f} tokens")
    print(f"Ethiopic Letters-Only Share: {letters_only_share:.2f}% (Benchmark: > 80.0%)")
    print(f"Ethiopic + Punctuation Share: {letters_with_punct_share:.2f}%")
    print(f"Lines with < 2 words: {too_short_lines} (Benchmark: 0)")
    print(f"Lines with Latin/digits: {forbidden_violations} (Benchmark: 0)")

    assert too_short_lines == 0, f"Failed: found {too_short_lines} lines with fewer than 2 words"
    assert forbidden_violations == 0, f"Failed: found {forbidden_violations} lines with forbidden characters"
    assert letters_only_share > 80.0, f"Failed: Ethiopic letters share {letters_only_share:.2f}% is below 80%"
    if expected_valid_lines is not None:
        assert line_count == expected_valid_lines, (
            f"Failed: Line count {line_count} != expected {expected_valid_lines}"
        )

    import json
    record_data = {
        "file_size": CLEANED_FILE.stat().st_size,
        "sha256": sha256_hash,
        "line_count": line_count,
        "token_count": token_count,
        "avg_length": round(avg_len, 2),
        "ethiopic_letters_only_share": round(letters_only_share, 2),
        "ethiopic_letters_punct_share": round(letters_with_punct_share, 2),
        "too_short_lines": too_short_lines,
        "forbidden_violations": forbidden_violations,
    }
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(record_data, f, indent=2)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()
    print(f"Stage completed in {elapsed:.2f} s (Peak Memory: {peak_mem:.2f} MB)")
    print("\n[SUCCESS] All Step 0, Item 10 verification criteria PASSED!")

    record_stage(STAGE_NAME, inputs, outputs)


if __name__ == "__main__":
    main()
