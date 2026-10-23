"""scripts/01_assess_raw.py: Step 0, Item 8.

Assess every delimited segment in the raw corpus (data/GPAC.txt).
Writes: data/sentence_assessment.tsv
Columns: segment_index, byte_offset, status, word_count, duplicate_flag.
"""

from __future__ import annotations

import gc
import hashlib
import sys
import time
from pathlib import Path

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    SENTENCE_SPLIT_REGEX,
    WORD_RE,
    get_peak_memory_mb,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)

STAGE_NAME = "01_assess_raw"
RAW_FILE = DATA_DIR / "GPAC.txt"
ASSESSMENT_FILE = DATA_DIR / "sentence_assessment.tsv"


def main() -> None:
    require_colab(STAGE_NAME)

    if is_checkpoint_done(STAGE_NAME, [ASSESSMENT_FILE]):
        return

    if not RAW_FILE.exists():
        print(f"[ERROR] Raw corpus file not found: {RAW_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    # Track status counts
    counts = {
        "valid": 0,
        "too_short": 0,
        "empty": 0,
        "non_ethiopic": 0,
    }
    punct_counts = {
        "።": 0,
        "?": 0,
        "!": 0,
        "፡፡": 0,
    }
    total_valid_tokens = 0
    duplicate_count = 0
    total_segments = 0

    # Store compact 64-bit sentence hashes to identify duplicates with minimal RAM footprint
    seen_hashes: set[int] = set()

    # Streaming read in 2MB chunks with remainder carry-forward
    chunk_size = 2 * 1024 * 1024  # 2 MB
    remainder = ""
    current_byte_offset = 0

    tmp_assessment_file = ASSESSMENT_FILE.with_suffix(".tsv.tmp")
    ASSESSMENT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_FILE, "r", encoding="utf-8", errors="replace") as in_f, open(
        tmp_assessment_file, "w", encoding="utf-8", newline="\n"
    ) as out_f:
        # TSV Header
        out_f.write("segment_index\tbyte_offset\tstatus\tword_count\tduplicate_flag\n")

        while True:
            chunk = in_f.read(chunk_size)
            if not chunk:
                # Process final remaining segment
                if remainder:
                    total_segments += 1
                    seg = remainder.strip()
                    words = WORD_RE.findall(seg)
                    n_words = len(words)

                    if not seg:
                        status = "empty"
                    elif n_words == 0:
                        status = "non_ethiopic"
                    elif n_words == 1:
                        status = "too_short"
                    else:
                        status = "valid"

                    counts[status] += 1
                    is_dup = 0
                    if status == "valid":
                        total_valid_tokens += n_words
                        h = int.from_bytes(hashlib.md5(seg.encode("utf-8")).digest()[:8], "little")
                        if h in seen_hashes:
                            is_dup = 1
                            duplicate_count += 1
                        else:
                            seen_hashes.add(h)

                    out_f.write(f"{total_segments - 1}\t{current_byte_offset}\t{status}\t{n_words}\t{is_dup}\n")
                break

            text = remainder + chunk
            # Find all delimiter matches in this block
            matches = list(SENTENCE_SPLIT_REGEX.finditer(text))

            if not matches:
                remainder = text
                continue

            last_end = 0
            for m in matches:
                delimiter = m.group()
                if "፡፡" in delimiter:
                    punct_counts["፡፡"] += 1
                elif "።" in delimiter:
                    punct_counts["።"] += 1
                elif "?" in delimiter:
                    punct_counts["?"] += 1
                elif "!" in delimiter:
                    punct_counts["!"] += 1

                segment_text = text[last_end : m.start()]
                seg = segment_text.strip()
                words = WORD_RE.findall(seg)
                n_words = len(words)

                if not seg:
                    status = "empty"
                elif n_words == 0:
                    status = "non_ethiopic"
                elif n_words == 1:
                    status = "too_short"
                else:
                    status = "valid"

                counts[status] += 1
                is_dup = 0
                if status == "valid":
                    total_valid_tokens += n_words
                    h = int.from_bytes(hashlib.md5(seg.encode("utf-8")).digest()[:8], "little")
                    if h in seen_hashes:
                        is_dup = 1
                        duplicate_count += 1
                    else:
                        seen_hashes.add(h)

                seg_offset = current_byte_offset + len(text[:last_end].encode("utf-8"))
                out_f.write(f"{total_segments}\t{seg_offset}\t{status}\t{n_words}\t{is_dup}\n")

                total_segments += 1
                last_end = m.end()

            # Carry un-delimited text to next chunk
            remainder = text[last_end:]
            # Advance byte counter by processed text length
            current_byte_offset += len(text[:last_end].encode("utf-8"))

            if total_segments % 500000 == 0:
                print(f"  Processed {total_segments:,} segments... (valid={counts['valid']:,})")

    # Finalize file atomically
    tmp_assessment_file.replace(ASSESSMENT_FILE)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()

    print("\n--- Assessment Completed ---")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Total Segments: {total_segments:,}")
    print(f"Status Breakdown: {counts}")
    print(f"Total Valid Sentences: {counts['valid']:,}")
    print(f"Total Valid Word Tokens: {total_valid_tokens:,}")
    print(f"Delimiter Counts: {punct_counts}")
    print(f"Duplicate Sentences: {duplicate_count:,}")

    record_stage_checkpoint(STAGE_NAME, [ASSESSMENT_FILE])


if __name__ == "__main__":
    main()
