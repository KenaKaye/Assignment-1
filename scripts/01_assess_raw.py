"""scripts/01_assess_raw.py: Step 0, Item 8.

Assess every delimited segment in the raw corpus (data/GPAC.txt).
Writes: data/sentence_assessment.tsv
Columns: segment_index, byte_offset, status, word_count, duplicate_flag.
"""

from __future__ import annotations

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
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)

STAGE_NAME = "01_assess_raw"
RAW_FILE = DATA_DIR / "GPAC.txt"
ASSESSMENT_FILE = DATA_DIR / "sentence_assessment.tsv"


def classify(seg: str) -> tuple[str, int]:
    """Return (status, word_count) for one stripped segment."""
    words = WORD_RE.findall(seg)
    n_words = len(words)
    if not seg:
        return "empty", n_words
    if n_words == 0:
        return "non_ethiopic", n_words
    if n_words == 1:
        return "too_short", n_words
    return "valid", n_words


def assess_stream(
    in_f,
    out_f,
    chunk_size: int = 2 * 1024 * 1024,
) -> dict:
    """Stream raw text and write TSV assessment.

    Returns dictionary of summary statistics.
    """
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

    seen_hashes: set[int] = set()
    remainder = ""
    current_byte_offset = 0

    out_f.write("segment_index\tbyte_offset\tstatus\tword_count\tduplicate_flag\n")

    while True:
        chunk = in_f.read(chunk_size)
        if not chunk:
            if remainder:
                seg = remainder.strip()
                status, n_words = classify(seg)
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

                out_f.write(f"{total_segments}\t{current_byte_offset}\t{status}\t{n_words}\t{is_dup}\n")
                total_segments += 1
            break

        text = remainder + chunk
        matches = list(SENTENCE_SPLIT_REGEX.finditer(text))

        if not matches:
            remainder = text
            continue

        byte_pos = current_byte_offset
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

            seg = text[last_end : m.start()].strip()
            status, n_words = classify(seg)

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

            out_f.write(f"{total_segments}\t{byte_pos}\t{status}\t{n_words}\t{is_dup}\n")

            byte_pos += len(text[last_end : m.end()].encode("utf-8"))
            total_segments += 1
            last_end = m.end()

        remainder = text[last_end:]
        current_byte_offset = byte_pos

        if total_segments % 500000 < len(matches):
            print(f"  Processed {total_segments:,} segments... (valid={counts['valid']:,})")

    return {
        "total_segments": total_segments,
        "counts": counts,
        "punct_counts": punct_counts,
        "total_valid_tokens": total_valid_tokens,
        "duplicate_count": duplicate_count,
    }


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    if stage_is_done(STAGE_NAME, [RAW_FILE], [ASSESSMENT_FILE]):
        return

    if not RAW_FILE.exists():
        print(f"[ERROR] Raw corpus file not found: {RAW_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    tmp_assessment_file = ASSESSMENT_FILE.with_suffix(".tsv.tmp")
    ASSESSMENT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_FILE, "r", encoding="utf-8", errors="replace") as in_f, open(
        tmp_assessment_file, "w", encoding="utf-8", newline="\n"
    ) as out_f:
        stats = assess_stream(in_f, out_f, chunk_size=2 * 1024 * 1024)

    tmp_assessment_file.replace(ASSESSMENT_FILE)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()

    print("\n--- Assessment Completed ---")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Total Segments: {stats['total_segments']:,}")
    print(f"Status Breakdown: {stats['counts']}")
    print(f"Total Valid Sentences: {stats['counts']['valid']:,}")
    print(f"Total Valid Word Tokens: {stats['total_valid_tokens']:,}")
    print(f"Delimiter Counts: {stats['punct_counts']}")
    print(f"Duplicate Sentences: {stats['duplicate_count']:,}")

    record_stage(STAGE_NAME, [RAW_FILE], [ASSESSMENT_FILE])


if __name__ == "__main__":
    main()