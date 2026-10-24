"""scripts/02_build_cleaned.py: Step 0, Item 9.

Build data/GPAC_cleaned.txt through a streaming normalization pass over data/GPAC.txt.
Keeps sentence-final punctuation (D5): each line ends with ።, ?, or !. ፡፡+ is replaced by ።.
"""

from __future__ import annotations

import codecs
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
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)

STAGE_NAME = "02_build_cleaned"
RAW_FILE = DATA_DIR / "GPAC.txt"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"


def normalize_delimiter(delimiter: str) -> str:
    """Normalize delimiter to canonical sentence-final punctuation (D5)."""
    if "፡፡" in delimiter or "።" in delimiter:
        return "።"
    if "?" in delimiter:
        return "?"
    if "!" in delimiter:
        return "!"
    return "።"


def clean_stream(
    in_bytes_f,
    out_f,
    chunk_bytes: int = 2 * 1024 * 1024,
) -> dict:
    """Stream raw bytes and write cleaned sentences with sentence-final punctuation.

    Args:
        in_bytes_f: Binary stream for raw input file.
        out_f: Text stream for cleaned output file (newline=\\n).
        chunk_bytes: Number of bytes to read per chunk.

    Returns:
        Dictionary of cleaning statistics.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    remainder = ""
    valid_sentences = 0
    total_tokens = 0
    nfc_changed_segments = 0
    replacement_chars = 0

    while True:
        chunk = in_bytes_f.read(chunk_bytes)
        is_eof = not chunk
        text_chunk = decoder.decode(chunk, final=is_eof)
        replacement_chars += text_chunk.count("\ufffd")

        text = remainder + text_chunk

        if is_eof:
            # Handle any remaining text at EOF
            if text:
                matches = list(SENTENCE_SPLIT_REGEX.finditer(text))
                last_end = 0
                for m in matches:
                    delim_norm = normalize_delimiter(m.group())
                    seg_raw = text[last_end : m.start()].strip()
                    last_end = m.end()

                    nfc_seg = unicodedata.normalize("NFC", seg_raw)
                    if nfc_seg != seg_raw:
                        nfc_changed_segments += 1

                    filtered = ETHIOPIC_FILTER_REGEX.sub(" ", nfc_seg)
                    sentence = re.sub(r"\s+", " ", filtered).strip()
                    words = WORD_RE.findall(sentence)
                    if len(words) >= 2:
                        out_f.write(f"{sentence}{delim_norm}\n")
                        valid_sentences += 1
                        total_tokens += len(words)

                final_seg = text[last_end:].strip()
                if final_seg:
                    nfc_seg = unicodedata.normalize("NFC", final_seg)
                    if nfc_seg != final_seg:
                        nfc_changed_segments += 1
                    filtered = ETHIOPIC_FILTER_REGEX.sub(" ", nfc_seg)
                    sentence = re.sub(r"\s+", " ", filtered).strip()
                    words = WORD_RE.findall(sentence)
                    if len(words) >= 2:
                        # Append ። if not ending in punctuation
                        if not re.search(r"[።?!]$", sentence):
                            sentence += "።"
                        out_f.write(f"{sentence}\n")
                        valid_sentences += 1
                        total_tokens += len(words)
            break

        matches = list(SENTENCE_SPLIT_REGEX.finditer(text))
        if not matches:
            remainder = text
            continue

        last_end = 0
        for m in matches:
            delim_norm = normalize_delimiter(m.group())
            seg_raw = text[last_end : m.start()].strip()
            last_end = m.end()

            nfc_seg = unicodedata.normalize("NFC", seg_raw)
            if nfc_seg != seg_raw:
                nfc_changed_segments += 1

            filtered = ETHIOPIC_FILTER_REGEX.sub(" ", nfc_seg)
            sentence = re.sub(r"\s+", " ", filtered).strip()

            words = WORD_RE.findall(sentence)
            if len(words) >= 2:
                out_f.write(f"{sentence}{delim_norm}\n")
                valid_sentences += 1
                total_tokens += len(words)

                if valid_sentences % 500000 == 0:
                    print(f"  Written {valid_sentences:,} cleaned sentences... ({total_tokens:,} tokens)")

        remainder = text[last_end:]

    return {
        "valid_sentences": valid_sentences,
        "total_tokens": total_tokens,
        "nfc_changed_segments": nfc_changed_segments,
        "replacement_chars": replacement_chars,
    }


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    if stage_is_done(STAGE_NAME, [RAW_FILE], [CLEANED_FILE]):
        return

    if not RAW_FILE.exists():
        print(f"[ERROR] Raw corpus file not found: {RAW_FILE}")
        sys.exit(1)

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    tmp_cleaned_file = CLEANED_FILE.with_suffix(".txt.tmp")
    CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_FILE, "rb") as in_f, open(
        tmp_cleaned_file, "w", encoding="utf-8", newline="\n"
    ) as out_f:
        stats = clean_stream(in_f, out_f, chunk_bytes=2 * 1024 * 1024)

    tmp_cleaned_file.replace(CLEANED_FILE)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()
    valid_sentences = stats["valid_sentences"]
    total_tokens = stats["total_tokens"]
    avg_length = total_tokens / valid_sentences if valid_sentences > 0 else 0

    print("\n--- Cleaned Corpus Build Completed ---")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Valid Cleaned Sentences (Lines): {valid_sentences:,}")
    print(f"Total Word Tokens: {total_tokens:,}")
    print(f"Average Sentence Length: {avg_length:.2f} tokens")
    print(f"Segments Changed by NFC: {stats['nfc_changed_segments']:,}")
    print(f"Replacement Characters (U+FFFD): {stats['replacement_chars']:,}")

    record_stage(STAGE_NAME, [RAW_FILE], [CLEANED_FILE])


if __name__ == "__main__":
    main()
