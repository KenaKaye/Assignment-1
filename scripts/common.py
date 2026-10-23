"""Common utilities, paths, regexes, and checkpoint management for NLP Assignment 1.

Follows the specifications in Docs/NLP_Assignment1_Implementation_Plan.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
COUNTS_DIR = DATA_DIR / "counts"
FULL_COUNTS_DIR = COUNTS_DIR / "full"
TRAIN_COUNTS_DIR = COUNTS_DIR / "train_seed42"
CHECKSUMS_FILE = DATA_DIR / "checksums.json"

# Fixed Seeds
SEED = 42
SEED_VAL = 7

# Reserved Special Tokens
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

PAD_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"

RESERVED_VOCAB = {
    PAD_ID: PAD_TOKEN,
    EOS_ID: EOS_TOKEN,
    UNK_ID: UNK_TOKEN,
}

# Regex specifications (Plan Step 0, items 8 & 9)
# Splits on sentence punctuation ።, ?, !, or multi-colon ፡፡+
SENTENCE_SPLIT_REGEX = re.compile(r"[።?!]|፡፡+")

# Ethiopic word token extraction (Unicode U+1200 to U+135A)
WORD_RE = re.compile(r"[ሀ-ፚ]+")

# Character filter regex: replace non-Ethiopic letters, non-whitespace,
# non-allowed punctuation with space (Plan Step 0, item 9)
# Allowed punctuation: ። (U+1362), ፡ (U+1361), ፣ (U+1363), ?, !
ETHIOPIC_FILTER_REGEX = re.compile(r"[^\u1200-\u135A\s።፡፣?!]")


def require_colab(task_name: str = "Heavy pipeline stage") -> None:
    """Enforce the rule: heavy stages run on Google Colab only.

    May only be bypassed locally if environment variable ALLOW_HEAVY_LOCAL=1
    is explicitly set, as stipulated in AGENTS.md Rule B13.
    """
    if os.environ.get("ALLOW_HEAVY_LOCAL") == "1":
        print(f"[REQUIRE_COLAB] Running locally with ALLOW_HEAVY_LOCAL=1 override for: {task_name}")
        return

    try:
        import google.colab  # noqa: F401
        print(f"[REQUIRE_COLAB] Verified Colab runtime for: {task_name}")
    except ImportError:
        print(
            f"\n[ERROR] '{task_name}' is a heavy pipeline script designated to run on Google Colab only.\n"
            f"Under AGENTS.md Rule B13 and the implementation plan, passes over GPAC.txt, sorting,\n"
            f"vectorized n-gram counting, and perplexity evaluations must execute in Colab.\n"
            f"To deliberately run locally, set environment variable ALLOW_HEAVY_LOCAL=1.\n"
        )
        sys.exit(1)


def compute_sha256(filepath: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hash of a file efficiently in streaming chunks."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cannot compute hash; file not found: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_peak_memory_mb() -> float:
    """Return peak memory usage in megabytes."""
    try:
        if platform.system() == "Windows":
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        else:
            import resource
            # On Linux/Colab ru_maxrss is in kilobytes
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def load_checksums() -> dict[str, str]:
    """Load the recorded artifact SHA-256 checksums from data/checksums.json."""
    if not CHECKSUMS_FILE.exists():
        return {}
    try:
        with open(CHECKSUMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {CHECKSUMS_FILE}: {e}")
        return {}


def save_checksum(rel_path: str, sha256_hash: str) -> None:
    """Record or update a single file checksum in data/checksums.json."""
    checksums = load_checksums()
    checksums[rel_path] = sha256_hash
    CHECKSUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKSUMS_FILE, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2, sort_keys=True)


def is_checkpoint_done(stage_name: str, expected_files: list[Path]) -> bool:
    """Verify if all output files for a stage exist and match recorded checksums."""
    checksums = load_checksums()
    for file_path in expected_files:
        if not file_path.exists():
            return False
        rel_key = str(file_path.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel_key not in checksums:
            return False
        # Verify hash match
        current_hash = compute_sha256(file_path)
        if current_hash != checksums[rel_key]:
            print(f"[CHECKPOINT] Hash mismatch for {rel_key}: {current_hash} != {checksums[rel_key]}")
            return False
    print(f"[CHECKPOINT] Stage '{stage_name}' already complete with matching checksums. Skipping.")
    return True


def record_stage_checkpoint(stage_name: str, output_files: list[Path]) -> None:
    """Compute and record checksums for completed stage output files."""
    for file_path in output_files:
        rel_key = str(file_path.relative_to(REPO_ROOT)).replace("\\", "/")
        sha256_val = compute_sha256(file_path)
        save_checksum(rel_key, sha256_val)
        print(f"[CHECKPOINT] Recorded {rel_key} -> {sha256_val}")
