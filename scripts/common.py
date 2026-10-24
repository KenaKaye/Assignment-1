"""Common utilities, paths, regexes, manifest and checkpoint management.

Follows the specifications in Section 3 and 4 of the fix specification.
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
MANIFEST_FILE = DATA_DIR / "manifest.json"
RUN_LOG_FILE = RESULTS_DIR / "run_log.jsonl"

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

# 64-bit integer packing base: 2^21 = 2,097,152
V_LIMIT = 1 << 21

# Sentence boundary regex (D5): keeps sentence-final punctuation
SENTENCE_END_RE = re.compile(r"[።?!]|፡፡+")
SENTENCE_SPLIT_REGEX = SENTENCE_END_RE

# Ethiopic word token extraction (Unicode U+1200 to U+135A)
WORD_RE = re.compile(r"[\u1200-\u135A]+")

# Character filter regex: keep Ethiopic letters, whitespace, and allowed punctuation
# Allowed punctuation: ። (U+1362), ፡ (U+1361), ፣ (U+1363), ?, !
ETHIOPIC_FILTER_REGEX = re.compile(r"[^\u1200-\u135A\s።፡፣?!]")


def to_rel_path(path: Path | str) -> str:
    """Normalize path to relative POSIX path from REPO_ROOT if inside workspace."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def require_heavy_runtime(task_name: str = "Heavy pipeline stage") -> None:
    """Enforce the rule: heavy stages run on Kaggle or Google Colab only.

    May only be bypassed locally if ALLOW_HEAVY_LOCAL=1 is explicitly set.
    """
    if os.environ.get("ALLOW_HEAVY_LOCAL") == "1":
        print(f"[REQUIRE_HEAVY] Running with ALLOW_HEAVY_LOCAL=1 override for: {task_name}")
        append_run_log({
            "timestamp": time.time(),
            "stage": task_name,
            "override": True,
            "message": "ALLOW_HEAVY_LOCAL=1 override used",
        })
        return

    try:
        import google.colab  # noqa: F401
        print(f"[REQUIRE_HEAVY] Verified Colab runtime for: {task_name}")
        return
    except ImportError:
        pass

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print(f"[REQUIRE_HEAVY] Verified Kaggle environment (KAGGLE_KERNEL_RUN_TYPE) for: {task_name}")
        return

    if os.path.isdir("/kaggle/working"):
        print(f"[REQUIRE_HEAVY] Verified Kaggle environment (/kaggle/working) for: {task_name}")
        return

    print(
        f"\n[ERROR] '{task_name}' is a heavy pipeline script designated to run on Kaggle or Colab only.\n"
        f"Under AGENTS.md Rule B13 and the implementation plan, passes over GPAC.txt, sorting,\n"
        f"vectorized n-gram counting, and perplexity evaluations must execute on Kaggle or Colab.\n"
        f"To deliberately run locally, set environment variable ALLOW_HEAVY_LOCAL=1.\n"
    )
    sys.exit(1)


# Maintain alias for compatibility
require_colab = require_heavy_runtime


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


def peak_rss_mb() -> float:
    """Return peak resident set size in megabytes using VmHWM from /proc/self/status.

    Raises RuntimeError on failure.
    """
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        raise RuntimeError("Cannot measure peak_rss_mb: /proc/self/status does not exist.")

    with open(status_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = float(parts[1])
                    return kb / 1024.0
    raise RuntimeError("VmHWM metric not found in /proc/self/status")


def get_peak_memory_mb() -> float:
    """Legacy helper returning peak RSS in MB with graceful fallback."""
    try:
        return peak_rss_mb()
    except Exception:
        if platform.system() == "Windows":
            try:
                import psutil
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0


def memory_budget_mb() -> float:
    """Detect available memory and compute operational budget in megabytes.

    Order of detection:
    1. Check NGRAM_MEM_BUDGET_MB environment override.
    2. Read cgroup memory limit (/sys/fs/cgroup/memory.max or memory.limit_in_bytes).
    3. Read physical total memory.
    4. total_mb = min(cgroup_limit, physical_total) / 2**20.
    5. budget_mb = total_mb * 0.90 - 1024 (10% safety margin plus 1 GiB for Python interpreter).
    """
    env_budget = os.environ.get("NGRAM_MEM_BUDGET_MB")
    if env_budget:
        val = float(env_budget)
        print(f"[MEMORY_BUDGET] Using explicit NGRAM_MEM_BUDGET_MB override: {val:.1f} MB")
        return val

    cgroup_bytes = float("inf")
    for cg_path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = Path(cg_path)
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8").strip()
                if content and content != "max":
                    val = int(content)
                    # Ignore values near 2^63 (no limit marker)
                    if val < (1 << 62):
                        cgroup_bytes = float(val)
                        break
            except Exception:
                pass

    phys_bytes = float("inf")
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_bytes = float(pages * page_size)
    except Exception:
        try:
            import psutil
            phys_bytes = float(psutil.virtual_memory().total)
        except Exception:
            pass

    min_bytes = min(cgroup_bytes, phys_bytes)
    if min_bytes == float("inf"):
        # Fallback default if completely undetectable: 8192 MB
        total_mb = 8192.0
    else:
        total_mb = min_bytes / (1024 * 1024)

    budget_mb = max(1024.0, total_mb * 0.90 - 1024.0)
    return budget_mb


def load_checksums() -> dict[str, str]:
    """Load the recorded artifact SHA-256 checksums from data/checksums.json.

    Raises RuntimeError if checksums.json is corrupt.
    """
    if not CHECKSUMS_FILE.exists():
        return {}
    try:
        with open(CHECKSUMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Corrupt {CHECKSUMS_FILE}: {e}") from e


def save_checksum(rel_path: str, sha256_hash: str) -> None:
    """Record or update a single file checksum in data/checksums.json atomically."""
    checksums = load_checksums()
    norm_key = rel_path.replace("\\", "/")
    checksums[norm_key] = sha256_hash
    CHECKSUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = CHECKSUMS_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2, sort_keys=True)
    tmp_file.replace(CHECKSUMS_FILE)


def load_manifest() -> dict[str, dict]:
    """Load stage execution manifest from data/manifest.json.

    Raises RuntimeError if manifest.json is corrupt.
    """
    if not MANIFEST_FILE.exists():
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Corrupt {MANIFEST_FILE}: {e}") from e


def stage_is_done(
    stage_name: str,
    inputs: list[Path | str],
    outputs: list[Path | str],
) -> bool:
    """Return True only if all recorded input and output hashes match current state."""
    manifest = load_manifest()
    if stage_name not in manifest:
        return False

    entry = manifest[stage_name]
    saved_inputs = entry.get("inputs", {})
    saved_outputs = entry.get("outputs", {})

    for inp in inputs:
        inp_path = Path(inp)
        if not inp_path.exists():
            return False
        rel_key = to_rel_path(inp_path)
        if rel_key not in saved_inputs:
            return False
        if compute_sha256(inp_path) != saved_inputs[rel_key]:
            return False

    for out in outputs:
        out_path = Path(out)
        if not out_path.exists():
            return False
        rel_key = to_rel_path(out_path)
        if rel_key not in saved_outputs:
            return False
        if compute_sha256(out_path) != saved_outputs[rel_key]:
            return False

    print(f"[MANIFEST] Stage '{stage_name}' inputs and outputs verified. Skipping.")
    return True


def record_stage(
    stage_name: str,
    inputs: list[Path | str],
    outputs: list[Path | str],
) -> None:
    """Atomically record input and output SHA-256 hashes in data/manifest.json."""
    manifest = load_manifest()
    inp_hashes = {to_rel_path(p): compute_sha256(p) for p in inputs if Path(p).exists()}
    out_hashes = {to_rel_path(p): compute_sha256(p) for p in outputs if Path(p).exists()}

    manifest[stage_name] = {
        "inputs": inp_hashes,
        "outputs": out_hashes,
        "timestamp": time.time(),
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = MANIFEST_FILE.with_suffix(".json.tmp")
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp_manifest.replace(MANIFEST_FILE)

    # Also record outputs in checksums.json
    for rel_k, h in out_hashes.items():
        save_checksum(rel_k, h)


def append_run_log(record: dict) -> None:
    """Append a JSON record line to results/run_log.jsonl."""
    RUN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
