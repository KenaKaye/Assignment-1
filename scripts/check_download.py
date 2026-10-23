"""scripts/check_download.py: Light PC-side Artifact Integrity Checker.

Verifies that all downloaded artifacts from Colab match their recorded SHA-256 hashes
in data/checksums.json.

Under AGENTS.md Rule B13:
- "Before any notebook or app runs on the PC, scripts/check_download.py must report
   every downloaded checksum as matching."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    CHECKSUMS_FILE,
    REPO_ROOT,
    compute_sha256,
)


def main() -> None:
    print("--- Running Downloaded Artifacts Checksum Verification ---")

    if not CHECKSUMS_FILE.exists():
        print(f"[ERROR] Checksums file not found: {CHECKSUMS_FILE}")
        print("Please copy data/checksums.json from the Google Colab build before running locally.")
        sys.exit(1)

    with open(CHECKSUMS_FILE, "r", encoding="utf-8") as f:
        checksums: dict[str, str] = json.load(f)

    if not checksums:
        print("[WARN] data/checksums.json is empty!")
        sys.exit(1)

    print(f"Loaded {len(checksums)} artifact checksums to verify.\n")

    passed_count = 0
    mismatches = []
    missing_files = []

    for rel_path, expected_hash in sorted(checksums.items()):
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            missing_files.append(rel_path)
            print(f"  [MISSING] {rel_path}")
            continue

        actual_hash = compute_sha256(full_path)
        if actual_hash == expected_hash:
            passed_count += 1
            print(f"  [OK] {rel_path} ({actual_hash[:10]}...)")
        else:
            mismatches.append((rel_path, expected_hash, actual_hash))
            print(f"  [MISMATCH] {rel_path}!\n      Expected: {expected_hash}\n      Actual:   {actual_hash}")

    print("\n---------------- Checksum Report ----------------")
    print(f"Total checked: {len(checksums)}")
    print(f"Matching:      {passed_count}")
    print(f"Missing:       {len(missing_files)}")
    print(f"Mismatches:    {len(mismatches)}")

    if missing_files or mismatches:
        print("\n[FAIL] Artifact verification failed. Do NOT run notebooks or application until resolved.")
        sys.exit(1)

    print("\n[SUCCESS] All artifact checksums match perfectly! Ready for local PC notebooks and app.")
    sys.exit(0)


if __name__ == "__main__":
    main()
