"""Artifact registry and exporter.

Generates data/artifact_list.json listing all artifacts marked 'Yes'
in the implementation plan artifact layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import DATA_DIR, REPO_ROOT

# Relative paths of canonical artifacts marked "Yes" for commit/export
CANONICAL_ARTIFACTS = [
    "data/sentence_assessment.tsv",
    "data/vocab.tsv",
    "data/split_indices_seed42.json",
    "data/split_indices_seed7.json",
    "data/split_indices_seed42_v2.json",
    "data/split_indices_seed7_v2.json",
    "data/counts/train_seed42/vocab_train.tsv",
    "data/counts/train_seed7/vocab_train.tsv",
    "data/manifest.json",
    "data/checksums.json",
    "data/profile_counts.json",
    "data/artifact_list.json",
    "results/run_log.jsonl",
    "results/kaggle_stage_times.json",
    "results/q1_top10_n1.csv",
    "results/q1_top10_n2.csv",
    "results/q1_top10_n3.csv",
    "results/q1_top10_n4.csv",
    "results/q1_bigram_conditional.csv",
    "results/q1_stopword_top10_n1.csv",
    "results/q1_stopword_top10_n2.csv",
    "results/q1_stopword_top10_n3.csv",
    "results/q1_stopword_top10_n4.csv",
    "results/q1_sentence_probs.csv",
    "results/q1_generation.json",
    "results/q2_perplexity.json",
    "results/q3_metrics.json",
    "results/q3_roc_n1.csv",
    "results/q3_roc_n2.csv",
    "results/q3_roc_n3.csv",
    "results/q3_roc_n4.csv",
    "results/q4_eval.json",
]


def write_artifact_list() -> list[str]:
    """Write canonical artifact list to data/artifact_list.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "artifact_list.json"

    # Ensure results/run_log.jsonl exists
    run_log = REPO_ROOT / "results" / "run_log.jsonl"
    if not run_log.exists():
        run_log.parent.mkdir(parents=True, exist_ok=True)
        run_log.touch()

    # Alias stage_times across Colab and Kaggle environments
    colab_times = REPO_ROOT / "results" / "colab_stage_times.json"
    kaggle_times = REPO_ROOT / "results" / "kaggle_stage_times.json"
    if colab_times.exists() and not kaggle_times.exists():
        import shutil
        shutil.copy2(colab_times, kaggle_times)
    elif kaggle_times.exists() and not colab_times.exists():
        import shutil
        shutil.copy2(kaggle_times, colab_times)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(CANONICAL_ARTIFACTS, f, indent=2)
    return CANONICAL_ARTIFACTS


def get_existing_artifacts() -> list[str]:
    """Return relative paths of canonical artifacts that currently exist on disk."""
    existing = []
    for rel_path in CANONICAL_ARTIFACTS:
        p = REPO_ROOT / rel_path
        if p.exists():
            existing.append(rel_path)
    return existing


if __name__ == "__main__":
    arts = write_artifact_list()
    print(f"Recorded {len(arts)} canonical artifacts to data/artifact_list.json")
