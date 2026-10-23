"""scripts/12_q3_extrinsic.py: Q3 Extrinsic Evaluation.

Task: Real-versus-Corrupted Amharic Sentence Detection
- Positive: Real sentence from held-out split
- Negative: Sentence corrupted by replacing 1-2 words with random vocabulary words
- Tunes threshold tau on held-out training tuning split
- Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix on Test Split
- Writes results/q3_metrics.json and results/q3_roc_n{1..4}.csv
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    EOS_ID,
    RESULTS_DIR,
    SEED,
    TRAIN_COUNTS_DIR,
    UNK_ID,
    WORD_RE,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)
import importlib

scripts_09 = importlib.import_module("scripts.09_kn")
KneserNeyModel = scripts_09.KneserNeyModel

STAGE_NAME = "12_q3_extrinsic"
SPLIT_FILE = DATA_DIR / "split_indices_seed42.json"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
TRAIN_VOCAB_FILE = TRAIN_COUNTS_DIR / "vocab_train.tsv"
METRICS_FILE = RESULTS_DIR / "q3_metrics.json"


def load_train_vocab() -> tuple[dict[str, int], list[str]]:
    w2id = {}
    words = []
    with open(TRAIN_VOCAB_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word = parts[1]
                w2id[word] = int(parts[0])
                if int(parts[0]) >= 3:
                    words.append(word)
    return w2id, words


def corrupt_sentence(words: list[str], vocab_words: list[str], rng: random.Random) -> list[str]:
    """Replace 1 or 2 random words in sentence with random vocabulary words."""
    corrupted = list(words)
    n_replace = 1 if len(words) <= 3 else min(2, len(words) - 1)
    indices = rng.sample(range(len(words)), n_replace)
    for idx in indices:
        orig = words[idx]
        replacement = rng.choice(vocab_words)
        while replacement == orig:
            replacement = rng.choice(vocab_words)
        corrupted[idx] = replacement
    return corrupted


def build_labeled_pairs(
    sentences: list[list[str]], vocab_words: list[str], seed: int
) -> list[dict]:
    """Build balanced 1:1 real (1) and corrupted (0) labeled examples."""
    rng = random.Random(seed)
    pairs = []
    for words in sentences:
        # Positive
        pairs.append({"words": words, "label": 1})
        # Negative
        corr_words = corrupt_sentence(words, vocab_words, rng)
        pairs.append({"words": corr_words, "label": 0})
    return pairs


def score_pairs(
    pairs: list[dict], kn_model: KneserNeyModel, w2id: dict[str, int], order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Score each sentence with per-token log probability."""
    scores = np.empty(len(pairs), dtype=np.float64)
    labels = np.empty(len(pairs), dtype=np.int32)

    for i, item in enumerate(pairs):
        labels[i] = item["label"]
        words = item["words"]
        tids = [w2id.get(w, UNK_ID) for w in words] + [EOS_ID]
        logp = kn_model.sentence_logprob(tids, order=order)
        scores[i] = logp / len(tids)

    return scores, labels


def tune_tau(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find threshold tau that maximizes F1 score on the tuning set."""
    best_tau = float(scores.min())
    best_f1 = -1.0
    percentiles = np.linspace(5, 95, 100)
    candidates = np.percentile(scores, percentiles)

    for tau in candidates:
        preds = (scores >= tau).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau, best_f1


def main() -> None:
    require_colab(STAGE_NAME)

    expected_files = [METRICS_FILE]
    for n in range(1, 5):
        expected_files.append(RESULTS_DIR / f"q3_roc_n{n}.csv")

    if is_checkpoint_done(STAGE_NAME, expected_files):
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    w2id, vocab_words = load_train_vocab()
    kn_model = KneserNeyModel.load_from_dir(TRAIN_COUNTS_DIR, max_order=4)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # For efficient extrinsic evaluation, evaluate on 5,000 tuning and 10,000 test pairs
    rng = random.Random(SEED)
    train_pool = split_data["train"]
    test_pool = split_data["test"]

    tuning_indices = set(rng.sample(train_pool, min(5000, len(train_pool))))
    test_eval_indices = set(rng.sample(test_pool, min(10000, len(test_pool))))

    print("Extracting sentences for tuning and test sets...")
    tuning_sents = []
    test_sents = []

    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in tuning_indices:
                words = WORD_RE.findall(line)
                if len(words) >= 2:
                    tuning_sents.append(words)
            elif idx in test_eval_indices:
                words = WORD_RE.findall(line)
                if len(words) >= 2:
                    test_sents.append(words)

    print(f"Tuning Pool: {len(tuning_sents):,} sentences | Test Pool: {len(test_sents):,} sentences")

    print("Constructing balanced labeled pairs...")
    tuning_pairs = build_labeled_pairs(tuning_sents, vocab_words, seed=SEED)
    test_pairs = build_labeled_pairs(test_sents, vocab_words, seed=SEED + 1)
    print(f"Tuning Examples: {len(tuning_pairs):,} | Test Examples: {len(test_pairs):,}")

    metrics_output = {
        "task": "real-vs-corrupted sentence detection",
        "tuning_pairs_count": len(tuning_pairs),
        "test_pairs_count": len(test_pairs),
        "models": {},
    }

    for n in range(1, 5):
        print(f"\nEvaluating Extrinsic Order n = {n}...")
        # Score tuning set and find tau
        tune_scores, tune_labels = score_pairs(tuning_pairs, kn_model, w2id, order=n)
        best_tau, tune_f1 = tune_tau(tune_scores, tune_labels)
        print(f"  Tuned tau on training tuning set: {best_tau:.4f} (Tuning F1: {tune_f1:.4f})")

        # Score test set
        test_scores, test_labels = score_pairs(test_pairs, kn_model, w2id, order=n)
        test_preds = (test_scores >= best_tau).astype(int)

        acc = float(accuracy_score(test_labels, test_preds))
        prec = float(precision_score(test_labels, test_preds, zero_division=0))
        rec = float(recall_score(test_labels, test_preds, zero_division=0))
        f1 = float(f1_score(test_labels, test_preds, zero_division=0))
        auc = float(roc_auc_score(test_labels, test_scores))
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

        print(
            f"  [Test Results n={n}] Acc: {acc:.4f} | Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
        )
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        # Save ROC points
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_file = RESULTS_DIR / f"q3_roc_n{n}.csv"
        with open(roc_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr"])
            for fp_val, tp_val in zip(fpr, tpr):
                writer.writerow([f"{fp_val:.6f}", f"{tp_val:.6f}"])

        metrics_output["models"][f"n={n}"] = {
            "order": n,
            "tau_threshold": round(best_tau, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
            "usable": auc >= 0.80 and acc >= 0.70,
        }

    tmp_metrics = METRICS_FILE.with_suffix(".json.tmp")
    with open(tmp_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    tmp_metrics.replace(METRICS_FILE)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s. Wrote {METRICS_FILE}")
    record_stage_checkpoint(STAGE_NAME, expected_files)


if __name__ == "__main__":
    main()
