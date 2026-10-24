"""scripts/12_q3_extrinsic.py: Q3 Extrinsic Evaluation.

Task: Real-versus-Corrupted Amharic Sentence Detection
- Positive: Real sentence from held-out split
- Negative: Sentence corrupted by replacing 1-2 words with vocabulary words
- Zero-leakage: tuning sentences from separate 'tune' partition (0 overlap with training counts)
- Full held-out tuning and test splits (no sampling)
- Vectorized scoring via kn_lib
- UNK confound baseline and sensitivity analysis on UNK-free pairs
- Tunes threshold tau on tuning set by maximum F1
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
import sklearn
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
    PAD_ID,
    RESULTS_DIR,
    SEED,
    UNK_ID,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)
from scripts.kn_lib import KneserNeyModel

STAGE_NAME = "12_q3_extrinsic"
SPLIT_FILE = DATA_DIR / "split_indices_seed42_v2.json"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
TRAIN_COUNTS_DIR = DATA_DIR / "counts" / "train_seed42"
LUT_FILE = TRAIN_COUNTS_DIR / "lut_train_seed42.npy"
METRICS_FILE = RESULTS_DIR / "q3_metrics.json"


def prepare_pairs_vectorized(
    sentence_indices: np.ndarray,
    tokens: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    lut: np.ndarray,
    train_vocab_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build positive and negative pairs, returning flattened token arrays."""
    rng = random.Random(seed)
    
    # Filter valid sentences (at least 2 words excluding EOS)
    s_starts = starts[sentence_indices]
    s_lengths = (ends[sentence_indices] - s_starts).astype(np.int64)
    # Note: s_lengths includes EOS
    valid_mask = s_lengths >= 3
    s_indices = sentence_indices[valid_mask]
    s_starts = s_starts[valid_mask]
    s_lengths = s_lengths[valid_mask]

    n_sents = len(s_indices)
    n_pairs = n_sents * 2
    print(f"  Valid sentences for evaluation: {n_sents:,} ({n_pairs:,} total pairs)")

    # Each sentence has length L (including EOS). Real has L, corrupted has L.
    # Total tokens = 2 * sum(s_lengths)
    total_tokens = int(s_lengths.sum() * 2)

    all_tokens = np.empty(total_tokens, dtype=np.int32)
    all_pos = np.empty(total_tokens, dtype=np.int64)
    pair_ids = np.empty(total_tokens, dtype=np.int32)
    pair_lengths = np.empty(n_pairs, dtype=np.float64)
    all_labels = np.empty(n_pairs, dtype=np.int32)

    curr_tok = 0
    curr_pair = 0

    for s_start, s_len in zip(s_starts, s_lengths):
        # Full original global IDs including EOS
        orig_seq = tokens[s_start : s_start + s_len]
        words_only = orig_seq[:-1]
        w_len = len(words_only)

        # 1. Positive (Real)
        real_seq = lut[orig_seq]
        all_tokens[curr_tok : curr_tok + s_len] = real_seq
        all_pos[curr_tok : curr_tok + s_len] = np.arange(s_len, dtype=np.int64)
        pair_ids[curr_tok : curr_tok + s_len] = curr_pair
        pair_lengths[curr_pair] = s_len
        all_labels[curr_pair] = 1
        curr_tok += s_len
        curr_pair += 1

        # 2. Negative (Corrupted on full IDs before LUT)
        corr_words = np.array(words_only, copy=True)
        n_replace = 1 if w_len <= 3 else min(2, w_len - 1)
        rep_indices = rng.sample(range(w_len), n_replace)
        for idx in rep_indices:
            orig_id = corr_words[idx]
            rep_id = int(rng.choice(train_vocab_ids))
            while rep_id == orig_id and len(train_vocab_ids) > 1:
                rep_id = int(rng.choice(train_vocab_ids))
            corr_words[idx] = rep_id

        corr_full = np.append(corr_words, EOS_ID)
        corr_seq = lut[corr_full]

        all_tokens[curr_tok : curr_tok + s_len] = corr_seq
        all_pos[curr_tok : curr_tok + s_len] = np.arange(s_len, dtype=np.int64)
        pair_ids[curr_tok : curr_tok + s_len] = curr_pair
        pair_lengths[curr_pair] = s_len
        all_labels[curr_pair] = 0
        curr_tok += s_len
        curr_pair += 1

    return all_tokens, all_pos, pair_ids, pair_lengths, all_labels, n_sents


def tune_tau(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find threshold tau that maximizes F1 score on tuning set (tie: first maximum)."""
    candidates = np.unique(np.percentile(scores, np.linspace(0, 100, 200)))
    best_tau = float(candidates[0])
    best_f1 = -1.0

    for tau in candidates:
        preds = (scores >= tau).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau, best_f1


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_inputs = [
        SPLIT_FILE,
        TOKENS_FILE,
        STARTS_FILE,
        LUT_FILE,
    ]
    for n in range(1, 5):
        expected_inputs.append(TRAIN_COUNTS_DIR / f"order{n}_ngrams.npy")
        expected_inputs.append(TRAIN_COUNTS_DIR / f"order{n}_counts.npy")

    expected_outputs = [METRICS_FILE]
    for n in range(1, 5):
        expected_outputs.append(RESULTS_DIR / f"q3_roc_n{n}.csv")

    if stage_is_done(STAGE_NAME, expected_inputs, expected_outputs):
        print(f"[{STAGE_NAME}] Checkpoint valid. Skipping.")
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    print(f"Loading {SPLIT_FILE.name}...")
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    tune_indices = np.array(split_data["tune"], dtype=np.int64)
    test_indices = np.array(split_data["test"], dtype=np.int64)
    train_core_set = set(split_data["train_core"])

    overlap = len(set(tune_indices) & train_core_set)
    assert overlap == 0, f"Data leakage detected! {overlap} tuning sentences found in train_core counts."
    print(f"  Zero-leakage check passed: Overlap between tune and train_core counts = {overlap}")

    print(f"Loading {TOKENS_FILE.name} and {STARTS_FILE.name}...")
    tokens = np.load(TOKENS_FILE, mmap_mode="r")
    starts = np.load(STARTS_FILE)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(tokens)

    print(f"Loading LUT: {LUT_FILE.name}...")
    lut = np.load(LUT_FILE)

    # Replacement vocabulary: IDs with train count >= 2 (non-singletons preserved in LUT)
    non_singletons = np.flatnonzero(lut == np.arange(len(lut)))
    train_vocab_ids = np.array([tid for tid in non_singletons if tid >= 3], dtype=np.int32)
    print(f"  Available vocabulary replacement tokens: {len(train_vocab_ids):,}")

    print("\nPreparing tuning pairs...")
    tune_toks, tune_pos, tune_pids, tune_lens, tune_labels, n_tune_sents = prepare_pairs_vectorized(
        tune_indices, tokens, starts, ends, lut, train_vocab_ids, seed=SEED
    )

    print("\nPreparing test pairs...")
    test_toks, test_pos, test_pids, test_lens, test_labels, n_test_sents = prepare_pairs_vectorized(
        test_indices, tokens, starts, ends, lut, train_vocab_ids, seed=SEED + 1
    )

    # UNK Confound Baseline (item 12-4)
    print("\nComputing UNK Confound Baseline on test set...")
    test_unk_counts = np.bincount(test_pids, weights=(test_toks == UNK_ID), minlength=len(test_labels))
    test_unk_scores = test_unk_counts / test_lens
    
    # UNK baseline: higher UNK count -> predict real (1)
    unk_auc = float(roc_auc_score(test_labels, test_unk_scores))
    # Threshold at > 0
    unk_preds = (test_unk_counts > 0).astype(int)
    unk_acc = float(accuracy_score(test_labels, unk_preds))
    print(f"  [UNK Baseline] ROC-AUC: {unk_auc:.4f} | Accuracy: {unk_acc:.4f}")

    # Sensitivity mask: pairs where neither real nor corrupted has UNK
    # Real is even (2i), corr is odd (2i+1)
    unk_free_sents = (test_unk_counts[0::2] == 0) & (test_unk_counts[1::2] == 0)
    unk_free_pair_mask = np.repeat(unk_free_sents, 2)
    n_unk_free_pairs = int(np.count_nonzero(unk_free_pair_mask))
    print(f"  Sensitivity subset (UNK-free): {n_unk_free_pairs:,} / {len(test_labels):,} pairs ({n_unk_free_pairs / len(test_labels) * 100:.2f}%)")

    print("\nLoading KneserNeyModel from train_seed42...")
    kn_model = KneserNeyModel.load_from_dir(TRAIN_COUNTS_DIR, max_order=4)

    models_output = {}

    for n in range(1, 5):
        t0 = time.time()
        print(f"\nEvaluating Order n = {n}...")

        # 1. Score tuning set and tune tau
        tune_logp, _ = kn_model.score_sequence(tune_toks, tune_pos, order=n)
        tune_scores = np.bincount(tune_pids, weights=tune_logp, minlength=len(tune_labels)) / tune_lens
        best_tau, tune_f1 = tune_tau(tune_scores, tune_labels)
        print(f"  Tuned tau on tuning set: {best_tau:.4f} (Tuning F1: {tune_f1:.4f})")

        # 2. Score test set
        test_logp, _ = kn_model.score_sequence(test_toks, test_pos, order=n)
        test_scores = np.bincount(test_pids, weights=test_logp, minlength=len(test_labels)) / test_lens
        test_preds = (test_scores >= best_tau).astype(int)

        # Full test metrics
        acc = float(accuracy_score(test_labels, test_preds))
        prec = float(precision_score(test_labels, test_preds, zero_division=0))
        rec = float(recall_score(test_labels, test_preds, zero_division=0))
        f1 = float(f1_score(test_labels, test_preds, zero_division=0))
        auc = float(roc_auc_score(test_labels, test_scores))
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

        print(f"  [Test Full n={n}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"  Confusion Matrix: TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")

        # Sensitivity metrics (UNK-free)
        sens_labels = test_labels[unk_free_pair_mask]
        sens_scores = test_scores[unk_free_pair_mask]
        sens_preds = (sens_scores >= best_tau).astype(int)

        sens_acc = float(accuracy_score(sens_labels, sens_preds))
        sens_prec = float(precision_score(sens_labels, sens_preds, zero_division=0))
        sens_rec = float(recall_score(sens_labels, sens_preds, zero_division=0))
        sens_f1 = float(f1_score(sens_labels, sens_preds, zero_division=0))
        sens_auc = float(roc_auc_score(sens_labels, sens_scores))
        stn, sfp, sfn, stp = confusion_matrix(sens_labels, sens_preds).ravel()

        print(f"  [Sensitivity UNK-free n={n}] Acc: {sens_acc:.4f} | Prec: {sens_prec:.4f} | Rec: {sens_rec:.4f} | F1: {sens_f1:.4f} | AUC: {sens_auc:.4f}")

        # Save ROC points
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_file = RESULTS_DIR / f"q3_roc_n{n}.csv"
        with open(roc_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr"])
            for fp_val, tp_val in zip(fpr, tpr):
                writer.writerow([f"{fp_val:.6f}", f"{tp_val:.6f}"])

        meets_usable = auc >= 0.80 and acc >= 0.70
        models_output[f"n={n}"] = {
            "order": n,
            "tau_threshold": round(best_tau, 4),
            "tuning_f1": round(tune_f1, 4),
            "tie_breaking_rule": "first_maximum",
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
            "sensitivity_unk_free": {
                "pair_count": n_unk_free_pairs,
                "accuracy": round(sens_acc, 4),
                "precision": round(sens_prec, 4),
                "recall": round(sens_rec, 4),
                "f1_score": round(sens_f1, 4),
                "roc_auc": round(sens_auc, 4),
                "confusion_matrix": {
                    "tp": int(stp),
                    "fp": int(sfp),
                    "tn": int(stn),
                    "fn": int(sfn),
                },
            },
            "proposed_target": "AUC >= 0.80 and accuracy >= 0.70 (proposed target set by the group, not a published standard)",
            "usable": bool(meets_usable),
            "scoring_time_sec": round(time.time() - t0, 2),
        }

    output = {
        "task": "real-vs-corrupted Amharic sentence detection",
        "data_leakage_overlap_count": overlap,
        "tuning_pairs_count": len(tune_labels),
        "test_pairs_count": len(test_labels),
        "unk_confound_baseline": {
            "description": "Predict real based on UNK token count",
            "accuracy": round(unk_acc, 4),
            "roc_auc": round(unk_auc, 4),
        },
        "environment": {
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "models": models_output,
    }

    tmp_metrics = METRICS_FILE.with_suffix(".json.tmp")
    with open(tmp_metrics, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    tmp_metrics.replace(METRICS_FILE)

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed} s. Wrote {METRICS_FILE}")
    record_stage(STAGE_NAME, expected_inputs, expected_outputs)


if __name__ == "__main__":
    main()
