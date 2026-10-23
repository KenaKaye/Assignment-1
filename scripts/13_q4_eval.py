"""scripts/13_q4_eval.py: Q4 Predictive Text Evaluation.

Evaluates next-word prediction on held-out test sentences:
- Input: Sentence without its last word
- Ground truth target: The true final word
- Metrics: Top-1 Accuracy, Top-5 Hit Rate, Mean Reciprocal Rank (MRR)
- Evaluated for one-word context (bigram backoff) and two-word context (trigram backoff)
- Produces: results/q4_eval.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to import common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (
    DATA_DIR,
    EOS_ID,
    PAD_ID,
    RESULTS_DIR,
    SEED,
    TRAIN_COUNTS_DIR,
    UNK_ID,
    WORD_RE,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)
from scripts.09_kn import KneserNeyModel

STAGE_NAME = "13_q4_eval"
SPLIT_FILE = DATA_DIR / "split_indices_seed42.json"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
TRAIN_VOCAB_FILE = TRAIN_COUNTS_DIR / "vocab_train.tsv"
RESULTS_FILE = RESULTS_DIR / "q4_eval.json"


def load_vocab() -> tuple[dict[str, int], dict[int, str]]:
    w2id = {}
    id2w = {}
    with open(TRAIN_VOCAB_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tid = int(parts[0])
                w = parts[1]
                w2id[w] = tid
                id2w[tid] = w
    return w2id, id2w


def predict_next_word(
    context_tokens: list[int],
    kn_model: KneserNeyModel,
    max_context_len: int = 2,
    k: int = 5,
) -> list[int]:
    """Return top-k candidate token IDs for a given context."""
    eff_len = min(len(context_tokens), max_context_len)
    context = context_tokens[-eff_len:] if eff_len > 0 else []
    order = len(context) + 1

    candidates = set()

    # 1. Trigram continuations
    if order >= 3:
        h = np.array(context[-2:], dtype=np.int32)
        l, r = kn_model._find_ngram_range(3, h)
        if r > l:
            candidates.update(kn_model.ngrams[3][l:r, 2].tolist())

    # 2. Bigram continuations
    if order >= 2:
        h1 = np.array([context[-1]], dtype=np.int32)
        l, r = kn_model._find_ngram_range(2, h1)
        if r > l:
            candidates.update(kn_model.ngrams[2][l:r, 1].tolist())

    # 3. Unigram top 50 fallback
    top_unigrams = kn_model.ngrams[1][:50, 0].tolist()
    candidates.update(top_unigrams)

    # Exclude special tokens
    candidates.discard(PAD_ID)
    candidates.discard(EOS_ID)
    candidates.discard(UNK_ID)

    # Score candidates using Kneser-Ney
    scored = []
    for cand_id in candidates:
        p = kn_model.prob(cand_id, context, order=order)
        scored.append((cand_id, p))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in scored[:k]]


def main() -> None:
    require_colab(STAGE_NAME)

    if is_checkpoint_done(STAGE_NAME, [RESULTS_FILE]):
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    w2id, id2w = load_vocab()
    kn_model = KneserNeyModel.load_from_dir(TRAIN_COUNTS_DIR, max_order=3)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Test set draw: 5,000 sentences
    rng = random.Random(SEED)
    test_indices = set(rng.sample(split_data["test"], min(5000, len(split_data["test"]))))

    test_sentences = []
    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in test_indices:
                words = WORD_RE.findall(line)
                if len(words) >= 3:  # Need at least 2 context words + 1 target
                    test_sentences.append(words)

    print(f"Evaluating {len(test_sentences):,} test sentences for predictive accuracy...")

    modes = ["one_word_context", "two_word_context"]
    results = {}

    for mode in modes:
        max_ctx = 1 if mode == "one_word_context" else 2
        print(f"\nEvaluating mode: {mode} (context={max_ctx})...")

        top1_hits = 0
        top5_hits = 0
        reciprocal_ranks = []
        eval_count = 0

        t0 = time.time()
        for words in test_sentences:
            context_words = words[:-1]
            target_word = words[-1]
            target_id = w2id.get(target_word, UNK_ID)

            if target_id == UNK_ID:
                continue

            ctx_ids = [w2id.get(w, UNK_ID) for w in context_words]
            suggestions = predict_next_word(ctx_ids, kn_model, max_context_len=max_ctx, k=5)

            eval_count += 1
            if target_id == suggestions[0]:
                top1_hits += 1

            if target_id in suggestions:
                top5_hits += 1
                rank = suggestions.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        elapsed_mode = time.time() - t0
        top1_acc = top1_hits / eval_count if eval_count > 0 else 0.0
        top5_acc = top5_hits / eval_count if eval_count > 0 else 0.0
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        print(f"  Evaluated: {eval_count:,} target positions")
        print(f"  Top-1 Accuracy: {top1_acc:.4f} (Benchmark: >= 0.10)")
        print(f"  Top-5 Hit Rate: {top5_acc:.4f} (Benchmark: >= 0.30)")
        print(f"  MRR: {mrr:.4f} (Benchmark: >= 0.15)")

        results[mode] = {
            "context_length": max_ctx,
            "evaluated_sentences": eval_count,
            "top1_accuracy": round(top1_acc, 4),
            "top5_hit_rate": round(top5_acc, 4),
            "mrr": round(mrr, 4),
            "time_sec": round(elapsed_mode, 2),
            "meets_benchmarks": top1_acc >= 0.10 and top5_acc >= 0.30 and mrr >= 0.15,
        }

    # Example test prompt: "ኢትዮጵያ ውስጥ የቡና"
    test_phrase = "ኢትዮጵያ ውስጥ የቡና"
    prompt_tokens = [w2id.get(w, UNK_ID) for w in WORD_RE.findall(test_phrase)]
    example_top5_ids = predict_next_word(prompt_tokens, kn_model, max_context_len=2, k=5)
    example_suggestions = [
        {"word": id2w.get(tid, f"<ID:{tid}>"), "prob": round(kn_model.prob(tid, prompt_tokens, order=3), 6)}
        for tid in example_top5_ids
    ]
    print(f"\nExample Prompt: '{test_phrase}' -> Suggestions: {example_suggestions}")

    final_payload = {
        "benchmark_targets": {
            "top1_target": 0.10,
            "top5_target": 0.30,
            "mrr_target": 0.15,
        },
        "metrics": results,
        "example_prompt_output": {
            "prompt": test_phrase,
            "suggestions": example_suggestions,
        },
    }

    tmp_res = RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp_res, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    tmp_res.replace(RESULTS_FILE)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s. Wrote {RESULTS_FILE}")
    record_stage_checkpoint(STAGE_NAME, [RESULTS_FILE])


if __name__ == "__main__":
    main()
