"""scripts/13_q4_eval.py: Q4 Predictive Text Evaluation.

Evaluates next-word prediction on held-out test sentences:
- Input: Sentence without its last word
- Ground truth target: The true final word
- Metrics: Top-1 Accuracy, Top-5 Hit Rate, MRR@5 (reciprocal rank if in top 5, else 0)
- Full held-out test split without sampling (sentences with >= 3 words)
- Candidates: union of trigram/bigram continuations and top-50 unigrams by continuation count
- Vectorized chunked scoring via kn_lib
- Produces: results/q4_eval.json
"""

from __future__ import annotations

import json
import math
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
    UNK_ID,
    WORD_RE,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)
from scripts.kn_lib import KneserNeyModel, pack

STAGE_NAME = "13_q4_eval"
SPLIT_FILE = DATA_DIR / "split_indices_seed42_v2.json"
TOKENS_FILE = DATA_DIR / "tokens.npy"
STARTS_FILE = DATA_DIR / "sentence_starts.npy"
TRAIN_COUNTS_DIR = DATA_DIR / "counts" / "train_seed42"
LUT_FILE = TRAIN_COUNTS_DIR / "lut_train_seed42.npy"
VOCAB_FILE = DATA_DIR / "vocab.tsv"
RESULTS_FILE = RESULTS_DIR / "q4_eval.json"

CHUNK_SIZE = 50000  # Target positions per chunk


def load_vocab_maps() -> tuple[dict[int, str], dict[str, int]]:
    id2w = {}
    w2id = {}
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tid = int(parts[0])
                w = parts[1]
                id2w[tid] = w
                w2id[w] = tid
    return id2w, w2id


def get_top50_unigrams(kn_model: KneserNeyModel) -> np.ndarray:
    """Return top 50 unigram IDs by continuation count, excluding PAD, EOS, UNK."""
    unigram_words = kn_model.unique_rows[1][:, 0]
    unigram_counts = kn_model.a[1]
    
    # Exclude reserved IDs (< 3)
    valid_mask = unigram_words >= 3
    valid_words = unigram_words[valid_mask]
    valid_counts = unigram_counts[valid_mask]
    
    k = min(50, len(valid_words))
    part_idx = np.argpartition(valid_counts, -k)[-k:]
    sorted_idx = part_idx[np.argsort(valid_counts[part_idx])[::-1]]
    return valid_words[sorted_idx]


def evaluate_mode(
    mode_name: str,
    context_len: int,
    test_contexts: list[list[int]],
    test_targets: np.ndarray,
    kn_model: KneserNeyModel,
    top50_unigrams: np.ndarray,
) -> dict:
    """Evaluate predictive performance for one context mode using vectorized chunks."""
    order = context_len + 1
    n_targets = len(test_targets)
    print(f"\n--- Evaluating Mode: {mode_name} (Order {order}, Context len {context_len}) ---")
    print(f"Total target positions to evaluate: {n_targets:,}")

    top1_hits = 0
    top5_hits = 0
    rr_sum = 0.0

    t_start = time.time()
    n_chunks = math.ceil(n_targets / CHUNK_SIZE)

    for chunk_idx in range(n_chunks):
        c_t0 = time.time()
        start_i = chunk_idx * CHUNK_SIZE
        end_i = min(start_i + CHUNK_SIZE, n_targets)
        b_size = end_i - start_i

        chunk_ctxs = test_contexts[start_i:end_i]
        chunk_targets = test_targets[start_i:end_i]

        # 1. Candidate generation for each item in chunk
        chunk_cand_list: list[np.ndarray] = []
        item_repeats = np.empty(b_size, dtype=np.int32)

        for i in range(b_size):
            ctx = chunk_ctxs[i]
            cands = set(top50_unigrams)

            # Bigram continuations
            if context_len >= 1 and len(ctx) >= 1:
                w1 = ctx[-1]
                q1 = w1
                keys2 = kn_model.ctx_keys[2]
                g2 = int(np.searchsorted(keys2, q1))
                if g2 < len(keys2) and keys2[g2] == q1:
                    st2 = kn_model.ctx_start[2][g2]
                    cnt2 = kn_model.ctx_n1[2][g2]
                    cands.update(kn_model.unique_rows[2][st2 : st2 + cnt2, 1])

            # Trigram continuations
            if context_len >= 2 and len(ctx) >= 2:
                q2 = pack(ctx[-2:])
                keys3 = kn_model.ctx_keys[3]
                g3 = int(np.searchsorted(keys3, q2))
                if g3 < len(keys3) and keys3[g3] == q2:
                    st3 = kn_model.ctx_start[3][g3]
                    cnt3 = kn_model.ctx_n1[3][g3]
                    cands.update(kn_model.unique_rows[3][st3 : st3 + cnt3, 2])

            cands.discard(PAD_ID)
            cands.discard(EOS_ID)
            cands.discard(UNK_ID)

            c_arr = np.array(list(cands), dtype=np.int32)
            chunk_cand_list.append(c_arr)
            item_repeats[i] = len(c_arr)

        # 2. Vectorized probability calculation for candidate pool
        total_cands = int(item_repeats.sum())
        all_cands = np.concatenate(chunk_cand_list)

        # Build context columns matching candidates
        ctx_cols = []
        for c_idx in range(context_len):
            col_vals = np.empty(b_size, dtype=np.int64)
            for i in range(b_size):
                ctx = chunk_ctxs[i]
                needed_pad = context_len - len(ctx)
                padded_ctx = ([PAD_ID] * max(0, needed_pad)) + ctx
                col_vals[i] = padded_ctx[c_idx]
            ctx_cols.append(np.repeat(col_vals, item_repeats))

        # Vectorized KN probability
        probs = kn_model.prob(ctx_cols, all_cands, order=order)

        # 3. Rank candidates per target using lexsort (item 13-4, 13-5)
        # Ties broken by ascending word ID
        offset = 0
        for i in range(b_size):
            k_len = item_repeats[i]
            c_words = all_cands[offset : offset + k_len]
            c_probs = probs[offset : offset + k_len]
            offset += k_len

            # lexsort primary: -c_probs (descending), secondary: c_words (ascending)
            sort_order = np.lexsort((c_words, -c_probs))
            top_k_words = c_words[sort_order[:5]]

            target = chunk_targets[i]
            if len(top_k_words) > 0 and top_k_words[0] == target:
                top1_hits += 1

            matches = np.where(top_k_words == target)[0]
            if len(matches) > 0:
                top5_hits += 1
                rank = int(matches[0]) + 1
                rr_sum += 1.0 / rank

        c_elapsed = time.time() - c_t0
        if chunk_idx == 0:
            est_total = c_elapsed * n_chunks
            print(f"  Chunk 1/{n_chunks} done in {c_elapsed:.2f} s. Estimated total time: {est_total / 60:.1f} min")
        elif (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
            print(f"  Chunk {chunk_idx + 1}/{n_chunks} processed in {c_elapsed:.2f} s")

    elapsed_mode = time.time() - t_start
    top1_acc = top1_hits / n_targets if n_targets > 0 else 0.0
    top5_acc = top5_hits / n_targets if n_targets > 0 else 0.0
    mrr_5 = rr_sum / n_targets if n_targets > 0 else 0.0

    print(f"  [Mode {mode_name}] Top-1 Accuracy: {top1_acc:.4f} | Top-5 Hit Rate: {top5_acc:.4f} | MRR@5: {mrr_5:.4f}")

    meets_benchmarks = top1_acc >= 0.10 and top5_acc >= 0.30 and mrr_5 >= 0.15
    return {
        "context_length": context_len,
        "evaluated_targets": n_targets,
        "top1_accuracy": round(top1_acc, 4),
        "top5_hit_rate": round(top5_acc, 4),
        "mrr_at_5": round(mrr_5, 4),
        "time_sec": round(elapsed_mode, 2),
        "proposed_targets": "Top-1 >= 0.10, Top-5 >= 0.30, MRR@5 >= 0.15 (proposed target set by group)",
        "meets_benchmarks": bool(meets_benchmarks),
    }


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_inputs = [
        SPLIT_FILE,
        TOKENS_FILE,
        STARTS_FILE,
        LUT_FILE,
        VOCAB_FILE,
        TRAIN_COUNTS_DIR / "order1_counts.npy",
    ]
    for n in range(1, 4):
        expected_inputs.append(TRAIN_COUNTS_DIR / f"order{n}_ngrams.npy")
        expected_inputs.append(TRAIN_COUNTS_DIR / f"order{n}_counts.npy")

    if stage_is_done(STAGE_NAME, expected_inputs, [RESULTS_FILE]):
        print(f"[{STAGE_NAME}] Checkpoint valid. Skipping.")
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    id2w, w2id = load_vocab_maps()

    print(f"Loading {SPLIT_FILE.name}...")
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_indices = np.array(split_data["test"], dtype=np.int64)

    print(f"Loading {TOKENS_FILE.name} and {STARTS_FILE.name}...")
    tokens = np.load(TOKENS_FILE, mmap_mode="r")
    starts = np.load(STARTS_FILE)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(tokens)

    print(f"Loading LUT: {LUT_FILE.name}...")
    lut = np.load(LUT_FILE)

    # Load unigram raw counts for training count verification (item 13-9)
    raw_c1 = np.load(TRAIN_COUNTS_DIR / "order1_counts.npy")
    raw_u1 = np.load(TRAIN_COUNTS_DIR / "order1_ngrams.npy")[:, 0]

    def get_train_count(tid: int) -> int:
        idx = np.searchsorted(raw_u1, tid)
        if idx < len(raw_u1) and raw_u1[idx] == tid:
            return int(raw_c1[idx])
        return 0

    print("Extracting test sentences and filtering targets...")
    test_starts = starts[test_indices]
    test_lengths = (ends[test_indices] - test_starts).astype(np.int64)
    # Require at least 3 words excluding EOS, so length including EOS >= 4
    valid_mask = test_lengths >= 4
    test_indices = test_indices[valid_mask]
    test_starts = test_starts[valid_mask]
    test_lengths = test_lengths[valid_mask]

    evaluated_contexts: list[list[int]] = []
    evaluated_targets_list: list[int] = []
    skipped_unk_targets = 0

    for s_start, s_len in zip(test_starts, test_lengths):
        seq = lut[tokens[s_start : s_start + s_len]]
        # seq ends with EOS_ID
        words = seq[:-1]
        target = int(words[-1])
        if target == UNK_ID:
            skipped_unk_targets += 1
            continue
        evaluated_contexts.append(list(words[:-1]))
        evaluated_targets_list.append(target)

    evaluated_targets = np.array(evaluated_targets_list, dtype=np.int32)
    print(f"Test Set Summary: {len(test_indices):,} candidates | {len(evaluated_targets):,} evaluated | {skipped_unk_targets:,} skipped (<UNK>)")

    print("\nLoading KneserNeyModel from train_seed42...")
    kn_model = KneserNeyModel.load_from_dir(TRAIN_COUNTS_DIR, max_order=3)
    top50_unigrams = get_top50_unigrams(kn_model)
    print(f"  Top 50 unigram continuation pool prepared: {len(top50_unigrams)} tokens")

    # Evaluate one-word context (bigram mode)
    res_one_word = evaluate_mode(
        "one_word_context", 1, evaluated_contexts, evaluated_targets, kn_model, top50_unigrams
    )

    # Evaluate two-word context (trigram mode)
    res_two_word = evaluate_mode(
        "two_word_context", 2, evaluated_contexts, evaluated_targets, kn_model, top50_unigrams
    )

    # Example test prompt: "ኢትዮጵያ ውስጥ የቡና"
    test_phrase = "ኢትዮጵያ ውስጥ የቡና"
    prompt_words = WORD_RE.findall(test_phrase)
    prompt_ids = [lut[w2id.get(w, UNK_ID)] for w in prompt_words]
    
    # Generate suggestions for example prompt
    cand_set = set(top50_unigrams)
    if len(prompt_ids) >= 1:
        w1 = prompt_ids[-1]
        keys2 = kn_model.ctx_keys[2]
        g2 = int(np.searchsorted(keys2, w1))
        if g2 < len(keys2) and keys2[g2] == w1:
            st2 = kn_model.ctx_start[2][g2]
            cand_set.update(kn_model.unique_rows[2][st2 : st2 + kn_model.ctx_n1[2][g2], 1])
    if len(prompt_ids) >= 2:
        q2 = pack(prompt_ids[-2:])
        keys3 = kn_model.ctx_keys[3]
        g3 = int(np.searchsorted(keys3, q2))
        if g3 < len(keys3) and keys3[g3] == q2:
            st3 = kn_model.ctx_start[3][g3]
            cand_set.update(kn_model.unique_rows[3][st3 : st3 + kn_model.ctx_n1[3][g3], 2])

    cand_set.discard(PAD_ID)
    cand_set.discard(EOS_ID)
    cand_set.discard(UNK_ID)
    cand_arr = np.array(list(cand_set), dtype=np.int32)

    prompt_ctx = prompt_ids[-2:]
    ctx_cols = [np.full(len(cand_arr), prompt_ctx[c], dtype=np.int64) for c in range(len(prompt_ctx))]
    cand_probs = kn_model.prob(ctx_cols, cand_arr, order=3)
    sort_order = np.lexsort((cand_arr, -cand_probs))
    top5_ex = cand_arr[sort_order[:5]]

    example_suggestions = []
    print(f"\n[Q13-9 Verification] Suggestions for '{test_phrase}':")
    for tid in top5_ex:
        word = id2w.get(tid, f"<ID:{tid}>")
        cnt = get_train_count(tid)
        prob = float(cand_probs[np.where(cand_arr == tid)[0][0]])
        example_suggestions.append({
            "token_id": int(tid),
            "word": word,
            "train_count": cnt,
            "prob": round(prob, 6),
            "verified_in_corpus": cnt > 0,
        })
        print(f"  Word: {word:15} | Count in Train: {cnt:8,} | Prob: {prob:.6f} | Verified: {cnt > 0}")

    final_payload = {
        "benchmark_targets": {
            "top1_target": 0.10,
            "top5_target": 0.30,
            "mrr_target": 0.15,
            "label": "proposed target set by the group, not a published standard",
        },
        "target_counts": {
            "evaluated_targets": len(evaluated_targets),
            "skipped_unk_targets": skipped_unk_targets,
            "total_candidate_sentences": len(test_indices),
        },
        "metrics": {
            "one_word_context": res_one_word,
            "two_word_context": res_two_word,
        },
        "example_prompt_output": {
            "prompt": test_phrase,
            "prompt_token_ids": [int(x) for x in prompt_ids],
            "suggestions": example_suggestions,
        },
    }

    tmp_res = RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp_res, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    tmp_res.replace(RESULTS_FILE)

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed} s. Wrote {RESULTS_FILE}")
    record_stage(STAGE_NAME, expected_inputs, [RESULTS_FILE])


if __name__ == "__main__":
    main()
