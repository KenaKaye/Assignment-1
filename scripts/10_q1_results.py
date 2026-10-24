"""scripts/10_q1_results.py: Q1 Analysis and Results Generator.

Produces all Question 1 outputs to results/:
- results/q1_top10_n{1..4}.csv (Q1.3)
- results/q1_bigram_conditional.csv (Q1.4)
- results/q1_stopword_top10_n{1..4}.csv (Q1.5)
- results/q1_wordcloud_n{1..4}_{before|after}.csv (Q1.6 top-200)
- results/q1_sentence_probs.csv (Q1.7)
- results/q1_generation.json (Q1.8)
"""

from __future__ import annotations

import csv
import hashlib
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
    FULL_COUNTS_DIR,
    PAD_ID,
    RESULTS_DIR,
    SEED,
    UNK_ID,
    V_LIMIT,
    WORD_RE,
    require_heavy_runtime,
    stage_is_done,
    record_stage,
)
from scripts.kn_lib import KneserNeyModel, pack

STAGE_NAME = "10_q1_results"
VOCAB_FILE = DATA_DIR / "vocab.tsv"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"
TOKENS_FILE = DATA_DIR / "tokens.npy"

# 30 justified Amharic stopwords (conjunctions, prepositions, copulas, particles)
AMHARIC_STOPWORDS = {
    "ነው", "ላይ", "ውስጥ", "እና", "ጋር", "ወደ", "እንደ", "ስለ", "ይህ", "ነበር",
    "አለ", "እስከ", "ከዚህ", "ሆኖ", "በኋላ", "መካከል", "በኩል", "ብቻ", "አይደለም", "ይህን",
    "ወይም", "ሆኖም", "ነገር", "ግን", "እንዲሁም", "በላይ", "ሌላ", "ሁሉ", "በታች", "በፊት"
}


def load_vocab() -> tuple[dict[int, str], dict[str, int]]:
    """Load vocabulary maps from data/vocab.tsv."""
    id2w = {}
    w2id = {}
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                t_id = int(parts[0])
                word = parts[1]
                id2w[t_id] = word
                w2id[word] = t_id
    return id2w, w2id


def top_k_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top k elements in descending order via argpartition."""
    n = len(arr)
    if n <= k:
        return np.argsort(arr)[::-1]
    part = np.argpartition(arr, n - k)[n - k:]
    return part[np.argsort(arr[part])[::-1]]


def main() -> None:
    require_heavy_runtime(STAGE_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    expected_inputs = [
        CLEANED_FILE,
        VOCAB_FILE,
        TOKENS_FILE,
    ]
    for n in range(1, 5):
        expected_inputs.append(FULL_COUNTS_DIR / f"order{n}_ngrams.npy")
        expected_inputs.append(FULL_COUNTS_DIR / f"order{n}_counts.npy")

    expected_outputs = [
        RESULTS_DIR / "q1_sentence_probs.csv",
        RESULTS_DIR / "q1_generation.json",
        RESULTS_DIR / "q1_bigram_conditional.csv",
    ]
    for n in range(1, 5):
        expected_outputs.append(RESULTS_DIR / f"q1_top10_n{n}.csv")
        expected_outputs.append(RESULTS_DIR / f"q1_stopword_top10_n{n}.csv")
        expected_outputs.append(RESULTS_DIR / f"q1_wordcloud_n{n}_before.csv")
        expected_outputs.append(RESULTS_DIR / f"q1_wordcloud_n{n}_after.csv")

    if stage_is_done(STAGE_NAME, expected_inputs, expected_outputs):
        print(f"[{STAGE_NAME}] Checkpoint valid. Skipping.")
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    id2w, w2id = load_vocab()
    stopword_ids = {w2id[w] for w in AMHARIC_STOPWORDS if w in w2id}

    # Load token length
    tokens_mmap = np.load(TOKENS_FILE, mmap_mode="r")
    total_tokens = len(tokens_mmap)
    del tokens_mmap

    # Load full counts
    ngrams = {}
    counts = {}
    for n in range(1, 5):
        print(f"Loading full counts for n={n}...")
        ngrams[n] = np.load(FULL_COUNTS_DIR / f"order{n}_ngrams.npy")
        counts[n] = np.load(FULL_COUNTS_DIR / f"order{n}_counts.npy")

    # Q1.3: Sum checks & Top 10 tables
    print("\n[Q1.3] Sum checks and Top-10 n-grams...")
    for n in range(1, 5):
        c_sum = int(counts[n].sum())
        assert c_sum == total_tokens, f"Order {n} sum {c_sum} != total tokens {total_tokens}"
        prob_sum = float(np.sum(counts[n] / total_tokens))
        diff = abs(prob_sum - 1.0)
        assert diff <= 1e-9, f"Order {n} probability sum {prob_sum} deviates by {diff:.2e}"
        print(f"  [Sum check n={n}] counts={c_sum:,}, sum(P)={prob_sum:.10f}, diff={diff:.2e}")

        top10_idx = top_k_indices(counts[n], 10)
        csv_file = RESULTS_DIR / f"q1_top10_n{n}.csv"
        with open(csv_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "ngram", "count", "probability"])
            for rank, idx in enumerate(top10_idx, start=1):
                row = ngrams[n][idx]
                cnt = int(counts[n][idx])
                text_ngram = " ".join(id2w.get(tid, f"<ID:{tid}>") for tid in row)
                prob = cnt / float(total_tokens)
                writer.writerow([rank, text_ngram, cnt, f"{prob:.10e}"])
        print(f"  Wrote {csv_file.name}")

    # Q1.4: Bigram conditional probabilities
    print("\n[Q1.4] Computing conditional probabilities for chosen pairs...")
    test_pairs = [
        ("ኢትዮጵያ", "ውስጥ"),
        ("አዲስ", "አበባ"),
        ("መንግስት", "እና"),
        ("ሰማይ", "ድንጋይ"),  # Unseen pair
    ]
    cond_file = RESULTS_DIR / "q1_bigram_conditional.csv"
    with open(cond_file, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["w1", "w2", "count_w1", "count_w1_w2", "P(w2|w1)"])
        for w1, w2 in test_pairs:
            id1 = w2id.get(w1, UNK_ID)
            id2 = w2id.get(w2, UNK_ID)

            # Find count(w1)
            u_idx = np.searchsorted(ngrams[1][:, 0], id1)
            c_w1 = int(counts[1][u_idx]) if (u_idx < len(ngrams[1]) and ngrams[1][u_idx, 0] == id1) else 0

            # Find count(w1, w2)
            bg = ngrams[2]
            l = int(np.searchsorted(bg[:, 0], id1, side="left"))
            r = int(np.searchsorted(bg[:, 0], id1, side="right"))
            c_w1_w2 = 0
            if r > l:
                sub = bg[l:r, 1]
                sub_idx = int(np.searchsorted(sub, id2))
                if sub_idx < len(sub) and sub[sub_idx] == id2:
                    c_w1_w2 = int(counts[2][l + sub_idx])

            prob = (c_w1_w2 / c_w1) if c_w1 > 0 else 0.0
            writer.writerow([w1, w2, c_w1, c_w1_w2, f"{prob:.8f}"])
            print(f"  P({w2} | {w1}) = {c_w1_w2} / {c_w1} = {prob:.6f}")

    # Q1.5 & Q1.6: Stopword removal top-10 & wordcloud frequency tables
    print("\n[Q1.5 & Q1.6] Generating before/after stopword removal tables...")
    for n in range(1, 5):
        # Top 200 before
        top_before_idx = top_k_indices(counts[n], 200)
        before_file = RESULTS_DIR / f"q1_wordcloud_n{n}_before.csv"
        with open(before_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["ngram", "count"])
            for idx in top_before_idx:
                row = ngrams[n][idx]
                text_ngram = " ".join(id2w.get(tid, f"<ID:{tid}>") for tid in row)
                writer.writerow([text_ngram, int(counts[n][idx])])

        # Filter out rows that contain any stopword ID
        ng_n = ngrams[n]
        is_stop = np.zeros(len(ng_n), dtype=bool)
        for col_idx in range(n):
            is_stop |= np.isin(ng_n[:, col_idx], list(stopword_ids))

        filtered_indices = np.where(~is_stop)[0]
        sorted_rel = top_k_indices(counts[n][filtered_indices], min(200, len(filtered_indices)))
        sorted_filtered = filtered_indices[sorted_rel]

        # Top 10 after
        stop_top10_file = RESULTS_DIR / f"q1_stopword_top10_n{n}.csv"
        with open(stop_top10_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "ngram", "count"])
            for rank, idx in enumerate(sorted_filtered[:10], start=1):
                row = ng_n[idx]
                text_ngram = " ".join(id2w.get(tid, f"<ID:{tid}>") for tid in row)
                writer.writerow([rank, text_ngram, int(counts[n][idx])])

        # Top 200 after
        after_file = RESULTS_DIR / f"q1_wordcloud_n{n}_after.csv"
        with open(after_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["ngram", "count"])
            for idx in sorted_filtered[:200]:
                row = ng_n[idx]
                text_ngram = " ".join(id2w.get(tid, f"<ID:{tid}>") for tid in row)
                writer.writerow([text_ngram, int(counts[n][idx])])

    # Hash EVERY line of GPAC_cleaned.txt for verbatim checks (10-6)
    print("\n[Stage 10] Hashing all lines of GPAC_cleaned.txt for 100% full-corpus verbatim checking...")
    corpus_hashes = []
    with open(CLEANED_FILE, "rb") as f:
        for line in f:
            line_clean = line.strip()
            if line_clean:
                h = int.from_bytes(hashlib.blake2b(line_clean, digest_size=8).digest(), "little")
                corpus_hashes.append(h)
    sorted_corpus_hashes = np.array(corpus_hashes, dtype=np.uint64)
    sorted_corpus_hashes.sort()
    print(f"  Indexed {len(sorted_corpus_hashes):,} sentence hashes.")

    # Initialize KneserNeyModel on full corpus
    print("\nInitializing KneserNeyModel on full-corpus counts...")
    kn_model = KneserNeyModel(ngrams, counts, max_order=4)

    # Q1.7: Target sentence log-probability and comparisons
    print("\n[Q1.7] Evaluating Kneser-Ney sentence probabilities & verbatim status...")
    test_sentences = [
        ("ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።", "test_1"),
        ("የኢትዮጵያ መንግስት ለህዝቡ ሰላም እና ልማት ይሰራል።", "test_2"),
        ("አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።", "test_3"),
        ("ድንጋይ ሰማይ ሮጠ በላ ትላንት ወንበር።", "invented_nonsense"),
    ]

    probs_file = RESULTS_DIR / "q1_sentence_probs.csv"
    with open(probs_file, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "category", "n", "log_prob", "per_token_log_prob", "is_verbatim_in_corpus"])
        for sent_text, cat in test_sentences:
            h_sent = int.from_bytes(hashlib.blake2b(sent_text.strip().encode("utf-8"), digest_size=8).digest(), "little")
            idx = np.searchsorted(sorted_corpus_hashes, h_sent)
            is_verb = bool(idx < len(sorted_corpus_hashes) and sorted_corpus_hashes[idx] == h_sent)

            words = WORD_RE.findall(sent_text)
            token_ids = [w2id.get(w, UNK_ID) for w in words] + [EOS_ID]
            length = len(token_ids)
            tok_arr = np.array(token_ids, dtype=np.int32)
            pos_arr = np.arange(length, dtype=np.int64)

            for n in range(1, 5):
                logp_arr, sent_logp, per_tok = kn_model.score_sequence(tok_arr, pos_arr, order=n)
                logp = float(sent_logp[0])
                per_token = float(per_tok[0])
                writer.writerow([sent_text, cat, n, f"{logp:.4f}", f"{per_token:.4f}", is_verb])
                print(f"  [{cat}] n={n} | LogP: {logp:8.4f} | Per-token: {per_token:8.4f} | Verbatim: {is_verb} | {sent_text}")

    # Q1.8: Generation & verbatim fractions
    print("\n[Q1.8] Generating sentences across orders 1 to 4 using model sampling...")
    generated_results = {}
    valid_unigrams = [tid for tid in kn_model.unique_rows[1][:, 0] if tid not in (PAD_ID, UNK_ID)]

    for n in range(1, 5):
        generated_results[f"n={n}"] = []
        for s_idx in range(3):
            context: list[int] = []
            gen_words = []
            logp_tokens = []

            for _ in range(25):  # max_len
                # Find candidates by context backoff
                cand_set: list[int] = []
                cur_k = n
                while cur_k >= 2:
                    if len(context) >= cur_k - 1:
                        sub_ctx = context[-(cur_k - 1):]
                        q = pack(sub_ctx)
                        keys = kn_model.ctx_keys[cur_k]
                        g = int(np.searchsorted(keys, q))
                        if g < len(keys) and keys[g] == q:
                            st = kn_model.ctx_start[cur_k][g]
                            cnt = kn_model.ctx_n1[cur_k][g]
                            cand_set = list(kn_model.unique_rows[cur_k][st : st + cnt, -1])
                            break
                    cur_k -= 1

                if not cand_set:
                    cand_set = valid_unigrams

                # Always allow EOS_ID, exclude PAD and UNK
                cands = [tid for tid in cand_set if tid not in (PAD_ID, UNK_ID)]
                if len(gen_words) < 3 and EOS_ID in cands:
                    cands.remove(EOS_ID)
                elif EOS_ID not in cands:
                    cands.append(EOS_ID)

                cands_arr = np.array(cands, dtype=np.int32)
                ctx_hist = ([PAD_ID] * max(0, n - 1 - len(context))) + context
                eff_ctx = ctx_hist[-(n - 1):] if n > 1 else []

                # Vectorized probability calculation for candidate words
                if len(eff_ctx) > 0:
                    ctx_mat = np.tile(np.array(eff_ctx, dtype=np.int64), (len(cands_arr), 1))
                else:
                    ctx_mat = np.zeros((len(cands_arr), 0), dtype=np.int64)
                probs = kn_model.prob(ctx_mat, cands_arr, order=n)
                p_sum = probs.sum()
                if p_sum > 0:
                    probs /= p_sum
                else:
                    probs = np.full(len(cands_arr), 1.0 / len(cands_arr))

                next_word = int(rng.choice(cands_arr, p=probs))
                chosen_idx = np.where(cands_arr == next_word)[0][0]
                logp_tokens.append(float(np.log(max(probs[chosen_idx], 1e-12))))

                if next_word == EOS_ID:
                    break
                gen_words.append(id2w.get(next_word, f"<ID:{next_word}>"))
                context.append(next_word)

            gen_text = " ".join(gen_words) + "።"
            h_gen = int.from_bytes(hashlib.blake2b(gen_text.strip().encode("utf-8"), digest_size=8).digest(), "little")
            idx = np.searchsorted(sorted_corpus_hashes, h_gen)
            is_full_copy = bool(idx < len(sorted_corpus_hashes) and sorted_corpus_hashes[idx] == h_gen)

            total_logp = float(sum(logp_tokens))
            generated_results[f"n={n}"].append({
                "sentence": gen_text,
                "length": len(gen_words),
                "log_prob": round(total_logp, 4),
                "is_full_verbatim_copy": is_full_copy,
            })
            print(f"  Generated n={n}: {gen_text} (LogP: {total_logp:.2f}, Verbatim: {is_full_copy})")

    gen_file = RESULTS_DIR / "q1_generation.json"
    with open(gen_file, "w", encoding="utf-8") as f:
        json.dump(generated_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s")
    record_stage(STAGE_NAME, expected_inputs, expected_outputs)


if __name__ == "__main__":
    main()
