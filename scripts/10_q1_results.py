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
    WORD_RE,
    is_checkpoint_done,
    record_stage_checkpoint,
    require_colab,
)
import importlib

scripts_09 = importlib.import_module("scripts.09_kn")
KneserNeyModel = scripts_09.KneserNeyModel

STAGE_NAME = "10_q1_results"
VOCAB_FILE = DATA_DIR / "vocab.tsv"
CLEANED_FILE = DATA_DIR / "GPAC_cleaned.txt"

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


def main() -> None:
    require_colab(STAGE_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    expected_files = [
        RESULTS_DIR / "q1_sentence_probs.csv",
        RESULTS_DIR / "q1_generation.json",
        RESULTS_DIR / "q1_bigram_conditional.csv",
    ]
    for n in range(1, 5):
        expected_files.append(RESULTS_DIR / f"q1_top10_n{n}.csv")
        expected_files.append(RESULTS_DIR / f"q1_stopword_top10_n{n}.csv")
        expected_files.append(RESULTS_DIR / f"q1_wordcloud_n{n}_before.csv")
        expected_files.append(RESULTS_DIR / f"q1_wordcloud_n{n}_after.csv")

    if is_checkpoint_done(STAGE_NAME, expected_files):
        return

    print(f"--- Starting Stage: {STAGE_NAME} ---")
    start_time = time.time()

    id2w, w2id = load_vocab()
    stopword_ids = {w2id[w] for w in AMHARIC_STOPWORDS if w in w2id}

    # Load full counts
    ngrams = {}
    counts = {}
    for n in range(1, 5):
        print(f"Loading full counts for n={n}...")
        ngrams[n] = np.load(FULL_COUNTS_DIR / f"order{n}_ngrams.npy")
        counts[n] = np.load(FULL_COUNTS_DIR / f"order{n}_counts.npy")

    # Q1.3: Top 10 tables and sanity check sums
    print("\n[Q1.3] Computing Top-10 n-grams and probabilities...")
    total_tokens = counts[1].sum()
    for n in range(1, 5):
        top_idx = np.argsort(counts[n])[::-1][:10]
        csv_file = RESULTS_DIR / f"q1_top10_n{n}.csv"
        with open(csv_file, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "ngram", "count", "probability"])
            for rank, idx in enumerate(top_idx, start=1):
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
        top_before_idx = np.argsort(counts[n])[::-1][:200]
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

        filtered_idx = np.where(~is_stop)[0]
        sorted_filtered = filtered_idx[np.argsort(counts[n][filtered_idx])[::-1]]

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

    # Q1.7: Target sentence log-probability and comparisons
    print("\n[Q1.7] Evaluating Kneser-Ney sentence probabilities...")
    kn_model = KneserNeyModel(ngrams, counts, max_order=4)

    test_sentences = [
        ("ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።", "canonical_target"),
        ("የኢትዮጵያ መንግስት ለህዝቡ ሰላም እና ልማት ይሰራል::", "corpus_fluent"),
        ("አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።", "corpus_fluent_2"),
        ("ድንጋይ ሰማይ ሮጠ በላ ትላንት ወንበር።", "invented_nonsense"),
    ]

    probs_file = RESULTS_DIR / "q1_sentence_probs.csv"
    with open(probs_file, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "category", "n", "log_prob", "per_token_log_prob"])
        for sent_text, cat in test_sentences:
            words = WORD_RE.findall(sent_text)
            token_ids = [w2id.get(w, UNK_ID) for w in words] + [EOS_ID]
            length = len(token_ids)
            for n in range(1, 5):
                logp = kn_model.sentence_logprob(token_ids, order=n)
                per_token = logp / length
                writer.writerow([sent_text, cat, n, f"{logp:.4f}", f"{per_token:.4f}"])
                print(f"  [{cat}] n={n} | LogP: {logp:8.4f} | Per-token: {per_token:8.4f} | {sent_text}")

    # Q1.8: Generation & verbatim fractions
    print("\n[Q1.8] Generating sentences across orders 1 to 4...")
    # Hash 100,000 corpus sentences to check verbatim full copies
    seen_sentence_hashes = set()
    if CLEANED_FILE.exists():
        with open(CLEANED_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                h = hashlib.blake2b(line.strip().encode("utf-8"), digest_size=16).digest()
                seen_sentence_hashes.add(h)
                if i >= 100000:
                    break

    generated_results = {}
    for n in range(1, 5):
        generated_results[f"n={n}"] = []
        for s_idx in range(3):
            # Generation loop
            context: list[int] = []
            gen_words = []
            for _ in range(25):  # max_len
                needed_pad = (n - 1) - len(context)
                eff_context = ([PAD_ID] * max(0, needed_pad)) + context

                # Draw top candidates
                if n == 1 or len(context) == 0:
                    # Unigram sample from top 100
                    top_candidates = ngrams[1][:100, 0]
                else:
                    h = eff_context[-(n - 1):]
                    l, r = kn_model._find_ngram_range(n, h)
                    if r > l:
                        top_candidates = ngrams[n][l:r, -1]
                    else:
                        top_candidates = ngrams[1][:50, 0]

                # Filter EOS early if too short
                valid_cand = [tid for tid in top_candidates if tid not in (PAD_ID, UNK_ID)]
                if len(gen_words) < 3 and EOS_ID in valid_cand:
                    valid_cand.remove(EOS_ID)
                if not valid_cand:
                    break

                # Sample candidate
                next_word = int(random.choice(valid_cand[:10]))
                if next_word == EOS_ID:
                    break
                gen_words.append(id2w.get(next_word, f"<ID:{next_word}>"))
                context.append(next_word)

            gen_text = " ".join(gen_words) + "።"
            # Verbatim check
            h_gen = hashlib.blake2b(gen_text.encode("utf-8"), digest_size=16).digest()
            is_full_copy = h_gen in seen_sentence_hashes

            generated_results[f"n={n}"].append({
                "sentence": gen_text,
                "length": len(gen_words),
                "is_full_verbatim_copy": is_full_copy,
            })
            print(f"  Generated n={n}: {gen_text}")

    gen_file = RESULTS_DIR / "q1_generation.json"
    with open(gen_file, "w", encoding="utf-8") as f:
        json.dump(generated_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Stage {STAGE_NAME} completed in {elapsed:.2f} s")
    record_stage_checkpoint(STAGE_NAME, expected_files)


if __name__ == "__main__":
    main()
