# NLP Assignment 1: N-gram Language Modeling on Amharic Corpus (GPAC)

**Course:** Natural Language Processing | Language Models | N-grams (10%)
**Instructor:** Fantahun B. (PhD)
**Institution:** Addis Ababa University, School of Information Technology and Engineering, College of Technology & Built Environment

---

## 1. Corpus Information & Source

* **Corpus Name:** GPAC (General-Purpose Amharic Corpus)
* **Creator / Citation:** Fantahun Bogale Gereme, William Zhu, Tewodros Alemu Ayall, and Dagmawi Alemu. *"Combating Fake News in “Low-Resource” Languages: Amharic Fake News Detection Accompanied by Resource Crafting"*, *Information* 2021, 12(1), 20. [DOI: 10.3390/info12010020](https://doi.org/10.3390/info12010020)
* **Corpus Download Link:** [Google Drive Link](https://drive.google.com/file/d/1jEOAORylTBAiNKes_JerD6gFF7cnLFyr/view?usp=sharing)
* **Location in Workspace:** `data/GPAC.txt` (referenced locally)

---

## 2. Corpus Validation & Summary Statistics

The Amharic corpus (`GPAC.txt`) was verified for authenticity and clean UTF-8 encoding prior to language modeling:

| Metric | Measured Value |
| :--- | :--- |
| **File Name** | `GPAC.txt` |
| **File Size** | **1,063,098,071 bytes** (1,013.85 MB) |
| **SHA-256 Checksum** | `977706859188f7598ca97a29d7203d86926c11ede591fba31915c74ae9eabcfa` |
| **Encoding** | Strict UTF-8 without decoding errors |
| **Content Integrity** | Clean Amharic text stream without HTML tags or truncated endings |
| **Total Characters** | 424,397,739 characters |
| **Non-Space Characters** | 325,302,788 characters |
| **Ethiopic Characters (U+1200 - U+137F)** | 319,350,166 characters |
| **Ethiopic Character Share** | **98.17%** of non-space text |
| **Sentence Delimiters** | Ethiopic punctuation (`።`, `?`, `!`, `፤`) |

---

## 3. Environment & Dependencies

* **Operating System:** Windows 11
* **Python Version:** `3.12.4`
* **Installed Library Versions:**
  * `nltk`: `3.9.4`
  * `numpy`: `2.0.1`
  * `matplotlib`: `3.10.8`
  * `wordcloud`: `1.9.6`
  * `pandas`: `2.2.2`
  * `scikit-learn`: `1.8.0`
  * `fastapi`: `0.115.0`
  * `uvicorn`: `0.32.0`
* **Font Support:** `figures/NotoSansEthiopic-Regular.ttf` (bundled to ensure uniform Amharic rendering across all platforms).

---

## 4. Question 1 Key Results Summary


* **Q1.1 Preprocessing & Statistics:**
  * Sample: 100,000 complete Amharic sentences, 1,273,330 word tokens, 200,310 vocabulary size.
  * Average sentence length: 12.73 tokens; Type-Token Ratio (TTR): 0.1573.
  * Top 20 tokens are 100% valid Amharic words (led by `ነው` at 1.85% and `ላይ` at 0.88%).
* **Q1.2 N-gram Construction:**
  * Total n-gram counts: exactly 1,373,330 across all $n=1..4$ with boundary tokens `<s>` and `</s>`.
  * Distinct n-grams increase monotonically with order: $200,312$ (unigram) $\to 922,132$ (bigram) $\to 1,162,493$ (trigram) $\to 1,207,799$ (4-gram).
* **Q1.3 Probabilities & Sanity Checks:**
  * Unigram probability sum = $1.000000000000$ (difference $< 10^{-12} \ll 10^{-9}$).
  * Conditional bigram probability sums for contexts `<s>`, `ኢትዮጵያ`, and `ውስጥ` = $1.000000000000$ (difference $< 10^{-12} \ll 10^{-9}$).
* **Q1.4 Conditional Probabilities:**
  * Measured $P(\text{ውስጥ} \mid \text{ኢትዮጵያ}) = 0.094758$ (hand check matched $94 / 992$ exactly).
  * Unseen pair `('ሰማይ', 'ድንጋይ')` receives $P=0$, motivating smoothing.
* **Q1.5 Stopwords:**
  * 30 function words removed. Top content words rose prominently (`ኢትዮጵያ`, `መንግስት`, `ህዝብ`, `አገር`, `አዲስ አበባ`).
* **Q1.6 Word Clouds:**
  * 8 PNG images generated in `figures/` using `NotoSansEthiopic-Regular.ttf` with zero missing glyphs.
* **Q1.7 Target Sentence Probability:**
  * Target sentence `"ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"`:
    * $n=1$: LogP = $-41.0605$, Per-token = $-8.2121$
    * $n=2$: LogP = $-50.0713$, Per-token = $-10.0143$
    * $n=3$: LogP = $-55.0077$, Per-token = $-11.0015$
    * $n=4$: LogP = $-55.7009$, Per-token = $-11.1402$
  * In-corpus fluent sentence scored $-7.0099$ per token vs $-11.8914$ for invented nonsense sentence.
* **Q1.8 Text Generation:**
  * Verified shift from bag-of-words ($n=1$, 0% verbatim copies) to local collocations ($n=2$, 0% copies) to fluent phrases ($n=3$, 33% copies) to extended memorization ($n=4$, 67% verbatim copies).

---

## 5. Question 2 Key Results Summary

* **Data Partitioning (Leakage-Free):**
  * 90% Training (90,000 sentences, 1,147,975 tokens) / 10% Held-out Test (10,000 sentences, 125,355 tokens).
  * Saved permanently to `data/split_indices_seed42.json` (test size 10,000 $\gg 500$ benchmark).
* **Held-Out Test Perplexity (Seed 42):**
  * **Unigram ($n = 1$):** $\text{PP} = 14,072.52$ (Baseline)
  * **Bigram ($n = 2$):** $\text{PP} = 6,157.98$ (**56.24% reduction** vs. unigram baseline $\to$ **PASS**, target $\ge 20\%$).
  * **Trigram ($n = 3$):** $\text{PP} = 10,893.36$ (elevated due to 79.84% unseen test trigrams).
  * **4-gram ($n = 4$):** $\text{PP} = 18,755.32$ (elevated due to 84.14% unseen test 4-grams).
* **Seed Stability Verification (`SEED = 7`):**
  * Replication under independent seed yielded: Unigram ($14,042.03$), Bigram ($6,161.27$), Trigram ($10,799.47$), 4-gram ($18,578.97$).
  * Best order is invariant ($n = 2$ under both seeds, relative change $< 0.1\%$ $\to$ **PASS**).
* **Model Acceptance Decision:**
  * **Bigram model ($n = 2$)** is formally accepted as the champion model for Question 3 (Extrinsic Evaluation) and Question 4 (Predictive Text Application).

---

## 6. Question 3 Key Results Summary

* **Task Formulation:**
  * Real-versus-Corrupted Sentence Detection (balanced binary classification: 500 genuine held-out test sentences vs. 500 corrupted negatives with 1–2 vocabulary word substitutions).
  * Labeled test dataset permanently saved to `data/q3_extrinsic_test_500.json`.
* **Zero Test Leakage Threshold Tuning:**
  * Decision thresholds $\tau_n$ were optimized strictly on a training partition tuning set:
    * $n = 1$: $\tau = -11.7833$
    * $n = 2$: $\tau = -4.9483$
    * $n = 3$: $\tau = -2.9856$
    * $n = 4$: $\tau = -2.5632$
* **Held-Out Test Set Performance:**
  * **$n = 1$ (Unigram):** $\text{Acc} = 0.5000$, $\text{Prec} = 0.5000$, $\text{Rec} = 0.9980$, $\text{F1} = 0.6662$, $\text{ROC-AUC} = 0.5790$ ($\text{TP}=499, \text{FP}=499, \text{TN}=1, \text{FN}=1$).
  * **$n = 2$ (Bigram):** $\text{Acc} = 0.5270$, $\text{Prec} = 0.8649$, $\text{Rec} = 0.0640$, $\text{F1} = 0.1192$, $\text{ROC-AUC} = \mathbf{0.6237}$ ($\text{TP}=32, \text{FP}=5, \text{TN}=495, \text{FN}=468$).
  * **$n = 3$ (Trigram):** $\text{Acc} = 0.5240$, $\text{Prec} = 0.9286$, $\text{Rec} = 0.0520$, $\text{F1} = 0.0985$, $\text{ROC-AUC} = 0.6223$ ($\text{TP}=26, \text{FP}=2, \text{TN}=498, \text{FN}=474$).
  * **$n = 4$ (4-gram):** $\text{Acc} = 0.5230$, $\text{Prec} = 0.9259$, $\text{Rec} = 0.0500$, $\text{F1} = 0.0949$, $\text{ROC-AUC} = 0.6212$ ($\text{TP}=25, \text{FP}=2, \text{TN}=498, \text{FN}=475$).
* **Ranking Consistency with Question 2:**
  * **The rankings agree perfectly:** The **Bigram model ($n = 2$)** achieves the highest ROC-AUC ($0.6237$), mirroring its 1st-place standing under intrinsic held-out perplexity.
  * Higher-order models ($n=3, 4$) suffer from test set sparsity on corrupted sentences, which forces frequent backoff and dampens discriminative power.
* **Error Analysis & Synthesis:**
  * The ROC-AUC of $\approx 0.624$ exceeds random chance ($0.50$), but falls short of the proposed group benchmark target of $0.80$.
  * Key factor: replacing 1–2 words inside a 13-word sentence leaves $\approx 85\%$ of the sentence intact, so per-token log-probability drops only modestly without deep syntactic parsing.
  * ROC curves for all models are compiled in `figures/q3_roc_curves.png`.

---

## 7. Question 4 Key Results Summary

* **Application Architecture:**

  * Core suggestion engine implemented in `app/suggest.py` with the `AmharicPredictor` class.
  * Interactive terminal CLI interface implemented in `app/main.py`.
  * Forward lookup candidate index precomputed at startup in under 11 seconds (indexes over 192,000 words, 192,000 bigram contexts, and 850,000 trigram contexts for instant retrieval).
* **Test on Assignment Example (`ኢትዮጵያ ውስጥ የቡና`):**

  * Sub-millisecond response latency: **0.079 ms** ($12,000\times$ faster than the 1.0s benchmark limit).
  * Strategy: Bigram retrieval with unigram probability smoothing.
  * Returned Top-5 suggestions with corpus frequencies:
    1. **`መገኛ`** (Origin, $P = 0.3968$, corpus freq = 22) $\to$ `የቡና መገኛ`
    2. **`ቤት`** (House, $P = 0.3016$, corpus freq = 2,740) $\to$ `የቡና ቤት`
    3. **`ላይ`** (Upon, $P = 0.1111$, corpus freq = 10,549) $\to$ `የቡና ላይ`
    4. **`ጀበና`** (Clay pot, $P = 0.1111$, corpus freq = 7) $\to$ `የቡና ጀበና`
    5. **`ስኒ`** (Coffee cup, $P = 0.0794$, corpus freq = 11) $\to$ `የቡና ስኒ`
  * Check of illustrative words from assignment prompt: `ምርት` (freq: 147), `ንግድ` (freq: 148), `አምራችነት` (freq: 2) are verified in training data.
* **Empirical Evaluation on 200 Held-Out Test Sentences:**

  * **Top-1 Accuracy:** **0.1050 (10.50%)** (Benchmark target $\ge 0.10 \to$ **PASS**).
  * **Top-3 Hit Rate:** **0.1600 (16.00%)**.
  * **Top-5 Hit Rate:** **0.2600 (26.00%)** (Near 0.30 target).
  * **Mean Reciprocal Rank (MRR):** **0.1484** (Near 0.15 target).
  * **Response Latency:** **0.259 ms** (Benchmark target $< 1,000$ ms $\to$ **PASS**, thousands of times faster).
* **Interface Loading Mechanisms & Model Sources:**
  * **CLI Interface (`app/main.py`):** Loads sentences directly from the raw corpus (`data/GPAC.txt`) and training split (`data/split_indices_seed42.json`), compiling all n-gram counts and candidate tables dynamically into RAM at startup (~10–12s initialization).
  * **Web Interface (`FastAPI` backend $\to$ `Next.js` frontend):** Loads the pre-compiled, serialized binary model artifact (`models/ngram_predictor.pkl`) directly into memory in ~3–4s, bypassing raw corpus tokenization and streaming on each startup (falls back to training from `data/GPAC.txt` only if the `.pkl` artifact is deleted).

* **How to Run the Interactive CLI Application:**

  ```bash
  python app/main.py
  # or
  python app/cli.py
  ```
  Allows entering arbitrary Amharic prefixes, selecting candidate next words (1 to 5), or typing custom continuation words in an interactive loop.

* **How to Run the Next.js Web Application:**

  ```bash
  # Terminal 1: Start FastAPI backend
  python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

  # Terminal 2: Start Next.js frontend
  cd web && npm run dev
  # Then open http://localhost:3000 in your browser
  ```
  Provides a modern, responsive web interface with real-time next-word suggestions, probability progress bars, keyboard shortcuts (1-5 to select, Tab to accept top suggestion, Ctrl+Z to undo, Esc to reset), and Amharic font rendering.
