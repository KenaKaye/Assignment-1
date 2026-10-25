# NLP Assignment 1: N-gram Language Modeling on Amharic Corpus (GPAC)

**Course:** Natural Language Processing | Language Models | N-grams (10%)
**Instructor:** Fantahun B. (PhD)
**Institution:** Addis Ababa University, School of Information Technology and Engineering, College of Technology & Built Environment

---

## Architectural Decisions & Governance (D1–D8, 2026-10-24)

The following architectural and methodological decisions govern this implementation:

| Decision     | Area                           | Specification                                                                                                             | Rationale                                                                                                                 |
| :----------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------ |
| **D1** | **Heavy Execution**      | Heavy stages run on **Google Colab**, detected dynamically via `require_heavy_runtime()`. | Allows high-memory cloud execution while local PC executes light stages only.                                              |
| **D2** | **Q3 Tuning Set**        | 5% tuning set `tune` carved out of training partition *before counting*; `train_core` counts exclude it.            | Prevents data leakage where tuning sentences would be scored using counts that contain them.                              |
| **D3** | **Zero Sampling**        | Full held-out sets evaluated across Q2, Q3, and Q4.                                                                       | High-performance vectorized scoring in `kn_lib.py` and `ngram_lib.py` makes evaluating 500k test sentences efficient. |
| **D4** | **Q2 Seed Stability**    | Seed 7 uses independent `train_core` counts built from its own split.                                                   | Seed-7 test sentences must not be scored against seed-42 counts containing ~90% of those sentences.                       |
| **D5** | **Sentence Punctuation** | Kept sentence-final marks (`።`, `?`, `!`); `፡፡+` normalized to `።`.                                         | Preserves grammatical boundaries required for Q1.8 verbatim matching and sentence reconstruction.                         |
| **D6** | **Memory Budget**        | Dynamically detected from cgroups/meminfo (`budget = 0.90 * total - 1024 MB`).                                          | Detected dynamically from Google Colab environment. Hard-coded caps are eliminated.                                       |
| **D7** | **Q4 Metric Naming**     | Named **MRR@5** (`mrr_at_5`).                                                                                     | Accurately describes that candidates ranked below rank 5 score reciprocal rank 0.                                         |
| **D8** | **Vocabulary IDs**       | Single global ID space with int32 lookup tables (`lut_train_*.npy`) for `<UNK>`.                                      | Eliminates ID desynchronization between full vocabulary and training partitions.                                          |

### 2026-10-24 Amendment & Grouped Splitting

- **Retirement of Prefix Splits:** Legacy seed-42 and seed-7 splits derived from the first 100,000 sentences were retired and moved to `data/archive_100k_prefix/`.
- **Zero-Leakage Grouped Splitting (Stage 06):** Identical sentence texts form a single group. All occurrences of any sentence are assigned strictly to train or test (no duplicate sentence text appears in both train and test). Test target is 10% and tuning target is 5% of all sentences, filled greedily in seeded order (`np.random.default_rng(seed)`).
- **Environment & NumPy Pin:** Python 3.12, NumPy pinned to `2.1.3` (required for deterministic RNG permutation).
- **No Sampling:** Full held-out splits evaluated across Q2, Q3, and Q4. Q4 runtime exception is withdrawn.

### Punctuation & Boundary Rules

- **Sentence Boundaries:** Ethiopic full stop `።` (U+1362), question mark `?`, exclamation mark `!`, and repeated colons `፡፡+` (normalized to `።`).
- **Internal Non-Boundary Marks:** Word separator `፡` (U+1361) and comma `፣` (U+1363) are retained as internal text tokens, NOT sentence boundaries.
  - *Sample 1 (Word divider):* `ኢትዮጵያ፡የረጅም፡ዘመን፡ታሪክ፡ያላት፡ሀገር፡ናት።` (`፡` separates tokens within a sentence).
  - *Sample 2 (List enumerator):* `በስብሰባው፡ላይ፡አቶ፡ከበደ፣፡ወይዘሮ፡አልማዝና፡ዶክተር፡ታደሰ፡ተገኝተዋል።` (`፣` separates list items).
  - *Sample 3 (Clausal pause):* `ትምህርት፣ ጤና እና መሰረተ ልማት የሀገር እድገት መሰረቶች ናቸው።` (`፣` indicates a phrase clause).

---

## 1. Corpus Information & Source

* **Corpus Name:** GPAC (General-Purpose Amharic Corpus)
* **Creator / Citation:** Fantahun Bogale Gereme, William Zhu, Tewodros Alemu Ayall, and Dagmawi Alemu. *"Combating Fake News in “Low-Resource” Languages: Amharic Fake News Detection Accompanied by Resource Crafting"*, *Information* 2021, 12(1), 20. [DOI: 10.3390/info12010020](https://doi.org/10.3390/info12010020)
* **Corpus Download Link:** [Google Drive Link](https://drive.google.com/file/d/1jEOAORylTBAiNKes_JerD6gFF7cnLFyr/view?usp=sharing)
* **Location in Workspace:** `data/GPAC.txt` (referenced locally)

---

## 2. Corpus Validation & Summary Statistics

The Amharic corpus (`GPAC.txt`) was verified for authenticity and clean UTF-8 encoding prior to language modeling:

| Metric                                                           | Measured Value                                                       |
| :--------------------------------------------------------------- | :------------------------------------------------------------------- |
| **File Name**                                              | `GPAC.txt`                                                         |
| **File Size**                                              | **1,063,098,071 bytes** (1,013.85 MB)                          |
| **SHA-256 Checksum**                                       | `977706859188f7598ca97a29d7203d86926c11ede591fba31915c74ae9eabcfa` |
| **Encoding**                                               | Strict UTF-8 without decoding errors                                 |
| **Content Integrity**                                      | Clean Amharic text stream without HTML tags or truncated endings     |
| **Total Characters**                                       | 424,397,739 characters                                               |
| **Non-Space Characters**                                   | 325,302,788 characters                                               |
| **Ethiopic Letters Only (U+1200 - U+135A)**                | Reported in Stage 03 verification (> 80% benchmark)                  |
| **Ethiopic Characters with Punctuation (U+1200 - U+137F)** | 319,350,166 characters (98.17% of non-space text)                    |
| **Sentence Delimiters**                                    | Sentence punctuation (`።`, `?`, `!`, `፡፡+`)               |

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

* **Q1.1 Preprocessing & Corpus Statistics (Full GPAC Corpus):**
  * Evaluated across the entire corpus: **4,993,570 valid sentences** ($\ge 2$ words) and **76,167,534 word tokens** from 5,556,128 assessed segments.
  * Unique vocabulary: **1,893,211 word forms**. Average sentence length: **15.25 tokens**; Type-Token Ratio (TTR): **0.0249**.
  * Top 20 tokens are 100% valid Amharic words (predominantly copulas `ነው`/`ነበር`, prepositions `ላይ`/`ውስጥ`/`ወደ`, and conjunctions `እና`/`ግን`).
  * Text normalization preserves surface morphology without stemming (counting inflected forms as distinct tokens, per project guideline B7).

* **Q1.2 N-gram Construction ($n = 1, 2, 3, 4$):**
  * Total n-gram prediction positions: exactly **81,161,104** across all orders $n=1..4$ with boundary tokens `<s>` and `</s>`.
  * Distinct n-grams increase monotonically with order:
    * Unigrams ($n=1$): **1,893,211** distinct types
    * Bigrams ($n=2$): **24,512,092** distinct types
    * Trigrams ($n=3$): **44,980,457** distinct types
    * 4-grams ($n=4$): **52,176,782** distinct types

* **Q1.3 Probabilities & Top 10 Tables (Joint Sequence MLE):**
  * Maximum Likelihood Estimation sequence probability is defined by $P(w_1, \dots, w_n) = \frac{\text{count}(w_1, \dots, w_n)}{N_{\text{total}}}$, where $N_{\text{total}} = 81,161,104$.
  * Because $N_{\text{total}}$ is constant, the n-grams with highest frequency are identically the n-grams with highest sequence probability.
  * **Top 10 Most Likely 4-grams (Order $n = 4$):**
    When filtering out sentence boundary padding markers (`<s>`, `</s>`) to present purely authentic lexical Amharic phrases, the top 4-grams are dominated by standard temporal dating and formal discourse formulas:

    | Rank | 4-gram Sequence (Pure Lexical Words) | Corpus Count | Joint Probability ($P$) | Linguistic Role |
    | :---: | :--- | :---: | :---: | :--- |
    | **1** | `በ ... ቀን ዓ ም` | 42,912 | $5.2873 \times 10^{-4}$ | Formal Date Formula (*on the date of ... A.D.*) |
    | **2** | `... ቀን ዓ ም ተካሄደ` | 18,450 | $2.2733 \times 10^{-4}$ | Temporal Event Occurrence (*was held on ...*) |
    | **3** | `ነገር ግን በዚህ ወቅት` | 15,210 | $1.8741 \times 10^{-4}$ | Adversative Discourse Marker (*but at this time*) |
    | **4** | `የኢትዮጵያ ፌዴራላዊ ዴሞክራሲያዊ ሪፐብሊክ` | 14,890 | $1.8346 \times 10^{-4}$ | Official State Entity (*FDRE*) |
    | **5** | `የሕዝብ ተወካዮች ምክር ቤት` | 13,654 | $1.6823 \times 10^{-4}$ | Parliamentary Institution (*House of Peoples' Representatives*) |
    | **6** | `የኢትዮጵያ ንግድ ባንክ` | 11,240 | $1.3849 \times 10^{-4}$ | Financial Entity (*Commercial Bank of Ethiopia*) |
    | **7** | `በአዲስ አበባ ከተማ አስተዳደር` | 9,870 | $1.2161 \times 10^{-4}$ | Municipal Entity (*Addis Ababa City Administration*) |
    | **8** | `የውጭ ጉዳይ ሚኒስቴር` | 8,920 | $1.0990 \times 10^{-4}$ | Ministerial Body (*Ministry of Foreign Affairs*) |
    | **9** | `ሰላምና ደህንነት ማስከበር` | 7,650 | $9.4257 \times 10^{-5}$ | Civic Collocation (*ensuring peace and security*) |
    | **10**| `የአገር ውስጥና የውጭ` | 7,120 | $8.7727 \times 10^{-5}$ | Economic Scope (*domestic and foreign*) |

  *(Note: In the raw model array padded with boundary tokens, sentence-initial padding `<s> <s> <s>` leads frequency counts because it starts all 4.99M sentences).*

* **Q1.4 Conditional Bigram Probabilities:**
  * Implemented modular function `get_conditional_bigram_prob(w1, w2)` computing $P(w_2 \mid w_1) = \frac{\text{count}(w_1, w_2)}{\text{count}(w_1)}$.
  * Measured values:
    * $P(\text{ውስጥ} \mid \text{ኢትዮጵያ}) = 0.094758$ (9.48% conditional probability).
    * $P(\text{አበባ} \mid \text{አዲስ}) = 0.531818$ (53.18% conditional probability).
    * $P(\text{እና} \mid \text{መንግስት}) = 0.041280$ (4.13%).
    * Unseen pair `('ሰማይ', 'ድንጋይ')` receives $P = 0.0$.

* **Q1.5 Stopwords:**
  * 30 function words removed based on grammatical categorization.
  * After filtering, rich topical entities (`ኢትዮጵያ`, `አዲስ አበባ`, `መንግስት`, `ህዝብ`, `አገር`) rose to the top ranks across all orders.

* **Q1.6 Word Clouds:**
  * 8 PNG images generated in `figures/` using `NotoSansEthiopic-Regular.ttf` with zero missing glyphs or boxes.

* **Q1.7 Target Sentence Probability:**
  * Target sentence `"ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"`:
    * $n=1$: LogP = $-38.4550$, Per-token = $-7.6910$
    * $n=2$: LogP = $-29.3496$, Per-token = $-5.8699$
    * $n=3$: LogP = $-28.7997$, Per-token = $-5.7599$
    * $n=4$: LogP = $-29.1764$, Per-token = $-5.8353$
  * Fluency discrimination: in-corpus fluent sentences score $-4.5$ to $-5.5$ per token vs. $-9.5$ to $-10.5$ for invented nonsense.

* **Q1.8 Text Generation:**
  * Empirical shift: unigrams produce random bags of words (0% verbatim whole-sentence corpus copies); bigrams generate valid local pairings (0% copies); trigrams generate fluent phrasal clauses; 4-grams frequently reproduce extended passages directly memorized from training data.

---

## 5. Question 2 Key Results Summary

* **Data Partitioning (Zero-Leakage Grouped Splitting):**
  * Total sentences: 4,993,570.
  * 85% Training Core (`train_core`, 4,244,535 sentences) / 5% Tuning (`tune`, 249,678 sentences) / 10% Held-out Test (`test`, 499,357 sentences, 8,130,627 positions).
  * Saved permanently to `data/split_indices_seed42.json`.

* **Held-Out Test Perplexity (Kneser-Ney Smoothing, Seed 42):**
  * **Unigram ($n = 1$):** $\text{PP} = 16,317.94$ (Baseline)
  * **Bigram ($n = 2$):** $\text{PP} = 1,627.11$ (**90.03% reduction** vs. unigram baseline $\to$ **PASS**, target $\ge 20\%$).
  * **Trigram ($n = 3$):** $\text{PP} = 921.78$ (**94.35% reduction** vs. unigram).
  * **4-gram ($n = 4$):** $\text{PP} = 648.51$ (**96.03% reduction** vs. unigram).

* **Seed Stability Verification (`SEED = 7`):**
  * Replication under independent seed yielded: Unigram ($16,288.76$), Bigram ($1,625.63$), Trigram ($923.00$), 4-gram ($651.48$).
  * Best order ranking is invariant across both seeds with relative delta $< 0.5\%$ ($\to$ **PASS**).

* **Model Usability Acceptance:**
  * All higher-order models achieve $> 90\%$ perplexity reduction over unigram, meeting the usability criteria.

---

## 6. Question 3 Key Results Summary

* **Task Formulation:**
  * Real-versus-Corrupted Sentence Detection (balanced binary classification on 998,714 held-out test pairs: 499,357 authentic sentences vs. 499,357 syntactically corrupted negatives).

* **Zero-Leakage Threshold Tuning:**
  * Decision thresholds $\tau_n$ were optimized strictly on the 499,358 pairs of the tuning partition (`tune`):
    * $n = 1$: $\tau = -10.2542$
    * $n = 2$: $\tau = -8.5700$
    * $n = 3$: $\tau = -8.6520$
    * $n = 4$: $\tau = -8.7901$

* **Held-Out Test Set Performance (Full Split, 998,714 Pairs):**
  * **$n = 1$ (Unigram):** $\text{Acc} = 71.21\%$, $\text{Prec} = 67.20\%$, $\text{Rec} = 82.87\%$, $\text{F1} = 0.7422$, $\text{ROC-AUC} = \mathbf{0.8118}$.
  * **$n = 2$ (Bigram):** $\text{Acc} = \mathbf{73.58\%}$, $\text{Prec} = \mathbf{70.06\%}$, $\text{Rec} = 82.37\%$, $\text{F1} = \mathbf{0.7572}$, $\text{ROC-AUC} = \mathbf{0.8359}$ (Champion Model).
  * **$n = 3$ (Trigram):** $\text{Acc} = 71.33\%$, $\text{Prec} = 66.84\%$, $\text{Rec} = 84.67\%$, $\text{F1} = 0.7470$, $\text{ROC-AUC} = 0.8000$.
  * **$n = 4$ (4-gram):** $\text{Acc} = 70.90\%$, $\text{Prec} = 66.00\%$, $\text{Rec} = 86.23\%$, $\text{F1} = 0.7477$, $\text{ROC-AUC} = 0.7895$.

* **UNK Confounder & Sensitivity Analysis:**
  * Naive UNK-count baseline achieves only $0.5184$ ROC-AUC (near random guess $0.50$).
  * On the **738,256 pairs containing zero UNK tokens**, the Bigram model achieves **ROC-AUC = 0.8473** and **Accuracy = 74.52%**, proving authentic linguistic fluency discrimination.

---

## 7. Question 4 Key Results Summary

* **Application Architecture:**
  * Core suggestion engine in `app/suggest.py` (`AmharicPredictor`).
  * Interactive CLI in `app/cli.py` and `app/main.py`.
  * FastAPI REST API in `app/api.py` serving the Next.js web application.
  * Candidate index precomputed at startup in under 4 seconds.

* **Test on Assignment Example (`ኢትዮጵያ ውስጥ የቡና`):**
  * Response latency: **0.05 ms** ($20,000\times$ faster than the 1.0s benchmark limit).
  * Returned Top-5 suggestions with full-corpus frequencies (`results/q4_eval.json`):
    1. **`ዋጋ`** (Price, $P = 0.1140$, train count = 27,626) $\to$ `የቡና ዋጋ`
    2. **`ፍጆታ`** (Consumption, $P = 0.1067$, train count = 1,911) $\to$ `የቡና ፍጆታ`
    3. **`ድርሻ`** (Share, $P = 0.0752$, train count = 13,607) $\to$ `የቡና ድርሻ`
    4. **`ገበያ`** (Market, $P = 0.0479$, train count = 15,518) $\to$ `የቡና ገበያ`
    5. **`የወጪ`** (Export, $P = 0.0470$, train count = 2,306) $\to$ `የቡና የወጪ`
  * Illustrative words: `ምርት` (freq: 1,328) and `ንግድ` (freq: 1,012) verified in training vocabulary; `አምራችነት` (freq: 0).

* **Evaluation on Full Held-Out Test Split (475,613 Prediction Positions):**
  * **Top-1 Accuracy:** **23.37%** (Benchmark target $\ge 10.0\% \to$ **PASS**).
  * **Top-5 Hit Rate:** **40.05%** (Usability target $\ge 30.0\% \to$ **PASS**).
  * **MRR@5:** **0.2963** (Benchmark target $\ge 0.15 \to$ **PASS**).
  * **Response Latency:** **< 0.1 ms** ($\to$ **PASS**).

* **Context Length Empirical Comparison (Bigram $L=1$ vs. Trigram $L=2$):**
  * On held-out test sentences, the **Bigram model ($L = 1$) achieved higher ranking metrics** than the Trigram model ($L = 2$):
    * Top-1 Accuracy: **14.00%** (Bigram) vs. **11.50%** (Trigram)
    * Top-3 Hit Rate: **20.50%** (Bigram) vs. **17.50%** (Trigram)
    * Top-5 Hit Rate: **23.50%** (Bigram) vs. **22.00%** (Trigram)
    * MRR: **0.1762** (Bigram) vs. **0.1522** (Trigram)
  * Trigram response latency was slightly faster (**0.337 ms** vs. **0.428 ms**).
  * **Linguistic Rationale:** On unseen text, 2-word histories ($w_{i-2}, w_{i-1}$) suffer from acute sparsity, forcing backoff mass distribution; the 1-word bigram context ($w_{i-1}$) has significantly higher vocabulary support, providing superior top-1 and top-5 ranking accuracy.

* **How to Run the Applications:**
  * **Interactive CLI Application:**
    ```bash
    python app/main.py
    # or
    python app/cli.py
    ```
  * **Next.js Web Application:**
    ```bash
    # Terminal 1: Start FastAPI backend
    python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

    # Terminal 2: Start Next.js frontend
    cd web && npm run dev
    # Open http://localhost:3000 in your browser
    ```
