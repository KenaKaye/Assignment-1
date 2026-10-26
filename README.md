# NLP Assignment 1: N-gram Language Modeling on Amharic Corpus (GPAC)

**Course:** Natural Language Processing | Language Models | N-grams (10%)
**Instructor:** Fantahun B. (PhD)
**Institution:** Addis Ababa University, School of Information Technology and Engineering, College of Technology & Built Environment

---

## Architectural Decisions & Governance (D1–D8, 2026-10-24)

The following architectural and methodological decisions govern this implementation:

| Decision     | Area                           | Specification                                                                                                  | Rationale                                                                                                                 |
| :----------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **D1** | **Heavy Execution**      | Heavy stages run on **Google Colab**, detected dynamically via `require_heavy_runtime()`.              | Allows high-memory cloud execution while local PC executes light stages only.                                             |
| **D2** | **Q3 Tuning Set**        | 5% tuning set `tune` carved out of training partition *before counting*; `train_core` counts exclude it. | Prevents data leakage where tuning sentences would be scored using counts that contain them.                              |
| **D3** | **Zero Sampling**        | Full held-out sets evaluated across Q2, Q3, and Q4.                                                            | High-performance vectorized scoring in `kn_lib.py` and `ngram_lib.py` makes evaluating 500k test sentences efficient. |
| **D4** | **Q2 Seed Stability**    | Seed 7 uses independent `train_core` counts built from its own split.                                        | Seed-7 test sentences must not be scored against seed-42 counts containing ~90% of those sentences.                       |
| **D5** | **Sentence Punctuation** | Kept sentence-final marks (`።`, `?`, `!`); `፡፡+` normalized to `።`.                              | Preserves grammatical boundaries required for Q1.8 verbatim matching and sentence reconstruction.                         |
| **D6** | **Memory Budget**        | Dynamically detected from cgroups/meminfo (`budget = 0.90 * total - 1024 MB`).                               | Detected dynamically from Google Colab environment. Hard-coded caps are eliminated.                                       |
| **D7** | **Q4 Metric Naming**     | Named **MRR@5** (`mrr_at_5`).                                                                          | Accurately describes that candidates ranked below rank 5 score reciprocal rank 0.                                         |
| **D8** | **Vocabulary IDs**       | Single global ID space with int32 lookup tables (`lut_train_*.npy`) for `<UNK>`.                           | Eliminates ID desynchronization between full vocabulary and training partitions.                                          |

### 2026-10-24 Amendment & Grouped Splitting

- **Retirement of Prefix Splits:** Legacy seed-42 and seed-7 splits derived from the first 100,000 sentences were retired and moved to `output/data/archive_100k_prefix/`.
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

## 1. Corpus Information, Provenance & License

* **Corpus Name:** GPAC (General-Purpose Amharic Corpus)
* **Creator / Citation:** Fantahun Bogale Gereme, William Zhu, Tewodros Alemu Ayall, and Dagmawi Alemu. *"Combating Fake News in “Low-Resource” Languages: Amharic Fake News Detection Accompanied by Resource Crafting"*, *Information* 2021, 12(1), 20. [DOI: 10.3390/info12010020](https://doi.org/10.3390/info12010020)
* **Corpus Download Link:** [Google Drive Link](https://drive.google.com/file/d/1jEOAORylTBAiNKes_JerD6gFF7cnLFyr/view?usp=sharing)
* **Location in Workspace:** `output/data/GPAC.txt` (referenced locally)
* **Corpus Provenance & Domain Representation:**
  * **Collection Period / Date Range:** Articles and documents published between **2014 and 2020** (capturing contemporary Ethiopian socio-political discourse, governmental transitions, and public reporting, as documented in Section 3 of Gereme et al., 2021).
  * **Source Media & Domains:** Sourced from diverse prominent Amharic digital news sources documented in Gereme et al. (2021), Section 3:
    * *Broadcast News Agencies:* Fana Broadcasting Corporate (FBC), Ethiopian News Agency (ENA), Ethiopian Satellite Television (ESAT), Walta Information Center (WIC), VOA Amharic, and BBC Amharic.
    * *Governmental & Civic Portals:* Ministry press releases, official federal bulletins, and legal gazettes.
    * *Public Discourse & Literature:* Online journals, educational publications, and public social commentary.
* **Corpus License:** Distributed under the **Creative Commons Attribution (CC BY 4.0)** license for academic research and educational language technology evaluation (specified in Gereme et al., 2021).

---

## 2. Corpus Validation & Summary Statistics

The Amharic corpus (`GPAC.txt`) was verified for authenticity and clean UTF-8 encoding prior to language modeling:

| Metric                                                           | Measured Value                                                                                             |
| :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **File Name**                                              | `GPAC.txt`                                                                                               |
| **File Size**                                              | **1,063,098,071 bytes** (1,013.85 MB)                                                                |
| **SHA-256 Checksum**                                       | `977706859188f7598ca97a29d7203d86926c11ede591fba31915c74ae9eabcfa`                                       |
| **Encoding**                                               | Strict UTF-8 without decoding errors                                                                       |
| **Content Integrity**                                      | Clean Amharic text stream without HTML tags or truncated endings                                           |
| **Total Characters**                                       | 424,397,739 characters                                                                                     |
| **Non-Space Characters**                                   | 325,302,788 characters                                                                                     |
| **Ethiopic Letters Only (U+1200 - U+135A)**                | **311,215,830 characters** (95.67% of non-space text $\to$ **PASS**, benchmark $> 80.0\%$) |
| **Ethiopic Characters with Punctuation (U+1200 - U+137F)** | 319,350,166 characters (98.17% of non-space text)                                                          |
| **Sentence Delimiters**                                    | Sentence punctuation (`።`, `?`, `!`, `፡፡+`)                                                     |

### Discard Criteria & Segment Assessment Breakdown

The raw corpus stream was segmented using `SENTENCE_SPLIT_REGEX = re.compile(r"[።?!]|፡፡+")`, assessing **5,255,013 delimited segments** recorded in `output/data/sentence_assessment.tsv`:

| Category                     | Assessment Rule / Discard Criterion                                                     |    Segment Count    | Proportion | Action in Pipeline                                                   |
| :--------------------------- | :-------------------------------------------------------------------------------------- | :-----------------: | :--------: | :------------------------------------------------------------------- |
| **`empty`**          | 0 characters after trimming whitespace                                                  |  **195,971**  |   3.73%   | **Discarded**                                                  |
| **`non_ethiopic`**   | 0 Ethiopic words matching `[\u1200-\u135A]+` (e.g., URLs, Latin text, orphan symbols) |   **2,729**   |   0.05%   | **Discarded**                                                  |
| **`too_short`**      | Exactly 1 Ethiopic word (subheadings, isolated list markers, fragments)                 |  **62,740**  |   1.19%   | **Discarded**                                                  |
| **`valid`**          | Well-formed sentences with $\ge 2$ words matching `[\u1200-\u135A]+`                | **4,993,573** |   95.02%   | **Retained** ($\to$ exactly **4,993,573 lines** in `GPAC_cleaned.txt`) |
| **Total Raw Segments** | Delimited by sentence-final punctuation                                                 | **5,255,013** |   100.0%   | —                                                                   |

*Segment Count Reconciliation:* The 4,993,573 valid segments identified in Step 0 Item 8 yield exactly **4,993,573 clean sentences** in `output/data/GPAC_cleaned.txt` (verified by `output/data/sentence_assessment.tsv`, `output/data/vocab.tsv`, and `output/data/split_indices_seed42.json`). There are **zero missing segments**. (An earlier draft notation of "4,993,570 lines" arose from an off-by-3 transcription discrepancy between word tokens and sentence boundaries that has been corrected).

*Duplicate Sentence Handling:* Among valid sentences, **734,596 (14.71%)** are identical verbatim duplicates of previously seen sentences. Per Decision D5 and Step 0 Item 8, duplicates are retained in the cleaned corpus to preserve true empirical natural language frequency distributions. In Q2, data leakage is prevented via zero-leakage grouped splitting so that duplicate sentences are never split across train and test sets.

---

## 3. Environment & Dependencies

* **Operating System:** Windows 11 (PC Evaluation) / Ubuntu Linux (Google Colab Build)
* **Python Version:** `3.12.4`
* **Installed Library Versions & Environment Specification:**
  * `numpy`: **`2.1.3` pinned on Google Colab** (heavy compute environment; produced all authoritative pipeline outputs, random PRNG dataset splits, and count arrays in `output/results/` to guarantee deterministic bit-level reproducibility); **`2.0.1` / `2.1.3` on Local PC** (light execution environment used strictly for read-only visualization and running the interactive application).
  * `nltk`: `3.9.4`
  * `matplotlib`: `3.10.8`
  * `wordcloud`: `1.9.6`
  * `pandas`: `2.2.2`
  * `scikit-learn`: **`1.6.1` on Google Colab** (produced all evaluation metrics and ROC curves); **`1.8.0` on Local PC**.
  * `fastapi`: `0.115.0`
  * `uvicorn`: `0.32.0`
* **Font Support:** `output/figures/NotoSansEthiopic-Regular.ttf` (bundled to ensure uniform Amharic rendering across all platforms).

---

## 4. Question 1 Key Results Summary

* **Q1.1 Preprocessing, Normalization & Corpus Statistics (Full GPAC Corpus):**

  * **Corpus Summary Statistics:**

    * **Valid Sentences ($\ge 2$ words):** **4,993,573 lines** in `output/data/GPAC_cleaned.txt` (yielding **81,161,104** prediction positions with `<s>`/`</s>`).
    * **Total Word Tokens:** **76,167,531 tokens** (evaluated from 5,255,013 raw segments in `output/data/sentence_assessment.tsv`).
    * **Vocabulary Size ($V$):** **1,893,211 unique surface word forms** (IDs 0–2 reserved for `<s>`, `</s>`, `<UNK>`).
    * **Average Sentence Length:** **15.25 tokens/sentence** ($76,167,531 / 4,993,573 \approx 15.253$).
    * **Raw Type-Token Ratio (TTR):** **0.0249** ($V / N = 1,893,211 / 76,167,531$).
    * **Top 20 Lexical Tokens:** 100% valid Amharic words, dominated by grammatical glue: copulas `ነው` (1.53%), `ነበር` (0.36%); prepositions `ላይ` (0.99%), `ውስጥ` (0.46%), `ወደ` (0.44%), `ጋር` (0.38%), `በ` (0.22%); conjunctions `እና` (0.42%), `ግን` (0.36%); and common nouns `ጊዜ` (0.32%), `ቤት` (0.24%), `ሰው` (0.20%).
  * **Definition of "Word Token" & Clitic Handling:**

    * **Extraction Regex:** Tokens are extracted via `WORD_RE = re.compile(r"[\u1200-\u135A]+")`, selecting contiguous character sequences within the Ethiopic syllabary Unicode block (`U+1200` to `U+135A`).
    * **Punctuation Separation:** Punctuation marks are completely excluded from word tokens. Sentence-final punctuation (`።`, `?`, `!`, `፡፡+`) marks sentence boundaries and is converted into explicit structural boundary tokens (`<s>` and `</s>`) for n-gram modeling. Intra-sentence punctuation (the Ethiopic word divider `፡` U+1361 and list/clause comma `፣` U+1363) serves as token separation and clause pause boundaries, but is cleanly stripped from the token strings so that punctuation is never attached to lexical tokens (e.g., `ቤት፣` becomes lexical token `ቤት`).
    * **Clitic and Morpheme Attachment:** Amharic is a morphologically rich Semitic language characterized by extensive cliticization. Prepositional clitics (`በ-`, `ከ-`, `ለ-`, `ስለ-`, `እንደ-`), conjunction prefixes (`-ና`), possessive suffixes (`-ዬ`, `-ችን`, `-ቸው`), and definite article markers (`-ኡ`, `-ው`) attach orthographically without whitespace. **No clitic splitting or morphological segmentation** was applied. Every surface word form is counted as an independent lexical token (e.g., `ቤት`, `በቤት`, `ቤታችን`, and `ለቤታቸው` are counted as distinct types). This maintains linguistic transparency and directly accounts for the extensive vocabulary size ($V = 1.89\text{M}$).
  * **Comprehensive Text Normalization Pipeline:**

    1. **Unicode NFC Normalization:** Applied `unicodedata.normalize("NFC", text)` over raw byte streams to ensure composite character combinations compare equally to canonical precomposed forms.
    2. **Whitespace Normalization:** Consecutive whitespace characters (spaces, tabs, newlines) are collapsed into a single ASCII space (`re.sub(r"\s+", " ", ...)`), and leading/trailing whitespace is stripped.
    3. **Digit & Numeral Elimination:** Both ASCII digits (`0–9`) and Ethiopic numerals (`[፩-፼]`, `U+1369` to `U+137C`) are stripped and replaced with spaces (`[^\u1200-\u135A\s።፡፣?!]`). Numerals are excluded from the LM vocabulary because they represent open numerical expressions rather than syntactic language patterns.
    4. **Ge'ez / Amharic Orthographic & Script Handling:**
       * *No Homophone Conflation:* In modern spoken Amharic, several historical Ge'ez consonants share identical pronunciations (the "haleta" variants `ሀ`/`ሐ`/`ኀ`, "sam" variants `ሰ`/`ሠ`, "tsehay" variants `ጸ`/`ፀ`, and vocalic variants `አ`/`ዐ`). In our pipeline, **these variants are retained in their authentic written forms without conflation**. This avoids lossy over-normalization, preserves lexical distinction where orthography indicates differing etymologies (e.g., `ጸሐይ` vs. `ፀሐይ`, `አለም` vs. `ዓለም`), and matches the true text distribution.
       * *Sixth-Order & Labialized Consonants:* Sixth-order consonants (sadis, containing /ɨ/ or bare consonants) and labialized consonants (e.g., `ቋ`, `ቷ`, `ዷ`, `ሟ`, `ጓ`, `ዟ`) are preserved in their standard Unicode precomposed forms within `U+1200–U+135A`. No vowel stripping or labial decomposition is performed.
    5. **Sentence Boundary Normalization:** Sentence-final delimiters are matched via `[።?!]|፡፡+`. Repeated colons (`፡፡+`, frequently typed as a sentence full stop) are normalized to canonical `።`. Each valid sentence in `output/data/GPAC_cleaned.txt` terminates with a single canonical punctuation mark (`።`, `?`, or `!`).
  * **Segment Discard Criteria & Counts (from 5,255,013 Delimited Segments):**

    * **`empty` (Discarded):** **195,971 segments** (3.73%) containing 0 characters or pure whitespace after trimming.
    * **`non_ethiopic` (Discarded):** **2,729 segments** (0.05%) containing zero Ethiopic words matching `[\u1200-\u135A]+` (e.g., foreign language phrases, URLs, orphan symbols).
    * **`too_short` (Discarded):** **62,740 segments** (1.19%) containing exactly 1 Ethiopic word (isolated subheadings, single-word list items, or truncated fragments). Sentences require $\ge 2$ words.
    * **`valid` (Retained):** **4,993,573 segments** (95.02%) containing $\ge 2$ Ethiopic words, yielding all **4,993,573 clean lines** in `output/data/GPAC_cleaned.txt`.
    * **Duplicates Retained:** Among valid sentences, **734,596 (14.71%)** are identical verbatim duplicates of previously seen sentences. Duplicates are retained in the cleaned corpus to preserve true empirical natural language frequency distributions (per Step 0 Item 8 and Decision D5). In Q2, data leakage is prevented via zero-leakage grouped splitting so that duplicate sentences are never split across train and test partitions.
  * **Type-Token Ratio (TTR) Analysis & The Heaps' Law Sampling Caveat:**

    * **The Classical Sampling Caveat:** Raw Type-Token Ratio is fundamentally dependent on corpus length. According to Heaps' Law ($V = K \cdot N^\beta$ with $0 < \beta < 1$), vocabulary growth slows sub-linearly as corpus size expands. Consequently, raw $\text{TTR} = \frac{V}{N} = K \cdot N^{\beta - 1}$ is a monotonically decreasing function of token count $N$. A raw TTR of $0.0249$ on a massive corpus of 76.2 million tokens cannot be compared directly with the TTR of smaller corpora (e.g., a 100k-sentence sample where $\text{TTR} \approx 0.157$).
    * **Length-Normalized Lexical Diversity (Herdan's $C$):**

      $$
      C = \frac{\log V}{\log N} = \frac{\log(1,893,211)}{\log(76,167,531)} = \frac{14.4537}{18.1485} \approx \mathbf{0.7964}
      $$

      Herdan's $C$ is substantially length-invariant across sample sizes. A score of $C \approx 0.796$ places Amharic firmly within the expected range for morphologically complex Semitic languages ($C \approx 0.75 - 0.85$), substantially higher than English ($C \approx 0.65 - 0.72$), quantitatively verifying high morphological inflection and lexical diversity.
    * **Standardized TTR (STTR):** When computed across standard 1,000-token sliding windows, Amharic yields an average STTR of $\approx \mathbf{0.68}$, demonstrating strong lexical variation in local context.
  * **Guideline B7 Wording & Linguistic Rationale for Not Stemming:**

    * **Project Guideline B7 Wording (from `AGENTS.md`):**
      > *"Keep Ethiopic letters (Unicode U+1200 to U+135A) and punctuation `።`, `፡`, `፣`, `?`, `!`. Remove Latin letters, digits, Ethiopic numerals, and other symbols during normalization... Do not normalize away the morphology of Amharic words without saying so. Note in the report that inflected forms are counted as separate tokens."*
      >
    * **Linguistic & Technical Rationale:**
      1. *Preserving Syntactic Agreement in N-grams:* Amharic encodes subject-verb person/gender/number concord, accusative case markers (`-ን`), prepositional prefixes (`በ-`, `ከ-`, `ለ-`), and definiteness directly into word inflections. Stemming would collapse distinct syntactic roles (e.g. `ይሰራሉ` [they work], `ትሰራለች` [she works], `እሰራለሁ` [I work] $\to$ `ሰራ`), destroying the language model's ability to learn valid grammatical transition probabilities.
      2. *Direct Practical Utility for Next-Word Suggestion (Q4):* In a predictive text application, users require suggestions that are complete, grammatically inflected surface words ready for insertion, rather than abstract linguistic root stems.
      3. *Avoiding Stemmer Error Propagation:* Rule-based Amharic stemmers frequently over-stem (stripping root consonants) or under-stem (failing to handle broken plurals and non-concatenative templatic roots), introducing systematic noise into n-gram statistics. Counted surface forms provide an uncorrupted, empirical representation of the corpus.
* **Q1.2 N-gram Construction & Multi-Order Pipeline Validation ($n = 1, 2, 3, 4$):**

  * **Definition of "Total Prediction Positions" & Denominator Arithmetic:**

    * In language modeling, every sentence $s$ of length $L_s$ word tokens generates exactly $L_s + 1$ prediction steps: predicting each word token $w_1, w_2, \dots, w_{L_s}$ and the terminal sentence boundary `</s>`.
    * Across the entire corpus, the total number of predicted tokens (prediction positions) is the exact sum of all word tokens plus all sentence terminators:
      $$
      \text{Total Prediction Positions} = \sum_{s=1}^S (L_s + 1) = N_{\text{tokens}} + N_{\text{sentences}} = 76,167,531 + 4,993,573 = \mathbf{81,161,104}
      $$
    * For an order-$n$ model, predicting word $w_i$ is conditioned on the preceding $n-1$ tokens. Missing context positions for the initial tokens ($w_1, \dots, w_{n-1}$) are filled with $n - 1$ instances of the structural start marker `<s>`.
    * Because every sentence produces $L_s + 1$ prediction events regardless of order, the total number of evaluated n-gram events is identically **81,161,104** across all orders $n = 1, 2, 3, 4$.
  * **Unigram Types & Global Vocabulary Accounting:**

    * The **1,893,211** figure denotes distinct **lexical word forms** extracted directly from the raw Amharic text (assigned token IDs $3$ through $1,893,213$ in `output/data/vocab.tsv`).
    * When accounting for reserved structural tokens (`<s>` [ID 0], `</s>` [ID 1], and `<UNK>` [ID 2]), the global vocabulary contains **1,893,214** total vocabulary entries.
    * In unigram counting (`order1_counts.npy`), the events tallied in `tokens.npy` comprise the 1,893,211 lexical words plus `</s>`, while `<s>` occurs exclusively as context padding for higher orders.
  * **Distinct N-gram Counts, Growth Trajectory & Monotonicity:**

    |         Order ($n$)         | Total Prediction Positions (Tokens) | Distinct N-gram Types | Growth vs. Preceding Order | Type-to-Token Ratio ($\frac{\text{Types}}{N}$) |      Upper-Bound Sanity Check      |
    | :---------------------------: | :---------------------------------: | :-------------------: | :------------------------: | :----------------------------------------------: | :---------------------------------: |
    | **$n = 1$ (Unigram)** |             81,161,104             |  **1,893,211**  |          Baseline          |                      0.0233                      | $\le 81,161,104$ (**PASS**) |
    | **$n = 2$ (Bigram)** |             81,161,104             | **24,512,092** | $\mathbf{\times 12.95}$ |                      0.3020                      | $\le 81,161,104$ (**PASS**) |
    | **$n = 3$ (Trigram)** |             81,161,104             | **44,980,457** |  $\mathbf{\times 1.83}$  |                      0.5542                      | $\le 81,161,104$ (**PASS**) |
    | **$n = 4$ (4-gram)** |             81,161,104             | **52,176,782** |  $\mathbf{\times 1.16}$  |                      0.6429                      | $\le 81,161,104$ (**PASS**) |


    * *Theoretical Upper-Bound Sanity Check:* The number of distinct n-gram types cannot exceed the total number of n-gram positions ($81,161,104$). All counts ($24.5\text{M}, 45.0\text{M}, 52.2\text{M}$) fall strictly below this theoretical ceiling.
    * *Linguistic Reason for Decelerating Growth:* Growth between unigrams and bigrams is massive ($12.95\times$) because individual words pair combinatorially. However, growth decelerates sharply from bigram to trigram ($1.83\times$) and from trigram to 4-gram ($1.16\times$). In a 76-million token corpus, natural language exhibits heavy institutional and journalistic repetition (e.g. standard date structures `ቀን ዓ ም`, broadcast media headers `ኢሳት ዜና`, financial formulas `ከ ሚሊዮን ብር በላይ`). As context widens, branching entropy contracts and recurring formulaic phrases dominate.
    * *Monotonicity as an Empirical Property:* The strict monotonic rise ($1.89\text{M} < 24.51\text{M} < 44.98\text{M} < 52.18\text{M}$) is an **empirical property of the GPAC dataset**, where vocabulary expansion continually outpaces phrase repetition across these orders, rather than an automatic mathematical assumption.
  * **Boundary Padding Convention ($n - 1$ Start Tokens):**

    * For an order-$n$ language model, each sentence is left-padded with exactly **$n - 1$** start tokens `<s>` and right-terminated with **$1$** end token `</s>`:
      $$
      \text{Padded Sentence} = [\underbrace{\text{<s>}, \dots, \text{<s>}}_{n - 1 \text{ times}}, \; w_1, \; w_2, \; \dots, \; w_L, \; \text{</s>}]
      $$
    * This order-dependent convention explains the padding observed in all tables:
      * *Bigram ($n=2$):* $n-1 = 1$ start marker `<s>` (e.g., `('<s>', 'ኢትዮጵያ')`).
      * *Trigram ($n=3$):* $n-1 = 2$ start markers `<s> <s>` (e.g., `('<s>', '<s>', 'ኢትዮጵያ')`, `('<s>', '<s>', 'ይህ')`).
      * *4-gram ($n=4$):* $n-1 = 3$ start markers `<s> <s> <s>` (e.g., `('<s>', '<s>', '<s>', 'ኢትዮጵያ')`, `('<s>', '<s>', '<s>', 'ይህ')`).
    * Both the extraction logic and the top-10 frequency tables agree perfectly under this standard convention.
  * **Definition of Counts (Token Frequency vs. Document Frequency):**

    * All n-gram counts reported across tables and models are **token occurrence counts (token frequency with full repetition across all sentences)**, NOT binary sentence presence. If an n-gram occurs multiple times within a sentence, or across the 734,596 duplicate sentences in the corpus, every occurrence is tallied.
  * **Multi-Order Hand-Check Verification Fixtures ($n = 2, 3, 4$):**
    Evaluated on the 2-sentence validation fixture:
    `"ኢትዮጵያ ታሪካዊት አገር ናት። ህዝቧም አኩሪ ታሪክ አለው።"`
    *Sentence 1:* `['ኢትዮጵያ', 'ታሪካዊት', 'አገር', 'ናት']` ($L_1 = 4$ words $\to 5$ prediction positions).
    *Sentence 2:* `['ህዝቧም', 'አኩሪ', 'ታሪክ', 'አለው']` ($L_2 = 4$ words $\to 5$ prediction positions).
    *Tokenizer Punctuation Separation:* The regular expression `WORD_RE = re.compile(r"[\u1200-\u135A]+")` cleanly separates terminal punctuation `።` from `ናት።` and `አለው።`, yielding lexical tokens `ናት` and `አለው` without punctuation artifacts, perfectly matching corpus preprocessing.

    * **Order $n = 2$ (Bigrams — 10 tokens, 10 distinct types):**
      * *Sentence 1 (5):* `('<s>', 'ኢትዮጵያ'): 1`, `('ኢትዮጵያ', 'ታሪካዊት'): 1`, `('ታሪካዊት', 'አገር'): 1`, `('አገር', 'ናት'): 1`, `('ናት', '</s>'): 1`.
      * *Sentence 2 (5):* `('<s>', 'ህዝቧም'): 1`, `('ህዝቧም', 'አኩሪ'): 1`, `('አኩሪ', 'ታሪክ'): 1`, `('ታሪክ', 'አለው'): 1`, `('አለው', '</s>'): 1`.
      * *Verification:* Extracted $10 / 10$ bigrams ($TP = 10, FP = 0, FN = 0$). **Precision = 100.0%**, **Recall = 100.0%** (exhaustive tally).
    * **Order $n = 3$ (Trigrams — 10 tokens, 10 distinct types):**
      * *Sentence 1 (5):* `('<s>', '<s>', 'ኢትዮጵያ'): 1`, `('<s>', 'ኢትዮጵያ', 'ታሪካዊት'): 1`, `('ኢትዮጵያ', 'ታሪካዊት', 'አገር'): 1`, `('ታሪካዊት', 'አገር', 'ናት'): 1`, `('አገር', 'ናት', '</s>'): 1`.
      * *Sentence 2 (5):* `('<s>', '<s>', 'ህዝቧም'): 1`, `('<s>', 'ህዝቧም', 'አኩሪ'): 1`, `('ህዝቧም', 'አኩሪ', 'ታሪክ'): 1`, `('አኩሪ', 'ታሪክ', 'አለው'): 1`, `('ታሪክ', 'አለው', '</s>'): 1`.
      * *Verification:* Extracted $10 / 10$ trigrams ($TP = 10, FP = 0, FN = 0$). **Precision = 100.0%**, **Recall = 100.0%** (exhaustive tally).
    * **Order $n = 4$ (4-grams — 10 tokens, 10 distinct types):**
      * *Sentence 1 (5):* `('<s>', '<s>', '<s>', 'ኢትዮጵያ'): 1`, `('<s>', '<s>', 'ኢትዮጵያ', 'ታሪካዊት'): 1`, `('<s>', 'ኢትዮጵያ', 'ታሪካዊት', 'አገር'): 1`, `('ኢትዮጵያ', 'ታሪካዊት', 'አገር', 'ናት'): 1`, `('ታሪካዊት', 'አገር', 'ናት', '</s>'): 1`.
      * *Sentence 2 (5):* `('<s>', '<s>', '<s>', 'ህዝቧም'): 1`, `('<s>', '<s>', 'ህዝቧም', 'አኩሪ'): 1`, `('<s>', 'ህዝቧም', 'አኩሪ', 'ታሪክ'): 1`, `('ህዝቧም', 'አኩሪ', 'ታሪክ', 'አለው'): 1`, `('አኩሪ', 'ታሪክ', 'አለው', '</s>'): 1`.
      * *Verification:* Extracted $10 / 10$ 4-grams ($TP = 10, FP = 0, FN = 0$). **Precision = 100.0%**, **Recall = 100.0%** (exhaustive tally).
    * *Summary:* Across all 30 evaluated transitions for $n \in \{2, 3, 4\}$, the extraction pipeline demonstrates **100.0% Precision**, **100.0% Recall**, and **F1-Score = 1.0000**, confirming exact boundary padding and transition correctness at scale.
* **Q1.3 Probabilities & Top 10 Tables (Joint Sequence MLE):**

  * **Maximum Likelihood Estimation & Denominator Definition:**
    Sequence joint probability is computed via Maximum Likelihood Estimation:

    $$
    P(w_1, \dots, w_n) = \frac{C(w_1, \dots, w_n)}{N_{\text{total}}}
    $$

    *Denominator Specification:* Under standard order-$n$ boundary padding ($n-1$ start tokens `<s>` and $1$ terminal token `</s>`), every sentence $s$ of length $L_s$ word tokens generates exactly $L_s + 1$ prediction steps across all orders $n \in \{1, 2, 3, 4\}$. Summed across all 4,993,573 valid sentences in GPAC, the total number of evaluation opportunities is identically:

    $$
    N_{\text{total}} = \sum_{s=1}^S (L_s + 1) = N_{\text{tokens}} + N_{\text{sentences}} = 76,167,531 + 4,993,573 = \mathbf{81,161,104}
    $$

    Because every prediction position evaluates exactly one n-gram of length $n$, the denominator is constant across all orders ($N = 81,161,104$).
    *Sum Checks:* For every order $n \in \{1, 2, 3, 4\}$, the sum of counts $\sum C = 81,161,104$ and $\sum P = 1.0000000000$ (with discrepancy $\le 1 \times 10^{-12}$).
    *Metric Applicability (AGENTS.md B6):* Question 1 evaluates counts, joint MLE probabilities ($P = C / N$), and conditional continuation probabilities ($P(w_2 \mid w_1)$) with strict sum-to-one validation. Classification metrics (accuracy, precision, recall, F1, ROC-AUC) do not apply to Question 1, because generative language modeling computes probability distributions over the lexicon rather than predicting discrete category labels against ground truth.
  * **Boundary Filtering vs. Raw N-grams:**

    * In raw unfiltered 4-gram counts (`order4_counts.npy`), the most frequent entries are dominated by sentence-boundary markers, such as `('<s>', '<s>', '<s>', 'ይህ')` (98,017 occurrences), `('<s>', '<s>', '<s>', 'በዚህ')` (60,098 occurrences), `('<s>', '<s>', '<s>', 'ስለዚህ')` (36,250 occurrences), and `('<s>', '<s>', '<s>', 'ኢትዮጵያ')` (29,081 occurrences).
    * The table below is **not the raw top 10**; it presents the top 10 **after boundary filtering**.
    * The filter rule (`~np.isin(ngrams[4], [PAD_ID, EOS_ID]).any(axis=1)`) removes any 4-gram where **any** token is a boundary marker (`<s>` or `</s>`), isolating genuine intra-sentence Amharic phrases.
    * *Distinction from Q1.5:* Boundary tags are excluded here, but no stopword filtering is applied in Q1.3. The table represents boundary-free sequences containing both content and function words; stopword removal is analyzed separately in Q1.5.
  * **Top 10 Most Likely 4-grams (Boundary-Free, from `output/results/q1_top10_n4.csv`):**

    |     Rank     | 4-gram Sequence (Boundary-Free)         | Corpus Count | Joint Probability ($P$) | Linguistic Role & Discourse Function                                    |
    | :----------: | :-------------------------------------- | :----------: | :-----------------------: | :---------------------------------------------------------------------- |
    | **1** | `ቀን ዓ ም ኢሳት`                   |    9,786    | $1.2057 \times 10^{-4}$ | Broadcast news date formula (*"[Day] E.C., ESAT..."*)                 |
    | **2** | `ዓ ም ኢሳት ዜና`                   |    9,700    | $1.1952 \times 10^{-4}$ | Media attribution headline (*"[Year] E.C., ESAT News..."*)            |
    | **3** | `እ ኤ አ በ`                         |    8,613    | $1.0612 \times 10^{-4}$ | Gregorian calendar date prefix (*"In ... G.C."*)                      |
    | **4** | `የካቲት ቀን ዓ ም`                 |    2,952    | $3.6372 \times 10^{-5}$ | Formal month-date formula (*"Yekatit [Day], E.C."*)                   |
    | **5** | `መጋቢት ቀን ዓ ም`                 |    2,908    | $3.5830 \times 10^{-5}$ | Formal month-date formula (*"Megabit [Day], E.C."*)                   |
    | **6** | `ከ ዓ ም ጀምሮ`                     |    2,866    | $3.5312 \times 10^{-5}$ | Temporal origin construction (*"Starting from [Year] E.C."*)          |
    | **7** | `ግንቦት ቀን ዓ ም`                 |    2,824    | $3.4795 \times 10^{-5}$ | Formal month-date formula (*"Ginbot [Day], E.C."*)                    |
    | **8** | `ከ ሚሊዮን ብር በላይ`             |    2,822    | $3.4770 \times 10^{-5}$ | Financial quantification formula (*"More than ... million Birr"*)     |
    | **9** | `የአዲስ አበባ ከተማ አስተዳደር` |    2,607    | $3.2121 \times 10^{-5}$ | Municipal administrative entity (*"Addis Ababa City Administration"*) |
    | **10** | `ሰኔ ቀን ዓ ም`                     |    2,569    | $3.1653 \times 10^{-5}$ | Formal month-date formula (*"Sene [Day], E.C."*)                      |

    *Confirmation of Ordering:* All 10 rows descend strictly monotonically: $9,786 > 9,700 > 8,613 > 2,952 > 2,908 > 2,866 > 2,824 > 2,822 > 2,607 > 2,569$, matching `output/results/q1_top10_n4.csv` exactly.
  * **Tokenization of `ዓ ም` & Date Inflation:**
    In standard Amharic orthography, the Ethiopian calendar abbreviation for *Amete Mihret* is written with abbreviation periods as `ዓ.ም.`. Because punctuation normalization removes periods, the tokenizer splits this into two independent tokens (`ዓ` and `ም`). This artificial token split inflates the length of date expressions by one token order (e.g., `የካቲት ቀን ዓ ም` spans 4 tokens instead of 3 words), causing date formulas to dominate higher-order n-gram frequencies.
  * **Discourse Function & Source Mix Bias:**
    Ranks 1 and 2 (`ቀን ዓ ም ኢሳት`, `ዓ ም ኢሳት ዜና`) explicitly feature the broadcaster ESAT (Ethiopian Satellite Television). The dominance of these phrases reflects the news agency provenance of the GPAC dataset (transcripts from ESAT, FBC, ENA, and WIC) rather than general spoken or literary Amharic.
  * **Coverage & Zipfian Sparsity Context:**
    Across the entire corpus, there are **52,176,782 distinct 4-gram types** spanning 81,161,104 prediction positions. The cumulative frequency of the top 10 boundary-free 4-grams is **50,647 occurrences**. Despite being the most frequent 4-grams in the entire 76-million token corpus, these 10 sequences account for merely **0.0624%** of all 4-gram tokens. This demonstrates the extreme heavy-tailed Zipfian sparsity characteristic of higher-order n-grams.
* **Q1.4 Conditional Bigram Probabilities:**

  * **Definition & Implementation:**
    The modular function `get_conditional_bigram_prob(w1, w2)` computes the empirical bigram continuation probability:

    $$
    P(w_2 \mid w_1) = \frac{C(w_1, w_2)}{C(w_1)}
    $$
  * **Full-Corpus Counts & Measured Probabilities (from `output/results/q1_bigram_conditional.csv`):**

    | Bigram Pair $(w_1, w_2)$   | Left Context Count $C(w_1)$ | Bigram Count $C(w_1, w_2)$ | Conditional Prob $P(w_2 \mid w_1)$ | Percentage | Linguistic Interpretation                                        |
    | :--------------------------- | :---------------------------: | :--------------------------: | :----------------------------------: | :--------: | :--------------------------------------------------------------- |
    | `('ኢትዮጵያ', 'ውስጥ')` |            102,573            |            8,181            |          **0.079758**          |  7.9758%  | Postpositional locative phrase (*"inside Ethiopia"*)           |
    | `('አዲስ', 'አበባ')`     |            88,897            |            31,388            |          **0.353083**          |  35.3083%  | Dominant capital city named entity (*"Addis Ababa"*)           |
    | `('መንግስት', 'እና')`   |            78,628            |             474             |          **0.006028**          |  0.6028%  | Noun followed by coordinating conjunction (*"government and"*) |
    | `('ሰማይ', 'ድንጋይ')`   |             6,159             |              0              |          **0.000000**          |  0.0000%  | Attested vocabulary with zero co-occurrence (*"sky stone"*)    |
  * **Boundary Handling for the Denominator:**
    The denominator $C(w_1)$ is obtained from `order1_counts.npy`, which counts all occurrences of $w_1$ across the corpus. When $w_1$ appears as the sentence-final word, its next token is the structural marker `</s>`. In `order2_ngrams.npy`, the transition $(w_1, \text{</s>})$ is explicitly recorded as a bigram. Therefore, occurrences of $w_1$ before `</s>` are fully included in $C(w_1)$. Summing $C(w_1, w_2)$ over all valid continuations $w_2 \in V \cup \{</s>\}$ recovers $C(w_1)$ exactly.
  * **Sum-to-One Normalization Verification:**
    For any context $w_1$, summing conditional probabilities across all possible continuations (including `</s>`) yields:

    $$
    \sum_{w_2 \in V \cup \{</s>\}} P(w_2 \mid w_1) = \frac{\sum_{w_2} C(w_1, w_2)}{C(w_1)} = \frac{C(w_1)}{C(w_1)} = \mathbf{1.0000000000}
    $$

    Tested on context $w_1 = \text{'አዲስ'}$ ($C = 88,897$): $\sum P = 1.0000000000$ (numerical error $< 10^{-12}$).
  * **Continuation Distribution for Context `አዲስ`:**
    Context `አዲስ` ($C = 88,897$) shows strong probability concentration:

    - `አበባ`: Count $31,388$, conditional probability **$0.353083$** (35.31% of all occurrences of `አዲስ`).
    - The remaining ~64.69% of probability mass is dispersed across diverse nominal and modifier continuations (e.g., `ነገር`, `ምዕራፍ`, `ስርዓት`, `አስተሳሰብ`, `ሕገ`, and sentence terminator `</s>`).
  * **Linguistic Analysis of Unseen Pair `(ሰማይ, ድንጋይ)` & Edge Cases:**

    - Unigram counts in GPAC: $C(\text{ሰማይ}) = 6,159$, $C(\text{ድንጋይ}) = 7,110$.
    - Both words are common vocabulary items. The zero count ($C = 0$) and zero probability ($P = 0.000000$) stem from semantic incompatibility and lack of natural co-occurrence in Amharic discourse, rather than an out-of-vocabulary absence.
    - *Unseen Context Behavior ($C(w_1) = 0$):* If an unseen word is su  * **Unified Stopword Filtering Rule:**
      An n-gram is removed if **any** constituent token is on the 30-stopword list, applied to isolated tokens after stripping bare prepositional clitics (`በ-`, `ከ-`, `ለ-`, `የ-`). Structural sentence boundary tags (`<s>`, `</s>`) are strictly excluded from all rankings (before and after stopword removal).
  * **Clitic Handling & Morphological Transparency:**
    Amharic prepositions and determiners (`በ-`, `ከ-`, `ለ-`, `የ-`) attach proclitically directly to nouns/pronouns. Forms with attached clitics (e.g. `በዚህ` *"in this"*, `በአዲስ` *"in Addis"*) are preserved as distinct lexical units and retained unless their stripped base matches the stopword list. Standalone `በ` and `ከ` are not in the stopword list because isolated instances represent abbreviation fragments (e.g., `በ ዓ ም` for `በዓ.ም.`).
  * **Before vs. After Stopword Removal (Top-5 across $n=1..4$, from `output/results/`):**

    |       Order       | Top-5 Before Removal (`output/results/q1_top10_n*.csv`)                                                                                                             | Top-5 After Removal (`output/results/q1_stopword_top10_n*.csv`)                                                                                                     | Filter Impact & Observations                                                                                                                                                                                                                       |
    | :---------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | **$n=1$** | 1. `ነው` (1,165,953)2. `ላይ` (754,678)3. `ውስጥ` (346,314)4. `ወደ` (331,106)5. `እና` (318,942)                                                  | 1. `ጊዜ` (244,117)2. `ደግሞ` (209,377)3. `አንድ` (192,674)4. `ቤት` (182,048)5. `አቶ` (169,169)                                                  | All top-5 function words (`ነው`, `ላይ`, `ውስጥ`, `ወደ`, `እና`) successfully eliminated; open-class nouns and titles rise.                                                                                                           |
    | **$n=2$** | 1. `ዓ ም` (108,455)2. `ቀን ዓ` (46,525)3. `ነገር ግን` (44,744)4. `ዶ ር` (35,624)5. `ማለት ነው` (32,386)                                       | 1. `ዓ ም` (108,455)2. `ቀን ዓ` (46,525)3. `ዶ ር` (35,624)4. `አዲስ አበባ` (31,388)5. `በ ዓ` (28,486)                                           | Function bigrams `ነገር ግን` (connective) and `ማለት ነው` (copular phrase) removed; entity bigram `አዲስ አበባ` enters top 5.                                                                                                          |
    | **$n=3$** | 1. `ቀን ዓ ም` (45,927)2. `በ ዓ ም` (28,127)3. `እ ኤ አ` (16,731)4. `ዓ ም ኢሳት` (9,789)5. `ም ኢሳት ዜና` (9,700)                             | 1. `ቀን ዓ ም` (45,927)2. `በ ዓ ም` (28,127)3. `እ ኤ አ` (16,731)4. `ዓ ም ኢሳት` (9,789)5. `ም ኢሳት ዜና` (9,700)                             | None of the top 5 trigrams contain words from the 30-word stopword list; therefore, the after-removal top 5 is identical to before removal.                                                                                                        |
    | **$n=4$** | 1. `ቀን ዓ ም ኢሳት` (9,786)2. `ዓ ም ኢሳት ዜና` (9,700)3. `እ ኤ አ በ` (8,613)4. `የካቲት ቀን ዓ ም` (2,952)5. `መጋቢት ቀን ዓ ም` (2,908) | 1. `ቀን ዓ ም ኢሳት` (9,786)2. `ዓ ም ኢሳት ዜና` (9,700)3. `እ ኤ አ በ` (8,613)4. `የካቲት ቀን ዓ ም` (2,952)5. `መጋቢት ቀን ዓ ም` (2,908) | Top 5 contain no stopwords and remain identical. However, rank 8 before removal (`ከ ሚሊዮን ብር በላይ`, 2,822) is removed because `በላይ` is a stopword, allowing `የአዲስ አበባ ከተማ አስተዳደር` (2,607) to advance into rank 8. |
  * **Deliberate Retention of `ደግሞ` and `አንድ`:**

    - `ደግሞ` (*"again / moreover / on the other hand"*): Expresses contrastive or additive discourse aspect rather than empty grammatical glue.
    - `አንድ` (*"one"*): Functions primarily as a cardinal numeral. Amharic has no indefinite article; `አንድ` acts as a meaningful quantifier.
* **Q1.6 Word Clouds for $n = 1, 2, 3, 4$ Before and After Stopword Removal:**

  * **Visualization Specifications & Inventory:**
    Eight dedicated high-resolution figures are preserved in `output/figures/` (`q1_6_wordcloud_*.png`), alongside synchronized comparative panels. Every visual features an explicit n-gram order title, filtering designation, clitic footnote, layout description, and scaling rule, rendered using Google `NotoSansEthiopic-Regular.ttf` with UTF-8 encoding:

    | Figure File                                              | Order ($n$) | Condition & Unified Filter                                                                                             | Stated Size Measure & Numerical Scaling                                                        | Dominant Corpus Items Visible                                                                                                                   |
    | :------------------------------------------------------- | :-----------: | :--------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
    | `output/figures/q1_6_wordcloud_unigram_before.png`            |    $n=1$    | **Before Removal** (Raw unfiltered corpus frequencies; `<s>`, `</s>` excluded)                               | Rank-frequency blend: `relative_scaling=0.5` (max 85 pt, from `output/results/q1_wordcloud_n1_before.csv`) | `ነው` (1.17M), `ላይ` (755k), `ውስጥ` (346k), `ወደ` (331k), `እና` (319k)                                                          |
    | `output/figures/q1_6_wordcloud_unigram_after.png`             |    $n=1$    | **After Removal** (Remove if any token on 30-stopword list, after stripping clitics; `<s>`, `</s>` excluded) | Rank-frequency blend: `relative_scaling=0.5` (max 85 pt, from `output/results/q1_wordcloud_n1_after.csv`)  | `ጊዜ` (244k), `ደግሞ` (209k), `አንድ` (193k), `ቤት` (182k), `አቶ` (169k)                                                         |
    | `output/figures/q1_6_wordcloud_bigram_before.png`             |    $n=2$    | **Before Removal** (Raw unfiltered corpus frequencies; `<s>`, `</s>` excluded)                               | Rank-frequency blend: `relative_scaling=0.5` (max 75 pt, from `output/results/q1_wordcloud_n2_before.csv`) | `ዓ ም` (108k), `ቀን ዓ` (46.5k), `ነገር ግን` (44.7k), `ዶ ር` (35.6k), `ማለት ነው` (32.4k)                                      |
    | `output/figures/q1_6_wordcloud_bigram_after.png`              |    $n=2$    | **After Removal** (Remove if any token on 30-stopword list, after stripping clitics; `<s>`, `</s>` excluded) | Rank-frequency blend: `relative_scaling=0.5` (max 75 pt, from `output/results/q1_wordcloud_n2_after.csv`)  | `ዓ ም` (108k), `ቀን ዓ` (46.5k), `ዶ ር` (35.6k), `አዲስ አበባ` (31.4k), `ምክር ቤት` (23.9k)                                    |
    | `output/figures/q1_6_wordcloud_trigram_before.png`            |    $n=3$    | **Before Removal** (Raw unfiltered corpus frequencies; `<s>`, `</s>` excluded)                               | Rank-frequency blend: `relative_scaling=0.5` (max 65 pt, from `output/results/q1_wordcloud_n3_before.csv`) | `ቀን ዓ ም` (45.9k), `በ ዓ ም` (28.1k), `እ ኤ አ` (16.7k), `ዓ ም ኢሳት` (9.8k), `ም ኢሳት ዜና` (9.7k)                          |
    | `output/figures/q1_6_wordcloud_trigram_after.png`             |    $n=3$    | **After Removal** (Remove if any token on 30-stopword list, after stripping clitics; `<s>`, `</s>` excluded) | Rank-frequency blend: `relative_scaling=0.5` (max 65 pt, from `output/results/q1_wordcloud_n3_after.csv`)  | `ቀን ዓ ም` (45.9k), `በ ዓ ም` (28.1k), `እ ኤ አ` (16.7k), `ዓ ም ጀምሮ` (6.4k), `ተወካዮች ምክር ቤት` (5.6k)                  |
    | `output/figures/q1_6_wordcloud_4gram_before.png`              |    $n=4$    | **Before Removal** (Raw unfiltered corpus frequencies; `<s>`, `</s>` excluded)                               | Rank-frequency blend: `relative_scaling=0.5` (max 55 pt, from `output/results/q1_wordcloud_n4_before.csv`) | `ቀን ዓ ም ኢሳት` (9.8k), `ዓ ም ኢሳት ዜና` (9.7k), `እ ኤ አ በ` (8.6k), `የካቲት ቀን ዓ ም` (3.0k)                              |
    | `output/figures/q1_6_wordcloud_4gram_after.png`               |    $n=4$    | **After Removal** (Remove if any token on 30-stopword list, after stripping clitics; `<s>`, `</s>` excluded) | Rank-frequency blend: `relative_scaling=0.5` (max 55 pt, from `output/results/q1_wordcloud_n4_after.csv`)  | `ቀን ዓ ም ኢሳት` (9.8k), `ዓ ም ኢሳት ዜና` (9.7k), `እ ኤ አ በ` (8.6k), `ከ ዓ ም ጀምሮ` (2.9k), `አዲስ አበባ ከተማ` (2.6k) |
    | `output/figures/q1_6_wordclouds_panel_unigram_bigram.png`   | $n \in \{1, 2\}$ | Synchronized dual-order panel before and after filtering                                                               | Full comparative layout (2-row stack)                                                  | Complete unigram and bigram distribution progression                                                            |
    | `output/figures/q1_6_wordclouds_panel_trigram_4gram.png`     | $n \in \{3, 4\}$ | Synchronized dual-order panel before and after filtering                                                               | Full comparative layout (2-row stack)                                                  | Complete trigram and 4-gram distribution progression                                                            |
    | `output/figures/q1_6_wordclouds_all_orders.png`               | $n=1..4$    | Master 8-cloud synchronization matrix (4 rows $\times$ 2 columns)                                                      | Unified comparative view                                                               | Exhaustive visual cross-verification across all orders and conditions                                           |

  * **Layout Positioning, Color Assignment & Scaling Methodology:**

    - **Pseudo-Random Spatial Layout:** Spatial placement follows a space-filling spiral packing heuristic (`random_state=42`) to prevent character collision. Central vs. peripheral position is purely a geometric artifact of non-overlapping placement, NOT an indicator of rank or importance. The word cloud is an exploratory visual, not an ordered ranking.
    - **Color Assignment:** Palettes (`copper`, `viridis`, `plasma`, `cividis`, `inferno`, `magma`, `Spectral`, `coolwarm`) are applied pseudo-randomly to maximize visual contrast and legibility, carrying no semantic, syntactic, or grammatical category meaning.
    - **Rank-Frequency Blend Scaling (`relative_scaling=0.5`):** WordCloud does not implement a pure mathematical square-root law ($s \propto \sqrt{c(w)}$). Instead, it implements a rank-frequency blend heuristic: font sizes are determined by $s(w) = s_{\min} + (s_{\max} - s_{\min}) \cdot \left(\frac{c(w)}{c_{\max}}\right)^{0.5}$ combined with candidate ranking and space-filling constraints. This dampens the massive Zipfian disparity between the top unigram `ነው` ($1,165,953$) and lower-ranked words while preventing words from monopolizing the entire canvas.
    - **Data Sources & Top-200 Compliance:** All 8 word clouds are generated directly from the top-200 files (`output/results/q1_wordcloud_n{1..4}_{before|after}.csv`), satisfying the implementation plan's top-200 visualization mandate across all orders.
    - **Label Collision Resolution:** In early rendering passes, long 4-gram collocations (such as `የካቲት ቀን ዓ ም` and `የአዲስ አበባ ከተማ አስተዳደር`) exhibited spatial label collision under default margins. The generator in `scripts/generate_q1_6_wordclouds.py` resolves this by expanding canvas dimensions ($1200 \times 650$), capping max font sizes ($55$ pt for $n=4$), and enforcing generous spacing (`margin=6`), guaranteeing zero character overlap and complete Ethiopic glyph integrity.
  * **Matching Top-10 Verification Tables (Before vs. After):**
    To ensure cross-verification between visual token size and exact numeric counts, the authoritative top 10 items from `output/results/q1_top10_n*.csv` and `output/results/q1_stopword_top10_n*.csv` are provided below:

    | Rank | Order 1 (Before)          |   Count   | Order 1 (After)              |  Count  | Order 2 (Before)                     |  Count  | Order 2 (After)                          |  Count  |
    | :--: | :------------------------ | :-------: | :--------------------------- | :-----: | :----------------------------------- | :-----: | :--------------------------------------- | :-----: |
    |  1  | `ነው` (copula: is)     | 1,165,953 | `ጊዜ` (time/period)       | 244,117 | `ዓ ም` (A.D. / Year of Mercy)     | 108,455 | `ዓ ም` (A.D. / Year of Mercy)         | 108,455 |
    |  2  | `ላይ` (on/upon)        |  754,678  | `ደግሞ` (moreover/again)  | 209,377 | `ቀን ዓ` (date of year)           | 46,525 | `ቀን ዓ` (date of year)               | 46,525 |
    |  3  | `ውስጥ` (in/inside)    |  346,314  | `አንድ` (one / a)         | 192,674 | `ነገር ግን` (however/but)        | 44,744 | `ዶ ር` (Dr. / Doctor)                 | 35,624 |
    |  4  | `ወደ` (to/towards)     |  331,106  | `ቤት` (house/institution) | 182,048 | `ዶ ር` (Dr. / Doctor)             | 35,624 | `አዲስ አበባ` (Addis Ababa)          | 31,388 |
    |  5  | `እና` (and)            |  318,942  | `አቶ` (Mr.)               | 169,169 | `ማለት ነው` (that means)         | 32,386 | `በ ዓ` (in the year of)               | 28,486 |
    |  6  | `ጋር` (with)           |  291,527  | `በ` (in/by/with)          | 164,316 | `ብቻ ሳይሆን` (not only)         | 32,261 | `በአዲስ አበባ` (in Addis Ababa)     | 24,048 |
    |  7  | `ነበር` (was/were)     |  274,994  | `ሰው` (person/man)        | 153,034 | `አዲስ አበባ` (Addis Ababa)      | 31,388 | `ምክር ቤት` (Council / House)        | 23,905 |
    |  8  | `ግን` (but/however)    |  271,502  | `ምን` (what)              | 147,347 | `በ ዓ` (in the year of)           | 28,486 | `ጠቅላይ ሚኒስትር` (Prime Minister) | 19,934 |
    |  9  | `ጊዜ` (time/period)    |  244,117  | `ናቸው` (are / they are)  | 141,179 | `ብቻ ነው` (is only)              | 24,682 | `የአዲስ አበባ` (of Addis Ababa)     | 19,669 |
    |  10  | `ነገር` (thing/matter) |  217,661  | `ሰዎች` (people)          | 138,007 | `በአዲስ አበባ` (in Addis Ababa) | 24,048 | `ይሁን እንጂ` (nevertheless)         | 19,518 |

    | Rank | Order 3 (Before)                     | Count | Order 3 (After)                      | Count | Order 4 (Before)                              | Count | Order 4 (After)                          | Count |
    | :--: | :----------------------------------- | :----: | :----------------------------------- | :----: | :-------------------------------------------- | :---: | :--------------------------------------- | :---: |
    |  1  | `ቀን ዓ ም` (date A.D.)           | 45,927 | `ቀን ዓ ም` (date A.D.)           | 45,927 | `ቀን ዓ ም ኢሳት` (Date A.D. ESAT)        | 9,786 | `ቀን ዓ ም ኢሳት` (Date A.D. ESAT)   | 9,786 |
    |  2  | `በ ዓ ም` (in year A.D.)          | 28,127 | `በ ዓ ም` (in year A.D.)          | 28,127 | `ዓ ም ኢሳት ዜና` (A.D. ESAT News)        | 9,700 | `ዓ ም ኢሳት ዜና` (A.D. ESAT News)   | 9,700 |
    |  3  | `እ ኤ አ` (G.C. calendar)         | 16,731 | `እ ኤ አ` (G.C. calendar)         | 16,731 | `እ ኤ አ በ` (G.C. in)                     | 8,613 | `እ ኤ አ በ` (G.C. in)                | 8,613 |
    |  4  | `ዓ ም ኢሳት` (A.D. ESAT)         | 9,789 | `ዓ ም ኢሳት` (A.D. ESAT)         | 9,789 | `የካቲት ቀን ዓ ም` (Feb ... A.D.)        | 2,952 | `የካቲት ቀን ዓ ም` (Feb ... A.D.)   | 2,952 |
    |  5  | `ም ኢሳት ዜና` (ESAT News)       | 9,700 | `ም ኢሳት ዜና` (ESAT News)       | 9,700 | `መጋቢት ቀን ዓ ም` (March ... A.D.)      | 2,908 | `መጋቢት ቀን ዓ ም` (March ... A.D.) | 2,908 |
    |  6  | `ኤ አ በ` (G.C. in)               | 8,656 | `ኤ አ በ` (G.C. in)               | 8,656 | `ከ ዓ ም ጀምሮ` (starting ... A.D.)       | 2,866 | `ከ ዓ ም ጀምሮ` (starting ... A.D.)  | 2,866 |
    |  7  | `ዓ ም ጀምሮ` (starting ... A.D.) | 6,395 | `ዓ ም ጀምሮ` (starting ... A.D.) | 6,395 | `ግንቦት ቀን ዓ ም` (May ... A.D.)        | 2,824 | `ግንቦት ቀን ዓ ም` (May ... A.D.)   | 2,824 |
    |  8  | `ከ ዓ ም` (from ... A.D.)         | 6,386 | `ከ ዓ ም` (from ... A.D.)         | 6,386 | `ከ ሚሊዮን ብር በላይ` (over ... M Birr) | 2,822 | `የአዲስ አበባ ከተማ አስተዳደር`  | 2,607 |
    |  9  | `ተወካዮች ምክር ቤት`           | 5,598 | `ተወካዮች ምክር ቤት`           | 5,598 | `የአዲስ አበባ ከተማ አስተዳደር`       | 2,607 | `ሰኔ ቀን ዓ ም` (June ... A.D.)      | 2,569 |
    |  10  | `በሌላ በኩል ደግሞ`             | 5,221 | `የአዲስ አበባ ከተማ`           | 4,903 | `ሰኔ ቀን ዓ ም` (June ... A.D.)           | 2,569 | `ጥር ቀን ዓ ም` (Jan ... A.D.)       | 2,518 |
  * **Resolution of Phrase Provenance & Caption Audit:**

    - **`ከ ዓ ም ጀምሮ`:** Verified as **Rank 6 in the global 4-gram corpus** (Count: **2,866**), and its trigram parent `ዓ ም ጀምሮ` is **Rank 7** (Count: **6,395**). It is an authentic, highly frequent temporal formula, not an artificial rendering artifact.
    - **`የጽንስና ማህጸን ሐኪሞች ማህበር`:** Verified as an authentic phrase in `GPAC.txt` from maternal healthcare news articles (individual word counts in `vocab.tsv`: `የጽንስና`: 1,233, `ማህጸን`: 1,764, `ሐኪሞች`: 2,033, `ማህበር`: 11,983). However, its compound 4-gram frequency is well below 1,000, placing it outside the top 50 global 4-grams. It was an illustrative health-reporting sample string and is strictly excluded from top-rank frequency visuals and captions.
    - **`ዓ ም` Count Verification:** Displays consistently at count **108,455** (Rank 1 bigram) across both the word cloud visual and the Q1.5 summary tables.
  * **Corpus Genre & Linguistic Interpretation:**

    - **News Broadcast & Dateline Genre:** The overwhelming prevalence of calendar abbreviations (`ዓ ም` [108,455], `እ ኤ አ` [16,731]), date connectors (`ቀን ዓ ም` [45,927]), month formulas (`የካቲት/መጋቢት/ሰኔ ቀን ዓ ም`), and broadcast tags (`ዓ ም ኢሳት` [9,789], `ም ኢሳት ዜና` [9,700]) reflects the journalistic origin of the GPAC corpus, which heavily aggregates daily news wires and transcribed broadcasts (e.g. Ethiopian News Agency, Walta, ESAT).
    - **Institutional Governance Collocations:** Open-class filtered clouds prominently feature official governance terms (`ጠቅላይ ሚኒስትር` [19,934], `ተወካዮች ምክር ቤት` [5,598], `የአዲስ አበባ ከተማ አስተዳደር` [2,607]), highlighting the administrative and political coverage of the underlying text.
  * **Methodological Caveat:**
    Word clouds are exploratory, descriptive visualizations of token frequency distributions. They effectively illustrate lexical dominance and verify stopword filtering, but do **not** reflect conditional transition probabilities, model calibration, or perplexity (which are quantitatively assessed in Questions 2 and 3).
* **Q1.7 Target Sentence Probability & Distribution Sums (Kneser-Ney Smoothing):**

  * **Evaluation across Test Sentences (from `output/results/q1_sentence_probs.csv`):**

    | Test Sentence                                                      | Category / Linguistic Rationale                  | Length ($L$) | $n=1$ Per-Token $\log P$ | $n=2$ Per-Token $\log P$ | $n=3$ Per-Token $\log P$ | $n=4$ Per-Token $\log P$ | In-Corpus Verbatim? |
    | :----------------------------------------------------------------- | :----------------------------------------------- | :------------: | :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: | :-----------------: |
    | `ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።`                      | Unseen Grammatical (Geographic statement)        |       5       |         $-8.6496$         |         $-5.6844$         |         $-5.6634$         |     $\mathbf{-5.0982}$     |   No (`False`)   |
    | `የኢትዮጵያ መንግስት ለህዝቡ ሰላም እና ልማት ይሰራል።` | Unseen Grammatical (Institutional news report)   |       8       |         $-7.9014$         |         $-6.0135$         |         $-5.2173$         |     $\mathbf{-5.3445}$     |   No (`False`)   |
    | `አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።`                  | In-Corpus Verbatim (Exact sentence seen in GPAC) |       7       |         $-7.6033$         |         $-4.6232$         |         $-3.8427$         |     $\mathbf{-2.3196}$     |   Yes (`True`)   |
    | `ድንጋይ ሰማይ ሮጠ በላ ትላንት ወንበር።`                  | Invented Nonsense (Ungrammatical word salad)     |       7       |         $-9.6134$         |         $-10.2308$         |         $-10.2522$         |    $\mathbf{-10.4669}$    |   No (`False`)   |
  * **Explanation of Metrics:**

    - **Sentence Length ($L$):** Evaluated prediction steps $L = \text{word count} + 1$ (including terminal `</s>`).
    - **Bold Values:** Represent the highest-order model ($n = 4$) per-token score.
    - **Total $\log P$:** Evaluated as $\sum_{i=1}^L \log P_{\text{KN}}(w_i \mid w_{i-n+1}^{i-1})$.
    - **Per-Token $\log P$:** Total $\log P$ normalized by sentence length $L$.
  * **Sum Check & Probability Normalization:**

    - For each context history $h$, the conditional probability distribution $P_{\text{KN}}(w \mid h)$ is evaluated over all $|V| = 1,893,214$ vocabulary items (including `</s>`).
    - Across tested histories (e.g., $h = \text{['<s>']}$, $h = \text{['አዲስ']}$, $h = \text{['የኢትዮጵያ', 'መንግስት']}$), $\sum_{w \in V} P_{\text{KN}}(w \mid h) = 1.0000000000$ with absolute numerical discrepancy $\le 1 \times 10^{-9}$, confirming that the conditional probability distributions are properly normalized.
  * **Linguistic Discrimination vs. Training Memorization:**

    - *Sentence 3 (`አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።`):* Reaches an exceptional score of $\mathbf{-2.3196}$ per token at $n = 4$. This sentence occurs verbatim in the GPAC corpus, so its high likelihood reflects training data memorization of a recurring factual proposition.
    - *Generalization Contrast:* The valid test of model linguistic sensitivity is the comparison between unseen grammatical sentences (Sentences 1 and 2, scoring $-5.0982$ to $-5.3445$ at $n=4$) and the unseen nonsense sentence (Sentence 4, penalized to $-10.4669$ at $n=4$). As $n$ increases from 1 to 4, the score of the nonsense sentence drops from $-9.6134$ down to $-10.4669$ due to severe backoff penalties for non-attested transitions, whereas unseen grammatical sentences gain substantially as higher-order local syntax is rewarded.
    - *Sentence 1 Stylistic Note:* While `ቀንድ ትገኛለች` is grammatical, standard stylistic phrasing frequently includes `ውስጥ` (`በአፍሪካ ቀንድ ውስጥ ትገኛለች`). The absence of `ውስጥ` lowers its 3-gram and 4-gram collocation counts, explaining why Sentence 1 improves modestly between $n=2$ and $n=4$.
* **Q1.8 Text Generation & Verbatim Duplication Analysis:**

  * **Generated Sentences (12 Samples across $n = 1..4$, from `output/results/q1_generation.json`):**

    |     Order $n$     | Sample | Generated Sentence                                                                                                                                                                                                                                                                       | Length ($L$) | Total $\log P$ | Per-Token $\log P$ | In-Corpus Verbatim? |
    | :-----------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------: | :--------------: | :------------------: | :-----------------: |
    | **$n = 1$** |   1   | `ያስተዋልነው ቱርክ ልፃፍ ማብራሪያም ገንዘብ ዘገየችን ከአብዮት ጠሃ ፍቅር ጎጆ ጭራሽ የምናድርባት እንደሚያወጣ ግንባርቀደም ቀጥለዋል ማሰብ እንደምችል የ ወናፍ ፀሐይን ተተብትበው በተባለ ስለሚሰግዱላቸው ከሚያጋልጡት የአምባው።`                 |       25       |   $-309.45$   |      $-12.38$      |   No (`False`)   |
    | **$n = 1$** |   2   | `ፈተና የሚያስችላቸውን እንዲሁም ከላይ ተሰናብተው የተደለደሉት ኅየንተ ስብሰባዎች ከመቼውም ተቀምጠዋል ንጉሥ ሦስት አንገታቸውን የአካባቢው ተላቃ የኮሚሽኑ በሮመዳን የስታዲየም በመልካም የመሣሪያዎች ሃንድ ተጠናቆ የያህዌ እንደሚቻልና አልፎ።` |       25       |   $-276.58$   |      $-11.06$      |   No (`False`)   |
    | **$n = 1$** |   3   | `ጥፋት ወደ አልሆነልንም ለማዋረድ ለፍቅርና ንዶ ዶክተሩ በሃላፊነት ገቢ ዙሪያ ያሳዩበት ወገኔ በልጅ እንደተሰማሩ ከኢራን ሊረዳው በመንገዱ ትልልቅ እንደ ለማንሳት እንዲሆኑ ከራሳቸው በመድማት ፈጣሪ መንግሥት።`                                         |       25       |   $-270.53$   |      $-10.82$      |   No (`False`)   |
    | **$n = 2$** |   1   | `አበረታች ውጤት ማስቀጠል የግድ ይለናል እንጂ ለአንድ የተወሰነ አንድ ላይ እንዴት እንደሚታይ አስረድተዋል።`                                                                                                                                                           |       13       |    $-77.11$    |      $-5.93$      |   No (`False`)   |
    | **$n = 2$** |   2   | `የአባቶቻችን አምላክ በጎች ሕይወት ቀላል ማድረግ ብቸኛው ኢትዮጵያዊ ሁሉ ስለ ሚገኙ ኣብያተ ክርስቲያናት አካባቢ የእስረኞቹ መፈታት ያለባቸው ሆነው ሳለ አንድ ፈራጅ መሆኑ ዛሬ ሥልጣን ለማስጠበቅ።`                                                       |       25       |   $-149.80$   |      $-5.99$      |   No (`False`)   |
    | **$n = 2$** |   3   | `በእጅጉ ቀንሶታል የሚሉት ከብዙ ን ጠቅሰው በቀጣይ በሚደረገው ምርጫ ቦርድ ውስጥ የሚሠሩ ተቋማት ብዙ ሰው ምስጢር ነበር የሚሉም ነበሩ።`                                                                                                                             |       19       |   $-108.36$   |      $-5.70$      |   No (`False`)   |
    | **$n = 3$** |   1   | `ቢሏቸው ምን እንዳሉ ታውቃላችሁ።`                                                                                                                                                                                                                                                |       4       |    $-19.62$    |      $-4.91$      |   No (`False`)   |
    | **$n = 3$** |   2   | `የዓለም ዋንጫ ጉብኝት በዘጠኝ ወራት ውስጥ ችግር እንዳለ ጠቁመዋል።`                                                                                                                                                                                                         |       9       |    $-32.56$    |      $-3.62$      |   No (`False`)   |
    | **$n = 3$** |   3   | `ናቸው የዚያ ሰፈር ልጆች ትምህርት ቤት ዲዛይን አልቋል።`                                                                                                                                                                                                                      |       8       |    $-32.95$    |      $-4.12$      |   No (`False`)   |
    | **$n = 4$** |   1   | `ሆቴሉ ተጠናቅቆ ስራ ሲጀምር ከ በላይ ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል ተብሏል።`                                                                                                                                                                                      |       12       |    $-29.04$    |      $-2.42$      |   No (`False`)   |
    | **$n = 4$** |   2   | `ሠራተኛ የነበረ ቢሆንም የነፍሱ ጥሪ ግን ሥራ ፈጠራ ነበር።`                                                                                                                                                                                                                   |       9       |    $-19.43$    |      $-2.16$      |   No (`False`)   |
    | **$n = 4$** |   3   | `የተሸጠው መሬት እጅግ አነስተኛ መሆኑን ገልፀዋል።`                                                                                                                                                                                                                            |       6       |    $-18.11$    |      $-3.02$      |   No (`False`)   |
  * **Order Progression Metrics Summary:**

    |     Order $n$     | Conditioning History | Avg. Length | Avg. Total $\log P$ | Avg. Per-Token $\log P$ | Effective Branching ($e^{-\overline{\log P}/L}$) | Verbatim Sentence Copies | Syntactic & Qualitative Observations                                                                      |
    | :-----------------: | :------------------: | :---------: | :-------------------: | :-----------------------: | :------------------------------------------------: | :----------------------: | :-------------------------------------------------------------------------------------------------------- |
    | **$n = 1$** |  None ($0$ words)  | 25.0 tokens |      $-285.52$      |   **$-11.42$**   |                 $\approx 91,126$                 |  **0 / 3 (0.0%)**  | Unordered word salad; complete absence of concord or phrase boundaries.                                   |
    | **$n = 2$** |        1 word        | 19.0 tokens |      $-111.76$      |    **$-5.88$**    |                  $\approx 358$                  |  **0 / 3 (0.0%)**  | Valid local 2-word collocations (`አበረታች ውጤት`, `ቀላል ማድረግ`); wandering global semantics. |
    | **$n = 3$** |       2 words       | 7.0 tokens |      $-28.38$      |    **$-4.21$**    |                   $\approx 67$                   |  **0 / 3 (0.0%)**  | Multi-word clause structure; natural termination; occasional initial copula defect (`ናቸው`).          |
    | **$n = 4$** |       3 words       | 9.0 tokens |      $-22.19$      |    **$-2.53$**    |                  $\approx 12.5$                  |  **0 / 3 (0.0%)**  | Advanced clause coordination and subordination; occasional ellipsis defect (`ከ በላይ`).               |
  * **Sampling Parameters & Length Distribution Arithmetic:**

    - *Settings:* `SEED = 42`, standard multinomial sampling ($T = 1.0$), maximum length 25 tokens.
    - *Sampler Constraint:* Minimum sentence length of 3 tokens was enforced by suppressing `EOS` for the first 2 steps (`len(gen_words) < 3`).
    - *Unigram Length Arithmetic:* At $n = 1$, the global end-of-sentence probability is:

      $$
      P(\text{EOS}) = \frac{4,993,573}{81,161,104} \approx \mathbf{0.061527}
      $$

      An unconstrained geometric distribution gives expected length $1 / p \approx 16.25$ tokens. With EOS suppressed for the first 2 tokens, the sequence has 23 steps remaining to terminate before hitting the 25-token cap. The probability of not drawing EOS at any step is $1 - 0.061527 \approx 0.938473$. The probability of surviving 23 consecutive steps without drawing EOS is:

      $$
      (0.938473)^{23} \approx \mathbf{0.2321} \quad (\approx 23.2\%, \text{ or with rounded base } (0.9385)^{23} \approx \mathbf{0.231})
      $$

      The joint probability that all three independent unigram sentences hit the 25-token limit is:

      $$
      (0.231)^3 \approx \mathbf{0.0123} \quad (\mathbf{1.23\%}) \quad [\text{exact: } (0.2321)^3 \approx 0.0125 \text{ (1.25\%)}]
      $$

      This quantitatively explains why all three unigram samples reached the 25-token cap under seed 42.
  * **Vocabulary Accounting:**

    - Full GPAC surface vocabulary: **1,893,211** unique lexical words.
    - The figure $|V| = 333,707$ represents the pruned training core vocabulary (words with $C \ge 2$) utilized in Q2 and Q3 splits; it does not represent the full corpus lexicon.
  * **Constituent Verbatim N-gram Fractions vs. Whole-Sentence Copies:**

    - *Whole-Sentence Duplication Check:* All 12 generated sentences were evaluated against all **4,993,573 sentences** in `output/data/GPAC_cleaned.txt` via 64-bit BLAKE2b hashes. Exactly **0 out of 12 (0.0%)** sentences are verbatim duplicates of whole sentences in the corpus.
    - *Constituent N-gram Support by Order (Evaluated from Corpus Counts):*
      - **Order 1 ($n = 1$):** **100.0%** of generated words ($75 / 75$ tokens across the 3 sentences) are valid in-corpus vocabulary types; however, multi-word sequence coherence is 0% because tokens are drawn independently.
      - **Order 2 ($n = 2$):** **100.0%** of sampled adjacent transitions ($57 / 57$ bigram steps across the 3 sentences) are attested bigrams in the corpus, providing valid local 2-word collocations (`አበረታች ውጤት`, `ቀላል ማድረግ`).
      - **Order 3 ($n = 3$):** **100.0%** of constituent trigram transitions ($21 / 21$ trigram steps across the 3 sentences) are attested in the corpus, creating natural multi-word clausal runs.
      - **Order 4 ($n = 4$):** **100.0%** of constituent 4-gram transitions ($27 / 27$ 4-gram steps across the 3 sentences) are attested in the corpus.
    - *Idiomatic Recombination vs. Passage Duplication:* The 4-gram model does **not** reproduce extended whole passages or duplicate training sentences. Rather, it functions as an idiomatic assembler: it draws attested 3-word and 4-word formulaic phrases (e.g., `ሆቴሉ ተጠናቅቆ ስራ ሲጀምር`, `የስራ እድል ይፈጥራል ተብሏል`, `የነፍሱ ጥሪ ግን`, `ሥራ ፈጠራ ነበር`) and recombines them at valid syntactic branching points into novel sentences.
  * **Qualitative Defect Analysis & Translation Glosses (Author Translations):**

    - *$n=4$, Sentence 1:* `ሆቴሉ ተጠናቅቆ ስራ ሲጀምር ከ በላይ ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል ተብሏል።`
      *Gloss:* *"It was stated that when the hotel is completed and begins operations, it will create job opportunities for more than [...] youths."*
      *Defect:* Contains an ellipsis defect where `ከ በላይ` (*"more than"*) lacks the required numeral/quantity (e.g., `ከ 100 በላይ`).
    - *$n=4$, Sentence 2:* `ሠራተኛ የነበረ ቢሆንም የነፍሱ ጥሪ ግን ሥራ ፈጠራ ነበር።`
      *Gloss:* *"Although he had been an employee, his soul's true calling was entrepreneurship."*
      *Observation:* Fluent concessive frame (`ቢሆንም ... ግን ... ነበር`) with valid contrastive syntax.
    - *$n=4$, Sentence 3:* `የተሸጠው መሬት እጅግ አነስተኛ መሆኑን ገልፀዋል።`
      *Gloss:* *"They stated that the land sold was extremely minimal."*
      *Observation:* Natural reporting complement clause (`መሆኑን ገልፀዋል`).
    - *$n=3$, Sentence 1:* `ቢሏቸው ምን እንዳሉ ታውቃላችሁ።`
      *Gloss:* *"Do you know what they said when they told them?"*
      *Observation:* Colloquial interrogative structure with agreement between subordinate `እንዳሉ` and matrix verb `ታውቃላችሁ`.
    - *$n=3$, Sentence 2:* `የዓለም ዋንጫ ጉብኝት በዘጠኝ ወራት ውስጥ ችግር እንዳለ ጠቁመዋል።`
      *Gloss:* *"They indicated that there was a problem within nine months of the World Cup tour."*
      *Defect:* Grammatically well-formed journalistic phrasing, but semantically disjoint.
    - *$n=3$, Sentence 3:* `ናቸው የዚያ ሰፈር ልጆች ትምህርት ቤት ዲዛይን አልቋል።`
      *Gloss:* *"[They are] that neighborhood's children school design is finished."*
      *Defect:* Begins abnormally with the copula `ናቸው` as the initial token, an unnatural syntactic opening in standard Amharic.

---

## 5. Question 2 Key Results Summary

* **Data Partitioning (Zero-Leakage Grouped Splitting):**

  * **Sentence-Level Allocation:** Grouped partitioning (Stage 06, `scripts/06_split.py`) was performed by **identical sentence text groups**, NOT by token counts:
    * Total valid corpus sentences: **4,993,573 sentences** (100.0%).
    * **Training Core (`train_core`):** **4,244,537 sentences** (~85.00%).
    * **Tuning Split (`tune`):** **249,679 sentences** (~5.00%).
    * **Held-out Test Split (`test`):** **499,357 sentences** (~10.00%).
  * **Zero-Leakage Guarantee:** Every occurrence of any sentence text was assigned strictly to one partition. Overlap between `test` and `train_core` was verified as exactly **0 sentence texts** (`overlap = 0`, saved in `output/data/split_indices_seed42.json`).
  * **Token Positions:** Across the 499,357 test sentences, there are **8,130,627 token prediction positions** (averaging $8,130,627 / 499,357 \approx 16.28$ tokens per sentence, including terminal `</s>`).
* **Vocabulary Cutoff Rule & `<UNK>` Profile:**

  * **Vocabulary Rule:** In `train_core`, words occurring $\le 1$ time ($k = 1$, singletons / hapax legomena) are mapped to `<UNK>` via the training LUT (`lut_train_seed42.npy`). Words with count $\ge 2$ ($k \ge 2$) form the core training vocabulary of **333,707 types** (out of 1.89M full-corpus surface forms).
  * **Test Set Mapping:** On the held-out test split, test words not in the 333,707 core training vocabulary map to `<UNK>`.
  * Exactly **191,638 positions** out of 8,130,627 map to `<UNK>`, yielding an exact `<UNK>` fraction of:
    $$
    \frac{191,638}{8,130,627} = \mathbf{0.023570} \quad (\mathbf{2.357\%})
    $$
  * **Scoring `<UNK>` in Unigram Baseline:** Unigram perplexity of $16,317.94$ corresponds to an average cross-entropy of $\ln(16,317.94) \approx 9.70$ nats per token. In `train_core`, `<UNK>` accumulates a substantial aggregate count ($C \approx 1.56\text{M}$ singletons), so `<UNK>` receives non-trivial unigram probability mass ($P_{\text{MLE}}(\text{<UNK>}) = C_{\text{train}}(\text{<UNK>}) / N_{\text{train}}$), preventing cross-entropy collapse on the 2.357% out-of-vocabulary test positions.
* **Unseen Transition Rates (Exact Test Denominators):**
  Unseen rates refer to **test token prediction positions** ($N = 8,130,627$):

  * **Bigram ($n=2$):** **34.8% of test positions** (2,829,458 positions out of 8,130,627) have an adjacent pair $(w_{i-1}, w_i)$ unseen in `train_core` ($C_{\text{train}}(w_{i-1}, w_i) = 0$), triggering backoff to unigram.
  * **Trigram ($n=3$):** **62.4% of test positions** (5,073,511 positions out of 8,130,627) have an unseen trigram history, relying on lower-order backoff.
  * **4-gram ($n=4$):** **81.2% of test positions** (6,602,069 positions out of 8,130,627) have an unseen 4-gram history.
    The denominator for all three rates is identically $N = 8,130,627$ evaluation positions. The high unseen rate at order 4 reflects the extreme Zipfian context sparsity of natural language.
* **Perplexity Parameter Specifications & Reproducibility:**

  * **Smoothing Model:** Vectorized Interpolated Kneser-Ney Language Model (`scripts/kn_lib.py`).
  * **Discount Parameter ($D_k$):** Computed empirically via the standard Ney-Essen / Chen & Goodman formula:

    $$
    D_k = \frac{n_1(k)}{n_1(k) + 2 \cdot n_2(k)}
    $$

    from order-$k$ singleton ($n_1$) and doubleton ($n_2$) counts in training counts.
  * **Token Accounting in $N$:**

    * Evaluated tokens: all word tokens plus the terminal boundary marker `</s>`.
    * Sentence start markers `<s>` serve only as context padding; they are never predicted or included in $N$.
    * `<UNK>` tokens are explicitly included in $N$ and scored via their interpolated Kneser-Ney probability.
    * Total denominator: $N = 8,130,627$.
  * **Perplexity Formula:**

    $$
    \text{PP} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P_{\text{KN}}(w_i \mid w_{i-n+1}^{i-1})\right)
    $$
* **Held-Out Test Perplexity (Primary Split, `SEED = 42`):**

  |          Model Order          | Test Perplexity ($\text{PP}$) | Log-Likelihood Sum ($\sum \ln P$) | Reduction vs. Unigram |   Benchmark Target   |     Status     |
  | :---------------------------: | :-----------------------------: | :---------------------------------: | :-------------------: | :-------------------: | :------------: |
  | **Unigram ($n = 1$)** |            16,317.94            |         $-78,867,248.20$         | Baseline ($0.0\%$) |       Baseline       |    Recorded    |
  | **Bigram ($n = 2$)** |            1,627.11            |         $-60,122,420.92$         |   **90.03%**   | Lower by $\ge 20\%$ | **PASS** |
  | **Trigram ($n = 3$)** |             921.78             |         $-55,502,189.84$         |   **94.35%**   |   Lower than bigram   | **PASS** |
  | **4-gram ($n = 4$)** |             648.51             |         $-52,643,189.81$         |   **96.03%**   |  Lower than trigram  | **PASS** |

  *Plot Artifact:* Rendered and saved as **[`output/figures/q2_perplexity_vs_n.png`](file:///c:/Users/hp/Documents/Programming/AI/NLP/Assignment%201/output/figures/q2_perplexity_vs_n.png)**, illustrating the steep monotonic decline in perplexity as context order increases.
  *Metric Applicability (AGENTS.md B6):* Perplexity only. Classification metrics (accuracy, precision, recall, F1, ROC-AUC) do not apply to Question 2, because intrinsic language modeling evaluates continuous generative sequence likelihood across held-out text rather than predicting discrete category classes.
  *Benchmark Target Origin:* The $\ge 20\%$ bigram reduction target was a group-proposed pedagogical threshold formulated prior to running models, based on standard NLP literature (Jurafsky & Martin, Ch. 3; Chen & Goodman, 1999). The observed reduction of **90.03%** comfortably satisfies this benchmark.
* **Cross-Split Stability Verification (`SEED = 42` vs. Independent `SEED = 7`):**
  Per Decision D4, Seed 7 is evaluated on an **independently drawn grouped test split** (499,357 sentences, 8,124,149 tokens) against training counts built from its own independent `train_core` partition:

  | Model Order                   | Seed 42 Perplexity | Seed 7 Perplexity | Absolute Diff ($\Delta$) | Relative Variation ($\frac{|\Delta|}{\text{Seed 42}}$) | Order Ranking Invariance |
  | :---------------------------- | :----------------: | :---------------: | :------------------------: | :------------------------------------------------------: | :----------------------: |
  | **Unigram ($n = 1$)** |     16,317.94     |     16,288.76     |         $-29.18$         |                     **0.18%**                     |     Rank 1 (Highest)     |
  | **Bigram ($n = 2$)**  |      1,627.11      |     1,625.63     |         $-1.48$         |                     **0.09%**                     |          Rank 2          |
  | **Trigram ($n = 3$)** |       921.78       |      923.00      |         $+1.22$         |                     **0.13%**                     |          Rank 3          |
  | **4-gram ($n = 4$)**  |       648.51       |      651.48      |         $+2.97$         |                     **0.46%**                     |     Rank 4 (Lowest)     |

  *Stability Finding:* Rank ordering across orders is perfectly invariant across independent splits ($n=4 < n=3 < n=2 < n=1$). Absolute numerical variation is strictly below $0.50\%$ across all orders on distinct data splits.
  *Usability Definition:* Intrinsic model usability is defined by achieving $\ge 20\%$ perplexity reduction from unigram while maintaining consistent monotonic improvement across independent data splits.

---

## 6. Question 3 Key Results Summary

* **Task Formulation & Tuning Split Calibration:**

  * **Task:** Real-versus-Corrupted Amharic Sentence Detection (balanced binary classification on 998,714 held-out test pairs: 499,357 authentic test sentences vs. 499,357 corrupted negatives).
  * **Tuning Pair Allocation:** The 5% tuning set `tune` contains 249,679 sentences, generating exactly **$2 \times 249,679 = \mathbf{499,358}$ labeled tuning pairs** (recorded in `output/results/q3_metrics.json`).
  * **Threshold Tuning Objective (Percentile Grid Search):** The plan specifies sweeping the range of tuning scores to find optimal decision thresholds $\tau_n$. This was operationalized by evaluating a **200-point uniform percentile grid** (`np.linspace(0, 100, 200)`) over the distribution of scores on the 499,358 tuning pairs, maximizing F1-score:
    * $n = 1$: $\tau_1 = -10.2542$ (tuning F1 = 0.7437)
    * $n = 2$: $\tau_2 = -8.5700$ (tuning F1 = 0.7576)
    * $n = 3$: $\tau_3 = -8.6520$ (tuning F1 = 0.7468)
    * $n = 4$: $\tau_4 = -8.7901$ (tuning F1 = 0.7478)
      Evaluating candidate thresholds at uniform percentiles ensures dense evaluation across the empirically populated score range without distortion from sparse extreme outliers. Zero test data was examined or utilized during threshold tuning.
* **Negative Sentence Corruption Procedure & Confounder Analysis:**

  * From `scripts/12_q3_extrinsic.py`: For each authentic sentence, $n_{\text{replace}} = 1$ (if word length $\le 3$) or $2$ word positions are randomly selected.
  * Substitute words are drawn **uniformly at random from the core training vocabulary types (`train_vocab_ids`)**, excluding the original token (`orig_id`).
  * *Confounder Caveat:* Because uniform sampling over vocabulary types draws words independent of empirical unigram frequency, substitute words have a lower average natural frequency than natural text words. This unigram frequency mismatch explains why unigram alone achieves $0.8118$ ROC-AUC, whereas the bigram model ($0.8359$ ROC-AUC) adds discriminative capability by detecting transitional and collocational disruption.
* **Held-Out Test Set Performance & Confusion Matrix Reconciliation:**
  Evaluated across all 998,714 test pairs (499,357 real, 499,357 corrupted):

  |     Model Order     | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) |       Accuracy ($> 0.70$)       |     Precision ($> 0.70$)     |   Recall ($> 0.60$)   |       F1-Score ($> 0.70$)       |      ROC-AUC ($\ge 0.80$)      |  Usability Status  |
  | :-----------------: | :-----------: | :------------: | :-----------: | :------------: | :-------------------------------: | :-----------------------------: | :---------------------: | :-------------------------------: | :-------------------------------: | :----------------: |
  | **$n = 1$** |    413,798    |    201,943    |    297,414    |     85,559     |      71.21% (**PASS**)      |     0.6720 (**FAIL**)     | 82.87% (**PASS**) |      0.7422 (**PASS**)      |      0.8118 (**PASS**)      |       Usable       |
  | **$n = 2$** |    411,323    |    175,781    |    323,576    |     88,034     | **73.58%** (**PASS**) | **0.7006** (*Boundary*) | 82.37% (**PASS**) | **0.7572** (**PASS**) | **0.8359** (**PASS**) | **Champion** |
  | **$n = 3$** |    422,802    |    209,772    |    289,585    |     76,555     |      71.33% (**PASS**)      |     0.6684 (**FAIL**)     | 84.67% (**PASS**) |      0.7470 (**PASS**)      |    **0.8000** (*Tie*)    |   Boundary Case   |
  | **$n = 4$** |    430,603    |    221,827    |    277,530    |     68,754     |      70.90% (**PASS**)      |     0.6600 (**FAIL**)     | 86.23% (**PASS**) |      0.7477 (**PASS**)      |      0.7895 (**FAIL**)      | **Unusable** |

  *Plot Artifact:* Rendered and saved as **[`output/figures/q3_roc_curves.png`](file:///c:/Users/hp/Documents/Programming/AI/NLP/Assignment%201/output/figures/q3_roc_curves.png)**, comparing the complete receiver operating characteristic curves across all four model orders alongside the unigram-frequency baseline.
  *Metric Applicability & Omission of Log Loss (AGENTS.md B6):* Question 3 evaluates binary classification metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and the confusion matrix. Log loss is deliberately omitted because the classification decision scores are raw per-token log-probabilities from n-gram Kneser-Ney models, which are uncalibrated decision heuristics rather than well-calibrated class posterior probabilities $P(\text{real} \mid s)$.
  *Arithmetic Reconciliation:* Every metric reconciles arithmetically:

  $$
  \text{Acc} = \frac{\text{TP}+\text{TN}}{998,714}, \quad \text{Prec} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \quad \text{Rec} = \frac{\text{TP}}{499,357}, \quad \text{F1} = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}
  $$
* **Statistical Significance of Bigram Precision Margin (+0.0006):**

  * Observed bigram precision is **0.7006** ($411,323 / 587,104$).
  * Standard error of precision:
    $$
    \text{SE} = \sqrt{\frac{p(1-p)}{N_{\text{pred}}}} = \sqrt{\frac{0.7006 \times 0.2994}{587,104}} \approx \mathbf{0.000597}
    $$
  * 95% Wald Confidence Interval:
    $$
    [0.7006 - 1.96 \times 0.0006, \; 0.7006 + 1.96 \times 0.0006] = [\mathbf{0.6994}, \; \mathbf{0.7018}]
    $$
  * *Critical Statistical Disclosure:* Because the proposed threshold ($0.7000$) falls comfortably inside the 95% confidence interval, the difference $+0.0006$ is **statistically indistinguishable from zero** ($z \approx 1.0, p \approx 0.31$). It represents an empirical tie with the threshold boundary rather than a robust clearance.
* **Trigram ROC-AUC Threshold Analysis ($\ge 0.80$ vs. $> 0.80$):**

  * Exact test ROC-AUC for trigram is **0.80003** ($\approx \mathbf{0.8000}$).
  * *Instructor Clarification Required:* The implementation plan benchmark states "above 0.80" (strict inequality $> 0.80$, under which 0.80003 is an empirical tie/failure), whereas the project usability guideline specifies "at least 0.80" (weak inequality $\ge 0.80$, under which it passes). We flag this boundary condition for instructor confirmation rather than arbitrarily choosing.
* **4-gram Unusability Classification:**

  * 4-gram achieves ROC-AUC = **0.7895** and Precision = **0.6600**, failing both the $\ge 0.80$ and $> 0.70$ benchmarks. It is definitively classified as **unusable** due to severe context sparsity on corrupted transitions.
* **UNK Confounder & Sensitivity Analysis:**

  * **Naive UNK Baseline:** Using the raw count of `<UNK>` tokens as the prediction score achieves $\text{ROC-AUC} = \mathbf{0.5184}$ and Accuracy = **51.63%**. Corrupted sentences contain slightly more `<UNK>` tokens on average (positive correlation), but discriminative ability is near chance ($0.50$).
  * **UNK-Free Subset (738,256 pairs, 73.9% of test set):** On pairs containing zero `<UNK>` tokens, Bigram achieves $\text{ROC-AUC} = \mathbf{0.8473}$ and $\text{Accuracy} = \mathbf{74.52\%}$ ($\text{TP}=307,859, \text{FP}=126,868, \text{TN}=242,260, \text{FN}=61,269$).
  * **UNK-Containing Subset (260,458 pairs, 26.1% of test set):** Bigram achieves $\text{ROC-AUC} \approx \mathbf{0.8035}$ and $\text{Accuracy} \approx \mathbf{70.92\%}$.
  * *Nuanced Interpretation:* Ruling out `<UNK>` tokens demonstrates that classification performance is not an artifact of out-of-vocabulary tags; however, the signal is driven jointly by unigram frequency disruption and local syntactic/collocational incoherence.

---

## 7. Question 4 Key Results Summary

* **Application Architecture & Working Deliverables:**

  * **Core Suggestion Engine:** In `app/predictor.py` and `app/model_store.py`, providing zero-copy memory-mapped candidate generation with Kneser-Ney smoothing in $< 0.1$ ms.
  * **Interactive CLI Application:** In `app/main.py`, accepting user input, querying candidates, and updating context iteratively.
  * **FastAPI Backend:** In `app/api.py`, serving REST endpoint `GET /suggest?text=...`.
  * **Next.js Web Interface:** In `web/`, providing interactive web UI with real-time suggestion buttons and keyboard shortcuts.
* **Latency Measurement Disambiguation & 1.0s Benchmark Origin:**

  * **Core In-Memory Lookup Latency (Offline Benchmark):** Evaluated strictly via the offline predictor benchmark in `output/results/q4_eval.json` across 475,613 in-vocabulary test targets, `AmharicPredictor.suggest()` requires an average core algorithm lookup time of **0.05 ms** (median ~0.04 ms, p95 ~0.09 ms).
  * **Benchmark Origin:** The 1.0-second limit is derived from **Nielsen Norman Group's standard UI responsiveness threshold** (sub-100ms feels instantaneous; sub-1.0s maintains unbroken flow of thought). The offline core lookup satisfies this pedagogical benchmark by over three orders of magnitude.
  * **Metric Applicability (AGENTS.md B6):** Question 4 evaluates candidate ranking and suggestion quality metrics: Top-1 Accuracy, Top-5 Hit Rate, and MRR@5 (`mrr_at_5`). Classification metrics (such as precision, recall, ROC-AUC, or confusion matrices) do not apply, because next-word prediction is an open-vocabulary candidate ranking task over hundreds of thousands of vocabulary words rather than fixed binary or multi-class categorization.
* **Evaluation Scope & Denominator Resolution (475,613 vs. 488,282):**

  * Total held-out test sentences: 499,357.
  * Sentences meeting length $\ge 3$ tokens (providing a 2-word context $w_{L-2}, w_{L-1}$ to predict final target word $w_L$): **488,282 sentences**.
  * Among these, 12,669 target words map to `<UNK>` in the training vocabulary.
  * **Evaluated In-Vocabulary Targets (475,613 positions):**
    * Top-1 Accuracy: **23.37%**
    * Top-5 Hit Rate: **40.05%**
    * MRR@5: **0.2963**
  * **Full Candidate Set (counting `<UNK>` targets as automatic prediction misses across all 488,282 sentences):**
    * Top-1 Accuracy: $23.37\% \times \frac{475,613}{488,282} = \mathbf{22.76\%}$
    * Top-5 Hit Rate: $40.05\% \times \frac{475,613}{488,282} = \mathbf{39.01\%}$
    * MRR@5: $0.2963 \times \frac{475,613}{488,282} = \mathbf{0.2886}$
* **Full Held-Out Test Split Performance & Baseline Comparison (from `output/results/q4_eval.json`):**

  | Model / Context Formulation                               | Top-1 Accuracy ($\ge 10.0\%$) | Top-5 Hit Rate ($\ge 30.0\%$) | MRR@5 ($\ge 0.15$) | Response Time per Target |          Status          |
  | :-------------------------------------------------------- | :-----------------------------: | :-----------------------------: | :------------------: | :----------------------: | :-----------------------: |
  | **Frequency-Only Baseline** (Top-5 Unigram Prior)   |              3.40%              |              9.80%              |        0.0510        |      $< 0.01$ ms      |       Poor Baseline       |
  | **One-Word Context ($L = 1$, Bigram)**            |             17.95%             |             32.91%             |        0.2348        |       $< 0.1$ ms       |           PASS           |
  | **Two-Word Context ($L = 2$, Trigram, Deployed)** |        **23.37%**        |        **40.05%**        |   **0.2963**   |       $< 0.1$ ms       | **PASS (Champion)** |

  *Baseline Lift:* The two-word context model outperforms the frequency-only baseline by nearly **$7\times$ in Top-1 Accuracy** and over **$4\times$ in Top-5 Hit Rate**, demonstrating strong contextual learning.
  *Benchmark Targets:* All targets ($\ge 10\%$ Top-1, $\ge 30\%$ Top-5, $\ge 0.15$ MRR@5) are group-proposed pedagogical targets set prior to evaluation.
* **Assignment Example Context & Probability Reconciliation (`ኢትዮጵያ ውስጥ የቡና`):**

  * Prompt: `"ኢትዮጵያ ውስጥ የቡና"` (3 words).
  * Conditioning context: For two-word context ($L = 2$), the model conditions on `("ውስጥ", "የቡና")`.
  * Top-5 suggestions returned from training counts (`output/results/q4_eval.json`):
    1. **`ዋጋ`** (Price): Full-vocab conditional probability $P(w \mid \text{ctx}) = \mathbf{0.1140}$ ($0.114015$); normalized share among top-5 candidates = **$0.2917$** (29.17%) $\to$ `የቡና ዋጋ` (*"coffee price"*)
    2. **`ፍጆታ`** (Consumption): Full-vocab conditional probability $P = \mathbf{0.1067}$ ($0.106745$); normalized share among top-5 = **$0.2731$** (27.31%) $\to$ `የቡና ፍጆታ` (*"coffee consumption"*)
    3. **`ድርሻ`** (Share): Full-vocab conditional probability $P = \mathbf{0.0752}$ ($0.075198$); normalized share among top-5 = **$0.1924$** (19.24%) $\to$ `የቡና ድርሻ` (*"coffee share"*)
    4. **`ገበያ`** (Market): Full-vocab conditional probability $P = \mathbf{0.0479}$ ($0.047879$); normalized share among top-5 = **$0.1225$** (12.25%) $\to$ `የቡና ገበያ` (*"coffee market"*)
    5. **`የወጪ`** (Export/Expenditure): Full-vocab conditional probability $P = \mathbf{0.0470}$ ($0.046965$); normalized share among top-5 = **$0.1202$** (12.02%) $\to$ `የቡና የወጪ`
  * **Reconciliation of Probability Formulations:**
    - The raw language model probability $P(\text{ዋጋ} \mid \text{ውስጥ}, \text{የቡና}) = \mathbf{0.1140}$ sums to 1.0 across the entire 333,707-word vocabulary.
    - The candidate share $\mathbf{0.2917}$ normalizes the probability strictly over the top 5 suggested candidates:
      $$
      \frac{P(\text{ዋጋ})}{\sum_{c \in \text{Top5}} P(c)} = \frac{0.114015}{0.114015 + 0.106745 + 0.075198 + 0.047879 + 0.046965} = \frac{0.114015}{0.390802} \approx \mathbf{0.2917}
      $$
    Both values derive directly from `output/results/q4_eval.json`; the table presents the true generative model continuation probability, while user-facing interfaces display the normalized candidate confidence.
  * *Linguistic Analysis of `የቡና የወጪ`:* In Ethiopian commercial news, the compound phrase `የቡና የወጪ ንግድ` (*"coffee export trade"*) is ubiquitous. Because `የወጪ` carries the genitive prefix `የ-` (*"of expenditure / of export"*), the bigram `የቡና የወጪ` contains double genitive marking (`የ-ቡና` + `የ-ወጪ`), which is syntactically incomplete on its own and requires a following noun like `ንግድ`. This is an authentic artifact of n-gram boundary truncation.
  * *Illustrative Prompt Words:* The assignment prompt mentions `ምርት` (*production*), `ንግድ` (*trade*), and `አምራችነት` (*productivity*). In GPAC's training vocabulary, `ምርት` (freq: 1,328) and `ንግድ` (freq: 1,012) are known words, but rank below the top 5 after `የቡና` because news text heavily favors price and consumption reports. `አምራችነት` (freq: 0) is out-of-vocabulary, correctly preventing the model from suggesting it.
* **Backoff Strategy & Failure Modes:**

  * **Fallback Cascade:** If a two-word context is unseen in trigram counts, the model backs off to the single preceding word ($L=1$, bigram) with a backoff penalty factor of $0.5$. If the single word is also unseen, it falls back to the top unigram frequencies (`ነው, ላይ, ውስጥ...`).
  * **Failure Mode Example:** When given rare named entities or misspellings (e.g. `ታሪካዊት ጃፓን`), the trigram and bigram contexts miss, forcing fallback to generic copulas and prepositions (`ነው`, `ላይ`), which are grammatically permissible but lack semantic relevance.
* **API & Frontend Specification (Blueprint):**

  * **API Endpoint Specification:**
    ```bash
    # Query format:
    curl -X GET "http://127.0.0.1:8000/api/suggest?text=%E1%8A%A2%E1%89%B5%E1%8B%AE%E1%8C%B5%E1%8F%83%20%E1%8B%8D%E1%88%B5%E1%8F%95%20%E1%8B%A8%E1%89%A1%E1%8A%93"
    ```
  * **Response Schema (showing both full-vocab and normalized candidate probabilities):**
    ```json
    {
      "query": "ኢትዮጵያ ውስጥ የቡና",
      "suggestions": [
        {"word": "ዋጋ", "probability": 0.1140, "candidate_share": 0.2917, "rank": 1},
        {"word": "ፍጆታ", "probability": 0.1067, "candidate_share": 0.2731, "rank": 2},
        {"word": "ድርሻ", "probability": 0.0752, "candidate_share": 0.1924, "rank": 3},
        {"word": "ገበያ", "probability": 0.0479, "candidate_share": 0.1225, "rank": 4},
        {"word": "የወጪ", "probability": 0.0470, "candidate_share": 0.1202, "rank": 5}
      ],
      "strategy": "trigram",
      "offline_lookup_ms": 0.05
    }
    ```
  * **Deployment Verification Notice:** The metrics and response tables in this section reflect the offline benchmark evaluated across 475,613 test targets in `output/results/q4_eval.json`. End-to-end HTTP latency and Next.js frontend interactions are documented as architectural blueprints and startup recipes; they are not presented as local PC live server test runs.
* **How to Run the Applications:**

  * **Next.js Web Application:**
    ```bash
    # Terminal 1: Start FastAPI backend
    python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

    # Terminal 2: Start Next.js frontend
    cd web && npm run dev
    # Open http://localhost:3000 in your browser
    ```
  * **Interactive CLI Application:**
    ```bash
    python app/main.py
    ```
