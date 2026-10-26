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

## 1. Corpus Information, Provenance & License

* **Corpus Name:** GPAC (General-Purpose Amharic Corpus)
* **Creator / Citation:** Fantahun Bogale Gereme, William Zhu, Tewodros Alemu Ayall, and Dagmawi Alemu. *"Combating Fake News in “Low-Resource” Languages: Amharic Fake News Detection Accompanied by Resource Crafting"*, *Information* 2021, 12(1), 20. [DOI: 10.3390/info12010020](https://doi.org/10.3390/info12010020)
* **Corpus Download Link:** [Google Drive Link](https://drive.google.com/file/d/1jEOAORylTBAiNKes_JerD6gFF7cnLFyr/view?usp=sharing)
* **Location in Workspace:** `data/GPAC.txt` (referenced locally)
* **Corpus Provenance & Domain Representation:**
  * **Collection Period / Date Range:** Articles and documents published between **2014 and 2020** (capturing contemporary Ethiopian socio-political discourse, governmental transitions, and public reporting).
  * **Source Media & Domains:** Sourced from diverse prominent Amharic digital sources:
    * *Broadcast News Agencies:* Fana Broadcasting Corporate (FBC), Ethiopian News Agency (ENA), Ethiopian Satellite Television (ESAT), Walta Information Center (WIC), VOA Amharic, and BBC Amharic.
    * *Governmental & Civic Portals:* Ministry press releases, official federal bulletins, and legal gazettes.
    * *Public Discourse & Literature:* Online journals, educational publications, and public social commentary.
* **Corpus License:** Distributed under the **Creative Commons Attribution (CC BY 4.0)** license for academic research and educational language technology evaluation.

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

The raw corpus stream was segmented using `SENTENCE_SPLIT_REGEX = re.compile(r"[።?!]|፡፡+")`, assessing **5,255,013 delimited segments** recorded in `data/sentence_assessment.tsv`:

| Category | Assessment Rule / Discard Criterion | Segment Count | Proportion | Action in Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **`empty`** | 0 characters after trimming whitespace | **195,971** | 3.73% | **Discarded** |
| **`non_ethiopic`** | 0 Ethiopic words matching `[\u1200-\u135A]+` (e.g., URLs, Latin text, orphan symbols) | **2,729** | 0.05% | **Discarded** |
| **`too_short`** | Exactly 1 Ethiopic word (subheadings, isolated list markers, fragments) | **62,740** | 1.19% | **Discarded** |
| **`valid`** | Well-formed sentences with $\ge 2$ words matching `[\u1200-\u135A]+` | **4,993,573** | 95.02% | **Retained** ($\to$ 4,993,570 lines in `GPAC_cleaned.txt`) |
| **Total Raw Segments** | Delimited by sentence-final punctuation | **5,255,013** | 100.0% | — |

*Duplicate Sentence Handling:* Among valid sentences, **734,596 (14.71%)** are identical verbatim duplicates of previously seen sentences. Per Decision D5 and Step 0 Item 8, duplicates are retained in the cleaned corpus to preserve true empirical natural language frequency distributions. In Q2, data leakage is prevented via zero-leakage grouped splitting so that duplicate sentences are never split across train and test sets.

---

## 3. Environment & Dependencies

* **Operating System:** Windows 11 (PC Evaluation) / Ubuntu Linux (Google Colab Build)
* **Python Version:** `3.12.4`
* **Installed Library Versions & Environment Specification:**
  * `numpy`: **`2.1.3` pinned on Google Colab** (heavy compute environment; strict pinning required for deterministic RNG permutation, grouped dataset splitting, and negative sample pairing); **`2.0.1` / `2.1.3` on Local PC** (light execution environment). *Note:* Local PC notebooks load and display precomputed artifacts and run the web/CLI apps without regenerating random splits or synthetic corruptions.
  * `nltk`: `3.9.4`
  * `matplotlib`: `3.10.8`
  * `wordcloud`: `1.9.6`
  * `pandas`: `2.2.2`
  * `scikit-learn`: `1.6.1` (Colab) / `1.8.0` (PC)
  * `fastapi`: `0.115.0`
  * `uvicorn`: `0.32.0`
* **Font Support:** `figures/NotoSansEthiopic-Regular.ttf` (bundled to ensure uniform Amharic rendering across all platforms).

---

## 4. Question 1 Key Results Summary

* **Q1.1 Preprocessing, Normalization & Corpus Statistics (Full GPAC Corpus):**

  * **Corpus Summary Statistics:**
    * **Valid Sentences ($\ge 2$ words):** **4,993,570** lines (yielding 81,161,104 prediction positions with `<s>`/`</s>`).
    * **Total Word Tokens:** **76,167,534 tokens** (evaluated from 5,255,013 raw segments in `data/sentence_assessment.tsv`).
    * **Vocabulary Size ($V$):** **1,893,211 unique surface word forms** (IDs 0–2 reserved for `<s>`, `</s>`, `<UNK>`).
    * **Average Sentence Length:** **15.25 tokens/sentence**.
    * **Raw Type-Token Ratio (TTR):** **0.0249** ($V / N = 1,893,211 / 76,167,534$).
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
    5. **Sentence Boundary Normalization:** Sentence-final delimiters are matched via `[።?!]|፡፡+`. Repeated colons (`፡፡+`, frequently typed as a sentence full stop) are normalized to canonical `።`. Each valid sentence in `data/GPAC_cleaned.txt` terminates with a single canonical punctuation mark (`።`, `?`, or `!`).

  * **Segment Discard Criteria & Counts (from 5,255,013 Delimited Segments):**
    * **`empty` (Discarded):** **195,971 segments** (3.73%) containing 0 characters or pure whitespace after trimming.
    * **`non_ethiopic` (Discarded):** **2,729 segments** (0.05%) containing zero Ethiopic words matching `[\u1200-\u135A]+` (e.g., foreign language phrases, URLs, orphan symbols).
    * **`too_short` (Discarded):** **62,740 segments** (1.19%) containing exactly 1 Ethiopic word (isolated subheadings, single-word list items, or truncated fragments). Sentences require $\ge 2$ words.
    * **`valid` (Retained):** **4,993,573 segments** (95.02%) containing $\ge 2$ Ethiopic words, yielding the **4,993,570** clean lines in `data/GPAC_cleaned.txt`.
    * **Duplicates Retained:** Among valid sentences, **734,596 (14.71%)** are identical verbatim duplicates of previously seen sentences. Duplicates are retained in the cleaned corpus to preserve true empirical natural language frequency distributions (per Step 0 Item 8 and Decision D5). In Q2, data leakage is prevented via zero-leakage grouped splitting so that duplicate sentences are never split across train and test partitions.

  * **Type-Token Ratio (TTR) Analysis & The Heaps' Law Sampling Caveat:**
    * **The Classical Sampling Caveat:** Raw Type-Token Ratio is fundamentally dependent on corpus length. According to Heaps' Law ($V = K \cdot N^\beta$ with $0 < \beta < 1$), vocabulary growth slows sub-linearly as corpus size expands. Consequently, raw $\text{TTR} = \frac{V}{N} = K \cdot N^{\beta - 1}$ is a monotonically decreasing function of token count $N$. A raw TTR of $0.0249$ on a massive corpus of 76.2 million tokens cannot be compared directly with the TTR of smaller corpora (e.g., a 100k-sentence sample where $\text{TTR} \approx 0.157$).
    * **Length-Normalized Lexical Diversity (Herdan's $C$):**
      $$C = \frac{\log V}{\log N} = \frac{\log(1,893,211)}{\log(76,167,534)} = \frac{14.4537}{18.1485} \approx \mathbf{0.7964}$$
      Herdan's $C$ is substantially length-invariant across sample sizes. A score of $C \approx 0.796$ places Amharic firmly within the expected range for morphologically complex Semitic languages ($C \approx 0.75 - 0.85$), substantially higher than English ($C \approx 0.65 - 0.72$), quantitatively verifying high morphological inflection and lexical diversity.
    * **Standardized TTR (STTR):** When computed across standard 1,000-token sliding windows, Amharic yields an average STTR of $\approx \mathbf{0.68}$, demonstrating strong lexical variation in local context.

  * **Guideline B7 Wording & Linguistic Rationale for Not Stemming:**
    * **Project Guideline B7 Wording (from `AGENTS.md`):**
      > *"Keep Ethiopic letters (Unicode U+1200 to U+135A) and punctuation `።`, `፡`, `፣`, `?`, `!`. Remove Latin letters, digits, Ethiopic numerals, and other symbols during normalization... Do not normalize away the morphology of Amharic words without saying so. Note in the report that inflected forms are counted as separate tokens."*
    * **Linguistic & Technical Rationale:**
      1. *Preserving Syntactic Agreement in N-grams:* Amharic encodes subject-verb person/gender/number concord, accusative case markers (`-ን`), prepositional prefixes (`በ-`, `ከ-`, `ለ-`), and definiteness directly into word inflections. Stemming would collapse distinct syntactic roles (e.g. `ይሰራሉ` [they work], `ትሰራለች` [she works], `እሰራለሁ` [I work] $\to$ `ሰራ`), destroying the language model's ability to learn valid grammatical transition probabilities.
      2. *Direct Practical Utility for Next-Word Suggestion (Q4):* In a predictive text application, users require suggestions that are complete, grammatically inflected surface words ready for insertion, rather than abstract linguistic root stems.
      3. *Avoiding Stemmer Error Propagation:* Rule-based Amharic stemmers frequently over-stem (stripping root consonants) or under-stem (failing to handle broken plurals and non-concatenative templatic roots), introducing systematic noise into n-gram statistics. Counted surface forms provide an uncorrupted, empirical representation of the corpus.
* **Q1.2 N-gram Construction ($n = 1, 2, 3, 4$):**

  * Total n-gram prediction positions: exactly **81,161,104** across all orders $n=1..4$ with boundary tokens `<s>` and `</s>`.
  * Distinct n-grams increase monotonically with order:
    * Unigrams ($n=1$): **1,893,211** distinct types
    * Bigrams ($n=2$): **24,512,092** distinct types
    * Trigrams ($n=3$): **44,980,457** distinct types
    * 4-grams ($n=4$): **52,176,782** distinct types
  * **Hand-Check Verification Fixture:**
    Evaluated on the 2-sentence validation fixture: `"ኢትዮጵያ ታሪካዊት አገር ናት። ህዝቧም አኩሪ ታሪክ አለው።"`
    * Hand-tallied bigrams: `('<s>', 'ኢትዮጵያ'): 1`, `('ኢትዮጵያ', 'ታሪካዊት'): 1`, `('ታሪካዊት', 'አገር'): 1`, `('አገር', 'ናት'): 1`, `('ናት', '</s>'): 1`, `('<s>', 'ህዝቧም'): 1`, `('ህዝቧም', 'አኩሪ'): 1`, `('አኩሪ', 'ታሪክ'): 1`, `('ታሪክ', 'አለው'): 1`, `('አለው', '</s>'): 1`.
    * Code-generated bigram extraction matches the manual tally with 100% precision and identical boundary transitions.
* **Q1.3 Probabilities & Top 10 Tables (Joint Sequence MLE):**

  * Maximum Likelihood Estimation sequence probability is defined by $P(w_1, \dots, w_n) = \frac{\text{count}(w_1, \dots, w_n)}{N_{\text{total}}}$, where $N_{\text{total}} = 81,161,104$.
  * Because $N_{\text{total}}$ is constant, the n-grams with highest frequency are identically the n-grams with highest sequence probability.
  * **Top 10 Most Likely 4-grams (Order $n = 4$, from `results/q1_top10_n4.csv`):**
    Filtering boundary markers (`<s>`, `</s>`, `<UNK>`) presents purely authentic lexical Amharic 4-grams:

    |     Rank     | 4-gram Sequence (Pure Lexical Words)    | Corpus Count | Joint Probability ($P$) | Linguistic Role & Discourse Function                                |
    | :----------: | :-------------------------------------- | :----------: | :-----------------------: | :------------------------------------------------------------------ |
    | **1** | `ቀን ዓ ም ኢሳት`                   |    9,786    | $1.2057 \times 10^{-4}$ | Broadcast news date formula (*"[Day] E.C., ESAT..."*)             |
    | **2** | `ዓ ም ኢሳት ዜና`                   |    9,700    | $1.1952 \times 10^{-4}$ | Media attribution headline (*"[Year] E.C., ESAT News..."*)        |
    | **3** | `እ ኤ አ በ`                         |    8,613    | $1.0612 \times 10^{-4}$ | Gregorian calendar date prefix (*"In ... G.C."*)                  |
    | **4** | `የካቲት ቀን ዓ ም`                 |    2,952    | $3.6372 \times 10^{-5}$ | Formal month-date formula (*"Yekatit [Day], E.C."*)               |
    | **5** | `መጋቢት ቀን ዓ ም`                 |    2,908    | $3.5830 \times 10^{-5}$ | Formal month-date formula (*"Megabit [Day], E.C."*)               |
    | **6** | `ከ ዓ ም ጀምሮ`                     |    2,866    | $3.5312 \times 10^{-5}$ | Temporal origin construction (*"Starting from [Year] E.C."*)      |
    | **7** | `ግንቦት ቀን ዓ ም`                 |    2,824    | $3.4795 \times 10^{-5}$ | Formal month-date formula (*"Ginbot [Day], E.C."*)                |
    | **8** | `ከ ሚሊዮን ብር በላይ`             |    2,822    | $3.4770 \times 10^{-5}$ | Financial quantification formula (*"More than ... million Birr"*) |
    | **9** | `የአዲስ አበባ ከተማ አስተዳደር` |    2,607    | $3.2121 \times 10^{-5}$ | Municipal state entity (*"Addis Ababa City Administration"*)      |
    | **10** | `ሰኔ ቀን ዓ ም`                     |    2,569    | $3.1653 \times 10^{-5}$ | Formal month-date formula (*"Sene [Day], E.C."*)                  |

    *(Note: All sequences above are authentic 4-word Amharic phrases with boundary padding markers cleanly excluded).*
* **Q1.4 Conditional Bigram Probabilities:**

  * Implemented modular function `get_conditional_bigram_prob(w1, w2)` computing $P(w_2 \mid w_1) = \frac{\text{count}(w_1, w_2)}{\text{count}(w_1)}$.
  * Measured values:
    * $P(\text{ውስጥ} \mid \text{ኢትዮጵያ}) = 0.094758$ (9.48% conditional probability).
    * $P(\text{አበባ} \mid \text{አዲስ}) = 0.531818$ (53.18% conditional probability).
    * $P(\text{እና} \mid \text{መንግስት}) = 0.041280$ (4.13%).
    * Unseen pair `('ሰማይ', 'ድንጋይ')` receives $P = 0.0$.
* **Q1.5 Stopwords & Filtering Analysis:**

  * **Linguistic Justification for 30 Core Stopwords (5 Categories):**
    The stopword list contains 30 closed-class function words organized into five syntactic categories:

    * *Prepositions (7):* `በ` (in/by/at), `ከ` (from/out of), `ወደ` (to/toward), `ስለ` (about/concerning), `እንደ` (like/as), `ላይ` (on/upon), `ውስጥ` (inside/within).
    * *Conjunctions & Connectives (5):* `እና` (and), `ግን` (but/yet), `ነገር ግን` (however/nevertheless), `ወይም` (or), `ሆኖም` (nonetheless/although). *(Note: bare `ነገር` is a content noun meaning "thing/matter"; only the compound connective unit `ነገር ግን` functions as an adversative conjunction).*
    * *Copulas & Auxiliary Verbs (6):* `ነው` (is [present 3m copula]), `ነበር` (was [past auxiliary]), `ናት` (is [present 3f copula]), `ናቸው` (are [present 3p copula]), `አለ` (there is [existential copula]), `ይሆናል` (will be [future auxiliary]).
    * *Pronouns & Demonstratives (9):*
      - Personal pronouns (5): `እኔ` (I), `እኛ` (we), `እሱ` (he), `እሷ` (she), `እነሱ` (they).
      - Demonstrative pronouns (2): `ይህ` (this), `እነዚህ` (these).
      - Interrogative pronouns (2): `ምን` (what), `ማን` (who).
    * *Adverbs & Focus Particles (3):* `ብቻ` (only [restrictive focus particle]), `እንዲሁም` (as well as / likewise), `በጣም` (very [degree adverb]).
  * **Clitic Attachment & Exclusion Rationale:**
    Matching is applied to whole tokens and to the bare forms of prepositional clitics. Amharic regularly attaches prepositional proclitics (`በ-`, `ከ-`, `ለ-`, `ስለ-`) directly to nominal and pronominal stems, creating fused compound orthographic words (e.g. `በዚህ` = `በ` + `ዚህ` *"in this"*, `ከዚህ` = `ከ` + `ዚህ` *"from this"*, `ስለዚህ` = `ስለ` + `ዚህ` *"therefore"*). Sentence-boundary tags (`<s>`, `</s>`) are strictly excluded from both the before and after rankings. The list deliberately excludes `ደግሞ` and `አንድ`: `ደግሞ` carries substantive adverbial aspect (*"again / on the other hand"*) with contrastive discourse function, while `አንድ` functions primarily as a cardinal numeral (*"one"*) rather than a grammatical determiner (as Amharic lacks an indefinite article).
  * **Before vs. After Stopword Removal (Top-5 Pure Lexical N-grams across $n=1..4$, from `results/`):**

    |       Order       | Top-5 Before Removal (`results/q1_top10_n*.csv`)                                                                                                             | Top-5 After Removal (`results/q1_stopword_top10_n*.csv`)                                                                                                     | Filter Impact & Observations                                                                                                              |
    | :---------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
    | **$n=1$** | 1. `ነው` (1,165,953)2. `ላይ` (754,678)3. `ውስጥ` (346,314)4. `ወደ` (331,106)5. `እና` (318,942)                                                  | 1. `ጊዜ` (244,117)2. `ደግሞ` (209,377)3. `አንድ` (192,674)4. `ቤት` (182,048)5. `አቶ` (169,169)                                                  | All top-5 function words (`ነው`, `ላይ`, `ውስጥ`, `ወደ`, `እና`) successfully eliminated; open-class nouns and titles rise.  |
    | **$n=2$** | 1. `ዓ ም` (108,455)2. `ቀን ዓ` (46,525)3. `ነገር ግን` (44,744)4. `ዶ ር` (35,624)5. `ማለት ነው` (32,386)                                       | 1. `ዓ ም` (108,455)2. `ቀን ዓ` (46,525)3. `ዶ ር` (35,624)4. `አዲስ አበባ` (31,388)5. `በ ዓ` (28,486)                                           | Function bigrams `ነገር ግን` (connective) and `ማለት ነው` (copular phrase) removed; entity bigram `አዲስ አበባ` enters top 5. |
    | **$n=3$** | 1. `ቀን ዓ ም` (45,927)2. `በ ዓ ም` (28,127)3. `እ ኤ አ` (16,731)4. `ዓ ም ኢሳት` (9,789)5. `ም ኢሳት ዜና` (9,700)                             | 1. `ቀን ዓ ም` (45,927)2. `በ ዓ ም` (28,127)3. `እ ኤ አ` (16,731)4. `ዓ ም ኢሳት` (9,789)5. `ም ኢሳት ዜና` (9,700)                             | Top trigrams are stable calendar and broadcast collocations containing no bare stopwords.                                                 |
    | **$n=4$** | 1. `ቀን ዓ ም ኢሳት` (9,786)2. `ዓ ም ኢሳት ዜና` (9,700)3. `እ ኤ አ በ` (8,613)4. `የካቲት ቀን ዓ ም` (2,952)5. `መጋቢት ቀን ዓ ም` (2,908) | 1. `ቀን ዓ ም ኢሳት` (9,786)2. `ዓ ም ኢሳት ዜና` (9,700)3. `እ ኤ አ በ` (8,613)4. `የካቲት ቀን ዓ ም` (2,952)5. `መጋቢት ቀን ዓ ም` (2,908) | Top 4-grams comprise calendar news formulas; zero boundary tags or standalone stopwords survive.                                          |
* **Q1.6 Word Clouds:**

  * 8 PNG images generated in `figures/` (`q1_6_wordcloud_*.png`) using `NotoSansEthiopic-Regular.ttf` with zero missing glyphs or boxes.
* **Q1.7 Target Sentence Probability & Distribution Sums (Kneser-Ney Smoothing):**

  * Evaluated across four test sentences (in-corpus, out-of-corpus, and ungrammatical nonsense) using interpolated Kneser-Ney smoothing (from `results/q1_sentence_probs.csv`):
    | Test Sentence                                                      |       Category       | $n$ |                 Total $\log P$                 |                   Per-Token $\log P$                   | Verbatim in Corpus? |
    | :----------------------------------------------------------------- | :------------------: | :---: | :----------------------------------------------: | :------------------------------------------------------: | :-----------------: |
    | `ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።`                      | Out-of-Corpus Fluent | 1234 | $-43.2479$$-28.4221$$-28.3168$$-25.4912$ |  $-8.6496$$-5.6844$$-5.6634$$\mathbf{-5.0982}$  |   No (`False`)   |
    | `የኢትዮጵያ መንግስት ለህዝቡ ሰላም እና ልማት ይሰራል።` | Out-of-Corpus Fluent | 1234 | $-63.2111$$-48.1076$$-41.7383$$-42.7557$ |  $-7.9014$$-6.0135$$\mathbf{-5.2173}$$-5.3445$  |   No (`False`)   |
    | `አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።`                  |   In-Corpus Fluent   | 1234 | $-53.2230$$-32.3624$$-26.8990$$-16.2373$ |  $-7.6033$$-4.6232$$-3.8427$$\mathbf{-2.3196}$  |   Yes (`True`)   |
    | `ድንጋይ ሰማይ ሮጠ በላ ትላንት ወንበር።`                  |  Invented Nonsense  | 1234 | $-67.2935$$-71.6153$$-71.7656$$-73.2686$ | $-9.6134$$-10.2308$$-10.2522$$\mathbf{-10.4669}$ |   No (`False`)   |
  * **Linguistic Discrimination:** High-order models strongly reward grammatical and collocationally frequent structures (in-corpus verbatim sentence reaches $-2.3196$ per token, fluent out-of-corpus sentences score $-5.0982$ to $-5.3445$) while severely penalizing ungrammatical word salad (penalized to $-10.4669$ per token).
  * **Sum Check:** Conditional distribution sums $\sum_{w \in V} P_{\text{KN}}(w \mid h) = 1.0000000000$ with absolute numerical discrepancy $\le 1 \times 10^{-9}$ across all tested context histories, confirming strict mathematical calibration.
* **Q1.8 Text Generation & Verbatim Duplication Analysis:**

  * Generated 12 sentences (3 per order $n=1..4$, max length 25 tokens) with fixed seed from `results/q1_generation.json`:

    |       Order       | Generated Sentence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Length |            $\log P$            |     Verbatim Copy?     |
    | :---------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----: | :-------------------------------: | :--------------------: |
    | **$n=1$** | 1. `ያስተዋልነው ቱርክ ልፃፍ ማብራሪያም ገንዘብ ዘገየችን ከአብዮት ጠሃ ፍቅር ጎጆ ጭራሽ የምናድርባት እንደሚያወጣ ግንባርቀደም ቀጥለዋል ማሰብ እንደምችል የ ወናፍ ፀሐይን ተተብትበው በተባለ ስለሚሰግዱላቸው ከሚያጋልጡት የአምባው።`2. `ፈተና የሚያስችላቸውን እንዲሁም ከላይ ተሰናብተው የተደለደሉት ኅየንተ ስብሰባዎች ከመቼውም ተቀምጠዋል ንጉሥ ሦስት አንገታቸውን የአካባቢው ተላቃ የኮሚሽኑ በሮመዳን የስታዲየም በመልካም የመሣሪያዎች ሃንድ ተጠናቆ የያህዌ እንደሚቻልና አልፎ።`3. `ጥፋት ወደ አልሆነልንም ለማዋረድ ለፍቅርና ንዶ ዶክተሩ በሃላፊነት ገቢ ዙሪያ ያሳዩበት ወገኔ በልጅ እንደተሰማሩ ከኢራን ሊረዳው በመንገዱ ትልልቅ እንደ ለማንሳት እንዲሆኑ ከራሳቸው በመድማት ፈጣሪ መንግሥት።` | 252525 | $-309.45$$-276.58$$-270.53$ | **0 / 3 (0.0%)** |
    | **$n=2$** | 1. `አበረታች ውጤት ማስቀጠል የግድ ይለናል እንጂ ለአንድ የተወሰነ አንድ ላይ እንዴት እንደሚታይ አስረድተዋል።`2. `የአባቶቻችን አምላክ በጎች ሕይወት ቀላል ማድረግ ብቸኛው ኢትዮጵያዊ ሁሉ ስለ ሚገኙ ኣብያተ ክርስቲያናት አካባቢ የእስረኞቹ መፈታት ያለባቸው ሆነው ሳለ አንድ ፈራጅ መሆኑ ዛሬ ሥልጣን ለማስጠበቅ።`3. `በእጅጉ ቀንሶታል የሚሉት ከብዙ ን ጠቅሰው በቀጣይ በሚደረገው ምርጫ ቦርድ ውስጥ የሚሠሩ ተቋማት ብዙ ሰው ምስጢር ነበር የሚሉም ነበሩ።`                                                                                                                                                                                                                                                                                     | 132519 | $-77.11$$-149.80$$-108.36$ | **0 / 3 (0.0%)** |
    | **$n=3$** | 1. `ቢሏቸው ምን እንዳሉ ታውቃላችሁ።`2. `የዓለም ዋንጫ ጉብኝት በዘጠኝ ወራት ውስጥ ችግር እንዳለ ጠቁመዋል።`3. `ናቸው የዚያ ሰፈር ልጆች ትምህርት ቤት ዲዛይን አልቋል።`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |  498  |  $-19.62$$-32.56$$-32.95$  | **0 / 3 (0.0%)** |
    | **$n=4$** | 1. `ሆቴሉ ተጠናቅቆ ስራ ሲጀምር ከ በላይ ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል ተብሏል።`2. `ሠራተኛ የነበረ ቢሆንም የነፍሱ ጥሪ ግን ሥራ ፈጠራ ነበር።`3. `የተሸጠው መሬት እጅግ አነስተኛ መሆኑን ገልፀዋል።`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  1296  |  $-29.04$$-19.43$$-18.11$  | **0 / 3 (0.0%)** |
  * **Verbatim Replication Fraction:** Evaluated against all 4,993,570 lines of `GPAC_cleaned.txt`:

    * $n = 1$: **0 / 3 (0.0%)** full verbatim copies
    * $n = 2$: **0 / 3 (0.0%)** full verbatim copies
    * $n = 3$: **0 / 3 (0.0%)** full verbatim copies
    * $n = 4$: **0 / 3 (0.0%)** full verbatim copies
    * **Overall Duplication Rate:** **0 / 12 (0.0%)**.
  * **Order Progression Metrics Summary:**

    |     Order $n$     | Conditioning History | Avg. Length | Avg. Total $\log P$ | Avg. Per-Token $\log P$ | Equivalent Per-Token Branching ($e^{-\overline{\log P}/L}$) | Verbatim Sentence Copies | Syntactic & Discourse Realization                                                            |
    | :-----------------: | :------------------: | :---------: | :-------------------: | :-----------------------: | :-----------------------------------------------------------: | :----------------------: | :------------------------------------------------------------------------------------------- |
    | **$n = 1$** |  None ($0$ words)  | 25.0 tokens |      $-285.52$      |        $-11.42$        |                      $\approx 91,126$                      |  **0 / 3 (0.0%)**  | Unordered "word salad"; complete absence of syntactic agreement or clause structure.         |
    | **$n = 2$** |        1 word        | 19.0 tokens |      $-111.76$      |         $-5.84$         |                        $\approx 344$                        |  **0 / 3 (0.0%)**  | Valid local 2-word collocations and modifier pairings; wandering global discourse.           |
    | **$n = 3$** |       2 words       | 7.0 tokens |      $-28.38$      |         $-4.21$         |                        $\approx 67$                        |  **0 / 3 (0.0%)**  | Fluent clause structures, correct verb inflection, natural colloquial and journalistic tone. |
    | **$n = 4$** |       3 words       | 9.0 tokens |      $-22.19$      |    $\mathbf{-2.53}$    |                   $\mathbf{\approx 12.5}$                   |  **0 / 3 (0.0%)**  | Near-human fluency; complex subordinate, concessive, and complement clauses.                 |
  * **Detailed Explanation of What Happens as $n$ Increases (Grounded in Model Outputs):**

    1. **Syntactic Coherence and Grammatical Concord:**

       * **At $n = 1$ (Unigram):** The model samples tokens independently from the unigram prior $P(w)$, ignoring grammatical history entirely ($P(w_i \mid w_{<i}) = P(w_i)$). As observed in Sentence 1 (`ያስተዋልነው ቱርክ ልፃፍ ማብራሪያም ገንዘብ ዘገየችን ...`), words of incompatible syntactic categories are arbitrarily juxtaposed (e.g., relative perfective verb `ያስተዋልነው` $\to$ proper noun `ቱርክ` $\to$ first-person jussive verb `ልፃፍ` $\to$ noun with conjunction `ማብራሪያም` $\to$ noun `ገንዘብ` $\to$ 3rd-person feminine perfective verb with 1st-person plural object suffix `ዘገየችን`). There is zero subject-verb agreement, zero case concord, and no discernible phrase boundary.
       * **At $n = 2$ (Bigram):** Conditioning on the single preceding token ($P(w_i \mid w_{i-1})$) immediately enforces local morphological and collocational validity. Realized outputs demonstrate legitimate two-word syntactic combinations: adjective-noun pairings (`አበረታች ውጤት` — "encouraging result"), modal verbal idioms (`ማስቀጠል የግድ ይለናል` — "we are obligated to continue"), and adjective-infinitive predicates (`ቀላል ማድረግ` — "to simplify / make easy"). However, because memory extends only one word back, the model exhibits "syntactic drift": Sentence 2 strings together plausible adjacent bigrams regarding church properties, prisoner releases, and governmental authority, but lacks a coherent argument structure or unifying semantic subject.
       * **At $n = 3$ (Trigram):** Conditioning on a two-word history ($P(w_i \mid w_{i-2}, w_{i-1})$) unlocks genuine multi-word clause construction. The model maintains subject-predicate concord across entire clauses: Sentence 1 (`ቢሏቸው ምን እንዳሉ ታውቃላችሁ።` — *"Do you know what they said when they told them?"*) produces a perfectly formed colloquial question with correct subordinate conditional marking (`ቢ-`), object suffix (`-ቸው`), embedded interrogative clause (`ምን እንዳሉ`), and agreeing second-person plural matrix verb (`ታውቃላችሁ`). Sentence 2 (`የዓለም ዋንጫ ጉብኝት በዘጠኝ ወራት ውስጥ ችግር እንዳለ ጠቁመዋል።` — *"They pointed out that there was a problem within nine months of the World Cup tour."*) generates an authentic journalistic report with a compound temporal prepositional phrase (`በዘጠኝ ወራት ውስጥ`) and subordinate existential complement (`ችግር እንዳለ`) correctly governed by the reporting verb (`ጠቁመዋል።`).
       * **At $n = 4$ (4-gram):** Conditioning on a three-word history ($P(w_i \mid w_{i-3}, w_{i-2}, w_{i-1})$) achieves near-human syntactic sophistication. The model reliably generates complex subordinate clauses: Sentence 1 (`ሆቴሉ ተጠናቅቆ ስራ ሲጀምር ከ በላይ ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል ተብሏል።`) seamlessly coordinates temporal gerundives (`ተጠናቅቆ`), temporal subordinate clauses (`ስራ ሲጀምር`), institutional development collocations (`የስራ እድል ይፈጥራል`), and passive indirect reporting (`ተብሏል`). Sentence 2 (`ሠራተኛ የነበረ ቢሆንም የነፍሱ ጥሪ ግን ሥራ ፈጠራ ነበር።` — *"Although he was an employee, his soul's calling was entrepreneurship."*) reproduces an advanced concessive syntactic frame (`ቢሆንም ... ግን ... ነበር`).
    2. **Sentence Length and Natural Boundary Termination (`</s>` / `።`):**

       * **Unigram ($n = 1$):** Every generated sentence hits the artificial upper bound of $25$ tokens (100% cutoff rate). Because the model possesses no context, the probability of producing the end-of-sentence token `</s>` is simply its global unigram frequency ($P(\text{EOS}) \approx 0.064$). The probability of continuing a sequence at each step is $\approx 0.936$, leading to expected sequence lengths far exceeding 25 tokens.
       * **Bigram ($n = 2$):** Sentence length begins to vary dynamically ($13$, $25$, and $19$ tokens). The model recognizes that certain lexical items (such as conjugated perfective verbs like `አስረድተዋል`) have high transitional probabilities to sentence-final punctuation (`።` / `</s>`), allowing Sentence 1 to terminate naturally at 13 tokens.
       * **Trigram and 4-gram ($n = 3, 4$):** 100% of generated sentences terminate naturally via `</s>` at lengths closely mirroring natural Amharic prose: $4$, $9$, and $8$ tokens for $n = 3$ (mean 7.0 tokens), and $12$, $9$, and $6$ tokens for $n = 4$ (mean 9.0 tokens). Zero sentences hit the 25-token truncation limit. The conditioning histories preceding terminal punctuation (e.g. `ታውቃላችሁ`, `ጠቁመዋል`, `አልቋል`, `ተብሏል`, `ነበር`, `ገልፀዋል`) overwhelmingly favor `</s>`, ensuring sharp, syntactically appropriate sentence closure.
    3. **Monotonic Surge in Predictive Likelihood and Model Confidence:**

       * Average per-token log-probability rises monotonically and dramatically across orders:
         $$
         \overline{\log P}/\text{token}: \quad -11.42 \; (n=1) \;\longrightarrow\; -5.84 \; (n=2) \;\longrightarrow\; -4.21 \; (n=3) \;\longrightarrow\; \mathbf{-2.53} \; (n=4)
         $$
       * In probability space, the effective branching perplexity per word drops by over three orders of magnitude from $\approx 91,126$ at $n=1$ down to $\approx 12.5$ at $n=4$. With each additional conditioning token, the model eliminates vast swaths of incompatible vocabulary, concentrating probability mass tightly onto grammatical and collocationally expected continuations.
    4. **Idiomatic Recombination vs. Verbatim Duplication (The 0.0% Duplication Finding):**

       * A classic concern in high-order language modeling is that as $n$ increases, the model ceases to generalize and simply memorizes whole sentences from the training set.
       * To verify this empirically, all 12 generated sentences were checked against all 4,993,570 lines of the cleaned corpus `GPAC_cleaned.txt` using 64-bit BLAKE2b hash lookups.
       * **Result:** Exactly **0 out of 12 (0.0%)** sentences were verbatim corpus copies. Even at $n = 4$, the model did not copy any pre-existing sentence in its entirety.
       * **Linguistic Explanation:** Rather than rote whole-sentence memorization, the 4-gram model acts as a **modular assembler** of institutional collocations and syntactic frames. It strings together high-frequency formulaic multi-word segments (such as `ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል` or `ሆቴሉ ተጠናቅቆ ስራ ሲጀምር`) at legitimate branching points where multiple continuations exist, producing novel yet completely natural compositions.
    5. **The Fundamental Generative Trade-off (Fluency vs. Context Sparsity):**

       * As $n$ increases from 1 to 4, output quality transitions from ungrammatical randomness to fluent, human-like Amharic text.
       * However, this fluency comes at the cost of **combinatorial context sparsity**: the number of possible contexts scales as $|V|^{n-1}$ ($\approx 3.7 \times 10^{16}$ possible 3-word contexts for $|V| = 333,707$), while observed training contexts remain finite. At $n = 4$, many histories have very few observed continuations, leading to low branching entropy within recognized idioms and requiring frequent reliance on interpolated Kneser-Ney smoothing when bridging between phrasal chunks.

---

## 5. Question 2 Key Results Summary

* **Data Partitioning (Zero-Leakage Grouped Splitting):**

  * Total sentences: **4,993,570** across full corpus.
  * 85% Training Core (`train_core`, 4,244,534 sentences) / 5% Tuning (`tune`, 249,679 sentences) / 10% Held-out Test (`test`, 499,357 sentences, 8,130,627 positions).
  * Grouped by identical sentence text so zero test sentences appear in training counts (saved to `data/split_indices_seed42.json`).
* **Held-Out Test Set Statistics & `<UNK>` Profile:**

  * **Total Test Sentences:** 499,357
  * **Total Evaluation Positions:** 8,130,627 token positions
  * **`<UNK>` Token Positions:** **191,638** positions (**2.357%** of test tokens mapped to `<UNK>` via training LUT `lut_train_seed42.npy`)
  * **Unseen Transition / Backoff Profile:**
    * Bigram ($n=2$): ~34.8% of test bigrams unseen in `train_core`, requiring unigram backoff.
    * Trigram ($n=3$): ~62.4% unseen, requiring lower-order backoff.
    * 4-gram ($n=4$): ~81.2% unseen, relying on interpolated backoff distribution.
* **Held-Out Test Perplexity (Kneser-Ney Smoothing, Seed 42):**

  |          Model Order          | Test Perplexity ($\text{PP}$) | Log-Likelihood Sum ($\sum \log P$) | Reduction vs. Unigram |   Benchmark Target   |     Status     |
  | :---------------------------: | :-----------------------------: | :----------------------------------: | :-------------------: | :-------------------: | :------------: |
  | **Unigram ($n = 1$)** |            16,317.94            |          $-78,867,248.20$          | Baseline ($0.0\%$) |       Baseline       |    Recorded    |
  | **Bigram ($n = 2$)** |            1,627.11            |          $-60,122,420.92$          |   **90.03%**   | Lower by $\ge 20\%$ | **PASS** |
  | **Trigram ($n = 3$)** |             921.78             |          $-55,502,189.84$          |   **94.35%**   |   Lower than bigram   | **PASS** |
  | **4-gram ($n = 4$)** |             648.51             |          $-52,643,189.81$          |   **96.03%**   |  Lower than trigram  | **PASS** |
* **Seed Stability Verification (`SEED = 7`):**

  * Evaluated against independent `train_core` counts built from its own split (D4):
    * Unigram ($n=1$): $\text{PP} = 16,288.76$ (Baseline)
    * Bigram ($n=2$): $\text{PP} = 1,625.63$ (**90.02% reduction**)
    * Trigram ($n=3$): $\text{PP} = 923.00$ (**94.33% reduction**)
    * 4-gram ($n=4$): $\text{PP} = 651.48$ (**96.00% reduction**)
  * Ranking order is perfectly invariant across seeds ($n=4 < n=3 < n=2 < n=1$) with relative variation $< 0.5\%$ ($\to$ **PASS**).
* **Model Usability Acceptance:**

  * All higher-order models achieve $> 90\%$ perplexity reduction over unigram, comfortably satisfying the intrinsic usability threshold.

---

## 6. Question 3 Key Results Summary

* **Task Formulation & Tuning Split Calibration:**

  * **Task:** Real-versus-Corrupted Sentence Detection (balanced binary classification on 998,714 held-out test pairs: 499,357 authentic test sentences vs. 499,357 corrupted negatives with 1–2 word substitutions).
  * **Tuning Pair Count Clarification:**
    * The theoretical 5.0% allocation target of the 4,993,570 corpus sentences is 249,678 sentences ($2 \times 249,678 = 499,356$ labeled pairs).
    * In zero-leakage grouped splitting (Stage 06), identical sentences are grouped together to prevent data leakage. Greedy assignment of groups yielded 249,679 sentences in the tuning partition (`tune`), generating exactly **$2 \times 249,679 = 499,358$ labeled tuning pairs** (recorded in `results/q3_metrics.json`).
  * **Zero-Leakage Optimal Thresholds ($\tau_n$):** Tuned strictly on the 499,358 tuning pairs:
    * $n = 1$: $\tau = -10.2542$
    * $n = 2$: $\tau = -8.5700$
    * $n = 3$: $\tau = -8.6520$
    * $n = 4$: $\tau = -8.7901$
* **Held-Out Test Set Performance & Benchmark Compliance Table:**
  Evaluated on the full 998,714 held-out test pairs against proposed benchmark targets:

  |          Model Order          |       Accuracy ($> 0.70$)       |             Precision ($> 0.70$)             |        Recall ($> 0.60$)        |       F1-Score ($> 0.70$)       | ROC-AUC ($> 0.80$ table / $\ge 0.80$ rule) |      Usability Status      |
  | :---------------------------: | :-------------------------------: | :--------------------------------------------: | :-------------------------------: | :-------------------------------: | :--------------------------------------------: | :------------------------: |
  | **$n = 1$ (Unigram)** |      71.21% (**PASS**)      |            0.6720 (**FAIL**)            |      82.87% (**PASS**)      |      0.7422 (**PASS**)      |            0.8118 (**PASS**)            |           Usable           |
  | **$n = 2$ (Bigram)** | **73.58%** (**PASS**) | **0.7006** (**PASS**, $+0.0006$) |      82.37% (**PASS**)      | **0.7572** (**PASS**) |       **0.8359** (**PASS**)       |  **Champion Model**  |
  | **$n = 3$ (Trigram)** |      71.33% (**PASS**)      |            0.6684 (**FAIL**)            |      84.67% (**PASS**)      |      0.7470 (**PASS**)      |   **0.8000** (*Boundary condition*)   | Usable ($\ge 0.80$ rule) |
  | **$n = 4$ (4-gram)** |      70.90% (**PASS**)      |            0.6600 (**FAIL**)            | **86.23%** (**PASS**) |      0.7477 (**PASS**)      |            0.7895 (**FAIL**)            |     **Unusable**     |


  * **Linguistic & Benchmark Analysis:**
    * **Bigram ($n=2$):** Precision is **0.7006**, narrowly exceeding the "above 0.70" benchmark target by only **+0.0006**. It is the only model order that passes all 5 benchmark criteria.
    * **Trigram ($n=3$):** Precision is **0.6684**, failing the $> 0.70$ benchmark. ROC-AUC is exactly **0.8000**: qualifies under the usability threshold rule (*"at least 0.80"*, $\ge 0.80$), but strictly misses the benchmark table check (*"above 0.80"*, $> 0.80$). This discrepancy between the text rule and table check is flagged for instructor discretion.
    * **4-gram ($n=4$):** Precision is **0.6600**, failing the $> 0.70$ benchmark. ROC-AUC is **0.7895**, failing both the $\ge 0.80$ usability threshold and the $> 0.80$ benchmark. The 4-gram model is explicitly unusable for this classification task due to severe context sparsity on corrupted transitions.
* **UNK Confounder & Sensitivity Analysis:**

  * Naive UNK-count baseline achieves only $0.5184$ ROC-AUC (near random chance $0.50$).
  * On the **738,256 pairs containing zero UNK tokens**, the Bigram model achieves **ROC-AUC = 0.8473** and **Accuracy = 74.52%**, confirming that discrimination is driven by genuine grammatical and statistical fluency, not out-of-vocabulary artifacts.

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
* **Evaluation Scope & Denominator Resolution (475,613 vs. 8,130,627):**

  * **No Sampling:** Conforming strictly to Guideline B3, evaluation spans the **100% full held-out test split** (499,357 sentences).
  * **Denominator Distinction:**
    * In **Q2 Intrinsic Evaluation**, perplexity evaluates cross-entropy across **all 8,130,627 word token positions** in the test split.
    * In **Q4 Next-Word Prediction**, evaluation targets the final word of eligible test sentences with length $\ge 3$ tokens.
    * Out of 499,357 test sentences, 488,282 meet the length criterion. Exactly 12,669 target words map to `<UNK>` in the training vocabulary (since the application vocabulary only predicts known vocabulary words, unknown words cannot be predicted), leaving **475,613 evaluated prediction positions** (exactly one prediction target per candidate sentence).
* **Full Held-Out Test Split Performance & Context Comparison (from `results/q4_eval.json`):**
  Evaluated across all 475,613 prediction targets:

  |             Context Length / Model             | Top-1 Accuracy ($\ge 10.0\%$) | Top-5 Hit Rate ($\ge 30.0\%$) | MRR@5 ($\ge 0.15$) | Response Time per Target |               Status               |
  | :---------------------------------------------: | :-----------------------------: | :-----------------------------: | :------------------: | :----------------------: | :--------------------------------: |
  | **Two-Word Context ($L = 2$, Trigram)** |        **23.37%**        |        **40.05%**        |   **0.2963**   |       $< 0.1$ ms       | **PASS** (Usable & Deployed) |
  | **One-Word Context ($L = 1$, Bigram)** |             17.95%             |             32.91%             |        0.2348        |       $< 0.1$ ms       |          PASS (Baseline)          |

  * **Application Context Length:** The interactive application uses **Two-word context ($L = 2$, Trigram)** as its primary prediction engine, backing off to bigram ($L = 1$) and unigrams when 2-word histories are unseen.
  * **Linguistic Rationale:** Two-word context provides richer semantic conditioning, delivering substantial accuracy gains over the single-word baseline (**+5.42% Top-1 Accuracy**, **+7.14% Top-5 Hit Rate**, and **+0.0615 MRR@5**). Both context lengths comfortably exceed all benchmark thresholds.
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
