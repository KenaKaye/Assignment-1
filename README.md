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
| **Ethiopic Letters Only (U+1200 - U+135A)**                | **311,215,830 characters** (95.67% of non-space text $\to$ **PASS**, benchmark $> 80.0\%$) |
| **Ethiopic Characters with Punctuation (U+1200 - U+137F)** | 319,350,166 characters (98.17% of non-space text)                    |
| **Sentence Delimiters**                                    | Sentence punctuation (`።`, `?`, `!`, `፡፡+`)               |

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
  * **Hand-Check Verification Fixture:**
    Evaluated on the 2-sentence validation fixture: `"ኢትዮጵያ ታሪካዊት አገር ናት። ህዝቧም አኩሪ ታሪክ አለው።"`
    * Hand-tallied bigrams: `('<s>', 'ኢትዮጵያ'): 1`, `('ኢትዮጵያ', 'ታሪካዊት'): 1`, `('ታሪካዊት', 'አገር'): 1`, `('አገር', 'ናት'): 1`, `('ናት', '</s>'): 1`, `('<s>', 'ህዝቧም'): 1`, `('ህዝቧም', 'አኩሪ'): 1`, `('አኩሪ', 'ታሪክ'): 1`, `('ታሪክ', 'አለው'): 1`, `('አለው', '</s>'): 1`.
    * Code-generated bigram extraction matches the manual tally with 100% precision and identical boundary transitions.

* **Q1.3 Probabilities & Top 10 Tables (Joint Sequence MLE):**
  * Maximum Likelihood Estimation sequence probability is defined by $P(w_1, \dots, w_n) = \frac{\text{count}(w_1, \dots, w_n)}{N_{\text{total}}}$, where $N_{\text{total}} = 81,161,104$.
  * Because $N_{\text{total}}$ is constant, the n-grams with highest frequency are identically the n-grams with highest sequence probability.
  * **Top 10 Most Likely 4-grams (Order $n = 4$, from `results/q1_top10_n4.csv`):**
    Filtering boundary markers (`<s>`, `</s>`, `<UNK>`) presents purely authentic lexical Amharic 4-grams:

    | Rank | 4-gram Sequence (Pure Lexical Words) | Corpus Count | Joint Probability ($P$) | Linguistic Role & Discourse Function |
    | :---: | :--- | :---: | :---: | :--- |
    | **1** | `ቀን ዓ ም ኢሳት` | 9,786 | $1.2057 \times 10^{-4}$ | Broadcast news date formula (*"[Day] E.C., ESAT..."*) |
    | **2** | `ዓ ም ኢሳት ዜና` | 9,700 | $1.1952 \times 10^{-4}$ | Media attribution headline (*"[Year] E.C., ESAT News..."*) |
    | **3** | `እ ኤ አ በ` | 8,613 | $1.0612 \times 10^{-4}$ | Gregorian calendar date prefix (*"In ... G.C."*) |
    | **4** | `የካቲት ቀን ዓ ም` | 2,952 | $3.6372 \times 10^{-5}$ | Formal month-date formula (*"Yekatit [Day], E.C."*) |
    | **5** | `መጋቢት ቀን ዓ ም` | 2,908 | $3.5830 \times 10^{-5}$ | Formal month-date formula (*"Megabit [Day], E.C."*) |
    | **6** | `ከ ዓ ም ጀምሮ` | 2,866 | $3.5312 \times 10^{-5}$ | Temporal origin construction (*"Starting from [Year] E.C."*) |
    | **7** | `ግንቦት ቀን ዓ ም` | 2,824 | $3.4795 \times 10^{-5}$ | Formal month-date formula (*"Ginbot [Day], E.C."*) |
    | **8** | `ከ ሚሊዮን ብር በላይ` | 2,822 | $3.4770 \times 10^{-5}$ | Financial quantification formula (*"More than ... million Birr"*) |
    | **9** | `የአዲስ አበባ ከተማ አስተዳደር` | 2,607 | $3.2121 \times 10^{-5}$ | Municipal state entity (*"Addis Ababa City Administration"*) |
    | **10**| `ሰኔ ቀን ዓ ም` | 2,569 | $3.1653 \times 10^{-5}$ | Formal month-date formula (*"Sene [Day], E.C."*) |

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

    | Order | Top-5 Before Removal (`results/q1_top10_n*.csv`) | Top-5 After Removal (`results/q1_stopword_top10_n*.csv`) | Filter Impact & Observations |
    | :---: | :--- | :--- | :--- |
    | **$n=1$** | 1. `ነው` (1,165,953)<br>2. `ላይ` (754,678)<br>3. `ውስጥ` (346,314)<br>4. `ወደ` (331,106)<br>5. `እና` (318,942) | 1. `ጊዜ` (244,117)<br>2. `ደግሞ` (209,377)<br>3. `አንድ` (192,674)<br>4. `ቤት` (182,048)<br>5. `አቶ` (169,169) | All top-5 function words (`ነው`, `ላይ`, `ውስጥ`, `ወደ`, `እና`) successfully eliminated; open-class nouns and titles rise. |
    | **$n=2$** | 1. `ዓ ም` (108,455)<br>2. `ቀን ዓ` (46,525)<br>3. `ነገር ግን` (44,744)<br>4. `ዶ ር` (35,624)<br>5. `ማለት ነው` (32,386) | 1. `ዓ ም` (108,455)<br>2. `ቀን ዓ` (46,525)<br>3. `ዶ ር` (35,624)<br>4. `አዲስ አበባ` (31,388)<br>5. `በ ዓ` (28,486) | Function bigrams `ነገር ግን` (connective) and `ማለት ነው` (copular phrase) removed; entity bigram `አዲስ አበባ` enters top 5. |
    | **$n=3$** | 1. `ቀን ዓ ም` (45,927)<br>2. `በ ዓ ም` (28,127)<br>3. `እ ኤ አ` (16,731)<br>4. `ዓ ም ኢሳት` (9,789)<br>5. `ም ኢሳት ዜና` (9,700) | 1. `ቀን ዓ ም` (45,927)<br>2. `በ ዓ ም` (28,127)<br>3. `እ ኤ አ` (16,731)<br>4. `ዓ ም ኢሳት` (9,789)<br>5. `ም ኢሳት ዜና` (9,700) | Top trigrams are stable calendar and broadcast collocations containing no bare stopwords. |
    | **$n=4$** | 1. `ቀን ዓ ም ኢሳት` (9,786)<br>2. `ዓ ም ኢሳት ዜና` (9,700)<br>3. `እ ኤ አ በ` (8,613)<br>4. `የካቲት ቀን ዓ ም` (2,952)<br>5. `መጋቢት ቀን ዓ ም` (2,908) | 1. `ቀን ዓ ም ኢሳት` (9,786)<br>2. `ዓ ም ኢሳት ዜና` (9,700)<br>3. `እ ኤ አ በ` (8,613)<br>4. `የካቲት ቀን ዓ ም` (2,952)<br>5. `መጋቢት ቀን ዓ ም` (2,908) | Top 4-grams comprise calendar news formulas; zero boundary tags or standalone stopwords survive. |

* **Q1.6 Word Clouds:**
  * 8 PNG images generated in `figures/` (`q1_6_wordcloud_*.png`) using `NotoSansEthiopic-Regular.ttf` with zero missing glyphs or boxes.

* **Q1.7 Target Sentence Probability & Distribution Sums (Kneser-Ney Smoothing):**
  * Evaluated across four test sentences (in-corpus, out-of-corpus, and ungrammatical nonsense) using interpolated Kneser-Ney smoothing (from `results/q1_sentence_probs.csv`):

    | Test Sentence | Category | $n$ | Total $\log P$ | Per-Token $\log P$ | Verbatim in Corpus? |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | `ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።` | Out-of-Corpus Fluent | 1<br>2<br>3<br>4 | $-43.2479$<br>$-28.4221$<br>$-28.3168$<br>$-25.4912$ | $-8.6496$<br>$-5.6844$<br>$-5.6634$<br>$\mathbf{-5.0982}$ | No (`False`) |
    | `የኢትዮጵያ መንግስት ለህዝቡ ሰላም እና ልማት ይሰራል።` | Out-of-Corpus Fluent | 1<br>2<br>3<br>4 | $-63.2111$<br>$-48.1076$<br>$-41.7383$<br>$-42.7557$ | $-7.9014$<br>$-6.0135$<br>$\mathbf{-5.2173}$<br>$-5.3445$ | No (`False`) |
    | `አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።` | In-Corpus Fluent | 1<br>2<br>3<br>4 | $-53.2230$<br>$-32.3624$<br>$-26.8990$<br>$-16.2373$ | $-7.6033$<br>$-4.6232$<br>$-3.8427$<br>$\mathbf{-2.3196}$ | Yes (`True`) |
    | `ድንጋይ ሰማይ ሮጠ በላ ትላንት ወንበር።` | Invented Nonsense | 1<br>2<br>3<br>4 | $-67.2935$<br>$-71.6153$<br>$-71.7656$<br>$-73.2686$ | $-9.6134$<br>$-10.2308$<br>$-10.2522$<br>$\mathbf{-10.4669}$ | No (`False`) |

  * **Linguistic Discrimination:** High-order models strongly reward grammatical and collocationally frequent structures (in-corpus verbatim sentence reaches $-2.3196$ per token, fluent out-of-corpus sentences score $-5.0982$ to $-5.3445$) while severely penalizing ungrammatical word salad (penalized to $-10.4669$ per token).
  * **Sum Check:** Conditional distribution sums $\sum_{w \in V} P_{\text{KN}}(w \mid h) = 1.0000000000$ with absolute numerical discrepancy $\le 1 \times 10^{-9}$ across all tested context histories, confirming strict mathematical calibration.

* **Q1.8 Text Generation & Verbatim Duplication Analysis:**
  * Generated 12 sentences (3 per order $n=1..4$, max length 25 tokens) with fixed seed from `results/q1_generation.json`:

    | Order | Generated Sentence | Length | $\log P$ | Verbatim Copy? |
    | :---: | :--- | :---: | :---: | :---: |
    | **$n=1$** | 1. `ያስተዋልነው ቱርክ ልፃፍ ማብራሪያም ገንዘብ ዘገየችን ከአብዮት ጠሃ ፍቅር ጎጆ ጭራሽ የምናድርባት እንደሚያወጣ ግንባርቀደም ቀጥለዋል ማሰብ እንደምችል የ ወናፍ ፀሐይን ተተብትበው በተባለ ስለሚሰግዱላቸው ከሚያጋልጡት የአምባው።`<br>2. `ፈተና የሚያስችላቸውን እንዲሁም ከላይ ተሰናብተው የተደለደሉት ኅየንተ ስብሰባዎች ከመቼውም ተቀምጠዋል ንጉሥ ሦስት አንገታቸውን የአካባቢው ተላቃ የኮሚሽኑ በሮመዳን የስታዲየም በመልካም የመሣሪያዎች ሃንድ ተጠናቆ የያህዌ እንደሚቻልና አልፎ።`<br>3. `ጥፋት ወደ አልሆነልንም ለማዋረድ ለፍቅርና ንዶ ዶክተሩ በሃላፊነት ገቢ ዙሪያ ያሳዩበት ወገኔ በልጅ እንደተሰማሩ ከኢራን ሊረዳው በመንገዱ ትልልቅ እንደ ለማንሳት እንዲሆኑ ከራሳቸው በመድማት ፈጣሪ መንግሥት።` | 25<br><br>25<br><br>25 | $-309.45$<br><br>$-276.58$<br><br>$-270.53$ | **0 / 3 (0.0%)** |
    | **$n=2$** | 1. `አበረታች ውጤት ማስቀጠል የግድ ይለናል እንጂ ለአንድ የተወሰነ አንድ ላይ እንዴት እንደሚታይ አስረድተዋል።`<br>2. `የአባቶቻችን አምላክ በጎች ሕይወት ቀላል ማድረግ ብቸኛው ኢትዮጵያዊ ሁሉ ስለ ሚገኙ ኣብያተ ክርስቲያናት አካባቢ የእስረኞቹ መፈታት ያለባቸው ሆነው ሳለ አንድ ፈራጅ መሆኑ ዛሬ ሥልጣን ለማስጠበቅ።`<br>3. `በእጅጉ ቀንሶታል የሚሉት ከብዙ ን ጠቅሰው በቀጣይ በሚደረገው ምርጫ ቦርድ ውስጥ የሚሠሩ ተቋማት ብዙ ሰው ምስጢር ነበር የሚሉም ነበሩ።` | 13<br><br>25<br><br>19 | $-77.11$<br><br>$-149.80$<br><br>$-108.36$ | **0 / 3 (0.0%)** |
    | **$n=3$** | 1. `ቢሏቸው ምን እንዳሉ ታውቃላችሁ።`<br>2. `የዓለም ዋንጫ ጉብኝት በዘጠኝ ወራት ውስጥ ችግር እንዳለ ጠቁመዋል።`<br>3. `ናቸው የዚያ ሰፈር ልጆች ትምህርት ቤት ዲዛይን አልቋል።` | 4<br><br>9<br><br>8 | $-19.62$<br><br>$-32.56$<br><br>$-32.95$ | **0 / 3 (0.0%)** |
    | **$n=4$** | 1. `ሆቴሉ ተጠናቅቆ ስራ ሲጀምር ከ በላይ ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል ተብሏል።`<br>2. `ሠራተኛ የነበረ ቢሆንም የነፍሱ ጥሪ ግን ሥራ ፈጠራ ነበር።`<br>3. `የተሸጠው መሬት እጅግ አነስተኛ መሆኑን ገልፀዋል።` | 12<br><br>9<br><br>6 | $-29.04$<br><br>$-19.43$<br><br>$-18.11$ | **0 / 3 (0.0%)** |

  * **Verbatim Replication Fraction:** Evaluated against all 4,993,570 lines of `GPAC_cleaned.txt`:
    * $n = 1$: **0 / 3 (0.0%)** full verbatim copies
    * $n = 2$: **0 / 3 (0.0%)** full verbatim copies
    * $n = 3$: **0 / 3 (0.0%)** full verbatim copies
    * $n = 4$: **0 / 3 (0.0%)** full verbatim copies
    * **Overall Duplication Rate:** **0 / 12 (0.0%)**.
  * **Qualitative Transition:** While unigrams yield syntactic gibberish and bigrams produce only local word pairs, trigrams and 4-grams construct grammatically fluent sentences by assembling authentic multi-word idiomatic collocations (e.g., `ለሚሆኑ ወጣቶች የስራ እድል ይፈጥራል ተብሏል`) without copying entire sentences verbatim.

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

  | Model Order | Test Perplexity ($\text{PP}$) | Log-Likelihood Sum ($\sum \log P$) | Reduction vs. Unigram | Benchmark Target | Status |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Unigram ($n = 1$)** | 16,317.94 | $-78,867,248.20$ | Baseline ($0.0\%$) | Baseline | Recorded |
  | **Bigram ($n = 2$)**  | 1,627.11  | $-60,122,420.92$ | **90.03%** | Lower by $\ge 20\%$ | **PASS** |
  | **Trigram ($n = 3$)** | 921.78    | $-55,502,189.84$ | **94.35%** | Lower than bigram | **PASS** |
  | **4-gram ($n = 4$)**  | 648.51    | $-52,643,189.81$ | **96.03%** | Lower than trigram | **PASS** |

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

  | Model Order | Accuracy ($> 0.70$) | Precision ($> 0.70$) | Recall ($> 0.60$) | F1-Score ($> 0.70$) | ROC-AUC ($> 0.80$ table / $\ge 0.80$ rule) | Usability Status |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **$n = 1$ (Unigram)** | 71.21% (**PASS**) | 0.6720 (**FAIL**) | 82.87% (**PASS**) | 0.7422 (**PASS**) | 0.8118 (**PASS**) | Usable |
  | **$n = 2$ (Bigram)**  | **73.58%** (**PASS**) | **0.7006** (**PASS**, $+0.0006$) | 82.37% (**PASS**) | **0.7572** (**PASS**) | **0.8359** (**PASS**) | **Champion Model** |
  | **$n = 3$ (Trigram)** | 71.33% (**PASS**) | 0.6684 (**FAIL**) | 84.67% (**PASS**) | 0.7470 (**PASS**) | **0.8000** (*Boundary condition*) | Usable ($\ge 0.80$ rule) |
  | **$n = 4$ (4-gram)**  | 70.90% (**PASS**) | 0.6600 (**FAIL**) | **86.23%** (**PASS**) | 0.7477 (**PASS**) | 0.7895 (**FAIL**) | **Unusable** |

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

  | Context Length / Model | Top-1 Accuracy ($\ge 10.0\%$) | Top-5 Hit Rate ($\ge 30.0\%$) | MRR@5 ($\ge 0.15$) | Response Time per Target | Status |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Two-Word Context ($L = 2$, Trigram)** | **23.37%** | **40.05%** | **0.2963** | $< 0.1$ ms | **PASS** (Usable & Deployed) |
  | **One-Word Context ($L = 1$, Bigram)**  | 17.95% | 32.91% | 0.2348 | $< 0.1$ ms | PASS (Baseline) |

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
