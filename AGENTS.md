# AGENTS.md: NLP Assignment 1 (N-grams on the Amharic Corpus GPAC)

These instructions apply to every agent or assistant working in this repository. They merge the general coding guidelines below with rules specific to this assignment. The project-specific rules take priority where they conflict with the general ones.

**Assignment:** NLP Assignment 1 (10%), Addis Ababa University, School of Information Technology and Engineering. Instructor: Fantahun B. (PhD).
**Reference files:** `NLP_Assignment1_Questions.md` (the questions), `NLP_Assignment1_Implementation_Plan.md` (the step-by-step plan, benchmarks, and artifact layout), `NLP_Assignment1_UI_Plan.md` (the Next.js interface). If these files disagree, the implementation plan wins, and the other files must be updated to match it.

Tradeoff: these guidelines favor caution over speed. For trivial tasks, use judgment.

---

## Part A: General Guidelines

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Don't pick one silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

### 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it. Don't delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" becomes "Write tests for invalid inputs, then make them pass."
- "Fix the bug" becomes "Write a test that reproduces it, then make it pass."
- "Refactor X" becomes "Ensure tests pass before and after."

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

---

## Part B: Assignment-Specific Rules

### B1. Order of Work

Follow the order in the implementation plan. Do not skip ahead.

1. Step 0: verify the raw corpus, then build the cleaned corpus, the vocabulary, and the token array (plan items 8 to 12).
2. Q1.1 through Q1.8, in order.
3. Q2 (intrinsic evaluation).
4. Q3 (extrinsic evaluation).
5. Q4 (predictive text application).
6. Presentation material.

Each step has a verification check in the plan. Do not start the next step until the current check passes. If a check fails, fix the cause or ask. Do not move on.

### B2. Verify the Data Before Anything Else

- Never run analysis on the corpus until the Step 0 checks pass: correct file, UTF-8 decoding, Ethiopic character share above 80%, no HTML content, no truncation, and a recorded SHA-256 checksum.
- All analysis reads `data/GPAC_cleaned.txt` and the artifacts listed in the plan's artifact layout. Never read the raw `GPAC.txt` for analysis, and never use the Kaggle `Cleaned_GPAC.txt` as input.
- The cleaned corpus keeps sentence-final punctuation (D5): each line ends with `።`, `?`, or `!`. `፡፡+` is normalized to `።`.
- Record the verification result in `README.md`.
- If the file looks wrong, stop and ask the user for the correct file. Do not substitute another file or fabricate data.

### B3. Prevent Data Leakage

- Split the corpus by sentence into train and test sets once, using `SEED = 42`, and save the line indices to `data/split_indices_seed42_v2.json`. Reuse the saved split. Do not re-split silently.
- A 5% tuning set `tune` is carved out of the training split *before counting*, leaving `train_core` (~85%) for training counts (D2).
- Build n-gram counts from `train_core` only when evaluating in Q2, Q3, and Q4. The application uses the seed-42 training-split counts as its evaluation.
- Seed-7 stability evaluation in Q2 uses its own training counts built from its own independent `train_core` split (D4).
- Words that occur $\le 1$ time in `train_core` are mapped to `<UNK>` via the training LUT (`lut_train_seed42.npy`, `lut_train_seed7.npy`). Test words not in the training vocabulary are scored as `<UNK>`. Original global IDs are preserved.
- Never tune any threshold (for example, τ in Q3) on the test split. Tune it strictly on the `tune` set (which has 0 sentence overlap with training counts) and then apply it once to the test set.
- No sampling of the data. Q2, Q3, and Q4 evaluate on the full held-out test splits without sampling exceptions (D3).
- Never use test sentences to choose smoothing parameters, stopwords, or the usability threshold.

### B4. Reproducibility

- Set `random.seed(42)` and `numpy.random.seed(42)` in every notebook that uses randomness.
- Any random output (generated sentences in Q1.8, sampled negatives in Q3) must be reproducible from the seed.
- Record library versions in `README.md`.
- Use relative paths from the repository root. Do not hardcode absolute paths such as `/content/` or `C:\Users\`.

### B5. Explanatory Sections for the Teacher

Every solution file must make it easy for the teacher to find each answer.

- Every notebook begins with a markdown header cell that states:
  - the question number and title (for example, "Question 1: N-gram language model")
  - the sub-questions covered (for example, "Q1.3 and Q1.4")
  - which cells prove each sub-question
- Every sub-question has its own section heading that matches the question text, for example "Q1.5: Stopword removal and recomputed top 10."
- Every result cell is followed by a markdown cell that states the result in one or two sentences, with the actual value.
- Do not leave a result without an explanation. A table or figure alone is not enough.

### B6. Metrics and Thresholds

- Use the metrics that apply to each question, and state why the others do not apply.
  - Q1: counts and probabilities. Sum checks must pass within 1e-9.
  - Q2: perplexity only. Classification metrics do not apply, because there are no predicted categories.
  - Q3: accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix. Log loss only if the scores are calibrated, and say so.
  - Q4: top-1 accuracy, top-5 hit rate, and MRR@5 (`mrr_at_5`).
- Use the benchmarks and thresholds in the implementation plan.
- Never change a benchmark or threshold to match the results. If a result misses a target, report the actual number and explain the likely cause.
- Label every threshold in the report as a proposed target set by the group, not a published standard. Confirm with the instructor where the assignment does not specify one.

### B7. Amharic Text Handling

- Read and write all files as UTF-8.
- Keep Ethiopic letters (Unicode U+1200 to U+135A) and the punctuation `።` (U+1362), `፡` (U+1361), `፣` (U+1363), `?`, and `!`. Remove Latin letters, digits, Ethiopic numerals, and other symbols during normalization. This is the same definition as the plan's cleaning step, Step 0, item 9.
- Use `።` as a sentence boundary. Decide whether `፡` and `፣` are sentence boundaries by inspecting samples, and record the decision.
- Word clouds must use an Ethiopic font (Noto Sans Ethiopic). Check that the rendered labels are readable. Boxes mean the font is missing, and the word clouds do not count until this is fixed.
- Do not normalize away the morphology of Amharic words without saying so. Note in the report that inflected forms are counted as separate tokens.

### B8. Do Not Fabricate Results

- Print every number from code. Never type a number into a report from memory.
- If a sentence, probability, or suggestion has not been computed, say so. Do not write an expected value.
- The example suggestions in Q4 (ምርት, ንግድ, አምራችነት) are illustrative. Do not claim they are verified unless they appear in the corpus counts. Check and report.
- If a command fails or a result cannot be reproduced, report the failure. Do not present partial output as complete.

### B9. Notebook and File Conventions

- Notebook names: `Q1_ngram_model.ipynb`, `Q2_intrinsic_eval.ipynb`, `Q3_extrinsic_eval.ipynb`, `Q4_predictive_app.ipynb`.
- Build notebooks: `kaggle/Kaggle_Build.ipynb` and `colab/Colab_Build.ipynb`.
- Application code for Q4 goes in `app/`. Keep the application logic separate from the notebook experiments.
- Figures go in `figures/` with descriptive names, for example `q1_6_wordcloud_bigram_before.png`.
- Follow the artifact layout table in the plan. Commit only the files marked "Yes" there. The large files in `data/` are rebuilt by the scripts and must be listed in `.gitignore`.
- Build scripts live in `scripts/` and are run once; notebooks and the application load their outputs.
- Keep `README.md` current: corpus source, verification result, environment, how to run each notebook, and the group members.

### B10. Q4 Application Scope

- Build only what the assignment asks: read input, take the last one or two words, return the top five candidates with probabilities, and let the user select one and continue.
- The Next.js web interface is allowed only as described in `NLP_Assignment1_UI_Plan.md`. The CLI stays unchanged in behavior. Do not add a database, user accounts, or a model other than the n-gram model.
- Load the memory-mapped artifacts once at startup. Do not rebuild counts or read the corpus at launch. Do not scan the count arrays on each request; use the binary-search lookups described in the plan.
- Handle these cases: empty input, words outside the vocabulary (map to `<UNK>`), and an exit command.

### B13. Where Code Runs

- Heavy work runs on Kaggle or Colab only: any pass over `GPAC.txt` or `GPAC_cleaned.txt`, any sort, count, or lookup over `tokens.npy` or the count arrays, any perplexity or evaluation over the test split, and any build of counts.
- Every heavy script calls `require_heavy_runtime()` and must refuse to run on the PC. Do not work around this check. The `ALLOW_HEAVY_LOCAL=1` override may be used only for a deliberate, recorded decision, which is logged to `results/run_log.jsonl`.
- The PC runs light work only: reading `results/`, rendering figures, running the CLI, the API, and the Next.js UI against the downloaded training-split artifacts, checksum checks, and unit tests on small fixtures in `tests/`.
- Heavy stages use two-tier checkpoints via `data/manifest.json`: a stage is skipped only if all its input hashes and output hashes match the recorded values.
- Before any notebook or app runs on the PC, `scripts/check_download.py` must report every downloaded checksum as matching.

### B11. When to Ask Instead of Proceeding

Ask before implementing when any of the following is true:

- The corpus file fails a Step 0 check.
- The assignment wording allows more than one reasonable interpretation, and the choice changes the output. Examples: whether `፡` is a sentence boundary, which extrinsic task to use, or which smoothing method to apply.
- A required benchmark is missing or unclear.
- Two group members' instructions conflict.
- A change would touch the test split, the saved splits, or the verification record.
- A counting stage exceeds the detected memory budget and the partition fallback has not been tried or recorded.

Ask once, clearly, and state the options you see and your recommendation. Do not ask questions that the plan already answers.

### B12. Presentation Support

- Keep the figures and tables needed for the presentation in `figures/` and in the notebooks, labeled with their question number.
- Every slide claim must trace to a printed cell in a notebook. If a claim has no supporting cell, do not put it on a slide.
- Before the presentation, confirm the Q4 demo runs from a clean environment, using the saved split and seed.

---

## Part C: Definition of Done

A task is done only when:

- the verification check in the implementation plan passes for that step
- the output is in the correct notebook, under the correct sub-question heading
- every result has an explanatory sentence with its value
- the seed is set and the run reproduces the same numbers
- no benchmark was changed to fit the results
- `README.md` reflects any new file or decision
- the diff contains only changes that trace to the step being worked on

These guidelines are working if: diffs contain fewer unnecessary changes, code is not overcomplicated, and clarifying questions come before implementation rather than after mistakes.
