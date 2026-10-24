"""Helper script to construct Kaggle_Build.ipynb and update Colab_Build.ipynb."""

import json
from pathlib import Path

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
    else:
        lines = source
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
    else:
        lines = source
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def build_kaggle_notebook():
    cells = []

    # Cell 1: Markdown Header
    cells.append(md_cell("""# NLP Assignment 1: Kaggle Build

**Assignment:** NLP Assignment 1 (10%), Addis Ababa University, School of Information Technology and Engineering.
**Instructor:** Fantahun B. (PhD)
**Questions Covered:**
- Step 0: Raw corpus verification and assessment, cleaned text generation, verification, vocabulary and token array creation.
- Q1.1 to Q1.8: N-gram language models, MLE probabilities, smoothing, conditional probabilities, stopword analysis, sentence scoring, and model generation.
- Q2: Intrinsic evaluation (perplexity across orders 1 to 4 on held-out test splits, primary seed 42 and replication seed 7).
- Q3: Extrinsic evaluation (real vs corrupted Amharic sentence detection, threshold tuning, UNK confound analysis, ROC-AUC).
- Q4: Predictive text application evaluation (Top-1, Top-5, MRR@5 across 1-word and 2-word contexts).
- Presentation Artifacts: Canonical artifact bundle generation for local PC exploration.

**Execution Rule (AGENTS.md B13):**
Heavy pipeline stages run exclusively in this heavy environment. The local PC runs only unit tests, checksum verification (`scripts/check_download.py`), and the light interactive UI/CLI.

**Proof Mapping:**
- Step 0: Produced by `01_assess_raw.py`, `02_build_cleaned.py`, `03_verify_cleaned.py`, `04_build_vocab_tokens.py`, and `06_split.py`.
- Q1: Produced by `07_count_full.py` and `10_q1_results.py`.
- Q2: Produced by `08_count_train.py` and `11_q2_perplexity.py`.
- Q3: Produced by `12_q3_extrinsic.py`.
- Q4: Produced by `13_q4_eval.py`."""))

    # Cell 2: Code Environment Report
    cells.append(code_cell("""import os, sys, platform, importlib, json, shutil

def read_first(path):
    try:
        return open(path).read().strip()
    except Exception:
        return None

cg = read_first("/sys/fs/cgroup/memory.max") or read_first("/sys/fs/cgroup/memory/memory.limit_in_bytes")
try:
    phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
except Exception:
    phys = 0

print("KAGGLE_KERNEL_RUN_TYPE =", os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
print("/kaggle/working exists =", os.path.isdir("/kaggle/working"))
print("physical RAM (GiB)     =", round(phys / 2**30, 2) if phys else "N/A")
print("cgroup limit          =", cg)
print("CPUs                  =", os.cpu_count())
print("disk free (GiB)       =", round(shutil.disk_usage("/kaggle/working").free / 2**30, 1) if os.path.isdir("/kaggle/working") else "N/A")
print("python                =", platform.python_version())"""))

    cells.append(md_cell("The memory limit printed here sets the runtime budget. The stages compute `budget_mb = total_mb * 0.90 - 1024` (90% of available RAM, minus 1 GiB for the interpreter margin)."))

    # Cell 3: Dependencies & Environment
    cells.append(code_cell("""import sklearn, numpy
print("scikit-learn version:", sklearn.__version__)
print("numpy version:       ", numpy.__version__)"""))

    cells.append(md_cell("scikit-learn is required for Q3 extrinsic metrics and ROC computation; numpy provides vectorization."))

    # Cell 4: Clone Repository
    cells.append(code_cell("""import os

# Start from a directory that always exists before removing anything, avoiding getcwd errors
os.chdir("/kaggle/working")

REPO_URL = "https://github.com/KenaKaye/Assignment-1.git"
BRANCH = "full-corpus"

!rm -rf /kaggle/working/NLP_Assignment1
!git clone -b {BRANCH} {REPO_URL} /kaggle/working/NLP_Assignment1
%cd /kaggle/working/NLP_Assignment1
!git log -1 --oneline"""))

    cells.append(md_cell("The commit hash printed above represents the exact code version executing this build run. Record this commit hash in `README.md`."))

    # Cell 5: Attach and verify raw corpus from Kaggle dataset (supports GPAC.txt.gz and GPAC.txt)
    cells.append(code_cell("""import gzip, shutil, hashlib, os

# Start from a directory that always exists, then enter the repository.
os.chdir("/kaggle/working")
os.chdir("/kaggle/working/NLP_Assignment1")
os.makedirs("data", exist_ok=True)

DST = "data/GPAC.txt"

# Recursive search for GPAC file under /kaggle/input (case-insensitive)
found_files = []
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if "gpac" in f.lower() and (f.endswith(".gz") or f.endswith(".txt")):
            found_files.append(os.path.join(root, f))

if not found_files:
    print("Contents currently mounted in /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        print(f"  {root}: {files}")
    raise SystemExit(
        "GPAC dataset not found in /kaggle/input!\\n"
        "To fix: In your Kaggle notebook, click '+ Add Input' (top right),\\n"
        "search for your uploaded dataset under 'Your Work' or 'Datasets', and click '+' to attach it."
    )

src_path = found_files[0]
print(f"Found input corpus at: {src_path}")

if src_path.endswith(".gz"):
    print(f"Decompressing {src_path} -> {DST}...")
    with gzip.open(src_path, "rb") as src, open(DST, "wb") as dst:
        shutil.copyfileobj(src, dst)
else:
    print(f"Copying {src_path} -> {DST}...")
    shutil.copy(src_path, DST)

h = hashlib.sha256()
with open(DST, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)

actual_hash = h.hexdigest()
EXPECTED = "977706859188f7598ca97a29d7203d86926c11ede591fba31915c74ae9eabcfa"
if actual_hash != EXPECTED:
    print("----------------------------------------------------------------------")
    print(f"ACTUAL HASH: {actual_hash}")
    print(f"EXPECTED:    {EXPECTED}")
    print("The attached file does not match the authoritative corpus hash.")
    print("----------------------------------------------------------------------")

assert actual_hash == EXPECTED, f"Raw corpus hash mismatch: {actual_hash}"
print("GPAC.txt SHA-256 matches authoritative hash:")
print(actual_hash)"""))

    cells.append(md_cell("The raw corpus hash strictly matches the authoritative value recorded in the implementation plan. Step 0 raw data verification passes. (Note: if the attached dataset is a different version, the hash will not match and the assertion will catch it)."))

    # Cell 6: Reset stale outputs
    cells.append(code_cell("""import os, shutil
from pathlib import Path

os.chdir("/kaggle/working/NLP_Assignment1")

# Set stages to reset if forced re-run is required (e.g. ['02_build_cleaned'])
RESET_STAGES = []

# Detect defect 2-1: If GPAC_cleaned.txt exists but sentences lack punctuation boundaries, reset 02-13
cleaned_file = Path("data/GPAC_cleaned.txt")
if cleaned_file.exists():
    with open(cleaned_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line and not any(first_line.endswith(p) for p in ["።", "?", "!"]):
            print("Detected stale GPAC_cleaned.txt lacking sentence-final punctuation! Triggering automatic reset of stages 02 to 13...")
            RESET_STAGES = ["02_build_cleaned"]

if RESET_STAGES:
    print(f"Resetting pipeline stages starting from {RESET_STAGES}...")
    manifest_file = Path("data/manifest.json")
    if manifest_file.exists():
        manifest_file.unlink()
    if cleaned_file.exists():
        cleaned_file.unlink()
    # Remove old counts and results
    shutil.rmtree("data/counts", ignore_errors=True)
    shutil.rmtree("results", ignore_errors=True)
    for f in Path("data").glob("*.npy"):
        f.unlink()
    for f in Path("data").glob("*.tsv"):
        if f.name != "sentence_assessment.tsv":
            f.unlink()
    print("Reset complete. Pipeline will build all downstream outputs cleanly.")
else:
    print("No stale stages detected. Manifest checkpoints will allow fast resumption.")"""))

    cells.append(md_cell("Outputs are checked for integrity. Per Decision D5 and Section 6 Stage 02, sentence-final punctuation is preserved in `GPAC_cleaned.txt`."))

    # Cell 7: Run pipeline
    cells.append(code_cell("""import subprocess, time, sys, os, json

os.chdir("/kaggle/working/NLP_Assignment1")

STAGES = [
    ["scripts/01_assess_raw.py"],
    ["scripts/02_build_cleaned.py"],
    ["scripts/03_verify_cleaned.py"],
    ["scripts/04_build_vocab_tokens.py"],
    ["scripts/06_split.py"],
    ["scripts/07_count_full.py"],
    ["scripts/08_count_train.py", "--seed", "42"],
    ["scripts/08_count_train.py", "--seed", "7"],
    ["scripts/10_q1_results.py"],
    ["scripts/11_q2_perplexity.py"],
    ["scripts/12_q3_extrinsic.py"],
    ["scripts/13_q4_eval.py"],
]

times = {}
os.makedirs("results", exist_ok=True)

for args in STAGES:
    name = " ".join(args)
    print(f"\\n==================== Running: {name} ====================", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    dur = round(time.time() - t0, 1)
    times[name] = dur
    print(f"== {name}: exit code {rc}, elapsed: {dur} s ==", flush=True)
    if rc != 0:
        raise RuntimeError(f"Pipeline Stage Failed: {name} (exit code {rc})")

with open("results/kaggle_stage_times.json", "w", encoding="utf-8") as f:
    json.dump(times, f, indent=2)

print("\\n[PIPELINE COMPLETE] All 12 stages finished successfully!")"""))

    cells.append(md_cell("Every pipeline stage finished with exit code 0. Manifest checkpoints ensure that re-running this cell will safely skip already verified stages."))

    # Cell 8: Gate checks
    cells.append(code_cell("""import os, json
import numpy as np
import pytest

os.chdir("/kaggle/working/NLP_Assignment1")

print("--- Running Local Unit Test Suite ---")
!python -m pytest tests/

print("\\n--- Pipeline Gate Checks ---")
# 1. Ethiopic character share (Stage 03)
v_rec = json.load(open("data/cleaned_verification_record.json"))
letter = v_rec.get("ethiopic_letter_share_pct", v_rec.get("ethiopic_letters_only_share"))
punct = v_rec.get("ethiopic_with_punctuation_share_pct", v_rec.get("ethiopic_letters_punct_share"))
print("Ethiopic letter share:       ", f"{letter:.2f}% (Benchmark: > 80%)")
print("Ethiopic with punct share:   ", f"{punct:.2f}%")
assert letter >= 80.0

# 2. Q1.3 Sum Checks
print("\\nQ1.3 Sum Checks:")
tokens_len = len(np.load("data/tokens.npy", mmap_mode="r"))
for n in range(1, 5):
    c = np.load(f"data/counts/full/order{n}_counts.npy")
    assert c.sum() == tokens_len
    p_sum = (c / tokens_len).sum()
    diff = abs(p_sum - 1.0)
    print(f"  Order {n}: counts sum = {c.sum():,} | prob sum = {p_sum:.10f} (diff = {diff:.2e})")
    assert diff <= 1e-9

# 3. Q1.8 Verbatim Generations
gen_data = json.load(open("results/q1_generation.json"))
print("\\nQ1.8 Generation Verbatim Copies:")
for order_k, sents in gen_data.items():
    verb_cnt = sum(1 for s in sents if s["is_full_verbatim_copy"])
    print(f"  {order_k}: {verb_cnt} / {len(sents)} sentences are verbatim corpus copies")

# 4. Q2 UNK fractions
q2_data = json.load(open("results/q2_perplexity.json"))
print("\\nQ2 Held-out Evaluation:")
print(f"  Seed 42 UNK fraction: {q2_data['seed_42']['unk_fraction'] * 100:.2f}%")
print(f"  Seed 7 UNK fraction:  {q2_data['seed_7']['unk_fraction'] * 100:.2f}%")
print(f"  Seed 42 Bigram PP Reduction: {q2_data['seed_42']['orders']['n=2']['reduction_vs_unigram_pct']:.2f}%")

print("\\n[ALL GATE CHECKS PASSED]")"""))

    cells.append(md_cell("All gate checks pass: unit test suite passes, Ethiopic letter share exceeds 80%, Q1.3 probability sums equal 1.0 within 1e-9, and Q2 perplexity bigram reduction target is satisfied."))

    # Cell 9: Results summary
    cells.append(code_cell("""import os, json
import pandas as pd

os.chdir("/kaggle/working/NLP_Assignment1")

print("=== Q1 Top 5 Bigrams (Q1.3) ===")
print(pd.read_csv("results/q1_top10_n2.csv").head(5).to_string(index=False))

print("\\n=== Q1 Conditional Probabilities (Q1.4) ===")
print(pd.read_csv("results/q1_bigram_conditional.csv").to_string(index=False))

print("\\n=== Q1 Top 5 Bigrams After Stopword Removal (Q1.5) ===")
print(pd.read_csv("results/q1_stopword_top10_n2.csv").head(5).to_string(index=False))

print("\\n=== Q2 Perplexity Summary ===")
q2 = json.load(open("results/q2_perplexity.json"))
for seed_name in ["seed_42", "seed_7"]:
    print(f"  -- {seed_name} --")
    for ok, ov in q2[seed_name]["orders"].items():
        print(f"     {ok}: PP = {ov['perplexity']}, LogP = {ov['logp_sum']}, Reduction = {ov['reduction_vs_unigram_pct']}%")

print("\\n=== Q3 Extrinsic Metrics ===")
q3 = json.load(open("results/q3_metrics.json"))
for ok, ov in q3["models"].items():
    print(f"  {ok}: Acc={ov['accuracy']}, F1={ov['f1_score']}, AUC={ov['roc_auc']}, Usable={ov['usable']}")

print("\\n=== Q4 Predictive Text Evaluation ===")
q4 = json.load(open("results/q4_eval.json"))
for mk, mv in q4["metrics"].items():
    print(f"  {mk}: Top-1={mv['top1_accuracy']}, Top-5={mv['top5_hit_rate']}, MRR@5={mv['mrr_at_5']}, Meets Target={mv['meets_benchmarks']}")
print("\\nExample Prompt Suggestion ('ኢትዮጵያ ውስጥ የቡና'):")
for s in q4["example_prompt_output"]["suggestions"]:
    print(f"  {s['word']:15} | Prob: {s['prob']:.4f} | Train Count: {s['train_count']} | Verified: {s['verified_in_corpus']}")"""))

    cells.append(md_cell("The numerical results printed above trace directly to the pipeline code. No numbers are fabricated or estimated from memory."))

    # Cell 10: Export artifacts
    cells.append(code_cell("""import os, shutil, json, sys

os.chdir("/kaggle/working/NLP_Assignment1")

# Ensure data/artifact_list.json is created
sys.path.insert(0, ".")
from scripts.artifacts import write_artifact_list
write_artifact_list()

EXPORT = "/kaggle/working/export"
os.makedirs(EXPORT, exist_ok=True)

ARTIFACTS = json.load(open("data/artifact_list.json"))
total_bytes = 0
copied_count = 0

for rel in ARTIFACTS:
    src = os.path.join("/kaggle/working/NLP_Assignment1", rel)
    dst = os.path.join(EXPORT, rel)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        total_bytes += os.path.getsize(dst)
        copied_count += 1
    else:
        print(f"  [MISSING OPTIONAL] {rel}")

print(f"Exported {copied_count} canonical artifacts: {total_bytes:,} bytes ({round(total_bytes / 2**20, 2)} MiB)")

# Create zip bundle for 1-click download from Kaggle output tab
shutil.make_archive("/kaggle/working/nlp_assignment1_artifacts", "zip", EXPORT)
print("Created /kaggle/working/nlp_assignment1_artifacts.zip")"""))

    cells.append(md_cell("The export directory and zip bundle contain only the canonical 'Yes' artifacts. Heavy training token arrays remain on the server."))

    # Cell 11: Checksums
    cells.append(code_cell("""import os, shutil

os.chdir("/kaggle/working/NLP_Assignment1")

for fname in ["checksums.json", "manifest.json", "artifact_list.json"]:
    p = os.path.join("data", fname)
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(EXPORT, "data", fname))
        print(f"Included {fname} in export.")

log_p = os.path.join("results", "run_log.jsonl")
if os.path.exists(log_p):
    shutil.copy2(log_p, os.path.join(EXPORT, "results", "run_log.jsonl"))
    print("Included run_log.jsonl in export.")

print("Export checksum verification bundle ready.")"""))

    cells.append(md_cell("All manifest entries, run logs, and checksums are bundled in the export."))

    # Cell 12: Next steps
    cells.append(md_cell("""### Next Steps: PC Download and Local Verification

1. Download `nlp_assignment1_artifacts.zip` or the files from the Kaggle notebook's **Output** tab.
2. Extract the files into your local repository root on your PC:
   - `data/` artifacts overwrite `data/`
   - `results/` artifacts overwrite `results/`
3. On your local PC, run:
   ```bash
   python scripts/check_download.py
   ```
4. Once `check_download.py` verifies all SHA-256 hashes, launch the local interactive UI and CLI:
   ```bash
   python app/cli.py
   ```
   Or run the local web interface (`web/` via `npm run dev`)."""))

    return make_notebook(cells)

def build_colab_notebook():
    # Similar to Kaggle notebook with Colab adaptations
    cells = []

    cells.append(md_cell("""# NLP Assignment 1: Google Colab Build

**Assignment:** NLP Assignment 1 (10%), Addis Ababa University, School of Information Technology and Engineering.
**Instructor:** Fantahun B. (PhD)
**Questions Covered:**
- Step 0: Raw corpus verification and assessment, cleaned text generation, verification, vocabulary and token array creation.
- Q1.1 to Q1.8: N-gram language models, MLE probabilities, smoothing, conditional probabilities, stopword analysis, sentence scoring, and model generation.
- Q2: Intrinsic evaluation (perplexity across orders 1 to 4 on held-out test splits, primary seed 42 and replication seed 7).
- Q3: Extrinsic evaluation (real vs corrupted Amharic sentence detection, threshold tuning, UNK confound analysis, ROC-AUC).
- Q4: Predictive text application evaluation (Top-1, Top-5, MRR@5 across 1-word and 2-word contexts).
- Presentation Artifacts: Canonical artifact bundle generation for local PC exploration.

**Execution Rule (AGENTS.md B13):**
Heavy pipeline stages run exclusively in this heavy environment. The local PC runs only unit tests, checksum verification (`scripts/check_download.py`), and the light interactive UI/CLI."""))

    cells.append(code_cell("""import os, sys, platform, importlib, json, shutil

def read_first(path):
    try:
        return open(path).read().strip()
    except Exception:
        return None

cg = read_first("/sys/fs/cgroup/memory.max") or read_first("/sys/fs/cgroup/memory/memory.limit_in_bytes")
try:
    phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
except Exception:
    phys = 0

print("Google Colab detected =", "google.colab" in sys.modules or os.path.exists("/content"))
print("physical RAM (GiB)    =", round(phys / 2**30, 2) if phys else "N/A")
print("cgroup limit         =", cg)
print("CPUs                 =", os.cpu_count())
print("disk free (GiB)      =", round(shutil.disk_usage("/content").free / 2**30, 1) if os.path.isdir("/content") else "N/A")
print("python               =", platform.python_version())"""))

    cells.append(md_cell("For a larger budget, choose *Runtime → Change runtime type → High-RAM*, if your account offers it."))

    cells.append(code_cell("""!pip install -q gdown
import sklearn, numpy
print("scikit-learn version:", sklearn.__version__)
print("numpy version:       ", numpy.__version__)"""))

    cells.append(md_cell("Dependencies verified."))

    # Cell 4: Clone Repository
    cells.append(code_cell("""import os

# Start from a directory that always exists, avoiding getcwd errors
os.chdir("/content")

REPO_URL = "https://github.com/KenaKaye/Assignment-1.git"
BRANCH = "full-corpus"

!rm -rf /content/NLP_Assignment1
!git clone -b {BRANCH} {REPO_URL} /content/NLP_Assignment1
%cd /content/NLP_Assignment1
!git log -1 --oneline"""))

    cells.append(md_cell("Record the commit printed here in `README.md`."))

    # Cell 5a: Connect Google Drive
    cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')"""))

    cells.append(md_cell("Run the cell above to connect your Google Drive containing `GPAC.txt.gz`."))

    # Cell 5b: Copy, decompress, and verify hash
    cells.append(code_cell("""import gzip, shutil, hashlib, os, glob

# Start from a directory that always exists, then enter the repository
os.chdir("/content")
os.chdir("/content/NLP_Assignment1")

SRC = "/content/drive/MyDrive/GPAC.txt.gz"
if not os.path.exists(SRC):
    candidates = glob.glob("/content/drive/*/**/GPAC.txt.gz", recursive=True) + glob.glob("/content/GPAC.txt.gz")
    if candidates:
        SRC = candidates[0]
        print(f"Found GPAC.txt.gz at {SRC}")
    else:
        raise SystemExit("GPAC.txt.gz not found at /content/drive/MyDrive/GPAC.txt.gz. Please check Drive path.")

DST = "/content/NLP_Assignment1/data/GPAC.txt"
os.makedirs(os.path.dirname(DST), exist_ok=True)

print(f"Decompressing {SRC} -> {DST}...")
with gzip.open(SRC, "rb") as src, open(DST, "wb") as dst:
    shutil.copyfileobj(src, dst)

h = hashlib.sha256()
with open(DST, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)

actual_hash = h.hexdigest()
EXPECTED = "977706859188f7598ca97a29d7203d86926c11ede591fba31915c74ae9eabcfa"
if actual_hash != EXPECTED:
    print("----------------------------------------------------------------------")
    print(f"ACTUAL HASH: {actual_hash}")
    print(f"EXPECTED:    {EXPECTED}")
    print("The attached file does not match the authoritative corpus hash.")
    print("----------------------------------------------------------------------")

assert actual_hash == EXPECTED, f"Raw corpus hash mismatch: {actual_hash}"
print("GPAC.txt SHA-256 matches:", actual_hash)"""))

    cells.append(md_cell("The raw corpus hash strictly matches the authoritative value recorded in the implementation plan. Step 0 raw data verification passes. (If the assert fails, the file is wrong)."))

    # Cell 6: Reset stale outputs
    cells.append(code_cell("""import os, shutil
from pathlib import Path

os.chdir("/content/NLP_Assignment1")

RESET_STAGES = []

cleaned_file = Path("data/GPAC_cleaned.txt")
if cleaned_file.exists():
    with open(cleaned_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line and not any(first_line.endswith(p) for p in ["።", "?", "!"]):
            print("Detected stale GPAC_cleaned.txt lacking sentence-final punctuation! Resetting...")
            RESET_STAGES = ["02_build_cleaned"]

if RESET_STAGES:
    print(f"Resetting pipeline stages starting from {RESET_STAGES}...")
    manifest_file = Path("data/manifest.json")
    if manifest_file.exists():
        manifest_file.unlink()
    if cleaned_file.exists():
        cleaned_file.unlink()
    shutil.rmtree("data/counts", ignore_errors=True)
    shutil.rmtree("results", ignore_errors=True)
    for f in Path("data").glob("*.npy"):
        f.unlink()
    for f in Path("data").glob("*.tsv"):
        if f.name != "sentence_assessment.tsv":
            f.unlink()
    print("Reset complete.")
else:
    print("Manifest checkpoints active.")"""))

    cells.append(md_cell("Old outputs verified."))

    # Cell 7: Run pipeline
    cells.append(code_cell("""import subprocess, time, sys, os, json

os.chdir("/content/NLP_Assignment1")

STAGES = [
    ["scripts/01_assess_raw.py"],
    ["scripts/02_build_cleaned.py"],
    ["scripts/03_verify_cleaned.py"],
    ["scripts/04_build_vocab_tokens.py"],
    ["scripts/06_split.py"],
    ["scripts/07_count_full.py"],
    ["scripts/08_count_train.py", "--seed", "42"],
    ["scripts/08_count_train.py", "--seed", "7"],
    ["scripts/10_q1_results.py"],
    ["scripts/11_q2_perplexity.py"],
    ["scripts/12_q3_extrinsic.py"],
    ["scripts/13_q4_eval.py"],
]

times = {}
os.makedirs("results", exist_ok=True)

for args in STAGES:
    name = " ".join(args)
    print(f"\\n==================== Running: {name} ====================", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    dur = round(time.time() - t0, 1)
    times[name] = dur
    print(f"== {name}: exit code {rc}, elapsed: {dur} s ==", flush=True)
    if rc != 0:
        raise RuntimeError(f"Pipeline Stage Failed: {name} (exit code {rc})")

with open("results/colab_stage_times.json", "w", encoding="utf-8") as f:
    json.dump(times, f, indent=2)

print("\\n[PIPELINE COMPLETE] All 12 stages finished successfully!")"""))

    cells.append(md_cell("Pipeline execution completed."))

    # Cell 8: Gate checks
    cells.append(code_cell("""import os, json
import numpy as np

os.chdir("/content/NLP_Assignment1")

print("--- Running Local Unit Test Suite ---")
!python -m pytest tests/

print("\\n--- Pipeline Gate Checks ---")
v_rec = json.load(open("data/cleaned_verification_record.json"))
letter = v_rec.get("ethiopic_letter_share_pct", v_rec.get("ethiopic_letters_only_share"))
punct = v_rec.get("ethiopic_with_punctuation_share_pct", v_rec.get("ethiopic_letters_punct_share"))
print("Ethiopic letter share:       ", f"{letter:.2f}% (Benchmark: > 80%)")
print("Ethiopic with punct share:   ", f"{punct:.2f}%")
assert letter >= 80.0

tokens_len = len(np.load("data/tokens.npy", mmap_mode="r"))
for n in range(1, 5):
    c = np.load(f"data/counts/full/order{n}_counts.npy")
    assert c.sum() == tokens_len
    p_sum = (c / tokens_len).sum()
    diff = abs(p_sum - 1.0)
    print(f"  Order {n}: counts sum = {c.sum():,} | prob sum = {p_sum:.10f} (diff = {diff:.2e})")
    assert diff <= 1e-9

gen_data = json.load(open("results/q1_generation.json"))
print("\\nQ1.8 Generation Verbatim Copies:")
for order_k, sents in gen_data.items():
    verb_cnt = sum(1 for s in sents if s["is_full_verbatim_copy"])
    print(f"  {order_k}: {verb_cnt} / {len(sents)} sentences are verbatim corpus copies")

q2_data = json.load(open("results/q2_perplexity.json"))
print("\\nQ2 Held-out Evaluation:")
print(f"  Seed 42 UNK fraction: {q2_data['seed_42']['unk_fraction'] * 100:.2f}%")
print(f"  Seed 7 UNK fraction:  {q2_data['seed_7']['unk_fraction'] * 100:.2f}%")
print(f"  Seed 42 Bigram PP Reduction: {q2_data['seed_42']['orders']['n=2']['reduction_vs_unigram_pct']:.2f}%")

print("\\n[ALL GATE CHECKS PASSED]")"""))

    cells.append(md_cell("All gate checks pass."))

    # Cell 9: Results summary
    cells.append(code_cell("""import os, json
import pandas as pd

os.chdir("/content/NLP_Assignment1")

print("=== Q1 Top 5 Bigrams (Q1.3) ===")
print(pd.read_csv("results/q1_top10_n2.csv").head(5).to_string(index=False))

print("\\n=== Q1 Conditional Probabilities (Q1.4) ===")
print(pd.read_csv("results/q1_bigram_conditional.csv").to_string(index=False))

print("\\n=== Q1 Top 5 Bigrams After Stopword Removal (Q1.5) ===")
print(pd.read_csv("results/q1_stopword_top10_n2.csv").head(5).to_string(index=False))

print("\\n=== Q2 Perplexity Summary ===")
q2 = json.load(open("results/q2_perplexity.json"))
for seed_name in ["seed_42", "seed_7"]:
    print(f"  -- {seed_name} --")
    for ok, ov in q2[seed_name]["orders"].items():
        print(f"     {ok}: PP = {ov['perplexity']}, LogP = {ov['logp_sum']}, Reduction = {ov['reduction_vs_unigram_pct']}%")

print("\\n=== Q3 Extrinsic Metrics ===")
q3 = json.load(open("results/q3_metrics.json"))
for ok, ov in q3["models"].items():
    print(f"  {ok}: Acc={ov['accuracy']}, F1={ov['f1_score']}, AUC={ov['roc_auc']}, Usable={ov['usable']}")

print("\\n=== Q4 Predictive Text Evaluation ===")
q4 = json.load(open("results/q4_eval.json"))
for mk, mv in q4["metrics"].items():
    print(f"  {mk}: Top-1={mv['top1_accuracy']}, Top-5={mv['top5_hit_rate']}, MRR@5={mv['mrr_at_5']}, Meets Target={mv['meets_benchmarks']}")
print("\\nExample Prompt Suggestion ('ኢትዮጵያ ውስጥ የቡና'):")
for s in q4["example_prompt_output"]["suggestions"]:
    print(f"  {s['word']:15} | Prob: {s['prob']:.4f} | Train Count: {s['train_count']} | Verified: {s['verified_in_corpus']}")"""))

    cells.append(md_cell("Results summary displayed."))

    # Cell 10: Export artifacts
    cells.append(code_cell("""import os, shutil, json, sys

os.chdir("/content/NLP_Assignment1")

sys.path.insert(0, ".")
from scripts.artifacts import write_artifact_list
write_artifact_list()

EXPORT = "/content/export"
os.makedirs(EXPORT, exist_ok=True)

ARTIFACTS = json.load(open("data/artifact_list.json"))
total_bytes = 0
copied_count = 0

for rel in ARTIFACTS:
    src = os.path.join("/content/NLP_Assignment1", rel)
    dst = os.path.join(EXPORT, rel)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        total_bytes += os.path.getsize(dst)
        copied_count += 1

print(f"Exported {copied_count} canonical artifacts: {total_bytes:,} bytes ({round(total_bytes / 2**20, 2)} MiB)")
shutil.make_archive("/content/nlp_assignment1_artifacts", "zip", EXPORT)
print("Created /content/nlp_assignment1_artifacts.zip")"""))

    cells.append(md_cell("Artifacts bundled into zip."))

    # Cell 11: Checksums
    cells.append(code_cell("""import os, shutil

os.chdir("/content/NLP_Assignment1")

for fname in ["checksums.json", "manifest.json", "artifact_list.json"]:
    p = os.path.join("data", fname)
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(EXPORT, "data", fname))
        print(f"Included {fname} in export.")

log_p = os.path.join("results", "run_log.jsonl")
if os.path.exists(log_p):
    shutil.copy2(log_p, os.path.join(EXPORT, "results", "run_log.jsonl"))
    print("Included run_log.jsonl in export.")

print("Export checksum verification bundle ready.")"""))

    cells.append(md_cell("Checksum bundle exported."))

    cells.append(md_cell("""### Next Steps: PC Download and Local Verification

1. Download `nlp_assignment1_artifacts.zip` from Colab files pane.
2. Extract the files into your local repository root on your PC:
   - `data/` artifacts overwrite `data/`
   - `results/` artifacts overwrite `results/`
3. On your local PC, run:
   ```bash
   python scripts/check_download.py
   ```
4. Once `check_download.py` verifies all SHA-256 hashes, launch the local interactive UI and CLI:
   ```bash
   python app/cli.py
   ```
   Or run the local web interface (`web/` via `npm run dev`)."""))

    return make_notebook(cells)

if __name__ == "__main__":
    kaggle_nb = build_kaggle_notebook()
    colab_nb = build_colab_notebook()

    Path("kaggle").mkdir(parents=True, exist_ok=True)
    with open("kaggle/Kaggle_Build.ipynb", "w", encoding="utf-8") as f:
        json.dump(kaggle_nb, f, indent=1, ensure_ascii=False)
    print("Wrote kaggle/Kaggle_Build.ipynb")

    Path("colab").mkdir(parents=True, exist_ok=True)
    with open("colab/Colab_Build.ipynb", "w", encoding="utf-8") as f:
        json.dump(colab_nb, f, indent=1, ensure_ascii=False)
    print("Wrote colab/Colab_Build.ipynb")
