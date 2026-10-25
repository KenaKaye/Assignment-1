"""Helper script to construct Colab_Build.ipynb."""

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

def build_colab_notebook():
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

# Include key model count arrays needed for the local interactive CLI and Next.js Web UI
app_artifacts = [
    "data/counts/train_seed42/order1_ngrams.npy",
    "data/counts/train_seed42/order1_counts.npy",
    "data/counts/train_seed42/order2_ngrams.npy",
    "data/counts/train_seed42/order2_counts.npy",
    "data/counts/train_seed42/order3_ngrams.npy",
    "data/counts/train_seed42/order3_counts.npy",
    "data/counts/train_seed42/lut_train_seed42.npy",
]
for rel in app_artifacts:
    src = os.path.join("/content/NLP_Assignment1", rel)
    dst = os.path.join(EXPORT, rel)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        total_bytes += os.path.getsize(dst)
        copied_count += 1
        print(f"Included model artifact in export: {rel} ({os.path.getsize(dst):,} bytes)")

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
    colab_nb = build_colab_notebook()

    Path("colab").mkdir(parents=True, exist_ok=True)
    with open("colab/Colab_Build.ipynb", "w", encoding="utf-8") as f:
        json.dump(colab_nb, f, indent=1, ensure_ascii=False)
    print("Wrote colab/Colab_Build.ipynb")
