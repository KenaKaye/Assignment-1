import json
from pathlib import Path

def create_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code(source):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Cell 1: Intro
    nb["cells"].append(md("""# NLP Assignment 1: Full Pipeline Execution & Gradio Application
**Course:** Natural Language Processing | Language Models | N-grams  
**Instructor:** Fantahun B. (PhD) | School of IT & Engineering, Addis Ababa University  

This notebook performs all assignment execution steps after the initial build / preprocessing using data retrieved directly from your Google Drive:
1. **Google Drive Integration:** Links dataset from Drive folder `1-8A7dbaRnYfcIZv8YUBqqvOkltWMX8kc`.
2. **Q1 Execution:** N-gram analysis, Top-10 tables, Stopword filtering, Sentence probabilities, and Wordclouds.
3. **Q2 Evaluation:** Intrinsic evaluation (Perplexity across $N=1,2,3,4$ on held-out test splits).
4. **Q3 Evaluation:** Extrinsic evaluation (Real vs Corrupted Amharic detection, ROC curves, F1-scores).
5. **Q4 Evaluation & Gradio App:** Next-word prediction benchmarks (Top-1, Top-5, MRR@5) and an interactive **Gradio** web application (`share=True`)."""))

    # Cell 2: Setup & Clone
    nb["cells"].append(code("""import os, sys, platform, shutil

os.chdir("/content")
REPO_URL = "https://github.com/KenaKaye/Assignment-1.git"
BRANCH = "full-corpus"

if not os.path.exists("/content/NLP_Assignment1"):
    print("Cloning repository...")
    !git clone -b {BRANCH} {REPO_URL} /content/NLP_Assignment1
else:
    print("Repository exists, pulling latest...")
    %cd /content/NLP_Assignment1
    !git pull origin {BRANCH}

%cd /content/NLP_Assignment1
sys.path.insert(0, "/content/NLP_Assignment1")

print("Installing required packages...")
!pip install -q gradio gdown matplotlib scikit-learn pandas
print("Python:", platform.python_version())"""))

    # Cell 3: Google Drive Data Linking
    nb["cells"].append(code("""from google.colab import drive
import os, shutil, glob
from pathlib import Path

# Mount Google Drive
drive.mount('/content/drive', force_remount=False)

DRIVE_FOLDER_ID = "1-8A7dbaRnYfcIZv8YUBqqvOkltWMX8kc"
TARGET_DATA_DIR = Path("/content/NLP_Assignment1/output/data")
TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)
Path("/content/NLP_Assignment1/output/results").mkdir(parents=True, exist_ok=True)
Path("/content/NLP_Assignment1/output/figures").mkdir(parents=True, exist_ok=True)

print("Searching for Google Drive data folder...")

# Search Google Drive for dataset files
drive_candidates = glob.glob("/content/drive/MyDrive/**/counts", recursive=True) + \
                   glob.glob("/content/drive/MyDrive/**/vocab.tsv", recursive=True) + \
                   glob.glob(f"/content/drive/*/**/{DRIVE_FOLDER_ID}*", recursive=True)

found_source = None
for c in drive_candidates:
    p = Path(c)
    source_candidate = p if p.is_dir() and p.name != "counts" else p.parent
    if (source_candidate / "counts").exists() or (source_candidate / "vocab.tsv").exists():
        found_source = source_candidate
        break

if found_source and found_source.exists():
    print(f"Found dataset in Google Drive at: {found_source}")
    for item in found_source.iterdir():
        dst = TARGET_DATA_DIR / item.name
        if not dst.exists():
            if item.is_dir():
                try:
                    os.symlink(item, dst, target_is_directory=True)
                    print(f"Linked directory: {item.name}")
                except Exception:
                    shutil.copytree(item, dst)
                    print(f"Copied directory: {item.name}")
            else:
                try:
                    os.symlink(item, dst)
                    print(f"Linked file: {item.name}")
                except Exception:
                    shutil.copy2(item, dst)
                    print(f"Copied file: {item.name}")
else:
    print("Downloading dataset from Google Drive folder via gdown...")
    !gdown --folder "https://drive.google.com/drive/folders/1-8A7dbaRnYfcIZv8YUBqqvOkltWMX8kc" -O /content/drive_data
    downloaded = Path("/content/drive_data")
    if downloaded.exists():
        for item in downloaded.iterdir():
            dst = TARGET_DATA_DIR / item.name
            if not dst.exists():
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
                print(f"Imported: {item.name}")

print("\\n--- Current output/data content ---")
for f in sorted(TARGET_DATA_DIR.iterdir()):
    sz = sum(x.stat().st_size for x in f.rglob('*')) if f.is_dir() else f.stat().st_size
    print(f"  {f.name:35s} : {sz / (1024*1024):.2f} MB")"""))

    # Cell 4: Check or build counts if missing
    nb["cells"].append(code("""train_counts_seed42 = TARGET_DATA_DIR / "counts" / "train_seed42"
if not train_counts_seed42.exists() or not (train_counts_seed42 / "order1_counts.npy").exists():
    print("Training counts not detected in Drive. Running counting stages...")
    !python scripts/07_count_full.py
    !python scripts/08_count_train.py --seed 42
    !python scripts/08_count_train.py --seed 7
else:
    print("Precomputed training count arrays verified successfully.")"""))

    # Cell 5: Q1
    nb["cells"].append(code("""print("--- Running Q1 Analysis ---")
!python scripts/10_q1_results.py
!python scripts/generate_q1_6_wordclouds.py

import pandas as pd
import json

print("\\n[Q1.3 Top 5 Trigrams Before Stopword Removal]")
if os.path.exists("output/results/q1_top10_n3.csv"):
    df_n3 = pd.read_csv("output/results/q1_top10_n3.csv")
    display(df_n3.head(5))

print("\\n[Q1.5 Top 5 Trigrams After Stopword Removal]")
if os.path.exists("output/results/q1_stopword_top10_n3.csv"):
    df_sw = pd.read_csv("output/results/q1_stopword_top10_n3.csv")
    display(df_sw.head(5))

print("\\n[Q1.7 Sentence Probability for: 'ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።']")
if os.path.exists("output/results/q1_sentence_probs.csv"):
    df_probs = pd.read_csv("output/results/q1_sentence_probs.csv")
    display(df_probs[df_probs['sentence'].str.contains('በአፍሪካ', na=False)])"""))

    # Cell 6: Q2
    nb["cells"].append(code("""print("--- Running Q2 Intrinsic Evaluation ---")
!python scripts/11_q2_perplexity.py

with open("output/results/q2_perplexity.json", "r", encoding="utf-8") as f:
    q2_data = json.load(f)

print("\\nPerplexity Results on Held-Out Test Split (Seed 42):")
for ord_k, v in q2_data["seed_42"]["orders"].items():
    print(f"  {ord_k}: Perplexity = {v['perplexity']:,.2f}  (Reduction vs Unigram: {v['reduction_vs_unigram_pct']:.2f}%)")

print("\\nPerplexity Results on Held-Out Test Split (Seed 7 Replication):")
for ord_k, v in q2_data["seed_7"]["orders"].items():
    print(f"  {ord_k}: Perplexity = {v['perplexity']:,.2f}  (Reduction vs Unigram: {v['reduction_vs_unigram_pct']:.2f}%)")"""))

    # Cell 7: Q3
    nb["cells"].append(code("""print("--- Running Q3 Extrinsic Evaluation ---")
!python scripts/12_q3_extrinsic.py

with open("output/results/q3_metrics.json", "r", encoding="utf-8") as f:
    q3_data = json.load(f)

print("\\nExtrinsic Classification Metrics:")
df_q3 = pd.DataFrame(q3_data)
display(df_q3.T[["accuracy", "precision", "recall", "f1", "roc_auc"]])"""))

    # Cell 8: Q4 Evaluation
    nb["cells"].append(code("""print("--- Running Q4 Offline Benchmark Evaluation ---")
!python scripts/13_q4_eval.py

with open("output/results/q4_eval.json", "r", encoding="utf-8") as f:
    q4_data = json.load(f)

print("\\nQ4 Benchmark Metrics across Context Lengths:")
m = q4_data["metrics"]
for k, res in m.items():
    print(f"Context: {k}")
    print(f"  Top-1 Accuracy : {res['top1_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Hit Rate : {res['top5_hit_rate'] * 100:.2f}%")
    print(f"  MRR@5          : {res['mrr_at_5']:.4f}")"""))

    # Cell 9: Gradio Web App
    nb["cells"].append(code("""import gradio as gr
from app.predictor import suggest_candidates

def predict_next(text, context_n):
    if not text or not text.strip():
        return (
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            "እባክዎ የአማርኛ ቃል ወይም ሐረግ ይጻፉ... (Please enter some Amharic words)"
        )
    
    res = suggest_candidates(text.strip(), n=int(context_n), k=5)
    suggestions = res.get("suggestions", [])
    strategy = res.get("strategy", "kneser_ney")
    latency_ms = res.get("latency_ms", 0.0)
    
    info_text = f"**ስትራቴጂ (Strategy):** `{strategy}` | **የምላሽ ፍጥነት (Latency):** `{latency_ms:.1f} ms` | **የተገኙ እጩዎች (Candidates):** `{len(suggestions)}`"
    
    # Fill 5 buttons
    btn_updates = []
    for i in range(5):
        if i < len(suggestions):
            item = suggestions[i]
            w = item["word"]
            p = item["probability"] * 100
            btn_updates.append(gr.update(value=f"{w}  ({p:.1f}%)", visible=True))
        else:
            btn_updates.append(gr.update(value="", visible=False))
            
    return (*btn_updates, info_text)

def select_word(current_text, button_label, context_n):
    if not button_label:
        return current_text, *predict_next(current_text, context_n)
    word = button_label.split()[0].strip()
    new_text = (current_text.strip() + " " + word).strip() + " "
    return new_text, *predict_next(new_text, context_n)

# Build Gradio Interface
custom_css = \"\"\"
.gradio-container { font-family: 'Noto Sans Ethiopic', 'Segoe UI', Tahoma, sans-serif !important; }
.amharic-box textarea { font-size: 1.25rem !important; line-height: 1.7 !important; }
.cand-btn { font-size: 1.1rem !important; font-weight: 600 !important; }
\"\"\"

with gr.Blocks(title="Amharic Predictive Text - N-Gram LM", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇪🇹 የአማርኛ ቀጣይ ቃል ጠቋሚ (Amharic Predictive Text)")
    gr.Markdown("**NLP Assignment 1 - Question 4** | Addis Ababa University")
    gr.Markdown("ፅሁፍ ሲተይቡ ሞዴሉ በ n-gram የቋንቋ ስታቲስቲክስ ላይ ተመስርቶ ቀጣይ ሊሆኑ የሚችሉ 5 ቃላትን ይጠቁማል። አንድ እጩ ቃል ሲጫኑ በቀጥታ ወደ አረፍተ ነገሩ ይታከላል።")
    
    with gr.Row():
        with gr.Column(scale=4):
            input_text = gr.Textbox(
                label="የአማርኛ አረፍተ ነገር ይጻፉ (Amharic Sentence)",
                placeholder="ለምሳሌ፡ ኢትዮጵያ ውስጥ የቡና...",
                lines=3,
                elem_classes=["amharic-box"]
            )
            
            with gr.Row():
                n_slider = gr.Radio(
                    label="ሞዴል / አውድ (Model Context Order)",
                    choices=[2, 3],
                    value=3,
                    info="2 = Bigram (1-word context), 3 = Trigram (2-word context)"
                )
                clear_btn = gr.Button("አጽዳ (Clear)", variant="secondary")
                
            info_markdown = gr.Markdown("እጩዎችን ለማየት መተየብ ይጀምሩ...")
            
            gr.Markdown("### የተጠቆሙ ቀጣይ ቃላት (Top 5 Suggestions):")
            with gr.Row():
                btn1 = gr.Button("", visible=False, elem_classes=["cand-btn"], variant="primary")
                btn2 = gr.Button("", visible=False, elem_classes=["cand-btn"], variant="primary")
                btn3 = gr.Button("", visible=False, elem_classes=["cand-btn"], variant="primary")
                btn4 = gr.Button("", visible=False, elem_classes=["cand-btn"], variant="secondary")
                btn5 = gr.Button("", visible=False, elem_classes=["cand-btn"], variant="secondary")
                
        with gr.Column(scale=2):
            gr.Markdown("### የሙከራ ምሳሌዎች (Example Prompts):")
            examples = gr.Examples(
                examples=[
                    ["ኢትዮጵያ ውስጥ የቡና", 3],
                    ["በአዲስ አበባ ከተማ", 3],
                    ["የትምህርት ጥራት እና", 3],
                    ["የኢኮኖሚ እድገት እና", 3],
                    ["የጤና ጥበቃ", 2],
                    ["የአየር ንብረት ለውጥ", 3]
                ],
                inputs=[input_text, n_slider]
            )

    cand_buttons = [btn1, btn2, btn3, btn4, btn5]
    
    input_text.change(
        fn=predict_next,
        inputs=[input_text, n_slider],
        outputs=[*cand_buttons, info_markdown]
    )
    
    n_slider.change(
        fn=predict_next,
        inputs=[input_text, n_slider],
        outputs=[*cand_buttons, info_markdown]
    )
    
    clear_btn.click(
        fn=lambda: ("", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "ፅሁፍ ይጻፉ..."),
        inputs=None,
        outputs=[input_text, *cand_buttons, info_markdown]
    )
    
    for btn in cand_buttons:
        btn.click(
            fn=select_word,
            inputs=[input_text, btn, n_slider],
            outputs=[input_text, *cand_buttons, info_markdown]
        )

# Launch with public shareable link
demo.launch(share=True, debug=False)"""))

    out_path = Path("colab/Colab_Pipeline.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully generated {out_path} with {len(nb['cells'])} cells.")

if __name__ == "__main__":
    create_notebook()
