import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

# Ensure font registration
font_path = "figures/NotoSansEthiopic-Regular.ttf"
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found: {font_path}")

font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['Noto Sans Ethiopic', 'DejaVu Sans', 'Arial']

# Stopwords set (30 words)
AMHARIC_STOPWORDS = {
    "ነው", "ላይ", "ውስጥ", "እና", "ጋር", "ወደ", "እንደ", "ስለ", "ይህ", "ነበር",
    "አለ", "እስከ", "ከዚህ", "ሆኖ", "በኋላ", "መካከል", "በኩል", "ብቻ", "አይደለም", "ይህን",
    "ወይም", "ሆኖም", "ነገር", "ግን", "እንዲሁም", "በላይ", "ሌላ", "ሁሉ", "በታች", "በፊት"
}

# 1. Unigrams from vocab.tsv
print("Loading unigrams from data/vocab.tsv...")
unigram_before = {}
unigram_after = {}
with open("data/vocab.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            w = parts[1]
            c = int(parts[2])
            if w in ["<s>", "</s>", "<UNK>"]:
                continue
            if len(unigram_before) < 60:
                unigram_before[w] = c
            if w not in AMHARIC_STOPWORDS and len(unigram_after) < 60:
                unigram_after[w] = c

# 2. Bigrams, Trigrams, 4-grams from results CSVs
def get_ngrams(n, cond):
    fn = f"results/q1_{'stopword_top10' if cond=='after' else 'top10'}_n{n}.csv"
    df = pd.read_csv(fn)
    return dict(zip(df["ngram"], df["count"]))

bigram_before = get_ngrams(2, "before")
bigram_after = get_ngrams(2, "after")
trigram_before = get_ngrams(3, "before")
trigram_after = get_ngrams(3, "after")
fourgram_before = get_ngrams(4, "before")
fourgram_after = get_ngrams(4, "after")

# Unified filter descriptions
FILTER_BEFORE = "Raw Unfiltered Corpus Frequencies (Sentence boundary markers <s>, </s> strictly excluded)"
FILTER_AFTER = "Filtered: Remove n-gram if any token is on 30-stopword list, after stripping bare prepositional clitics | <s>, </s> excluded"

configs = [
    {
        "filename": "figures/q1_6_wordcloud_unigram_before.png",
        "freqs": unigram_before,
        "order_title": "Order n = 1: Unigram Model — Before Stopword Removal",
        "filter": FILTER_BEFORE,
        "source": "data/vocab.tsv & results/q1_top10_n1.csv",
        "top_snippet": "ነው (1,165,953) · ላይ (754,678) · ውስጥ (346,314) · ወደ (331,106) · እና (318,942)",
        "cmap": "copper"
    },
    {
        "filename": "figures/q1_6_wordcloud_unigram_after.png",
        "freqs": unigram_after,
        "order_title": "Order n = 1: Unigram Model — After Stopword Removal",
        "filter": FILTER_AFTER,
        "source": "results/q1_stopword_top10_n1.csv",
        "top_snippet": "ጊዜ (244,117) · ደግሞ (209,377) · አንድ (192,674) · ቤት (182,048) · አቶ (169,169)",
        "cmap": "viridis"
    },
    {
        "filename": "figures/q1_6_wordcloud_bigram_before.png",
        "freqs": bigram_before,
        "order_title": "Order n = 2: Bigram Model — Before Stopword Removal",
        "filter": FILTER_BEFORE,
        "source": "results/q1_top10_n2.csv",
        "top_snippet": "ዓ ም (108,455) · ቀን ዓ (46,525) · ነገር ግን (44,744) · ዶ ር (35,624) · ማለት ነው (32,386)",
        "cmap": "plasma"
    },
    {
        "filename": "figures/q1_6_wordcloud_bigram_after.png",
        "freqs": bigram_after,
        "order_title": "Order n = 2: Bigram Model — After Stopword Removal",
        "filter": FILTER_AFTER,
        "source": "results/q1_stopword_top10_n2.csv",
        "top_snippet": "ዓ ም (108,455) · ቀን ዓ (46,525) · ዶ ር (35,624) · አዲስ አበባ (31,388) · ምክር ቤት (23,905)",
        "cmap": "cividis"
    },
    {
        "filename": "figures/q1_6_wordcloud_trigram_before.png",
        "freqs": trigram_before,
        "order_title": "Order n = 3: Trigram Model — Before Stopword Removal",
        "filter": FILTER_BEFORE,
        "source": "results/q1_top10_n3.csv",
        "top_snippet": "ቀን ዓ ም (45,927) · በ ዓ ም (28,127) · እ ኤ አ (16,731) · ዓ ም ኢሳት (9,789) · ም ኢሳት ዜና (9,700)",
        "cmap": "inferno"
    },
    {
        "filename": "figures/q1_6_wordcloud_trigram_after.png",
        "freqs": trigram_after,
        "order_title": "Order n = 3: Trigram Model — After Stopword Removal",
        "filter": FILTER_AFTER,
        "source": "results/q1_stopword_top10_n3.csv",
        "top_snippet": "ቀን ዓ ም (45,927) · በ ዓ ም (28,127) · እ ኤ አ (16,731) · ዓ ም ጀምሮ (6,395) · ተወካዮች ምክር ቤት (5,598)",
        "cmap": "magma"
    },
    {
        "filename": "figures/q1_6_wordcloud_4gram_before.png",
        "freqs": fourgram_before,
        "order_title": "Order n = 4: 4-gram Model — Before Stopword Removal",
        "filter": FILTER_BEFORE,
        "source": "results/q1_top10_n4.csv",
        "top_snippet": "ቀን ዓ ም ኢሳት (9,786) · ዓ ም ኢሳት ዜና (9,700) · እ ኤ አ በ (8,613) · የካቲት ቀን ዓ ም (2,952)",
        "cmap": "Spectral"
    },
    {
        "filename": "figures/q1_6_wordcloud_4gram_after.png",
        "freqs": fourgram_after,
        "order_title": "Order n = 4: 4-gram Model — After Stopword Removal",
        "filter": FILTER_AFTER,
        "source": "results/q1_stopword_top10_n4.csv",
        "top_snippet": "ቀን ዓ ም ኢሳት (9,786) · ዓ ም ኢሳት ዜና (9,700) · እ ኤ አ በ (8,613) · ከ ዓ ም ጀምሮ (2,866) · አዲስ አበባ ከተማ (2,607)",
        "cmap": "coolwarm"
    },
]

print("Rendering 8 self-contained standalone word cloud figures...")
for cfg in configs:
    # Explicit square-root scaling (relative_scaling=0.5: font_size ∝ √count)
    wc = WordCloud(
        font_path=font_path,
        width=1000,
        height=520,
        background_color="white",
        colormap=cfg["cmap"],
        collocations=False,
        prefer_horizontal=0.9,
        max_words=len(cfg["freqs"]),
        min_font_size=14,
        max_font_size=105,
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(cfg["freqs"])
    
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    
    # Header Banner
    plt.suptitle(cfg["order_title"], fontsize=13.5, fontweight="bold", y=0.97)
    ax.set_title(
        f"{cfg['filter']}\n"
        f"Scale: Square-Root Count Scaling (relative_scaling=0.5; s ∝ √c(w), 14–105 pt) | Font: Google Noto Sans Ethiopic",
        fontsize=9, color="#222222", pad=8
    )
    
    # Dual Footnotes: Top Entities + Clitic & Layout Rules
    footnote_text = (
        f"Data Source: {cfg['source']} | Top Items: {cfg['top_snippet']}\n"
        f"Footnote on Clitic Handling: Amharic prepositions (በ-, ከ-, ለ-, የ-) attach proclitically; forms with attached clitics (e.g. በዚህ, በአዲስ)\n"
        f"are treated as distinct lexical units and retained unless their stripped base matches the stopword list.\n"
        f"Layout Note: Pseudo-random spiral placement (random_state=42; spatial position does not indicate rank). Colormap provides visual contrast."
    )
    fig.text(
        0.5, 0.02, footnote_text, ha="center", fontsize=7.8, color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", lw=0.8)
    )
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    plt.savefig(cfg["filename"])
    plt.close()
    print(f"  Saved standalone: {cfg['filename']}")

# -------------------------------------------------------------
# Generate Composite Panel 1: Orders n=1 and n=2
# (Constructed directly from the standalone figure files to ensure 100% match)
# -------------------------------------------------------------
print("\nGenerating composite panel 1 (n=1 & n=2) from standalone figures...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=250)
panel1_files = [
    ("figures/q1_6_wordcloud_unigram_before.png", axes[0, 0]),
    ("figures/q1_6_wordcloud_unigram_after.png", axes[0, 1]),
    ("figures/q1_6_wordcloud_bigram_before.png", axes[1, 0]),
    ("figures/q1_6_wordcloud_bigram_after.png", axes[1, 1]),
]
for p, ax in panel1_files:
    img = plt.imread(p)
    ax.imshow(img)
    ax.axis("off")

plt.suptitle(
    "Q1.6 Word Cloud Comparison Panel: Unigrams (n=1) and Bigrams (n=2)\n"
    "Before vs. After Stopword Removal (Square-Root Scaling: s ∝ √c(w) | Filter Rule Applied After Stripping Clitics)",
    fontsize=14.5, fontweight="bold", y=0.985
)
plt.tight_layout(rect=[0, 0, 1, 0.965])
panel1_path = "figures/q1_6_wordclouds_panel_unigram_bigram.png"
plt.savefig(panel1_path)
plt.close()
print(f"  Saved composite panel 1: {panel1_path}")

# -------------------------------------------------------------
# Generate Composite Panel 2: Orders n=3 and n=4
# (Constructed directly from the standalone figure files to ensure 100% match)
# -------------------------------------------------------------
print("Generating composite panel 2 (n=3 & n=4) from standalone figures...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=250)
panel2_files = [
    ("figures/q1_6_wordcloud_trigram_before.png", axes[0, 0]),
    ("figures/q1_6_wordcloud_trigram_after.png", axes[0, 1]),
    ("figures/q1_6_wordcloud_4gram_before.png", axes[1, 0]),
    ("figures/q1_6_wordcloud_4gram_after.png", axes[1, 1]),
]
for p, ax in panel2_files:
    img = plt.imread(p)
    ax.imshow(img)
    ax.axis("off")

plt.suptitle(
    "Q1.6 Word Cloud Comparison Panel: Trigrams (n=3) and 4-grams (n=4)\n"
    "Before vs. After Stopword Removal (Square-Root Scaling: s ∝ √c(w) | Filter Rule Applied After Stripping Clitics)",
    fontsize=14.5, fontweight="bold", y=0.985
)
plt.tight_layout(rect=[0, 0, 1, 0.965])
panel2_path = "figures/q1_6_wordclouds_panel_trigram_4gram.png"
plt.savefig(panel2_path)
plt.close()
print(f"  Saved composite panel 2: {panel2_path}")

# -------------------------------------------------------------
# Generate Master 8-Panel Overview
# (Constructed directly from the standalone figure files to ensure 100% match)
# -------------------------------------------------------------
print("Generating master 8-panel overview (all orders) from standalone figures...")
fig, axes = plt.subplots(4, 2, figsize=(16, 24), dpi=200)
all_files = [
    ("figures/q1_6_wordcloud_unigram_before.png", axes[0, 0]),
    ("figures/q1_6_wordcloud_unigram_after.png", axes[0, 1]),
    ("figures/q1_6_wordcloud_bigram_before.png", axes[1, 0]),
    ("figures/q1_6_wordcloud_bigram_after.png", axes[1, 1]),
    ("figures/q1_6_wordcloud_trigram_before.png", axes[2, 0]),
    ("figures/q1_6_wordcloud_trigram_after.png", axes[2, 1]),
    ("figures/q1_6_wordcloud_4gram_before.png", axes[3, 0]),
    ("figures/q1_6_wordcloud_4gram_after.png", axes[3, 1]),
]
for p, ax in all_files:
    img = plt.imread(p)
    ax.imshow(img)
    ax.axis("off")

plt.suptitle(
    "Q1.6 Complete Word Cloud Overview (Orders n = 1, 2, 3, 4)\n"
    "Full Amharic Corpus GPAC (76.1M Tokens | SEED = 42 | Square-Root Scaling: s ∝ √c(w))",
    fontsize=16, fontweight="bold", y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
master_path = "figures/q1_6_wordclouds_all_orders.png"
plt.savefig(master_path)
plt.close()
print(f"  Saved master overview: {master_path}")

print("\nAll 8 standalone figures and 3 composite panels updated and synchronized successfully!")
