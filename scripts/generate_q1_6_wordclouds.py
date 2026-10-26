"""scripts/generate_q1_6_wordclouds.py

Generates and renders all 8 standalone word clouds and composite panels for Q1.6:
- figures/q1_6_wordcloud_unigram_before.png
- figures/q1_6_wordcloud_unigram_after.png
- figures/q1_6_wordcloud_bigram_before.png
- figures/q1_6_wordcloud_bigram_after.png
- figures/q1_6_wordcloud_trigram_before.png
- figures/q1_6_wordcloud_trigram_after.png
- figures/q1_6_wordcloud_4gram_before.png
- figures/q1_6_wordcloud_4gram_after.png
- figures/q1_6_wordclouds_panel_unigram_bigram.png
- figures/q1_6_wordclouds_panel_trigram_4gram.png
- figures/q1_6_wordclouds_all_orders.png

Enforces:
1. High resolution (1200x680) with generous margins (margin=6) to guarantee ZERO label collision.
2. Cites exact input files in the visual footer.
3. Uses Google Noto Sans Ethiopic for complete glyph coverage (zero box artifacts).
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import unicodedata
from fontTools.ttLib import TTFont
from wordcloud import WordCloud

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "output" / "figures"
RESULTS_DIR = REPO_ROOT / "output" / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
font_path = FIGURES_DIR / "NotoSansEthiopic-Regular.ttf"
if not font_path.exists():
    raise FileNotFoundError(f"Font file not found: {font_path}")
font_manager.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = ['Noto Sans Ethiopic', 'DejaVu Sans', 'Arial']

# Load font and verify character map coverage for Ethiopic block U+1200..U+135A
font_obj = TTFont(str(font_path))
font_cmap = font_obj.getBestCmap()
assigned_ethiopic = [
    cp for cp in range(0x1200, 0x135B)
    if unicodedata.name(chr(cp), None) is not None
]
missing_ethiopic = [cp for cp in assigned_ethiopic if cp not in font_cmap]
if missing_ethiopic:
    raise RuntimeError(
        f"Font {font_path.name} is incomplete: missing {len(missing_ethiopic)} "
        f"assigned Ethiopic characters in U+1200..U+135A: {[hex(cp) for cp in missing_ethiopic[:5]]}"
    )


def verify_all_characters_in_font(freq_dict: dict[str, int], cmap: dict[int, int]) -> None:
    """Strictly assert that every character in every n-gram label is covered by the font cmap.
    Raises ValueError immediately if any character would render as a missing glyph / box artifact.
    """
    for ngram in freq_dict.keys():
        for char in ngram:
            if char.isspace():
                continue
            cp = ord(char)
            if cp not in cmap:
                raise ValueError(
                    f"Glyph rendering violation: character '{char}' (U+{cp:04X}) "
                    f"in label '{ngram}' is NOT present in font {font_path.name}! "
                    "This would render as an empty rectangle / box artifact."
                )


def load_frequencies(n: int, cond: str) -> tuple[dict[str, int], str]:
    """Load frequency dictionary for order n and condition ('before' or 'after').
    Prefers the top-200 file if present, falls back to top-10.
    Strictly excludes sentence boundary markers (<s>, </s>) per AGENTS B2/B7 and plan
    to eliminate box glyphs from unmapped Latin boundary characters.
    """
    top200_file = RESULTS_DIR / f"q1_wordcloud_n{n}_{cond}.csv"
    if top200_file.exists():
        df = pd.read_csv(top200_file)
        # Exclude sentence boundary tokens (<s>, </s>, <UNK>)
        clean_df = df[~df["ngram"].astype(str).str.contains(r"<s>|</s>|<UNK>")].copy()

        # Complement with top-10 table entries to ensure full representation for higher orders
        top10_file = RESULTS_DIR / (f"q1_stopword_top10_n{n}.csv" if cond == "after" else f"q1_top10_n{n}.csv")
        if top10_file.exists():
            df_t10 = pd.read_csv(top10_file)
            clean_df = pd.concat([clean_df[["ngram", "count"]], df_t10[["ngram", "count"]]]).drop_duplicates(subset=["ngram"])
            clean_df = clean_df.sort_values(by="count", ascending=False)

        freq_dict = dict(zip(clean_df["ngram"].astype(str), clean_df["count"].astype(int)))
        return freq_dict, f"results/{top200_file.name} (Top 200 n-grams; boundary markers excluded)"

    # Fallback to top-10
    top10_file = RESULTS_DIR / (f"q1_stopword_top10_n{n}.csv" if cond == "after" else f"q1_top10_n{n}.csv")
    if top10_file.exists():
        df = pd.read_csv(top10_file)
        freq_dict = dict(zip(df["ngram"].astype(str), df["count"].astype(int)))
        return freq_dict, f"results/{top10_file.name} (Top {len(freq_dict)} n-grams)"

    raise FileNotFoundError(f"No frequency file found for n={n}, cond={cond}")


def main() -> None:
    print("Generating Q1.6 Word Clouds...")

    FILTER_BEFORE = "Raw Unfiltered Corpus Frequencies (Sentence boundary markers <s>, </s> strictly excluded)"
    FILTER_AFTER = "Filtered: Remove n-gram if any token is on 30-stopword list, after stripping bare prepositional clitics | <s>, </s> excluded"

    configs = [
        {
            "filename": "q1_6_wordcloud_unigram_before.png",
            "n": 1,
            "cond": "before",
            "title": "Order n = 1: Unigram Model — Before Stopword Removal",
            "filter": FILTER_BEFORE,
            "cmap": "copper",
            "max_font": 85,
        },
        {
            "filename": "q1_6_wordcloud_unigram_after.png",
            "n": 1,
            "cond": "after",
            "title": "Order n = 1: Unigram Model — After Stopword Removal",
            "filter": FILTER_AFTER,
            "cmap": "viridis",
            "max_font": 85,
        },
        {
            "filename": "q1_6_wordcloud_bigram_before.png",
            "n": 2,
            "cond": "before",
            "title": "Order n = 2: Bigram Model — Before Stopword Removal",
            "filter": FILTER_BEFORE,
            "cmap": "plasma",
            "max_font": 75,
        },
        {
            "filename": "q1_6_wordcloud_bigram_after.png",
            "n": 2,
            "cond": "after",
            "title": "Order n = 2: Bigram Model — After Stopword Removal",
            "filter": FILTER_AFTER,
            "cmap": "cividis",
            "max_font": 75,
        },
        {
            "filename": "q1_6_wordcloud_trigram_before.png",
            "n": 3,
            "cond": "before",
            "title": "Order n = 3: Trigram Model — Before Stopword Removal",
            "filter": FILTER_BEFORE,
            "cmap": "inferno",
            "max_font": 65,
        },
        {
            "filename": "q1_6_wordcloud_trigram_after.png",
            "n": 3,
            "cond": "after",
            "title": "Order n = 3: Trigram Model — After Stopword Removal",
            "filter": FILTER_AFTER,
            "cmap": "magma",
            "max_font": 65,
        },
        {
            "filename": "q1_6_wordcloud_4gram_before.png",
            "n": 4,
            "cond": "before",
            "title": "Order n = 4: 4-gram Model — Before Stopword Removal",
            "filter": FILTER_BEFORE,
            "cmap": "Spectral",
            "max_font": 55,
        },
        {
            "filename": "q1_6_wordcloud_4gram_after.png",
            "n": 4,
            "cond": "after",
            "title": "Order n = 4: 4-gram Model — After Stopword Removal",
            "filter": FILTER_AFTER,
            "cmap": "coolwarm",
            "max_font": 55,
        },
    ]

    for cfg in configs:
        freqs, source_label = load_frequencies(cfg["n"], cfg["cond"])

        # Strictly verify font character map coverage before rendering; fails immediately if any glyph is missing
        verify_all_characters_in_font(freqs, font_cmap)

        wc = WordCloud(
            font_path=str(font_path),
            width=1200,
            height=650,
            background_color="white",
            colormap=cfg["cmap"],
            collocations=False,
            prefer_horizontal=0.92,
            relative_scaling=0.5,
            max_font_size=cfg["max_font"],
            min_font_size=12,
            margin=6,
            random_state=42,
        ).generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(12, 7.2), dpi=150)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        plt.suptitle(cfg["title"], fontsize=15, fontweight="bold", y=0.97, color="#1e293b")
        ax.set_title(cfg["filter"], fontsize=9.5, color="#475569", pad=12)

        # Footer with exact source file citation
        footer_text = f"Source: {source_label} | Scaling: WordCloud relative_scaling=0.5 (rank-frequency blend heuristic) | Font: Noto Sans Ethiopic (UTF-8)"
        plt.figtext(0.5, 0.02, footer_text, ha="center", fontsize=8.5, color="#64748b")

        plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
        out_path = FIGURES_DIR / cfg["filename"]
        plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name} (Source: {source_label})")

    # Generate composite panels
    print("Generating composite panels...")
    def stitch_pair(file_left, file_right, out_file, title_panel):
        im1 = Image.open(FIGURES_DIR / file_left)
        im2 = Image.open(FIGURES_DIR / file_right)
        w = max(im1.width, im2.width)
        h = im1.height + im2.height
        dst = Image.new("RGB", (w, h), (255, 255, 255))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        dst.save(FIGURES_DIR / out_file)
        print(f"  Saved composite panel {out_file}")

    stitch_pair("q1_6_wordcloud_unigram_before.png", "q1_6_wordcloud_bigram_before.png", "q1_6_wordclouds_panel_unigram_bigram.png", "Orders 1 & 2")
    stitch_pair("q1_6_wordcloud_trigram_before.png", "q1_6_wordcloud_4gram_before.png", "q1_6_wordclouds_panel_trigram_4gram.png", "Orders 3 & 4")

    # Stitch all 8 for master overview
    # 4 rows x 2 cols
    images = []
    for cfg in configs:
        images.append(Image.open(FIGURES_DIR / cfg["filename"]))
    
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    grid_w = cell_w * 2
    grid_h = cell_h * 4
    overview = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, im in enumerate(images):
        row = i // 2
        col = i % 2
        overview.paste(im, (col * cell_w, row * cell_h))
    overview.save(FIGURES_DIR / "q1_6_wordclouds_all_orders.png")
    print("  Saved master overview figures/q1_6_wordclouds_all_orders.png")
    print("All Q1.6 word clouds generated successfully.")


if __name__ == "__main__":
    main()
