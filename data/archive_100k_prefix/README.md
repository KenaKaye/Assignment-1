# Retired 100k-Prefix Splits

The files in this directory (`split_indices_seed42.json`, `split_indices_seed7.json`) were sampled from only the first 100,000 sentences of the corpus.
Per the approved 2026-10-24 amendment, they are retired and must NOT be used for any full-corpus results.
The canonical splits covering the entire corpus are produced by `scripts/06_split.py` using identical-sentence text grouping to eliminate train/test leakage.
