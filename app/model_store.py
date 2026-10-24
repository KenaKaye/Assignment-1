"""app/model_store.py: Memory-mapped model and vocabulary loader.

Under Docs/NLP_Assignment1_UI_Plan.md Section 4.1:
- Memory-maps data/counts/train_seed42/ and data/counts/train_seed42/vocab_train.tsv once.
- Provides binary-search lookups for n-gram counts and context row ranges.
- Zero-copy read-only access with sub-millisecond query latency and near-zero memory footprint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

# Add repo root to import scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.kn_lib import KneserNeyModel

TRAIN_COUNTS_DIR = ROOT_DIR / "data" / "counts" / "train_seed42"
VOCAB_FILE = TRAIN_COUNTS_DIR / "vocab_train.tsv"

_STORE_INSTANCE: ModelStore | None = None


class ModelStore:
    """Manages memory-mapped training-split count arrays and vocabulary."""

    def __init__(self, counts_dir: Path = TRAIN_COUNTS_DIR) -> None:
        self.counts_dir = counts_dir
        self.w2id: dict[str, int] = {}
        self.id2w: dict[int, str] = {}
        self.kn_model: KneserNeyModel | None = None
        self._is_ready = False

        self._load()

    def _load(self) -> None:
        vocab_path = self.counts_dir / "vocab_train.tsv"
        order1_path = self.counts_dir / "order1_ngrams.npy"

        if not vocab_path.exists() or not order1_path.exists():
            # Training artifacts not yet built/downloaded from Colab
            self._is_ready = False
            return

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    tid = int(parts[0])
                    word = parts[1]
                    self.w2id[word] = tid
                    self.id2w[tid] = word

        # Determine available orders
        max_order = 3
        for o in range(1, 5):
            if (self.counts_dir / f"order{o}_ngrams.npy").exists():
                max_order = o

        # Memory-map n-gram and count arrays read-only
        self.kn_model = KneserNeyModel.load_from_dir(self.counts_dir, max_order=max_order, mmap_mode="r")
        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def vocab_size(self) -> int:
        return len(self.w2id)


def get_model_store() -> ModelStore:
    """Singleton getter for the ModelStore instance."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = ModelStore()
    return _STORE_INSTANCE
