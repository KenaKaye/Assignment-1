"""Shared Prediction Engine for Amharic Next-Word Suggestion.

Course: Natural Language Processing | Assignment 1 (Q4)
Follows Docs/NLP_Assignment1_Implementation_Plan.md and Docs/NLP_Assignment1_UI_Plan.md.

Uses memory-mapped training-split count arrays (data/counts/train_seed42/)
with Kneser-Ney smoothing and binary search candidate generation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "app") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "app"))

from scripts.common import (
    EOS_ID,
    PAD_ID,
    UNK_ID,
    WORD_RE,
)
from app.model_store import get_model_store
from suggest import AmharicPredictor, ETHIOPIC_WORD_REGEX

_FALLBACK_PREDICTOR: Optional[AmharicPredictor] = None


def get_fallback_predictor() -> AmharicPredictor:
    """Fallback in-memory predictor if Colab count arrays have not yet been downloaded."""
    global _FALLBACK_PREDICTOR
    if _FALLBACK_PREDICTOR is not None:
        return _FALLBACK_PREDICTOR

    import pickle

    pkl_path = ROOT_DIR / "models" / "ngram_predictor.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                _FALLBACK_PREDICTOR = pickle.load(f)
            return _FALLBACK_PREDICTOR
        except Exception:
            pass

    from main import load_training_sentences

    train_sents = load_training_sentences()
    predictor = AmharicPredictor()
    predictor.train(train_sents)
    _FALLBACK_PREDICTOR = predictor
    return _FALLBACK_PREDICTOR


def get_predictor():
    """Returns the active predictor or model store."""
    store = get_model_store()
    if store.is_ready:
        return store
    return get_fallback_predictor()


def suggest_candidates(text: str, n: int = 3, k: int = 5) -> Dict[str, Any]:
    """Suggest up to k next words given input text, adhering to Section 6.2 of UI Plan."""
    t_start = time.perf_counter()
    tokens = WORD_RE.findall(text) if text else []
    context_len = max(1, n - 1)
    input_tail = tokens[-context_len:] if tokens else []

    store = get_model_store()
    if store.is_ready and store.kn_model is not None:
        # Use primary memory-mapped Kneser-Ney engine
        kn = store.kn_model
        w2id = store.w2id
        id2w = store.id2w

        ctx_ids = [w2id.get(w, UNK_ID) for w in input_tail]
        candidates = set()
        model_used = 1
        is_fallback = False

        # Check order continuations
        if n == 3 and len(ctx_ids) >= 2:
            h = np.array(ctx_ids[-2:], dtype=np.int32)
            l, r = kn._find_ngram_range(3, h)
            if r > l:
                candidates.update(kn.ngrams[3][l:r, 2].tolist())
                model_used = 3

        if len(candidates) < k and len(ctx_ids) >= 1:
            h1 = np.array([ctx_ids[-1]], dtype=np.int32)
            l1, r1 = kn._find_ngram_range(2, h1)
            if r1 > l1:
                candidates.update(kn.ngrams[2][l1:r1, 1].tolist())
                if model_used == 1:
                    model_used = 2

        if len(candidates) < k:
            is_fallback = True
            top_unigrams = kn.ngrams[1][:50, 0].tolist()
            candidates.update(top_unigrams)

        candidates.discard(PAD_ID)
        candidates.discard(EOS_ID)
        candidates.discard(UNK_ID)

        scored = []
        for cid in candidates:
            p = kn.prob(cid, ctx_ids, order=n)
            scored.append((id2w.get(cid, f"<ID:{cid}>"), float(p)))

        scored.sort(key=lambda x: x[1], reverse=True)
        suggestions = [
            {"word": w, "probability": round(p, 4)}
            for w, p in scored[:k]
        ]
        strategy = f"kneser_ney_order_{model_used}" + ("_fallback" if is_fallback else "")
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "input_tail": input_tail,
            "requested_n": n,
            "model_used": model_used,
            "fallback": is_fallback or (model_used < n),
            "strategy": strategy,
            "latency_ms": round(latency_ms, 2),
            "suggestions": suggestions,
        }

    # Fallback to in-memory predictor
    predictor = get_fallback_predictor()
    raw_suggs, strategy, _ = predictor.suggest(text, context_len=context_len, top_k=k)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    if "trigram" in strategy and "backoff" not in strategy and "fallback" not in strategy:
        model_used = 3
    elif "bigram" in strategy and "fallback" not in strategy:
        model_used = 2
    else:
        model_used = 1

    is_fallback = (model_used < n) or ("fallback" in strategy) or ("backoff" in strategy)
    suggestions = [
        {"word": w, "probability": round(float(p), 4)}
        for w, p in raw_suggs
    ]

    return {
        "input_tail": input_tail,
        "requested_n": n,
        "model_used": model_used,
        "fallback": is_fallback,
        "strategy": strategy,
        "latency_ms": round(latency_ms, 2),
        "suggestions": suggestions,
    }
