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

    pkl_candidates = [
        ROOT_DIR / "models" / "ngram_predictor.pkl",
        ROOT_DIR / "archive_100k" / "models" / "ngram_predictor.pkl",
    ]
    for pkl_path in pkl_candidates:
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    _FALLBACK_PREDICTOR = pickle.load(f)
                return _FALLBACK_PREDICTOR
            except Exception:
                pass

    predictor = AmharicPredictor()
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
        from scripts.kn_lib import pack
        # Candidate generation
        candidates = set()
        model_used = 1
        is_fallback = False

        if n == 3 and len(ctx_ids) >= 2:
            q = pack(ctx_ids[-2:])
            keys = kn.ctx_keys[3]
            g = int(np.searchsorted(keys, q))
            if g < len(keys) and keys[g] == q:
                st = kn.ctx_start[3][g]
                cnt = kn.ctx_n1[3][g]
                candidates.update(kn.unique_rows[3][st : st + cnt, 2].tolist())
                model_used = 3

        if len(candidates) < k and len(ctx_ids) >= 1:
            q1 = ctx_ids[-1]
            keys = kn.ctx_keys[2]
            g1 = int(np.searchsorted(keys, q1))
            if g1 < len(keys) and keys[g1] == q1:
                st1 = kn.ctx_start[2][g1]
                cnt1 = kn.ctx_n1[2][g1]
                candidates.update(kn.unique_rows[2][st1 : st1 + cnt1, 1].tolist())
                if model_used == 1:
                    model_used = 2

        if len(candidates) < k:
            is_fallback = True
            valid_mask = kn.unique_rows[1][:, 0] >= 3
            v_words = kn.unique_rows[1][valid_mask, 0]
            v_counts = kn.a[1][valid_mask]
            k_top = min(50, len(v_counts))
            if k_top > 0:
                top_k_idx = np.argpartition(v_counts, -k_top)[-k_top:]
                candidates.update(v_words[top_k_idx].tolist())

        candidates.discard(PAD_ID)
        candidates.discard(EOS_ID)
        candidates.discard(UNK_ID)

        cands_arr = np.array(list(candidates), dtype=np.int32)
        if len(cands_arr) > 0:
            eff_ctx = ctx_ids[-(n - 1):] if n > 1 else []
            ctx_cols = [np.full(len(cands_arr), eff_ctx[c], dtype=np.int64) for c in range(len(eff_ctx))]
            probs = kn.prob(ctx_cols, cands_arr, order=n)
            sort_order = np.lexsort((cands_arr, -probs))
            top_cands = cands_arr[sort_order[:k]]
            top_probs = probs[sort_order[:k]]
            suggestions = [
                {"word": id2w.get(cid, f"<ID:{cid}>"), "probability": round(float(p), 4)}
                for cid, p in zip(top_cands, top_probs)
            ]
        else:
            suggestions = []
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
