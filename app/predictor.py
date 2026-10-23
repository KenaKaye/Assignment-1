"""
Shared Prediction Engine for Amharic Next-Word Suggestion
Course: Natural Language Processing | Assignment 1 (Q4)

"""

import os
import sys
import time
import pickle
import re
from typing import List, Dict, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "app") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from suggest import AmharicPredictor, ETHIOPIC_WORD_REGEX

_PREDICTOR_INSTANCE: Optional[AmharicPredictor] = None

def get_predictor() -> AmharicPredictor:
    """
    Singleton getter for the AmharicPredictor instance.
    Loads from models/ngram_predictor.pkl if present, otherwise trains from data/GPAC.txt.
    """
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is not None:
        return _PREDICTOR_INSTANCE

    pkl_path = os.path.join(ROOT_DIR, "models", "ngram_predictor.pkl")
    if os.path.exists(pkl_path):
        t0 = time.time()
        print(f"[predictor] Loading serialized model from {pkl_path}...")
        try:
            with open(pkl_path, "rb") as f:
                _PREDICTOR_INSTANCE = pickle.load(f)
            print(f"[predictor] Model loaded in {time.time() - t0:.2f}s ({_PREDICTOR_INSTANCE.V:,} words).")
            return _PREDICTOR_INSTANCE
        except Exception as e:
            print(f"[predictor] Warning: Failed to load pickle ({e}). Falling back to training.")

    # Fallback: train on the fly
    from main import load_training_sentences
    print("[predictor] Compiling model in-memory from data/GPAC.txt...")
    train_sents = load_training_sentences()
    predictor = AmharicPredictor()
    t_train = predictor.train(train_sents)
    print(f"[predictor] Model compiled in {t_train:.2f}s ({predictor.V:,} words).")
    _PREDICTOR_INSTANCE = predictor
    return _PREDICTOR_INSTANCE


def suggest_candidates(text: str, n: int = 3, k: int = 5) -> Dict[str, Any]:
    """
    Suggest up to k next words given input text, adhering to the UI Plan specification.
    
    Args:
        text: Input string in Amharic.
        n: Order of n-gram model requested (2 for Bigram, 3 for Trigram). Default 3.
        k: Maximum number of suggestions to return (1 to 5). Default 5.
        
    Returns:
        Dictionary adhering to Section 6.2 of Docs/NLP_Assignment1_UI_Plan.md.
    """
    predictor = get_predictor()
    t_start = time.perf_counter()
    
    # Extract Ethiopic tokens
    tokens = ETHIOPIC_WORD_REGEX.findall(text) if text else []
    
    # Context length for order n: n-1 prior words
    context_len = max(1, n - 1)
    input_tail = tokens[-context_len:] if tokens else []
    
    raw_suggs, strategy, _ = predictor.suggest(text, context_len=context_len, top_k=k)
    latency_ms = (time.perf_counter() - t_start) * 1000.0
    
    # Determine model_used based on strategy string
    if "trigram" in strategy and "backoff" not in strategy and "fallback" not in strategy:
        model_used = 3
    elif "bigram" in strategy and "fallback" not in strategy:
        model_used = 2
    elif "unigram" in strategy:
        model_used = 1
    else:
        # Fallback or hybrid
        if "trigram" in strategy:
            model_used = 3
        elif "bigram" in strategy:
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
        "suggestions": suggestions
    }
