"""
Train and Export Script: Amharic N-gram Language Model
Course: Natural Language Processing | Assignment 1 (Q4)

Usage:
    python train_model.py

This script:
1. Loads the 90,000 deterministic training sentences from data/GPAC.txt
   using the saved train/test split (data/split_indices_seed42.json).
2. Fits the unigram, bigram, and trigram models with Add-k smoothing.
3. Precomputes fast forward lookup candidate indices.
4. Serializes the trained model object to models/ngram_predictor.pkl.
"""

import os
import sys
import time
import pickle

# Ensure workspace root and app/ are in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from main import load_training_sentences
from suggest import AmharicPredictor

def main():
    print("=" * 65)
    print("  Amharic N-gram Model Exporter (models/ngram_predictor.pkl)  ")
    print("=" * 65)
    
    corpus_path = os.path.join(ROOT_DIR, "data", "GPAC.txt")
    split_path = os.path.join(ROOT_DIR, "data", "split_indices_seed42.json")
    models_dir = os.path.join(ROOT_DIR, "models")
    output_path = os.path.join(models_dir, "ngram_predictor.pkl")
    
    # 1. Prerequisite verification
    if not os.path.exists(corpus_path):
        print(f"\n[ERROR] Corpus file not found at: {corpus_path}")
        print("Please place 'GPAC.txt' inside the 'data/' folder.")
        print("See data/README.md for the Google Drive download link.\n")
        sys.exit(1)
        
    if not os.path.exists(split_path):
        print(f"\n[ERROR] Split file not found at: {split_path}")
        sys.exit(1)
        
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Load training split
    print("\n[1/3] Loading training sentences from data/GPAC.txt...")
    t0_load = time.time()
    train_sentences = load_training_sentences()
    elapsed_load = time.time() - t0_load
    print(f"      Loaded {len(train_sentences):,} sentences in {elapsed_load:.2f}s.")
    
    # 3. Train AmharicPredictor
    print("\n[2/3] Compiling n-gram models and forward candidate index...")
    predictor = AmharicPredictor()
    t_train = predictor.train(train_sentences)
    print(f"      Training completed in {t_train:.2f}s.")
    print(f"      Vocabulary Size (V):      {predictor.V:,} words")
    print(f"      Bigram Contexts Indexed:  {len(predictor.bigram_candidates):,}")
    print(f"      Trigram Contexts Indexed: {len(predictor.trigram_candidates):,}")
    
    # 4. Serialize to disk
    print(f"\n[3/3] Serializing model to: {output_path} ...")
    t0_dump = time.time()
    with open(output_path, "wb") as f:
        pickle.dump(predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed_dump = time.time() - t0_dump
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Saved in {elapsed_dump:.2f}s ({size_mb:.2f} MB)")
    
    # Quick sanity check
    print("\n--- Model Verification Test ---")
    test_phrase = "የኢትዮጵያ ፌዴራላዊ"
    suggestions, strategy, lat = predictor.suggest(test_phrase, top_k=3)
    print(f"Test query: '{test_phrase}'")
    print(f"Strategy:   {strategy} ({lat*1000:.2f} ms)")
    print(f"Top 3:      {suggestions}")
    
    print("\n[SUCCESS] Model successfully built and saved to models/ngram_predictor.pkl\n")

if __name__ == "__main__":
    main()
