"""
Interactive CLI for Amharic Next-Word Suggestion Application
Course: Natural Language Processing | Assignment 1 (Q4)
Follows Docs/NLP_Assignment1_Implementation_Plan.md and Docs/NLP_Assignment1_UI_Plan.md.
"""

from __future__ import annotations

import os
import sys

# Ensure UTF-8 stdout and stdin
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "app") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from predictor import get_predictor, suggest_candidates
from model_store import get_model_store


def run_interactive_app():
    print("=" * 65)
    print("  Amharic Next-Word Suggestion & Predictive Text System (Q4)  ")
    print("=" * 65)
    print("Initializing predictor engine...")
    predictor = get_predictor()
    store = get_model_store()

    v_size = getattr(predictor, "vocab_size", getattr(predictor, "V", 0))
    if store.is_ready:
        print(f"Pipeline model loaded successfully! ({v_size:,} words in training vocabulary)")
    else:
        print(f"Predictor ready! ({v_size:,} words in vocabulary)")
        print("(Note: To use full memory-mapped Kneser-Ney counts, download Colab export bundle)")

    print("-" * 65)
    print("Instructions:")
    print("  - Type an incomplete Amharic sentence (e.g., 'ኢትዮጵያ ውስጥ የቡና')")
    print("  - Select a suggestion by entering its number (1 to 5)")
    print("  - Or type your own next word to continue composing")
    print("  - Type 'exit', 'quit', or 'q' to stop.")
    print("-" * 65)

    current_sentence = ""
    last_suggs: list[dict] = []

    while True:
        try:
            if not current_sentence:
                user_input = input("\nEnter starting Amharic text: ").strip()
            else:
                print(f"\nCurrent text: [ {current_sentence} ]")
                user_input = input("Select [1-5] to pick suggestion, or type next word/exit: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('q', 'quit', 'exit'):
                if current_sentence:
                    print(f"\nFinal composed text: {current_sentence} ።")
                print("Exiting application. ደህና ሁኑ!")
                break

            # If user selected a number 1-5 from previous suggestions
            if user_input.isdigit() and 1 <= int(user_input) <= len(last_suggs):
                selected_idx = int(user_input) - 1
                picked_word = last_suggs[selected_idx]["word"]
                current_sentence = (current_sentence + " " + picked_word).strip()
            else:
                # User typed new text or custom next word
                if current_sentence:
                    current_sentence = (current_sentence + " " + user_input).strip()
                else:
                    current_sentence = user_input

            # Generate top-5 suggestions for current sentence
            res = suggest_candidates(current_sentence, n=3, k=5)
            last_suggs = res.get("suggestions", [])
            strategy = res.get("strategy", "ngram")
            latency_ms = res.get("latency_ms", 0.0)

            print(f"\nTop 5 Next-Word Suggestions ({strategy}, {latency_ms:.2f} ms):")
            if not last_suggs:
                print("  (No candidate continuations found)")
            for rank, item in enumerate(last_suggs, start=1):
                w = item["word"]
                p = item["probability"]
                print(f"  [{rank}] {w:<15} (P = {p:.4f})")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. ደህና ሁኑ!")
            break


if __name__ == "__main__":
    run_interactive_app()
