"""
Interactive CLI for Amharic Next-Word Suggestion Application
Course: Natural Language Processing | Assignment 1 (Q4)

"""

import sys
import os
import json
import re

# Ensure UTF-8 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suggest import AmharicPredictor, ETHIOPIC_WORD_REGEX

def load_training_sentences():
    """Load sentences according to saved split."""
    split_path = "data/split_indices_seed42.json"
    corpus_path = "data/GPAC.txt" if os.path.exists("data/GPAC.txt") else "Docs/GPAC.txt"
    
    if not os.path.exists(split_path) or not os.path.exists(corpus_path):
        print(f"Error: Missing {split_path} or {corpus_path}.")
        sys.exit(1)
        
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_set = set(splits["train"])
    
    sentence_split_regex = re.compile(r'[።?!]|\u1361\u1361+')
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        buffer = ""
        while chunk := f.read(2 * 1024 * 1024):
            buffer += chunk
            parts = sentence_split_regex.split(buffer)
            for s in parts[:-1]:
                tokens = ETHIOPIC_WORD_REGEX.findall(s)
                if len(tokens) >= 2:
                    sentences.append(tokens)
                    if len(sentences) >= 100000:
                        break
            if len(sentences) >= 100000:
                break
            buffer = parts[-1]
            
    train_sentences = [sentences[i] for i in splits["train"] if i < len(sentences)]
    return train_sentences

def run_interactive_app():
    print("=" * 65)
    print("  Amharic Next-Word Suggestion & Predictive Text System (Q4)  ")
    print("=" * 65)
    print("Initializing model from corpus training partition...")
    train_data = load_training_sentences()
    
    predictor = AmharicPredictor()
    t_train = predictor.train(train_data)
    print(f"Model ready in {t_train:.2f}s! ({predictor.V:,} words in vocabulary)")
    print("-" * 65)
    print("Instructions:")
    print("  - Type an incomplete Amharic sentence (e.g., 'ኢትዮጵያ ውስጥ የቡና')")
    print("  - Select a suggestion by entering its number (1 to 5)")
    print("  - Or type your own next word to continue composing")
    print("  - Type 'exit', 'quit', or 'q' to stop.")
    print("-" * 65)

    current_sentence = ""
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
                print(f"\nFinal composed text: {current_sentence} ።")
                print("Exiting application. ደህና ሁኑ!")
                break

            # If user selected a number 1-5 from previous suggestions
            if user_input.isdigit() and 1 <= int(user_input) <= 5 and 'last_suggs' in locals():
                selected_idx = int(user_input) - 1
                picked_word = last_suggs[selected_idx][0]
                current_sentence = (current_sentence + " " + picked_word).strip()
            else:
                # User typed new text or custom next word
                if current_sentence:
                    current_sentence = (current_sentence + " " + user_input).strip()
                else:
                    current_sentence = user_input

            # Generate top-5 suggestions for current sentence
            suggs, strategy, latency = predictor.suggest(current_sentence, context_len=2, top_k=5)
            last_suggs = suggs

            print(f"\nTop 5 Next-Word Suggestions ({strategy}, {latency*1000:.2f}ms):")
            for rank, (word, prob) in enumerate(suggs, 1):
                c = predictor.unigram_counts.get(word, 0)
                print(f"  [{rank}] {word:<15} (P = {prob:.4f}, corpus freq = {c:,})")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. ደህና ሁኑ!")
            break

if __name__ == "__main__":
    run_interactive_app()
