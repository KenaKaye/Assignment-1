"""
Amharic Predictive Text CLI Application (Q4)
Course: Natural Language Processing | Assignment 1

"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "app") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from main import run_interactive_app

if __name__ == "__main__":
    run_interactive_app()
