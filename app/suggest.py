"""
Amharic Predictive Text and Next-Word Suggestion Engine
Course: Natural Language Processing | Assignment 1 (Q4)

"""

import os
import sys
import time
import re
from collections import Counter, defaultdict

ETHIOPIC_WORD_REGEX = re.compile(r'[\u1200-\u135A]+')

class AmharicPredictor:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.ctx2_counts = defaultdict(int)
        self.ctx3_counts = defaultdict(int)
        
        self.bigram_candidates = defaultdict(dict)
        self.trigram_candidates = defaultdict(dict)
        self.top_unigrams = []
        self.total_unigrams = 0
        self.V = 0
        self.is_ready = False

    def train(self, sentences):
        """Train n-gram models and precompute fast candidate index."""
        t0 = time.time()
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.ctx2_counts.clear()
        self.ctx3_counts.clear()
        
        for s in sentences:
            padded2 = ['<s>'] + s + ['</s>']
            padded3 = ['<s>', '<s>'] + s + ['</s>']
            for w in s:
                self.unigram_counts[w] += 1
            for i in range(len(padded2) - 1):
                self.bigram_counts[(padded2[i], padded2[i+1])] += 1
            for i in range(len(padded3) - 2):
                self.trigram_counts[(padded3[i], padded3[i+1], padded3[i+2])] += 1

        for (w1, w2), cnt in self.bigram_counts.items():
            self.ctx2_counts[w1] += cnt

        for (w1, w2, w3), cnt in self.trigram_counts.items():
            self.ctx3_counts[(w1, w2)] += cnt

        self.total_unigrams = sum(self.unigram_counts.values())
        self.V = len(self.unigram_counts)
        self.top_unigrams = [w for w, _ in self.unigram_counts.most_common(50) if w not in ('<s>', '</s>')][:20]

        # Precompute forward candidate lookup maps (filtering boundary tokens)
        self.bigram_candidates = defaultdict(dict)
        for (w1, w2), cnt in self.bigram_counts.items():
            if w2 not in ('<s>', '</s>'):
                prob = (cnt + 0.5) / (self.ctx2_counts[w1] + 0.5 * self.V)
                self.bigram_candidates[w1][w2] = prob

        self.trigram_candidates = defaultdict(dict)
        for (w1, w2, w3), cnt in self.trigram_counts.items():
            if w3 not in ('<s>', '</s>'):
                prob = (cnt + 0.5) / (self.ctx3_counts[(w1, w2)] + 0.5 * self.V)
                self.trigram_candidates[(w1, w2)][w3] = prob

        self.is_ready = True
        elapsed = time.time() - t0
        return elapsed

    def suggest(self, text, context_len=2, top_k=5):
        """
        Suggest top-k next words for given input text.
        Returns:
            list of tuples: (word, normalized_prob)
            strategy: string indicating strategy used ('trigram', 'bigram', 'fallback')
            latency: response time in seconds
        """
        t_start = time.perf_counter()
        tokens = ETHIOPIC_WORD_REGEX.findall(text)
        
        candidates = {}
        strategy = "unigram_fallback"
        
        # 1. Trigram context (last 2 words) if context_len >= 2
        if context_len >= 2 and len(tokens) >= 2:
            ctx2 = (tokens[-2], tokens[-1])
            if ctx2 in self.trigram_candidates:
                strategy = "trigram"
                for w, p in self.trigram_candidates[ctx2].items():
                    candidates[w] = p
                    
        # 2. Backoff to Bigram context (last 1 word) if fewer than 3 candidates
        if len(candidates) < 3 and len(tokens) >= 1:
            ctx1 = tokens[-1]
            if ctx1 in self.bigram_candidates:
                if not candidates:
                    strategy = "bigram"
                else:
                    strategy = "trigram_with_bigram_backoff"
                for w, p in self.bigram_candidates[ctx1].items():
                    if w not in candidates:
                        candidates[w] = 0.5 * p  # backoff penalty
                        
        # 3. Fallback to frequent unigrams if still fewer than top_k
        if len(candidates) < top_k:
            if not candidates:
                strategy = "unigram_fallback"
            else:
                strategy += "_with_unigram_fallback"
            for w in self.top_unigrams:
                if w not in candidates:
                    p_u = self.unigram_counts.get(w, 1) / self.total_unigrams
                    candidates[w] = 0.05 * p_u
                    if len(candidates) >= top_k:
                        break

        # Filter boundary tokens, sort, and take top_k
        clean_candidates = {w: p for w, p in candidates.items() if w not in ('<s>', '</s>')}
        top_candidates = sorted(clean_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Normalize probabilities across top-k for clear presentation
        sum_p = sum(p for _, p in top_candidates)
        normalized = [(w, p / sum_p if sum_p > 0 else 1.0 / top_k) for w, p in top_candidates]
        
        latency = time.perf_counter() - t_start
        return normalized, strategy, latency

    def evaluate_on_sentences(self, test_sentences, context_len=2, top_k=5):
        """Evaluate Top-1 accuracy, Top-5 hit rate, and MRR on test sentences."""
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        rr_sum = 0.0
        total_lat = 0.0
        
        valid_eval = [s for s in test_sentences if len(s) >= 3]
        for s in valid_eval:
            prompt = " ".join(s[:-1])
            target = s[-1]
            
            suggs, _, lat = self.suggest(prompt, context_len=context_len, top_k=top_k)
            total_lat += lat
            
            words = [w for w, _ in suggs]
            if target in words:
                rank = words.index(target) + 1
                if rank == 1:
                    top1_hits += 1
                if rank <= 3:
                    top3_hits += 1
                if rank <= 5:
                    top5_hits += 1
                rr_sum += 1.0 / rank

        N = len(valid_eval)
        return {
            "N": N,
            "top1_acc": top1_hits / N if N > 0 else 0.0,
            "top3_hit": top3_hits / N if N > 0 else 0.0,
            "top5_hit": top5_hits / N if N > 0 else 0.0,
            "mrr": rr_sum / N if N > 0 else 0.0,
            "avg_latency_ms": (total_lat / N) * 1000 if N > 0 else 0.0
        }
