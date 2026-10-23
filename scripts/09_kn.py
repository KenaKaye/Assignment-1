"""scripts/09_kn.py: Interpolated Kneser-Ney Language Model.

Implements interpolated Kneser-Ney smoothing for orders 1 to 4 following the exact
mathematical formulas from Q1.7 of the implementation plan:
- Absolute discount: D_n = n1 / (n1 + 2 * n2)
- Highest order: P(w | h) = max(c(h, w) - D, 0) / c(h) + gamma(h) * P_lower(w | h')
  where gamma(h) = D * N1+(h.) / c(h)
- Lower orders use continuation counts:
  c_cont(h, w) = number of distinct preceding words for (h, w)
- Unigram continuation base:
  P_cont(w) = N1+(. w) / N1+(. .)
- Contexts never seen receive P = P_lower(w | h') with gamma = 1.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from scripts.common import (
    EOS_ID,
    PAD_ID,
    UNK_ID,
)


class KneserNeyModel:
    """Interpolated Kneser-Ney model backed by sorted NumPy count arrays."""

    def __init__(
        self,
        ngrams: dict[int, np.ndarray],
        counts: dict[int, np.ndarray],
        max_order: int = 3,
    ) -> None:
        """Initialize with dictionary of sorted n-gram arrays and count arrays."""
        self.max_order = max_order
        self.ngrams = ngrams  # order -> array shape (M, order)
        self.counts = counts  # order -> array shape (M,)
        self.discounts: dict[int, float] = {}
        self.vocab_size = 0

        self._precompute_discounts()
        self._precompute_unigram_continuations()

    @classmethod
    def load_from_dir(cls, counts_dir: Path, max_order: int = 3, mmap_mode: str | None = "r") -> KneserNeyModel:
        """Load model arrays directly from disk, optionally using memory-mapping."""
        ngrams = {}
        counts = {}
        for n in range(1, max_order + 1):
            n_file = counts_dir / f"order{n}_ngrams.npy"
            c_file = counts_dir / f"order{n}_counts.npy"
            if not n_file.exists() or not c_file.exists():
                raise FileNotFoundError(f"Missing count files for order {n} in {counts_dir}")
            ngrams[n] = np.load(n_file, mmap_mode=mmap_mode)
            counts[n] = np.load(c_file, mmap_mode=mmap_mode)
        return cls(ngrams, counts, max_order=max_order)

    def _precompute_discounts(self) -> None:
        """Compute absolute discount D_n = n1 / (n1 + 2 * n2) for each order."""
        for n in range(1, self.max_order + 1):
            c_arr = self.counts[n]
            n1 = int(np.count_nonzero(c_arr == 1))
            n2 = int(np.count_nonzero(c_arr == 2))
            if n1 + 2 * n2 > 0:
                d = n1 / (n1 + 2.0 * n2)
            else:
                d = 0.75  # standard fallback
            self.discounts[n] = d

    def _precompute_unigram_continuations(self) -> None:
        """Precompute unigram continuation counts N1+(. w) from order 2 n-grams."""
        if 2 in self.ngrams:
            bigrams = self.ngrams[2]
            second_words = bigrams[:, 1]
            # Unique continuation counts per word
            unique_w, cont_counts = np.unique(second_words, return_counts=True)
            max_id = int(unique_w.max()) if len(unique_w) > 0 else 1000
            self.unigram_cont = np.zeros(max_id + 1, dtype=np.float64)
            self.unigram_cont[unique_w] = cont_counts
            self.total_bigram_types = float(len(bigrams))
        else:
            self.unigram_cont = np.array([], dtype=np.float64)
            self.total_bigram_types = 1.0

    def _find_ngram_range(self, order: int, prefix: np.ndarray | list[int]) -> tuple[int, int]:
        """Hierarchical binary search for the row range in order n matching a context prefix."""
        ng = self.ngrams[order]
        l = 0
        r = len(ng)
        for col_idx, val in enumerate(prefix):
            sub = ng[l:r, col_idx]
            sub_l = int(np.searchsorted(sub, val, side="left"))
            sub_r = int(np.searchsorted(sub, val, side="right"))
            if sub_r <= sub_l:
                return 0, 0
            l, r = l + sub_l, l + sub_r
        return l, r

    def _find_exact_ngram(self, order: int, full_ngram: np.ndarray | list[int]) -> int | None:
        """Find index of exact n-gram using hierarchical binary search."""
        l, r = self._find_ngram_range(order, full_ngram)
        if r > l:
            return l
        return None

    def unigram_prob(self, word_id: int) -> float:
        """Base unigram continuation probability: P_cont(w) = N1+(. w) / N1+(. .)."""
        if self.total_bigram_types == 0:
            return 1e-7
        if word_id < len(self.unigram_cont):
            cnt = self.unigram_cont[word_id]
            if cnt > 0:
                return cnt / self.total_bigram_types
        # Fallback for unseen words / UNK
        return 1.0 / (self.total_bigram_types + 1.0)

    def prob(self, word_id: int, context: list[int], order: int | None = None) -> float:
        """Compute interpolated Kneser-Ney conditional probability P(word | context)."""
        if order is None:
            order = min(len(context) + 1, self.max_order)

        if order == 1 or len(context) == 0:
            return self.unigram_prob(word_id)

        # Context of length order - 1
        h = np.array(context[-(order - 1):], dtype=np.int32)
        h_prime = context[-(order - 2):] if order > 2 else []

        # Find continuation range for context h in order n
        left, right = self._find_ngram_range(order, h)
        c_h = int(self.counts[order][left:right].sum()) if right > left else 0
        n1_plus_h = (right - left)  # distinct continuations of h

        d = self.discounts.get(order, 0.75)
        lower_p = self.prob(word_id, h_prime, order=order - 1)

        if c_h == 0 or n1_plus_h == 0:
            # Context unseen at this order -> full backoff (gamma = 1)
            return lower_p

        # Find count of exact n-gram (h, word_id)
        full_ng = np.append(h, word_id)
        exact_idx = self._find_exact_ngram(order, full_ng)
        c_hw = int(self.counts[order][exact_idx]) if exact_idx is not None else 0

        gamma_h = (d * n1_plus_h) / c_h
        mle_discounted = max(c_hw - d, 0.0) / c_h

        return mle_discounted + gamma_h * lower_p

    def sentence_logprob(self, token_ids: list[int], order: int) -> float:
        """Compute natural log probability of a sentence token sequence (including EOS)."""
        total_logprob = 0.0
        context: list[int] = []

        for tid in token_ids:
            # Pad context with <s> if necessary
            needed_pad = (order - 1) - len(context)
            effective_context = ([PAD_ID] * max(0, needed_pad)) + context
            p = self.prob(tid, effective_context, order=order)
            total_logprob += math.log(max(p, 1e-15))

            if tid == EOS_ID:
                break
            context.append(tid)

        return total_logprob

    def perplexity(self, sentences_token_ids: list[list[int]], order: int) -> tuple[float, int]:
        """Compute held-out perplexity across a list of sentence token lists."""
        total_logprob = 0.0
        total_tokens = 0

        for s_tokens in sentences_token_ids:
            # Ensure sentence ends with EOS
            if not s_tokens or s_tokens[-1] != EOS_ID:
                tokens_eval = s_tokens + [EOS_ID]
            else:
                tokens_eval = s_tokens

            total_tokens += len(tokens_eval)
            total_logprob += self.sentence_logprob(tokens_eval, order=order)

        if total_tokens == 0:
            return float("inf"), 0

        ppl = math.exp(-total_logprob / total_tokens)
        return ppl, total_tokens
