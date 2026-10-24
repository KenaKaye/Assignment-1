"""Vectorized interpolated Kneser-Ney language model library.

Implements model training, continuation count extraction, packed prefix indexing,
and vectorized batch probability computation as specified in Section 5.2.
"""

from __future__ import annotations

import numpy as np
from scripts.ngram_lib import group_rows, sentence_positions, context_matrix

# Global vocabulary limit for 64-bit integer packing: 2^21 = 2,097,152
V_LIMIT = 1 << 21
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


def pack(cols: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Pack up to 3 integer column arrays into a single int64 array.

    Formula: pack(c0, c1, c2) = ((c0 * V_LIMIT) + c1) * V_LIMIT + c2.
    Safe within 63-bit signed integer range for up to 3 columns when all IDs < V_LIMIT.
    """
    if isinstance(cols, np.ndarray):
        if cols.ndim == 1:
            return cols.astype(np.int64)
        cols_list = [cols[:, i] for i in range(cols.shape[1])]
    else:
        cols_list = cols

    if not cols_list:
        return np.empty(0, dtype=np.int64)

    val = cols_list[0].astype(np.int64)
    for c in cols_list[1:]:
        val = val * V_LIMIT + c.astype(np.int64)
    return val


class KneserNeyModel:
    """Vectorized Interpolated Kneser-Ney Language Model."""

    def __init__(self) -> None:
        self.max_order: int = 1
        self.D: dict[int, float] = {}
        self.a: dict[int, np.ndarray] = {}
        self.unique_rows: dict[int, np.ndarray] = {}
        self.ctx_keys: dict[int, np.ndarray] = {}
        self.ctx_start: dict[int, np.ndarray] = {}
        self.ctx_n1: dict[int, np.ndarray] = {}
        self.ctx_total: dict[int, np.ndarray] = {}
        self.pair_keys: dict[int, np.ndarray] = {}
        self.p1: np.ndarray = np.empty(0, dtype=np.float64)
        self.top50_unigrams: np.ndarray = np.empty(0, dtype=np.int32)

    @classmethod
    def build_from_counts(
        cls,
        raw_unique_rows: dict[int, np.ndarray],
        raw_counts: dict[int, np.ndarray],
        max_order: int = 4,
        vocab_size: int = V_LIMIT,
    ) -> KneserNeyModel:
        """Construct the Kneser-Ney model from raw counts for orders 1 to max_order.

        Lower orders use continuation counts derived from unique higher-order rows.
        Order max_order uses raw counts.
        """
        model = cls()
        model.max_order = max_order

        # 1. Build count tables a_k and unique_rows_k
        # For k == max_order: raw counts
        model.unique_rows[max_order] = raw_unique_rows[max_order]
        model.a[max_order] = raw_counts[max_order].astype(np.float64)

        # For k < max_order downward: continuation counts N1+(· h w)
        for k in range(max_order - 1, 0, -1):
            higher_unique = model.unique_rows[k + 1]
            # Strip the leading word of the higher-order unique rows
            lower_patterns = higher_unique[:, 1:]
            u_rows, cont_counts = group_rows(lower_patterns)
            model.unique_rows[k] = u_rows
            model.a[k] = cont_counts.astype(np.float64)

        # 2. Compute discounts D_k = n1 / (n1 + 2 * n2)
        for k in range(1, max_order + 1):
            ak = model.a[k]
            n1 = np.count_nonzero(ak == 1.0)
            n2 = np.count_nonzero(ak == 2.0)
            if (n1 + 2 * n2) > 0:
                model.D[k] = float(n1 / (n1 + 2.0 * n2))
            else:
                model.D[k] = 0.75

        # 3. Context groups for orders k >= 2
        for k in range(2, max_order + 1):
            rows = model.unique_rows[k]
            ak = model.a[k]
            n_rows = len(rows)
            if n_rows == 0:
                model.ctx_keys[k] = np.empty(0, dtype=np.int64)
                model.ctx_start[k] = np.empty(0, dtype=np.int64)
                model.ctx_n1[k] = np.empty(0, dtype=np.int64)
                model.ctx_total[k] = np.empty(0, dtype=np.float64)
                model.pair_keys[k] = np.empty(0, dtype=np.int64)
                continue

            # Context is first k-1 columns
            ctx_cols = [rows[:, c] for c in range(k - 1)]
            row_keys = pack(ctx_cols)

            new_ctx = np.ones(n_rows, dtype=bool)
            new_ctx[1:] = row_keys[1:] != row_keys[:-1]
            starts = np.flatnonzero(new_ctx)

            model.ctx_start[k] = starts
            model.ctx_keys[k] = row_keys[starts]
            model.ctx_n1[k] = np.diff(np.append(starts, n_rows)).astype(np.int64)
            model.ctx_total[k] = np.add.reduceat(ak, starts)

            group_idx = np.repeat(np.arange(len(starts), dtype=np.int64), model.ctx_n1[k])
            model.pair_keys[k] = group_idx * V_LIMIT + rows[:, -1].astype(np.int64)

        # 4. Unigram base probability array
        a1 = model.a[1]
        u1 = model.unique_rows[1]
        p1_size = max(int(np.max(u1[:, 0])) + 1 if len(u1) > 0 else 0, vocab_size)
        model.p1 = np.zeros(p1_size, dtype=np.float64)

        total_a1 = a1.sum()
        if total_a1 > 0 and len(u1) > 0:
            words = u1[:, 0]
            model.p1[words] = a1 / total_a1
            # Assert all observed words have positive probability
            assert np.all(model.p1[words] > 0), "Observed tokens in order 1 must have p1 > 0"

        # 5. Precompute top-50 unigrams by continuation count (excluding PAD, EOS, UNK)
        if len(u1) > 0:
            valid_mask = ~np.isin(u1[:, 0], [PAD_ID, EOS_ID, UNK_ID])
            valid_words = u1[valid_mask, 0]
            valid_counts = a1[valid_mask]
            k_top = min(50, len(valid_words))
            if k_top > 0:
                top_part = np.argpartition(-valid_counts, k_top - 1)[:k_top]
                top_sorted = top_part[np.argsort(-valid_counts[top_part])]
                model.top50_unigrams = valid_words[top_sorted]

        return model

    def lookup(
        self,
        k: int,
        ctx_cols: list[np.ndarray],
        words: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized lookup for count c(h, w) in order k table.

        Args:
            k: Order index (>= 2).
            ctx_cols: List of 1D arrays for the context prefix columns.
            words: 1D array of word IDs.

        Returns:
            found: Boolean mask indicating whether the context exists in table k.
            g: Group indices of the context matches.
            cw: Looked-up counts c(h, w), 0.0 where not found.
        """
        B = len(words)
        keys = self.ctx_keys[k]
        if len(keys) == 0:
            return np.zeros(B, dtype=bool), np.zeros(B, dtype=np.int64), np.zeros(B, dtype=np.float64)

        q = pack(ctx_cols)
        g = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        found = keys[g] == q

        pk = g.astype(np.int64) * V_LIMIT + words.astype(np.int64)
        pairs = self.pair_keys[k]
        j = np.minimum(np.searchsorted(pairs, pk), len(pairs) - 1)
        exact = found & (pairs[j] == pk)
        cw = np.where(exact, self.a[k][j], 0.0)
        return found, g, cw

    def prob(
        self,
        ctx: np.ndarray,
        words: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """Compute batch interpolated Kneser-Ney probabilities.

        Args:
            ctx: 2D int64 array of shape (B, >= order - 1), most recent context token last.
            words: 1D int64 or int32 array of target words of length B.
            order: Order to evaluate (1 to max_order).

        Returns:
            1D float64 array of probabilities P(words | ctx).
        """
        B = len(words)
        if B == 0:
            return np.empty(0, dtype=np.float64)

        # Base order 1 probability
        in_bounds = words < len(self.p1)
        safe_words = np.where(in_bounds, words, 0)
        p = np.where(in_bounds, self.p1[safe_words], 0.0)

        for k in range(2, order + 1):
            ctx_len = k - 1
            cols = [ctx[:, ctx.shape[1] - ctx_len + c] for c in range(ctx_len)]
            found, g, cw = self.lookup(k, cols, words)

            tot = self.ctx_total[k][np.minimum(g, len(self.ctx_total[k]) - 1)]
            n1 = self.ctx_n1[k][np.minimum(g, len(self.ctx_n1[k]) - 1)]

            ok = found & (tot > 0)
            tot_safe = np.where(ok, tot, 1.0)
            D = self.D[k]

            new_p = np.maximum(cw - D, 0.0) / tot_safe + (D * n1 / tot_safe) * p
            p = np.where(ok, new_p, p)

        return p

    def get_continuations(self, k: int, ctx_cols: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Return array of observed continuation words for context in table k."""
        if k < 2 or k not in self.ctx_keys or len(self.ctx_keys[k]) == 0:
            return np.empty(0, dtype=np.int32)

        q = pack(ctx_cols)
        if isinstance(q, np.ndarray) and q.ndim > 0:
            q_val = int(q[0])
        else:
            q_val = int(q)

        keys = self.ctx_keys[k]
        idx = np.searchsorted(keys, q_val)
        if idx < len(keys) and keys[idx] == q_val:
            start = self.ctx_start[k][idx]
            n1 = self.ctx_n1[k][idx]
            return self.unique_rows[k][start : start + n1, -1]
        return np.empty(0, dtype=np.int32)

    def score_sequence(
        self,
        tokens: np.ndarray,
        starts: np.ndarray,
        order: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score an entire token sequence vectorized across all positions.

        Args:
            tokens: 1D int32 array of token IDs.
            starts: 1D int64 array of sentence start offsets.
            order: Order to evaluate.

        Returns:
            logp_positions: Natural log probability at each token position.
            sentence_logp: Sum of log probabilities for each sentence.
            sentence_per_token_logp: Per-token log probability (divided by sentence length).
        """
        N = len(tokens)
        if N == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        pos = sentence_positions(starts, N)
        M = context_matrix(tokens, pos, order)
        if order > 1:
            ctx = M[:, :-1]
        else:
            ctx = np.zeros((N, 0), dtype=np.int64)
        words = M[:, -1]

        probs = self.prob(ctx, words, order)
        # Avoid log(0) with minimal clipping floor
        probs = np.maximum(probs, 1e-12)
        logp = np.log(probs)

        lengths = np.diff(np.append(starts, N))
        sentence_ids = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
        sentence_logp = np.bincount(sentence_ids, weights=logp, minlength=len(starts))
        sentence_per_token_logp = sentence_logp / np.maximum(lengths, 1)

        return logp, sentence_logp, sentence_per_token_logp


def kn_prob(model: KneserNeyModel, ctx: np.ndarray, words: np.ndarray, order: int) -> np.ndarray:
    """Batch interpolated Kneser-Ney probability evaluation function."""
    return model.prob(ctx, words, order)
