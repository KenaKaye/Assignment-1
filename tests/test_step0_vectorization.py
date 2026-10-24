"""Unit tests for Step 0 vectorization and common helpers on small fixtures.

Adheres to AGENTS.md Rule B13: PC runs unit tests on small fixtures in the repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.common import (
    ETHIOPIC_FILTER_REGEX,
    PAD_ID,
    SENTENCE_SPLIT_REGEX,
    WORD_RE,
)
from scripts.ngram_lib import (
    context_matrix as build_ngram_matrix,
    sentence_positions as compute_pos_in_sent,
    group_rows as count_ngrams_direct,
    group_rows_partitioned as count_ngrams_partitioned,
)


def test_sentence_split_and_word_regex():
    sample = "ኢትዮጵያ ውብ ሀገር ናት። ህዝቧም እንግዳ ተቀባይ ነው! ሰላም ይሁን?"
    segments = SENTENCE_SPLIT_REGEX.split(sample)
    # Delimiters split into 3 segments plus trailing empty
    non_empty = [s.strip() for s in segments if s.strip()]
    assert len(non_empty) == 3

    words = WORD_RE.findall(non_empty[0])
    assert words == ["ኢትዮጵያ", "ውብ", "ሀገር", "ናት"]


def test_ethiopic_character_filter():
    raw_mixed = "ኢትዮጵያ (Ethiopia) 2026 ዓ.ም. #123 ውብ ነው!"
    cleaned = ETHIOPIC_FILTER_REGEX.sub(" ", raw_mixed)
    words = WORD_RE.findall(cleaned)
    assert words == ["ኢትዮጵያ", "ዓ", "ም", "ውብ", "ነው"]
    # Check no Latin or ASCII digits remained
    assert not any(c.isascii() and c.isalnum() for c in cleaned)


def test_ngram_matrix_and_padding():
    # 2 sentences:
    # Sentence 1: [10, 11, 1]  (length 3, ends with EOS=1)
    # Sentence 2: [20, 21, 22, 1] (length 4, ends with EOS=1)
    tokens = np.array([10, 11, 1, 20, 21, 22, 1], dtype=np.int32)
    starts = np.array([0, 3], dtype=np.int64)

    pos_in_sent = compute_pos_in_sent(len(tokens), starts)
    expected_pos = np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(pos_in_sent, expected_pos)

    # Test bigram (n=2)
    matrix_2 = build_ngram_matrix(tokens, pos_in_sent, n=2)
    expected_2 = np.array([
        [PAD_ID, 10],
        [10, 11],
        [11, 1],
        [PAD_ID, 20],
        [20, 21],
        [21, 22],
        [22, 1],
    ], dtype=np.int32)
    np.testing.assert_array_equal(matrix_2, expected_2)

    # Test trigram (n=3)
    matrix_3 = build_ngram_matrix(tokens, pos_in_sent, n=3)
    expected_3 = np.array([
        [PAD_ID, PAD_ID, 10],
        [PAD_ID, 10, 11],
        [10, 11, 1],
        [PAD_ID, PAD_ID, 20],
        [PAD_ID, 20, 21],
        [20, 21, 22],
        [21, 22, 1],
    ], dtype=np.int32)
    np.testing.assert_array_equal(matrix_3, expected_3)


def test_count_ngrams_direct_vs_partitioned():
    # Generate reproducible synthetic sequence
    np.random.seed(42)
    tokens = np.random.randint(3, 50, size=500, dtype=np.int32)
    starts = np.array([0, 100, 250, 400], dtype=np.int64)
    pos_in_sent = compute_pos_in_sent(len(tokens), starts)

    for n in [1, 2, 3, 4]:
        matrix = build_ngram_matrix(tokens, pos_in_sent, n=n)

        u_dir, c_dir = count_ngrams_direct(matrix)
        u_part, c_part = count_ngrams_partitioned(matrix, n_partitions=3)

        assert c_dir.sum() == len(tokens)
        assert c_part.sum() == len(tokens)

        # Confirm exact equality between direct lexsort and partitioned fallback
        np.testing.assert_array_equal(u_dir, u_part)
        np.testing.assert_array_equal(c_dir, c_part)


def test_kneser_ney_model():
    from scripts.kn_lib import KneserNeyModel

    # Toy counts:
    # Unigrams: 3, 4, 5
    # Bigrams: (3, 4), (3, 5), (4, 5)
    ngrams = {
        1: np.array([[3], [4], [5]], dtype=np.int32),
        2: np.array([[3, 4], [3, 5], [4, 5]], dtype=np.int32),
    }
    counts = {
        1: np.array([2, 2, 2], dtype=np.int64),
        2: np.array([1, 1, 2], dtype=np.int64),
    }

    model = KneserNeyModel.build_from_counts(ngrams, counts, max_order=2, vocab_size=10)
    # Check discount
    assert 0.0 < model.D[2] < 1.0

    # Probability of word 4 given context [3]
    p_seen = model.prob(np.array([[3]]), np.array([4]), order=2)[0]
    assert 0.0 < p_seen < 1.0

    # Sequence score
    tokens = np.array([3, 4, 1], dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    logp_pos, sent_logp, per_tok = model.score_sequence(tokens, starts, order=2)
    assert sent_logp[0] < 0.0
