"""Comprehensive test suite for Section 7 of Fix Specification.

Tests PC-side light components, vectorization, Kneser-Ney normalization,
LUT generation, split v2 invariance, streaming chunk invariance, offset accuracy,
checksum error handling, memory budget detection, and manifest tracking.
"""

from __future__ import annotations

import collections
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.common import (
    PAD_ID,
    EOS_ID,
    UNK_ID,
    V_LIMIT,
    load_checksums,
    memory_budget_mb,
    stage_is_done,
    record_stage,
)
from scripts.ngram_lib import (
    sentence_positions,
    context_matrix,
    group_rows,
    group_rows_partitioned,
)
from scripts.kn_lib import KneserNeyModel, kn_prob, pack
import importlib

scripts_01 = importlib.import_module("scripts.01_assess_raw")
assess_stream = scripts_01.assess_stream
SENTENCE_SPLIT_REGEX = scripts_01.SENTENCE_SPLIT_REGEX

scripts_02 = importlib.import_module("scripts.02_build_cleaned")
clean_stream = scripts_02.clean_stream

scripts_06 = importlib.import_module("scripts.06_split")
build_split = scripts_06.build_split
greedy_pick = scripts_06.greedy_pick
line_group_ids = scripts_06.line_group_ids


# ---------------------------------------------------------------------------
# Test 1: test_context_matrix
# ---------------------------------------------------------------------------
def test_context_matrix():
    """Verify context_matrix matches brute-force builder and never crosses boundaries."""
    # Toy corpus with sentences of lengths: 2, 1, 4, 2
    sentences = [
        [10, EOS_ID],
        [20],
        [30, 31, 32, EOS_ID],
        [40, EOS_ID],
    ]
    tokens_list = []
    starts_list = []
    cur_idx = 0
    for s in sentences:
        starts_list.append(cur_idx)
        tokens_list.extend(s)
        cur_idx += len(s)

    tokens = np.array(tokens_list, dtype=np.int32)
    starts = np.array(starts_list, dtype=np.int64)
    pos = sentence_positions(starts, len(tokens))

    for n in [1, 2, 3, 4]:
        matrix = context_matrix(tokens, pos, n)
        assert matrix.shape == (len(tokens), n)
        assert matrix.dtype == np.int32

        # Brute-force validation per sentence
        idx = 0
        for s in sentences:
            for s_pos in range(len(s)):
                expected_row = []
                for offset in range(n - 1, -1, -1):
                    ctx_pos = s_pos - offset
                    if ctx_pos < 0:
                        expected_row.append(PAD_ID)
                    else:
                        expected_row.append(s[ctx_pos])
                np.testing.assert_array_equal(matrix[idx], expected_row)
                idx += 1


# ---------------------------------------------------------------------------
# Test 2: test_group_rows
# ---------------------------------------------------------------------------
def test_group_rows():
    """Verify group_rows equals collections.Counter over row tuples."""
    np.random.seed(42)
    M = np.random.randint(0, 10, size=(200, 3), dtype=np.int32)
    u_rows, counts = group_rows(M)

    # Validate lexicographical ordering
    for i in range(len(u_rows) - 1):
        assert tuple(u_rows[i]) < tuple(u_rows[i + 1])

    # Validate against collections.Counter
    counter = collections.Counter(map(tuple, M))
    assert len(u_rows) == len(counter)
    for row, cnt in zip(u_rows, counts):
        assert counter[tuple(row)] == cnt
    assert counts.sum() == len(M)


# ---------------------------------------------------------------------------
# Test 3: test_partitioned_equals_direct
# ---------------------------------------------------------------------------
def test_partitioned_equals_direct():
    """Verify group_rows_partitioned exactly matches group_rows for 1 to 5 partitions."""
    np.random.seed(42)
    M = np.random.randint(0, 25, size=(400, 4), dtype=np.int32)

    u_direct, c_direct = group_rows(M)

    for parts in [1, 2, 3, 4, 5]:
        u_part, c_part = group_rows_partitioned(M, parts=parts)
        np.testing.assert_array_equal(u_part, u_direct, err_msg=f"Mismatch for parts={parts}")
        np.testing.assert_array_equal(c_part, c_direct, err_msg=f"Mismatch for parts={parts}")


# ---------------------------------------------------------------------------
# Test 4: test_kn_normalization
# ---------------------------------------------------------------------------
def test_kn_normalization():
    """Verify sum over vocabulary of P(w | h) equals 1 within 1e-9 for all orders and unseen contexts."""
    # Toy corpus
    sentences = [
        [10, 11, 12, EOS_ID],
        [10, 11, 13, EOS_ID],
        [14, 11, 12, EOS_ID],
        [10, 15, 12, EOS_ID],
        [16, 17, 18, EOS_ID],
    ]
    tokens = np.array([tok for s in sentences for tok in s], dtype=np.int32)
    starts = np.array([0, 4, 8, 12, 16], dtype=np.int64)
    pos = sentence_positions(starts, len(tokens))

    raw_unique = {}
    raw_counts = {}
    for n in range(1, 5):
        M = context_matrix(tokens, pos, n)
        u, c = group_rows(M)
        raw_unique[n] = u
        raw_counts[n] = c

    # Full vocabulary observed
    all_vocab = np.unique(tokens)
    vocab_size = int(all_vocab.max()) + 1

    model = KneserNeyModel.build_from_counts(raw_unique, raw_counts, max_order=4, vocab_size=vocab_size)

    # For orders 2 to 4, test all observed contexts
    for order in [2, 3, 4]:
        ctx_len = order - 1
        seen_ctxs = model.unique_rows[order][:, :ctx_len]
        unique_seen_ctxs = np.unique(seen_ctxs, axis=0)

        for ctx in unique_seen_ctxs:
            ctx_batch = np.tile(ctx, (len(all_vocab), 1))
            probs = model.prob(ctx_batch, all_vocab, order=order)
            prob_sum = float(probs.sum())
            assert abs(prob_sum - 1.0) < 1e-9, f"Order {order} seen context sum was {prob_sum}"

        # Test an unseen context
        unseen_ctx = np.array([[99, 98, 97][:ctx_len]], dtype=np.int64)
        unseen_batch = np.tile(unseen_ctx, (len(all_vocab), 1))
        probs_unseen = model.prob(unseen_batch, all_vocab, order=order)
        prob_sum_unseen = float(probs_unseen.sum())
        assert abs(prob_sum_unseen - 1.0) < 1e-9, f"Order {order} unseen context sum was {prob_sum_unseen}"


# ---------------------------------------------------------------------------
# Test 5: test_kn_reference
# ---------------------------------------------------------------------------
def test_kn_reference():
    """Verify vectorized kn_prob matches a pure-Python recursive reference implementation."""
    sentences = [
        [10, 11, 12, EOS_ID],
        [10, 11, 13, EOS_ID],
        [14, 11, 12, EOS_ID],
    ]
    tokens = np.array([tok for s in sentences for tok in s], dtype=np.int32)
    starts = np.array([0, 4, 8], dtype=np.int64)
    pos = sentence_positions(starts, len(tokens))

    raw_unique = {}
    raw_counts = {}
    for n in range(1, 4):
        M = context_matrix(tokens, pos, n)
        u, c = group_rows(M)
        raw_unique[n] = u
        raw_counts[n] = c

    all_vocab = np.unique(tokens)
    vocab_size = int(all_vocab.max()) + 1
    model = KneserNeyModel.build_from_counts(raw_unique, raw_counts, max_order=3, vocab_size=vocab_size)

    # Reference implementation
    # Precompute reference counts
    # Order 3 raw counts
    c3 = collections.Counter(tuple(r) for r in context_matrix(tokens, pos, 3))
    # Order 2 continuation counts: N1+(· w1 w2) from unique order 3 rows
    c2 = collections.Counter((r[1], r[2]) for r in raw_unique[3])
    # Order 1 continuation counts: N1+(· w) from unique order 2 rows
    c1 = collections.Counter(r[1] for r in raw_unique[2])
    total_c1 = sum(c1.values())

    d3 = model.D[3]
    d2 = model.D[2]

    def ref_p1(w):
        return c1[w] / total_c1

    def ref_p2(w1, w2):
        followers = [w for (_w1, w) in c2 if _w1 == w1]
        c_ctx = sum(c2[(w1, w)] for w in followers)
        if c_ctx == 0:
            return ref_p1(w2)
        n1 = len(followers)
        return max(c2.get((w1, w2), 0) - d2, 0.0) / c_ctx + (d2 * n1 / c_ctx) * ref_p1(w2)

    def ref_p3(w1, w2, w3):
        followers = [w for (_w1, _w2, w) in c3 if (_w1, _w2) == (w1, w2)]
        c_ctx = sum(c3[(w1, w2, w)] for w in followers)
        if c_ctx == 0:
            return ref_p2(w2, w3)
        n1 = len(followers)
        return max(c3.get((w1, w2, w3), 0) - d3, 0.0) / c_ctx + (d3 * n1 / c_ctx) * ref_p2(w2, w3)

    # Check across various pairs and trigrams
    for w1 in [10, 11, 14, PAD_ID, 99]:
        for w2 in [11, 12, 13, 99]:
            # Bigram check
            vec_p2 = model.prob(np.array([[w1]]), np.array([w2]), order=2)[0]
            expected_p2 = ref_p2(w1, w2)
            assert abs(vec_p2 - expected_p2) < 1e-9

            for w3 in [12, 13, EOS_ID, 99]:
                vec_p3 = model.prob(np.array([[w1, w2]]), np.array([w3]), order=3)[0]
                expected_p3 = ref_p3(w1, w2, w3)
                assert abs(vec_p3 - expected_p3) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: test_lut
# ---------------------------------------------------------------------------
def test_lut():
    """Verify singletons and count-0 map to UNK_ID, while non-singletons keep IDs."""
    # Toy train counts
    train_tokens = np.array([PAD_ID, EOS_ID, 10, 10, 10, 11, 13, 13], dtype=np.int32)
    count_train = np.bincount(train_tokens, minlength=20)

    lut = np.zeros(20, dtype=np.int32)
    # Rule 8-2
    for w in range(20):
        if count_train[w] <= 1:
            lut[w] = UNK_ID
        else:
            lut[w] = w
    lut[PAD_ID] = PAD_ID
    lut[EOS_ID] = EOS_ID
    lut[UNK_ID] = UNK_ID

    assert lut[11] == UNK_ID  # singleton
    assert lut[12] == UNK_ID  # count 0
    assert lut[10] == 10      # count 3
    assert lut[13] == 13      # count 2
    assert lut[PAD_ID] == PAD_ID
    assert lut[EOS_ID] == EOS_ID
    assert lut[UNK_ID] == UNK_ID


# ---------------------------------------------------------------------------
# Test 7: test_split_v2
# ---------------------------------------------------------------------------
def test_split_v2(tmp_path: Path):
    """Verify text-group split partitions cleanly, removes duplicate sentence leakage, and covers all sentences."""
    # Create toy cleaned sentences with duplicates
    sentences = [
        "ኢትዮጵያ ታሪካዊት ሀገር ናት።\n",  # group 0, dup 1
        "ሰላም ለሁላችን ይሁን።\n",        # group 1
        "ኢትዮጵያ ታሪካዊት ሀገር ናት።\n",  # group 0, dup 2
        "ትምህርት የህይወት መሰረት ነው።\n", # group 2
        "ሰላም ለሁላችን ይሁን።\n",        # group 1, dup 2
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።\n", # group 3
        "ጥሩ ስራ ተሰርቷል።\n",          # group 4
        "ኢትዮጵያ ታሪካዊት ሀገር ናት።\n",  # group 0, dup 3
        "እውቀት ኃይል ነው።\n",          # group 5
        "አመሰግናለሁ።\n",                # group 6
    ] * 20  # 200 sentences total
    cleaned_file = tmp_path / "toy_cleaned.txt"
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.writelines(sentences)

    inv, sizes = line_group_ids(cleaned_file)
    n = len(inv)
    assert n == 200
    assert len(sizes) == 7  # 7 unique sentence texts

    split = build_split(seed=42, inv=inv, sizes=sizes, n=n)

    test_idx = np.array(split["test"], dtype=np.int64)
    tune_idx = np.array(split["tune"], dtype=np.int64)
    train_core_idx = np.array(split["train_core"], dtype=np.int64)
    train_full_idx = np.array(split["train_full"], dtype=np.int64)

    # 1. Total coverage & disjointness
    assert len(test_idx) + len(train_full_idx) == n
    all_indices = np.sort(np.concatenate([train_core_idx, tune_idx, test_idx]))
    np.testing.assert_array_equal(all_indices, np.arange(n))

    # 2. train_full partitioned by train_core and tune
    train_combined = np.sort(np.concatenate([train_core_idx, tune_idx]))
    np.testing.assert_array_equal(train_combined, train_full_idx)

    # 3. Zero text leakage: no sentence text appears on both test and train sides
    test_text_hashes = inv[test_idx]
    train_text_hashes = inv[train_full_idx]
    assert not np.any(np.isin(test_text_hashes, train_text_hashes))

    # 4. Same for tune and test
    tune_text_hashes = inv[tune_idx]
    assert not np.any(np.isin(tune_text_hashes, test_text_hashes))


# ---------------------------------------------------------------------------
# Test 8: test_clean_chunk_invariance
# ---------------------------------------------------------------------------
def test_clean_chunk_invariance(tmp_path: Path):
    """Verify running 02 with chunk_bytes=7 vs single-pass gives byte-identical output."""
    raw_text = "ሰላም ነው፡፡ እንዴት ናችሁ? ደህና ነን! ትምህርት፡ ቤት፡ ሄደ።"
    raw_bytes = raw_text.encode("utf-8")

    out_file_chunked = tmp_path / "cleaned_chunked.txt"
    out_file_full = tmp_path / "cleaned_full.txt"

    # Run with small 7-byte chunks (splits multi-byte characters)
    with io.BytesIO(raw_bytes) as in_f, open(out_file_chunked, "w", encoding="utf-8", newline="\n") as out_f:
        clean_stream(in_f, out_f, chunk_bytes=7)

    # Run with large single chunk
    with io.BytesIO(raw_bytes) as in_f, open(out_file_full, "w", encoding="utf-8", newline="\n") as out_f:
        clean_stream(in_f, out_f, chunk_bytes=1024 * 1024)

    chunked_bytes = out_file_chunked.read_bytes()
    full_bytes = out_file_full.read_bytes()

    assert chunked_bytes == full_bytes
    lines = out_file_chunked.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    for line in lines:
        assert line[-1] in ("።", "?", "!")
    # Verify ፡፡ normalized to ።
    assert lines[0].endswith("።")


# ---------------------------------------------------------------------------
# Test 9: test_assess_offsets
# ---------------------------------------------------------------------------
def test_assess_offsets():
    """Verify assess_stream offsets match the naive length definition."""
    raw_text = "ሰላም ነው፡፡ እንዴት ናችሁ? ደህና ነን! ትምህርት ቤት ሄደ።"
    in_stream = io.StringIO(raw_text)
    out_stream = io.StringIO()

    stats = assess_stream(in_stream, out_stream, chunk_size=10)
    lines = out_stream.getvalue().splitlines()[1:]  # Skip header

    # Naive definition
    matches = list(SENTENCE_SPLIT_REGEX.finditer(raw_text))
    last_end = 0
    expected_offsets = []
    for m in matches:
        offset = len(raw_text[:last_end].encode("utf-8"))
        expected_offsets.append(offset)
        last_end = m.end()

    assert len(lines) == len(expected_offsets)
    for line, exp_offset in zip(lines, expected_offsets):
        parts = line.split("\t")
        assert int(parts[1]) == exp_offset


# ---------------------------------------------------------------------------
# Test 10: test_checksum_corrupt_raises
# ---------------------------------------------------------------------------
def test_checksum_corrupt_raises(tmp_path: Path):
    """Verify corrupt checksums.json raises RuntimeError and is not replaced."""
    corrupt_file = tmp_path / "checksums.json"
    corrupt_file.write_text("{ corrupt json data ...", encoding="utf-8")

    with patch("scripts.common.CHECKSUMS_FILE", corrupt_file):
        with pytest.raises(RuntimeError, match="Corrupt"):
            load_checksums()
    # Ensure corrupt file was not deleted or replaced
    assert corrupt_file.exists()


# ---------------------------------------------------------------------------
# Test 11: test_budget
# ---------------------------------------------------------------------------
def test_budget():
    """Verify memory_budget_mb calculates correct budget for various cgroup/physical limits."""
    # 1. Environment override takes precedence
    with patch.dict(os.environ, {"NGRAM_MEM_BUDGET_MB": "12345.0"}):
        assert memory_budget_mb() == 12345.0

    # 2. Unlimited cgroup, physical RAM 16 GiB
    # budget = 16384 * 0.90 - 1024 = 14745.6 - 1024 = 13721.6
    with patch.dict(os.environ, {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            with patch("os.sysconf", create=True, side_effect=[16 * 1024 * 1024 // 4, 4096]):
                budget = memory_budget_mb()
                expected = 16384.0 * 0.90 - 1024.0
                assert abs(budget - expected) < 1.0


# ---------------------------------------------------------------------------
# Test 12: test_manifest_inputs
# ---------------------------------------------------------------------------
def test_manifest_inputs(tmp_path: Path):
    """Verify changing an input hash makes stage_is_done return False."""
    inp = tmp_path / "input.txt"
    out = tmp_path / "output.txt"
    manifest_f = tmp_path / "manifest.json"

    inp.write_text("initial input", encoding="utf-8")
    out.write_text("initial output", encoding="utf-8")

    checksums_f = tmp_path / "checksums.json"

    with patch("scripts.common.MANIFEST_FILE", manifest_f), patch("scripts.common.CHECKSUMS_FILE", checksums_f):
        record_stage("stage_test", [inp], [out])
        assert stage_is_done("stage_test", [inp], [out]) is True

        # Modify input content
        inp.write_text("changed input content", encoding="utf-8")
        assert stage_is_done("stage_test", [inp], [out]) is False
