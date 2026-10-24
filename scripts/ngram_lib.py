"""High-performance vectorized n-gram counting and matrix utilities.

Implements the memory-efficient n-gram matrix extraction and grouping
specified in Docs/NLP_Assignment1_Implementation_Plan.md and the fix specification.
"""

from __future__ import annotations

import numpy as np

# PAD_ID is 0 for <s>
PAD_ID = 0


def sentence_positions(starts: np.ndarray | int, n_tokens: int | np.ndarray) -> np.ndarray:
    """Return the 0-based position of each token within its sentence.

    Args:
        starts: 1D array of starting token indices for each sentence.
        n_tokens: Total number of tokens in the corpus array.

    Returns:
        int64 array of length n_tokens containing intra-sentence offsets.
    """
    if isinstance(starts, (int, np.integer)) and isinstance(n_tokens, (np.ndarray, list)):
        starts, n_tokens = np.asarray(n_tokens, dtype=np.int64), int(starts)
    else:
        starts = np.asarray(starts, dtype=np.int64)
        n_tokens = int(n_tokens)

    if n_tokens == 0:
        return np.empty(0, dtype=np.int64)
    if len(starts) == 0:
        return np.arange(n_tokens, dtype=np.int64)

    lengths = np.diff(np.append(starts, n_tokens))
    sentence_start_per_token = np.repeat(starts, lengths)
    return np.arange(n_tokens, dtype=np.int64) - sentence_start_per_token


def context_matrix(tokens: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """Build the (N, n) context matrix without crossing sentence boundaries.

    The last column is the prediction token itself. Preceding columns are
    the preceding tokens within the same sentence, padded with PAD_ID before
    sentence start.

    Args:
        tokens: 1D int32 array of tokens.
        pos: 1D int64 array of intra-sentence token positions.
        n: N-gram order (1 to 4).

    Returns:
        int32 array of shape (N, n).
    """
    N = len(tokens)
    M = np.full((N, n), PAD_ID, dtype=np.int32)
    M[:, n - 1] = tokens
    for k in range(n - 1):
        shift = n - 1 - k
        M[shift:, k] = tokens[:-shift]
        M[pos < shift, k] = PAD_ID
    return M


def group_rows(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lexicographically sort rows and compute exact counts of unique rows.

    Args:
        M: 2D array of shape (N, n).

    Returns:
        unique_rows: 2D array of unique rows in lexicographical ascending order.
        counts: 1D int64 array of occurrence frequencies.
    """
    if len(M) == 0:
        return np.empty((0, M.shape[1]), dtype=M.dtype), np.empty(0, dtype=np.int64)

    # np.lexsort takes keys in reverse order: column 0 is primary key
    order = np.lexsort(M.T[::-1])
    S = M[order]
    new = np.ones(len(S), dtype=bool)
    new[1:] = np.any(S[1:] != S[:-1], axis=1)
    idx = np.flatnonzero(new)
    counts = np.diff(np.append(idx, len(S))).astype(np.int64)
    return S[idx], counts


def group_rows_partitioned(
    M: np.ndarray,
    parts: int = 4,
    n_partitions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition rows by column 0 into ranges to count within lower memory budget.

    Guarantees exact mathematical equivalence to group_rows(M).

    Args:
        M: 2D array of shape (N, n).
        parts: Number of memory partitions.
        n_partitions: Alias for parts for backwards compatibility.

    Returns:
        unique_rows: Concatenated lexicographically sorted unique rows.
        counts: Concatenated frequencies.
    """
    if n_partitions is not None:
        parts = n_partitions

    if len(M) == 0:
        return np.empty((0, M.shape[1]), dtype=M.dtype), np.empty(0, dtype=np.int64)
    if parts <= 1:
        return group_rows(M)

    col0 = M[:, 0]
    unique_col0 = np.unique(col0)
    if len(unique_col0) <= parts:
        unique_list = []
        counts_list = []
        for val in unique_col0:
            mask = col0 == val
            u, c = group_rows(M[mask])
            unique_list.append(u)
            counts_list.append(c)
        return np.vstack(unique_list), np.concatenate(counts_list)

    chunks = np.array_split(unique_col0, parts)
    unique_list = []
    counts_list = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        c_min, c_max = chunk[0], chunk[-1]
        mask = (col0 >= c_min) & (col0 <= c_max)
        if not np.any(mask):
            continue
        u, c = group_rows(M[mask])
        unique_list.append(u)
        counts_list.append(c)

    if not unique_list:
        return np.empty((0, M.shape[1]), dtype=M.dtype), np.empty(0, dtype=np.int64)
    return np.vstack(unique_list), np.concatenate(counts_list)


def estimate_bytes(N: int, n: int) -> int:
    """Estimate memory needed for sorting and grouping N rows of order n.

    Formula: N * (8 * n + 16) bytes (matrix + lexsort index + sorted copy + flags).
    """
    return int(N * (8 * n + 16))


def use_partition(N: int, n: int, budget_mb: float, current_usage_mb: float = 0.0) -> bool:
    """Decide up front whether counting requires partitioned execution.

    Args:
        N: Number of token positions.
        n: N-gram order.
        budget_mb: Configured memory budget in megabytes.
        current_usage_mb: Current resident set memory in megabytes.

    Returns:
        True if estimated peak exceeds the memory budget.
    """
    est = estimate_bytes(N, n)
    return (est + current_usage_mb * (1024 * 1024)) > (budget_mb * (1024 * 1024))
