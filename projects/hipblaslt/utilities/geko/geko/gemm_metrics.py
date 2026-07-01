# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reusable GEMM-shape metrics: categories, FLOP counts, bytes moved, AI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geko.constants import DTYPE_BYTES

__all__ = [
    "CATEGORIES",
    "DTYPE_BYTES",
    "CATEGORIES_2",
    "arithmetic_intensity",
    "bytes_moved",
    "categorize",
    "categorize_2",
    "dtype_bytes",
    "flop_count",
]


# All 20 canonical labels in their documented order.
CATEGORIES: tuple[str, ...] = (
    "Small GEMMs",
    "Medium GEMMs",
    "Large GEMMs",
    "Large M, smaller N and K",
    "Large M, very small N and K",
    "Large N, smaller M and K",
    "Large N, very small M and K",
    "Large K, smaller M and N",
    "Large K, very small M and N",
    "Large M and N",
    "Large N and K",
    "Large M and K",
    "Very Large GEMMs",
    "M=1",
    "N=1",
    "K=1",
    "Small batch",
    "Medium batch",
    "Large batch",
    "Very large batch",
)


# ---------------------------------------------------------------------------
# Dtype byte lookup (scalar or vectorized)
# ---------------------------------------------------------------------------

def dtype_bytes(dtype):
    """Bytes per element for a hipBLASLt dtype token.

    Accepts:
      - str (single dtype name)
      - pandas Series of dtype strings
      - numpy array / list of dtype strings

    Returns the same shape (scalar float / Series / ndarray). Unknown dtypes
    yield NaN so downstream arithmetic stays well-defined.
    """
    if isinstance(dtype, str):
        return float(DTYPE_BYTES.get(dtype, float("nan")))
    if isinstance(dtype, pd.Series):
        return dtype.map(DTYPE_BYTES).astype(float)
    return np.array([DTYPE_BYTES.get(str(d), float("nan")) for d in dtype], dtype=float)


# ---------------------------------------------------------------------------
# Theoretical arithmetic
# ---------------------------------------------------------------------------

def flop_count(m, n, k, batch=1):
    """Total floating-point operation count for a GEMM (or batched GEMM).

    ``2 * m * n * k * batch`` — the universal convention counting each
    multiply-add as 2 FLOPs. Works on scalars or array-like inputs.
    """
    return 2 * m * n * k * batch


def bytes_moved(m, n, k, batch, a_type, b_type, c_type):
    """Theoretical bytes moved by a GEMM under the "minimal traffic" convention.

    ``batch * (sizeof(a)*m*k + sizeof(b)*k*n + sizeof(c)*m*n)`` — one read of
    each operand from HBM, no writes counted. Matches the MLSE Examen roofline
    formula.

    Works on scalars or array-like inputs (m/n/k/batch as scalars, ndarrays, or
    pandas Series; a_type/b_type/c_type as scalar strings or Series of strings).
    """
    ba = dtype_bytes(a_type)
    bb = dtype_bytes(b_type)
    bc = dtype_bytes(c_type)
    return batch * (ba * m * k + bb * k * n + bc * m * n)


def arithmetic_intensity(m, n, k, batch, a_type, b_type, c_type):
    """Theoretical arithmetic intensity (FLOPs / byte).

    Matches the canonical MLSE Examen roofline formula. The batch factor
    cancels between numerator and denominator, so AI is independent of
    batch_count; it's accepted for API consistency with ``bytes_moved``.
    """
    return flop_count(m, n, k, batch) / bytes_moved(m, n, k, batch, a_type, b_type, c_type)


# ---------------------------------------------------------------------------
# Canonical GEMM categorization
# ---------------------------------------------------------------------------

def _categorize_scalar(m: int, n: int, k: int, batch: int) -> str:
    # 17-20: Batched
    if batch > 8192:
        return "Very large batch"
    if batch > 1024:
        return "Large batch"
    if batch > 128:
        return "Medium batch"
    if batch >= 2:
        return "Small batch"

    # 14-16: Degenerate dims (batch == 1)
    if m == 1:
        return "M=1"
    if n == 1:
        return "N=1"
    if k == 1:
        return "K=1"

    large_m, large_n, large_k = m > 8192, n > 8192, k > 8192

    # 10-13: Multiple large dims
    if large_m and large_n and large_k:
        return "Very Large GEMMs"
    if large_n and large_k:
        return "Large N and K"
    if large_m and large_k:
        return "Large M and K"
    if large_m and large_n:
        return "Large M and N"

    # 4-9: One large dim, with "very small" sub-buckets when the other two are <=128
    if large_m:
        return (
            "Large M, very small N and K"
            if (n <= 128 and k <= 128)
            else "Large M, smaller N and K"
        )
    if large_n:
        return (
            "Large N, very small M and K"
            if (m <= 128 and k <= 128)
            else "Large N, smaller M and K"
        )
    if large_k:
        return (
            "Large K, very small M and N"
            if (m <= 128 and n <= 128)
            else "Large K, smaller M and N"
        )

    # 1-3: No dim > 8192
    if m <= 1024 and n <= 1024 and k <= 1024:
        return "Small GEMMs"
    if m >= 4096 and n >= 4096 and k >= 4096:
        return "Large GEMMs"
    return "Medium GEMMs"


def categorize(m, n, k, batch=1):
    """Canonical GEMM category from the MLSE taxonomy.

    Maps a shape to one of 20 mutually-exclusive labels (see CATEGORIES).
    Order of checks: batched -> degenerate dims (M/N/K=1) -> multiple large
    dims -> single large dim (with "very small" sub-bucket) -> plain size
    buckets.

    Accepts scalars or array-like inputs:
      - scalar ints -> returns a single string
      - pandas Series -> returns a pandas Series of strings (aligned index)
      - numpy array / list -> returns a numpy array of strings

    ``batch`` may be a scalar (broadcast) or an array of matching length.
    """
    is_series = isinstance(m, pd.Series)
    is_arraylike = is_series or (
        hasattr(m, "__len__") and not isinstance(m, (str, bytes))
    )

    if not is_arraylike:
        return _categorize_scalar(int(m), int(n), int(k), int(batch))

    # Broadcast batch if scalar
    n_rows = len(m)
    if not (isinstance(batch, pd.Series) or hasattr(batch, "__len__")):
        batch_iter = [int(batch)] * n_rows
    else:
        batch_iter = [int(b) for b in batch]

    labels = [
        _categorize_scalar(int(mm), int(nn), int(kk), bb)
        for mm, nn, kk, bb in zip(m, n, k, batch_iter)
    ]
    if is_series:
        return pd.Series(labels, index=m.index)
    return np.array(labels, dtype=object)


# ---------------------------------------------------------------------------
# Categorization scheme 2 (sister to ``categorize``)
# ---------------------------------------------------------------------------

CATEGORIES_2: tuple[str, ...] = (
    "2", "3.1", "1.2", "1.1", "4", "3.2", "6", "5",
)


def _categorize_2_scalar(m: int, n: int, k: int, batch: int) -> str:
    skinny_base = (
        ((m < 3073 and n > 32767) or (n < 3073 and m > 32767))
        and k < 3073
    )
    memberships = {
        "1.1": skinny_base and (k % 64 == 0),
        "1.2": skinny_base and (k % 64 != 0),
        "2":   m * n <= 2048 * 2048 and k >= 4096,
        "3.1": batch > 1,
        "3.2": k <= 512 or (m < 33 and n < 33),
        "4":   m * n > 256 * 256 and m * n < 3072 * 2048 and k > 512,
        "5":   True,  # catch-all
        "6":   (
            (m > 127 and n > 127 and m * n > 256**3 - 1)
            or (m > 255 and n > 255 and m * n * batch > 256**3 - 1)
        ),
    }
    for code in CATEGORIES_2:
        if memberships[code]:
            return code
    return "5"  # unreachable (5 is always True) but safe fallback


def categorize_2(m, n, k, batch=1):
    """Second GEMM categorization scheme (sister to ``categorize``).

    Same predicates and priority order as
    ``postprocessor.py:add_gemm_categories``: a GEMM may match multiple
    buckets, the priority order picks the single most-informative label.
    Returns one of ``CATEGORIES_2``.

    Accepts scalars or array-like inputs (same polymorphism as
    ``categorize``):
      - scalar ints -> returns a single string
      - pandas Series -> returns a pandas Series of strings
      - numpy array / list -> returns a numpy array of strings
    """
    is_series = isinstance(m, pd.Series)
    is_arraylike = is_series or (
        hasattr(m, "__len__") and not isinstance(m, (str, bytes))
    )

    if not is_arraylike:
        return _categorize_2_scalar(int(m), int(n), int(k), int(batch))

    n_rows = len(m)
    if not (isinstance(batch, pd.Series) or hasattr(batch, "__len__")):
        batch_iter = [int(batch)] * n_rows
    else:
        batch_iter = [int(b) for b in batch]

    labels = [
        _categorize_2_scalar(int(mm), int(nn), int(kk), bb)
        for mm, nn, kk, bb in zip(m, n, k, batch_iter)
    ]
    if is_series:
        return pd.Series(labels, index=m.index)
    return np.array(labels, dtype=object)
