# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gemm(a, b): a small convenience, built only on heuristic() and matmul()."""

from __future__ import annotations

import ml_dtypes
import numpy as np

from . import _core
from ._dtypes import from_numpy

__all__ = ["gemm"]

# The input types gemm() handles: (data type, compute type, scale type). Anything else (int8,
# fp8, ...) needs choices this shortcut doesn't make, so it's refused rather than guessed.
_SUPPORTED = {
    np.dtype(np.float32): (
        _core.DataType.R_32F,
        _core.ComputeType.COMPUTE_32F,
        _core.DataType.R_32F,
    ),
    np.dtype(np.float16): (
        _core.DataType.R_16F,
        _core.ComputeType.COMPUTE_32F,
        _core.DataType.R_32F,
    ),
    np.dtype(ml_dtypes.bfloat16): (
        _core.DataType.R_16BF,
        _core.ComputeType.COMPUTE_32F,
        _core.DataType.R_32F,
    ),
    np.dtype(np.float64): (
        _core.DataType.R_64F,
        _core.ComputeType.COMPUTE_64F,
        _core.DataType.R_64F,
    ),
}

_WORKSPACE_BYTES = 64 * 1024 * 1024


def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a @ b, computed by hipBLASLt with its heuristic's top pick.

    Takes two 2-D C-contiguous arrays of the same type: float32, float16, bfloat16 or float64.
    For anything more (a chosen solution, epilogues, scales, other types), use hipblaslt._core.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"gemm: shapes {a.shape} and {b.shape} can't be multiplied")
    if a.dtype != b.dtype:
        raise ValueError(f"gemm: a is {a.dtype} and b is {b.dtype}; they must match")
    if a.dtype not in _SUPPORTED:
        raise ValueError(
            f"gemm: {a.dtype} isn't supported; use float32, float16, bfloat16 or float64, or "
            "hipblaslt._core for other types"
        )
    dtype, compute, scale = _SUPPORTED[a.dtype]
    m, k = a.shape
    n = b.shape[1]

    # hipBLASLt is column-major. A row-major (m, k) array is the column-major (k, m) matrix
    # a^T, so computing d^T = b^T @ a^T in column-major writes d in row-major with no copies.
    with _core.Handle() as handle:
        desc = _core.MatmulDesc(compute, scale)
        lb = _core.MatrixLayout(dtype, n, k, n)
        la = _core.MatrixLayout(dtype, k, m, k)
        ld = _core.MatrixLayout(dtype, n, m, n)
        pref = _core.Preference()
        pref.set_attribute(_core.PreferenceAttr.MAX_WORKSPACE_BYTES, _WORKSPACE_BYTES)
        results = _core.heuristic(handle, desc, lb, la, ld, ld, pref, 1)
        if not results:
            raise _core.HipblasLtError(
                f"gemm: hipBLASLt has no solution for {a.dtype} {m}x{n}x{k}"
            )
        result = results[0]
        workspace = (
            _core.DeviceArray.empty([result.workspace_size], _core.DataType.R_8I)
            if result.workspace_size
            else None
        )
        da, db = from_numpy(a, dtype), from_numpy(b, dtype)
        dd = _core.DeviceArray.empty([m, n], dtype)
        _core.matmul(
            handle, desc, 1, db, lb, da, la, 0, dd, ld, dd, ld, result, workspace
        )
        return dd.to_numpy()
