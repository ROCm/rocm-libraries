# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The error metrics verify harnesses and numeric tests judge a kernel output by.

* :func:`max_abs_error`: ``max|out - ref|``, the largest absolute error.
* :func:`peak_relative_error`: ``max|out - ref| / max(max|ref|, floor)``, the
  largest absolute error as a fraction of the reference's peak magnitude. The
  ``floor`` keeps the ratio an absolute error on references whose peak is small
  (1.0 makes it relative only above unit magnitude); 0.0 is the plain
  peak-relative error.

Both work in float64 on arrays of one shape. A NaN anywhere in ``out`` or
``ref`` makes the result NaN, which fails every ``< tolerance`` test.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["max_abs_error", "peak_relative_error"]


def _same_shape_float64(
    caller: str, out: npt.ArrayLike, ref: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        raise ValueError(
            f"{caller} failed - out.shape={out.shape}, expected ref.shape={ref.shape}"
        )
    return out, ref


def max_abs_error(out: npt.ArrayLike, ref: npt.ArrayLike) -> float:
    """``max|out - ref|``, in float64."""
    out, ref = _same_shape_float64("max_abs_error", out, ref)
    return float(np.abs(out - ref).max())


def peak_relative_error(
    out: npt.ArrayLike, ref: npt.ArrayLike, *, floor: float = 0.0
) -> float:
    """``max|out - ref| / max(max|ref|, floor)``, in float64."""
    out, ref = _same_shape_float64("peak_relative_error", out, ref)
    # numpy division: a zero peak with a zero floor gives inf or NaN, not an error.
    return float(np.float64(max_abs_error(out, ref)) / max(np.abs(ref).max(), floor))
