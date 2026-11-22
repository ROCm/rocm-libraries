# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The peak-relative error verify harnesses judge a kernel output by.

``max|out - ref| / max(max|ref|, floor)``: the largest absolute error, as a
fraction of the reference's peak magnitude. The ``floor`` keeps the ratio an
absolute error on references whose peak is small (1.0 makes it relative only
above unit magnitude); 0.0 is the plain peak-relative error. A NaN anywhere in
``out`` or ``ref`` makes the result NaN, which fails every ``< tolerance`` test.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["peak_relative_error"]


def peak_relative_error(
    out: npt.ArrayLike, ref: npt.ArrayLike, *, floor: float = 0.0
) -> float:
    """``max|out - ref| / max(max|ref|, floor)``, in float64."""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        raise ValueError(
            f"peak_relative_error failed - out.shape={out.shape}, expected "
            f"ref.shape={ref.shape}"
        )
    return float(np.abs(out - ref).max() / max(np.abs(ref).max(), floor))
