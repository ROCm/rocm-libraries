# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The error metrics verify harnesses and numeric tests judge a kernel output by.

* :func:`max_abs_error`: ``max|out - ref|``, the largest absolute error.
* :func:`peak_relative_error`: ``max|out - ref| / max(max|ref|, floor)``, the
  largest absolute error as a fraction of the reference's peak magnitude. The
  ``floor`` keeps the ratio an absolute error on references whose peak is small
  (1.0 makes it relative only above unit magnitude); 0.0 is the plain
  peak-relative error.
* :class:`Tolerance` and :func:`tolerance_excess`: the element-wise bound
  ``|out - ref| <= atol + rtol * |ref|``, and how far the worst element is from
  it. :func:`within_tolerance` is the pass/fail answer and
  :func:`assert_within_tolerance` the test assertion that names the worst
  element.

Why element-wise: a bound on the largest error, alone or as a fraction of the
peak, is sized by the largest outputs, so it cannot see an error in a small
one: a zeroed element, a dropped term or a few-percent scale error on outputs
far below the peak all pass it. An element-wise bound holds every output to
its own size.

All of them work in float64 on arrays of one shape. A NaN anywhere in ``out``
or ``ref`` makes the two error metrics NaN, which fails every ``< tolerance``
test, and makes :func:`tolerance_excess` infinite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "Tolerance",
    "assert_within_tolerance",
    "max_abs_error",
    "peak_relative_error",
    "tolerance_excess",
    "within_tolerance",
]


@dataclass(frozen=True, eq=False)
class Tolerance:
    """The element-wise bound ``|out - ref| <= atol + rtol * |ref|``.

    ``atol`` is a scalar, or an array that broadcasts against the reference
    for a bound of its own per element (the rounding bound of each output of
    an exact-operand GEMM, say); ``rtol`` is a scalar. Both must be finite and
    non-negative. ``Tolerance(0.0)`` asks for exact equality.
    """

    atol: float | np.ndarray
    rtol: float = 0.0

    def __post_init__(self) -> None:
        atol = np.asarray(self.atol, dtype=np.float64)
        if atol.size == 0 or not (np.isfinite(atol).all() and (atol >= 0).all()):
            raise ValueError(
                f"Tolerance failed - atol={self.atol!r}, expected finite values >= 0"
            )
        if not (np.isfinite(self.rtol) and self.rtol >= 0):
            raise ValueError(
                f"Tolerance failed - rtol={self.rtol!r}, expected a finite value >= 0"
            )

    def bound(self, ref: npt.ArrayLike) -> np.ndarray:
        """``atol + rtol * |ref|``: the largest error each element may have."""
        return np.asarray(self.atol, dtype=np.float64) + self.rtol * np.abs(
            np.asarray(ref, dtype=np.float64)
        )


def _excess_per_element(
    caller: str, out: npt.ArrayLike, ref: npt.ArrayLike, tolerance: Tolerance
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(out, ref, bound, excess)`` in float64, ``excess = |out - ref| / bound``
    per element: ``inf`` for a non-finite element or a nonzero error against a
    zero bound, 0 for an exact match."""
    out, ref = _same_shape_float64(caller, out, ref)
    bound = np.broadcast_to(tolerance.bound(ref), ref.shape)
    finite = np.isfinite(out) & np.isfinite(ref)
    error = np.abs(np.where(finite, out, 0.0) - np.where(finite, ref, 0.0))
    excess = np.zeros(ref.shape)
    np.divide(error, bound, out=excess, where=bound > 0)
    excess[(error != 0) & (bound == 0)] = np.inf
    excess[~finite] = np.inf
    return out, ref, bound, excess


def tolerance_excess(
    out: npt.ArrayLike, ref: npt.ArrayLike, tolerance: Tolerance
) -> float:
    """The largest ``|out - ref| / (atol + rtol * |ref|)`` over the elements, in
    float64: at most 1.0 when every element is within ``tolerance``.

    A NaN or infinity in ``out`` or ``ref`` gives ``inf``, so a non-finite
    element never passes (a check that expects non-finite values compares
    their positions separately); an empty array gives 0.0.
    """
    excess = _excess_per_element("tolerance_excess", out, ref, tolerance)[3]
    return float(excess.max()) if excess.size else 0.0


def within_tolerance(
    out: npt.ArrayLike, ref: npt.ArrayLike, tolerance: Tolerance
) -> bool:
    """Whether every element of ``out`` is within ``tolerance`` of ``ref``."""
    return tolerance_excess(out, ref, tolerance) <= 1.0


def assert_within_tolerance(
    out: npt.ArrayLike, ref: npt.ArrayLike, tolerance: Tolerance, *, context: str = ""
) -> None:
    """Raise ``AssertionError`` unless every element of ``out`` is within
    ``tolerance`` of ``ref`` (:func:`tolerance_excess` at most 1.0). The
    message starts with ``context`` and names the worst element (index, both
    values, its bound) and how many elements are outside."""
    if tolerance_excess(out, ref, tolerance) <= 1.0:
        return
    out, ref, bound, excess = _excess_per_element(
        "assert_within_tolerance", out, ref, tolerance
    )
    worst = np.unravel_index(int(np.argmax(excess)), excess.shape)
    prefix = f"{context}: " if context else ""
    raise AssertionError(
        f"{prefix}{int((excess > 1.0).sum())} of {excess.size} elements outside "
        f"|out - ref| <= atol + rtol*|ref| (rtol={tolerance.rtol:g}); worst at "
        f"{tuple(int(i) for i in worst)}: out={float(out[worst])!r}, ref={float(ref[worst])!r}, "
        f"bound={bound[worst]:.3e}, error/bound={excess[worst]:.3g}"
    )


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
