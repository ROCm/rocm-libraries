# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: compare floating-point results bit for bit.

A GPU numeric test that checks conversions or arithmetic on edge values wants
the exact bits (so -0.0 is told apart from +0.0 and a rounding error of one ulp
fails), but IEEE 754 does not fix the sign or payload of a NaN that an
operation produces, nor the sign of the zero ``maxNum`` returns for operands
that compare equal: :func:`assert_ieee_identical` leaves those open.

A test that checks that two runs of one computation agree (a knob that must not
change the result) wants every bit, NaN payloads and zero signs included:
:func:`same_bits`. It takes the stored values of any element type, including
the raw bits ``rocke.runtime.device_array.host_values`` returns for bf16 and fp8.
"""

from __future__ import annotations

import numpy as np

__all__ = ["assert_ieee_identical", "same_bits"]


def _require_one_dtype_and_shape(
    comparison: str, actual: np.ndarray, expected: np.ndarray
) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise TypeError(
            f"{comparison} failed - actual is {actual.dtype}{list(actual.shape)}, "
            f"expected is {expected.dtype}{list(expected.shape)}, "
            "expected one dtype and shape"
        )


def same_bits(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Whether ``actual`` and ``expected`` hold the same bit pattern at every
    element. The arrays must have the same dtype and shape (``TypeError``
    otherwise)."""
    _require_one_dtype_and_shape("same_bits", actual, expected)
    bits = np.dtype(f"u{actual.itemsize}")
    return bool(
        np.array_equal(
            np.ascontiguousarray(actual).view(bits),
            np.ascontiguousarray(expected).view(bits),
        )
    )


def assert_ieee_identical(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    nan_bits_exact: bool = False,
    zero_sign_exact: bool = True,
) -> None:
    """Assert ``actual`` equals ``expected`` bit for bit.

    The arrays must have the same float dtype and shape (``TypeError``
    otherwise), and be NaN at the same places. With ``nan_bits_exact`` the NaN
    bits must match too (for a copy or a sign flip, which only move bits);
    otherwise any NaN is accepted where one is expected. With
    ``zero_sign_exact`` false, an expected zero accepts either signed zero.
    """
    _require_one_dtype_and_shape("IEEE comparison", actual, expected)
    bits = np.dtype(f"u{actual.itemsize}")
    is_nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), is_nan)
    exact = ~is_nan if not nan_bits_exact else np.ones_like(is_nan)
    if not zero_sign_exact:
        exact &= expected != 0
        np.testing.assert_array_equal(actual[expected == 0], 0)
    np.testing.assert_array_equal(actual.view(bits)[exact], expected.view(bits)[exact])
