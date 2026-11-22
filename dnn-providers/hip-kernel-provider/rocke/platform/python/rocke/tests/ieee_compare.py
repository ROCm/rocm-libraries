# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: compare floating-point results bit for bit, except where IEEE 754
leaves the bits open.

A GPU numeric test that checks conversions or arithmetic on edge values wants
the exact bits (so -0.0 is told apart from +0.0 and a rounding error of one ulp
fails), but IEEE 754 does not fix the sign or payload of a NaN that an
operation produces, nor the sign of the zero ``maxNum`` returns for operands
that compare equal.
"""

from __future__ import annotations

import numpy as np

__all__ = ["assert_ieee_identical"]


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
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise TypeError(
            f"IEEE comparison failed - actual is {actual.dtype}{list(actual.shape)}, "
            f"expected is {expected.dtype}{list(expected.shape)}, "
            "expected one dtype and shape"
        )
    bits = np.dtype(f"u{actual.itemsize}")
    is_nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), is_nan)
    exact = ~is_nan if not nan_bits_exact else np.ones_like(is_nan)
    if not zero_sign_exact:
        exact &= expected != 0
        np.testing.assert_array_equal(actual[expected == 0], 0)
    np.testing.assert_array_equal(actual.view(bits)[exact], expected.view(bits)[exact])
