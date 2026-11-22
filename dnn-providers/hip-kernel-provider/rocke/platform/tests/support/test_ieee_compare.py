# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``rocke.tests.ieee_compare.assert_ieee_identical`` compares bit for bit,
leaving open only what IEEE 754 leaves open."""

from __future__ import annotations

import numpy as np
import pytest
from rocke.tests.ieee_compare import assert_ieee_identical


def test_equal_bits_pass():
    values = np.array([1.5, -0.0, np.inf], np.float32)
    assert_ieee_identical(values.copy(), values)


def test_the_sign_of_zero_is_compared():
    with pytest.raises(AssertionError):
        assert_ieee_identical(np.array([0.0], np.float32), np.array([-0.0], np.float32))


def test_zero_sign_can_be_left_open():
    assert_ieee_identical(
        np.array([0.0], np.float32),
        np.array([-0.0], np.float32),
        zero_sign_exact=False,
    )


def test_one_ulp_fails():
    one = np.array([1.0], np.float32)
    with pytest.raises(AssertionError):
        assert_ieee_identical(np.nextafter(one, np.float32(2.0)), one)


def test_any_nan_matches_a_nan_unless_its_bits_are_compared():
    quiet = np.array([0x7FC00000], np.uint32).view(np.float32)
    payload = np.array([0xFFC00001], np.uint32).view(np.float32)
    assert_ieee_identical(payload, quiet)
    with pytest.raises(AssertionError):
        assert_ieee_identical(payload, quiet, nan_bits_exact=True)


def test_a_missing_nan_fails():
    with pytest.raises(AssertionError):
        assert_ieee_identical(
            np.array([1.0], np.float32), np.array([np.nan], np.float32)
        )


@pytest.mark.parametrize(
    "actual",
    [np.zeros(2, np.float64), np.zeros(3, np.float32)],
    ids=["dtype", "shape"],
)
def test_arrays_of_another_dtype_or_shape_are_refused(actual):
    with pytest.raises(
        TypeError, match=r"^IEEE comparison failed - actual is .*expected one dtype"
    ):
        assert_ieee_identical(actual, np.zeros(2, np.float32))
