# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.numeric.compare``: the error metrics numeric checks judge outputs by."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rocke.numeric.compare import max_abs_error, peak_relative_error


def test_the_error_is_relative_to_the_reference_peak():
    ref = np.array([[4.0, -8.0], [1.0, 0.5]])
    out = ref + np.array([[0.0, 0.2], [-0.4, 0.0]])
    assert peak_relative_error(out, ref) == pytest.approx(0.4 / 8.0)


def test_the_floor_bounds_the_denominator_from_below():
    ref = np.array([0.01, -0.02])
    out = np.array([0.01, 0.0])
    assert peak_relative_error(out, ref) == pytest.approx(1.0)
    assert peak_relative_error(out, ref, floor=1.0) == pytest.approx(0.02)


def test_a_nan_output_gives_nan():
    assert math.isnan(peak_relative_error([1.0, float("nan")], [1.0, 2.0]))


def test_refuses_mismatched_shapes():
    with pytest.raises(
        ValueError, match=r"out.shape=\(2,\), expected ref.shape=\(3,\)"
    ):
        peak_relative_error(np.zeros(2), np.zeros(3))


def test_the_max_abs_error_is_the_largest_absolute_difference():
    ref = np.array([[4.0, -8.0], [1.0, 0.5]], dtype=np.float32)
    out = ref + np.array([[0.0, 0.25], [-0.5, 0.0]], dtype=np.float32)
    assert max_abs_error(out, ref) == 0.5


def test_a_nan_anywhere_makes_the_max_abs_error_nan():
    assert math.isnan(max_abs_error([1.0, float("nan")], [1.0, 2.0]))
    assert math.isnan(max_abs_error([1.0, 2.0], [float("nan"), 2.0]))


def test_the_max_abs_error_refuses_mismatched_shapes():
    with pytest.raises(
        ValueError,
        match=r"max_abs_error failed - out.shape=\(2, 1\), expected ref.shape=\(2,\)",
    ):
        max_abs_error(np.zeros((2, 1)), np.zeros(2))


def test_a_zero_peak_with_no_floor_gives_inf_not_an_error():
    with np.errstate(divide="ignore"):
        assert peak_relative_error([1.0], [0.0]) == math.inf
