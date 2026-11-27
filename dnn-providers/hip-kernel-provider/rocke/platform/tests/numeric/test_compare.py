# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.numeric.compare``: the error metrics numeric checks judge outputs by."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rocke.numeric.compare import (
    Tolerance,
    assert_within_tolerance,
    max_abs_error,
    peak_relative_error,
    tolerance_excess,
    within_tolerance,
)


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


def test_each_element_is_held_to_its_own_bound():
    ref = np.array([100.0, 1.0, -1.0])
    tolerance = Tolerance(atol=0.01, rtol=0.01)
    # Bounds 1.01, 0.02, 0.02: the large element's error is half its bound.
    out = ref + np.array([0.505, 0.0, 0.0])
    assert tolerance_excess(out, ref, tolerance) == pytest.approx(0.5)
    assert within_tolerance(out, ref, tolerance)
    # The same absolute error that passes on the large element fails a small one
    # that a peak-relative bound of 1% (1.0 here) would still accept.
    out = ref + np.array([0.0, 0.5, 0.0])
    assert tolerance_excess(out, ref, tolerance) == pytest.approx(25.0)
    assert not within_tolerance(out, ref, tolerance)
    assert peak_relative_error(out, ref) < 0.01


@pytest.mark.parametrize(
    "out, ref",
    [
        ([1.0, float("nan")], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, float("nan")]),
        ([float("nan")], [float("nan")]),
        ([float("inf")], [float("inf")]),
        ([1.0], [float("-inf")]),
    ],
    ids=["nan-out", "nan-ref", "nan-both", "inf-both", "inf-ref"],
)
def test_a_non_finite_element_never_passes(out, ref):
    assert tolerance_excess(out, ref, Tolerance(atol=1e9, rtol=1e9)) == math.inf


def test_a_zero_bound_accepts_only_exact_equality():
    exact = Tolerance(0.0)
    assert tolerance_excess([0.0, 2.0], [0.0, 2.0], exact) == 0.0
    assert tolerance_excess([0.0, 2.0], [0.0, 2.5], exact) == math.inf


def test_atol_may_be_a_bound_per_element():
    ref = np.array([[1.0, 1.0], [1.0, 1.0]])
    tolerance = Tolerance(atol=np.array([[0.1], [0.2]]))
    out = ref + np.array([[0.05, 0.1], [0.2, 0.1]])
    assert tolerance_excess(out, ref, tolerance) == pytest.approx(1.0)
    assert not within_tolerance(
        out + np.array([[0.0, 0.01], [0.0, 0.0]]), ref, tolerance
    )


def test_the_excess_of_an_empty_array_is_zero():
    assert tolerance_excess(np.zeros((0, 3)), np.zeros((0, 3)), Tolerance(0.0)) == 0.0


def test_the_excess_refuses_mismatched_shapes():
    with pytest.raises(
        ValueError,
        match=r"tolerance_excess failed - out.shape=\(2,\), expected ref.shape=\(3,\)",
    ):
        tolerance_excess(np.zeros(2), np.zeros(3), Tolerance(1.0))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"atol": -1.0}, r"atol=-1.0, expected finite values >= 0"),
        ({"atol": float("nan")}, r"atol=nan, expected finite values >= 0"),
        ({"atol": np.array([1.0, -1.0])}, r"atol=array\(\[ 1., -1.\]\), expected"),
        ({"atol": np.zeros(0)}, r"atol=array\(\[\], dtype=float64\), expected"),
        ({"atol": 0.0, "rtol": -0.5}, r"rtol=-0.5, expected a finite value >= 0"),
        (
            {"atol": 0.0, "rtol": float("inf")},
            r"rtol=inf, expected a finite value >= 0",
        ),
    ],
)
def test_tolerance_refuses_negative_or_non_finite_bounds(kwargs, message):
    with pytest.raises(ValueError, match=r"Tolerance failed - " + message):
        Tolerance(**kwargs)


def test_the_assertion_names_the_worst_element():
    ref = np.array([[1.0, 2.0], [4.0, 8.0]])
    out = ref + np.array([[0.0, 0.3], [0.0, 0.2]])
    with pytest.raises(
        AssertionError,
        match=(
            r"^bf16 case: 2 of 4 elements outside \|out - ref\| <= atol \+ "
            r"rtol\*\|ref\| \(rtol=0\.01\); worst at \(0, 1\): out=2\.3, "
            r"ref=2\.0, bound=1\.200e-01, error/bound=2\.5$"
        ),
    ):
        assert_within_tolerance(
            out, ref, Tolerance(atol=0.1, rtol=0.01), context="bf16 case"
        )
    assert_within_tolerance(ref, ref, Tolerance(0.0))
