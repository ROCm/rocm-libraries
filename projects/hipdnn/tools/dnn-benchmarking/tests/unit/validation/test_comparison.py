# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ArrayComparator and ComparisonResult."""

import numpy as np
import pytest

from dnn_benchmarking.validation import ArrayComparator, ComparisonResult


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_create_passed_result(self) -> None:
        """Test creating a passed comparison result."""
        result = ComparisonResult(
            passed=True,
            max_abs_diff=1e-10,
            max_rel_diff=1e-12,
            message="Match",
        )

        assert result.passed is True
        assert result.max_abs_diff == 1e-10
        assert result.max_rel_diff == 1e-12
        assert result.message == "Match"

    def test_create_failed_result(self) -> None:
        """Test creating a failed comparison result."""
        result = ComparisonResult(
            passed=False,
            max_abs_diff=0.1,
            max_rel_diff=0.5,
            message="Mismatch: values differ significantly",
        )

        assert result.passed is False
        assert result.max_abs_diff == 0.1
        assert result.max_rel_diff == 0.5


class TestArrayComparator:
    """Tests for ArrayComparator class."""

    def test_identical_arrays_pass(self) -> None:
        """Test that identical arrays pass comparison."""
        comparator = ArrayComparator()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        result = comparator.compare(a, b)

        assert result.passed is True
        assert result.max_abs_diff == 0.0
        assert result.max_rel_diff == 0.0

    def test_arrays_within_tolerance_pass(self) -> None:
        """Test that arrays within tolerance pass comparison."""
        comparator = ArrayComparator(rtol=1e-3, atol=1e-6)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0001, 2.0002, 3.0003])

        result = comparator.compare(a, b)

        assert result.passed is True

    def test_arrays_outside_tolerance_fail(self) -> None:
        """Test that arrays outside tolerance fail comparison."""
        comparator = ArrayComparator(rtol=1e-5, atol=1e-8)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.2, 3.3])

        result = comparator.compare(a, b)

        assert result.passed is False
        assert result.max_abs_diff > 0.05

    def test_shape_mismatch_fails(self) -> None:
        """Test that shape mismatch fails comparison."""
        comparator = ArrayComparator()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = comparator.compare(a, b)

        assert result.passed is False
        assert "shape mismatch" in result.message.lower()

    def test_nan_in_actual_fails(self) -> None:
        """Test that NaN in actual array fails comparison."""
        comparator = ArrayComparator()
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        result = comparator.compare(a, b)

        assert result.passed is False
        assert "nan" in result.message.lower() or "NaN" in result.message

    def test_nan_in_expected_fails(self) -> None:
        """Test that NaN in expected array fails comparison."""
        comparator = ArrayComparator()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, np.nan, 3.0])

        result = comparator.compare(a, b)

        assert result.passed is False
        assert "nan" in result.message.lower() or "NaN" in result.message

    def test_inf_in_actual_fails(self) -> None:
        """Test that Inf in actual array fails comparison."""
        comparator = ArrayComparator()
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        result = comparator.compare(a, b)

        assert result.passed is False
        assert "inf" in result.message.lower() or "Inf" in result.message

    def test_inf_in_expected_fails(self) -> None:
        """Test that Inf in expected array fails comparison."""
        comparator = ArrayComparator()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, np.inf, 3.0])

        result = comparator.compare(a, b)

        assert result.passed is False
        assert "inf" in result.message.lower() or "Inf" in result.message

    def test_multidimensional_arrays(self) -> None:
        """Test comparison works with multidimensional arrays."""
        comparator = ArrayComparator()
        a = np.random.randn(4, 4, 4, 4).astype(np.float32)
        b = a.copy()

        result = comparator.compare(a, b)

        assert result.passed is True

    def test_empty_arrays(self) -> None:
        """Test comparison works with empty arrays."""
        comparator = ArrayComparator()
        a = np.array([])
        b = np.array([])

        result = comparator.compare(a, b)

        assert result.passed is True

    def test_custom_labels_in_message(self) -> None:
        """Test that custom labels appear in error messages."""
        comparator = ArrayComparator()
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        result = comparator.compare(a, b, "hipDNN", "pytorch")

        assert "hipDNN" in result.message

    def test_compare_with_diffs_returns_tuple(self) -> None:
        """Test that compare_with_diffs returns simplified tuple."""
        comparator = ArrayComparator()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        passed, max_abs_diff, max_rel_diff = comparator.compare_with_diffs(a, b)

        assert passed is True
        assert max_abs_diff == 0.0
        assert max_rel_diff == 0.0

    def test_tolerance_properties(self) -> None:
        """Test that tolerance properties are accessible."""
        comparator = ArrayComparator(rtol=1e-3, atol=1e-6)

        assert comparator.rtol == 1e-3
        assert comparator.atol == 1e-6
