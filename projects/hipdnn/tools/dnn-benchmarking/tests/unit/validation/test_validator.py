# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Validator."""

import numpy as np
import pytest

from dnn_benchmarking.validation import Validator


class TestValidatorCompareAB:
    """Tests for Validator.compare_ab method."""

    def test_identical_outputs_pass(self) -> None:
        """Test that identical outputs pass comparison."""
        validator = Validator()
        output_a = np.array([1.0, 2.0, 3.0])
        output_b = np.array([1.0, 2.0, 3.0])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is True
        assert "match" in message.lower()

    def test_outputs_within_tolerance_pass(self) -> None:
        """Test that outputs within tolerance pass comparison."""
        validator = Validator(rtol=1e-3, atol=1e-6)
        output_a = np.array([1.0, 2.0, 3.0])
        output_b = np.array([1.0001, 2.0002, 3.0003])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is True

    def test_outputs_outside_tolerance_fail(self) -> None:
        """Test that outputs outside tolerance fail comparison."""
        validator = Validator(rtol=1e-5, atol=1e-8)
        output_a = np.array([1.0, 2.0, 3.0])
        output_b = np.array([1.1, 2.2, 3.3])  # Large difference

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is False
        assert "mismatch" in message.lower()
        assert "max_abs" in message.lower()

    def test_shape_mismatch_fails(self) -> None:
        """Test that shape mismatch fails comparison."""
        validator = Validator()
        output_a = np.array([1.0, 2.0, 3.0])
        output_b = np.array([[1.0, 2.0], [3.0, 4.0]])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is False
        assert "shape mismatch" in message.lower()

    def test_nan_in_output_a_fails(self) -> None:
        """Test that NaN in output A fails comparison."""
        validator = Validator()
        output_a = np.array([1.0, np.nan, 3.0])
        output_b = np.array([1.0, 2.0, 3.0])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is False
        assert "nan" in message.lower() or "NaN" in message

    def test_nan_in_output_b_fails(self) -> None:
        """Test that NaN in output B fails comparison."""
        validator = Validator()
        output_a = np.array([1.0, 2.0, 3.0])
        output_b = np.array([1.0, np.nan, 3.0])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is False
        assert "nan" in message.lower() or "NaN" in message

    def test_inf_in_output_a_fails(self) -> None:
        """Test that Inf in output A fails comparison."""
        validator = Validator()
        output_a = np.array([1.0, np.inf, 3.0])
        output_b = np.array([1.0, 2.0, 3.0])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is False
        assert "inf" in message.lower() or "Inf" in message

    def test_inf_in_output_b_fails(self) -> None:
        """Test that Inf in output B fails comparison."""
        validator = Validator()
        output_a = np.array([1.0, 2.0, 3.0])
        output_b = np.array([1.0, np.inf, 3.0])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is False
        assert "inf" in message.lower() or "Inf" in message

    def test_multidimensional_arrays(self) -> None:
        """Test comparison works with multidimensional arrays."""
        validator = Validator()
        output_a = np.random.randn(16, 16, 16, 16).astype(np.float32)
        output_b = output_a.copy()

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is True

    def test_empty_arrays(self) -> None:
        """Test comparison works with empty arrays."""
        validator = Validator()
        output_a = np.array([])
        output_b = np.array([])

        passed, message = validator.compare_ab(output_a, output_b)

        assert passed is True


class TestValidatorValidate:
    """Tests for Validator.validate method."""

    def test_validate_no_reference_skips(self) -> None:
        """Test that validation without reference is skipped."""
        validator = Validator()
        output_data = np.array([1.0, 2.0, 3.0])

        # Mock tensor info - just need any object
        class MockTensorInfo:
            pass

        result = validator.validate(output_data, MockTensorInfo(), None)

        assert result.passed is True
        assert "skipped" in result.message.lower()

    def test_validate_with_matching_reference_passes(self) -> None:
        """Test validation with matching reference passes."""
        validator = Validator()
        output_data = np.array([1.0, 2.0, 3.0])
        reference_data = np.array([1.0, 2.0, 3.0])

        class MockTensorInfo:
            pass

        result = validator.validate(output_data, MockTensorInfo(), reference_data)

        assert result.passed is True
        assert result.max_abs_diff == 0.0

    def test_validate_with_mismatching_reference_fails(self) -> None:
        """Test validation with mismatching reference fails."""
        validator = Validator()
        output_data = np.array([1.0, 2.0, 3.0])
        reference_data = np.array([1.5, 2.5, 3.5])

        class MockTensorInfo:
            pass

        result = validator.validate(output_data, MockTensorInfo(), reference_data)

        assert result.passed is False
        assert result.max_abs_diff > 0
