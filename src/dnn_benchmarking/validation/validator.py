# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Validation module for comparing execution output against reference.

CPU reference validation is not yet implemented.
"""

from typing import Optional, Tuple

import numpy as np

from ..graph.tensor_info import TensorInfo
from .comparison import ArrayComparator


class Validator:
    """Validates execution output against a reference using allclose comparison.

    CPU reference validation is not yet implemented.
    When implemented, this will:
    1. Copy output buffer to host
    2. Compare against reference data using np.allclose(rtol, atol)
    """

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        """Initialize validator with tolerance settings.

        Args:
            rtol: Relative tolerance for allclose comparison.
            atol: Absolute tolerance for allclose comparison.
        """
        self._rtol = rtol
        self._atol = atol
        self._comparator = ArrayComparator(rtol=rtol, atol=atol)

    def validate(
        self,
        output_data: np.ndarray,
        tensor_info: TensorInfo,
        reference_data: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """Compare output data against reference using np.allclose.

        Args:
            output_data: Output tensor data from execution.
            tensor_info: Information about the output tensor.
            reference_data: Golden/reference data to compare against.
                           If None, validation is skipped.

        Returns:
            Tuple of (passed: bool, message: str).
        """
        if reference_data is None:
            return (True, "Validation skipped - no reference data provided")

        result = self._comparator.compare(
            output_data, reference_data, "output", "reference"
        )

        if result.passed:
            return (True, f"Validation passed (rtol={self._rtol}, atol={self._atol})")
        else:
            return (
                False,
                f"Validation failed: max_abs_diff={result.max_abs_diff:.2e}, "
                f"max_rel_diff={result.max_rel_diff:.2e} "
                f"(rtol={self._rtol}, atol={self._atol})",
            )

    def validate_stub(self) -> Tuple[bool, str]:
        """Stubbed validation - returns success with message.

        Use this when no reference data is available.

        Returns:
            Tuple of (True, stub message).
        """
        return (True, "Validation skipped - CPU reference not yet implemented")

    def compare_ab(
        self, output_a: np.ndarray, output_b: np.ndarray
    ) -> Tuple[bool, str]:
        """Compare A and B outputs using np.allclose.

        Args:
            output_a: Output tensor data from configuration A.
            output_b: Output tensor data from configuration B.

        Returns:
            Tuple of (passed: bool, message: str).
        """
        result = self._comparator.compare(output_a, output_b, "A", "B")

        if result.passed:
            return (True, f"A/B outputs match (rtol={self._rtol}, atol={self._atol})")
        else:
            return (False, f"A/B mismatch: {result.message}")
