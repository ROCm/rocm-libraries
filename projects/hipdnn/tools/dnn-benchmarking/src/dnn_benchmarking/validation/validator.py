# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Validation module for comparing execution output against reference."""

from typing import Optional, Tuple

import numpy as np

from ..graph.tensor_info import TensorInfo
from .comparison import ArrayComparator, ComparisonResult


class Validator:
    """Validates execution output against a reference using allclose comparison."""

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

    @property
    def rtol(self) -> float:
        """Relative tolerance."""
        return self._rtol

    @property
    def atol(self) -> float:
        """Absolute tolerance."""
        return self._atol

    def validate(
        self,
        output_data: np.ndarray,
        tensor_info: TensorInfo,
        reference_data: Optional[np.ndarray] = None,
    ) -> ComparisonResult:
        """Compare output data against reference using np.allclose.

        Args:
            output_data: Output tensor data from execution.
            tensor_info: Information about the output tensor.
            reference_data: Golden/reference data to compare against.
                           If None, validation is skipped.

        Returns:
            ComparisonResult. When reference_data is None, returns a passed
            result with a "skipped" message and zero diffs.
        """
        if reference_data is None:
            return ComparisonResult(
                passed=True,
                max_abs_diff=0.0,
                max_rel_diff=0.0,
                message="Validation skipped - no reference data provided",
            )

        return self._comparator.compare(
            output_data, reference_data, "output", "reference"
        )

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
            return (True, f"Outputs match (rtol={self._rtol}, atol={self._atol})")
        else:
            return (False, f"Output mismatch: {result.message}")
