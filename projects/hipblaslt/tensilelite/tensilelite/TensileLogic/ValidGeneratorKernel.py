# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation of solutions whose CustomKernel is a generator block.

A generator kernel is emitted by its generator, not by Tensile, so the matrix
instruction, work group and XCC validators do not describe it. Its generator's
parameter validation is the whole check, and it needs no tensilelite_codegen module.
"""

from pathlib import Path

from ..KernelGenerators import validateGeneratorKernel


def _validateGeneratorKernel(solution: dict, filepath: Path) -> bool:
    try:
        validateGeneratorKernel(solution["CustomKernel"])
        return True
    except (ValueError, TypeError, KeyError) as e:
        print(
            f"Error: Validation failed: {e} (file: {filepath}, index: {solution.get('SolutionIndex')})"
        )
        return False
