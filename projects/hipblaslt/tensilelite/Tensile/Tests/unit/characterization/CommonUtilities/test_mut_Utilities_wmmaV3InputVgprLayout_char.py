################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.wmmaV3InputVgprLayout``.

The function maps a 4-tuple ``wmma`` instruction shape (and, for the 128-K
shapes, a ``dtypeBitWidth``) to a fixed VGPR-layout 4-tuple
``(numReadsUnroll, numVecTile, numVecUnroll, NumElementPerRead)``. Unhandled
shapes / bitwidths raise ``AssertionError``.

These tests pin the ACTUAL current behavior so they pass on clean source and
fail under the surviving mutants on the two compound predicates
``wmma == (16,16,128,1) or wmma == (32,16,128,1)`` (mutmut 36-40) and
``dtypeBitWidth == 4 or dtypeBitWidth == 6`` (mutmut 50-51), plus the
"unsupported bitwidth" assertion (mutmut 56).
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
wmmaV3InputVgprLayout = U.wmmaV3InputVgprLayout

pytestmark = pytest.mark.unit


def test_32_16_128_1_int8_enters_branch_and_returns_layout():
    # Kills mutmut_37/38/39/40: the second operand `wmma == (32, 16, 128, 1)`
    # is corrupted to (33,16,128,1)/(32,17,128,1)/(32,16,129,1)/(32,16,128,2).
    # For input (32,16,128,1) the original matches the second clause and (with
    # bitwidth 8) returns (4,16,2,16). Each mutant no longer matches this tuple,
    # so it falls through to the final `assert False` -> AssertionError.
    assert wmmaV3InputVgprLayout((32, 16, 128, 1), 8) == (4, 16, 2, 16)


def test_unhandled_shape_raises_not_treated_as_128_branch():
    # Kills mutmut_36: `or wmma == (32,16,128,1)` -> `or wmma != (32,16,128,1)`.
    # An unhandled shape that is neither (16,16,128,1) nor (32,16,128,1) makes
    # the second clause True under the mutant (`!= (32,16,128,1)`), so the mutant
    # enters the 128-branch and (with bitwidth 8) returns (4,16,2,16). The
    # original instead falls to the else `assert False` -> AssertionError.
    with pytest.raises(AssertionError):
        wmmaV3InputVgprLayout((16, 16, 4, 2), 8)


def test_bitwidth_6_returns_packed_layout():
    # Kills mutmut_51: `dtypeBitWidth == 6` -> `dtypeBitWidth == 7`.
    # With shape (16,16,128,1) and bitwidth 6, the original returns
    # (2,16,2,32). The mutant treats 6 as unsupported -> AssertionError.
    assert wmmaV3InputVgprLayout((16, 16, 128, 1), 6) == (2, 16, 2, 32)


def test_unsupported_bitwidth_raises():
    # Kills mutmut_50: `dtypeBitWidth == 6` -> `dtypeBitWidth != 6`, AND
    # mutmut_56: `assert False` -> `assert True` on the unsupported-bitwidth path.
    # Bitwidth 16 is not 8, 4, or 6, so the original reaches the
    # `assert False, "Unsupported datatype bitwidth"` -> AssertionError.
    #  - mutmut_50: 16 != 6 is True -> mutant returns (2,16,2,32) instead.
    #  - mutmut_56: assert True passes -> function falls off end, returns None.
    with pytest.raises(AssertionError):
        wmmaV3InputVgprLayout((16, 16, 128, 1), 16)
