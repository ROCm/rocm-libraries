################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

import sys
import importlib
import pytest
from rocisa.enum import DataTypeEnum
from Tensile.Common.DataType import DataType

from Tensile.Components.Subtile.Kernel import (
    selectMXScaleGeometry,
    MXSA_B4,
    MXSB_B4,
    MXSA_B8,
    MXSB_B8,
)


# ---------------------------------------------------------------------------
# Tiny kernel-dict factory
# ---------------------------------------------------------------------------
#
# selectMXScaleGeometry only reads
#
#     kernel["ProblemType"]["DataTypeA"]   when tc == 'MXSA'
#     kernel["ProblemType"]["DataTypeB"]   when tc == 'MXSB'
#
# (see ``data_tc = 'A' if tc == 'MXSA' else 'B'`` in the function body).
def _build_kernel(*, dtype_a, dtype_b=None):
    """Build the smallest kernel dict accepted by selectMXScaleGeometry."""
    if dtype_b is None:
        dtype_b = dtype_a
    return {
        "ProblemType": {
            "DataTypeA": dtype_a,
            "DataTypeB": dtype_b,
        }
    }


# ---------------------------------------------------------------------------
# Parametrize lists
# ---------------------------------------------------------------------------
EIGHT_BIT_FLOAT_ENUMS = [
    DataTypeEnum.Float8,
    DataTypeEnum.BFloat8,
    DataTypeEnum.Float8BFloat8,
    DataTypeEnum.BFloat8Float8,
]
# FP4 / FP6 enums — supported pre-fix. Used as a regression guard: the FP4/FP6
# branch must keep returning the unchanged MXSA_B4 / MXSB_B4 singletons.
SIX_OR_FOUR_BIT_ENUMS = [
    DataTypeEnum.Float4,
    DataTypeEnum.Float6,
    DataTypeEnum.BFloat6,
]
# A deliberately mixed bag: floats of the wrong width, integers, and types
# unrelated to the MX scale pipeline. Each must continue to raise.
UNSUPPORTED_ENUMS = [
    DataTypeEnum.Float,
    DataTypeEnum.Double,
    DataTypeEnum.Half,
    DataTypeEnum.BFloat16,
    DataTypeEnum.Int32,
    DataTypeEnum.Int8,
]


# ===========================================================================
# Section 1 — FP8 dispatch
# ===========================================================================
class TestSelectMXScaleGeometryFP8:
    """Verify the new ``is8bitFloat()`` branch in selectMXScaleGeometry.
    For every 8-bit float dtype the function must return:
      * the module-level singleton ``MXSA_B8`` when ``tc == 'MXSA'``,
      * the module-level singleton ``MXSB_B8`` when ``tc == 'MXSB'``.
    "Singleton" is enforced via ``is`` (identity), not ``==`` (structural
    equality), because the production caller relies on the geometry being
    cached at module load time. If selectMXScaleGeometry ever started
    constructing a fresh equal instance per call, the existing pipeline
    would still pass an ``==`` check but break any future memoisation
    keyed on geometry identity.
    """

    @pytest.mark.parametrize("dtype_enum", EIGHT_BIT_FLOAT_ENUMS, ids=lambda e: e.name)
    def test_returns_MXSA_B8_for_MXSA(self, dtype_enum):
        """Every 8-bit float dtype yields ``MXSA_B8`` when tc='MXSA'."""
        kernel = _build_kernel(dtype_a=DataType(dtype_enum))
        result = selectMXScaleGeometry(kernel, "MXSA")
        assert result is MXSA_B8, (
            f"Expected the singleton MXSA_B8 for 8-bit float "
            f"{dtype_enum.name}, got {result!r}. If this fails, the fix "
            f"likely returns a freshly-constructed MXScaleTilePair instead "
            f"of the cached one."
        )

    @pytest.mark.parametrize("dtype_enum", EIGHT_BIT_FLOAT_ENUMS, ids=lambda e: e.name)
    def test_returns_MXSB_B8_for_MXSB(self, dtype_enum):
        """Every 8-bit float dtype yields ``MXSB_B8`` when tc='MXSB'."""
        # We deliberately set A=Float (a non-8-bit dtype) so any accidental
        # "look up DataTypeA when tc='MXSB'" bug in the dispatch would
        # surface as a NotImplementedError raised by Section 3's safety net,
        # not a silent pass.
        kernel = _build_kernel(
            dtype_a=DataType(DataTypeEnum.Float),
            dtype_b=DataType(dtype_enum),
        )
        result = selectMXScaleGeometry(kernel, "MXSB")
        assert result is MXSB_B8, (
            f"Expected the singleton MXSB_B8 for 8-bit float "
            f"{dtype_enum.name}, got {result!r}."
        )

    def test_call_is_idempotent(self):
        """Calling twice with the same kernel returns the same object.
        Belt-and-braces check that the function performs no hidden mutation
        of the kernel dict and no per-call allocation."""
        kernel = _build_kernel(dtype_a=DataType(DataTypeEnum.Float8))
        first = selectMXScaleGeometry(kernel, "MXSA")
        second = selectMXScaleGeometry(kernel, "MXSA")
        assert first is second

    def test_kernel_dict_is_not_mutated(self):
        """selectMXScaleGeometry must not mutate the caller's kernel dict.
        Functions that quietly mutate their inputs are a notorious source of
        production bugs (the call site sees a different object on the next
        access). We snapshot keys + types before the call and verify they
        are unchanged afterwards.
        """
        before_dtype_a = DataType(DataTypeEnum.Float8)
        before_dtype_b = DataType(DataTypeEnum.Float8)
        kernel = _build_kernel(dtype_a=before_dtype_a, dtype_b=before_dtype_b)
        before_keys = set(kernel.keys())
        before_pt_keys = set(kernel["ProblemType"].keys())
        selectMXScaleGeometry(kernel, "MXSA")
        selectMXScaleGeometry(kernel, "MXSB")
        # Same keys
        assert set(kernel.keys()) == before_keys
        assert set(kernel["ProblemType"].keys()) == before_pt_keys
        # Same dtype objects (identity)
        assert kernel["ProblemType"]["DataTypeA"] is before_dtype_a
        assert kernel["ProblemType"]["DataTypeB"] is before_dtype_b


# ===========================================================================
# Section 2 — Regression: pre-existing FP4/FP6 path is still alive
# ===========================================================================
class TestSelectMXScaleGeometryFP4FP6Regression:
    """Verify does not accidentally disturb the original FP4/FP6
    dispatch."""

    @pytest.mark.parametrize("dtype_enum", SIX_OR_FOUR_BIT_ENUMS, ids=lambda e: e.name)
    def test_FP4_FP6_still_returns_B4(self, dtype_enum):
        """FP4 and FP6 still resolve to ``MXSA_B4 / MXSB_B4``."""
        kernel = _build_kernel(dtype_a=DataType(dtype_enum))
        assert selectMXScaleGeometry(kernel, "MXSA") is MXSA_B4
        assert selectMXScaleGeometry(kernel, "MXSB") is MXSB_B4

    def test_branch_ordering_does_not_misroute_FP4_to_B8(self):
        """Make sure the FP8 branch wasn't placed *above* the FP4 branch in
        a way that catches FP4 first via a broader predicate.
        If a future refactor accidentally writes::
            if dtype.is8bitFloat() or dtype.isFloat4():    # <-- BUG
                return MXSA_B8 ...
        this assertion catches it.
        """
        kernel = _build_kernel(dtype_a=DataType(DataTypeEnum.Float4))
        assert selectMXScaleGeometry(kernel, "MXSA") is not MXSA_B8


# ===========================================================================
# Section 3 — Safety net: unsupported dtypes still raise
# ===========================================================================
class TestSelectMXScaleGeometryUnsupported:
    """Anything outside {FP4, FP6, FP8 family} must still raise."""

    @pytest.mark.parametrize("dtype_enum", UNSUPPORTED_ENUMS, ids=lambda e: e.name)
    def test_other_dtypes_raise_NotImplementedError_MXSA(self, dtype_enum):
        kernel = _build_kernel(dtype_a=DataType(dtype_enum))
        # ``match=`` checks the error message contains "unsupported dtype".
        # If the wording in selectMXScaleGeometry changes, update this
        # assertion *and* the function docstring together.
        with pytest.raises(NotImplementedError, match="unsupported dtype"):
            selectMXScaleGeometry(kernel, "MXSA")

    @pytest.mark.parametrize("dtype_enum", UNSUPPORTED_ENUMS, ids=lambda e: e.name)
    def test_other_dtypes_raise_NotImplementedError_MXSB(self, dtype_enum):
        # A is supported, B is not. The function must raise on B, not be
        # misled by A.
        kernel = _build_kernel(
            dtype_a=DataType(DataTypeEnum.Float8),
            dtype_b=DataType(dtype_enum),
        )
        with pytest.raises(NotImplementedError, match="unsupported dtype"):
            selectMXScaleGeometry(kernel, "MXSB")


# ===========================================================================
# Section 4 — A and B dispatch independently
# ===========================================================================
class TestSelectMXScaleGeometryAandBindependent:
    """Show the dispatch reads ``DataType{A,B}`` indepedently"""

    def test_MXSA_reads_DataTypeA_not_DataTypeB(self):
        kernel = _build_kernel(
            dtype_a=DataType(DataTypeEnum.Float8),
            dtype_b=DataType(DataTypeEnum.Float4),
        )
        assert selectMXScaleGeometry(kernel, "MXSA") is MXSA_B8

    def test_MXSB_reads_DataTypeB_not_DataTypeA(self):
        kernel = _build_kernel(
            dtype_a=DataType(DataTypeEnum.Float4),
            dtype_b=DataType(DataTypeEnum.Float8),
        )
        assert selectMXScaleGeometry(kernel, "MXSB") is MXSB_B8

    def test_mixed_dispatch_does_not_cross_contaminate(self):
        """Both halves of an asymmetric kernel resolve independently."""
        kernel = _build_kernel(
            dtype_a=DataType(DataTypeEnum.Float8),
            dtype_b=DataType(DataTypeEnum.Float4),
        )
        assert selectMXScaleGeometry(kernel, "MXSA") is MXSA_B8
        assert selectMXScaleGeometry(kernel, "MXSB") is MXSB_B4
