################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 -- Reduction / Conversion / BetaOnly helper-kernel emit (CPU-only).

Exercises the KernelWriterReduction, KernelWriterConversion, and
KernelWriterBetaOnly emit paths that are NOT reached by the existing
test_emit_helpers_char.py suite.

Target files and uncovered arms:
  KernelWriterReduction.py  (all lines missed — no test exercises Gradient+Bias):
    __init__:       lines 33-49  (state copy, indicesStr, datatype derivation)
    kernelBody:     lines 52-54  (dummy body -- always returns '')
    kernelName:     lines 56-69  (static name from btype + ComputeDataType)
    getKernelName:  lines 72-73
    getHeaderFileString: lines 76-100 (MTVW loop, extern C stubs)
    getSourceFileString: lines 103-107

  KernelWriterBetaOnly.py  (partially missed):
    lines 59-63     f8MacroGuard branches (float8/bfloat8 dest)
    lines 93-95     BetaOnlyUseBias -> Bias pointer in functionSignature
    lines 108-111   BetaOnlyUseBias strideBias / factorDim in functionSignature
    lines 173-180   GLOBAL_BIAS macro in kernelBodyBetaOnly
    lines 259-269   biasStr computation (UseBias==2, UseBias==3 branches)

  KernelWriterConversion.py  (partially missed):
    lines 61-63     f8/bfloat8_fnuz MacroGuard
    lines 118-120   UseScaleAB in functionArgument
    lines 392-403   StridedBatched=False batch-ptr indirection in kernelBody

Strategy:
  A) emit_helpers_from_logic(Grad.yaml)  -> triggers KernelWriterReduction
     (Gradient=True, UseBias=1, BiasDataTypeList=[0]).
     Assert Reduction kernel in output with err==0.
  B) emit_helpers_from_logic(GSU.yaml)   -> triggers KernelWriterConversion +
     KernelWriterBetaOnly (GlobalSplitU=2, _GlobalAccumulation=MultipleBuffer).
     Assert both helper types are emitted with err==0.
  C) Direct instantiation tests for BetaOnly with BetaOnlyUseBias=True to hit
     functionSignature and kernelBodyBetaOnly bias branches.
  D) Direct instantiation of Reduction to exercise header and source emit.

Pattern: A (codegen emit via emit_helpers_from_logic + direct KernelWriter calls).
"""

import os
from copy import deepcopy

import pytest

from codegen_harness import emit_helpers_from_logic

pytestmark = pytest.mark.unit

_DATA = os.path.join(os.path.dirname(__file__), "data", "gfx942")

# Grad.yaml: Gradient=True, UseBias=1, BiasDataTypeList=[0] -> Reduction kernels
_GRAD_LOGIC = os.path.join(_DATA, "Grad.yaml")

# GSU.yaml: GlobalSplitU=2, _GlobalAccumulation=MultipleBuffer -> Conversion + BetaOnly
_GSU_LOGIC = os.path.join(_DATA, "GSU.yaml")


# ---------------------------------------------------------------------------
# A) Reduction kernel emit via Grad.yaml
# ---------------------------------------------------------------------------


def test_r4_reduction_grad_logic_emits():
    """Grad logic (Gradient+Bias) triggers KernelWriterReduction; err==0."""
    helpers = emit_helpers_from_logic(_GRAD_LOGIC)
    assert helpers, "Expected >=1 helper kernel from Grad.yaml (Reduction)"
    names = [name for name, _err in helpers]
    reduction_names = [n for n in names if "Reduction" in n]
    assert reduction_names, (
        f"Expected >=1 Reduction kernel helper; got: {names}"
    )
    for name, err in helpers:
        if "Reduction" in name:
            assert err == 0, f"KernelWriterReduction {name!r} emitted with err={err}"


def test_r4_reduction_header_content():
    """Reduction header file string contains expected MTVW-loop kernel stubs."""
    from Tensile.KernelWriterReduction import KernelWriterReduction
    from Tensile.Common.DataType import DataType

    # Build a minimal state that mirrors what initReductionKernelObjects constructs.
    # Requires: ProblemType with Gradient=True, UseBias>0, BiasDataTypeList.
    # We derive it from the Grad.yaml logic helpers to stay consistent.
    helpers = emit_helpers_from_logic(_GRAD_LOGIC)
    reduction_pairs = [(n, e) for n, e in helpers if "Reduction" in n]
    assert reduction_pairs, "No Reduction helper produced by Grad.yaml"

    # Now directly instantiate KernelWriterReduction to exercise getHeaderFileString.
    # Minimal ProblemType dict with required fields.
    pt = {
        "NumIndicesC": 3,
        "Index0": 0,
        "Index1": 1,
        "DataType": DataType("s"),          # float32
        "ComputeDataType": DataType("s"),   # float32
        "DestDataType": DataType("s"),
        "BiasDataType": DataType("s"),
        "BiasDataTypeList": [DataType("s")],
        "HighPrecisionAccumulate": False,
        "Gradient": True,
        "UseBias": 1,
        "BetaOnlyUseBias": False,
        "UseE": False,
        "StridedBatched": True,
        "GroupedGemm": False,
        "ActivationType": "none",
        "UseBeta": True,
    }
    state = {"ProblemType": deepcopy(pt)}
    kwr = KernelWriterReduction(state)

    header = kwr.getHeaderFileString()
    # The header iterates MTVW = [[256,1,1], [128,2,1], [64,4,1], [32,8,1], [16,16,1], [8,32,1], [32,32,4]]
    assert "Reduction" in header, "Header missing 'Reduction' keyword"
    assert "MT256x1_VW1" in header or "MT256" in header, (
        "Header missing expected MT/VW kernel stubs from MTVW loop"
    )
    assert "__global__" in header, "Header missing __global__ kernel declaration"

    _err, src = kwr.getSourceFileString()
    assert _err == 0, f"getSourceFileString returned err={_err}"


def test_r4_reduction_kernel_name():
    """KernelWriterReduction.kernelName static method produces expected name format."""
    from Tensile.KernelWriterReduction import KernelWriterReduction
    from Tensile.Common.DataType import DataType

    pt = {
        "NumIndicesC": 3,
        "Index0": 0,
        "Index1": 1,
        "DataType": DataType("s"),
        "ComputeDataType": DataType("s"),
        "DestDataType": DataType("s"),
        "BiasDataType": DataType("s"),
        "BiasDataTypeList": [DataType("s")],
        "HighPrecisionAccumulate": False,
        "Gradient": True,
        "UseBias": 1,
        "BetaOnlyUseBias": False,
        "UseE": False,
        "StridedBatched": True,
        "GroupedGemm": False,
        "ActivationType": "none",
        "UseBeta": True,
    }
    state = {"ProblemType": deepcopy(pt)}
    kwr = KernelWriterReduction(state)

    name = kwr.getKernelName()
    assert "Reduction" in name, f"Kernel name missing 'Reduction': {name!r}"
    # Name should be: D<indicesStr>_<btype.toChar()><computeType.toChar()>_Reduction
    assert name.startswith("D"), f"Reduction kernel name should start with 'D': {name!r}"


# ---------------------------------------------------------------------------
# B) Conversion + BetaOnly via GSU.yaml
# ---------------------------------------------------------------------------


def test_r4_gsu_conversion_and_betaonly_emit():
    """GSU logic (GSU=2) triggers KernelWriterConversion + KernelWriterBetaOnly; err==0."""
    helpers = emit_helpers_from_logic(_GSU_LOGIC)
    assert helpers, "Expected >=1 helper kernel from GSU.yaml"
    names_errs = helpers
    names = [name for name, _err in names_errs]

    # GSU.yaml has GlobalSplitU=2, so a conversion kernel should be present.
    # BetaOnly is triggered for GSU > 1 solutions.
    # At least one helper with 'PostGSU' or '_GA' (GlobalAccum) should be present.
    has_gsu_helper = any(
        "PostGSU" in n or "_GA" in n or "_Conversion" in n for n in names
    )
    assert has_gsu_helper, (
        "Expected Conversion or BetaOnly helper from GSU.yaml; got: "
        + str(names)
    )

    errors = [(n, e) for n, e in names_errs if e != 0]
    assert not errors, f"Helper kernels with err!=0: {errors}"


# ---------------------------------------------------------------------------
# C) Direct BetaOnly instantiation — BetaOnlyUseBias=True branch
# ---------------------------------------------------------------------------


def test_r4_betaonly_bias_functionSignature():
    """KernelWriterBetaOnly with BetaOnlyUseBias=True emits Bias pointer in signature."""
    from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
    from Tensile.Common.DataType import DataType

    pt = {
        "NumIndicesC": 3,
        "Index0": 0,
        "Index1": 1,
        "DataType": DataType("s"),
        "ComputeDataType": DataType("s"),
        "DestDataType": DataType("s"),
        "BiasDataType": DataType("s"),
        "BiasDataTypeList": [DataType("s")],
        "BetaOnlyUseBias": True,
        "UseBias": 1,
        "HighPrecisionAccumulate": False,
        "Gradient": False,
        "UseE": False,
        "StridedBatched": True,
        "GroupedGemm": False,
        "ActivationType": "none",
        "UseInitialStridesCD": False,
        "BiasSrc": "D",
        "UseBeta": True,
    }
    state = {
        "ProblemType": deepcopy(pt),
        "_GlobalAccumulation": None,
    }
    kw = KernelWriterBetaOnly(state)

    sig = kw.functionSignature()
    assert "Bias" in sig, (
        "functionSignature with BetaOnlyUseBias=True must include Bias pointer"
    )
    assert "strideBias" in sig, (
        "functionSignature with BetaOnlyUseBias=True must include strideBias"
    )

    err, src = kw.getSourceFileString()
    assert err == 0, f"getSourceFileString returned err={err}"
    assert "Bias" in src, "Source with BetaOnlyUseBias=True should reference Bias"
    assert "GLOBAL_BIAS" in src or "Bias[" in src, (
        "Source must include GLOBAL_BIAS macro or direct Bias indexing"
    )


def test_r4_betaonly_bias_header():
    """KernelWriterBetaOnly with BetaOnlyUseBias=True includes Bias in header."""
    from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
    from Tensile.Common.DataType import DataType

    pt = {
        "NumIndicesC": 3,
        "Index0": 0,
        "Index1": 1,
        "DataType": DataType("s"),
        "ComputeDataType": DataType("s"),
        "DestDataType": DataType("s"),
        "BiasDataType": DataType("s"),
        "BiasDataTypeList": [DataType("s")],
        "BetaOnlyUseBias": True,
        "UseBias": 1,
        "HighPrecisionAccumulate": False,
        "Gradient": False,
        "UseE": False,
        "StridedBatched": True,
        "GroupedGemm": False,
        "ActivationType": "none",
        "UseInitialStridesCD": False,
        "BiasSrc": "D",
        "UseBeta": True,
    }
    state = {
        "ProblemType": deepcopy(pt),
        "_GlobalAccumulation": None,
    }
    kw = KernelWriterBetaOnly(state)
    hdr = kw.getHeaderFileString()
    assert "Bias" in hdr, "Header with BetaOnlyUseBias=True must include Bias"


def test_r4_betaonly_nobias_emit():
    """KernelWriterBetaOnly with BetaOnlyUseBias=False emits cleanly, no Bias pointer."""
    from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
    from Tensile.Common.DataType import DataType

    pt = {
        "NumIndicesC": 2,
        "Index0": 0,
        "Index1": 1,
        "DataType": DataType("s"),
        "ComputeDataType": DataType("s"),
        "DestDataType": DataType("s"),
        "BiasDataType": DataType("s"),
        "BiasDataTypeList": [],
        "BetaOnlyUseBias": False,
        "UseBias": 0,
        "HighPrecisionAccumulate": False,
        "Gradient": False,
        "UseE": False,
        "StridedBatched": True,
        "GroupedGemm": False,
        "ActivationType": "none",
        "UseInitialStridesCD": False,
        "BiasSrc": "D",
        "UseBeta": True,
    }
    state = {
        "ProblemType": deepcopy(pt),
        "_GlobalAccumulation": None,
    }
    kw = KernelWriterBetaOnly(state)

    err, src = kw.getSourceFileString()
    assert err == 0, f"getSourceFileString err={err}"
    # BetaOnlyUseBias=False: no Bias pointer, no strideBias in signature
    assert "strideBias" not in kw.functionSignature(), (
        "functionSignature with BetaOnlyUseBias=False should not include strideBias"
    )


# ---------------------------------------------------------------------------
# D) Reduction: int8 DataType + float32 ComputeDataType + HPA branch
# ---------------------------------------------------------------------------


def test_r4_reduction_int8_hpa_datatype():
    """KernelWriterReduction int8+HPA branch sets datatype to int32."""
    from Tensile.KernelWriterReduction import KernelWriterReduction
    from Tensile.Common.DataType import DataType

    pt = {
        "NumIndicesC": 3,
        "Index0": 0,
        "Index1": 1,
        "DataType": DataType("i8"),         # int8
        "ComputeDataType": DataType("s"),   # float32 / single
        "DestDataType": DataType("s"),
        "BiasDataType": DataType("s"),
        "BiasDataTypeList": [DataType("s")],
        "HighPrecisionAccumulate": True,
        "Gradient": True,
        "UseBias": 1,
        "BetaOnlyUseBias": False,
        "UseE": False,
        "StridedBatched": True,
        "GroupedGemm": False,
        "ActivationType": "none",
        "UseBeta": True,
    }
    state = {"ProblemType": deepcopy(pt)}
    kwr = KernelWriterReduction(state)

    # int8 DataType + float32 ComputeDataType + HPA -> datatype should be int32
    assert "int" in kwr.datatype.lower() or "i32" in kwr.datatype.lower(), (
        f"Expected int32 datatype for int8+HPA, got: {kwr.datatype!r}"
    )
    header = kwr.getHeaderFileString()
    assert "Reduction" in header
    _err, src = kwr.getSourceFileString()
    assert _err == 0
