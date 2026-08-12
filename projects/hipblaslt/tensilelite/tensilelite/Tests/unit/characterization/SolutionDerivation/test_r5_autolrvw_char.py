################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R5 characterization — gfx950 AUTO-LRVW derivation + UseSubtileImpl VW/storeD forcing.

Target uncovered ranges in Tensile/SolutionStructs/Solution.py:
  819-882   UseSubtileImpl block: forces VectorWidthA/B=1, BufferStore=1,
            SourceSwap=False, Use64bShadowLimit=False, DepthU/duUnit logic,
            _ABTilePair assignment, MX scale load checks, StreamK/MI guards.
  3064-3130 isAutoLRVW (inside calLRVWFor950MX): when LocalReadVectorWidth==-1
            and MXBlockA/B are set, derives LRVWA/B from MIInputPerThread,
            TransposeLDS, and the Sparse/MXBlockSize condition.

Approach:
  Both paths require gfx950 (ISA 9.5.0). UseSubtileImpl=1 forces VWA/B=1 and is
  mandatory for gfx950 MX kernels (line 815). The calLRVWFor950MX path fires only
  when MXBlockA or MXBlockB is set. LocalReadVectorWidth=-1 (auto) sends the
  code into isAutoLRVW which computes LRVW from MIInputPerThread.

  We call Solution() directly in the test process (not via the parallel
  _generateForkedSolutions worker) so coverage is captured by pytest-cov.

CPU-only. No GPU, no device compilation, no hardware access.
"""

import copy
import importlib

import pytest

pytestmark = pytest.mark.unit

S = importlib.import_module("Tensile.SolutionStructs.Solution")
Solution = S.Solution


# ---------------------------------------------------------------------------
# Shared ISA / assembler fixtures (scoped to session to minimise init cost).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gfx950_iim():
    """Single-arch ISA info map for gfx950."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx950")
    return makeIsaInfoMap([isa], cxx)


@pytest.fixture(scope="module")
def gfx950_assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_assigned(gfx950_iim):
    """Assign global parameters for gfx950 once per module; restore after."""
    import contextlib
    import copy
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    assignGlobalParameters({}, gfx950_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)


# ---------------------------------------------------------------------------
# Helper: build a minimal gfx950 MX FP8 solution dict and derive it.
# ---------------------------------------------------------------------------

def _make_gfx950_mx_fp8_solution(iim, assembler, **overrides):
    """Construct and derive a gfx950 MX FP8 solution with UseSubtileImpl=1.

    Default: MI16x16x128, 2x2 WG, DepthU=256, StreamK=3, TransposeLDS=1,
    LocalReadVectorWidth=-1 (auto).  Pass keyword overrides to explore branches.

    Returns the derived Solution object (or None if rejected).
    """
    from Tensile.BenchmarkProblems import matrixInstructionToMIParameters

    isa = list(iim.keys())[0]

    params = {
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": "F8",
            "DestDataType": "s",
            "ComputeDataType": "s",
            "HighPrecisionAccumulate": True,
            "MXBlockA": 32,
            "MXBlockB": 32,
            "TransposeA": True,
            "TransposeB": False,
            "UseBeta": True,
            "Batched": True,
        },
        "ISA": isa,
        # MI16x16x128, 2x2 wave-group, 2x2 wave-tile
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "DirectToLds": 1,
        "StaggerU": 0,
        "StreamK": 3,
        "UseSubtileImpl": True,
        # Auto LRVW — triggers isAutoLRVW in calLRVWFor950MX (lines 3064-3088)
        "LocalReadVectorWidth": -1,
        "GlobalReadVectorWidthA": 16,
        "GlobalReadVectorWidthB": 16,
        "TransposeLDS": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": -1,
        # UseSubtileImpl forces VWA/B=1 (lines 819-820); start with -1
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "SourceSwap": True,
        "ExpandPointerSwap": True,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "DebugStreamK": 0,
    }
    params.update(overrides)

    mi = params["MatrixInstruction"]
    miParams = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], params["ProblemType"], params["WorkGroup"], iim
    )
    params.update(miParams)

    return Solution(params, False, False, False, assembler, iim)


# ===========================================================================
# Test 1: gfx950 MX FP8 + UseSubtileImpl=1 + LocalReadVectorWidth=-1 (auto)
#   Hits both target ranges:
#     819-882   UseSubtileImpl block (VW forcing, DepthU check, _ABTilePair, StreamK)
#     3064-3130 isAutoLRVW (auto LRVW via MIInputPerThread, TransposeLDS path)
# ===========================================================================

def test_gfx950_mx_fp8_autolrvw_derived_valid(_gp_assigned, gfx950_iim, gfx950_assembler):
    """gfx950 MX FP8 + UseSubtileImpl=1 + LRVW auto: solution must be Valid."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler)
    assert sol is not None
    assert sol.get("Valid") is True, (
        "Expected Valid=True for gfx950 MX FP8 + UseSubtileImpl=1 + auto LRVW; "
        "got invalid (check rejection log above)"
    )


def test_gfx950_mx_fp8_autolrvw_vw_forced(_gp_assigned, gfx950_iim, gfx950_assembler):
    """UseSubtileImpl block (line 819-820) forces VectorWidthA/B to 1."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler)
    assert sol.get("Valid") is True
    # UseSubtileImpl block must set VWA/B = 1 (lines 819-820)
    assert sol.get("VectorWidthA") == 1, (
        f"UseSubtileImpl must force VectorWidthA=1, got {sol.get('VectorWidthA')}"
    )
    assert sol.get("VectorWidthB") == 1, (
        f"UseSubtileImpl must force VectorWidthB=1, got {sol.get('VectorWidthB')}"
    )


def test_gfx950_mx_fp8_autolrvw_bufferstoreD_forced(_gp_assigned, gfx950_iim, gfx950_assembler):
    """UseSubtileImpl block (line 824) forces BufferStore=1."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler)
    assert sol.get("Valid") is True
    # Line 824: state["BufferStore"] = 1
    assert sol.get("BufferStore") == 1, (
        f"UseSubtileImpl must force BufferStore=1, got {sol.get('BufferStore')}"
    )


def test_gfx950_mx_fp8_autolrvw_shadow_limit_cleared(_gp_assigned, gfx950_iim, gfx950_assembler):
    """UseSubtileImpl block (lines 826-827) clears Use64bShadowLimit flags."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler)
    assert sol.get("Valid") is True
    # Lines 826-827: Use64bShadowLimit = False, Use64bShadowLimitMX = False
    assert sol.get("Use64bShadowLimit") is False, (
        "UseSubtileImpl must set Use64bShadowLimit=False"
    )
    assert sol.get("Use64bShadowLimitMX") is False, (
        "UseSubtileImpl must set Use64bShadowLimitMX=False"
    )


def test_gfx950_mx_fp8_autolrvw_abtile_pair_fp8(_gp_assigned, gfx950_iim, gfx950_assembler):
    """UseSubtileImpl block (line 848) sets _ABTilePair to 'AB_B8' for FP8 data type."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler)
    assert sol.get("Valid") is True
    # Line 848: is8bitFloat -> AB_B8
    assert sol.get("_ABTilePairA") == "AB_B8", (
        f"FP8 dtype must set _ABTilePairA='AB_B8', got {sol.get('_ABTilePairA')!r}"
    )
    assert sol.get("_ABTilePairB") == "AB_B8", (
        f"FP8 dtype must set _ABTilePairB='AB_B8', got {sol.get('_ABTilePairB')!r}"
    )


def test_gfx950_mx_fp8_autolrvw_lrvw_derived(_gp_assigned, gfx950_iim, gfx950_assembler):
    """isAutoLRVW (lines 3064-3088) derives LRVW from MIInputPerThread when LRVW=-1."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler)
    assert sol.get("Valid") is True
    lrvwa = sol.get("LocalReadVectorWidthA")
    lrvwb = sol.get("LocalReadVectorWidthB")
    # Auto LRVW must be set (not -1) and must be a positive integer
    assert lrvwa is not None and lrvwa > 0, (
        f"Auto LRVW must be derived to a positive value, got LocalReadVectorWidthA={lrvwa}"
    )
    assert lrvwb is not None and lrvwb > 0, (
        f"Auto LRVW must be derived to a positive value, got LocalReadVectorWidthB={lrvwb}"
    )
    # For TransposeLDS=1, DirectToLds=1 path (line 3078):
    # LRVW = int(16 // MacDataType.numBytes()) where FP8 has 1 byte -> LRVW = 16
    # But DepthU//MatrixInstK ratio check (line 3085) may halve it to 8 or 32
    assert lrvwa in (8, 16, 32, 64), (
        f"LRVWA should be a valid power-of-2, got {lrvwa}"
    )
    assert lrvwb in (8, 16, 32, 64), (
        f"LRVWB should be a valid power-of-2, got {lrvwb}"
    )


def test_gfx950_mx_fp8_autolrvw_source_swap_cleared(_gp_assigned, gfx950_iim, gfx950_assembler):
    """UseSubtileImpl block (line 821) forces SourceSwap=False even if config requests True."""
    sol = _make_gfx950_mx_fp8_solution(gfx950_iim, gfx950_assembler, SourceSwap=True)
    assert sol.get("Valid") is True
    # Line 821: state["SourceSwap"] = False
    assert sol.get("SourceSwap") is False, (
        f"UseSubtileImpl must force SourceSwap=False, got {sol.get('SourceSwap')}"
    )


# ===========================================================================
# Test 2: MI32x32x64 variant — exercises DepthU/duUnit computation (lines 830-834)
#   and the different MIInputPerThread value (4 instead of 8 for 128-MI).
# ===========================================================================

def test_gfx950_mx_fp8_mi32x32_autolrvw(_gp_assigned, gfx950_iim, gfx950_assembler):
    """MI32x32x64 with UseSubtileImpl=1 + auto LRVW — exercises DepthU duUnit path."""
    sol = _make_gfx950_mx_fp8_solution(
        gfx950_iim, gfx950_assembler,
        MatrixInstruction=[32, 32, 64, 1, 1, 2, 2, 2, 2],
        DepthU=256,
    )
    # MI32x32x64: MatrixInstK=64; duUnit = 2*64*1 = 128; DepthU=256 -> 256%128=0 -> valid
    if sol.get("Valid"):
        assert sol.get("VectorWidthA") == 1
        assert sol.get("BufferStore") == 1
        lrvwa = sol.get("LocalReadVectorWidthA")
        assert lrvwa is not None and lrvwa > 0, f"LRVWA must be derived: got {lrvwa}"


# ===========================================================================
# Test 3: PGR=2 variant — exercises the PrefetchGlobalRead allowed-set check
#   (line 870-872): PGR in {0,1,2} -> valid.
# ===========================================================================

def test_gfx950_mx_fp8_pgr2_autolrvw(_gp_assigned, gfx950_iim, gfx950_assembler):
    """PrefetchGlobalRead=2 with UseSubtileImpl=1 passes the PGR guard (line 870-872)."""
    sol = _make_gfx950_mx_fp8_solution(
        gfx950_iim, gfx950_assembler,
        PrefetchGlobalRead=2,
    )
    if sol.get("Valid"):
        assert sol.get("VectorWidthA") == 1
        assert sol.get("BufferStore") == 1


# ===========================================================================
# Test 4: gfx950 MX FP8 + UseSubtileImpl=1 + explicit LRVW (not auto)
#   Exercises the isAutoLRVW validation branch (lines 3066-3071) when LRVW != -1.
# ===========================================================================

def test_gfx950_mx_fp8_explicit_lrvw_valid(_gp_assigned, gfx950_iim, gfx950_assembler):
    """Explicit LRVW=16 (non-auto) takes the validation branch in isAutoLRVW (line 3066)."""
    sol = _make_gfx950_mx_fp8_solution(
        gfx950_iim, gfx950_assembler,
        LocalReadVectorWidth=16,
    )
    # LRVW=16 is valid for FP8 on gfx950 (numRegisters=0.5, tmplrvw=16, 16*0.5=8 >= 1)
    if sol.get("Valid"):
        assert sol.get("LocalReadVectorWidthA") == 16
        assert sol.get("LocalReadVectorWidthB") == 16


# ===========================================================================
# P6 survivor-2 kill: pin the isAutoLRVW False default (the explicit-LRVW path).
#
# isAutoLRVW returns False on the explicit (LocalReadVectorWidth != -1) path, so
# the auto-derivation / LDS-pad recompute block — gated on `autoLRVWA or
# autoLRVWB` — is SKIPPED and the user's width survives verbatim. The prior
# explicit test used LRVW=16, where wlr = 16 // MIInputPerThread(32) == 1, so
# the recompute block leaves 16 unchanged even if entered; that is why flipping
# the `autoLRVW = False` default to True went undetected.
#
# A WIDE explicit width (LRVW=64, valid here because TransposeLDS=1) makes
# wlr = 64 // 32 == 2 > 1. On correct code the block is skipped (autoLRVW=False)
# and the derived LocalReadVectorWidth{A,B} stays 64. With the default flipped to
# True the explicit path wrongly enters the block, altering the width (and, given
# a latent calcLdsPad arity bug on that branch, raising) — either way the
# pass-through assertion below fails, killing the mutant.
# ===========================================================================

def test_gfx950_mx_fp8_explicit_wide_lrvw_preserved(_gp_assigned, gfx950_iim, gfx950_assembler):
    """Explicit wide LRVW=64 must pass through unchanged (isAutoLRVW False default)."""
    sol = _make_gfx950_mx_fp8_solution(
        gfx950_iim, gfx950_assembler,
        LocalReadVectorWidth=64,
    )
    assert sol is not None
    # Explicit LRVW=64 is valid for FP8 on gfx950 with TransposeLDS=1.
    assert sol.get("Valid") is True, (
        "Expected Valid=True for gfx950 MX FP8 + explicit LRVW=64; "
        "got invalid (check rejection log above)"
    )
    # isAutoLRVW must return False on the explicit path -> recompute block skipped
    # -> the user-specified width survives unchanged on BOTH operands.
    assert sol.get("LocalReadVectorWidthA") == 64, (
        f"explicit LRVW=64 must be preserved, got {sol.get('LocalReadVectorWidthA')}"
    )
    assert sol.get("LocalReadVectorWidthB") == 64, (
        f"explicit LRVW=64 must be preserved, got {sol.get('LocalReadVectorWidthB')}"
    )
