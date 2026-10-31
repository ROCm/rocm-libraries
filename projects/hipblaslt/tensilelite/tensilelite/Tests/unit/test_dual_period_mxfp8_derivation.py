# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derivation gate for the dual-period MXFP8 subtile path (gfx950).

The non-scale F8 subtile kernel wins at MT256x256 / MIWT 8x4 / DU128 / PGR2.
MXFP8 historically forced DepthU to a multiple of 2*MI_K=256 (one scale local
read covers two scale MMA tiles in K) and then split that into a dataDU=128
multi-DU pipeline, which rejects PGR>1.  The dual-period derivation breaks that
coupling on the gfx950 HostPreSwizzle pipeline: the data loop runs single-DU at
DepthU=128 (so isSubtileMultiDU is False and PGR2 stays legal) while the
pre-shuffled scale epoch spans 2*DepthU=256 K and is reused by two data
iterations.

These tests pin the derivation contract:
  * PGR2 + MXFP8 (gfx950) is ACCEPTED at DepthU=128 (previously rejected).
  * _DualPeriodMXFP8 is set, _ScalePeriodUA/B == 256, _ScaleReuseA/B == 2,
    _DepthUMXSA/B == 8 (256/32), _DepthUA/B == DepthU == 128.
  * The kernel stays single-DU (isSubtileMultiDU False).
  * The legacy DU256/PGR1 multi-DU MXFP8 derivation is UNCHANGED (control).
"""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution
from tensilelite.Common.Utilities import isSubtileMultiDU

pytestmark = pytest.mark.unit


_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# ---------------------------------------------------------------------------
# Toolchain fixtures (real gfx950 caps + assembler).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx950_iim():
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx950")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx950(gfx950_iim):
    """Assign process-global parameters for gfx950; restore after module."""
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx950_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _make_params(gfx950_iim, mi, depth_u, pgr, plr, **overrides):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx950")
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F8",
        "DestDataType": "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,   # TN
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": depth_u,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": pgr,
        "PrefetchLocalRead": plr,
        "ScheduleIterAlg": 3,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "DirectToLds": 1,
        "StreamK": 0,
        "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": True,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 4,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx950_iim
    )
    params.update(mi_params)
    return params


def _derive(gfx950_iim, assembler, capsys, mi, depth_u, pgr, plr, **overrides):
    params = _make_params(gfx950_iim, mi, depth_u, pgr, plr, **overrides)
    sol = Solution(params, False, True, False, assembler, gfx950_iim)
    out = capsys.readouterr().out
    return sol, out


# MT256x256, MIWT 8x4, MIWG 2x4 — the winning non-scale geometry.
_MI_8WAVE = [16, 16, 128, 1, 1, 8, 4, 2, 4]
# MT256x256, MIWT 8x8, MIWG 2x2 — the shipped DU256/PGR1 MXFP8 geometry.
_MI_PGR1 = [16, 16, 128, 1, 1, 8, 8, 2, 2]


# ---------------------------------------------------------------------------
# Dual-period path: PGR2 + DU128 MXFP8 is ACCEPTED with the right scale period.
# ---------------------------------------------------------------------------
def test_dual_period_accepts_pgr2_du128(_gp_gfx950, gfx950_iim, assembler, capsys):
    sol, out = _derive(gfx950_iim, assembler, capsys, _MI_8WAVE, 128, 2, 0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("_DualPeriodMXFP8") is True
    assert sol["DepthU"] == 128
    assert sol["_DepthUA"] == 128 and sol["_DepthUB"] == 128
    assert sol["_ScalePeriodUA"] == 256 and sol["_ScalePeriodUB"] == 256
    assert sol["_ScaleReuseA"] == 2 and sol["_ScaleReuseB"] == 2
    assert sol["_DepthUMXSA"] == 8 and sol["_DepthUMXSB"] == 8
    # single-DU: data per-uid DepthU equals loop DepthU -> PGR2 legal.
    assert isSubtileMultiDU(sol) is False


# ---------------------------------------------------------------------------
# Control: shipped DU256/PGR1 MXFP8 derivation is UNCHANGED (multi-DU, no
# dual-period flag).
# ---------------------------------------------------------------------------
def test_legacy_du256_pgr1_unchanged(_gp_gfx950, gfx950_iim, assembler, capsys):
    sol, out = _derive(gfx950_iim, assembler, capsys, _MI_PGR1, 256, 1, 1)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("_DualPeriodMXFP8") is False
    assert sol["DepthU"] == 256
    # legacy multi-DU: data SRD advances at dataDU=128 < DepthU=256.
    assert sol["_DepthUA"] == 128 and sol["_DepthUB"] == 128
    assert isSubtileMultiDU(sol) is True


# ---------------------------------------------------------------------------
# Control: the legacy multi-DU AB_B8 path still rejects PGR2 at DU256 (the
# dual-period gate must not accidentally relax this).
# ---------------------------------------------------------------------------
def test_legacy_du256_pgr2_still_rejected(_gp_gfx950, gfx950_iim, assembler, capsys):
    sol, out = _derive(gfx950_iim, assembler, capsys, _MI_PGR1, 256, 2, 1)
    assert sol.get("Valid") is False
    assert sol.get("_DualPeriodMXFP8") is False
