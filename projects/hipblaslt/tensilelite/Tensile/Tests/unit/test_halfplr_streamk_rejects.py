# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HalfPLR + StreamK=3 gfx1250 solution-validation guards."""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# Snapshot defaultSolution at import; sibling tests mutate it in place.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# ---------------------------------------------------------------------------
# Module-scoped toolchain fixtures (real gfx1250 caps + assembler).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    """Assign process-global parameters for gfx1250; restore after module."""
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


# Base solution: known-good pure-F8 TN StreamK=3 HalfPLR.
def _make_params(gfx1250_iim, mi=None, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    if mi is None:
        # [M, N, K, B, ?, MIWaveTile0, MIWaveTile1, WaveGroup0, WaveGroup1]
        # MIWaveTile = [mi[5], mi[6]] = [2, 2] -> even (HalfPLR requires even).
        mi = [16, 16, 128, 1, 1, 2, 2, 2, 2]
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F8",
        "DestDataType": "F8",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,   # TN: UnrollMajorLDSA=True satisfies HalfPLR packing.
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4,          # StinkyTofu: required for HalfPLR on StreamK.
        "StaggerU": 0,
        "GlobalSplitU": 0,
        "InnerUnroll": 1,
        "TransposeLDS": -1,
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
        "TDMInst": 3,
        "LDSTrInst": False,
        "StreamK": 3,
        "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "HalfPLR": 1,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim
    )
    params.update(mi_params)
    return params


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_params(gfx1250_iim, **overrides)
    # printSolutionRejectionReason=True so reject() writes the reason to stdout.
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    out = capsys.readouterr().out
    return sol, out


# ---------------------------------------------------------------------------
# Positive: known-good combination must be accepted.
# ---------------------------------------------------------------------------
def test_halfplr_streamk_sk3_sia4_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Accept control so the guards below are not vacuous."""
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    # Sanity: the HalfPLR block ran and derived its A/B split.
    assert sol.get("HalfPLRA") is True


# ---------------------------------------------------------------------------
# Positive: every HalfPLR tensor mask and PGR depth is accepted.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half_plr", [1, 2, 3])
@pytest.mark.parametrize("pgr", [1, 2])
def test_halfplr_supports_prefetch_across_persistent(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, half_plr, pgr
):
    sol, out = _derive(
        gfx1250_iim,
        assembler,
        capsys,
        HalfPLR=half_plr,
        PrefetchGlobalRead=pgr,
        PrefetchAcrossPersistent=1,
        StreamKForceDPOnly=1,
        AssertSummationElementMultiple=256,
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("HalfPLRA") is bool(half_plr & 1)
    assert sol.get("HalfPLRB") is bool(half_plr & 2)


@pytest.mark.parametrize("half_plr", [1, 2, 3])
@pytest.mark.parametrize("pgr", [1, 2])
def test_halfplr_pap_supports_tail_capable_solution(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, half_plr, pgr
):
    sol, out = _derive(
        gfx1250_iim,
        assembler,
        capsys,
        HalfPLR=half_plr,
        PrefetchGlobalRead=pgr,
        PrefetchAcrossPersistent=1,
        StreamKForceDPOnly=1,
        AssertSummationElementMultiple=32,
    )
    assert sol.get("Valid") is True, f"expected tail-capable accept, rejected with: {out!r}"


# ---------------------------------------------------------------------------
# Guard 1: HalfPLR + PAP outside StreamK=3 + StreamKForceDPOnly=1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_halfplr_pap_rejects_without_force_dp_only(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, half_plr
):
    # Same knobs as the accept test above with StreamKForceDPOnly flipped to 0:
    # only the DP-only handoff has been validated for the out-of-line PAP block.
    sol, out = _derive(
        gfx1250_iim,
        assembler,
        capsys,
        HalfPLR=half_plr,
        PrefetchAcrossPersistent=1,
        StreamKForceDPOnly=0,
        AssertSummationElementMultiple=256,
    )
    assert sol.get("Valid") is False
    assert ("HalfPLR + PrefetchAcrossPersistent currently requires StreamK = 3 "
            "and StreamKForceDPOnly = 1") in out


def test_halfplr_pap_rejects_on_other_streamk_modes(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    # The StreamK term of the HalfPLR guard is a backstop: on any other non-zero
    # StreamK mode the generic PAP guard rejects first and returns before the
    # HalfPLR block runs. (StreamK=0 rejects nothing at all -- it silently clears
    # PrefetchAcrossPersistent along with the rest of the StreamK settings.)
    sol, out = _derive(
        gfx1250_iim,
        assembler,
        capsys,
        PrefetchAcrossPersistent=1,
        StreamK=2,
        AssertSummationElementMultiple=256,
    )
    assert sol.get("Valid") is False
    assert "PrefetchAcrossPersistent is currently supported only with StreamK in [3, 4, 5]" in out


def test_halfplr_pap_is_cleared_without_streamk(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    # StreamK=0 drops PrefetchAcrossPersistent with the other StreamK settings, so
    # HalfPLR stays valid and no PAP code is generated.
    sol, out = _derive(
        gfx1250_iim,
        assembler,
        capsys,
        PrefetchAcrossPersistent=1,
        StreamK=0,
        GlobalSplitU=1,
        AssertSummationElementMultiple=256,
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("PrefetchAcrossPersistent") == 0


# ---------------------------------------------------------------------------
# Guard 2: HalfPLR + StreamK + ScheduleIterAlg != 4.
# ---------------------------------------------------------------------------
def test_halfplr_streamk_rejects_non_stinkytofu_sia(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    # SIA=0 is otherwise legal for HalfPLR (SIA in {0,4}); it is only the
    # StreamK combination that pins it to 4.
    sol, out = _derive(gfx1250_iim, assembler, capsys, ScheduleIterAlg=0)
    assert sol.get("Valid") is False
    assert "HalfPLR on StreamK requires ScheduleIterAlg = 4" in out


# ---------------------------------------------------------------------------
# Guard 3: HalfPLR + UseSubtileImpl.
# ---------------------------------------------------------------------------
def test_halfplr_rejects_use_subtile_impl(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, UseSubtileImpl=True)
    assert sol.get("Valid") is False
    assert "HalfPLR is not supported with UseSubtileImpl" in out


# ---------------------------------------------------------------------------
# Guard 4: HalfPLR + TDMFuse=1 at a divergent pair. The increment mask rides
# in the module the single-buffered fill relocation moves.
# ---------------------------------------------------------------------------
def test_halfplr_rejects_tdmfuse1_at_a_divergent_pair(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    sol, out = _derive(
        gfx1250_iim,
        assembler,
        capsys,
        TDMFuse=1,
        PrefetchGlobalReadA=1,
        PrefetchGlobalReadB=2,
        mi=[16, 16, 128, 1, 1, 2, 16, 2, 2],
        WorkGroup=[32, 4, 1],
        MXScaleFormat="InMemorySwizzle",
        LDSTrInst=True,
        ProblemType={
            "MacDataTypeA": "F8",
            "MacDataTypeB": "F4",
            "DestDataType": "s",
            "MXBlockA": 32,
            "MXBlockB": 32,
            "DataTypeMXSA": "E8",
            "DataTypeMXSB": "E8",
        },
    )
    assert sol.get("Valid") is False
    assert "TDMFuse=1 requires HalfPLR=0 at a divergent decoupled pair" in out
