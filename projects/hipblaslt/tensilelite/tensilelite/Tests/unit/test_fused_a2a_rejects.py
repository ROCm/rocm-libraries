# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-validation guards for ``ProblemType["FusedA2A"]``.

The fused GEMM + all-to-all store path is only correct inside a narrow window,
so ``Solution.assignDerivedParameters`` rejects every configuration outside it
(gfx94x/gfx95x ISA, bf16 D, ``StreamK=0``, ``MacroTile0/1`` in {128, 256},
``GlobalSplitU=1``). The end-to-end proof of the path itself needs four GPUs and
cannot run in CI; these tests are the CPU-only unit coverage of the guard block,
in the same shape as ``test_halfplr_streamk_rejects``.

Each negative test flips exactly one knob off a known-good base and asserts the
*specific* diagnostic, so a config that happened to be rejected earlier for an
unrelated reason fails rather than passing vacuously. The positive test pins the
accept path (and the ``SupportUserGSU`` clear that follows it), without which the
guards below would be vacuously satisfied.

The base is the gfx950 fused config (``Tests/common/comm/gfx950/fused_a2a.yaml``)
retargeted to gfx942: bf16 TN GEMM, ``MatrixInstruction`` [16, 16, 32, 1, 1, 4, 4,
2, 2] (MT 128x128, the other tile size the guard admits), DepthU 32, PGR 2,
PLR 0, SIA 3, StreamK 0. gfx942 is used rather than gfx950 because it is the ISA
the coverage lane's container targets, and (9, 4) is on the supported side of the
ISA guard. The yaml's MT 256x256 + DepthU 64 + DirectToLds combination is
gfx950-only (gfx942 rejects b128 DirectToLds, and the 256x256 tile overruns the
64 KiB LDS budget), so the base is shrunk to the smaller admitted tile; the guard
under test does not distinguish them.
"""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# Snapshot the pristine process-global defaultSolution at import time; sibling
# unit tests mutate it in place, which breaks Solution construction in an
# order-dependent way. Same guard as test_halfplr_streamk_rejects.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

_ARCH = "gfx942"

# MT 128x128: MIWaveTile [4, 4] x MatrixInstM/N 16 x MIWaveGroup [2, 2].
_MI_MT128 = [16, 16, 32, 1, 1, 4, 4, 2, 2]
# MIWaveTile0 = 2 -> MacroTile0 = 2 * 16 * 2 = 64, outside {128, 256}.
_MI_MT64 = [16, 16, 32, 1, 1, 2, 4, 2, 2]


# ---------------------------------------------------------------------------
# Module-scoped toolchain fixtures (real gfx942 caps + assembler).
# ---------------------------------------------------------------------------
def _iim(arch):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(arch)
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip(f"amdclang++ in this environment does not support {arch}")
    return iim


@pytest.fixture(scope="module")
def gfx942_iim():
    return _iim(_ARCH)


@pytest.fixture(scope="module")
def gfx90a_iim():
    return _iim("gfx90a")


@pytest.fixture(scope="module")
def gfx950_iim():
    return _iim("gfx950")


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx942(gfx942_iim):
    """Assign process-global parameters for gfx942; restore after module."""
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx942_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


# ---------------------------------------------------------------------------
# Base solution: the fused_a2a.yaml fork permutation, retargeted to gfx942.
# ---------------------------------------------------------------------------
def _make_params(iim, arch=_ARCH, mi=None, mode="GemmA2A", **overrides):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa(arch)
    if mi is None:
        mi = list(_MI_MT128)

    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "b",
        "DestDataType": "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "FusedA2A": mode,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        # Seeded, not omitted: an omitted key aliases the process-global
        # defaultInternalSupportParams, which earlier StreamK solutions mutate.
        "InternalSupportParams": {"SupportUserGSU": True},
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],   # only [2] is consumed; [0]/[1] are derived.
        "WavefrontSize": 64,
        "DepthU": 32,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "ScheduleIterAlg": 3,
        "StaggerU": 0,
        "TileProcessingStrategy": "None",
        "GlobalSplitU": 1,
        "WorkGroupMapping": 1,
        "MIArchVgpr": False,
        "UseSubtileImpl": False,
    }
    params.update(overrides)

    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], iim
    )
    params.update(mi_params)
    # matrixInstructionToMIParameters writes result["ISA"], so an ISA override has
    # to be re-applied after the merge or it is silently clobbered.
    if "ISA" in overrides:
        params["ISA"] = overrides["ISA"]
    return params


def _derive(iim, assembler, capsys, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_params(iim, **overrides)
    sol = Solution(params, False, True, False, assembler, iim)
    return sol, capsys.readouterr().out


# ---------------------------------------------------------------------------
# Positive: the fused config is accepted, and clears SupportUserGSU.
# ---------------------------------------------------------------------------
def test_fused_a2a_base_is_accepted(_gp_gfx942, gfx942_iim, assembler, capsys):
    """FusedA2A=GemmA2A on gfx942, bf16 D, StreamK=0, MT 128x128, GSU=1 is valid.

    Without this the negative tests below could all pass on a base that was
    rejected for an unrelated reason.
    """
    sol, out = _derive(gfx942_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert (sol["MacroTile0"], sol["MacroTile1"]) == (128, 128)
    # The last-work-group DRAIN election counts work-groups, so a caller must not
    # be able to change GSU at runtime.
    assert sol["InternalSupportParams"]["SupportUserGSU"] is False


def test_fused_a2a_off_leaves_user_gsu_alone(_gp_gfx942, gfx942_iim, assembler, capsys):
    """Control: with the flag off the guard block does not run at all, so the
    same solution keeps the default SupportUserGSU. This is what makes the
    ``FusedA2A=Disabled`` default a no-op for every shipping solution."""
    sol, out = _derive(
        gfx942_iim, assembler, capsys, ProblemType={"FusedA2A": "Disabled"}
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["InternalSupportParams"]["SupportUserGSU"] is True


# ---------------------------------------------------------------------------
# Guard 1: non-gfx94x/gfx95x ISA -> rejected (SDMA packet layout is gfx9-only
# and pre-GFX12).
# ---------------------------------------------------------------------------
def test_fused_a2a_rejects_unsupported_isa(_gp_gfx942, gfx90a_iim, assembler, capsys):
    # gfx90a is a real MFMA target that is neither gfx94x nor gfx95x, so it
    # reaches the guard rather than tripping an earlier MatrixInstruction check.
    sol, out = _derive(gfx90a_iim, assembler, capsys, arch="gfx90a",
                       mi=[16, 16, 16, 1, 1, 4, 4, 2, 2])
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmA2A requires a gfx94x/gfx95x ISA" in out


# ---------------------------------------------------------------------------
# Guard 2: non-bf16 D -> rejected (the SDMA descriptor arithmetic folds a
# sizeof(bf16) shift).
# ---------------------------------------------------------------------------
def test_fused_a2a_rejects_non_bf16_d(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(
        gfx942_iim, assembler, capsys, ProblemType={"DestDataType": "s"}
    )
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmA2A only supports a bf16 D" in out


# ---------------------------------------------------------------------------
# Guard 3: StreamK processing -> rejected (the fused path requires
# nonpersistent processing).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("assignment", ["StaticGrid", "DynamicWorkQueue", "Hybrid"])
def test_fused_a2a_rejects_streamk(_gp_gfx942, gfx942_iim, assembler, capsys, assignment):
    sol, out = _derive(gfx942_iim, assembler, capsys,
                       TileProcessingStrategy="StreamK", WorkAssignment=assignment)
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmA2A requires TileProcessingStrategy=None" in out


# ---------------------------------------------------------------------------
# Guard 4: MacroTile0/1 outside {128, 256} -> rejected.
# ---------------------------------------------------------------------------
def test_fused_a2a_rejects_unsupported_macrotile(
    _gp_gfx942, gfx942_iim, assembler, capsys
):
    sol, out = _derive(gfx942_iim, assembler, capsys, mi=list(_MI_MT64))
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmA2A supports MacroTile0/1 in {128, 256}" in out


# ---------------------------------------------------------------------------
# Guard 5: GlobalSplitU != 1 -> rejected (any other GSU multiplies the arriving
# work-group count, so the DRAIN election can fire early). GSU=-1 resolves at
# runtime and is rejected for the same reason.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gsu", [-1, 2])
def test_fused_a2a_rejects_non_unit_gsu(_gp_gfx942, gfx942_iim, assembler, capsys, gsu):
    sol, out = _derive(gfx942_iim, assembler, capsys, GlobalSplitU=gsu)
    assert sol.get("Valid") is False
    assert "FusedA2A=GemmA2A requires GlobalSplitU=1" in out


# ---------------------------------------------------------------------------
# FusedA2A=A2AGemm.
# ---------------------------------------------------------------------------
_A2A_GEMM_GUARDS = [
    ({}, "a gfx94x/gfx95x ISA", "gfx90a"),
    ({"GlobalSplitU": -1}, "GlobalSplitU=1", "gfx942"),
    ({"GlobalSplitU": 2}, "GlobalSplitU=1", "gfx942"),
    ({"TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid"},
     "TileProcessingStrategy=None", "gfx942"),
    ({"DirectToVgprA": True}, "DirectToVgprA=DirectToVgprB=0", "gfx942"),
    ({"DirectToVgprB": True}, "DirectToVgprA=DirectToVgprB=0", "gfx942"),
    ({"UseSubtileImpl": True, "DepthU": 64}, "UseSubtileImpl=0", "gfx950"),
    ({"BufferLoad": False}, "BufferLoad=1", "gfx942"),
]


def _derive_a2a_gemm(request, assembler, capsys, arch="gfx942", **overrides):
    iim = request.getfixturevalue(f"{arch}_iim")
    return _derive(iim, assembler, capsys, arch=arch, mode="A2AGemm",
                   PrefetchLocalRead=1, **overrides)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_a2a_gemm_base_is_accepted(_gp_gfx942, request, assembler, capsys, arch):
    sol, out = _derive_a2a_gemm(request, assembler, capsys, arch=arch)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    # The shard loop and the block/grid tallies are sized for GSU=1, so a caller
    # must not be able to change GSU at runtime.
    assert sol["InternalSupportParams"]["SupportUserGSU"] is False


@pytest.mark.parametrize("overrides, msg, arch", _A2A_GEMM_GUARDS)
def test_a2a_gemm_rejects(_gp_gfx942, request, assembler, capsys, overrides, msg, arch):
    sol, out = _derive_a2a_gemm(request, assembler, capsys, arch=arch, **overrides)
    assert sol.get("Valid") is False
    assert "FusedA2A=A2AGemm requires " + msg in out


def test_a2a_sub_rounds_accepted_with_a2a_gemm(_gp_gfx942, request, assembler, capsys):
    sol, out = _derive_a2a_gemm(request, assembler, capsys, A2ASubRounds=2)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"


def test_a2a_sub_rounds_rejected_without_a2a_gemm(_gp_gfx942, request, assembler, capsys):
    sol, out = _derive_a2a_gemm(
        request, assembler, capsys, ProblemType={"FusedA2A": "Disabled"}, A2ASubRounds=2
    )
    assert sol.get("Valid") is False
    assert "A2ASubRounds > 1 requires FusedA2A=A2AGemm" in out
