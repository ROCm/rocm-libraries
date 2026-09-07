# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-validation guards for ``ProblemType["FusedA2AMode"] == 1`` (GatheredB).

The guards admit GlobalSplitU=1, StreamK=0, StaggerU=0, UseSubtileImpl=0, no
DirectToVgpr, non-batched, non-sparse, and FusedGemmA2A off.

Each negative test flips one knob off a known-good base and asserts the specific
diagnostic. The positive test pins the accept path.

The base is the gfx950 champion retargeted to gfx942: bf16 TN, MatrixInstruction
[16, 16, 32, 1, 1, 4, 4, 2, 2] (MT 128x128), DepthU 32, PGR 2, PLR 1, SIA 3.
"""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# Sibling unit tests mutate the process-global defaultSolution in place.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

_ARCH = "gfx942"

# MT 128x128: MIWaveTile [4, 4] x MatrixInstM/N 16 x MIWaveGroup [2, 2].
_MI_MT128 = [16, 16, 32, 1, 1, 4, 4, 2, 2]


@pytest.fixture(scope="module")
def gfx942_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(_ARCH)
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip(f"amdclang++ in this environment does not support {_ARCH}")
    return iim


@pytest.fixture(scope="module")
def gfx950_iim():
    """Capability map for the one ISA family that honours UseSubtileImpl."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx950")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx942(gfx942_iim):
    """Assign process-global parameters for gfx942; restore after module."""
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

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


def _make_params(iim, arch=_ARCH, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa(arch)
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
        "Batched": False,
        "FusedA2AMode": 1,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        # Seeded, not omitted: an omitted key aliases the process-global
        # defaultInternalSupportParams.
        "InternalSupportParams": {"SupportUserGSU": True},
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],   # only [2] is consumed; [0]/[1] are derived.
        "WavefrontSize": 64,
        "DepthU": 32,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "StaggerU": 0,
        "StreamK": 0,
        "GlobalSplitU": 1,
        "WorkGroupMapping": 1,
        "MIArchVgpr": False,
        "UseSubtileImpl": False,
    }
    params.update(overrides)

    params.update(
        matrixInstructionToMIParameters(
            mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], iim
        )
    )
    return params


def _derive(iim, assembler, capsys, arch=_ARCH, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_params(iim, arch=arch, **overrides)
    sol = Solution(params, False, True, False, assembler, iim)
    return sol, capsys.readouterr().out


# ---------------------------------------------------------------------------
# Positive and control arms.
# ---------------------------------------------------------------------------
def test_a2a_mode_base_is_accepted(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(gfx942_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["ProblemType"]["NumIndicesSummation"] == 1
    assert sol["ProblemType"]["IndicesSummation"] == [2]


def test_a2a_mode_off_keeps_one_summation_index(_gp_gfx942, gfx942_iim, assembler, capsys):
    """Control: mode 0 is the stock single-summation GEMM."""
    sol, out = _derive(gfx942_iim, assembler, capsys, ProblemType={"FusedA2AMode": 0})
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["ProblemType"]["NumIndicesSummation"] == 1


@pytest.mark.parametrize(
    "param,value",
    [
        ("GlobalSplitU", 2),
        ("StreamK", 1),
        ("StaggerU", 32),
        ("UseSubtileImpl", True),
        ("DirectToVgprA", True),
    ],
)
def test_a2a_mode_off_does_not_apply_the_guards(
    _gp_gfx942, gfx942_iim, assembler, capsys, param, value
):
    """Control: none of the guards fire when the mode is off."""
    _, out = _derive(
        gfx942_iim, assembler, capsys, ProblemType={"FusedA2AMode": 0}, **{param: value}
    )
    assert "FusedA2AMode=1" not in out


# ---------------------------------------------------------------------------
# Solution-level guards.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gsu", [-1, 2])
def test_a2a_mode_rejects_non_unit_gsu(_gp_gfx942, gfx942_iim, assembler, capsys, gsu):
    sol, out = _derive(gfx942_iim, assembler, capsys, GlobalSplitU=gsu)
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 requires GlobalSplitU=1" in out


@pytest.mark.parametrize("streamk", [1, 2, 3])
def test_a2a_mode_rejects_streamk(_gp_gfx942, gfx942_iim, assembler, capsys, streamk):
    sol, out = _derive(gfx942_iim, assembler, capsys, StreamK=streamk)
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 requires StreamK=0" in out


@pytest.mark.parametrize("staggeru", [2, 32])
def test_a2a_mode_rejects_staggeru(_gp_gfx942, gfx942_iim, assembler, capsys, staggeru):
    sol, out = _derive(gfx942_iim, assembler, capsys, StaggerU=staggeru)
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 requires StaggerU=0" in out


def test_a2a_mode_base_is_accepted_on_gfx950(_gp_gfx942, gfx950_iim, assembler, capsys):
    sol, out = _derive(gfx950_iim, assembler, capsys, arch="gfx950")
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"


def test_a2a_mode_rejects_subtile(_gp_gfx942, gfx950_iim, assembler, capsys):
    # UseSubtileImpl is forced off outside gfx950/gfx1250.
    sol, out = _derive(gfx950_iim, assembler, capsys, arch="gfx950", UseSubtileImpl=True)
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 requires UseSubtileImpl=0" in out


@pytest.mark.parametrize("tc", ["A", "B"])
def test_a2a_mode_rejects_direct_to_vgpr(_gp_gfx942, gfx942_iim, assembler, capsys, tc):
    sol, out = _derive(gfx942_iim, assembler, capsys, **{f"DirectToVgpr{tc}": True})
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 requires DirectToVgprA=DirectToVgprB=0" in out


# ---------------------------------------------------------------------------
# ProblemType-level guards.
# ---------------------------------------------------------------------------
def test_a2a_mode_rejects_batched(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(gfx942_iim, assembler, capsys, ProblemType={"Batched": True})
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 does not support batched problems" in out


@pytest.mark.parametrize("sparse", [1, 2])
def test_a2a_mode_rejects_sparse(_gp_gfx942, gfx942_iim, assembler, capsys, sparse):
    sol, out = _derive(gfx942_iim, assembler, capsys, ProblemType={"Sparse": sparse})
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 does not support sparse problems" in out


def test_a2a_mode_rejects_fused_gemm_a2a(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(
        gfx942_iim, assembler, capsys, ProblemType={"FusedGemmA2A": True}
    )
    assert sol.get("Valid") is False
    assert "FusedA2AMode=1 and FusedGemmA2A are mutually exclusive" in out


# ---------------------------------------------------------------------------
# PrefetchAcrossPersistent carries no guard of its own.
# ---------------------------------------------------------------------------
def test_a2a_mode_pap_is_cleared_by_streamk_off(_gp_gfx942, gfx942_iim, assembler, capsys):
    """StreamK=0 clears PrefetchAcrossPersistent."""
    sol, out = _derive(gfx942_iim, assembler, capsys, PrefetchAcrossPersistent=True)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["PrefetchAcrossPersistent"] == 0
