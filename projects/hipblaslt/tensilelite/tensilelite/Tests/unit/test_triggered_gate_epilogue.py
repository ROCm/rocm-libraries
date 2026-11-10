# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution guards, serialization and library-build exclusion for TriggeredGateEpilogue."""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

_ARCH = "gfx942"
_MI_MT128 = [16, 16, 16, 1, 1, 4, 4, 2, 2]


@pytest.fixture(scope="module")
def _gp_gfx942(gfx942_iim):
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


def _make_params(iim, arch=_ARCH, mi=None, **overrides):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa(arch)
    if mi is None:
        mi = list(_MI_MT128)
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "b",
        "DestDataType": "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": False,
        "Batched": True,
        "TriggeredGateEpilogue": True,
    }
    problem_type.update(overrides.pop("ProblemType", {}))
    params = {
        "ProblemType": problem_type,
        "InternalSupportParams": {"SupportUserGSU": True},
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": 32,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "TileProcessingStrategy": "None",
        "GlobalSplitU": 1,
        "WorkGroupMapping": 1,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], iim))
    if "ISA" in overrides:
        params["ISA"] = overrides["ISA"]
    return params


def _derive(iim, assembler, capsys, **overrides):
    sol = Solution(_make_params(iim, **overrides), False, True, False, assembler, iim)
    return sol, capsys.readouterr().out


def test_gate_epilogue_base_is_accepted(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(gfx942_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["InternalSupportParams"]["SupportUserGSU"] is False
    assert "TGE" in str(sol["ProblemType"]).split("_")


def test_gate_epilogue_off_is_a_no_op(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(gfx942_iim, assembler, capsys, ProblemType={"TriggeredGateEpilogue": False})
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["InternalSupportParams"]["SupportUserGSU"] is True
    assert "TGE" not in str(sol["ProblemType"])


def test_gate_epilogue_rejects_unsupported_isa(_gp_gfx942, gfx90a_iim, assembler, capsys):
    sol, out = _derive(gfx90a_iim, assembler, capsys, arch="gfx90a")
    assert sol.get("Valid") is False
    assert "TriggeredGateEpilogue requires a gfx94x/gfx95x ISA" in out


def test_gate_epilogue_rejects_non_bf16_d(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(gfx942_iim, assembler, capsys, ProblemType={"DestDataType": "s"})
    assert sol.get("Valid") is False
    assert "TriggeredGateEpilogue only supports a bf16 D" in out


@pytest.mark.parametrize("assignment", ["StaticGrid", "DynamicWorkQueue", "Hybrid"])
def test_gate_epilogue_rejects_streamk(_gp_gfx942, gfx942_iim, assembler, capsys, assignment):
    sol, out = _derive(gfx942_iim, assembler, capsys,
                       TileProcessingStrategy="StreamK", WorkAssignment=assignment)
    assert sol.get("Valid") is False
    assert "TriggeredGateEpilogue requires StreamK=0" in out


@pytest.mark.parametrize("gsu", [2, -1])
def test_gate_epilogue_rejects_gsu(_gp_gfx942, gfx942_iim, assembler, capsys, gsu):
    sol, out = _derive(gfx942_iim, assembler, capsys, GlobalSplitU=gsu)
    assert sol.get("Valid") is False
    assert "TriggeredGateEpilogue requires GlobalSplitU=1" in out


@pytest.mark.parametrize("wgm,xcc", [(4, 1), (8, 1), (8, 8)])
def test_gate_epilogue_accepts_wgm(_gp_gfx942, gfx942_iim, assembler, capsys, wgm, xcc):
    sol, out = _derive(gfx942_iim, assembler, capsys, WorkGroupMapping=wgm, WorkGroupMappingXCC=xcc)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"


def test_gate_epilogue_rejects_fused_a2a(_gp_gfx942, gfx942_iim, assembler, capsys):
    sol, out = _derive(gfx942_iim, assembler, capsys, ProblemType={"FusedGemmA2A": True})
    assert sol.get("Valid") is False
    assert "TriggeredGateEpilogue and FusedGemmA2A are mutually exclusive" in out


def test_gate_epilogue_serializes_only_when_set(_gp_gfx942, gfx942_iim, assembler, capsys):
    import tensilelite.Contractions as C

    on, _ = _derive(gfx942_iim, assembler, capsys)
    off, _ = _derive(gfx942_iim, assembler, capsys, ProblemType={"TriggeredGateEpilogue": False})
    pt_on = C.ProblemType.FromOriginalState(on["ProblemType"])
    pt_off = C.ProblemType.FromOriginalState(off["ProblemType"])
    for pt in (pt_on, pt_off):
        pt.mxScaleFormat = 0  # Solution-level; Contractions.Solution fills it in
    assert pt_on.state().get("triggeredGateEpilogue") is True
    assert "triggeredGateEpilogue" not in pt_off.state()

    def tags(pt):
        return [p.tag for p in pt.predicates(includeBatch=True, includeOperation=True, includeType=True)]

    assert "TriggeredGateEpilogue" in tags(pt_on)
    assert "TriggeredGateEpilogue" not in tags(pt_off)


@pytest.mark.parametrize(
    "problem_type,expected",
    [
        ({"TriggeredGateEpilogue": False}, True),
        ({}, True),
        (None, True),
        ({"TriggeredGateEpilogue": True}, False),
    ],
)
def test_gate_epilogue_never_library_built(problem_type, expected):
    from tensilelite.TensileCreateLibrary.Run import _includeTriggeredGateProblemType

    assert _includeTriggeredGateProblemType(problem_type) is expected
