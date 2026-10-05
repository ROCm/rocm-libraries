# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LDSTrInst(A/B) resolution and CMS (Custom Main-Loop Schedule) compatibility.

hasCustomSchedule() (Tensile/Components/CustomSchedule.py) only ever reads the
single global kernel["LDSTrInst"] value when selecting a registered schedule,
and every registered schedule function branches on that one shared value for
BOTH A and B tensor local-read scheduling - there is no per-tensor awareness
anywhere in the CMS schedule-matching mechanism. A kernel whose resolved
LDSTrInstA/LDSTrInstB diverge from each other, or from the global LDSTrInst
value actually used to pick the schedule, would silently get a CMS schedule
inconsistent with its real per-tensor LDS-transpose configuration.

These tests build real, fully-derived Solutions (gfx950) and pin:
  * the known-good CMS-enabled baseline is accepted,
  * CMS + LDSTrInstA != LDSTrInstB is rejected (any layout, not just NT),
  * CMS + LDSTrInstA == LDSTrInstB but != LDSTrInst (global) is rejected,
  * CMS + LDSTrInstA/LDSTrInstB left on -1 (auto) correctly inherit the
    global LDSTrInst value and are accepted.

A second group covers the SourceSwap + Sparse==2 (sparse-B) special case in
the LDSTrInstA/B resolution block: LDSTrInstA defaults to 0 (disabled) when
SourceSwap and Sparse==2, independent of the global LDSTrInst value, unless
explicitly overridden; LDSTrInstB always follows the global LDSTrInst value.

The harness mirrors test_halfplr_streamk_rejects.py: real gfx950 capability
maps from makeIsaInfoMap (needs amdclang++; skips if the toolchain cannot
target gfx950) and a real assembler, feeding Solution.__init__ which runs
assignDerivedParameters end-to-end.
"""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# Snapshot the pristine process-global defaultSolution at import time (mirrors
# test_halfplr_streamk_rejects.py - sibling unit tests mutate it in place).
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# ---------------------------------------------------------------------------
# Module-scoped toolchain fixtures (real gfx950 caps + assembler).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx950_iim():
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
def _gp_gfx950(gfx950_iim):
    """Assign process-global parameters for gfx950; restore after module."""
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

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


# ---------------------------------------------------------------------------
# Base solution #1: known-good CMS-enabled bf16 NN gfx950 kernel.
#   MI [16,16,32,1,1,8,8,2,2] MT256x256 DepthU=64, matches a registered
#   CustomSchedule.py schedule (query_cms_kernels("b", "NN") minimum combo).
# ---------------------------------------------------------------------------
def _make_cms_params(gfx950_iim, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx950")
    mi = [16, 16, 32, 1, 1, 8, 8, 2, 2]
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "b",
        "DestDataType": "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": False,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [32, 8, 1],
        "WavefrontSize": 64,
        "DepthU": 64,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "ExpandPointerSwap": False,
        "TransposeLDS": 1,
        "LocalReadVectorWidth": 8,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "DirectToLds": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "StaggerU": 0,
        "WorkGroupMapping": 1,
        "1LDSBuffer": 0,
        "NonTemporalD": 0,
        "SourceSwap": False,
        "UseSgprForGRO": 0,
        "UseCustomMainLoopSchedule": 1,
        "LDSTrInst": False,
        "StreamK": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMappingXCC": 1,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx950_iim
    )
    params.update(mi_params)
    return params


def _derive_cms(gfx950_iim, assembler, capsys, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_cms_params(gfx950_iim, **overrides)
    sol = Solution(params, False, True, False, assembler, gfx950_iim)
    out = capsys.readouterr().out
    return sol, out


# ---------------------------------------------------------------------------
# Positive: the known-good CMS combination (matching A/B/global) is ACCEPTED.
# ---------------------------------------------------------------------------
def test_cms_matching_ldstrinst_is_accepted(_gp_gfx950, gfx950_iim, assembler, capsys):
    sol, out = _derive_cms(gfx950_iim, assembler, capsys, LDSTrInstA=0, LDSTrInstB=0, LDSTrInst=False)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("UseCustomMainLoopSchedule") == 1
    assert sol.get("LDSTrInstA") == 0
    assert sol.get("LDSTrInstB") == 0


def test_cms_auto_ldstrinst_inherits_global(_gp_gfx950, gfx950_iim, assembler, capsys):
    """LDSTrInstA/LDSTrInstB left on -1 (auto) must inherit the global LDSTrInst."""
    sol, out = _derive_cms(gfx950_iim, assembler, capsys, LDSTrInstA=-1, LDSTrInstB=-1, LDSTrInst=False)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("LDSTrInstA") == 0
    assert sol.get("LDSTrInstB") == 0


# ---------------------------------------------------------------------------
# Negative: CMS + LDSTrInstA != LDSTrInstB must be rejected (broadened fix -
# previously only checked in NT layout; this base config is NN).
# ---------------------------------------------------------------------------
def test_cms_rejects_a_b_mismatch(_gp_gfx950, gfx950_iim, assembler, capsys):
    sol, out = _derive_cms(gfx950_iim, assembler, capsys, LDSTrInstA=1, LDSTrInstB=0, LDSTrInst=False)
    assert sol.get("Valid") is False, "expected reject on LDSTrInstA != LDSTrInstB under CMS"
    assert "UseCustomMainLoopSchedule=1 (CMS) requires LDSTrInstA" in out, out


# ---------------------------------------------------------------------------
# Negative: CMS + LDSTrInstA == LDSTrInstB but != global LDSTrInst must be
# rejected (the new check added by the fix; previously silently accepted).
# ---------------------------------------------------------------------------
def test_cms_rejects_global_mismatch(_gp_gfx950, gfx950_iim, assembler, capsys):
    sol, out = _derive_cms(gfx950_iim, assembler, capsys, LDSTrInstA=1, LDSTrInstB=1, LDSTrInst=False)
    assert sol.get("Valid") is False, "expected reject on LDSTrInstA/B != global LDSTrInst under CMS"
    assert "UseCustomMainLoopSchedule=1 (CMS) requires LDSTrInstA" in out, out


# ---------------------------------------------------------------------------
# Base solution #2: SourceSwap + Sparse==2 (sparse-B) gfx950 kernel, from
# Tensile/Tests/common/sparse/gfx950/spmm_bf16_sb.yaml.
# ---------------------------------------------------------------------------
def _make_sparse_params(gfx950_iim, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx950")
    mi = [16, 16, 64, 1, 1, 2, 2, 2, 2]
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": False,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "Sparse": 2,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": 64,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ClusterLocalRead": 1,
        "ScheduleIterAlg": 3,
        "ExpandPointerSwap": False,
        "TransposeLDS": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsPadMetadata": -1,
        "1LDSBuffer": -1,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "SourceSwap": True,
        "StoreRemapVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "StaggerU": 0,
        "WorkGroupMapping": 1,
        "UseCustomMainLoopSchedule": 0,
        "LDSTrInst": True,
        "StreamK": 0,
        "InnerUnroll": 1,
        "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "DirectToLds": 0,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx950_iim
    )
    params.update(mi_params)
    return params


def _derive_sparse(gfx950_iim, assembler, capsys, **overrides):
    params = _make_sparse_params(gfx950_iim, **overrides)
    sol = Solution(params, False, True, False, assembler, gfx950_iim)
    out = capsys.readouterr().out
    return sol, out


@pytest.mark.parametrize("ldstrinst_global", [True, False])
def test_sparse_b_sourceswap_defaults_ldstrinstA_to_zero(
    _gp_gfx950, gfx950_iim, assembler, capsys, ldstrinst_global
):
    """SourceSwap + Sparse==2 forces LDSTrInstA to 0, independent of global.

    LDSTrInstB (unaffected by the exception) must still follow the global
    LDSTrInst value.
    """
    sol, out = _derive_sparse(gfx950_iim, assembler, capsys, LDSTrInst=ldstrinst_global)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("LDSTrInstA") == 0
    assert sol.get("LDSTrInstB") == int(ldstrinst_global)


def test_sparse_b_sourceswap_ldstrinstA_explicit_override_preserved(
    _gp_gfx950, gfx950_iim, assembler, capsys
):
    """An explicit LDSTrInstA still works when set to 1 in the YAML/params,

    i.e. the SourceSwap + Sparse==2 default-to-0 exception only applies when
    LDSTrInstA is left on -1 (auto); an explicit override is not clobbered.
    """
    sol, out = _derive_sparse(gfx950_iim, assembler, capsys, LDSTrInstA=1, LDSTrInst=True)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("LDSTrInstA") == 1
    assert sol.get("LDSTrInstB") == 1
