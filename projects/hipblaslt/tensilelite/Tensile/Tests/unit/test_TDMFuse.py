################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Components.TDMFuse.

Related coverage that is not this module:
  HalfPLR + TDMFuse=1 at a divergent pair -> test_halfplr_streamk_rejects.py
  TDMFuse kernel-name tokens              -> characterization/Naming/test_mut_Naming_char.py
"""
import copy

import pytest

from Tensile.Components.DecouplePGR import decouplePGRBlocks
from Tensile.Components.TDMFuse import (
    tdmBothTensors,
    tdmFuseAMx,
    tdmFusePaired,
    tdmWaveCompIdMode,
    tdmWaveComponents,
    tdmWavePartition,
)
from Tensile.Common.GlobalParameters import defaultSolution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))
_TENSORS = ("A", "MXSA", "MXSB", "B")
_NO_MX_ON_B = {"MacDataTypeB": "F8", "DataTypeMXSB": "E8", "MXBlockB": 0}
_ONE_WAVE_MI = [16, 16, 128, 1, 1, 2, 16, 1, 1]
_ONE_WAVE_WG = [32, 1, 1]
_TDMSPLIT_DISABLED = pytest.mark.xfail(
    reason="TDMSplit is currently disabled upstream (PR #10911)", strict=False)


def _ks(fuse=1, pgrA=1, pgrB=2, **overrides):
    ks = {
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "NumWaves": 4,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    ks.update(overrides)
    return ks


def test_tdm_both_tensors():
    assert tdmBothTensors({"TDMInst": 3}) is True
    assert tdmBothTensors({"TDMInst": 1}) is False
    assert tdmBothTensors({"TDMInst": 2}) is False
    assert tdmBothTensors({"TDMInst": 0}) is False


def test_fuse_predicates_are_exclusive():
    assert tdmFusePaired(_ks(fuse=1)) is True
    assert tdmFuseAMx(_ks(fuse=1)) is False
    assert tdmFuseAMx(_ks(fuse=2)) is True
    assert tdmFusePaired(_ks(fuse=2)) is False
    assert tdmFusePaired(_ks(fuse=0)) is False
    assert tdmFuseAMx(_ks(fuse=0)) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"NumWaves": 1},
        {"UseSubtileImpl": True},
        {"TDMSplit": True},
        {"TDMInst": 1},
        {"TDMInst": 2},
        {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}},
        {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}},
    ],
)
def test_paired_declines_outside_its_envelope(overrides):
    assert tdmFusePaired(_ks(**overrides)) is False
    assert tdmFuseAMx(_ks(fuse=2, **overrides)) is False


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1), (2, 2), (0, 2), (1, 1)])
def test_fuse_predicates_do_not_key_on_block_counts(pgrA, pgrB):
    assert tdmFusePaired(_ks(pgrA=pgrA, pgrB=pgrB)) is True
    assert tdmFuseAMx(_ks(fuse=2, pgrA=pgrA, pgrB=pgrB)) is True


@pytest.mark.parametrize("numWaves", [2, 4, 8])
def test_paired_holds_at_every_wave_count_parity_can_split(numWaves):
    assert tdmFusePaired(_ks(NumWaves=numWaves)) is True
    assert tdmFuseAMx(_ks(fuse=2, NumWaves=numWaves)) is (numWaves == 4)


@pytest.mark.parametrize(
    "tc, waves",
    [("A", (0, 2)), ("MXSA", (1, 3)), ("MXSB", (0, 2)), ("B", (1, 3))],
)
def test_paired_wave_assignment(tc, waves):
    numComp, got = tdmWavePartition(_ks(), tc)
    assert (numComp, got) == (2, waves)
    assert tdmWaveCompIdMode(_ks(), tc) == "parity"
    assert tdmWaveComponents(_ks(), tc) == (2, 1)


@pytest.mark.parametrize(
    "tc, numComp, waves, mode, shift",
    [
        ("A", 2, (0, 1), "waveIdx", 0),
        ("MXSA", 1, (2,), "zero", None),
        ("MXSB", 1, (3,), "zero", None),
        ("B", 4, (0, 1, 2, 3), "waveIdx", 0),
    ],
)
def test_amx_wave_assignment(tc, numComp, waves, mode, shift):
    ks = _ks(fuse=2)
    assert tdmWavePartition(ks, tc) == (numComp, waves)
    assert tdmWaveCompIdMode(ks, tc) == mode
    assert tdmWaveComponents(ks, tc) == (numComp, shift)


def test_paired_only_swaps_scale_parity_against_default():
    paired, default = _ks(fuse=1), _ks(fuse=0)
    assert tdmWavePartition(paired, "A") == tdmWavePartition(default, "A")
    assert tdmWavePartition(paired, "B") == tdmWavePartition(default, "B")
    assert tdmWavePartition(paired, "MXSA") == tdmWavePartition(default, "MXSB")
    assert tdmWavePartition(paired, "MXSB") == tdmWavePartition(default, "MXSA")
    for wave in range(4):
        carried = [tc for tc in _TENSORS if wave in tdmWavePartition(paired, tc)[1]]
        assert len(carried) == 2
        assert sum(1 for tc in carried if "MXS" in tc) == 1


# ---------------------------------------------------------------------------
# Solution wiring. Needs amdclang++ gfx1250; skipped otherwise.
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


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = overrides.pop("MatrixInstruction", [16, 16, 128, 1, 1, 2, 16, 2, 2])
    workGroup = overrides.pop("WorkGroup", [32, 4, 1])
    problemType = {
        "OperationType": "GEMM",
        "MacDataTypeA": "F8",
        "MacDataTypeB": "F4",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    problemType.update(overrides.pop("ProblemType", {}))
    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": workGroup,
        "WavefrontSize": 32,
        "DepthU": 256,
        "MaxLDS": 327680,
        "KernelLanguage": "Assembly",
        "TDMInst": 3,
        "MXScaleFormat": "InMemorySwizzle",
        "LDSTrInst": True,
        "TDMFuse": 1,
        "TDMSplit": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 2,
        "PrefetchGlobalReadB": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "LdsPadMetadata": 0,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problemType, workGroup, gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_solution_accepts_each_grouping_at_equal_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse)
    assert sol.get("Valid") is True, out


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_solution_accepts_paired_at_divergent_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=1,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out


@pytest.mark.parametrize("fuse", [1, 2])
def test_solution_accepts_stagger(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, StaggerU=32)
    assert sol.get("Valid") is True, out
    assert "requires StaggerU=0" not in out


@pytest.mark.parametrize(
    "fuse, overrides, clause",
    [
        (2, {"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
         "TDMFuse=2 requires an equal decoupled pair"),
        (1, {"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG},
         "TDMFuse=1 splits each of its two descriptor sets by wave parity"),
        (2, {"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG},
         "1/1/2 split is a remainder policy"),
        (2, {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 16, 2, 1], "WorkGroup": [32, 2, 1]},
         "got NumWaves=2"),
        (1, {"ProblemType": _NO_MX_ON_B},
         "TDMFuse=1 requires MX scales on both tensors"),
        (2, {"ProblemType": _NO_MX_ON_B},
         "TDMFuse=2 names MXSA and MXSB as the two single-wave members"),
        (1, {"ProblemType": {"Sparse": 1}},
         "TDMFuse=1 does not describe the sparse metadata tensor"),
        (2, {"ProblemType": {"Sparse": 1}},
         "TDMFuse=2 does not describe the sparse metadata tensor"),
        (1, {"TDMInst": 1}, "TDMA and TDMB must be enabled simultaneously"),
        (1, {"UseSubtileImpl": True}, "Unable to load MXSB scales using one load per wave"),
    ],
)
def test_solution_rejects_outside_the_grouping(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, overrides, clause):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is False
    assert clause in out


@pytest.mark.parametrize("fuse", [0, 1, 2])
@_TDMSPLIT_DISABLED
def test_tdmsplit_across_every_grouping(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, TDMSplit=True)
    if fuse == 0:
        assert sol.get("Valid") is True, out
        return
    assert sol.get("Valid") is False
    assert "TDMFuse=%d is not available with TDMSplit" % fuse in out


@pytest.mark.parametrize("fuse, predicate", [(1, tdmFusePaired), (2, tdmFuseAMx)])
def test_accepted_solution_matches_the_writer_predicate(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, predicate):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse)
    assert sol.get("Valid") is True, out
    assert predicate(sol._state) is True
    if fuse == 1:
        sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=1,
                           PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
        assert sol.get("Valid") is True, out
        assert tdmFusePaired(sol._state) is True
        assert decouplePGRBlocks(sol._state)[1:] == (1, 2)
