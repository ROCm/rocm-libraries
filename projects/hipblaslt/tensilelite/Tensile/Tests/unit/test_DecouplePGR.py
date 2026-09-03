################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Components.DecouplePGR.

Related coverage that is not this module:
  PAP + PrefetchGlobalReadA/B  -> test_PrefetchAcrossPersistent.py
  SIA4 barrier vs wave-parity  -> test_stinkytofu_barrier_wave_parity.py
"""
import copy
import types

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Components.DecouplePGR import (
    _asDataType,
    _ldsAlignedBytes,
    decouplePGRBlocks,
    decouplePGRLdsBytesEstimate,
    decoupledSingleBuffered,
    divergentPairUnsupportedReason,
    equalPairDegeneratesToScalar,
    ldsBlocksForPgrLevel,
    macroTileFromMatrixInstruction,
    pgrAutoPairCandidates,
    pgrAutoPairSelectMaxLds,
    pgrAutoStartLevel,
    pgrSpecialValueRejectReason,
    resolvePrefetchGlobalReadSpecialValues,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers (no Solution / toolchain)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level, blocks", [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4)])
def test_lds_blocks_for_pgr_level(level, blocks):
    assert ldsBlocksForPgrLevel(level) == blocks


@pytest.mark.parametrize(
    "pgr, pgrA, pgrB, expected",
    [
        (2, None, None, (False, 2, 2)),
        (0, 0, 0, (True, 1, 1)),
        (1, 1, 1, (True, 1, 1)),
        (2, 2, 2, (True, 2, 2)),
        (2, 1, 2, (True, 1, 2)),
        (2, 2, 1, (True, 2, 1)),
        (2, 0, 2, (True, 1, 2)),
        (2, 2, 0, (True, 2, 1)),
    ],
)
def test_decouple_pgr_blocks(pgr, pgrA, pgrB, expected):
    ks = {"PrefetchGlobalRead": pgr}
    if pgrA is not None:
        ks["PrefetchGlobalReadA"] = pgrA
    if pgrB is not None:
        ks["PrefetchGlobalReadB"] = pgrB
    assert decouplePGRBlocks(ks) == expected


@pytest.mark.parametrize(
    "pgrA, pgrB, single",
    [(1, 2, True), (2, 1, True), (0, 2, True), (0, 0, False), (1, 1, False), (0, 1, False), (2, 2, False)],
)
def test_decoupled_single_buffered(pgrA, pgrB, single):
    ks = {"PrefetchGlobalRead": max(pgrA, pgrB), "PrefetchGlobalReadA": pgrA, "PrefetchGlobalReadB": pgrB}
    assert decoupledSingleBuffered(ks) is single


def test_legacy_solution_is_not_decoupled():
    assert decoupledSingleBuffered({"PrefetchGlobalRead": 2}) is False


def _divergentSolution(**overrides):
    ks = {
        "PrefetchGlobalRead": 1,
        "PrefetchGlobalReadA": 1,
        "PrefetchGlobalReadB": 2,
        "ScheduleIterAlg": 0,
        "PrefetchLocalRead": 1,
        "NumWaves": 4,
        "DepthU": 512,
        "LocalSplitU": 1,
        "InnerUnroll": 1,
        "MatrixInstK": 128,
        "EnableMatrixInstruction": True,
        "ClusterLocalRead": 1,
        "ForceUnrollSubIter": False,
    }
    ks.update(overrides)
    ks.setdefault("_ScheduleIterAlg", 0 if ks["ScheduleIterAlg"] == 4 else ks["ScheduleIterAlg"])
    return ks


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, None),
        ({"PrefetchGlobalReadB": 3}, "more than two LDS blocks"),
        ({"ScheduleIterAlg": 3}, "ScheduleIterAlg=0"),
        ({"PrefetchLocalRead": 0}, "PrefetchLocalRead must be at least 1"),
        ({"PrefetchLocalRead": 4}, "is not below LoopIters=4"),
        ({"NumWaves": 1}, "NumWaves > 1"),
    ],
)
def test_divergent_pair_unsupported_reason(overrides, expected):
    reason = divergentPairUnsupportedReason(_divergentSolution(**overrides))
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


@pytest.mark.parametrize("scheduleIterAlg, accepted", [(0, True), (1, False), (2, False), (3, False), (4, True)])
def test_divergent_pair_follows_derived_schedule_iter_alg(scheduleIterAlg, accepted):
    reason = divergentPairUnsupportedReason(_divergentSolution(ScheduleIterAlg=scheduleIterAlg))
    if accepted:
        assert reason is None
    else:
        assert reason is not None and "ScheduleIterAlg" in reason


@pytest.mark.parametrize(
    "pgrA, pgrB, oneLdsBuffer, degenerates",
    [
        (2, 2, 0, True), (2, 2, -1, True), (2, 2, None, True), (2, 2, 1, True),
        (0, 0, 0, True), (0, 0, 1, True), (1, 1, 0, True), (1, 1, 1, True),
        (3, 3, 0, True), (1, 2, 0, False), (1, 2, 1, False),
    ],
)
def test_equal_pair_degenerates_to_scalar(pgrA, pgrB, oneLdsBuffer, degenerates):
    ks = {"PrefetchGlobalRead": max(pgrA, pgrB), "PrefetchGlobalReadA": pgrA, "PrefetchGlobalReadB": pgrB}
    if oneLdsBuffer is not None:
        ks["1LDSBuffer"] = oneLdsBuffer
    assert equalPairDegeneratesToScalar(ks) is degenerates


@pytest.mark.parametrize(
    "depthU, prefetchLocalRead, rejected",
    [(512, 1, False), (512, 4, True), (256, 2, True), (128, 1, True)],
)
def test_prefetch_local_read_below_loop_iters(depthU, prefetchLocalRead, rejected):
    reason = divergentPairUnsupportedReason(
        _divergentSolution(DepthU=depthU, PrefetchLocalRead=prefetchLocalRead))
    if rejected:
        assert reason is not None and "LoopIters" in reason
    else:
        assert reason is None


@pytest.mark.parametrize("pgr, expected", [
    (5, [(5, 5), (5, 4), (4, 5), (4, 4), (4, 3), (3, 4), (3, 3), (3, 2), (2, 3), (2, 2), (2, 1), (1, 2)]),
    (2, [(2, 2), (2, 1), (1, 2)]),
    (1, []),
    (0, []),
])
def test_pgr_auto_pair_candidates(pgr, expected):
    assert pgrAutoPairCandidates(pgr) == expected
    assert (0, 0) not in expected
    assert (1, 1) not in expected
    assert (0, 1) not in expected
    assert (1, 0) not in expected


@pytest.mark.parametrize("pgr, start", [(-1, 2), (None, 2), (4, 4), (0, 0), (1, 1)])
def test_pgr_auto_start_level(pgr, start):
    assert pgrAutoStartLevel(pgr) == start


@pytest.mark.parametrize("pgrA, pgrB, clause", [
    (-1, 2, "special value"),
    (-1, 0, "special value"),
    (-1, None, "both be set or both omitted"),
    (0, None, "both be set or both omitted"),
    (1, None, "both be set or both omitted"),
    (2, None, "both be set or both omitted"),
    (None, 2, "both be set or both omitted"),
])
def test_pgr_special_value_reject_reason(pgrA, pgrB, clause):
    reason = pgrSpecialValueRejectReason(pgrA, pgrB)
    assert reason is not None and clause in reason


@pytest.mark.parametrize("pgrA, pgrB", [
    (None, None), (0, 0), (1, 1), (-1, -1), (1, 2), (2, 1), (0, 2), (2, 0), (2, 2),
])
def test_pgr_special_value_accepts_equal_sentinels_and_real_pairs(pgrA, pgrB):
    assert pgrSpecialValueRejectReason(pgrA, pgrB) is None


_F8F4_PROBLEM_TYPE = {
    "MacDataTypeA": DataType("F8"),
    "MacDataTypeB": DataType("F4"),
    "MXBlockA": 32,
    "MXBlockB": 32,
}


def test_macro_tile_from_matrix_instruction():
    assert macroTileFromMatrixInstruction([16, 16, 128, 1, 1, 1, 32, 4, 1]) == (64, 512)


def _autoSelectState(**overrides):
    state = {
        "PrefetchGlobalRead": 2,
        "DepthU": 256,
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 8, 2, 2],
        "MaxLDS": 327680,
        "ProblemType": _F8F4_PROBLEM_TYPE,
    }
    state.update(overrides)
    return state


def test_pgr_auto_select_max_lds_prunes_lds():
    state = _autoSelectState(
        PrefetchGlobalRead=4,
        MatrixInstruction=[16, 16, 128, 1, 1, 1, 32, 4, 1],
        MaxLDS=50000,
    )
    assert pgrAutoPairSelectMaxLds(4, state, _F8F4_PROBLEM_TYPE) is None


def test_pgr_auto_select_max_lds_pair():
    assert pgrAutoPairSelectMaxLds(2, _autoSelectState(), _F8F4_PROBLEM_TYPE) == (2, 2)


def test_pgr_auto_select_max_lds_picks_higher_usage_divergent_pair():
    selected = pgrAutoPairSelectMaxLds(2, _autoSelectState(MaxLDS=90000), _F8F4_PROBLEM_TYPE)
    assert selected == (1, 2)


def test_resolve_auto_picks_max_lds_pair():
    state = _autoSelectState(PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (2, 2)


@pytest.mark.parametrize("pgr", [0, 1])
def test_resolve_auto_below_two_drops_per_tensor_keys(pgr):
    state = {"PrefetchGlobalRead": pgr, "PrefetchGlobalReadA": -1, "PrefetchGlobalReadB": -1}
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert "PrefetchGlobalReadA" not in state
    assert "PrefetchGlobalReadB" not in state
    assert state["PrefetchGlobalRead"] == pgr


@pytest.mark.parametrize("pgrA, pgrB", [(0, 0), (1, 1)])
def test_resolve_leaves_equal_pair_for_scalar_degeneration(pgrA, pgrB):
    state = {"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": pgrA, "PrefetchGlobalReadB": pgrB}
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (pgrA, pgrB)


def test_resolve_scalar_auto_starts_at_two():
    state = _autoSelectState(PrefetchGlobalRead=-1)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (2, 2)


def _problemType(macA, macB, mxA=0, mxB=0):
    return {
        "MacDataTypeA": DataType(macA),
        "MacDataTypeB": DataType(macB),
        "MXBlockA": mxA,
        "MXBlockB": mxB,
    }


def test_pgr_auto_select_uses_element_size_not_f8f4_default():
    mi = [16, 16, 128, 1, 1, 2, 8, 2, 2]
    maxLds = 120000
    f8f4 = _autoSelectState(MaxLDS=maxLds, MatrixInstruction=mi, ProblemType=_problemType("F8", "F4", 32, 32))
    f8f8 = _autoSelectState(MaxLDS=maxLds, MatrixInstruction=mi, ProblemType=_problemType("F8", "F8", 32, 32))
    f16 = _autoSelectState(MaxLDS=maxLds, MatrixInstruction=mi, ProblemType=_problemType("H", "H"))
    assert pgrAutoPairSelectMaxLds(2, f8f4, f8f4["ProblemType"]) == (2, 2)
    assert pgrAutoPairSelectMaxLds(2, f8f8, f8f8["ProblemType"]) == (2, 1)
    assert pgrAutoPairSelectMaxLds(2, f16, f16["ProblemType"]) is None


def test_pgr_auto_select_without_type_does_not_assume_f8f4():
    state = _autoSelectState(MaxLDS=90000, ProblemType={})
    assert pgrAutoPairSelectMaxLds(2, state, {}) == (2, 2)


def _realMacA(dt):
    if dt.isFloat8BFloat8():
        return DataType("F8")
    if dt.isBFloat8Float8():
        return DataType("B8")
    if dt.isFloat8BFloat8_fnuz():
        return DataType("F8N")
    if dt.isBFloat8Float8_fnuz():
        return DataType("B8N")
    return dt


def _realMacB(dt):
    if dt.isFloat8BFloat8():
        return DataType("B8")
    if dt.isBFloat8Float8():
        return DataType("F8")
    if dt.isFloat8BFloat8_fnuz():
        return DataType("B8N")
    if dt.isBFloat8Float8_fnuz():
        return DataType("F8N")
    return dt


def _validGemmAB():
    import ast
    import Tensile.Common.DataType as dataTypeMod
    from pathlib import Path
    src = Path(dataTypeMod.__file__).resolve().parents[1] / "SolutionStructs" / "Problem.py"
    tree = ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_validGEMMTypes":
                    return [(a, b) for a, b, _to, _tc in ast.literal_eval(node.value)]
    raise RuntimeError("could not find _validGEMMTypes in Problem.py")


def _calcLdsNumBytesAB(mac, bpe, depthU, macroTile):
    align = 64 if mac.is6bitFloat() else int(64 / mac.numRegisters())
    raw = int(depthU * macroTile * bpe)
    return (raw + align - 1) // align * align


def _usesMx(mac):
    return mac.isFloat4() or mac.is6bitFloat() or mac.is8bitFloat()


def test_as_data_type_accepts_every_problem_name_form():
    for props in DataType.properties:
        dt = DataType(props["char"])
        expected = dt.numBytes()
        assert _asDataType(props["char"]).numBytes() == expected
        assert _asDataType(props["char"].lower()).numBytes() == expected
        assert _asDataType(dt).numBytes() == expected
        assert _asDataType(dt.value).numBytes() == expected
        assert _asDataType(props["enum"]).numBytes() == expected
    assert _asDataType(None) is None
    assert _asDataType("not-a-type") is None


def test_dryrun_all_problem_gemm_types():
    depthU, mt0, mt1 = 256, 64, 256
    pairs = list(_validGemmAB())
    for props in DataType.properties:
        pairs.append((props["char"], props["char"]))
    unique = list(dict.fromkeys(pairs))

    failures = []
    for charA, charB in unique:
        macA = _realMacA(DataType(charA))
        macB = _realMacB(DataType(charB))
        mxCases = [(0, 0)]
        if _usesMx(macA) or _usesMx(macB):
            mxCases.append((32 if _usesMx(macA) else 0, 32 if _usesMx(macB) else 0))
        for mxA, mxB in mxCases:
            for convert in (False, True):
                pt = {
                    "MacDataTypeA": macA, "MacDataTypeB": macB,
                    "DataTypeA": macA, "DataTypeB": macB,
                    "MXBlockA": mxA, "MXBlockB": mxB,
                }
                ks = {
                    "DepthU": depthU, "MacroTile0": mt0, "MacroTile1": mt1,
                    "PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 2,
                    "ConvertAfterDS": convert, "ProblemType": pt,
                }
                label = "%s/%s mx=%s/%s convert=%s" % (charA, charB, mxA, mxB, convert)
                try:
                    got = (
                        _ldsAlignedBytes(ks, pt, "A", depthU, mt0),
                        _ldsAlignedBytes(ks, pt, "B", depthU, mt1),
                        _ldsAlignedBytes(ks, pt, "MXSA", depthU, mt0),
                        _ldsAlignedBytes(ks, pt, "MXSB", depthU, mt1),
                    )
                    exp = (
                        _calcLdsNumBytesAB(macA, macA.numBytes(), depthU, mt0),
                        _calcLdsNumBytesAB(macB, macB.numBytes(), depthU, mt1),
                        _calcLdsNumBytesAB(macA, 1, depthU // mxA, mt0) if mxA else 0,
                        _calcLdsNumBytesAB(macB, 1, depthU // mxB, mt1) if mxB else 0,
                    )
                    if got != exp:
                        failures.append("%s: got %s expected %s" % (label, got, exp))
                        continue
                    est = decouplePGRLdsBytesEstimate(ks, pt)
                    if est is None or est <= 0:
                        failures.append("%s: estimate %s" % (label, est))
                        continue
                    pgrAutoPairSelectMaxLds(2, dict(ks, MaxLDS=1 << 30), pt)
                except Exception as exc:
                    failures.append("%s: %s" % (label, exc))
    assert not failures, "%d failures:\n%s" % (len(failures), "\n".join(failures[:25]))


def test_convert_after_ds_uses_data_type_not_mac():
    macA, macB = DataType("F8"), DataType("F4")
    dataA, dataB = DataType("H"), DataType("H")
    pt = {
        "MacDataTypeA": macA, "MacDataTypeB": macB,
        "DataTypeA": dataA, "DataTypeB": dataB,
        "MXBlockA": 0, "MXBlockB": 0,
    }
    ks = {
        "DepthU": 256, "MacroTile0": 64, "MacroTile1": 256, "ConvertAfterDS": True,
        "PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 2,
        "ProblemType": pt,
    }
    gotA = _ldsAlignedBytes(ks, pt, "A", 256, 64)
    gotB = _ldsAlignedBytes(ks, pt, "B", 256, 256)
    expA = _calcLdsNumBytesAB(macA, dataA.numBytes(), 256, 64)
    expB = _calcLdsNumBytesAB(macB, dataB.numBytes(), 256, 256)
    assert (gotA, gotB) == (expA, expB)
    assert gotA != _calcLdsNumBytesAB(macA, macA.numBytes(), 256, 64)


# ---------------------------------------------------------------------------
# Solution wiring. Needs amdclang++ gfx1250; skipped otherwise.
# Helper tests above already pin the reject *reasons*; these check Solution
# actually applies them. PAP belongs in test_PrefetchAcrossPersistent.py.
# ---------------------------------------------------------------------------
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


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
        "TDMFuse": 0,
        "TDMSplit": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 1,
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


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_solution_accepts_divergent_pairs(_gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out


@pytest.mark.parametrize(
    "overrides, clause",
    [
        ({"ClusterDim": [2, 1]}, "ClusterDim != [1, 1] is incompatible with divergent"),
        ({"ProblemType": {"Sparse": 1}}, "Sparse is not supported yet"),
        ({"1LDSBuffer": 1}, "1LDSBuffer=1 gives every tensor one shared LDS block"),
        ({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 0, "PrefetchGlobalReadB": 1},
         "leave both tensors on one LDS block"),
        ({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 0},
         "leave both tensors on one LDS block"),
        ({"ScheduleIterAlg": 3}, "only ScheduleIterAlg=0 places the fill where it can be moved"),
        ({"PrefetchGlobalReadA": -1, "PrefetchGlobalReadB": 2}, "special value"),
    ],
)
def test_solution_rejects_unsupported_decoupled_pgr(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, overrides, clause):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False
    assert clause in out


def test_solution_cluster_allows_equal_pgr(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, ClusterDim=[2, 1],
                       PrefetchGlobalReadA=2, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is True, out


def test_solution_equal_one_degenerates_to_scalar(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchGlobalRead=1,
                       PrefetchGlobalReadA=1, PrefetchGlobalReadB=1)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 1
    assert sol.get("PrefetchGlobalReadA") is None
    assert sol.get("PrefetchGlobalReadB") is None
    assert "equal pair" in out
    assert "may overwrite LDS data still being read" in out


def test_solution_auto_equal_pair_degenerates_to_scalar(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalRead=2, PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 2
    assert sol.get("PrefetchGlobalReadA") is None
    assert sol.get("PrefetchGlobalReadB") is None


def test_solution_degenerate_zero_falls_back_to_scalar(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalRead=2, PrefetchGlobalReadA=0, PrefetchGlobalReadB=0)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 0
    assert sol.get("PrefetchGlobalReadA") is None
    assert sol.get("PrefetchGlobalReadB") is None
    assert "equal pair" in out


# ---------------------------------------------------------------------------
# Divergent thick-wait post-pass (KernelWriter._dcpApplyThickWait1).
# A text pass over already-emitted assembly, so it needs no toolchain: the
# writer is stubbed down to the two things the pass touches.
# ---------------------------------------------------------------------------
class _ThickWaitWriter:
    def __init__(self, memTokenLdsDcp=None):
        self.states = types.SimpleNamespace()
        if memTokenLdsDcp is not None:
            self.states.memTokenLdsDcp = memTokenLdsDcp

    def _dcpDivergent(self, kernel):
        from Tensile.KernelWriterAssembly import KernelWriterAssembly

        return KernelWriterAssembly._dcpDivergent(self, kernel)


def _applyThickWait(kernel, asm, memTokenLdsDcp=None):
    from Tensile.KernelWriter import KernelWriter

    writer = _ThickWaitWriter(memTokenLdsDcp)
    return KernelWriter._dcpApplyThickWait1(writer, kernel, asm)


def _thickWaitKernel(pgrA=1, pgrB=2, **overrides):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "TDMFuse": 0,
        "InitCIterWmma": 0,
    }
    ks.update(overrides)
    return ks


def _fillBlock(tc, body=0, wait="s_wait_tensorcnt 0", clone=False):
    """One emitted fill header: its label, `body` filler lines, then its wait.

    `clone` names it the way the InitCIterWmma region clone does.
    """
    name = "label_InitCIterWmma_label_DcpEarlyFill%sEnd_0" % tc if clone \
        else "label_DcpEarlyFill%sEnd" % tc
    lines = ["%s:" % name]
    lines += ["  tensor_load_to_lds %d" % i for i in range(body)]
    if wait is not None:
        lines.append(wait)
    return lines


def _asm(*blocks, **kwargs):
    tail = kwargs.pop("tail", ["label_DcpLateFillAEnd:", "s_endpgm"])
    assert not kwargs, kwargs
    lines = ["label_LoopBeginL:"]
    for block in blocks:
        lines += block
    lines += tail
    return "".join(line + "\n" for line in lines)


def test_thick_wait_retags_the_body_and_the_iter0_clone():
    asm = _asm(_fillBlock("B", clone=True), _fillBlock("B"))
    out = _applyThickWait(_thickWaitKernel(InitCIterWmma=1), asm)
    assert out.count("s_wait_tensorcnt 2") == 2
    assert "s_wait_tensorcnt 0" not in out


def test_thick_wait_accepts_one_retag_when_iter0_is_not_cloned():
    """InitCIterWmma=0 emits no iter0 clone, so there is one wait to retag."""
    asm = _asm(_fillBlock("B"))
    out = _applyThickWait(_thickWaitKernel(), asm)
    assert out.count("s_wait_tensorcnt 2") == 1


def test_thick_wait_ignores_header_copies_that_carry_no_tensorcnt_wait():
    """A loop copy can emit a thick header with no wait in its block, so the
    header labels are not a count of the waits to retag."""
    asm = _asm(_fillBlock("B", clone=True), _fillBlock("B"),
               _fillBlock("B", wait=None))
    out = _applyThickWait(_thickWaitKernel(InitCIterWmma=1), asm)
    assert out.count("s_wait_tensorcnt 2") == 2


def test_thick_wait_finds_the_wait_in_a_long_fill_block():
    asm = _asm(_fillBlock("B", body=400, clone=True), _fillBlock("B", body=400))
    out = _applyThickWait(_thickWaitKernel(InitCIterWmma=1), asm)
    assert out.count("s_wait_tensorcnt 2") == 2


@pytest.mark.parametrize("pgrA, pgrB, thick, thin", [(1, 2, "B", "A"), (2, 1, "A", "B")])
def test_thick_wait_only_retags_the_double_buffered_tensor(pgrA, pgrB, thick, thin):
    asm = _asm(_fillBlock(thick), _fillBlock(thin))
    out = _applyThickWait(_thickWaitKernel(pgrA, pgrB), asm)
    assert out.count("s_wait_tensorcnt 2") == 1
    assert out.count("s_wait_tensorcnt 0") == 1


def test_thick_wait_leaves_equal_pairs_untouched():
    asm = _asm(_fillBlock("B"))
    assert _applyThickWait(_thickWaitKernel(2, 2), asm) == asm


def test_thick_wait_does_not_claim_a_wait_past_the_next_fill_label():
    asm = _asm(_fillBlock("B", wait=None), tail=["label_DcpLateFillAEnd:",
                                                 "s_wait_tensorcnt 0", "s_endpgm"])
    with pytest.raises(RuntimeError, match="found 0"):
        _applyThickWait(_thickWaitKernel(), asm)


def test_thick_wait_shortfall_drops_one_kernel_instead_of_the_build():
    asm = _asm(_fillBlock("B", clone=True), _fillBlock("B", wait=None))
    with pytest.raises(
            RuntimeError,
            match=r"expected 2 s_wait_tensorcnt 2 on thick B \(1/2 LDS blocks\), found 1"):
        _applyThickWait(_thickWaitKernel(InitCIterWmma=1), asm)


def test_thick_wait_paired_retags_both_cloned_headers():
    asm = _asm(_fillBlock("B", clone=True), _fillBlock("B"))
    out = _applyThickWait(_thickWaitKernel(TDMFuse=1, InitCIterWmma=1), asm,
                          memTokenLdsDcp={"A": (0, 1), "B": (2, 3)})
    assert out.count("s_wait_tensorcnt 1") == 2


def test_thick_wait_paired_shortfall_rejects_rather_than_aborts():
    asm = _asm(_fillBlock("B", clone=True), _fillBlock("B", wait=None))
    with pytest.raises(RuntimeError, match="TDMFuse=1 cannot honour"):
        _applyThickWait(_thickWaitKernel(TDMFuse=1, InitCIterWmma=1), asm,
                        memTokenLdsDcp={"A": (0, 1), "B": (2, 3)})


def test_thick_wait_paired_without_lds_tokens_is_a_no_op():
    asm = _asm(_fillBlock("B"))
    assert _applyThickWait(_thickWaitKernel(TDMFuse=1), asm) == asm


def test_thick_wait_paired_finds_the_wait_in_a_long_fill_block():
    asm = _asm(_fillBlock("B", body=400, clone=True), _fillBlock("B", body=400))
    out = _applyThickWait(_thickWaitKernel(TDMFuse=1, InitCIterWmma=1), asm,
                          memTokenLdsDcp={"A": (0, 1), "B": (2, 3)})
    assert out.count("s_wait_tensorcnt 1") == 2
