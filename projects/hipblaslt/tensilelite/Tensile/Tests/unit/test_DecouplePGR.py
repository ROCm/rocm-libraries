################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Components.DecouplePGR."""
import copy
import itertools
import re
import types

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components import DecouplePGR as DP
from Tensile.Components import TDMFuse as TF
from Tensile.Components.DecouplePGR import (
    _asDataType,
    _ldsAlignedBytes,
    _macroTileFromState,
    DCP_MAX_LDS_BLOCKS_DIVERGENT,
    DCP_THICK_GATE_SUPPORTED,
    DCP_THICK_GATE_TEXT,
    DCP_THICK_GATE_TOKENS,
    dcpThickGateFromTokenPasses,
    decoupledSingleBuffered,
    decoupledThickGateRelaxation,
    decouplePGRBlocks,
    decouplePGRLdsBytesEstimate,
    divergentPairUnsupportedReason,
    equalPairDegeneratesToScalar,
    ldsBlocksForPgrLevel,
    macroTileFromMatrixInstruction,
    PGR_SPECIAL_AUTO,
    pgrAutoPairCandidates,
    pgrAutoPairRanking,
    pgrAutoPairRequested,
    pgrSpecialValueRejectReason,
    resolvePrefetchGlobalReadSpecialValues,
    tdmWaveIssueOrder,
)
from Tensile.Components.TDMFuse import TDM_FUSE_GROUPING, TDM_GROUPS, tdmGrouping, tdmSeparateABDescriptors

pytestmark = pytest.mark.unit


def pgrAutoPairSelectMaxLds(pgr, state, problemType=None, fixedA=None, fixedB=None):
    """Head of the ranking. Production reads the whole ranking; only tests want one."""
    ranking = pgrAutoPairRanking(pgr, state, problemType, fixedA=fixedA, fixedB=fixedB)
    return ranking[0] if ranking else None


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
        ({"PrefetchGlobalReadB": 3}, "divergent pairs support at most two LDS blocks per tensor"),
        ({"ScheduleIterAlg": 3}, "ScheduleIterAlg=0"),
        ({"PrefetchLocalRead": 0}, "PrefetchLocalRead=0 leaves no late-fill slot"),
        ({"PrefetchLocalRead": 4}, "is a multiple of LoopIters=4"),
        ({"PrefetchLocalRead": 8}, "is a multiple of LoopIters=4"),
        # KernelWriter asserts on numItersPLR unconditionally, so neither of
        # these user-settable keys may narrow the rejection.
        ({"PrefetchLocalRead": 4, "ClusterLocalRead": 0}, "is a multiple of LoopIters=4"),
        ({"PrefetchLocalRead": 4, "ForceUnrollSubIter": True},
         "is a multiple of LoopIters=4"),
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
    (5, [(5, 5), (4, 4), (3, 3), (2, 2), (2, 1), (1, 2)]),
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


@pytest.mark.parametrize("pgrA, pgrB, clause", [
    (-1, None, "both be set or both omitted"),
    (0, None, "both be set or both omitted"),
    (1, None, "both be set or both omitted"),
    (2, None, "both be set or both omitted"),
    (None, 2, "both be set or both omitted"),
    (None, -1, "both be set or both omitted"),
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


@pytest.mark.parametrize("mi, expected", [
    # MatrixInstB == 1: MIBlockBM is 1 and MIWaveGroup is (mi[7], mi[8]), which is
    # the only case the old mi[0]*mi[5]*mi[7] / mi[1]*mi[6]*mi[8] shortcut got right.
    ([16, 16, 128, 1, 1, 1, 32, 4, 1], (64, 512)),
    ([16, 16, 128, 1, 1, 2, 8, 2, 2], (64, 256)),
    # MatrixInstB > 1: the blocks are distributed into MIBlockBM first and
    # MIWaveGroup follows, so the shortcut under-reported MacroTile1. These are
    # the shapes it got wrong.
    ([32, 32, 1, 2, 1, 4, 1, 2, 2], (256, 128)),
    ([16, 16, 32, 4, 1, 2, 2, 2, 2], (64, 256)),
    ([16, 16, 64, 2, 1, 4, 2, 2, 2], (128, 128)),
])
def test_macro_tile_from_matrix_instruction(mi, expected):
    assert macroTileFromMatrixInstruction(mi, 32) == expected


def test_macro_tile_from_matrix_instruction_needs_a_wavefront_size():
    assert macroTileFromMatrixInstruction([16, 16, 128, 1, 1, 1, 32, 4, 1], None) is None


def _autoSelectState(**overrides):
    state = {
        "PrefetchGlobalRead": 2,
        "DepthU": 256,
        # WavefrontSize is required: MacroTile is derived from the MI geometry,
        # and the wavefront size enters that derivation. Auto-selection rejects
        # rather than guessing when it is missing.
        "WavefrontSize": 32,
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


@pytest.mark.parametrize("pgrA, pgrB", [(-1, -1), (-1, 2), (2, -1)])
def test_resolve_auto_rejects_auto_depthu(pgrA, pgrB):
    state = _autoSelectState(DepthU=-1, PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    reason = resolvePrefetchGlobalReadSpecialValues(state)
    assert reason is not None and "needs a concrete DepthU" in reason
    # Left unresolved, so no later rule reads a level the LDS search never ranked.
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (pgrA, pgrB)


@pytest.mark.parametrize("pgr", [0, 1])
def test_resolve_auto_below_two_degenerates_under_auto_depthu(pgr):
    state = _autoSelectState(DepthU=-1, PrefetchGlobalRead=pgr,
                             PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert "PrefetchGlobalReadA" not in state
    assert "PrefetchGlobalReadB" not in state
    assert state["PrefetchGlobalRead"] == pgr


@pytest.mark.parametrize("pgrA, pgrB", [(0, 0), (1, 1)])
def test_resolve_leaves_equal_pair_for_scalar_degeneration(pgrA, pgrB):
    state = {"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": pgrA, "PrefetchGlobalReadB": pgrB}
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (pgrA, pgrB)


def test_resolve_bare_scalar_auto_is_not_a_pair_request():
    state = _autoSelectState(PrefetchGlobalRead=-1)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert "PrefetchGlobalReadA" not in state
    assert "PrefetchGlobalReadB" not in state
    assert decouplePGRBlocks(state) == (False, 1, 1)


def test_scalar_prefetch_global_read_does_not_accept_auto():
    assert PGR_SPECIAL_AUTO not in validParameters["PrefetchGlobalRead"]
    assert PGR_SPECIAL_AUTO in validParameters["PrefetchGlobalReadA"]
    assert PGR_SPECIAL_AUTO in validParameters["PrefetchGlobalReadB"]


# ---------------------------------------------------------------------------
# One-sided auto: -1 vs a pinned depth.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixedA, fixedB, expected", [
    (None, 1, (2, 1)),
    (None, 2, (2, 2)),
    (1, None, (1, 2)),
    (2, None, (2, 2)),
])
def test_pgr_auto_select_honours_a_pinned_side(fixedA, fixedB, expected):
    selected = pgrAutoPairSelectMaxLds(2, _autoSelectState(), _F8F4_PROBLEM_TYPE,
                                       fixedA=fixedA, fixedB=fixedB)
    assert selected == expected


def test_pgr_auto_select_pinned_side_still_ranks_by_lds():
    assert pgrAutoPairSelectMaxLds(2, _autoSelectState(), _F8F4_PROBLEM_TYPE,
                                   fixedB=2) == (2, 2)
    assert pgrAutoPairSelectMaxLds(2, _autoSelectState(MaxLDS=90000), _F8F4_PROBLEM_TYPE,
                                   fixedB=2) == (1, 2)


def test_pgr_auto_select_pinned_side_with_no_candidate_is_none():
    assert pgrAutoPairSelectMaxLds(2, _autoSelectState(), _F8F4_PROBLEM_TYPE,
                                   fixedB=0) is None


@pytest.mark.parametrize("pgrA, pgrB, expected", [
    (-1, 1, (2, 1)),
    (1, -1, (1, 2)),
    (-1, 2, (2, 2)),
    (2, -1, (2, 2)),
])
def test_resolve_one_sided_auto_searches_the_minus_one_side(pgrA, pgrB, expected):
    state = _autoSelectState(PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == expected


def test_resolve_one_sided_auto_raises_the_ceiling_to_the_pinned_depth():
    state = _autoSelectState(PrefetchGlobalRead=0, PrefetchGlobalReadA=-1,
                             PrefetchGlobalReadB=2)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (2, 2)


@pytest.mark.parametrize("pgrA, pgrB, heldTc", [(-1, 1, "B"), (1, -1, "A")])
def test_resolve_one_sided_auto_rejects_when_nothing_fits(pgrA, pgrB, heldTc):
    state = _autoSelectState(MaxLDS=1024, PrefetchGlobalReadA=pgrA,
                             PrefetchGlobalReadB=pgrB)
    reason = resolvePrefetchGlobalReadSpecialValues(state)
    assert reason is not None
    assert "no LDS-feasible pair" in reason
    assert "PrefetchGlobalRead%s held at" % heldTc in reason


@pytest.mark.parametrize("pgrA, pgrB, expected", [(-1, 1, (True, 2, 1)), (1, -1, (True, 1, 2))])
def test_one_sided_auto_reaches_decouple_pgr_blocks(pgrA, pgrB, expected):
    state = _autoSelectState(PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert decouplePGRBlocks(state) == expected


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


def test_pgr_auto_select_without_type_rejects_rather_than_guessing():
    state = _autoSelectState(MaxLDS=90000, ProblemType={})
    assert pgrAutoPairSelectMaxLds(2, state, {}) is None


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
# Solution wiring (gfx1250). PAP coverage is in test_PrefetchAcrossPersistent.py.
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
        "ScheduleIterAlg": 4,
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


def _emitDerived(sol, assembler):
    """(errs, asm) for a derived solution. Overflow arrives as err, not an exception."""
    import shutil

    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (generateKernelObjectsFromSolutions,
                                                  processKernelSource)
    from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    with preserve_rocisa_kernel_state():
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        errs, pieces = [], []
        for kernel in generateKernelObjectsFromSolutions([sol]):
            ri = rocisa.rocIsa.getInstance()
            ri.init(tuple(kernel["ISA"]),
                    shutil.which("amdclang++") or "/usr/bin/amdclang++")
            ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
            kernel.duplicate = False
            kernel["BaseName"] = getKernelFileBase(False, kernel)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(),
                                      False, kernel)
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            pieces.append(src or "")
            errs.append(res.err)
    return errs, "\n".join(pieces)


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_solution_accepts_divergent_pairs(_gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out


# Auto plus every real depth up to DCP_MAX_LDS_BLOCKS_DIVERGENT.
_PGR_PAIR_SPACE = list(itertools.product(
    [PGR_SPECIAL_AUTO] + list(range(DCP_MAX_LDS_BLOCKS_DIVERGENT + 1)), repeat=2))


@pytest.mark.parametrize("pgrA, pgrB", _PGR_PAIR_SPACE)
def test_derived_valid_pair_also_emits(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                       pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    if sol.get("Valid") is not True:
        assert out.strip(), "(%s, %s) was refused with no reason" % (pgrA, pgrB)
        pytest.skip("(%s, %s) refused at derivation: %s"
                    % (pgrA, pgrB, out.strip().splitlines()[-1]))
    errs, asm = _emitDerived(sol, assembler)
    warnings = [line for line in capsys.readouterr().out.splitlines()
                if "WARNING" in line]
    gate = decoupledThickGateRelaxation(sol)
    assert errs == [0] and asm, (
        "(%s, %s) derives Valid but emits err=%s with %d bytes of source: resolved "
        "pair (%s, %s), thick gate %s, _ScheduleIterAlg=%s, _StinkyTofuOptLevel=%s. %s"
        % (pgrA, pgrB, errs, len(asm), sol.get("PrefetchGlobalReadA"),
           sol.get("PrefetchGlobalReadB"),
           None if gate is None else gate.mechanism,
           sol.get("_ScheduleIterAlg"), sol.get("_StinkyTofuOptLevel"),
           warnings[-1] if warnings else "(no warning captured)"))


# A divergent pair on StreamK needs the tail LDS normalization that
# tdmResetTailLdsBuffer refuses on two block strides, so derivation refuses first.
@pytest.mark.parametrize("tdmFuse", [0, 1])
@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_streamk_divergent_pair_never_derives_valid_without_emitting(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, tdmFuse, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=tdmFuse,
                       StreamK=3, StreamKForceDPOnly=1, GlobalSplitU=0,
                       PrefetchGlobalRead=max(pgrA, pgrB),
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    if sol.get("Valid") is not True:
        assert "StreamK tail cannot normalize LDS to buffer 0" in out, out
        return
    errs, asm = _emitDerived(sol, assembler)
    assert errs == [0] and asm, (
        "StreamK + divergent (%d, %d) derived Valid and then emitted err=%s with "
        "%d bytes of source" % (pgrA, pgrB, errs, len(asm)))


# ---------------------------------------------------------------------------
# Text-gate count = descriptor sets one fill issues (1 scale-less, 2 with MX).
# ---------------------------------------------------------------------------
_SCALE_LESS_PROBLEM = {"MacDataTypeA": "B", "MacDataTypeB": "B", "DataType": "B",
                       "DestDataType": "B", "ComputeDataType": "s",
                       "MXBlockA": 0, "MXBlockB": 0}


def _scaleLessDivergent(pgrA, pgrB):
    return dict(ProblemType=dict(_SCALE_LESS_PROBLEM),
                MatrixInstruction=[16, 16, 32, 1, 1, 2, 2, 2, 2],
                DepthU=256, MaxLDS=327680, LDSTrInst=False, TDMFuse=0,
                PrefetchGlobalRead=max(pgrA, pgrB),
                PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)


def _thickGates(asm, thick):
    """First s_wait_tensorcnt anywhere below each thick fill label, as production reads it."""
    lines = asm.splitlines(keepends=True)
    gates = []
    for i, line in enumerate(lines):
        if ("DcpEarlyFill%s" % thick) not in line or not line.rstrip().endswith(":"):
            continue
        gates.append(next((int(m.group(1)) for m in
                           (DP.DCP_TENSORCNT_RE.match(c) for c in lines[i + 1:]) if m),
                          None))
    return gates


@pytest.mark.parametrize("initCIterWmma", [0, 1])
@pytest.mark.parametrize("pgrA, pgrB, thick", [(1, 2, "B"), (2, 1, "A")])
def test_scale_less_divergent_pair_gates_on_its_single_fill(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, thick, initCIterWmma):
    sol, out = _derive(gfx1250_iim, assembler, capsys, InitCIterWmma=initCIterWmma,
                       **_scaleLessDivergent(pgrA, pgrB))
    assert sol.get("Valid") is True, out
    gate = decoupledThickGateRelaxation(sol)
    assert (gate.mechanism, gate.tensorcnt) == (DCP_THICK_GATE_TEXT, 1)

    errs, asm = _emitDerived(sol, assembler)
    assert errs == [0] and asm
    prologue = asm.split("s_wait_tensorcnt", 1)[0]
    assert prologue.count("tensor_load_to_lds") == 1, (
        "the pre-loop fill issues %d tensor ops, so 1 is the wrong allowance"
        % prologue.count("tensor_load_to_lds"))
    gates = _thickGates(asm, thick)
    assert gates and set(gates) == {1}, gates
    assert "s_wait_tensorcnt 2" not in asm


@pytest.mark.parametrize("initCIterWmma", [0, 1])
@pytest.mark.parametrize("pgrA, pgrB, thick", [(1, 2, "B"), (2, 1, "A")])
def test_mx_divergent_pair_still_gates_on_two(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, thick, initCIterWmma):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0,
                       PrefetchGlobalRead=2, InitCIterWmma=initCIterWmma,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    gate = decoupledThickGateRelaxation(sol)
    assert (gate.mechanism, gate.tensorcnt) == (DCP_THICK_GATE_TEXT, 2)

    errs, asm = _emitDerived(sol, assembler)
    assert errs == [0] and asm
    prologue = asm.split("s_wait_tensorcnt", 1)[0]
    assert prologue.count("tensor_load_to_lds") == 2
    gates = _thickGates(asm, thick)
    assert gates and set(gates) == {2}, gates


def test_solution_rejects_a_plr_that_leaves_no_late_sub_iteration(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """KernelWriter asserts numItersPLR unconditionally; ClusterLocalRead must not skip this reject."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, DepthU=128,
                       ClusterLocalRead=0, PrefetchGlobalRead=2,
                       PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is False, out
    assert "is a multiple of LoopIters=1" in out, out


@pytest.mark.parametrize(
    "overrides, clause",
    [
        ({"ClusterDim": [2, 1]}, "ClusterDim != [1, 1] is incompatible with divergent"),
        ({"ProblemType": {"Sparse": 1}}, "Sparse is not supported"),
        ({"1LDSBuffer": 1}, "1LDSBuffer=1 allows one LDS block per tensor"),
        ({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 0, "PrefetchGlobalReadB": 1},
         "leave both tensors on one LDS block"),
        ({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 0},
         "leave both tensors on one LDS block"),
        ({"ScheduleIterAlg": 3},
         "leaves no complete fill group to re-slot"),
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
# Scalar vs PGRA/PGRB through assignDerivedParameters.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pgrA, pgrB, expected", [(-1, 1, (2, 1)), (1, -1, (1, 2))])
def test_solution_one_sided_auto_searches_the_constrained_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, expected):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    assert (sol.get("PrefetchGlobalReadA"), sol.get("PrefetchGlobalReadB")) == expected


@pytest.mark.parametrize("pgrA, pgrB", [(-1, 2), (2, -1)])
def test_solution_one_sided_auto_can_pick_the_equal_pair_and_degenerate(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 2
    assert sol.get("PrefetchGlobalReadA") is None
    assert sol.get("PrefetchGlobalReadB") is None


def test_solution_both_auto_picks_max_lds_pair(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 2
    assert (sol.get("PrefetchGlobalReadA"), sol.get("PrefetchGlobalReadB")) == (None, None)


@pytest.mark.parametrize("pgrA, pgrB, expected", [(1, 2, (1, 2)), (2, 1, (2, 1))])
def test_solution_per_tensor_values_win_over_the_legacy_scalar(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, expected):
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchGlobalRead=2,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    assert (sol.get("PrefetchGlobalReadA"), sol.get("PrefetchGlobalReadB")) == expected
    assert decouplePGRBlocks(sol) == (True, expected[0], expected[1])


# ---------------------------------------------------------------------------
# _dcpRelaxThickTextGate on hand-written assembly (no toolchain).
# ---------------------------------------------------------------------------
class _ThickWaitWriter:
    def __init__(self):
        self.states = types.SimpleNamespace(overflowedResources=0)


def _applyThickWait(kernel, asm):
    from Tensile.KernelWriter import KernelWriter

    writer = _ThickWaitWriter()
    return KernelWriter._dcpRelaxThickTextGate(writer, kernel, asm)


DCP_THICK_WAIT_UNCOVERED = 11


def _thickWaitRejects(kernel, asm):
    """The overflowedResources code the pass set, or 0 if it accepted."""
    from Tensile.KernelWriter import KernelWriter

    writer = _ThickWaitWriter()
    KernelWriter._dcpRelaxThickTextGate(writer, kernel, asm)
    return writer.states.overflowedResources


def _thickWaitKernel(pgrA=1, pgrB=2, **overrides):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "TDMFuse": 0,
        "InitCIterWmma": 0,
        # Keys so TDMFuse=1 actually resolves to paired, not the default grouping.
        "TDMInst": 3,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "NumWaves": 4,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    ks.update(overrides)
    return ks


def _fillBlock(tc, body=0, wait="s_wait_tensorcnt 0", clone=False, paired=False):
    """One fill block: unpaired `[label, body..., wait]`; paired `[body..., label]` (no wait)."""
    name = "label_InitCIterWmma_label_DcpEarlyFill%sEnd_0" % tc if clone \
        else "label_DcpEarlyFill%sEnd" % tc
    fill = ["  tensor_load_to_lds %d" % i for i in range(body)]
    if paired:
        assert wait == "s_wait_tensorcnt 0", \
            "the paired layout carries no wait inside the fill block"
        return fill + ["%s:" % name]
    lines = ["%s:" % name] + fill
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
    asm = _asm(_fillBlock("B"))
    out = _applyThickWait(_thickWaitKernel(), asm)
    assert out.count("s_wait_tensorcnt 2") == 1


def test_thick_wait_ignores_header_copies_that_carry_no_tensorcnt_wait():
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
    assert _applyThickWait(_thickWaitKernel(), asm) == asm
    assert _thickWaitRejects(_thickWaitKernel(), asm) == 0


def test_thick_wait_accepts_a_schedule_that_merges_the_two_gates():
    asm = _asm(_fillBlock("B", clone=True, wait=None), _fillBlock("B"))
    out = _applyThickWait(_thickWaitKernel(InitCIterWmma=1), asm)
    assert out.count("s_wait_tensorcnt 2") == 1
    assert "s_wait_tensorcnt 0" not in out


def test_thick_wait_shortfall_drops_one_kernel_instead_of_the_build():
    asm = _asm(["label_DcpEarlyFillBEnd:",
                "ds_load_b128 v[0:3], v[64] offset:128",
                "s_wait_tensorcnt 0"], tail=["s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(), asm) == DCP_THICK_WAIT_UNCOVERED


def test_thick_wait_refuses_to_walk_past_the_thick_gate_to_the_thin_drain():
    asm = _asm(_fillBlock("B", wait="s_wait_tensorcnt 3"),
               tail=["s_wait_tensorcnt 0", "s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(), asm) == DCP_THICK_WAIT_UNCOVERED


@pytest.mark.parametrize("emitted", [1, 2], ids=["stronger", "exactly-relaxed"])
def test_thick_wait_accepts_a_gate_already_strong_enough(emitted):
    asm = _asm(_fillBlock("B", wait="s_wait_tensorcnt %d" % emitted),
               tail=["s_wait_tensorcnt 0", "s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(), asm) == 0


# ---------------------------------------------------------------------------
# Paired arm: text pass is a no-op. Pin against `[body..., label]`.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("asmArgs, asmKwargs", [
    # paired layout: [body..., label]
    (((("B", 8, True),),), {}),
    # across the InitCIterWmma region clone and a long fill body
    (((("B", 400, True, True), ("B", 400, True)),), {}),
    # a gate the wait-count insertion pass already emitted as 1, and a wider 2
    (((("B", 4, True),),), {"tail": ["s_wait_tensorcnt 1", "s_endpgm"]}),
    (((("B", 4, True),),), {"tail": ["s_wait_tensorcnt 2", "s_endpgm"]}),
    # a thick fill that emitted no label at all
    (((("A", 0, True),),), {"tail": ["s_endpgm"]}),
], ids=["production", "clone+long", "gate1", "gate2", "unlabelled"])
def test_thick_wait_paired_is_a_no_op(asmArgs, asmKwargs):
    blocks = [_fillBlock(tc, body=body, paired=paired,
                         clone=(rest[0] if rest else False))
              for (tc, body, paired, *rest) in asmArgs[0]]
    asm = _asm(*blocks, **asmKwargs)
    assert _applyThickWait(_thickWaitKernel(TDMFuse=1), asm) == asm


def test_thick_wait_paired_leaves_a_downstream_zero_wait_alone():
    asm = _asm(_fillBlock("B", body=8, paired=True),
               tail=["s_wait_tensorcnt 0", "label_DcpLateFillAEnd:", "s_endpgm"])
    out = _applyThickWait(_thickWaitKernel(TDMFuse=1), asm)
    assert out == asm
    assert "s_wait_tensorcnt 1" not in out
    assert out.count("s_wait_tensorcnt 0") == 1


@pytest.mark.parametrize("pgrA, pgrB, thick", [(1, 2, "B"), (2, 1, "A")])
def test_thick_wait_target_follows_the_label_the_fill_emitted(pgrA, pgrB, thick):
    other = "A" if thick == "B" else "B"
    asm = _asm(_fillBlock(other), tail=["s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(pgrA, pgrB), asm) == DCP_THICK_WAIT_UNCOVERED


# ---------------------------------------------------------------------------
# Thick-first issue order: s_wait_tensorcnt N is age-ordered on one counter.
# ---------------------------------------------------------------------------
def _issueOrder(pgrA, pgrB, *args):
    from Tensile.KernelWriter import KernelWriter

    writer = _ThickWaitWriter()
    return KernelWriter._dcpThickThinIssueOrder(
        writer, _thickWaitKernel(pgrA, pgrB), *args)


@pytest.mark.parametrize("pgrA, pgrB, expected", [
    (1, 2, ("B", "A")),
    (2, 1, ("A", "B")),
    (2, 2, ("A", "B")),
    (1, 1, ("A", "B")),
])
def test_dcp_thick_thin_issue_order_puts_the_thick_tensor_first(pgrA, pgrB, expected):
    assert _issueOrder(pgrA, pgrB) == expected


def test_dcp_thick_thin_issue_order_is_a_pure_swap_of_its_arguments():
    tpA, tpB = object(), object()
    assert _issueOrder(1, 2, tpA, tpB) == (tpB, tpA)
    assert _issueOrder(2, 1, tpA, tpB) == (tpA, tpB)


@pytest.mark.parametrize("pgrA, pgrB, thick, thin", [(1, 2, "B", "A"), (2, 1, "A", "B")])
def test_the_ordered_pair_is_the_pair_the_emission_is_gated_on(pgrA, pgrB, thick, thin):
    assert _issueOrder(pgrA, pgrB) == (thick, thin)

    relaxed = _applyThickWait(_thickWaitKernel(pgrA, pgrB), _asm(_fillBlock(thick)))
    assert relaxed.count("s_wait_tensorcnt 2") == 1, \
        "the thick tensor named by the helper is not the one whose gate was relaxed"

    assert _thickWaitRejects(_thickWaitKernel(pgrA, pgrB), _asm(_fillBlock(thin), tail=["s_endpgm"])) == DCP_THICK_WAIT_UNCOVERED


_F8F4 = {
    "MacDataTypeA": DataType("F8"),
    "MacDataTypeB": DataType("F4"),
    "MXBlockA": 32,
    "MXBlockB": 32,
}


def _postConversionState(**overrides):
    """Tuning-path state: four-item MatrixInstruction; MacroTile0/1 not yet assigned."""
    state = {
        "PrefetchGlobalRead": 2,
        "DepthU": 256,
        "MaxLDS": 327680,
        "WavefrontSize": 32,
        "MatrixInstruction": [16, 16, 128, 1],
        "MIBlock": [16, 16, 128, 1, 1, 1],
        "MIWaveTile": [2, 8],
        "MIWaveGroup": [2, 2],
        "SourceSwap": False,
        "ProblemType": _F8F4,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Auto LDS search on the four-item MatrixInstruction tuning path.
# ---------------------------------------------------------------------------
def test_four_item_matrix_instruction_still_yields_a_tile():
    assert _macroTileFromState(_postConversionState()) == (64, 256)


def test_auto_selection_evaluates_every_candidate_on_the_normal_path(monkeypatch):
    import Tensile.Components.DecouplePGR as dcp

    seen = []
    real = dcp.decouplePGRLdsBytesEstimate

    def spy(ks, problemType=None):
        seen.append((ks.get("PrefetchGlobalReadA"), ks.get("PrefetchGlobalReadB")))
        return real(ks, problemType)

    monkeypatch.setattr(dcp, "decouplePGRLdsBytesEstimate", spy)
    selected = pgrAutoPairSelectMaxLds(2, _postConversionState(), _F8F4)
    assert seen == [(2, 2), (2, 1), (1, 2)]
    assert selected == (2, 2)


def test_auto_selection_rejects_when_no_candidate_fits_in_lds():
    assert pgrAutoPairSelectMaxLds(2, _postConversionState(MaxLDS=1024), _F8F4) is None


def test_auto_selection_steps_down_to_a_divergent_pair_when_the_equal_pair_will_not_fit():
    state = _postConversionState()
    probe = dict(state, MacroTile0=64, MacroTile1=256)
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = 2, 2
    # Packed size at MaxLDS=0: a cap just below it must not still admit (2, 2).
    equalPairLds = decouplePGRLdsBytesEstimate(dict(probe, MaxLDS=0), _F8F4)
    selected = pgrAutoPairSelectMaxLds(2, _postConversionState(MaxLDS=equalPairLds - 1), _F8F4)
    assert selected is not None and selected[0] != selected[1]
    assert ldsBlocksForPgrLevel(selected[0]) != ldsBlocksForPgrLevel(selected[1])


def test_auto_selection_rejects_when_the_tile_cannot_be_derived():
    state = _postConversionState()
    for key in ("MIBlock", "MIWaveTile", "MIWaveGroup", "MatrixInstruction"):
        state.pop(key, None)
    assert pgrAutoPairSelectMaxLds(2, state, _F8F4) is None


@pytest.mark.parametrize("depthU", [None, 0, -1])
def test_auto_selection_rejects_an_unresolved_depthu(depthU):
    state = _postConversionState()
    if depthU is None:
        state.pop("DepthU")
    else:
        state["DepthU"] = depthU
    assert pgrAutoPairSelectMaxLds(2, state, _F8F4) is None


@pytest.mark.parametrize("mi, shortcut, derived", [
    ([32, 32, 1, 2, 1, 4, 1, 2, 2], (256, 64), (256, 128)),
    ([16, 16, 32, 4, 1, 2, 2, 2, 2], (64, 64), (64, 256)),
    ([16, 16, 64, 2, 1, 4, 2, 2, 2], (128, 64), (128, 128)),
])
def test_matrix_inst_b_is_distributed_into_the_block_before_the_wave_group(mi, shortcut, derived):
    assert shortcut != derived, "fixture must describe a case the shortcut got wrong"
    assert macroTileFromMatrixInstruction(mi, 32) == derived


def test_state_geometry_beats_a_stale_nine_item_matrix_instruction():
    state = _postConversionState(
        MatrixInstruction=[32, 32, 1, 2, 1, 4, 1, 2, 2],
        MIBlock=[32, 32, 1, 2, 2, 1], MIWaveTile=[4, 1], MIWaveGroup=[1, 4])
    assert _macroTileFromState(state) == (256, 128)


# ---------------------------------------------------------------------------
# Level 0 in a divergent pair.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pgrA, pgrB", [(0, 2), (2, 0), (0, 3), (3, 0)])
def test_divergent_level_zero_is_rejected(pgrA, pgrB):
    ks = {"PrefetchGlobalRead": min(max(pgrA, pgrB), 1),
          "PrefetchGlobalReadA": pgrA, "PrefetchGlobalReadB": pgrB,
          "_ScheduleIterAlg": 0, "PrefetchLocalRead": 1, "DepthU": 256,
          "LocalSplitU": 1, "InnerUnroll": 1, "MatrixInstK": 128, "NumWaves": 4,
          "EnableMatrixInstruction": True}
    reason = divergentPairUnsupportedReason(ks)
    assert reason is not None
    assert "level 0" in reason


@pytest.mark.parametrize("pgrA, pgrB, heldTc", [(-1, 0, "B"), (0, -1, "A")])
def test_one_sided_auto_against_a_pinned_zero_names_the_real_reason(pgrA, pgrB, heldTc):
    state = _postConversionState(PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    reason = resolvePrefetchGlobalReadSpecialValues(state)
    assert reason is not None
    assert "PrefetchGlobalRead%s=0" % heldTc in reason
    assert "level 0" in reason
    assert "no LDS-feasible pair" not in reason


# ---------------------------------------------------------------------------
# Legality is a ranking filter, not a post-hoc rejection.
# ---------------------------------------------------------------------------
def test_candidates_above_two_blocks_never_enter_the_ranking():
    # Import here so a missing symbol fails this test, not collection.
    from Tensile.Components.DecouplePGR import autoPairCandidateIsLegal

    legal = [p for p in pgrAutoPairCandidates(4) if autoPairCandidateIsLegal(*p)]
    assert (3, 2) not in legal and (2, 3) not in legal and (4, 3) not in legal
    assert (2, 2) in legal and (2, 1) in legal and (1, 2) in legal
    assert (3, 3) in legal and (4, 4) in legal, "equal pairs degenerate to scalar"


def test_no_candidate_can_strand_a_legal_pair():
    for pgrA, pgrB in pgrAutoPairCandidates(5):
        blocks = (ldsBlocksForPgrLevel(pgrA), ldsBlocksForPgrLevel(pgrB))
        assert blocks[0] == blocks[1] or max(blocks) <= 2, (pgrA, pgrB)

    state = _postConversionState(PrefetchGlobalRead=3)
    probe = dict(state, MacroTile0=64, MacroTile1=256)

    def estimate(pair):
        probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = pair
        return decouplePGRLdsBytesEstimate(probe, _F8F4)

    # Cap that admits (2, 2) but not (3, 3).
    cap = estimate((2, 2))
    assert cap < estimate((3, 3)), "fixture must reproduce the stranding band"
    selected = pgrAutoPairSelectMaxLds(3, _postConversionState(PrefetchGlobalRead=3, MaxLDS=cap), _F8F4)
    assert selected == (2, 2)


# ---------------------------------------------------------------------------
# Fixture vs matrixInstructionToMIParameters.
# ---------------------------------------------------------------------------
def test_fixture_matches_the_real_conversion():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    problemType = {"DataType": DataType("F8"), "MXBlockA": 32, "MXBlockB": 32}
    produced = matrixInstructionToMIParameters(
        [16, 16, 128, 1, 1, 2, 8, 2, 2], isa, 32, problemType, [32, 4, 1], iim)
    fixture = _postConversionState()
    assert produced["MatrixInstruction"] == fixture["MatrixInstruction"]
    assert list(produced["MIBlock"]) == fixture["MIBlock"]
    assert list(produced["MIWaveTile"]) == fixture["MIWaveTile"]
    assert list(produced["MIWaveGroup"]) == fixture["MIWaveGroup"]


# ---------------------------------------------------------------------------
# Estimate vs setLdsOffsets / setLdsOffsetsDecoupled.
# ---------------------------------------------------------------------------
def _estimateProbe(**overrides):
    probe = _postConversionState(MacroTile0=64, MacroTile1=256)
    probe.update(overrides)
    return probe


def test_equal_pair_block_is_rounded_to_a_power_of_two_only_while_it_fits():
    rounded = decouplePGRLdsBytesEstimate(
        _estimateProbe(MaxLDS=1 << 30, PrefetchGlobalReadA=2, PrefetchGlobalReadB=2),
        _F8F4)
    packed = decouplePGRLdsBytesEstimate(
        _estimateProbe(MaxLDS=rounded - 1, PrefetchGlobalReadA=2, PrefetchGlobalReadB=2),
        _F8F4)
    assert rounded > packed, "fixture must reach the round-up, or it tests nothing"
    block = packed // 2
    assert packed == 2 * block
    swapStride = rounded - block
    assert swapStride & (swapStride - 1) == 0, "the swap stride has to be a power of two"
    assert block <= swapStride < 2 * block, "and the smallest one that holds the block"


def test_the_round_up_can_never_cost_a_pair_that_fitted():
    for maxLds in range(49152, 327681, 4096):
        probe = _estimateProbe(MaxLDS=maxLds, PrefetchGlobalReadA=2, PrefetchGlobalReadB=2)
        est = decouplePGRLdsBytesEstimate(probe, _F8F4)
        packed = decouplePGRLdsBytesEstimate(dict(probe, MaxLDS=0), _F8F4)
        assert est == packed or est <= maxLds, maxLds


def test_divergent_pair_is_packed_with_no_stride_round_up():
    probe = _estimateProbe(PrefetchGlobalReadA=2, PrefetchGlobalReadB=1)
    ldsA, ldsMXSA, ldsMXSB, ldsB = (
        _ldsAlignedBytes(probe, _F8F4, mxTc, probe["DepthU"],
                         probe["MacroTile0"] if "A" in mxTc else probe["MacroTile1"])
        for mxTc in ("A", "MXSA", "MXSB", "B"))
    assert (decouplePGRLdsBytesEstimate(probe, _F8F4)
            == 2 * (ldsA + ldsMXSA) + (ldsMXSB + ldsB))
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = 1, 2
    assert (decouplePGRLdsBytesEstimate(probe, _F8F4)
            == (ldsA + ldsMXSA) + 2 * (ldsMXSB + ldsB))


_RESOLVED_LDS_KEYS = (
    "DepthU", "MacroTile0", "MacroTile1", "MaxLDS", "ConvertAfterDS",
    "LdsPadA", "LdsPadB", "LdsPadMXSA", "LdsPadMXSB",
    "LdsBlockSizePerPadA", "LdsBlockSizePerPadB",
    "LdsBlockSizePerPadMXSA", "LdsBlockSizePerPadMXSB",
    "UnrollMajorLDSA", "UnrollMajorLDSB", "UnrollMajorLDSMXSA", "UnrollMajorLDSMXSB",
    "DirectToVgprA", "DirectToVgprB", "DirectToVgprMXSA", "DirectToVgprMXSB",
)


def _derivedLdsBytesAB(sol, pgrA, pgrB):
    """ldsNumBytesAB as the derivation computed it, read back off its offsets."""
    nBlkA, nBlkB = ldsBlocksForPgrLevel(pgrA), ldsBlocksForPgrLevel(pgrB)
    ldsB = sol.get("ldsNumBytesB")
    if nBlkA != nBlkB:
        lastB = (sol.get("LdsOffsetMXSB") + (nBlkB - 1) * sol.get("LdsOffsetBlkB")
                 + sol.get("LdsNumElementsAlignedMXSB") + ldsB)
        return max(lastB, nBlkA * sol.get("LdsOffsetBlkA"))
    return ((max(sol.get("NumLdsBlk"), 2) - 1) * sol.get("LdsOffsetA_Blk")
            + sol.get("LdsOffsetB") + ldsB)


@pytest.mark.parametrize("pgrA, pgrB", [(2, 2), (2, 1), (1, 2)])
def test_estimate_matches_the_derivation_once_the_padding_is_resolved(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalRead=max(pgrA, pgrB),
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    probe = {key: sol.get(key) for key in _RESOLVED_LDS_KEYS}
    probe["ProblemType"] = sol.get("ProblemType")
    probe["PrefetchGlobalRead"] = max(pgrA, pgrB)
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = pgrA, pgrB
    assert (decouplePGRLdsBytesEstimate(probe, sol.get("ProblemType"))
            == _derivedLdsBytesAB(sol, pgrA, pgrB))


# ---------------------------------------------------------------------------
# (2, 1) vs (1, 2): same LDS, different thick tensor / local-read order.
# ---------------------------------------------------------------------------
_F8F8 = {
    "MacDataTypeA": DataType("F8"),
    "MacDataTypeB": DataType("F8"),
    "MXBlockA": 32,
    "MXBlockB": 32,
}


def _tiedPairState(miWaveTile, miWaveGroup, **overrides):
    """A state where (2, 1) and (1, 2) cost the same LDS."""
    state = {
        "PrefetchGlobalRead": 2,
        "DepthU": 256,
        "MaxLDS": 327680,
        "WavefrontSize": 32,
        "MatrixInstruction": [16, 16, 128, 1],
        "MIBlock": [16, 16, 128, 1, 1, 1],
        "MIWaveTile": list(miWaveTile),
        "MIWaveGroup": list(miWaveGroup),
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "SourceSwap": False,
        "ProblemType": _F8F8,
    }
    state.update(overrides)
    return state


def _tiedPairCap(state):
    """The MaxLDS that admits the two divergent pairs and nothing deeper."""
    probe = dict(state, MacroTile0=128, MacroTile1=128,
                 PrefetchGlobalReadA=2, PrefetchGlobalReadB=1)
    tied = decouplePGRLdsBytesEstimate(probe, _F8F8)
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = 1, 2
    assert decouplePGRLdsBytesEstimate(probe, _F8F8) == tied, \
        "expected equal LDS for (2, 1) and (1, 2)"
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = 2, 2
    assert decouplePGRLdsBytesEstimate(probe, _F8F8) > tied, \
        "and the equal pair has to be what this cap excludes"
    return tied


_TIED_SHAPES = [
    # A reads four times what B does at the same tile, so A takes the second block
    ([8, 2], [1, 4], (2, 1)),
    ([2, 8], [4, 1], (1, 2)),
    # Neither side reads more: the pinned arm
    ([4, 4], [2, 2], (2, 1)),
]


def test_equal_macro_tile_does_not_mean_equal_mi_wave_tile():
    tiles = {tuple(_macroTileFromState(_tiedPairState(wt, wg)))
             for wt, wg, _ in _TIED_SHAPES}
    assert tiles == {(128, 128)}
    assert len({tuple(wt) for wt, _, _ in _TIED_SHAPES}) == len(_TIED_SHAPES)


@pytest.mark.parametrize("miWaveTile, miWaveGroup, expected", _TIED_SHAPES)
def test_auto_makes_the_heavier_local_read_side_thick(miWaveTile, miWaveGroup, expected):
    state = _tiedPairState(miWaveTile, miWaveGroup)
    state["MaxLDS"] = _tiedPairCap(state)
    assert pgrAutoPairSelectMaxLds(2, state, _F8F8) == expected


@pytest.mark.parametrize("miWaveTile, miWaveGroup, expected", _TIED_SHAPES)
def test_auto_tie_break_does_not_depend_on_the_candidate_order(
        monkeypatch, miWaveTile, miWaveGroup, expected):
    state = _tiedPairState(miWaveTile, miWaveGroup)
    state["MaxLDS"] = _tiedPairCap(state)
    ordered = DP.pgrAutoPairCandidates
    monkeypatch.setattr(DP, "pgrAutoPairCandidates",
                        lambda pgr: list(reversed(ordered(pgr))))
    assert pgrAutoPairSelectMaxLds(2, state, _F8F8) == expected


def test_auto_weighs_the_wave_tile_by_its_own_input_count():
    state = _tiedPairState([4, 4], [2, 2], MIInputPerThreadA=32)
    state["MaxLDS"] = _tiedPairCap(state)
    assert pgrAutoPairSelectMaxLds(2, state, _F8F8) == (1, 2)


def ks(fuse=0, pgrA=2, pgrB=1, **over):
    """A solution state that satisfies every guard the grouping consults."""
    state = {
        "PrefetchGlobalRead": 1,
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "NumWaves": 4,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    if pgrA is None:
        del state["PrefetchGlobalReadA"]
    if pgrB is None:
        del state["PrefetchGlobalReadB"]
    state.update(over)
    return state


# Every pair shape derivation can hand the writer. None/None is the legacy
# scalar shape, which is also what an equal (k, k) pair is folded to by
# equalPairDegeneratesToScalar before it reaches the writer.
PAIRS = (
    (None, None, "scalar"),
    (0, 0, "equal"),
    (1, 1, "equal"),
    (2, 2, "equal"),
    (1, 0, "equal"),
    (2, 1, "divergent"),
    (1, 2, "divergent"),
    (2, 0, "divergent"),
    (0, 2, "divergent"),
)


# Expected A/B separation per row, written out so it cannot share a bug with the table.
ROW_SEPARATES_AB = {
    "MX_AB": False,   # {A,B} + {MXSA,MXSB}
    "paired": True,   # {A,MXSA} + {MXSB,B}
    "A_MX": True,     # {A,MXSA,MXSB} + {B}
    "B_MX": True,     # {B,MXSA,MXSB} + {A}
}

def test_every_grouping_row_has_a_stated_expectation():
    assert set(ROW_SEPARATES_AB) == set(TDM_GROUPS), (
        "a grouping row was added without stating whether it separates A from B")


@pytest.mark.parametrize("name", sorted(TDM_GROUPS))
def test_separate_ab_descriptors_over_every_grouping_row(name, monkeypatch):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    assert tdmSeparateABDescriptors(ks()) is ROW_SEPARATES_AB[name]


@pytest.mark.parametrize("name", sorted(TDM_GROUPS))
@pytest.mark.parametrize("pgrA,pgrB,shape", PAIRS)
def test_owner_over_the_full_row_by_pair_cross_product(
        name, pgrA, pgrB, shape, monkeypatch):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    state = ks(pgrA=pgrA, pgrB=pgrB)
    gate = decoupledThickGateRelaxation(state)
    if shape != "divergent":
        assert gate is None
        return
    assert gate is not None, (
        "every divergent pair here is thin on one block, and an issue declares "
        "every LDS region it fills whatever set it rides, so each row earns a "
        "relaxation; the %s row returned None, which is the fuse-gated-presence "
        "bug" % name)
    if ROW_SEPARATES_AB[name]:
        assert gate.mechanism == DCP_THICK_GATE_TOKENS
    else:
        assert gate.mechanism == DCP_THICK_GATE_TEXT
    assert gate.tensorcnt == DCP_THICK_GATE_SUPPORTED[gate.mechanism](state)


@pytest.mark.parametrize("pgrA,pgrB,shape", PAIRS)
def test_presence_follows_the_pair_not_the_fuse_integer(pgrA, pgrB, shape):
    present = {
        fuse: decoupledThickGateRelaxation(ks(fuse=fuse, pgrA=pgrA, pgrB=pgrB)) is not None
        for fuse in sorted(TDM_FUSE_GROUPING)
    }
    assert present == {fuse: shape == "divergent"
                       for fuse in sorted(TDM_FUSE_GROUPING)}


@pytest.mark.parametrize("fuse,expected", sorted(TDM_FUSE_GROUPING.items()))
def test_reachable_fuse_integers_resolve_to_their_row(fuse, expected):
    assert tdmGrouping(ks(fuse=fuse)).name == expected


# Declined TDMFuse must not look paired. Unreachable in Valid solutions.
DECLINES = (
    ("TDMSplit", {"TDMSplit": True}),
    ("UseSubtileImpl", {"UseSubtileImpl": True}),
    ("no MX scale on A", {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}}),
    ("no MX scale on B", {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}}),
    ("NumWaves==1", {"NumWaves": 1}),
    ("TDM moves A only", {"TDMInst": 1}),
    ("TDM moves B only", {"TDMInst": 2}),
)


@pytest.mark.parametrize("why,override", DECLINES)
def test_declined_fusion_falls_back_to_the_shared_grouping(why, override):
    assert tdmGrouping(ks(fuse=1, **override)).name == TDM_FUSE_GROUPING[0], why


@pytest.mark.parametrize("why,override", DECLINES)
def test_declined_fusion_never_gets_disjoint_tokens(why, override):
    gate = decoupledThickGateRelaxation(ks(fuse=1, **override))
    assert gate is not None, why
    assert gate.mechanism == DCP_THICK_GATE_TEXT, (
        "TDMFuse=1 declined on %s shares one descriptor set, so its tensor ops "
        "share a token and cannot be drained independently; handing it "
        "DCP_THICK_GATE_TOKENS would gate reads on data that has not landed" % why)


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
@pytest.mark.parametrize("why,override", DECLINES + (("(none)", {}),))
def test_mechanism_agrees_with_the_grouping_by_construction(fuse, why, override):
    state = ks(fuse=fuse, **override)
    gate = decoupledThickGateRelaxation(state)
    if gate is None:
        return
    separates = tdmSeparateABDescriptors(state)
    assert (gate.mechanism == DCP_THICK_GATE_TOKENS) is separates, why


def test_token_path_requires_a_second_token_stream():
    assert decoupledThickGateRelaxation(ks(fuse=1, enableTDMA=False)) is None
    assert decoupledThickGateRelaxation(ks(fuse=1, enableTDMB=False)) is None


@pytest.mark.parametrize("name", sorted(TDM_GROUPS))
@pytest.mark.parametrize("pgrA,pgrB", [(3, 2), (2, 3), (4, 2)])
def test_a_thin_side_deeper_than_one_block_earns_nothing_on_any_row(
        name, pgrA, pgrB, monkeypatch):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    assert decoupledThickGateRelaxation(ks(pgrA=pgrA, pgrB=pgrB)) is None


def test_supported_counts_match_the_documented_mechanisms():
    assert sorted(DCP_THICK_GATE_SUPPORTED) == sorted([DCP_THICK_GATE_TEXT,
                                                       DCP_THICK_GATE_TOKENS])
    assert DCP_THICK_GATE_SUPPORTED[DCP_THICK_GATE_TOKENS](ks()) == 1
    assert DCP_THICK_GATE_SUPPORTED[DCP_THICK_GATE_TEXT](ks()) == 2
    scaleLess = ks(ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert DCP_THICK_GATE_SUPPORTED[DCP_THICK_GATE_TEXT](scaleLess) == 1


@pytest.mark.parametrize("name", sorted(TDM_GROUPS))
@pytest.mark.parametrize("pgrA,pgrB,shape", PAIRS)
def test_the_token_pass_predicate_names_exactly_the_tokens_mechanism(
        name, pgrA, pgrB, shape, monkeypatch):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    assert dcpThickGateFromTokenPasses(ks(pgrA=pgrA, pgrB=pgrB)) is (
        shape == "divergent" and ROW_SEPARATES_AB[name])


MARKER = "DcpEarlyFillA"          # ks() is a 2/1 pair, so A is the thick side


# ks() carries MX scales on both tensors, so its shared grouping writes two
# descriptor sets per fill and the pass relaxes to 2.
TEXT_RELAXED = DCP_THICK_GATE_SUPPORTED[DCP_THICK_GATE_TEXT](ks())


GATE = ["s_wait_tensorcnt 0", "s_barrier_signal -1", "s_barrier_wait -1"]


READS = ["ds_load_b128 v[0:3], v[64] offset:128",
         "ds_load_b64 v[4:5], v[65] offset:512"]


MATH = ["v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:31], 0"]


FILL = ["tensor_load_to_lds s[0:3], s[8:15]"]


def site(label, *body):
    """One fill-end label and the lines the schedule left under it."""
    return ["label_%s:" % label] + [line for part in body for line in part]


def asm(*blocks):
    """Those blocks as the one string the pass is handed."""
    return "".join(line + "\n" for block in blocks for line in block)


CLONE = "InitCIterWmma_label_DcpEarlyFillAEnd_0"


MAIN = "DcpEarlyFillAEnd"


TAIL = "InitCIterWmma_target_0"


PER_SITE = (                                       # gfx1250
    site(CLONE, GATE, READS, MATH, ["s_branch label_%s:" % TAIL]),
    ["label_LoopBeginL:"] + FILL,
    site(MAIN, GATE, READS, MATH),
)


MERGED = (                                         # gfx1250v0
    site(CLONE, MATH, ["s_branch label_%s" % TAIL]),
    ["label_LoopBeginL:"] + FILL,
    site(MAIN, MATH),
    ["label_%s:" % TAIL] + GATE + READS,
)


THREE_SITE = (                       # 33 of 72 in the shipped gfx1250 corpus
    site(CLONE, GATE, READS),
    site(MAIN, MATH),                # no gate of its own, and always accepted
    site(MAIN + "_1", GATE, READS),
)


def uncovered(schedule, relaxed=TEXT_RELAXED):
    """Sites still uncovered after retagging, matching _dcpRelaxThickTextGate (accept already-strong gates)."""
    lines = asm(*schedule).splitlines(keepends=True)
    accepted = set()
    for n, line in enumerate(lines):
        gate = DP.DCP_TENSORCNT_RE.match(line)
        if gate is None:
            continue
        if int(gate.group(1)) == 0:
            lines[n] = re.sub(r"^(s_wait_tensorcnt\s+)0(\s|$)",
                              r"\g<1>%d\2" % relaxed, line)
            accepted.add(n)
        elif int(gate.group(1)) <= relaxed:
            accepted.add(n)
    return DP.dcpThickGateUncoveredSites(lines, MARKER, relaxed, accepted)


@pytest.mark.parametrize("why,schedule", [("per-site", PER_SITE),
                                          ("merged", MERGED),
                                          ("three-site", THREE_SITE)])
def test_every_schedule_the_compiler_emits_is_covered(why, schedule):
    assert uncovered(schedule) == []


def test_reads_ahead_of_the_gate_are_refused():
    bad = (site(CLONE, READS, GATE),)
    assert [why.split(" at line")[0] for _, why in uncovered(bad)] == [
        "reaches ds_load_b128"]


def test_a_gate_weaker_than_the_relaxation_is_refused():
    weak = "s_wait_tensorcnt %d" % (TEXT_RELAXED + 1)
    foreign = (site(CLONE, [weak], READS),)
    assert [why.split(" at line")[0] for _, why in uncovered(foreign)] == [
        "first gate is %s" % weak]


def test_a_gate_already_stronger_than_the_relaxation_is_accepted():
    strong = (site(CLONE, ["s_wait_tensorcnt %d" % (TEXT_RELAXED - 1)], READS),)
    assert uncovered(strong) == []


def test_a_site_is_not_judged_by_a_gate_in_the_next_fill_region():
    weak = "s_wait_tensorcnt %d" % (TEXT_RELAXED + 1)
    schedule = (site(MAIN, MATH),
                site(MAIN + "_1", [weak], READS))
    assert [why.split(" at line")[0] for _, why in uncovered(schedule)] == [
        "first gate is %s" % weak]


def test_a_header_copy_with_nothing_below_it_is_covered_vacuously():
    assert uncovered((site(CLONE, GATE, READS), site(MAIN, MATH))) == []


def test_no_fill_label_at_all_is_refused():
    assert [why for _, why in uncovered((["label_LoopBeginL:"] + FILL,))] == [
        "no DcpEarlyFillA label was emitted at all"]


def test_acceptance_does_not_depend_on_how_many_waits_were_retagged():
    """PER_SITE retags two waits, MERGED one; both must be accepted."""
    from Tensile.KernelWriter import KernelWriter

    solution = ks()
    gate = decoupledThickGateRelaxation(solution)
    assert gate.mechanism == DCP_THICK_GATE_TEXT
    relaxed = "s_wait_tensorcnt %d" % gate.tensorcnt

    perSite = KernelWriter._dcpRelaxThickTextGate(None, solution, asm(*PER_SITE))
    merged = KernelWriter._dcpRelaxThickTextGate(None, solution, asm(*MERGED))

    assert (perSite.count(relaxed), merged.count(relaxed)) == (2, 1)
    assert "s_wait_tensorcnt 0" not in perSite + merged


# ---------------------------------------------------------------------------
# Auto retries only DCP_LDS_CAPACITY_REFUSED.
# ---------------------------------------------------------------------------
def test_auto_pair_ranking_is_ordered_by_estimate_and_holds_only_feasible_pairs():
    state = _postConversionState()
    ranking = pgrAutoPairRanking(2, state, _F8F4)
    assert set(ranking) == {(2, 2), (2, 1), (1, 2)}
    probe = dict(state, MacroTile0=64, MacroTile1=256)
    sizes = []
    for pgrA, pgrB in ranking:
        probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = pgrA, pgrB
        sizes.append(decouplePGRLdsBytesEstimate(probe, _F8F4))
    assert sizes == sorted(sizes, reverse=True)
    assert all(size <= state["MaxLDS"] for size in sizes)


def test_auto_pair_ranking_drops_the_candidates_a_cap_excludes():
    wide = pgrAutoPairRanking(2, _postConversionState(), _F8F4)
    narrow = pgrAutoPairRanking(2, _postConversionState(MaxLDS=90000), _F8F4)
    assert narrow and len(narrow) < len(wide)
    assert narrow == [pair for pair in wide if pair in narrow]
    assert pgrAutoPairRanking(2, _postConversionState(MaxLDS=1024), _F8F4) == []


def test_resolve_auto_skip_walks_the_ranking_in_order():
    ranking = pgrAutoPairRanking(2, _postConversionState(), _F8F4)
    assert len(ranking) > 1, "expected ranking longer than 1"
    for skip, expected in enumerate(ranking):
        state = _postConversionState(PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
        assert resolvePrefetchGlobalReadSpecialValues(state, skip) is None
        assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == expected


def test_resolve_auto_skip_past_the_ranking_reports_the_same_no_fit():
    ranking = pgrAutoPairRanking(2, _postConversionState(), _F8F4)
    exhausted = _postConversionState(PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
    nothingFits = _postConversionState(PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1,
                                       MaxLDS=1024)
    reason = resolvePrefetchGlobalReadSpecialValues(exhausted, len(ranking))
    assert reason == resolvePrefetchGlobalReadSpecialValues(nothingFits, 0)
    assert "no LDS-feasible pair" in reason
    assert exhausted["PrefetchGlobalReadA"] == -1


@pytest.mark.parametrize("state, requested", [
    (dict(PrefetchGlobalRead=2, PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1), True),
    (dict(PrefetchGlobalRead=2, PrefetchGlobalReadA=-1, PrefetchGlobalReadB=2), True),
    (dict(PrefetchGlobalRead=2, PrefetchGlobalReadA=2, PrefetchGlobalReadB=-1), True),
    # The bare scalar asks for nothing, so there is no ranking to step.
    (dict(PrefetchGlobalRead=-1), False),
    (dict(PrefetchGlobalRead=2, PrefetchGlobalReadA=2, PrefetchGlobalReadB=1), False),
    (dict(PrefetchGlobalRead=2), False),
    # Below two the keys are dropped rather than searched, so there is no
    # ranking to step through.
    (dict(PrefetchGlobalRead=1, PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1), False),
    (dict(PrefetchGlobalRead=0, PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1), False),
    # One-sided keys are rejected, never searched.
    (dict(PrefetchGlobalRead=2, PrefetchGlobalReadA=-1), False),
])
def test_pgr_auto_pair_requested(state, requested):
    assert pgrAutoPairRequested(state) is requested


# Estimate 327680 vs derived 332800 at this BBS tile.
_LDS_WITNESSES = [
    ([16, 16, 32, 1, 1, 8, 4, 2, 2], 256, (1, 2)),
    ([16, 16, 32, 1, 1, 4, 8, 2, 2], 256, (2, 1)),
]


def _bbsWitness(**overrides):
    params = dict(
        ProblemType={"MacDataTypeA": "B", "MacDataTypeB": "B", "DataType": "B",
                     "DestDataType": "B", "ComputeDataType": "s",
                     "MXBlockA": 0, "MXBlockB": 0},
        DepthU=256,
        MaxLDS=327680,
        LDSTrInst=False,
    )
    params.update(overrides)
    return params


@pytest.mark.parametrize("mi, depthU, expected", _LDS_WITNESSES)
def test_solution_auto_steps_past_a_pair_the_lds_check_refuses(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mi, depthU, expected):
    refused, refusedOut = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
        MatrixInstruction=mi, DepthU=depthU,
        PrefetchGlobalReadA=expected[1], PrefetchGlobalReadB=expected[0]))
    assert refused.get("Valid") is False, refusedOut
    assert "bytes of LDS" in refusedOut, refusedOut

    auto, autoOut = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
        MatrixInstruction=mi, DepthU=depthU,
        PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1))
    assert auto.get("Valid") is True, autoOut
    assert (auto["PrefetchGlobalReadA"], auto["PrefetchGlobalReadB"]) == expected
    assert auto["LdsNumBytes"] <= auto["MaxLDS"]


@pytest.mark.parametrize("mi, depthU, expected", _LDS_WITNESSES)
def test_solution_auto_matches_the_same_pair_written_out(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mi, depthU, expected):
    auto, autoOut = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
        MatrixInstruction=mi, DepthU=depthU,
        PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1))
    assert auto.get("Valid") is True, autoOut
    explicit, _ = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
        MatrixInstruction=mi, DepthU=depthU,
        PrefetchGlobalReadA=expected[0], PrefetchGlobalReadB=expected[1]))
    assert explicit.get("Valid") is True
    assert dict(auto) == dict(explicit)


def test_solution_auto_exhausting_the_ranking_still_rejects(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
        MatrixInstruction=[16, 16, 32, 1, 1, 8, 4, 2, 2], MaxLDS=40960,
        PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1))
    assert sol.get("Valid") is False, out


def test_solution_auto_does_not_retry_an_unrelated_rejection(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
        MatrixInstruction=[16, 16, 32, 1, 1, 8, 4, 2, 2], ClusterDim=[2, 1],
        PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1))
    assert sol.get("Valid") is False, out
    assert "ClusterDim != [1, 1] is incompatible with" in out, out


def test_solution_derivation_leaves_no_capacity_marker_behind(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    from Tensile.SolutionStructs.Solution import Solution

    for maxLds in (327680, 40960):
        sol, _ = _derive(gfx1250_iim, assembler, capsys, **_bbsWitness(
            MatrixInstruction=[16, 16, 32, 1, 1, 8, 4, 2, 2], MaxLDS=maxLds,
            PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1))
        assert Solution.DCP_LDS_CAPACITY_REFUSED not in sol


# ---------------------------------------------------------------------------
# Which ScheduleIterAlg can carry a divergent pair.
# ---------------------------------------------------------------------------


def _dcpEmittedFillsAndGates(asm):
    """([(line, label, side)] fill labels, [(line, count)] gates) of emitted text."""
    fills, gates = [], []
    for i, line in enumerate(asm.splitlines()):
        fill = re.match(r"\s*(label_Dcp(?:Early|Late)Fill([AB])(?:End)?)\s*:", line)
        if fill:
            fills.append((i, fill.group(1), fill.group(2)))
        gate = re.match(r"\s*s_wait_tensorcnt\s+(\d+)(?:\s|$)", line)
        if gate:
            gates.append((i, int(gate.group(1))))
    return fills, gates


def _dcpSolutionModule():
    """The Solution *module*. Importing the class of the same name makes monkeypatch a no-op."""
    import importlib

    return importlib.import_module("Tensile.SolutionStructs.Solution")


@pytest.mark.parametrize("tdmFuse", [0, 1])
@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_schedule_iter_alg_3_refuses_a_divergent_pair_over_the_fill_placement(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, tdmFuse, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=tdmFuse,
                       ScheduleIterAlg=3, PrefetchGlobalRead=max(pgrA, pgrB),
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is False, out
    # SIA=4 remaps; 3 keeps _ScheduleIterAlg=3, so the fill never reaches the header.
    assert sol["_ScheduleIterAlg"] == 3, sol["_ScheduleIterAlg"]
    assert sol["_StinkyTofuOptLevel"] == 0, sol["_StinkyTofuOptLevel"]
    assert "need a single-buffered fill slot" in out, out
    assert "use ScheduleIterAlg=0 or 4" in out, out
    assert "no complete fill group to re-slot" in out, out


@pytest.mark.parametrize("scheduleIterAlg", [0, 3, 4])
@pytest.mark.parametrize("tdmFuse", [0, 1])
@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_the_relaxed_gate_and_its_side_order_across_every_schedule_iter_alg(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, scheduleIterAlg, tdmFuse,
        pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=tdmFuse,
                       ScheduleIterAlg=scheduleIterAlg,
                       PrefetchGlobalRead=max(pgrA, pgrB),
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)

    if sol.get("Valid") is not True:
        if scheduleIterAlg == 3:
            assert "no complete fill group to re-slot" in out, out
        else:
            # SIA=0 on a grouping that shares one descriptor set: the text
            # mechanism rewrites emitted instructions in a level 3 pass.
            assert "which needs _StinkyTofuOptLevel=3" in out, out
        return

    gate = decoupledThickGateRelaxation(sol)
    assert gate is not None, "a divergent pair that derives Valid owes a relaxation"
    errs, asm = _emitDerived(sol, assembler)
    warnings = [line for line in capsys.readouterr().out.splitlines()
                if "WARNING" in line]
    assert errs == [0] and asm, (errs, len(asm))
    assert not warnings, warnings[-1]

    fills, gates = _dcpEmittedFillsAndGates(asm)
    thick = tdmWaveIssueOrder(sol, "A", "B")[0]
    assert fills, "a divergent pair emitted no Dcp fill label at all"
    assert fills[0][2] == thick, (
        "Henry's order wants %s filled first; the first label emitted is %s"
        % (thick, fills[0][1]))

    below = [(line, count) for line, count in gates if line > fills[0][0]]
    assert below, "no s_wait_tensorcnt anywhere below %s" % fills[0][1]
    assert below[0][1] == gate.tensorcnt, (
        "%s at ScheduleIterAlg=%d (_StinkyTofuOptLevel=%s): the first gate below "
        "%s is s_wait_tensorcnt %d, not the relaxed %d that "
        "decoupledThickGateRelaxation promised. Every gate: %s"
        % (thick, scheduleIterAlg, sol.get("_StinkyTofuOptLevel"), fills[0][1],
           below[0][1], gate.tensorcnt, [c for _, c in gates]))


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_the_schedule_iter_alg_3_refusal_is_correctness_not_conservatism(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, monkeypatch, pgrA, pgrB):
    solutionModule = _dcpSolutionModule()
    real = solutionModule.divergentPairUnsupportedReason

    def withoutScheduleIterAlgTerm(ks):
        saved = ks["_ScheduleIterAlg"]
        ks["_ScheduleIterAlg"] = 0
        try:
            return real(ks)
        finally:
            ks["_ScheduleIterAlg"] = saved

    monkeypatch.setattr(solutionModule, "divergentPairUnsupportedReason",
                        withoutScheduleIterAlgTerm)

    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=1,
                       ScheduleIterAlg=3, PrefetchGlobalRead=max(pgrA, pgrB),
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, (
        "expected Valid after masking the ScheduleIterAlg term, got: %s" % out)
    assert sol["_ScheduleIterAlg"] == 3, sol["_ScheduleIterAlg"]

    with pytest.raises(AssertionError,
                       match="fill group to re-slot carries no tensor_load_to_lds"):
        _emitDerived(sol, assembler)
