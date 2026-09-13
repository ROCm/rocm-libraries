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
from Tensile.Components.DecouplePGR import decouplePGRLdsBytesEstimate, divergentPairUnsupportedReason, ldsBlocksForPgrLevel, macroTileFromMatrixInstruction, pgrAutoPairCandidates, pgrAutoPairSelectMaxLds, resolvePrefetchGlobalReadSpecialValues, _macroTileFromState
import re
from Tensile.Components import TDMFuse as TF
from Tensile.Components import DecouplePGR as DP
from Tensile.Components.DecouplePGR import decouplePGRBlocks
from Tensile.Components.TDMFuse import TDM_FUSE_GROUPING, TDM_GROUPS, tdmGrouping, tdmSeparateABDescriptors
from Tensile.Components.DecouplePGR import DCP_THICK_GATE_SUPPORTED, DCP_THICK_GATE_TEXT, DCP_THICK_GATE_TOKENS, decoupledThickGateRelaxation

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


@pytest.mark.parametrize("pgrA, pgrB", [(0, 0), (1, 1)])
def test_resolve_leaves_equal_pair_for_scalar_degeneration(pgrA, pgrB):
    state = {"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": pgrA, "PrefetchGlobalReadB": pgrB}
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (pgrA, pgrB)


def test_resolve_scalar_auto_starts_at_two():
    state = _autoSelectState(PrefetchGlobalRead=-1)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (2, 2)


# ---------------------------------------------------------------------------
# One-sided auto: -1 on one tensor, a real depth on the other. PGR, PGRA and
# PGRB each support -1 independently, so this runs the same max-LDS search
# narrowed to the pairs that keep the fixed tensor where the caller put it.
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
    """No pair in the space keeps a tensor at 0, so the filter empties."""
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
    """PrefetchGlobalRead=0 would give an empty candidate space, but pinning
    B at 2 says level 2 is wanted, so the search has to be able to reach it."""
    state = _autoSelectState(PrefetchGlobalRead=0, PrefetchGlobalReadA=-1,
                             PrefetchGlobalReadB=2)
    assert resolvePrefetchGlobalReadSpecialValues(state) is None
    assert (state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"]) == (2, 2)


@pytest.mark.parametrize("pgrA, pgrB, heldTc", [(-1, 1, "B"), (1, -1, "A")])
def test_resolve_one_sided_auto_rejects_when_nothing_fits(pgrA, pgrB, heldTc):
    """Explicit rejection naming the held side, never a silent fallback."""
    state = _autoSelectState(MaxLDS=1024, PrefetchGlobalReadA=pgrA,
                             PrefetchGlobalReadB=pgrB)
    reason = resolvePrefetchGlobalReadSpecialValues(state)
    assert reason is not None
    assert "no LDS-feasible pair" in reason
    assert "PrefetchGlobalRead%s held at" % heldTc in reason


@pytest.mark.parametrize("pgrA, pgrB, expected", [(-1, 1, (True, 2, 1)), (1, -1, (True, 1, 2))])
def test_one_sided_auto_reaches_decouple_pgr_blocks(pgrA, pgrB, expected):
    """decouplePGRBlocks reads PrefetchGlobalReadA/B, so the resolved per-tensor
    depths drive the LDS block counts instead of degenerating to the scalar."""
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
    """No element size means no LDS estimate, so there is nothing to rank.

    This used to return the first candidate. That is indistinguishable from a
    search that ran and chose it, which is what hid the missing search.
    """
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
# Precedence between the legacy scalar and the two per-tensor keys, end to end
# through Solution derivation. Each of PrefetchGlobalRead, PrefetchGlobalReadA
# and PrefetchGlobalReadB accepts -1 independently.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pgrA, pgrB, expected", [(-1, 1, (2, 1)), (1, -1, (1, 2))])
def test_solution_one_sided_auto_searches_the_constrained_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, expected):
    """One-sided auto holds the fixed tensor and searches the other.

    At this tile the search space for a held side is a single pair, so the
    assertion is that the fixed depth survives and the -1 side is filled in --
    not that a wider ranking happened here.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    assert (sol.get("PrefetchGlobalReadA"), sol.get("PrefetchGlobalReadB")) == expected


@pytest.mark.parametrize("pgrA, pgrB", [(-1, 2), (2, -1)])
def test_solution_one_sided_auto_can_pick_the_equal_pair_and_degenerate(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    """Holding a tensor at 2 leaves (2,2) and the divergent pair; (2,2) wins on
    LDS, and an equal pair then degenerates to the scalar as it always has."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 2
    assert sol.get("PrefetchGlobalReadA") is None
    assert sol.get("PrefetchGlobalReadB") is None


def test_solution_both_auto_picks_max_lds_pair(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """(-1, -1) searches both sides; (2,2) is the max-LDS combination here and
    then degenerates to the scalar."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=-1, PrefetchGlobalReadB=-1)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 2
    assert (sol.get("PrefetchGlobalReadA"), sol.get("PrefetchGlobalReadB")) == (None, None)


def test_solution_legacy_scalar_auto_without_per_tensor_keys(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Legacy PrefetchGlobalRead=-1 with no per-tensor keys still resolves."""
    params = {"PrefetchGlobalRead": -1}
    params["PrefetchGlobalReadA"] = None
    params["PrefetchGlobalReadB"] = None
    sol, out = _derive(gfx1250_iim, assembler, capsys, **params)
    assert sol.get("Valid") is True, out
    assert sol.get("PrefetchGlobalRead") == 2


@pytest.mark.parametrize("pgrA, pgrB, expected", [(1, 2, (1, 2)), (2, 1, (2, 1))])
def test_solution_per_tensor_values_win_over_the_legacy_scalar(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, expected):
    """Legacy PrefetchGlobalRead set alongside a per-tensor pair: the pair is
    what survives derivation, so the scalar cannot silently override it."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchGlobalRead=2,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out
    assert (sol.get("PrefetchGlobalReadA"), sol.get("PrefetchGlobalReadB")) == expected
    assert decouplePGRBlocks(sol) == (True, expected[0], expected[1])


# ---------------------------------------------------------------------------
# Divergent thick-wait post-pass (KernelWriter._dcpApplyThickWait1).
# A text pass over already-emitted assembly, so it needs no toolchain: the
# writer is stubbed down to the two things the pass touches.
# ---------------------------------------------------------------------------
class _ThickWaitWriter:
    def __init__(self, memTokenLdsDcp=None):
        self.states = types.SimpleNamespace(overflowedResources=0)
        if memTokenLdsDcp is not None:
            self.states.memTokenLdsDcp = memTokenLdsDcp

    def _dcpDivergent(self, kernel):
        from Tensile.KernelWriterAssembly import KernelWriterAssembly

        return KernelWriterAssembly._dcpDivergent(self, kernel)


def _applyThickWait(kernel, asm, memTokenLdsDcp=None):
    from Tensile.KernelWriter import KernelWriter

    writer = _ThickWaitWriter(memTokenLdsDcp)
    return KernelWriter._dcpApplyThickWait1(writer, kernel, asm)


DCP_THICK_WAIT_UNCOVERED = 10


def _thickWaitRejects(kernel, asm, memTokenLdsDcp=None):
    """The overflowedResources code the pass set, or 0 if it accepted."""
    from Tensile.KernelWriter import KernelWriter

    writer = _ThickWaitWriter(memTokenLdsDcp)
    KernelWriter._dcpApplyThickWait1(writer, kernel, asm)
    return writer.states.overflowedResources


def _thickWaitKernel(pgrA=1, pgrB=2, **overrides):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "TDMFuse": 0,
        "InitCIterWmma": 0,
        # Which arm emits this kernel is decided by the resolved grouping, and
        # tdmGrouping declines the grouping a TDMFuse integer names when its
        # guards are unmet -- falling back to the shared default. A fixture that
        # set TDMFuse=1 and nothing else therefore described a solution whose
        # descriptors are shared, not a paired one, and only looked paired while
        # the emission sites read the same integer it did. These keys are what
        # make TDMFuse=1 actually resolve to {A,MXSA} + {MXSB,B}; they are inert
        # at TDMFuse=0, whose grouping is the fallback and needs no guard.
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
    """One emitted fill block, in the layout the arm named by `paired` emits.

    Separate descriptors (`paired=False`) make the label the branch target that
    skips the fill, so the block reads `[label, body..., wait]` and the wait
    just after the label really is the thick tensor's own gate.

    TDMFuse=1 (`paired=True`) appends the label *after* the fill body and
    nothing branches to it, so the block reads `[body..., label]` and holds no
    wait at all -- whatever a forward scan finds next is outside the group.
    Generating the non-paired layout for a paired case is what let the paired
    tests pass against assembly that arm cannot emit, so `wait` is refused here.

    `clone` names it the way the InitCIterWmma region clone does.
    """
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
    """The retag loop stops at the next fill label, so the drain past it is
    never rewritten -- and the coverage check will not accept a gate the pass
    did not write, so the kernel is still refused rather than credited with a
    wait belonging to another group."""
    asm = _asm(_fillBlock("B", wait=None), tail=["label_DcpLateFillAEnd:",
                                                 "s_wait_tensorcnt 0", "s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(), asm) == DCP_THICK_WAIT_UNCOVERED


def test_thick_wait_accepts_a_schedule_that_merges_the_two_gates():
    """One retag can cover two fill ends, so a retag count is not the test.

    gfx1250v0's cost table sinks the iter0 clone's LDS reads and its gate below
    the convergence label, leaving one drain that gates both paths. This shape
    used to be refused as a shortfall against InitCIterWmma=1, which cost six
    kernels -- all of them TDMFuse=0, the default grouping. Nothing reads ahead
    of the surviving gate, so there is nothing to refuse.
    """
    asm = _asm(_fillBlock("B", clone=True, wait=None), _fillBlock("B"))
    out = _applyThickWait(_thickWaitKernel(InitCIterWmma=1), asm)
    assert out.count("s_wait_tensorcnt 2") == 1
    assert "s_wait_tensorcnt 0" not in out


def test_thick_wait_shortfall_drops_one_kernel_instead_of_the_build():
    """A real shortfall still raises per kernel, so the caller drops that
    kernel and not the build. What makes it real is reads starting ahead of the
    drain that gates them, which is what the gate exists to prevent -- not a
    retag total falling short of a solution parameter.
    """
    asm = _asm(["label_DcpEarlyFillBEnd:",
                "ds_load_b128 v[0:3], v[64] offset:128",
                "s_wait_tensorcnt 0"], tail=["s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(), asm) == DCP_THICK_WAIT_UNCOVERED


def test_thick_wait_refuses_to_walk_past_the_thick_gate_to_the_thin_drain():
    """The gate right after the thick fill is the thick tensor's. The next one
    drains the thin tensor's refill into the single block its reads are about
    to touch, so reaching it means the emitted shape is not what this pass
    assumes -- reject the kernel instead of relaxing a real dependency."""
    asm = _asm(_fillBlock("B", wait="s_wait_tensorcnt 2"),
               tail=["s_wait_tensorcnt 0", "s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(), asm) == DCP_THICK_WAIT_UNCOVERED


# ---------------------------------------------------------------------------
# TDMFuse=1 is not this pass's business. memTokenLdsDcp gives A and B disjoint
# tensor tokens, so the wait-count insertion pass computes the relaxed thick
# gate from dataflow and emits it directly. These pin the no-op against the
# layout the paired arm actually emits -- `[body..., label]`, label last.
# ---------------------------------------------------------------------------
_PAIRED_TOKENS = {"A": (0, 1), "B": (2, 3)}


@pytest.mark.parametrize("asmArgs, asmKwargs, tokens", [
    # the production layout itself
    (((("B", 8, True),),), {}, _PAIRED_TOKENS),
    # across the InitCIterWmma region clone and a long fill body
    (((("B", 400, True, True), ("B", 400, True)),), {}, _PAIRED_TOKENS),
    # a gate the wait-count insertion pass already emitted as 1, and a wider 2
    (((("B", 4, True),),), {"tail": ["s_wait_tensorcnt 1", "s_endpgm"]}, _PAIRED_TOKENS),
    (((("B", 4, True),),), {"tail": ["s_wait_tensorcnt 2", "s_endpgm"]}, _PAIRED_TOKENS),
    # a thick fill that emitted no label at all
    (((("A", 0, True),),), {"tail": ["s_endpgm"]}, _PAIRED_TOKENS),
    # and with no LDS tokens on the writer at all
    (((("B", 0, True),),), {}, None),
], ids=["production", "clone+long", "gate1", "gate2", "unlabelled", "no-tokens"])
def test_thick_wait_paired_is_a_no_op(asmArgs, asmKwargs, tokens):
    """One branch, so one test. The paired arm returns the assembly untouched
    before the scan, which makes the token map, the region clone, the body
    length, the label and any pre-existing gate all the same code path."""
    blocks = [_fillBlock(tc, body=body, paired=paired,
                         clone=(rest[0] if rest else False))
              for (tc, body, paired, *rest) in asmArgs[0]]
    asm = _asm(*blocks, **asmKwargs)
    assert _applyThickWait(_thickWaitKernel(TDMFuse=1), asm, memTokenLdsDcp=tokens) == asm


def test_thick_wait_paired_leaves_a_downstream_zero_wait_alone():
    """Regression for the rewrite this pass used to do.

    The paired label is appended after the fill body, so a forward scan is
    already outside the group and the first wait it meets belongs to unrelated
    code. Relaxing that one to 1 would let reads start before a real dependency
    had drained -- a live synchronisation hazard, not a missed optimisation.
    """
    asm = _asm(_fillBlock("B", body=8, paired=True),
               tail=["s_wait_tensorcnt 0", "label_DcpLateFillAEnd:", "s_endpgm"])
    out = _applyThickWait(_thickWaitKernel(TDMFuse=1), asm,
                          memTokenLdsDcp=_PAIRED_TOKENS)
    assert out == asm
    assert "s_wait_tensorcnt 1" not in out
    assert out.count("s_wait_tensorcnt 0") == 1


@pytest.mark.parametrize("pgrA, pgrB, thick", [(1, 2, "B"), (2, 1, "A")])
def test_thick_wait_target_follows_the_label_the_fill_emitted(pgrA, pgrB, thick):
    """Only the double-buffered tensor gets a DcpEarlyFill label, so the target
    is fixed by LDS block count and cannot be re-chosen by transfer size: the
    other tensor's name matches no label in the emitted kernel."""
    other = "A" if thick == "B" else "B"
    asm = _asm(_fillBlock(other), tail=["s_endpgm"])
    assert _thickWaitRejects(_thickWaitKernel(pgrA, pgrB), asm) == DCP_THICK_WAIT_UNCOVERED


# ---------------------------------------------------------------------------
# Thick/thin issue order (KernelWriter._dcpThickThinIssueOrder). Thick-first is
# what makes the relaxed gate mean anything: s_wait_tensorcnt N is an
# age-ordered drain on one counter, so which tensor a count bypasses follows
# from issue order alone. It was guaranteed by construction but asserted
# nowhere.
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
    """Call sites pass tensor-parameter objects, not the names "A"/"B", so the
    helper has to reorder whatever it is handed rather than return literals."""
    tpA, tpB = object(), object()
    assert _issueOrder(1, 2, tpA, tpB) == (tpB, tpA)
    assert _issueOrder(2, 1, tpA, tpB) == (tpA, tpB)


@pytest.mark.parametrize("pgrA, pgrB, thick, thin", [(1, 2, "B", "A"), (2, 1, "A", "B")])
def test_the_ordered_pair_is_the_pair_the_emission_is_gated_on(pgrA, pgrB, thick, thin):
    """The ordering has to hold in the emitted assembly, not just in the helper.

    Only the thick tensor's early fill carries a DcpEarlyFill label, and that
    label is what the wait pass finds and relaxes. So asserting which tensor's
    gate moves proves the emission followed the helper: a site that reverted to
    a positional A-then-B reading would swap thick and thin, and both arms would
    fail here. Matching on the source text instead bound this to local-variable
    spelling and let a semantic break through.
    """
    assert _issueOrder(pgrA, pgrB) == (thick, thin)

    relaxed = _applyThickWait(_thickWaitKernel(pgrA, pgrB), _asm(_fillBlock(thick)))
    assert relaxed.count("s_wait_tensorcnt 2") == 1, \
        "the thick tensor named by the helper is not the one whose gate was relaxed"

    assert _thickWaitRejects(_thickWaitKernel(pgrA, pgrB), _asm(_fillBlock(thin), tail=["s_endpgm"])) == DCP_THICK_WAIT_UNCOVERED


# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Auto-selection of the per-tensor PrefetchGlobalReadA/B pair.

Every test here fails against the code as it stood at 5d43341bd8. That is the
point of the file: the defects it covers are ones where the code looked correct
and quietly did nothing, so a test that only passes after the fix would not have
caught them.

The central one: auto-selection runs before MacroTile exists, and standard
benchmark processing has already replaced the nine-item MatrixInstruction with
its four-item form by then. The tile could not be reconstructed, and the search
returned its first candidate without evaluating LDS even once.
"""


_F8F4 = {
    "MacDataTypeA": DataType("F8"),
    "MacDataTypeB": DataType("F4"),
    "MXBlockA": 32,
    "MXBlockB": 32,
}


def _postConversionState(**overrides):
    """The state shape auto-selection actually sees on the normal tuning path.

    BenchmarkProblems calls matrixInstructionToMIParameters before it constructs
    the Solution, so MatrixInstruction is already the FOUR-item form and the
    geometry lives in MIBlock / MIWaveTile / MIWaveGroup. MacroTile0/1 do not
    exist yet: they are assigned in assignProblemIndependentDerivedParameters,
    which assignDerivedParameters calls *after* it resolves the auto pair.

    test_fixture_matches_the_real_conversion pins this against the real
    converter so the fixture cannot drift away from what production produces.
    """
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
# The headline defect: no LDS search happened at all on the normal path.
# ---------------------------------------------------------------------------
def test_four_item_matrix_instruction_still_yields_a_tile():
    """Without this there is nothing to size the search with.

    Before the fix _macroTileFromState read MacroTile0/1 (absent) and then
    MatrixInstruction (four items, so the nine-item helper returned None), and
    so returned None on every ordinary solution.
    """
    assert _macroTileFromState(_postConversionState()) == (64, 256)


def test_auto_selection_evaluates_every_candidate_on_the_normal_path(monkeypatch):
    """The search must actually measure LDS, not return candidates[0] unmeasured."""
    import Tensile.Components.DecouplePGR as dcp

    seen = []
    real = dcp.decouplePGRLdsBytesEstimate

    def spy(ks, problemType=None):
        seen.append((ks.get("PrefetchGlobalReadA"), ks.get("PrefetchGlobalReadB")))
        return real(ks, problemType)

    monkeypatch.setattr(dcp, "decouplePGRLdsBytesEstimate", spy)
    selected = dcp.pgrAutoPairSelectMaxLds(2, _postConversionState(), _F8F4)
    assert seen == [(2, 2), (2, 1), (1, 2)]
    assert selected == (2, 2)


def test_auto_selection_rejects_when_no_candidate_fits_in_lds():
    """The whole point of ranking by LDS: a cap nothing fits under must reject.

    Before the fix this returned (2, 2) -- the first candidate -- and the
    solution was carried all the way to the real LDS check before dying, having
    never considered the divergent pairs that are the reason auto exists.
    """
    assert pgrAutoPairSelectMaxLds(2, _postConversionState(MaxLDS=1024), _F8F4) is None


def test_auto_selection_steps_down_to_a_divergent_pair_when_the_equal_pair_will_not_fit():
    """This is the tile unlock, and it never ran before the fix."""
    state = _postConversionState()
    probe = dict(state, MacroTile0=64, MacroTile1=256)
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = 2, 2
    equalPairLds = decouplePGRLdsBytesEstimate(probe, _F8F4)
    selected = pgrAutoPairSelectMaxLds(2, _postConversionState(MaxLDS=equalPairLds - 1), _F8F4)
    assert selected is not None and selected[0] != selected[1]
    assert ldsBlocksForPgrLevel(selected[0]) != ldsBlocksForPgrLevel(selected[1])


def test_auto_selection_rejects_when_the_tile_cannot_be_derived():
    """No geometry means no search. Accepting a candidate here is a lie."""
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
    """mi[0]*mi[5]*mi[7], mi[1]*mi[6]*mi[8] is not the tile when MatrixInstB > 1.

    MIBlockBM absorbs the blocks first and MIWaveGroup follows that, so the
    shortcut always came out too small -- which under-reports LDS, and an
    under-reported footprint lets a pair that does not fit win the ranking.
    """
    assert shortcut != derived, "fixture must describe a case the shortcut got wrong"
    assert macroTileFromMatrixInstruction(mi, 32) == derived


def test_state_geometry_beats_a_stale_nine_item_matrix_instruction():
    state = _postConversionState(
        MatrixInstruction=[32, 32, 1, 2, 1, 4, 1, 2, 2],
        MIBlock=[32, 32, 1, 2, 2, 1], MIWaveTile=[4, 1], MIWaveGroup=[1, 4])
    assert _macroTileFromState(state) == (256, 128)


# ---------------------------------------------------------------------------
# Level 0 in a divergent pair: one block, one scalar, one instruction stream.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pgrA, pgrB", [(0, 2), (2, 0), (0, 3), (3, 0)])
def test_divergent_level_zero_is_rejected(pgrA, pgrB):
    """0 and 1 both map to one LDS block and pin the same scalar, and no
    consumer reads the raw level, so (0, N) emits (1, N)'s instructions under a
    different kernel name. Measured byte-identical. Reject until a real
    per-tensor cadence exists."""
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
    """It rejected before too, but as "no LDS-feasible pair" -- which reads as a
    sizing problem on a solution whose sizing is fine. The candidate space simply
    has no zero in it, and zero is unsupported in a divergent pair. Say that, and
    say it the same way an explicit (0, N) is refused."""
    state = _postConversionState(PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    reason = resolvePrefetchGlobalReadSpecialValues(state)
    assert reason is not None
    assert "PrefetchGlobalRead%s=0" % heldTc in reason
    assert "level 0" in reason
    assert "no LDS-feasible pair" not in reason


# ---------------------------------------------------------------------------
# Legality is a filter, not a post-hoc rejection.
# ---------------------------------------------------------------------------
def test_candidates_above_two_blocks_never_enter_the_ranking():
    # Imported here rather than at module scope so that running this file
    # against a build without the legality filter reports each test's own
    # failure instead of one collection error for the whole module.
    from Tensile.Components.DecouplePGR import autoPairCandidateIsLegal

    legal = [p for p in pgrAutoPairCandidates(4) if autoPairCandidateIsLegal(*p)]
    assert (3, 2) not in legal and (2, 3) not in legal and (4, 3) not in legal
    assert (2, 2) in legal and (2, 1) in legal and (1, 2) in legal
    assert (3, 3) in legal and (4, 4) in legal, "equal pairs degenerate to scalar"


def test_an_illegal_pair_cannot_outrank_a_legal_one_that_fits():
    """(3, 2) used to win on LDS, get rejected downstream for exceeding two
    blocks, and strand a legal (2, 2) that fitted under the same cap. Nothing
    retried, so the solution was lost outright."""
    state = _postConversionState(PrefetchGlobalRead=3)
    lds = {}
    for pair in pgrAutoPairCandidates(3):
        probe = dict(state, MacroTile0=64, MacroTile1=256)
        probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = pair
        lds[pair] = decouplePGRLdsBytesEstimate(probe, _F8F4)
    # A cap that admits the illegal (3, 2) and the legal (2, 2), but not (3, 3).
    cap = lds[(3, 2)]
    assert lds[(2, 2)] <= cap < lds[(3, 3)], "fixture must reproduce the stranding band"
    selected = pgrAutoPairSelectMaxLds(3, _postConversionState(PrefetchGlobalRead=3, MaxLDS=cap), _F8F4)
    assert selected == (2, 2)
    blocks = (ldsBlocksForPgrLevel(selected[0]), ldsBlocksForPgrLevel(selected[1]))
    assert max(blocks) <= 2


# ---------------------------------------------------------------------------
# Keep the hand-built fixture honest against the real converter.
# ---------------------------------------------------------------------------
def test_fixture_matches_the_real_conversion():
    """If matrixInstructionToMIParameters ever stops producing this shape, the
    tests above would be exercising a state production never builds."""
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


# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for `decoupledThickGateRelaxation` -- the single owner of whether a
decoupled PGR pair gets a relaxed thick-tensor gate, and of the count it reads.

The invariant these pin, and the reason the function exists: the *presence* of a
relaxation follows from the resolved PGR pair alone and must be identical under
every descriptor grouping, because the thick/thin LDS-block asymmetry that makes
an early start legal is a property of the pair and the grouping cannot create or
remove it. What the grouping decides is the mechanism and the count, because the
legal count is how many independent tensor ops the grouping actually leaves
outstanding -- `s_wait_tensorcnt N` retires while N remain, so 0 is a full drain
and a larger N is a weaker gate. That is not a tunable: raising it past what the
grouping supports starts LDS reads before their data has landed.

Consequence worth stating because it has been misread: counting
`s_wait_tensorcnt 1` is a separate-descriptor-specific proxy. The shared
grouping's gate reads 2 -- the weaker of the two -- so a census of "wait1"
reports the shared grouping as unrelaxed when it is in fact relaxed further.

The cross-product runs over the rows of TDM_GROUPS rather than over the TDMFuse
integers. The integer is a stable-but-arbitrary index into that table, so a test
keyed on it cannot see the rows no integer selects yet.
"""


def ks(fuse=0, pgrA=2, pgrB=1, **over):
    """A solution state that satisfies every guard the grouping consults.

    Deliberately complete: a state missing TDMInst or the MX scales would be
    declined by `_tdmFuseCanShareDescriptors` and fall back to the default
    grouping, which is the very case `test_declined_fusion_*` isolates on
    purpose. Starting from a passing state keeps those tests honest.
    """
    state = {
        "PrefetchGlobalRead": 1,
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "NumWaves": 4,
        "TDMCross": 0,
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


# Expected answer per grouping row. Written out rather than derived from the row
# so the table and the predicate cannot agree by sharing a bug.
ROW_SEPARATES_AB = {
    "MX_AB": False,   # {A,B} + {MXSA,MXSB}
    "AB": False,      # {A,B} + {MXSA} + {MXSB}
    "paired": True,   # {A,MXSA} + {MXSB,B}
    "A_MX": True,     # {A,MXSA,MXSB} + {B}
    "B_MX": True,     # {B,MXSA,MXSB} + {A}
    "None": True,     # nothing fused, so every tensor is its own set
}


def shape_of(state):
    decoupled, blkA, blkB = decouplePGRBlocks(state)
    if not decoupled:
        return "scalar"
    return "divergent" if blkA != blkB else "equal"


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
    """Presence follows the pair; mechanism and count follow the row."""
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    gate = decoupledThickGateRelaxation(ks(pgrA=pgrA, pgrB=pgrB))
    if shape != "divergent":
        assert gate is None
        return
    assert gate is not None, (
        "a divergent pair earns a relaxation under every grouping; the %s row "
        "returned None, which is the fuse-gated-presence bug" % name)
    if ROW_SEPARATES_AB[name]:
        assert gate.mechanism == DCP_THICK_GATE_TOKENS
    else:
        assert gate.mechanism == DCP_THICK_GATE_TEXT
    assert gate.tensorcnt == DCP_THICK_GATE_SUPPORTED[gate.mechanism]


@pytest.mark.parametrize("pgrA,pgrB,shape", PAIRS)
def test_presence_is_identical_across_the_reachable_fuse_levels(pgrA, pgrB, shape):
    """No TDMFuse integer may add or remove a relaxation.

    This is the mutation guard. Re-gating *presence* on a fuse integer -- the
    defect the owner exists to prevent -- makes the divergent rows disagree here
    and fails loudly.
    """
    present = {
        fuse: decoupledThickGateRelaxation(ks(fuse=fuse, pgrA=pgrA, pgrB=pgrB)) is not None
        for fuse in sorted(TDM_FUSE_GROUPING)
    }
    assert len(set(present.values())) == 1, (
        "presence differs by TDMFuse: %s" % present)
    assert next(iter(present.values())) is (shape == "divergent")


@pytest.mark.parametrize("fuse,expected", sorted(TDM_FUSE_GROUPING.items()))
def test_reachable_fuse_integers_resolve_to_their_row(fuse, expected):
    """Pins the integer-to-row mapping in one place, so nothing else reads it."""
    assert tdmGrouping(ks(fuse=fuse)).name == expected


# --- the guard is now where the decision is --------------------------------
#
# `_tdmFuseCanShareDescriptors` can decline the grouping TDMFuse names and fall
# back to the default, where A and B share one descriptor set. When the emission
# sites tested `TDMFuse == 1` themselves, a declined solution was handed disjoint
# tokens for descriptors that are in fact shared: the guard was not where the
# decision was, and it failed open. These cases are unreachable in a surviving
# solution today -- Solution.py rejects each of them -- so they are an invariant
# check, and the one that catches the divergence if a future change reopens it.
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
    """The divergence catcher for the relocated guard."""
    gate = decoupledThickGateRelaxation(ks(fuse=1, **override))
    assert gate is not None, why
    assert gate.mechanism == DCP_THICK_GATE_TEXT, (
        "TDMFuse=1 declined on %s shares one descriptor set, so its tensor ops "
        "share a token and cannot be drained independently; handing it "
        "DCP_THICK_GATE_TOKENS would gate reads on data that has not landed" % why)


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
@pytest.mark.parametrize("why,override", DECLINES + (("(none)", {}),))
def test_mechanism_agrees_with_the_grouping_by_construction(fuse, why, override):
    """The owner and the grouping cannot disagree, whatever the guards say."""
    state = ks(fuse=fuse, **override)
    gate = decoupledThickGateRelaxation(state)
    if gate is None:
        return
    separates = tdmSeparateABDescriptors(state)
    assert (gate.mechanism == DCP_THICK_GATE_TOKENS) is separates, why


def test_token_path_requires_a_second_token_stream():
    """A separate-descriptor pair the TDM does not move on both tensors, or whose
    thin side is not single-buffered, has no second stream to skip past."""
    assert decoupledThickGateRelaxation(ks(fuse=1, enableTDMA=False)) is None
    assert decoupledThickGateRelaxation(ks(fuse=1, enableTDMB=False)) is None
    assert decoupledThickGateRelaxation(ks(fuse=1, pgrA=3, pgrB=2)) is None


def test_supported_counts_match_the_documented_mechanisms():
    assert DCP_THICK_GATE_SUPPORTED == {DCP_THICK_GATE_TEXT: 2,
                                        DCP_THICK_GATE_TOKENS: 1}


MARKER = "DcpEarlyFillA"          # ks() is a 2/1 pair, so A is the thick side


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


def uncovered(schedule, relaxed=DCP_THICK_GATE_SUPPORTED[DCP_THICK_GATE_TEXT]):
    """The verdict on one schedule, after the pass has done its retagging.

    Retagging every full drain stands in for the real loop here: in these
    shapes each one is some site's own gate, and passing the rewritten indices
    is what lets the check tell a gate this pass relaxed from one that merely
    reads the same number.
    """
    lines = asm(*schedule).splitlines(keepends=True)
    retagged = set()
    for n, line in enumerate(lines):
        swapped = re.sub(r"^(s_wait_tensorcnt\s+)0(\s|$)",
                         r"\g<1>%d\2" % relaxed, line)
        if swapped != line:
            lines[n] = swapped
            retagged.add(n)
    return DP.dcpThickGateUncoveredSites(lines, MARKER, relaxed, retagged)


@pytest.mark.parametrize("why,schedule", [("per-site", PER_SITE),
                                          ("merged", MERGED),
                                          ("three-site", THREE_SITE)])
def test_every_schedule_the_compiler_emits_is_covered(why, schedule):
    assert uncovered(schedule) == []


def test_reads_ahead_of_the_gate_are_refused():
    """The hazard the gate exists for, and the one shape none of the three has.

    A fill end whose LDS reads precede its drain is reading data that may not
    have landed. That is worth refusing to emit, and it is what the check is
    for now that the count is gone.
    """
    bad = (site(CLONE, READS, GATE),)
    assert [why.split(" at line")[0] for _, why in uncovered(bad)] == [
        "reaches ds_load_b128"]


def test_a_gate_the_pass_did_not_write_is_refused():
    """Still fail-closed, just about the right thing.

    Reaching a gate this pass did not rewrite means the emitted shape is not
    the one it assumes, so it may not claim the site is relaxed -- and this
    holds even when that gate already reads the relaxed count, because the next
    gate down drains the thin tensor into the block its reads are about to
    touch.
    """
    foreign = (site(CLONE, ["s_wait_tensorcnt 1"], READS),)
    assert [why.split(" at line")[0] for _, why in uncovered(foreign)] == [
        "first gate is s_wait_tensorcnt 1"]


def test_a_header_copy_with_nothing_below_it_is_covered_vacuously():
    """A loop copy can emit a fill end with no gate and no reads after it.
    There is nothing to hold back, so there is nothing to refuse -- and the
    pre-existing test_thick_wait_ignores_header_copies_* says the same thing
    about the pass itself.
    """
    assert uncovered((site(CLONE, GATE, READS), site(MAIN, MATH))) == []


def test_no_fill_label_at_all_is_refused():
    assert [why for _, why in uncovered((["label_LoopBeginL:"] + FILL,))] == [
        "no DcpEarlyFillA label was emitted at all"]


@pytest.mark.parametrize("initCIterWmma", [0, 1])
def test_acceptance_does_not_depend_on_how_many_waits_were_retagged(
        initCIterWmma):
    """The regression. Fails against the check this replaced.

    Both schedules are correct and the pass relaxes both, but it retags two
    waits in one and one in the other, and InitCIterWmma is 1 either way. Any
    check derived from that parameter -- which is what `expected` was -- has to
    reject one of these, so reintroducing one fails here. Parametrising the
    parameter is the point: the verdict may not move with it.

    `self` is unused on the text path, so the pass is driven unbound rather
    than standing up a KernelWriter.
    """
    from Tensile.KernelWriter import KernelWriter

    solution = ks(InitCIterWmma=initCIterWmma)
    gate = decoupledThickGateRelaxation(solution)
    assert gate.mechanism == DCP_THICK_GATE_TEXT
    relaxed = "s_wait_tensorcnt %d" % gate.tensorcnt

    perSite = KernelWriter._dcpApplyThickWait1(None, solution, asm(*PER_SITE))
    merged = KernelWriter._dcpApplyThickWait1(None, solution, asm(*MERGED))

    assert (perSite.count(relaxed), merged.count(relaxed)) == (2, 1)
    assert "s_wait_tensorcnt 0" not in perSite + merged
