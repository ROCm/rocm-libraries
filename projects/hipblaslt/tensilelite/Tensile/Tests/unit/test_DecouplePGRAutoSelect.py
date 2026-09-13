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

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Components.DecouplePGR import (
    decouplePGRLdsBytesEstimate,
    divergentPairUnsupportedReason,
    ldsBlocksForPgrLevel,
    macroTileFromMatrixInstruction,
    pgrAutoPairCandidates,
    pgrAutoPairSelectMaxLds,
    resolvePrefetchGlobalReadSpecialValues,
    _macroTileFromState,
)

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


# ---------------------------------------------------------------------------
# The tile helper itself: MatrixInstB distribution was ignored.
# ---------------------------------------------------------------------------
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