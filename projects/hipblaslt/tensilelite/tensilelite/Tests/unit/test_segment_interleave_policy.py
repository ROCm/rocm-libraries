# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise paired-placement policy through real gfx1250 Solution derivation."""

import pytest

from tensilelite.SolutionStructs.Solution import Solution
from tensilelite.Tests.unit.test_halfplr_streamk_rejects import (
    _gp_gfx1250, assembler, gfx1250_iim, _make_params,  # noqa: F401
)

pytestmark = pytest.mark.unit


@pytest.fixture
def derive(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    def run(mode, **overrides):
        params = dict(
            mi=[16, 16, 128, 1, 1, 8, 8, 2, 2],
            StreamK=0, GlobalSplitU=1, SourceSwap=True,
            VectorWidthA=8, VectorWidthB=4, TDMIterateMode=-1,
            LDSSegmentInterleave=mode, LDSSegmentBRotate=0,
        )
        params.update(overrides)
        sol = Solution(_make_params(gfx1250_iim, **params), False, True,
                       False, assembler, gfx1250_iim)
        return sol, capsys.readouterr().out
    return run


def _lds_configuration(sol):
    # Include every LDS-derived field, not just the resolved interleave flag.
    return {key: value for key, value in sol._state.items()
            if key.startswith(("Lds", "lds", "LDSSeg"))
            or key in ("NumLdsBlk", "StoreSwapAddr", "1LDSBuffer",
                       "ExpandPointerSwap", "TDMPlusLdsBuf", "DtlPlusLdsBuf")}


@pytest.mark.parametrize("mode", [-1, 1])
@pytest.mark.parametrize("rotate", [0, 1])
@pytest.mark.parametrize("pgr", [1, 2])
def test_tight_interleave_requires_paired(derive, mode, rotate, pgr):
    sol, out = derive(mode, LDSSegmentBRotate=rotate, PrefetchGlobalRead=pgr)
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True


def test_unavailable_paired_rejects_forced_but_auto_matches_off(derive):
    overrides = dict(PrefetchGlobalRead=1, HalfPLR=0, ExpandPointerSwap=True)
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    assert auto["LDSSegmentInterleave"] == 0
    assert auto["LDSSegInterleaveOffsets"] == {}
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1, **overrides)
    assert not forced["Valid"]
    assert "LDSSegmentInterleave=1 requires paired LDS placement" in out
    assert "ExpandPointerSwap is not supported" in out


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("mx_block", [16, 32])
@pytest.mark.parametrize("mode", [-1, 1])
def test_fp4_uses_paired_placement(derive, pgr, mx_block, mode):
    sol, out = derive(mode, PrefetchGlobalRead=pgr, DepthU=512, ProblemType={
        "DataType": "F4", "DestDataType": "s", "MXBlockA": mx_block, "MXBlockB": mx_block,
    })
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("a,b", [("F4", "F8"), ("F8", "F4")])
def test_mixed_fp4_retains_aligned_placement(derive, pgr, a, b):
    sol, out = derive(1, PrefetchGlobalRead=pgr, ProblemType={
        "DataType": "F8", "DestDataType": "s", "MXBlockA": 32, "MXBlockB": 32,
        "DataTypeA": a, "DataTypeB": b, "MacDataTypeA": a, "MacDataTypeB": b,
    })
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert not sol["LDSSegInterleaveOffsets"].get("pairedBuffers")


@pytest.mark.parametrize("pgr", [1, 2])
def test_auto_aligned_layout_matches_off(derive, pgr):
    overrides = dict(mi=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                     VectorWidthA=4, VectorWidthB=1, HalfPLR=0, PrefetchGlobalRead=pgr)
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1, **overrides)
    assert forced["Valid"], out
    assert not forced["LDSSegInterleaveOffsets"].get("pairedBuffers")
    assert forced["LdsNumBytes"] == 230656


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("rotate", [0, 1])
@pytest.mark.parametrize("vwb,expected", [(1, 230656), (2, 230144)])
def test_aligned_retains_component_offsets_and_separate_buffers(
    derive, monkeypatch, pgr, rotate, vwb, expected
):
    import importlib
    solution_module = importlib.import_module("tensilelite.SolutionStructs.Solution")
    monkeypatch.setattr(solution_module, "segPairedOffsets",
                        lambda state: pytest.fail("aligned placement must not be paired"))
    sol, out = derive(1, mi=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                      VectorWidthA=4, VectorWidthB=vwb, HalfPLR=0,
                      PrefetchGlobalRead=pgr, LDSSegmentBRotate=rotate)
    assert sol["Valid"], out
    assert sol["LDSSegInterleaveOffsets"] == {
        "ldsBaseB": 16640, "writeStrideBytes": 65536, "footprintPacked": True,
    }
    assert sol["StoreSwapAddr"] is False
    assert sol["NumLdsBlk"] == 2
    assert sol["LdsOffsetA_Blk"] == 131072
    assert sol["LdsNumBytes"] == expected


@pytest.mark.parametrize("tile,wg,shared", [
    ([4, 7], [2, 2], "bBaseline"),
    ([4, 4], [4, 1], "bBaseline"),
    ([4, 4], [1, 4], "aBaseline"),
])
def test_aligned_contiguous_operands_retain_their_grouping(derive, tile, wg, shared):
    from tensilelite.SolutionStructs.segment_interleave import evaluate
    mi = [16, 16, 128, 1, 1, *tile, *wg]
    sol, out = derive(1, mi=mi, DepthU=128, HalfPLR=0,
                      VectorWidthA=4, VectorWidthB=1 if tile[1] == 7 else 4)
    assert sol["Valid"], out
    assert evaluate(sol._state)["aligned"] is True
    assert sol["LDSSegInterleaveOffsets"][shared] is True
    assert not sol["LDSSegInterleaveOffsets"].get("pairedBuffers")


def test_paired_budget_failure_obeys_requested_mode(derive, monkeypatch):
    import importlib
    solution_module = importlib.import_module("tensilelite.SolutionStructs.Solution")
    off, out = derive(0)
    assert off["Valid"], out
    # Exercise the caller's failure policy independently of the chosen geometry.
    # The offset generator returns None when its placement exceeds MaxLDS.
    monkeypatch.setattr(solution_module, "segPairedOffsets", lambda state: None)
    auto, out = derive(-1)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1)
    assert not forced["Valid"]
    assert "requires paired LDS placement: layout exceeds MaxLDS" in out
    rotated, out = derive(-1, LDSSegmentBRotate=1)
    assert not rotated["Valid"]
    assert "LDSSegmentBRotate requires paired LDS placement" in out


def test_auto_cannot_commit_a_layout_that_grows_lds(derive, monkeypatch):
    import importlib
    solution_module = importlib.import_module("tensilelite.SolutionStructs.Solution")
    generate_offsets = solution_module.segPairedOffsets
    off, out = derive(0)
    assert off["Valid"], out

    def with_extra_reservation(state):
        offsets = generate_offsets(state)
        # Model additional padding that fits MaxLDS but exceeds the ordinary
        # reservation. The automatic policy must discard the entire placement.
        offsets["pairedUsedBytes"] += 16
        return offsets

    monkeypatch.setattr(solution_module, "segPairedOffsets", with_extra_reservation)
    auto, out = derive(-1)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1)
    assert forced["Valid"], out
    assert forced["LDSSegInterleaveOffsets"]["pairedBuffers"] is True
    assert forced["LdsNumBytes"] > off["LdsNumBytes"]


@pytest.mark.parametrize("mode", [-1, 1])
def test_brotate_cannot_fall_back_to_unpaired(derive, mode):
    sol, out = derive(mode, LDSSegmentBRotate=1,
                      PrefetchGlobalRead=1, HalfPLR=0, ExpandPointerSwap=True)
    assert not sol["Valid"]
    assert "LDSSegmentBRotate: needs ExpandPointerSwap=0" in out


def test_tdmsplit_remains_disabled(derive):
    sol, out = derive(1, TDMSplit=True)
    assert not sol["Valid"]
    assert "TDMSplit is currently disabled" in out


@pytest.mark.parametrize("occupancy,amax,dest,expected", [
    (40, False, "s", 265216),
    (1, False, "s", 327680),
    (40, True, "F8", 265344),
])
def test_paired_capacity_includes_only_final_layout_and_other_lds_users(
    derive, occupancy, amax, dest, expected
):
    sol, out = derive(1,
                      LDSSegmentBRotate=1, MaxOccupancy=occupancy,
                      ProblemType={"DestDataType": dest, "OutputAmaxD": amax})
    assert sol["Valid"], out
    # This tight tile needs 259 KiB for paired A/B data; occupancy and amax
    # can require more than the final data footprint.
    assert sol["LDSSegInterleaveOffsets"]["pairedUsedBytes"] == 265216
    assert sol["LdsNumBytes"] == expected


@pytest.mark.parametrize("mode", [-1, 1])
def test_streamk_retains_existing_tight_interleave(derive, mode):
    # Match the MXFP8 StreamK geometry shipped in the gfx1250 Origami logic.
    sol, out = derive(mode, mi=[16, 16, 128, 1, 1, 1, 5, 2, 2],
                      StreamK=3, GlobalSplitU=0, SourceSwap=False, HalfPLR=0,
                      VectorWidthA=1, VectorWidthB=1, DepthU=512,
                      PrefetchGlobalRead=2,
                      ProblemType={"DestDataType": "s", "MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["LDSSegInterleaveOffsets"] == {
        "ldsBaseB": 8448, "writeStrideBytes": 92928, "footprintPacked": True,
        "bBaseline": True, "ldsBaseMXSA": 101376, "ldsBaseMXSB": 101888,
    }
    assert sol["StoreSwapAddr"] is False
    assert sol["NumLdsBlk"] == 2
    assert sol["LdsNumBytes"] == 235520
