# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise paired-placement policy through real gfx1250 Solution derivation."""

import pytest

from tensilelite.SolutionStructs.Solution import Solution
from test_halfplr_streamk_rejects import _gp_gfx1250, _make_params  # noqa: F401

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


@pytest.mark.parametrize("tile", [4, 8])
@pytest.mark.parametrize("pad", [0, 16])
@pytest.mark.parametrize("rotate", [0, 1])
@pytest.mark.parametrize("pgr", [1, 2])
def test_mxsa_m_halves_separate_both_buffer_phases(derive, tile, pad, rotate, pgr):
    from tensilelite.SolutionStructs.lds_layout import format_lds_layout
    from tensilelite.SolutionStructs.segment_interleave import SEG, paired_buffer_layout_reason

    sol, out = derive(1, mi=[16, 16, 128, 1, 1, tile, tile, 2, 2],
                      VectorWidthA=tile, VectorWidthB=tile // 2,
                      HalfPLR=1 if tile == 8 else 0, PrefetchGlobalRead=pgr,
                      LDSSegmentBRotate=rotate, LdsPadMXSA=pad,
                      LdsBlockSizePerPadMXSA=256 if pad else 0,
                      ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    o = sol["LDSSegInterleaveOffsets"]
    assert o["mxsSplitA"] is True
    assert sol["MIWaveGroup"] == [2, 2]
    assert bool(o.get("pairedBuffers")) is (tile == 8)
    stride = o.get("bufferStrideMXSA", sol["LdsOffsetA_Blk"])
    for phase in (0, 1):
        starts = [o["ldsBaseMXSA"] + half * o["mxsaStrideBytes"] + phase * stride
                  for half in (0, 1)]
        ranges = [set(range(start // SEG, (start + o["mxsaHalfFootprintBytes"] - 1) // SEG + 1))
                  for start in starts]
        assert ranges[0].isdisjoint(ranges[1])
        assert max(starts) + o["mxsaHalfBytes"] <= sol["LdsNumBytes"]
    if tile == 8:
        assert paired_buffer_layout_reason(sol._state, o) is None
        assert sol["LdsNumBytes"] == (274432 if pad else 273408)
        assert o["mxsaStrideBytes"] == (135168 if pad else 134656)
        if rotate:
            assert o["readStrideBytesB"] == 66304
    text = format_lds_layout(sol._state, "MXSA split")
    assert "MXSA0 ping" in text and "MXSA1 pong" in text
    assert "MXSA ping" not in text


@pytest.mark.parametrize("pad", [0, 16])
def test_mxsa_auto_compares_final_split_layout_with_ordinary(derive, pad):
    overrides = dict(HalfPLR=1, LdsPadMXSA=pad, LdsBlockSizePerPadMXSA=256 if pad else 0,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    forced, out = derive(1, **overrides)
    assert forced["Valid"], out
    assert forced["LDSSegInterleaveOffsets"]["mxsSplitA"] is True
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    if pad:
        assert forced["LdsNumBytes"] == off["LdsNumBytes"] + 512
        # Discard the complete candidate, rather than keeping unsplit paired.
        assert auto["LDSSegmentInterleave"] == 0
        assert _lds_configuration(auto) == _lds_configuration(off)
    else:
        assert forced["LdsNumBytes"] == off["LdsNumBytes"]
        assert auto["LDSSegmentInterleave"] == 1
        assert _lds_configuration(auto) == _lds_configuration(forced)


def test_mxsa_split_budget_is_checked_before_committing_paired(derive):
    # Unsplit paired and ordinary both need 273920 B; splitting needs 274432 B.
    overrides = dict(HalfPLR=1, MaxLDS=273920, LdsPadMXSA=16,
                     LdsBlockSizePerPadMXSA=256,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    for mode, rotate in ((1, 0), (-1, 1)):
        sol, out = derive(mode, LDSSegmentBRotate=rotate, **overrides)
        assert not sol["Valid"]
        assert "requires paired LDS placement: layout exceeds MaxLDS" in out


@pytest.mark.parametrize("pad", [0, 16])
def test_mxsa_nonpaired_tight_split_reuses_trailing_scale_slot(derive, monkeypatch, pad):
    import importlib
    module = importlib.import_module("tensilelite.SolutionStructs.segment_interleave")
    overrides = dict(StreamK=3, GlobalSplitU=0, HalfPLR=1,
                     LdsPadMXSA=pad, LdsBlockSizePerPadMXSA=256 if pad else 0,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    split, out = derive(1, **overrides)
    assert split["Valid"], out
    o = split["LDSSegInterleaveOffsets"]
    assert o["mxsSplitA"] is True and not o.get("pairedBuffers")
    assert o["mxsaHalfBytes"] == (1088 if pad else 1024)
    monkeypatch.setattr(module, "_mxsa_half_layout", lambda *args, **kwargs: None)
    original, out = derive(1, **overrides)
    assert original["Valid"], out
    old = original["LDSSegInterleaveOffsets"]
    assert o["writeStrideBytes"] == old["writeStrideBytes"] + o["mxsaHalfBytes"]
    assert old["ldsBaseMXSB"] - o["ldsBaseMXSB"] == (128 if pad else 0)
    assert split["LdsNumBytes"] == original["LdsNumBytes"]


@pytest.mark.parametrize("paired,slot", [(False, 1088), (True, 1280)])
def test_mxsa_half_slot_does_not_depend_on_full_buffer_alignment(derive, paired, slot):
    from tensilelite.SolutionStructs.segment_interleave import (
        _mxsa_half_layout, evaluate, paired_buffer_offsets,
    )
    sol, out = derive(1, HalfPLR=1, LdsPadMXSA=16, LdsBlockSizePerPadMXSA=256,
                      ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    state = dict(sol._state)
    expected = dict(mxsSplitA=True, mxsaHalfBytes=slot, mxsaHalfFootprintBytes=1088)
    placement = paired_buffer_offsets(state) if paired else evaluate(state)
    for fullSize in (0, 2304, 8192, None):
        if fullSize is None:
            state.pop("LdsNumElementsAlignedMXSA", None)
        else:
            state["LdsNumElementsAlignedMXSA"] = fullSize
        assert _mxsa_half_layout(state, paired) == expected
        assert (paired_buffer_offsets(state) if paired else evaluate(state)) == placement


@pytest.mark.parametrize("pad_b,pad_mxsa,span,auto_interleave", [
    (0, 0, 65536, 1),
    (16, 16, 65856, 0),
])
def test_mxsa_slot_can_make_the_complete_component_tight(
    derive, pad_b, pad_mxsa, span, auto_interleave
):
    from tensilelite.SolutionStructs.segment_interleave import evaluate, SEG

    overrides = dict(mi=[16, 16, 128, 1, 1, 8, 4, 2, 2],
                     VectorWidthA=8, VectorWidthB=2, DepthU=256, HalfPLR=0,
                     TDMIterateMode=0,
                     LdsPadA=240, LdsBlockSizePerPadA=512,
                     LdsPadB=pad_b, LdsBlockSizePerPadB=1024,
                     LdsPadMXSA=pad_mxsa, LdsBlockSizePerPadMXSA=256 if pad_mxsa else 0,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    sol, out = derive(1, **overrides)
    assert sol["Valid"], out
    # A0+B0 alone is below 64 KiB; MXSA0 reaches or crosses the boundary.
    assert 48128 + 16384 + 16 * pad_b < SEG
    layout = evaluate(sol._state)
    assert layout["aligned"] is False
    assert layout["offsets"]["writeStrideBytes"] == span
    assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True
    assert sol["LDSSegInterleaveOffsets"]["mxsSplitA"] is True
    # Auto still compares the final paired reservation with ordinary once.
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    assert auto["LDSSegmentInterleave"] == auto_interleave
    assert _lds_configuration(auto) == _lds_configuration(sol if auto_interleave else off)


@pytest.mark.parametrize("overrides", [
    {"MIWaveGroup": [4, 1]}, {"TDMFuse": 1}, {"TDMSplit": True},
    {"MXScaleFormat": "NoSwizzle"}, {"DirectToVgprMXSA": True},
    {"ProblemType": {"MXBlockFreeA": 128}},
    {"LdsBlockSizePerPadMXSA": 768},
])
def test_mxsa_split_leaves_incompatible_layouts_unchanged(derive, overrides):
    from copy import deepcopy
    from tensilelite.SolutionStructs.segment_interleave import _mxsa_half_layout
    sol, out = derive(1, HalfPLR=1, LdsPadMXSA=16, LdsBlockSizePerPadMXSA=256,
                      ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    state = deepcopy(sol._state)
    overrides = deepcopy(overrides)
    state["ProblemType"].state.update(overrides.pop("ProblemType", {}))
    state.update(overrides)
    before = deepcopy(state["LDSSegInterleaveOffsets"])
    assert _mxsa_half_layout(state) is None
    assert state["LDSSegInterleaveOffsets"] == before


@pytest.mark.parametrize("pad", [0, 16])
@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("streamk", [0, 3])
@pytest.mark.parametrize("mode", [-1, 1])
def test_mxsa_split_with_whole_b(derive, pad, pgr, streamk, mode):
    from tensilelite.SolutionStructs.segment_interleave import SEG, paired_buffer_layout_reason

    overrides = dict(mi=[16, 16, 128, 1, 1, 8, 6, 2, 2],
                     VectorWidthA=8, VectorWidthB=2, HalfPLR=1,
                     StreamK=streamk, GlobalSplitU=0 if streamk else 1,
                     PrefetchGlobalRead=pgr, LdsPadMXSA=pad,
                     LdsBlockSizePerPadMXSA=256 if pad else 0,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    sol, out = derive(mode, **overrides)
    assert sol["Valid"], out
    o = sol["LDSSegInterleaveOffsets"]
    assert o["bBaseline"] and o["mxsSplitA"]
    assert bool(o.get("pairedBuffers")) is (streamk == 0)
    stride = o.get("bufferStrideMXSA", sol["LdsOffsetA_Blk"])
    for phase in (0, 1):
        start = o["ldsBaseMXSA"] + phase * stride
        other = start + o["mxsaStrideBytes"]
        first = set(range(start // SEG, (start + o["mxsaHalfFootprintBytes"] - 1) // SEG + 1))
        second = set(range(other // SEG, (other + o["mxsaHalfFootprintBytes"] - 1) // SEG + 1))
        assert first.isdisjoint(second)
        assert other + o["mxsaHalfBytes"] <= sol["LdsNumBytes"]
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    if streamk:
        assert sol["LdsNumBytes"] == off["LdsNumBytes"]
    else:
        assert paired_buffer_layout_reason(sol._state, o) is None
        assert sol["LdsNumBytes"] == (241664 if pad else 240640)
        # The final split can grow beyond old paired while still fitting ordinary.
        assert sol["LdsNumBytes"] < off["LdsNumBytes"]


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


def _rectangular_mxf4(derive, mode=1):
    # SourceSwap turns MI32x16 into effective M=16, N=32. B's two
    # 64-column instruction subtiles together fill one 128-column component.
    sol, out = derive(mode, mi=[32, 16, 128, 1, 1, 8, 4, 2, 2],
                      SourceSwap=True, VectorWidthA=8, VectorWidthB=4,
                      DepthU=512, HalfPLR=1, ProblemType={
                          "DataType": "F4", "DestDataType": "F8",
                          "MXBlockA": 32, "MXBlockB": 32,
                      })
    assert sol["Valid"], out
    return sol


@pytest.mark.parametrize("mode", [-1, 1])
def test_rectangular_mxf4_splits_b_without_growing_lds(derive, mode):
    sol = _rectangular_mxf4(derive, mode)
    assert (sol["MatrixInstM"], sol["MatrixInstN"]) == (16, 32)
    assert (sol["MacroTile0"], sol["MacroTile1"]) == (256, 256)
    o = sol["LDSSegInterleaveOffsets"]
    assert o["pairedBuffers"] and o["mxsSplitA"]
    assert not o.get("bBaseline", False)
    assert o["bufferStrideB"] == 33280
    assert o["ldsBaseB"] == 66048
    assert o["writeStrideBytes"] == 137216
    assert sol["LdsNumBytes"] == 282624


def test_rectangular_mxf4_keeps_incomplete_b_groups_whole(derive):
    # N=192 has 96 columns/half, which a 64-column vIdx group cannot divide.
    sol, out = derive(1, mi=[32, 16, 128, 1, 1, 8, 3, 2, 2],
                      VectorWidthB=1, DepthU=512, HalfPLR=1, ProblemType={
                          "DataType": "F4", "DestDataType": "F8",
                          "MXBlockA": 32, "MXBlockB": 32,
                      })
    assert sol["Valid"], out
    assert sol["MacroTile1"] == 192
    o = sol["LDSSegInterleaveOffsets"]
    assert o["bBaseline"] and o["pairedBuffers"]


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


def _derive_shipped_streamk(derive, mode, mx=True, **overrides):
    # Match the MXFP8 StreamK geometry shipped in the gfx1250 Origami logic.
    scales = {"MXBlockA": 32, "MXBlockB": 32} if mx else {"DataType": "F8"}
    return derive(mode, mi=[16, 16, 128, 1, 1, 1, 5, 2, 2],
                  StreamK=3, GlobalSplitU=0, SourceSwap=False, HalfPLR=0,
                  VectorWidthA=1, VectorWidthB=1, DepthU=512,
                  PrefetchGlobalRead=2,
                  ProblemType={"DestDataType": "s", **scales}, **overrides)


@pytest.mark.parametrize("mode,mx,shadow_init,paired", [
    pytest.param(-1, True, True, False, id="mx-shadow-init-auto"),
    pytest.param(1, True, True, False, id="mx-shadow-init-forced"),
    pytest.param(-1, True, False, True, id="mx-no-shadow-init"),
    pytest.param(-1, False, True, True, id="fp8-shadow-init"),
])
def test_streamk_pairs_unless_mx_with_shadow_init(derive, mode, mx, shadow_init, paired):
    sol, out = _derive_shipped_streamk(derive, mode, mx, ForceDisableShadowInit=not shadow_init)
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["NumLdsBlk"] == 2
    offsets = sol["LDSSegInterleaveOffsets"]
    if paired:
        assert offsets["pairedBuffers"] is True
        assert sol["StoreSwapAddr"] is True
        assert offsets["pairedUsedBytes"] <= sol["LdsNumBytes"] <= sol["MaxLDS"]
    else:
        # The paired swap masks would push these kernels past the SGPR limit.
        # Keep unpaired buffers, with MXSA M halves inside their components.
        assert offsets == {
            "ldsBaseB": 8704, "writeStrideBytes": 93184, "footprintPacked": True,
            "bBaseline": True, "ldsBaseMXSA": 8448, "ldsBaseMXSB": 101888,
            "mxsSplitA": True, "mxsaHalfBytes": 256,
            "mxsaHalfFootprintBytes": 256, "mxsaStrideBytes": 93184,
        }
        assert sol["StoreSwapAddr"] is False
        assert sol["LdsNumBytes"] == 235520


@pytest.mark.parametrize("mode", [-1, 1])
@pytest.mark.parametrize("mx", [False, True], ids=["fp8", "mxfp8"])
@pytest.mark.parametrize("strategy", ["StreamK", "DataParallel"])
def test_persistent_prefetch_keeps_unpaired_interleave(derive, mode, mx, strategy):
    # Disable shadow init so MX kernels exercise the pairing attempt too.
    sol, out = _derive_shipped_streamk(
        derive, mode, mx, ForceDisableShadowInit=True, PrefetchAcrossPersistent=1,
        StreamKForceDPOnly=int(strategy == "DataParallel"))
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["TileProcessingStrategy"] == strategy
    assert sol["NumLdsBlk"] == 2
    assert sol["StoreSwapAddr"] is False
    expected = {
        "ldsBaseB": 8448, "writeStrideBytes": 92928, "footprintPacked": True,
        "bBaseline": True,
    }
    if mx:
        expected.update(ldsBaseB=8704, writeStrideBytes=93184,
                        ldsBaseMXSA=8448, ldsBaseMXSB=101888, mxsSplitA=True,
                        mxsaHalfBytes=256, mxsaHalfFootprintBytes=256,
                        mxsaStrideBytes=93184)
    assert sol["LDSSegInterleaveOffsets"] == expected
    assert sol["LdsNumBytes"] == (235520 if mx else 232448)


@pytest.mark.parametrize("mode", [-1, 1])
def test_persistent_paired_budget_failure_keeps_unpaired_interleave(derive, monkeypatch, mode):
    import importlib
    solution_module = importlib.import_module("tensilelite.SolutionStructs.Solution")
    # The same persistent geometry can keep its existing layout if pairing exceeds MaxLDS.
    reference, out = _derive_shipped_streamk(derive, mode, mx=False, PrefetchAcrossPersistent=1)
    assert reference["Valid"], out
    monkeypatch.setattr(solution_module, "segPairedOffsets", lambda state: None)
    sol, out = _derive_shipped_streamk(derive, mode, mx=False)
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert _lds_configuration(sol) == _lds_configuration(reference)


@pytest.mark.parametrize("problem_index", [0, 1, 2], ids=["bf16", "fp8-shared-B", "mxfp8"])
def test_persistent_yaml_reaches_paired_tail_reset(problem_index):
    from pathlib import Path
    from config_harness import solutions_from_config, emit_kernels_from_config

    config = (Path(__file__).resolve().parents[1] / "common" / "streamk" /
              "gfx1250" / "core" / "paired_lds_tail.yaml")
    solutions = solutions_from_config(config, arch="gfx1250", problem_index=problem_index,
                                     expected_fork_count=4)
    # A supported YAML fork must not disappear or silently use the old layout.
    assert len(solutions) == 4
    assert {(s["TileProcessingStrategy"], s["PrefetchGlobalRead"]) for s in solutions} == {
        (strategy, pgr) for strategy in ("StreamK", "DataParallel") for pgr in (1, 2)
    }
    for sol in solutions:
        assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True
        assert sol["PrefetchAcrossPersistent"] == 0
        # MX scales require block-aligned K, but must still retain a K tail.
        assert sol["AssertSummationElementMultiple"] == (32 if problem_index == 2 else 1)
        assert sol["AssertSummationElementMultiple"] < sol["DepthU"]

    emitted = emit_kernels_from_config(config, arch="gfx1250", limit=None, canonical=False,
                                       problem_index=problem_index, expected_fork_count=4)
    assert len(emitted) == 4
    for name, source, error in emitted:
        assert error == 0, name
        instructions = [line.split("//", 1)[0].replace(",", " ").split()
                        for line in source.splitlines()]
        instructions = [inst for inst in instructions if inst]
        resets = [(instructions[i - 1], inst) for i, inst in enumerate(instructions)
                  if i and inst[0] == "s_min_u32" and inst[1].startswith("s[sgprtdm")]
        assert resets, name
        for swap, reset in resets:
            opcode, tmp, descriptor, mask = swap
            assert opcode == "s_xor_b32", swap
            assert reset[1:] == [descriptor, descriptor, tmp], reset
            is_scale = descriptor.startswith("s[sgprtdmMXS")
            expected_mask = "PairedTDMScaleSwap" if is_scale else "PairedTDMSwap"
            assert mask == f"s[sgpr{expected_mask}]", swap
        if problem_index == 2:
            assert any(reset[1].startswith("s[sgprtdmMXS") for _, reset in resets), name
