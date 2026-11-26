# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise paired-placement policy through real gfx1250 Solution derivation."""

import pytest

from tensilelite.SolutionStructs.Naming import getSolutionNameFull
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


def _check_split_mxsa_addresses(sol):
    from types import SimpleNamespace
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.SolutionStructs.segment_interleave import SEG

    offsets = sol["LDSSegInterleaveOffsets"]
    assert offsets["mxsSplitA"] and offsets["pairedBuffers"]
    regions = []
    for tc in ("A", "B", "MXSA", "MXSB"):
        components = 1 if tc == "MXSB" else 2
        component_stride = (offsets["mxsaStrideBytes"] if tc == "MXSA"
                            else offsets["writeStrideBytes"])
        size = offsets["bufferStride" + tc]
        for phase in (0, 1):
            for component in range(components):
                start = offsets["ldsBase" + tc] + component * component_stride + phase * size
                regions.append((start, start + size))
            if tc == "MXSA":
                base = offsets["ldsBaseMXSA"] + phase * size
                footprint = offsets["mxsaHalfFootprintBytes"]
                segments = [set(range((base + half * component_stride) // SEG,
                                      (base + half * component_stride + footprint - 1) // SEG + 1))
                            for half in (0, 1)]
                assert segments[0].isdisjoint(segments[1])
    regions.sort()
    assert all(left[1] <= right[0] for left, right in zip(regions, regions[1:]))
    assert regions[-1][1] == offsets["pairedUsedBytes"] <= sol["LdsNumBytes"]

    # Execute the emitted TDM address instructions for both M components.
    raw_half = sol["MacroTile0"] * sol["DepthU"] // sol["ProblemType"]["MXBlockA"] // 2
    module = KernelWriterAssembly.tdmWaveLdsAddr(
        None, sol, "MXSA", 2, raw_half, sol["LdsOffsetMXSA"],
        sol["LdsBlockSizePerPadMXSA"], sol["LdsPadMXSA"], 80, 81)
    for component in (0, 1):
        registers = {"s80": component}
        for line in str(module).splitlines():
            code = line.partition("//")[0].strip()
            if not code or code.startswith("/*"):
                continue
            op, dst, *args = code.replace(",", " ").split()
            values = [registers[arg] if arg in registers else int(arg, 0) for arg in args]
            if op == "s_mul_i32":
                registers[dst] = values[0] * values[1]
            elif op == "s_add_u32":
                registers[dst] = values[0] + values[1]
            else:
                pytest.fail(f"Unsupported TDM address instruction: {line}")
        expected = offsets["ldsBaseMXSA"] + component * offsets["mxsaStrideBytes"]
        assert registers["s80"] == expected
        for phase in (0, 1):
            actual = registers["s80"] + phase * offsets["bufferStrideMXSA"]
            assert actual + offsets["mxsaHalfFootprintBytes"] <= sol["LdsNumBytes"]

    # The reader must advance through K groups inside one M half.
    states = SimpleNamespace(inTailLoop=False, localReadDoCntA=0, localReadDoCntB=0,
                             localReadDoCntMXSA=0, localReadDoCntMXSB=0,
                             localReadDoCntMetadata=0)
    writer = SimpleNamespace(do={"LocalReadMXSA": True, "LocalReadMXSB": True}, states=states)
    for tc, rows in (("MXSA", sol["MacroTile0"] // 2), ("MXSB", sol["MacroTile1"])):
        params = dict(tensorChar=tc, localReadInstruction=SimpleNamespace(numOffsets=1),
                      localReadOffset=0)
        mx_unit = sol["MatrixInstK"] // sol["ProblemType"]["MXBlock" + tc[-1]]
        for group in range(1, sol["DepthU"] // sol["MatrixInstK"]):
            KernelWriterAssembly.localReadInc(writer, sol, 0, params)
            assert params["localReadOffset"] == group * rows * mx_unit


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
        # Keep unpaired buffers. The 64-byte M-half row keeps MXSA unsplit.
        assert offsets == {
            "ldsBaseB": 8448, "writeStrideBytes": 92928, "footprintPacked": True,
            "bBaseline": True, "ldsBaseMXSA": 101376, "ldsBaseMXSB": 101888,
        }
        assert sol["StoreSwapAddr"] is False
        assert sol["LdsNumBytes"] == 235520


@pytest.mark.parametrize("mode", [-1, 1])
@pytest.mark.parametrize("mx", [False, True], ids=["fp8", "mxfp8"])
@pytest.mark.parametrize("strategy", ["StreamK", "DataParallel"])
def test_persistent_prefetch_pairs_interleave(derive, mode, mx, strategy):
    # Disable shadow init so MX kernels exercise the pairing attempt too.
    sol, out = _derive_shipped_streamk(
        derive, mode, mx, ForceDisableShadowInit=True, PrefetchAcrossPersistent=1,
        StreamKForceDPOnly=int(strategy == "DataParallel"))
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["TileProcessingStrategy"] == strategy
    assert sol["PrefetchAcrossPersistent"] == 1
    assert sol["NumLdsBlk"] == 2
    offsets = sol["LDSSegInterleaveOffsets"]
    if mx and strategy == "StreamK":
        # StreamK MX kernels with PAP cannot afford the swap-mask SGPRs.
        assert not offsets.get("pairedBuffers")
        assert sol["StoreSwapAddr"] is False
        assert ("keeps separate ping/pong blocks: StreamK MX kernels with "
                "PrefetchAcrossPersistent" in out) is (mode == 1)
        return
    # PAP restores its LDS bank through the per-set paired masks.
    assert offsets["pairedBuffers"] is True
    assert sol["StoreSwapAddr"] is True
    assert offsets["pairedUsedBytes"] <= sol["LdsNumBytes"] <= sol["MaxLDS"]
    assert "keeps separate ping/pong blocks" not in out


@pytest.mark.parametrize("mode", [-1, 1])
def test_persistent_paired_budget_failure_keeps_unpaired_interleave(derive, monkeypatch, mode):
    import importlib
    solution_module = importlib.import_module("tensilelite.SolutionStructs.Solution")
    # The same persistent geometry keeps its unpaired layout whether pairing is
    # unsupported or exceeds MaxLDS, and an explicit request reports the fallback.
    with monkeypatch.context() as patch:
        patch.setattr(solution_module, "segPairedReason", lambda state: "unit: unsupported")
        reference, out = _derive_shipped_streamk(derive, mode, mx=False, PrefetchAcrossPersistent=1)
    assert reference["Valid"], out
    assert not reference["LDSSegInterleaveOffsets"].get("pairedBuffers")
    assert ("keeps separate ping/pong blocks: unit: unsupported" in out) is (mode == 1)
    monkeypatch.setattr(solution_module, "segPairedOffsets", lambda state: None)
    sol, out = _derive_shipped_streamk(derive, mode, mx=False, PrefetchAcrossPersistent=1)
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert _lds_configuration(sol) == _lds_configuration(reference)
    assert ("keeps separate ping/pong blocks: layout exceeds MaxLDS" in out) is (mode == 1)


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


@pytest.mark.parametrize("problem_index", [0, 1, 2], ids=["bf16", "fp8-shared-B", "mxfp8"])
def test_persistent_prefetch_yaml_moves_paired_banks_by_swap_mask(problem_index, tmp_path, capsys,
                                                                  monkeypatch):
    from pathlib import Path
    from config_harness import solutions_from_config, emit_kernels_from_config
    from tensilelite.Common.GlobalParameters import globalParameters

    # The warning is read from sys.stdout, so derive in-process: pool workers
    # write to the inherited fd and capsys never sees them.
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)

    source = (Path(__file__).resolve().parents[1] / "common" / "streamk" /
              "gfx1250" / "core" / "paired_lds_tail.yaml").read_text()
    # WorkGroupMapping=1 lets the PGR2 forks take the PAP two-group pipeline.
    config = tmp_path / "paired_lds_tail_pap.yaml"
    config.write_text(source.replace("- PrefetchAcrossPersistent: [0]",
                                     "- PrefetchAcrossPersistent: [1]\n    - WorkGroupMapping: [1]"))
    solutions = solutions_from_config(config, arch="gfx1250", problem_index=problem_index,
                                     expected_fork_count=4)
    out = capsys.readouterr().out
    assert len(solutions) == 4
    strides = {}
    for sol in solutions:
        assert sol["PrefetchAcrossPersistent"] == 1
        key = ("SK" if sol["TileProcessingStrategy"] == "StreamK" else "DP", sol["PrefetchGlobalRead"])
        offsets = sol["LDSSegInterleaveOffsets"]
        # StreamK MX kernels with PAP keep separate blocks to stay within the SGPR budget.
        assert bool(offsets.get("pairedBuffers")) is (problem_index != 2 or key[0] == "DP"), key
        if offsets.get("pairedBuffers"):
            strides[key] = {k[len("bufferStride"):]: v for k, v in offsets.items()
                            if k.startswith("bufferStride")}
    assert ("StreamK MX kernels with PrefetchAcrossPersistent cannot afford the paired "
            "swap-mask SGPRs" in out) is (problem_index == 2)

    tensors = ["A", "B"] + (["MXSA", "MXSB"] if problem_index == 2 else [])
    emitted = emit_kernels_from_config(config, arch="gfx1250", limit=None, canonical=False,
                                       problem_index=problem_index, expected_fork_count=4)
    assert len(emitted) == 4
    pipelined = False
    for name, source, error in emitted:
        assert error == 0, name
        key = ("SK" if "_TPSSK_" in source else "DP", 2 if "_PGR2_" in source else 1)
        assert f"_TPS{key[0]}_" in source and f"_PGR{key[1]}_" in source, name
        lines = []
        for line in source.splitlines():
            code, _, comment = line.partition("//")
            inst = code.replace(",", " ").split()
            if inst:
                lines.append((inst, comment.strip()))
        if key not in strides:
            assert "PairedTDMSwap" not in source, name
            continue
        stride = strides[key]
        restores = [inst for inst, comment in lines
                    if comment.endswith("to PAP bank") and inst[1].startswith("s[sgprtdm")]
        assert restores, name
        for opcode, dst, src0, mask in restores:
            expected_mask = "PairedTDMScaleSwap" if dst.startswith("s[sgprtdmMXS") else "PairedTDMSwap"
            assert (opcode, src0, mask) == ("s_xor_b32", dst, f"s[sgpr{expected_mask}]"), (name, dst)
        shifted = {}
        for inst, comment in lines:
            for tc in tensors:
                if comment == f"shift LocalReadAddr{tc} to PAP bank":
                    shifted.setdefault(tc, set()).add(inst[3])
                elif comment == f"tail {tc} paired buffer stride if the tail uses pong":
                    assert inst[3] == str(stride[tc]), (name, inst)
        assert shifted == {tc: {str(stride[tc])} for tc in tensors}, name
        assert any(inst[0] == "s_cmp_lt_u32" for inst, comment in lines
                   if comment == "PAP descriptor in its paired pong buffer?"), name
        tail = [inst for inst, comment in lines if comment == "set TDM tail write bank opposite PAP"]
        assert tail and all(inst[0] == "s_xor_b32" for inst in tail), name
        if "PAP PGR2 group0 actually issued" in source:
            pipelined = True
            selected = {comment.rsplit(" ", 1)[1] for _, comment in lines
                        if comment.startswith("PAP select group0 bank for ")}
            assert selected == set(tensors), name
    assert pipelined


@pytest.mark.parametrize("wave_tile,width", [(7, 1), (6, 1), (6, 2)])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("width_b", [2, 4, 8])
def test_contiguous_a_waves_preserve_data_writer_and_choose_native_stores(derive, wave_tile, width, source_swap, width_b):
    overrides = dict(mi=[16, 16, 128, 1, 1, wave_tile, 8, 2, 2],
                     VectorWidthA=width, VectorWidthB=width_b, HalfPLR=0,
                     SourceSwap=source_swap, StoreVectorWidth=-1,
                     StoreRemapVectorWidth=0, LDSTrInst=False,
                     LdsPadA=16, LdsPadB=16,
                     LdsBlockSizePerPadA=256 * width, LdsBlockSizePerPadB=256 * width_b,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    baseline, out = derive(0, **overrides)
    assert baseline["Valid"], out
    assert baseline["StoreVectorWidth"] == (width if source_swap else 8 * width)
    assert baseline["VectorWidthMXSA"] == width
    contiguous_a, out = derive(0, WaveTileContiguousA=True, **overrides)
    assert contiguous_a["Valid"], out
    assert _lds_configuration(baseline) == _lds_configuration(contiguous_a)
    assert contiguous_a["StoreVectorWidth"] == (1 if source_swap else 8)
    assert baseline["VectorWidthA"] == contiguous_a["VectorWidthA"] == width
    assert contiguous_a["VectorWidthMXSA"] == width
    assert contiguous_a["VectorWidthB"] == baseline["VectorWidthB"] == width_b
    assert contiguous_a["VectorWidthMXSB"] == baseline["VectorWidthMXSB"]
    interleaved, out = derive(1, WaveTileContiguousA=True, **overrides)
    assert interleaved["Valid"], out
    assert interleaved["StoreVectorWidth"] == (1 if source_swap else 8)
    offsets = interleaved["LDSSegInterleaveOffsets"]
    assert offsets["pairedBuffers"]
    assert bool(offsets.get("mxsSplitA")) is (wave_tile == 6)
    assert offsets["pairedUsedBytes"] <= interleaved["LdsNumBytes"]
    for phase in (0, 1):
        starts = [offsets["ldsBaseA"] + component * offsets["writeStrideBytes"]
                  + phase * offsets["bufferStrideA"] for component in (0, 1)]
        size = offsets["bufferStrideA"]
        segments = [set(range(start // 65536, (start + size - 1) // 65536 + 1))
                    for start in starts]
        assert segments[0].isdisjoint(segments[1])


@pytest.mark.parametrize("tile,half_row_bytes,split", [
    (5, 320, False), (6, 384, True), (7, 448, False), (8, 512, True), (10, 640, True),
])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("dtype,depth_u", [("F8", 256), ("F4", 512)])
def test_mxsa_split_preserves_128_byte_m_boundary(
        derive, monkeypatch, tile, half_row_bytes, split, source_swap, dtype, depth_u):
    import tensilelite.SolutionStructs.segment_interleave as layout

    params = dict(mi=[16, 16, 128, 1, 1, tile, 2, 2, 2],
                  SourceSwap=source_swap, VectorWidthA=1, VectorWidthB=1,
                  HalfPLR=0, DepthU=depth_u, WaveTileContiguousA=True,
                  LDSSegmentBRotate=1, TDMIterateMode=0,
                  ProblemType={"DataType": dtype, "DestDataType": "S",
                               "MXBlockA": 32, "MXBlockB": 32})
    sol, out = derive(1, **params)
    assert sol["Valid"], out
    assert sol["WaveTileContiguousA"] and sol["LDSSegmentInterleave"] == 1
    assert sol["LDSSegmentBRotate"] == 1
    offsets = sol["LDSSegInterleaveOffsets"]
    assert bool(offsets.get("mxsSplitA")) is split
    assert sol["MacroTile0"] // 2 * 4 == half_row_bytes
    if split:
        _check_split_mxsa_addresses(sol)
        for component in (0, 1):
            for phase in (0, 1):
                base = (offsets["ldsBaseMXSA"] + component * offsets["mxsaStrideBytes"]
                        + phase * offsets["bufferStrideMXSA"])
                assert base % 128 == 0
    else:
        # Fall back to the full existing layout, including A/B and both buffers.
        monkeypatch.setattr(layout, "_mxsa_half_layout", lambda *args, **kwargs: None)
        ordinary, out = derive(1, **params)
        assert ordinary["Valid"], out
        assert _lds_configuration(sol) == _lds_configuration(ordinary)


@pytest.mark.parametrize("tile,width,half_plr", [(6, -1, 1), (6, 1, 1), (6, 2, 1),
                                               (7, -1, 2), (7, 1, 2)])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("dtype,mx,depth_u,pad", [("F8", 0, 256, 16),
                                               ("F8", 32, 256, 16),
                                               ("F4", 32, 512, 32)])
def test_contiguous_a_auto_padding_matches_row_padded_writer(
        derive, tile, width, half_plr, source_swap, dtype, mx, depth_u, pad):
    overrides = dict(mi=[16, 16, 128, 1, 1, tile, 8, 2, 2],
                     SourceSwap=source_swap, VectorWidthA=width, VectorWidthB=4,
                     HalfPLR=half_plr, DepthU=depth_u, WaveTileContiguousA=True,
                     LDSSegmentBRotate=1, TDMIterateMode=0,
                     ProblemType={"DataType": dtype, "DestDataType": "S",
                                  "MXBlockA": mx, "MXBlockB": mx,
                                  "DataTypeMXSA": "E8", "DataTypeMXSB": "E8"})
    automatic, out = derive(1, **overrides)
    assert automatic["Valid"], out
    explicit, out = derive(1, LdsPadA=pad, LdsPadB=pad,
                           LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=1024,
                           **overrides)
    assert explicit["Valid"], out
    # Each A row contains 256 bytes, including the packed FP4 cases.
    assert automatic["LdsBlockSizePerPadA"] == 256
    assert automatic["LdsBlockSizePerPadB"] == 1024
    assert automatic["LdsPadA"] == automatic["LdsPadB"] == pad
    expected_width = (2 if tile == 6 else 1) if width == -1 else width
    assert automatic["VectorWidthA"] == expected_width
    if mx:
        assert automatic["VectorWidthMXSA"] == expected_width
        assert automatic["LdsPadMXSA"] == automatic["LdsBlockSizePerPadMXSA"] == 0
    assert _lds_configuration(automatic) == _lds_configuration(explicit)


@pytest.mark.parametrize("width", [1, 2])
def test_original_a_auto_padding_keeps_vector_width_stride(derive, width):
    sol, out = derive(0, mi=[16, 16, 128, 1, 1, 6, 8, 2, 2],
                      VectorWidthA=width, VectorWidthB=4, WaveTileContiguousA=False,
                      TDMIterateMode=0, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert sol["Valid"], out
    assert sol["LdsBlockSizePerPadA"] == 256 * width
    assert sol["LdsBlockSizePerPadB"] == 1024


@pytest.mark.parametrize("iterate_mode", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("tile,width,half_plr", [(6, -1, 1), (6, 1, 1), (6, 2, 1), (7, -1, 2)])
@pytest.mark.parametrize("source_swap", [False, True])
def test_contiguous_a_disables_only_a_iterate(derive, iterate_mode, tile, width, half_plr, source_swap):
    params = dict(mi=[16, 16, 128, 1, 1, tile, 8, 2, 2],
                  VectorWidthA=width, VectorWidthB=4, HalfPLR=half_plr,
                  SourceSwap=source_swap, WaveTileContiguousA=True,
                  LDSSegmentBRotate=1, ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    expected_mode = 2 if iterate_mode in (2, 3) else 0
    if expected_mode:
        params.update(LdsBlockSizePerPadB=2048, LdsPadB=16)
    sol, out = derive(1, TDMIterateMode=iterate_mode, **params)
    assert sol["Valid"], out
    assert sol["TDMIterateMode"] == expected_mode
    assert not sol.get("_TDMIterateModeA", False)
    assert bool(sol.get("_TDMIterateModeB", False)) == bool(expected_mode)
    expected_width = (2 if tile == 6 else 1) if width == -1 else width
    assert sol["VectorWidthA"] == sol["VectorWidthMXSA"] == expected_width
    assert sol["LdsBlockSizePerPadA"] == 256
    reference, out = derive(1, TDMIterateMode=expected_mode, **params)
    assert reference["Valid"], out
    assert _lds_configuration(sol) == _lds_configuration(reference)
    assert getSolutionNameFull(sol, False) == getSolutionNameFull(reference, False)


@pytest.mark.parametrize("iterate_mode", [-1, 0, 1])
def test_equal_auto_width_preserves_original_iterate_and_width_fallback(derive, iterate_mode):
    # Auto first chooses VWA2 == WT2. Non-iterate mode then reduces VWA to 1;
    # that fallback must not reactivate the contiguous layout.
    params = dict(mi=[16, 16, 128, 1, 1, 2, 2, 2, 2],
                  VectorWidthA=-1, VectorWidthB=1, DepthU=1024, HalfPLR=0,
                  SourceSwap=False, TDMIterateMode=iterate_mode,
                  ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    original, out = derive(0, WaveTileContiguousA=False, **params)
    assert original["Valid"], out
    enabled, out = derive(0, WaveTileContiguousA=True, **params)
    assert enabled["Valid"], out
    assert enabled["WaveTileContiguousA"] is False
    assert enabled["VectorWidthA"] == (1 if iterate_mode == 0 else 2)
    assert enabled._state == original._state
    assert getSolutionNameFull(enabled, False) == getSolutionNameFull(original, False)


@pytest.mark.parametrize("tile,width", [(4, 4), (8, 8), (4, -1), (8, -1)])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("dtype,mx,depth_u", [("F8", 0, 256), ("F8", 32, 256), ("F4", 32, 512)])
def test_equal_wave_tile_width_uses_original_solution(derive, tile, width, source_swap, dtype, mx, depth_u):
    params = dict(mi=[16, 16, 128, 1, 1, tile, 8, 2, 2],
                  VectorWidthA=width, VectorWidthB=4, DepthU=depth_u, HalfPLR=2,
                  SourceSwap=source_swap, TDMIterateMode=-1, LDSSegmentBRotate=1,
                  ProblemType={"DataType": dtype, "DestDataType": "S",
                               "MXBlockA": mx, "MXBlockB": mx,
                               "DataTypeMXSA": "E8", "DataTypeMXSB": "E8"})
    original, out = derive(1, WaveTileContiguousA=False, **params)
    assert original["Valid"], out
    enabled, out = derive(1, WaveTileContiguousA=True, **params)
    assert enabled["Valid"], out
    assert enabled["VectorWidthA"] == tile
    assert enabled["WaveTileContiguousA"] is False
    # Covers padding, scale grouping, TDM, store width, LDS placement and naming.
    assert enabled._state == original._state
    assert getSolutionNameFull(enabled, False) == getSolutionNameFull(original, False)


@pytest.mark.parametrize("source_swap,mx", [(False, 0), (True, 32)])
def test_equal_auto_width_emits_original_assembly(derive, assembler, source_swap, mx):
    import re
    from codegen_harness import emit_one_kernel
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions

    sources = []
    for enabled in (False, True):
        sol, out = derive(0, mi=[16, 16, 128, 1, 1, 2, 2, 2, 2],
                          VectorWidthA=-1, VectorWidthB=1, HalfPLR=0,
                          SourceSwap=source_swap, WaveTileContiguousA=enabled,
                          ProblemType={"MXBlockA": mx, "MXBlockB": mx})
        assert sol["Valid"], out
        kernel, = generateKernelObjectsFromSolutions([sol])
        name, source, error = emit_one_kernel(KernelWriterAssembly(assembler, DebugConfig()),
                                             kernel, canonical=False)
        assert error == 0, name
        labels = {}
        sources.append(re.sub(r"label_skipCBPreSignal(?:_LCL)?_[A-Za-z0-9]{16}\b",
                              lambda match: labels.setdefault(match[0], f"label_cluster_{len(labels)}"),
                              source))
    assert sources[0] == sources[1]


@pytest.mark.parametrize("tile,width", [(6, -1), (6, 2), (6, 3), (7, -1),
                                      (8, 4), (12, 6), (16, 8)])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("dtype,mx,depth_u", [("F8", 0, 128), ("F8", 32, 128), ("F4", 32, 256)])
def test_contiguous_a_retains_legal_widths(derive, tile, width, source_swap, dtype, mx, depth_u):
    # Smaller B/DepthU keep the wider inputs inside the LDS budget.
    params = dict(mi=[16, 16, 128, 1, 1, tile, 2, 2, 2],
                  VectorWidthB=1, HalfPLR=0, DepthU=depth_u,
                  SourceSwap=source_swap, WaveTileContiguousA=True,
                  ProblemType={"DataType": dtype, "DestDataType": "S",
                               "MXBlockA": mx, "MXBlockB": mx})
    sol, out = derive(0, VectorWidthA=width, **params)
    assert sol["Valid"], out
    assert sol["WaveTileContiguousA"] is True
    expected_width = (2 if tile == 6 else 1) if width == -1 else width
    assert sol["VectorWidthA"] == expected_width
    assert sol["StoreVectorWidth"] == (1 if source_swap else 8)
    assert sol["LdsBlockSizePerPadA"] == 256
    assert not sol.get("_TDMIterateModeA", False)
    if mx:
        assert sol["VectorWidthMXSA"] == expected_width
        assert sol["LdsPadMXSA"] == sol["LdsBlockSizePerPadMXSA"] == 0
    reference, out = derive(0, VectorWidthA=1, **params)
    assert reference["Valid"], out
    # Physical placement stays contiguous while the requested grouping remains
    # in the solution and its name.
    assert _lds_configuration(sol) == _lds_configuration(reference)
    assert (getSolutionNameFull(sol, False) == getSolutionNameFull(reference, False)) == (expected_width == 1)


@pytest.mark.parametrize("tile,width", [(6, 1), (6, 2), (6, 3), (8, 4), (12, 6), (16, 8)])
@pytest.mark.parametrize("block", [64, 128, 256])
@pytest.mark.parametrize("mode", [0, 1])
def test_contiguous_a_scale_padding_checks_effective_scale_groups(derive, tile, width, block, mode):
    params = dict(mi=[16, 16, 128, 1, 1, tile, 2, 2, 2],
                  VectorWidthA=width, VectorWidthB=1, HalfPLR=0, DepthU=128,
                  WaveTileContiguousA=True,
                  ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    unpadded, out = derive(mode, **params)
    assert unpadded["Valid"], out
    assert unpadded["VectorWidthA"] == unpadded["VectorWidthMXSA"] == width
    assert unpadded["LdsPadMXSA"] == unpadded["LdsBlockSizePerPadMXSA"] == 0
    padded, out = derive(mode, LdsPadMXSA=16, LdsBlockSizePerPadMXSA=block, **params)
    assert not padded["Valid"]
    if block == 256:
        # The reader advances two scale tiles (128 B), regardless of raw VW.
        assert "WaveTileContiguousA" in out and "padding phase" in out
    else:
        assert "is not a pair the MX scale layout offers" in out


def test_contiguous_a_scale_read_metadata_preserves_vgpr_interleave(derive, assembler):
    from codegen_harness import emit_one_kernel
    from rocisa.instruction import DSLoadB32
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions

    sol, out = derive(1, mi=[16, 16, 128, 1, 1, 12, 4, 2, 2],
                      VectorWidthA=6, VectorWidthB=2, HalfPLR=1, DepthU=256,
                      SourceSwap=True, WaveTileContiguousA=True, LDSSegmentBRotate=1,
                      NumElementsPerBatchStore=16,
                      ProblemType={"DataType": "F4", "DestDataType": "S",
                                   "MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    kernel, = generateKernelObjectsFromSolutions([sol])
    writer = KernelWriterAssembly(assembler, DebugConfig())
    name, source, error = emit_one_kernel(writer, kernel, canonical=False)
    assert error == 0, name
    instruction = writer.tPA["MX"]["localReadInstruction"]
    assert instruction.getInst() is DSLoadB32
    assert instruction.blockWidth == 1
    assert writer.states.mxsInterleave is not None
    scale_loads = [line for line in source.splitlines()
                   if line.strip().startswith("ds_load_") and "v[vgprValuMXSA" in line]
    assert scale_loads
    assert all(line.strip().startswith("ds_load_b32 ") for line in scale_loads)


def _execute_scale_addresses(source, serial):
    import re

    def normalized(operand):
        operand = re.sub(r"-(?:256|512|768)\]", "]", operand)
        return re.sub(r"((?:\+\d+)+)\]",
                      lambda m: f"+{sum(map(int, m[1].split('+')[1:]))}]",
                      operand).replace("+0]", "]")

    registers = {}

    def value(operand):
        if "vgprSerial" in operand:
            return serial
        return registers[operand] if operand in registers else int(operand, 0)

    for line in source.splitlines():
        code = line.partition("//")[0].strip()
        if not code or code.startswith("/*"):
            continue
        op, *args = map(normalized, code.replace(",", " ").split())
        if op == "s_set_vgpr_msb":
            continue
        dst, *src = args
        if op == "v_add_co_u32":
            src = src[1:]
        values = [value(s) for s in src]
        if op in ("s_mov_b32", "v_mov_b32"):
            result = values[0]
        elif op in ("v_and_b32", "s_and_b32"):
            result = values[0] & values[1]
        elif op in ("v_xor_b32", "s_xor_b32"):
            result = values[0] ^ values[1]
        elif op == "v_lshrrev_b32":
            result = values[1] >> values[0]
        elif op == "s_lshr_b32":
            result = values[0] >> values[1]
        elif op == "v_lshlrev_b32":
            result = values[1] << values[0]
        elif op == "v_lshl_add_u32":
            result = (values[0] << values[1]) + values[2]
        elif op in ("v_add_u32", "v_add_nc_u32", "v_add_co_u32", "s_add_u32"):
            result = values[0] + values[1]
        elif op in ("v_mul_lo_u32", "s_mul_i32"):
            result = values[0] * values[1]
        elif op == "v_mad_u32_u24":
            result = (values[0] & 0xffffff) * (values[1] & 0xffffff) + values[2]
        else:
            pytest.fail(f"Unsupported scale address instruction: {line}")
        registers[dst] = result & 0xffffffff
    return registers, normalized


@pytest.mark.parametrize("tile,width", [
    (6, 1), (6, 2), (6, 3), (7, 1), (8, 4), (12, 3), (12, 6), (16, 8),
])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("dtype,depth_u", [("F8", 128), ("F4", 256)])
@pytest.mark.parametrize("rectangular", [False, True])
def test_contiguous_a_scale_loads_match_wmma_rows(
        derive, assembler, monkeypatch, tile, width, source_swap, dtype, depth_u, rectangular):
    import re
    from codegen_harness import emit_one_kernel
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.Components.LocalRead import LocalReadMFMA
    from tensilelite.Components.LraTileAssignment import LraTileAssignmentMFMA
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions

    init, loads = {}, []
    original_assignment = LraTileAssignmentMFMA.__call__

    def assignment(self, writer, kernel, tp):
        result = original_assignment(self, writer, kernel, tp)
        if tp["tensorChar"] == "MXSA":
            init.setdefault("assignment", str(result))
        return result

    monkeypatch.setattr(LraTileAssignmentMFMA, "__call__", assignment)
    for method in ("lraFinalOffset", "lraDeclareAddresses", "lraSwapAddressesForDTLPad"):
        original = getattr(KernelWriterAssembly, method)

        def address(self, kernel, *args, _original=original, _method=method, **kwargs):
            result = _original(self, kernel, *args, **kwargs)
            tp = next((arg for arg in args if isinstance(arg, dict) and "tensorChar" in arg), {})
            if tp.get("tensorChar") == "MXSA":
                init.setdefault(_method, str(result))
            return result

        monkeypatch.setattr(KernelWriterAssembly, method, address)
    original_read = LocalReadMFMA.localReadMX

    def local_read(self, writer, kernel, buffer_idx, iui, epsi, tp):
        result = original_read(self, writer, kernel, buffer_idx, iui, epsi, tp)
        if tp["tensorChar"] == "MXSA":
            loads.append((tp["localReadOffset"], str(result[0])))
        return result

    monkeypatch.setattr(LocalReadMFMA, "localReadMX", local_read)
    sol, out = derive(1, mi=[32 if rectangular else 16, 16, 128, 1, 1, tile, 1 if rectangular else 2, 2, 2],
                      VectorWidthA=width, VectorWidthB=1, DepthU=depth_u, HalfPLR=0,
                      SourceSwap=source_swap, WaveTileContiguousA=True,
                      LDSSegmentBRotate=0 if rectangular else 1, NumElementsPerBatchStore=16,
                      LdsPadMXSA=-1, LdsBlockSizePerPadMXSA=-1,
                      ProblemType={"DataType": dtype, "DestDataType": "S",
                                   "MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    assert sol["VectorWidthA"] == sol["VectorWidthMXSA"] == width
    assert sol["LdsPadMXSA"] == sol["LdsBlockSizePerPadMXSA"] == 0
    kernel, = generateKernelObjectsFromSolutions([sol])
    writer = KernelWriterAssembly(assembler, DebugConfig())
    name, _, error = emit_one_kernel(writer, kernel, canonical=False)
    assert error == 0, name
    assert len(init) == 4 and loads
    matrix_m = sol["MatrixInstM"]
    count = tile // 2 if matrix_m == 16 and tile % 2 == 0 else tile
    assert writer.states.mxsa.numVgprValuPerBlock == count
    assert writer.states.numReadsPerIterMXSA == count
    assert writer.tPA["MX"]["localReadInstruction"].blockWidth == 1
    source = "\n".join(init[key] for key in (
        "assignment", "lraFinalOffset", "lraDeclareAddresses", "lraSwapAddressesForDTLPad"))
    lane_regs = [_execute_scale_addresses(source, serial)[0] for serial in range(128)]
    _, normalized = _execute_scale_addresses(source, 0)
    offsets = sol["LDSSegInterleaveOffsets"]
    split = bool(offsets.get("mxsSplitA"))
    assert split is (matrix_m * tile * 4 % 128 == 0)
    buffer_stride = offsets.get("bufferStrideMXSA", sol["LdsOffsetA_Blk"])
    pad = sol["LdsPadMXSA"]
    pad_block = sol["LdsBlockSizePerPadMXSA"]

    def padded(address):
        return address + (address // pad_block) * pad if pad_block else address

    for k_offset, module in set(loads):
        assert k_offset % (tile * matrix_m * 4) == 0
        instructions = [line for line in module.splitlines() if line.startswith("ds_load_")]
        assert len(instructions) == count
        by_register = {}
        for line in instructions:
            assert line.startswith("ds_load_b32 ")
            dest, addr, immediate = re.search(r" (v\[[^]]+\]), (v\[[^]]+\]) offset:(\d+)", line).groups()
            match = re.search(r"vgprValuMXSA_X\d+_I\d+(?:\+(\d+))?", normalized(dest))
            register = int(match[1] or 0)
            by_register[register] = normalized(addr), int(immediate)
        assert len(by_register) == count
        for logical_tile in range(tile):
            register, selector = writer.mxsTileSpanScaleSel(kernel, writer.tPA["MX"], logical_tile)
            addr, immediate = by_register[register]
            for wave in range(4):
                for row_in_tile in range(matrix_m):
                    regs = lane_regs[wave * 32 + row_in_tile + selector * 16]
                    for phase in (0, 1):
                        actual = regs[addr]
                        if phase:
                            actual ^= regs["v[vgprLocalReadSwapAddrMXSA]"]
                        actual += immediate
                        row = logical_tile * matrix_m + row_in_tile
                        expected = offsets["ldsBaseMXSA"] + phase * buffer_stride
                        if split:
                            expected += (wave % 2) * offsets["mxsaStrideBytes"]
                        else:
                            row += (wave % 2) * tile * matrix_m
                        expected += padded(k_offset + row * 4)
                        # One dword contains the four K-block scales for this row.
                        assert list(range(actual, actual + 4)) == list(range(expected, expected + 4))


@pytest.mark.parametrize("tile,width", [(6, 4), (7, 2), (10, 4)])
def test_contiguous_a_still_requires_integral_operand_groups(derive, tile, width):
    sol, out = derive(0, mi=[16, 16, 128, 1, 1, tile, 8, 2, 2],
                      VectorWidthA=width, WaveTileContiguousA=True, HalfPLR=0)
    assert not sol["Valid"]
    assert f"MIWaveTile0({tile}) should be multiple of VectorWidthA({width})" in out


@pytest.mark.parametrize("contiguous_a", [False, True])
def test_contiguous_a_keeps_non_iterate_padding_limit(derive, contiguous_a):
    sol, out = derive(0, mi=[16, 16, 128, 1, 1, 6, 8, 2, 2],
                      VectorWidthA=2, VectorWidthB=4, HalfPLR=1,
                      WaveTileContiguousA=contiguous_a, TDMIterateMode=1,
                      LdsBlockSizePerPadA=2048, LdsPadA=16)
    if contiguous_a:
        assert not sol["Valid"]
        assert sol["TDMIterateMode"] == 0
        assert "LdsBlockSizePerPadA=2048" in out and "pad_interval" in out
        assert "WaveTileContiguousA disables A iterate" in out
    else:
        assert sol["Valid"], out
        assert sol["TDMIterateMode"] == 1
        assert sol["_TDMIterateModeA"]


@pytest.mark.parametrize("mode", [0, 1])
def test_contiguous_a_scale_padding_phase_must_match_component_writer(derive, mode):
    # Without splitting, Wave1 starts at byte384 in a 768-byte scale component.
    # M-half splitting removes that base offset, but the 384-byte K step still
    # does not preserve the padding phase required by the reader.
    overrides = dict(mi=[16, 16, 128, 1, 1, 6, 8, 2, 2],
                     VectorWidthA=1, VectorWidthB=8, HalfPLR=0,
                     StoreRemapVectorWidth=0, LDSTrInst=False,
                     LdsBlockSizePerPadMXSA=256, LdsPadMXSA=16,
                     ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    baseline, out = derive(0, **overrides)
    assert baseline["Valid"], out
    contiguous_a, out = derive(mode, WaveTileContiguousA=True, **overrides)
    assert not contiguous_a["Valid"]
    assert "WaveTileContiguousA" in out and "padding phase" in out


@pytest.mark.parametrize("tile,width,half_plr", [(6, 1, 1), (6, 2, 1), (7, 1, 2)])
@pytest.mark.parametrize("width_b,rotate", [(2, 0), (2, 1), (4, 0), (4, 1), (8, 0)])
@pytest.mark.parametrize("source_swap", [False, True])
def test_contiguous_a_with_original_b_and_rotate(derive, tile, width, half_plr, width_b, rotate, source_swap):
    sol, out = derive(1, mi=[16, 16, 128, 1, 1, tile, 8, 2, 2],
                      SourceSwap=source_swap, VectorWidthA=width, VectorWidthB=width_b,
                      HalfPLR=half_plr, WaveTileContiguousA=True,
                      LDSSegmentBRotate=rotate, LdsPadA=16, LdsPadB=16,
                      LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=1024,
                      TDMIterateMode=0, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert sol["Valid"], out
    offsets = sol["LDSSegInterleaveOffsets"]
    assert offsets["pairedBuffers"]
    assert not offsets.get("aBaseline") and not offsets.get("bBaseline")
    assert sol["StoreVectorWidth"] == (1 if source_swap else 8)
    assert sol["VectorWidthB"] == width_b
    if rotate:
        from tensilelite.SolutionStructs.segment_interleave import b_rotate_reason
        assert b_rotate_reason(sol._state) is None


@pytest.mark.parametrize("source_swap,store_width,valid", [
    (False, 1, True), (False, 2, True), (False, 4, True), (False, 8, True),
    (False, 16, False), (True, 1, True), (True, 2, False),
])
def test_contiguous_a_store_width_is_explicit_and_lane_local(derive, source_swap, store_width, valid):
    sol, out = derive(1, mi=[16, 16, 128, 1, 1, 6, 8, 2, 2],
                      SourceSwap=source_swap, VectorWidthA=2, VectorWidthB=4,
                      HalfPLR=1, WaveTileContiguousA=True,
                      StoreVectorWidth=store_width, LDSSegmentBRotate=1,
                      LdsPadA=16, LdsPadB=16,
                      LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=1024,
                      TDMIterateMode=0, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert bool(sol["Valid"]) == valid, out
    assert sol["StoreVectorWidth"] == store_width
    if not valid:
        assert "WaveTileContiguousA requires StoreVectorWidth" in out
        assert "for direct stores" in out


@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("case", ["compact", "data-parallel", "stream-k", "int8"])
def test_contiguous_a_preserves_supported_execution_paths(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, source_swap, case):
    integer = case == "int8"
    params = _make_params(
        gfx1250_iim, mi=[16, 16, 64 if integer else 128, 1, 1, 6, 8, 2, 2],
        GlobalSplitU=1, SourceSwap=source_swap, VectorWidthA=1, VectorWidthB=4,
        HalfPLR=1, WaveTileContiguousA=True, TDMIterateMode=0,
        LDSSegmentInterleave=0 if integer else 1, CompactLoopStore=case == "compact",
        ProblemType={"DataType": "I8" if integer else "F8",
                     "DestDataType": "I" if integer else "F8",
                     "ComputeDataType": "I" if integer else "S"})
    params.pop("StreamK")
    strategy = {"data-parallel": "DataParallel", "stream-k": "StreamK"}.get(case, "None")
    params.update(TileProcessingStrategy=strategy, WorkAssignment="StaticGrid")
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    assert sol["Valid"], capsys.readouterr().out
    assert sol["WaveTileContiguousA"]
    assert sol["TileProcessingStrategy"] == strategy
    assert sol["CompactLoopStore"] == (case == "compact")
    assert sol["ProblemType"]["ComputeDataType"].isInt32() == integer
    assert not sol["PostLoopStoreInNll"]


@pytest.mark.parametrize("contiguous", [False, True])
def test_contiguous_a_gradient_scalar_e_uses_loaded_half(derive, assembler, contiguous):
    import re
    from codegen_harness import emit_one_kernel
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions

    sol, out = derive(int(contiguous), mi=[16, 16, 32, 1, 1, 6, 8, 2, 2],
                      DepthU=128, VectorWidthA=1, VectorWidthB=4, HalfPLR=1,
                      SourceSwap=True, WaveTileContiguousA=contiguous,
                      NumElementsPerBatchStore=16, GroupLoadStore=False,
                      ProblemType={"DataType": "H", "DestDataType": "H",
                                   "Gradient": True, "UseE": True, "DataTypeE": "H",
                                   "Activation": True, "ActivationType": "hipblaslt_all"})
    assert sol["Valid"], out
    assert sol["StoreVectorWidth"] == 1
    kernel, = generateKernelObjectsFromSolutions([sol])
    writer = KernelWriterAssembly(assembler, DebugConfig())
    name, source, error = emit_one_kernel(writer, kernel, canonical=False)
    assert error == 0, name
    loads = [line for line in source.splitlines() if "// load E" in line]
    conversions = [line for line in source.splitlines()
                   if "v_cvt_f32_f16" in line and "gwvw 1, elementIdx" in line]
    assert loads and conversions
    assert all("buffer_load_d16_b16 " in line for line in loads)
    # Every scalar E load targets the low half, including odd batch elements.
    assert any("elementIdx 1" in line for line in conversions)
    assert all(re.search(r"\.l\s*//", line) for line in conversions)


@pytest.mark.parametrize("overrides,reason", [
    ({"VectorWidthB": 8, "LDSSegmentBRotate": 1}, "MIWaveTile[1]/VectorWidthB must be even"),
    ({"ProblemType": {"MXBlockA": 32, "MXBlockB": 32, "MXBlockFreeA": 2}},
     "2D MX scale for A supports MXBlockFreeA 128 only, got 2"),
])
def test_contiguous_a_rejects_unimplemented_combinations(derive, overrides, reason):
    params = dict(mi=[16, 16, 128, 1, 1, 6, 8, 2, 2], SourceSwap=False,
                  VectorWidthA=1, VectorWidthB=4, HalfPLR=1,
                  WaveTileContiguousA=True, TDMIterateMode=0,
                  LdsPadA=16, LdsPadB=16, LdsBlockSizePerPadA=256,
                  LdsBlockSizePerPadB=1024, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    params.update(overrides)
    sol, out = derive(1, **params)
    assert not sol["Valid"], out
    assert reason in out


@pytest.mark.parametrize("tile,width,half_plr", [(6, -1, 1), (6, 1, 1), (6, 2, 1),
                                               (7, -1, 2), (7, 1, 2)])
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("dtype,depth_u,pad", [("F8", 256, 16), ("F4", 512, 32)])
def test_contiguous_a_mx_selects_scale_layout_and_preserves_b_rotate(derive, tile, width, half_plr,
                                                                  source_swap, dtype, depth_u, pad):
    overrides = dict(mi=[16, 16, 128, 1, 1, tile, 8, 2, 2],
                     SourceSwap=source_swap, VectorWidthA=width, VectorWidthB=4,
                     HalfPLR=half_plr, WaveTileContiguousA=True,
                     LdsPadA=pad, LdsPadB=pad, DepthU=depth_u,
                     LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=1024,
                     TDMIterateMode=0,
                     ProblemType={"DataType": dtype, "DestDataType": "S",
                                  "MXBlockA": 32, "MXBlockB": 32,
                                  "DataTypeMXSA": "E8", "DataTypeMXSB": "E8"})
    reference, out = derive(1, LDSSegmentBRotate=0, **overrides)
    assert reference["Valid"], out
    rotated, out = derive(1, LDSSegmentBRotate=1, **overrides)
    assert rotated["Valid"], out
    # BRotate changes reader deltas; the TDM writer and allocation stay fixed.
    writer_keys = [key for key in reference._state if key.startswith("Lds")]
    assert {key: reference[key] for key in writer_keys} == {
        key: rotated[key] for key in writer_keys}
    reference_offsets = reference["LDSSegInterleaveOffsets"]
    offsets = rotated["LDSSegInterleaveOffsets"]
    for key in ("ldsBaseA", "ldsBaseB", "ldsBaseMXSA", "ldsBaseMXSB",
                "bufferStrideA", "bufferStrideB", "bufferStrideMXSA", "bufferStrideMXSB",
                "writeStrideBytes", "pairedUsedBytes"):
        assert reference_offsets[key] == offsets[key]
    assert bool(offsets.get("mxsSplitA")) is (tile == 6)
    if tile == 6:
        _check_split_mxsa_addresses(rotated)
    assert offsets["pairedUsedBytes"] <= rotated["LdsNumBytes"] <= rotated["MaxLDS"]
    assert rotated["StoreVectorWidth"] == (1 if source_swap else 8)
    # A scales pair consecutive tiles; B keeps its VWB4 scale group.
    from tensilelite.Components.LocalRead import LocalReadMFMA
    expected_width = (2 if tile == 6 else 1) if width == -1 else width
    assert rotated["VectorWidthA"] == rotated["VectorWidthMXSA"] == expected_width
    assert rotated["VectorWidthMXSB"] == 4
    caps = {"HasWMMA_V3": True}
    scale_a = LocalReadMFMA.getMxsTileSpanInfo(rotated, "MXSA", 0, caps)
    assert scale_a == ({"vectorWidth": 1, "numGroups": 3} if tile == 6 else None)
    assert LocalReadMFMA.getMxsTileSpanInfo(rotated, "MXSB", 1, caps) == {
        "vectorWidth": 4, "numGroups": 1}
    from tensilelite.SolutionStructs.segment_interleave import b_rotate_reason
    assert b_rotate_reason(rotated._state) is None


@pytest.mark.parametrize("source_swap", [False, True])
def test_mxsa_split_preserves_b_rotate_encoding_across_segment_boundary(derive, source_swap):
    sol, out = derive(
        1, mi=[16, 16, 128, 1, 1, 6, 8, 2, 2],
        SourceSwap=source_swap, VectorWidthA=1, VectorWidthB=4,
        HalfPLR=1, DepthU=512, WaveTileContiguousA=True, LDSSegmentBRotate=1,
        LdsPadA=128, LdsPadB=64, LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=1024,
        TDMIterateMode=0,
        ProblemType={"DataType": "F4", "DestDataType": "S", "MXBlockA": 32, "MXBlockB": 32})
    assert sol["Valid"], out
    _check_split_mxsa_addresses(sol)
    offsets = sol["LDSSegInterleaveOffsets"]
    # MXSA pushes the component past 64 KiB; B keeps its A/B-only read encoding.
    ab_bytes = offsets["bufferStrideA"] + offsets["bufferStrideB"]
    assert ab_bytes == 64512
    assert ab_bytes + offsets["mxsaHalfBytes"] > 65536
    assert offsets["readStrideBytesB"] == 65536
    assert offsets["writeStrideBytes"] != offsets["readStrideBytesB"]
    from tensilelite.SolutionStructs.segment_interleave import b_rotate_reason
    assert b_rotate_reason(sol._state) is None


@pytest.mark.parametrize("source_swap", [False, True])
def test_contiguous_a_mxsa_split_budget_keeps_auto_fallback_intact(derive, source_swap):
    params = dict(mi=[16, 16, 128, 1, 1, 6, 8, 2, 2],
                  SourceSwap=source_swap, VectorWidthA=1, VectorWidthB=4,
                  HalfPLR=0, DepthU=128, WaveTileContiguousA=True,
                  TDMIterateMode=0,
                  ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    ordinary, out = derive(0, **params)
    assert ordinary["Valid"], out
    forced, out = derive(1, **params)
    assert forced["Valid"], out
    assert forced["LDSSegInterleaveOffsets"]["mxsSplitA"]
    assert forced["LdsNumBytes"] > ordinary["LdsNumBytes"]
    params["MaxLDS"] = ordinary["LdsNumBytes"]
    automatic, out = derive(-1, **params)
    assert automatic["Valid"], out
    assert automatic["LDSSegmentInterleave"] == 0
    assert _lds_configuration(automatic) == _lds_configuration(ordinary)
    for mode, rotate in ((1, 0), (1, 1), (-1, 1)):
        rejected, out = derive(mode, LDSSegmentBRotate=rotate, **params)
        assert not rejected["Valid"]
        if mode == 1:
            assert "requires paired LDS placement: layout exceeds MaxLDS" in out
        else:
            assert "LDSSegmentBRotate: needs LDSSegmentInterleave=1" in out


@pytest.mark.parametrize("source_swap,tile,width,depth_u,half_plr,mt", [
    (False, [10, 4], -1, 256, 1, (640, 128)),
    (True, [10, 4], -1, 512, 1, (320, 256)),
    (True, [10, 4], -1, 256, 1, (320, 256)),
    (False, [4, 10], 2, 512, 1, (256, 320)),
    (False, [4, 10], 2, 256, 1, (256, 320)),
    (True, [4, 10], 2, 256, 2, (128, 640)),
])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("mx_block,scale_type", [(16, "F8"), (32, "E8"), (128, "E8")])
def test_contiguous_a_rectangular_mxf4_derivation(derive, source_swap, tile, width, depth_u, half_plr, mt, mode, mx_block, scale_type):
    sol, out = derive(mode, mi=[32, 16, 128, 1, 1, *tile, 2, 2],
                      VectorWidthA=width, VectorWidthB=2, DepthU=depth_u, HalfPLR=half_plr,
                      SourceSwap=source_swap, WaveTileContiguousA=True,
                      LDSSegmentBRotate=0, ClusterDim=[4, 4], ClusterAIDMapping=-2,
                      ProblemType={"DataType": "F4", "DestDataType": "F8",
                                   "MXBlockA": mx_block, "MXBlockB": mx_block,
                                   "DataTypeMXSA": scale_type, "DataTypeMXSB": scale_type})
    if mx_block == 16 and depth_u == 512:
        assert not sol["Valid"]
        assert "exceeds MaxLDS" in out or "bytes of LDS" in out
        return
    assert sol["Valid"], out
    assert sol["WaveTileContiguousA"]
    assert sol["LDSSegmentInterleave"] == mode
    assert sol["StoreVectorWidth"] == (1 if source_swap else 8)
    assert (sol["MacroTile0"], sol["MacroTile1"]) == mt
    assert sol["LdsNumBytes"] <= sol["MaxLDS"]
