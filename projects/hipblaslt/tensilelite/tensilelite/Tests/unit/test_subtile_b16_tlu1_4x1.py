# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Geometry pins for AB_B16_TLU1_4x1, the bf16 TLU=1 64-row x 32-K sub-tile.

This is the bf16 counterpart to the fp4 TLU=1 stacks: `subtileShape=(4, 1)`
gives a 64-row M extent and a 128 B LDS column, the same column width the fp4
`(16, 1)` stack reaches.  bf16 gets there at a stack of 4 rather than 16 because
its element is 4x wider, which is also why bf16 has exactly one usable stack --
see `_SUBTILE_STACK_B16` in SolutionStructs/Solution.py.

Everything here is derived, not configured, so a change to `_tlu1LRGeom`, to the
GR chunk ramp, or to the geometry table moves these numbers.  That is the point:
the LR read (`ds_read_b64_tr_b16`) and the GR fetch have to agree on the column
layout, and nothing downstream checks that they do.
"""

import pytest

from gpu_test_helpers import TileConfig, create_writer

from Tensile.Components.Subtile.Kernel import AB_GEOMETRY_MAP
from Tensile.Components.Subtile.SubtileGREmit import _tluKWaveSlots
from Tensile.Components.Subtile.SubtileLREmit import _tlu1LRGeom, _tlu1TrLoadInst
from Tensile.Components.Subtile.SubtileTLUSwizzle import selectTLU1B16SwizzleBits

GEOMETRY = AB_GEOMETRY_MAP["AB_B16_TLU1_4x1"]
WAVE_SIZE = 64


def _tileInfoA(mtA=256, mtB=64, depthU=128, waveGroup=(4, 1)):
    """TileInfo for A under this geometry.  Defaults to the one shape that
    enables the swizzle -- see test_swizzle_enablement_window for why."""
    _, _, tileInfoA, _ = create_writer(
        TileConfig(mt_a=mtA, mt_b=mtB, depth_u=depthU),
        mi_wave_group=list(waveGroup),
        geometry=GEOMETRY,
    )
    return tileInfoA


def test_geometry_table_entry():
    """The frozen half: what the geometry table itself declares."""
    gr, lr = GEOMETRY.gr, GEOMETRY.lr
    assert tuple(gr.subtileShape) == (4, 1)
    assert tuple(lr.subtileShape) == (4, 1)
    # 64-bit LR read: the only width gfx950 offers for ds_read_b64_tr_b16.
    assert lr.loadWidth == 8
    assert gr.tlu and lr.tlu


def test_lr_column_layout():
    """The LDS column the transpose read walks.

    colStride 128 B is the whole point of the (4, 1) stack: 4 subtile rows x
    instM 16 x 2 B.  kHalfStride is the byte step between the two halves of a
    paired tr read, and it must be colsPerRead columns, not a fixed 16 x strip
    (the fp4 form) -- at bf16 that would over-step by 4x.
    """
    g = _tlu1LRGeom(_tileInfoA(), WAVE_SIZE)

    assert g.colStride == 128
    assert g.kHalfStride == 512      # colsPerRead 4 * colStride 128
    assert g.tileMStride == 32       # instM 16 * 2 B -- where the XOR field starts
    assert g.mSubStride == 8         # 4 rows/lane * 2 B

    # ds_read_b64_tr_b16 hands a lane 4 M-rows, so four lanes span one column.
    # This is the structural difference from tr_b4, where lanesPerCol == 1 and a
    # single lane covers a whole 16-row MMA-M tile.
    assert g.rowsPerLaneRead == 4
    assert g.lanesPerCol == 4
    assert g.colsPerRead == 4
    assert g.regsPerRead == 2

    assert g.kGroups == 4            # waveSize 64 / instM 16
    assert g.kPerGroup == 8          # instK 32 / kGroups 4
    assert g.readsPerTile == 2       # kPerGroup 8 / colsPerRead 4


def test_lr_uses_the_bf16_transpose_read():
    """bf16 must reach ds_read_b64_tr_b16, not the fp4 tr_b4 form."""
    inst = _tlu1TrLoadInst(_tileInfoA())
    assert inst.__name__ == "DSLoadB64TrB16"


def test_swizzle_field_stays_disjoint_from_msub():
    """The XOR permutes whole m-blocks, so it must sit above the mSub field.

    _tlu1LRGeom raises if this fails; assert the margin explicitly so a future
    loadWidth or instM change shows up here as a number rather than as a
    RuntimeError with no recorded expectation.
    """
    g = _tlu1LRGeom(_tileInfoA(), WAVE_SIZE)
    assert g.mSubStride * (g.lanesPerCol - 1) == 24
    assert 24 < g.tileMStride
    assert g.colStride & (g.colStride - 1) == 0


def test_gr_chunk_ramp_is_dtype_correct():
    """chunksPerK is what splits the GR chunk index P into (k row, m block).

    The pre-existing code forced chunksPerK = 1 for anything non-fp4, which is
    only right when the strip fits in one 16 B chunk.  A bf16 (4, 1) strip is
    128 B, so P must split as m block = P % 8, k row = P // 8.
    """
    ti = _tileInfoA()
    instM = int(ti.mmaTileShape[0])
    mStripBytes = int(ti.subtileShape[0] * instM * ti.bpe)

    assert mStripBytes == 128
    assert max(1, mStripBytes // 16) == 8       # chunksPerK
    assert int(16 / ti.bpe) == 8                # elemsPerChunk


ALL_SHAPES = [
    (mt, wg) for mt in (64, 128, 256) for wg in ((1, 4), (2, 2), (4, 1))
]

# Where the wave owns fewer MMA M-tiles than the stack, it does not own a whole
# strip -- it *shares* one, which _subtileWaveStraddlesStrip permits explicitly
# ("whole strips OR share one with other waves").  localSubtileGrid[0] is 1 there
# not because the geometry is degenerate but because there is one shared strip,
# and the surplus waves take K windows within it.  These are precisely the shapes
# that exercise the fetch-group wave term in the GR swizzle, so they are the
# important ones to keep, not to prune.
COOPERATIVE_SHAPES = [(mt, wg) for (mt, wg) in ALL_SHAPES if mt // wg[0] < 64]


@pytest.mark.parametrize("mtA,waveGroup", COOPERATIVE_SHAPES)
def test_waves_share_a_strip_below_64_rows_per_wave(mtA, waveGroup):
    """Fewer M-tiles per wave than the stack means sharing, and sharing is legal.

    The wave still covers a whole number of sub-tiles' worth of the strip
    (stack % perWave == 0), so it has a soffset register and the GR emit stays
    in range.  This pins the distinction: a *fraction* of a strip is the
    rejected case, a *share* of one is not.
    """
    ti = _tileInfoA(mtA=mtA, waveGroup=waveGroup)
    stack = int(ti.subtileShape[0])
    perWave = int(ti.localMMATileGrid[0])
    assert perWave < stack
    assert stack % perWave == 0, "a shared strip must divide evenly, else it straddles"
    assert int(ti.localSubtileGrid[0]) == 1


@pytest.mark.parametrize("mtA,waveGroup", ALL_SHAPES)
def test_swizzle_is_enabled_for_every_shape(mtA, waveGroup):
    """The XOR is on everywhere, including where waves cooperate on a fetch.

    It is not gated on `_sharedStrip` (which the fp4 swizzle does use).  That
    gate is about GR seeing k relative to a wave's K slice while LR sees it
    absolute; the bf16 GR emit closes that gap by sourcing the high k bits from
    the fetch-group index, so there is nothing left to gate on.  Under the gate
    only MacroTileA 256 with MIWaveGroup [4, 1] survived -- one shape in nine,
    and not MT 64x64.
    """
    ti = _tileInfoA(mtA=mtA, waveGroup=waveGroup)
    assert selectTLU1B16SwizzleBits(ti) == 2


@pytest.mark.parametrize("mtA,waveGroup,localKSpan", [
    # localKSpan is the K extent one wave's own chunk ramp covers, and it decides
    # where each swizzle k bit comes from: below it, from laneId/loadIdx; at or
    # above it, from the fetch-group index.  k bit 3 is the one in play, so a
    # span of 8 is exactly the case that needs the wave term -- and MT 64x64 is
    # a span of 8 in every wave group.
    (64,  (1, 4), 8),
    (64,  (2, 2), 8),
    (64,  (4, 1), 8),
    (128, (1, 4), 8),
    (128, (2, 2), 16),
    (128, (4, 1), 16),
    (256, (1, 4), 8),
    (256, (2, 2), 16),
    (256, (4, 1), 32),
])
def test_local_k_span_matches_the_wave_k_base(mtA, waveGroup, localKSpan):
    """A wave's chunk ramp must cover exactly the K slice its address walks.

    localKSpan (from numGRPerSubtile) and the wave's K-row base (from the
    cooperative-fetch split) are computed by different code on different sides
    of the emit.  If they disagree, the swizzle reconstructs the wrong absolute
    k and A is silently wrong.  They agreed only after subtileCount was pinned
    to 1 -- with it deriving, MT 64x64 [4,1] claimed a span of 32 while owning 8.
    """
    ti = _tileInfoA(mtA=mtA, waveGroup=waveGroup)
    instM = int(ti.mmaTileShape[0])
    chunksPerK = max(1, int(ti.subtileShape[0] * instM * ti.bpe) // 16)
    span = int(ti.numGRPerSubtile) * WAVE_SIZE // chunksPerK
    assert span == localKSpan

    kSplit, _, rowsPerSlice, _ = _tluKWaveSlots(ti)
    kRows = int(ti.mmaTileShape[1] * ti.subtileShape[1])
    waveKBase = kRows // ti.grCoopWaves if ti.grWavesPerStrip > 1 else rowsPerSlice
    assert waveKBase == span


def test_subtile_count_is_pinned_contiguous():
    """TLU=1 is one contiguous strip in M, so subtileCount=1 / subtileStride=0.

    Left deriving, for_kernel() sets subtileCount=MIWaveGroup[0] -- the TLU=0
    "N discontiguous strips in M" layout (SubtileGeometry.py:370).  That inflates
    globalGRTileSize, and with it numGRTotal: MT 64x64 [4,1] fetched 4x the
    bytes the macro tile contains.
    """
    assert GEOMETRY.gr.subtileCount == 1
    assert GEOMETRY.gr.subtileStride == 0


def test_swizzle_width_is_the_whole_mblock_field():
    """2 bits == log2(subtileShape[0]).  The XOR always consumes the full field;
    a partial one under-spreads banks silently rather than failing."""
    ti = _tileInfoA()
    assert selectTLU1B16SwizzleBits(ti) == int(ti.subtileShape[0]).bit_length() - 1
