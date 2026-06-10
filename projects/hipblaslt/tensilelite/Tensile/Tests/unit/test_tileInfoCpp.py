# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the C++ (nanobind) TileInfo query layer.

The read-only TileInfo grid/index query methods in
``Tensile.Components.Subtile.Kernel.TileInfo`` (ABTilePair case) are now backed
unconditionally by the compiled
``tensile_writer.subtile.tile_info.ABTileInfoQuery`` value object — there is no
parallel Python formula. These tests therefore lock the *values* the query
methods produce:

  * the methods match the documented reference math, derived here from the
    Python-computed TileInfo construction state (golden formulas), and
  * the C++ snapshot's derived grids/ratios stay consistent with the
    Python TileInfo.__init__ state that feeds them, plus
  * a handful of absolute-value pins for a known gfx950 AB_B16 case.

They run only when the extension is importable (it is a hard dependency of
Kernel.py); otherwise they skip.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import pytest

# Both the ISA layer (rocisa) and the tile_info extension must be present.
pytest.importorskip("rocisa")
cppti = pytest.importorskip("tensile_writer.subtile.tile_info")

from Tensile.Components.Subtile.Kernel import (
    TileInfo,
    AB_B16,
    AB_B8,
    AB_B4,
    AB_B4_2x2,
    AB_B16_2x2,
    AB_B16_TLU1,
    AB_B16_TLU1_16x1,
)


def _kernel(macroTileA, macroTileB, depthU, waveGroup=(4, 1)):
    """Build a representative gfx950 kernel dict for the read-only query layer."""
    return {
        "MIWaveGroup": list(waveGroup),
        "WavefrontSize": 64,
        "MacroTileA": macroTileA,
        "MacroTileB": macroTileB,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
    }


# A representative AB_B16-style gfx950 kernel configuration. The absolute-value
# pins below depend on these exact numbers.
BASE_KERNEL = _kernel(256, 128, 128)

# Each AB pair is exercised with a kernel whose macroTile/depthU yield a
# non-degenerate (strictly positive) subtile grid for both tc='A' (waveGroup
# 4) and tc='B' (waveGroup 1). Tighter-K (B4) and tall-M (TLU1/16x1) shapes
# need a larger depthU / MacroTileA than the AB_B16 base to tile cleanly.
AB_PAIRS = {
    "AB_B16": (AB_B16, BASE_KERNEL),
    "AB_B8": (AB_B8, BASE_KERNEL),
    "AB_B16_2x2": (AB_B16_2x2, BASE_KERNEL),
    "AB_B4": (AB_B4, _kernel(256, 128, 256)),
    "AB_B4_2x2": (AB_B4_2x2, _kernel(256, 128, 256)),
    "AB_B16_TLU1": (AB_B16_TLU1, _kernel(512, 128, 128)),
    "AB_B16_TLU1_16x1": (AB_B16_TLU1_16x1, _kernel(1024, 256, 128)),
}


def _make_tileinfo(pair, tc, kernel=BASE_KERNEL):
    """Build a TileInfo for a pair/tc. Query methods never touch the writer,
    so a None writer is sufficient for the read-only query layer."""
    return TileInfo(pair, tc, writer=None, kernel=kernel)


# ---------------------------------------------------------------------------
# Reference formulas (the documented math the C++ query layer implements).
#
# These mirror the prior pure-Python TileInfo bodies, derived from the
# Python-computed TileInfo construction state (localSubtileGrid, loadRatioGR,
# subtileShape, …). They are the golden oracle: the C++-backed query methods
# must reproduce them exactly.
# ---------------------------------------------------------------------------
def _ref_local_subtile_linear_id(ti, s0, s1):
    return s1 * ti.localSubtileGrid[0] + s0


def _ref_gr_load_index(ti, s0, s1, loadIdx=0):
    linearId = _ref_local_subtile_linear_id(ti, s0, s1)
    baseGR = int(linearId // ti.loadRatioGR) if ti.loadRatioGR else 0
    return baseGR + loadIdx


def _ref_lr_tile_index(ti, s0, s1, mfmaId=0):
    linearId = s1 * ti.lrLocalSubtileGrid[0] + s0
    tilesPerSubtile = int(ti.lrSubtileShape[0]) * int(ti.lrSubtileShape[1])
    return linearId * tilesPerSubtile + mfmaId


def _ref_wave_mma_tiles(ti, s0, s1):
    st = ti.subtileShape
    baseRow = s0 * int(st[0])
    baseCol = s1 * int(st[1])
    return [(baseRow + m, baseCol + k)
            for m in range(int(st[0]))
            for k in range(int(st[1]))]


def _ref_gr_reg_group(ti, s0):
    if ti.loadRatioGR >= 2.0:
        return int(s0 // ti.loadRatioGR)
    return s0


# ---------------------------------------------------------------------------
# Query-method values must match the documented reference math over the grid.
# ---------------------------------------------------------------------------
class TestTileInfoQueryValues:
    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_grid_index_queries(self, name, tc):
        pair, kernel = AB_PAIRS[name]
        ti = _make_tileinfo(pair, tc, kernel)
        g0, g1 = int(ti.localSubtileGrid[0]), int(ti.localSubtileGrid[1])
        assert g0 > 0 and g1 > 0
        lrTiles = int(ti.lrSubtileShape[0]) * int(ti.lrSubtileShape[1])
        for s0 in range(g0):
            for s1 in range(g1):
                ctx = f"{name}.{tc}({s0},{s1})"
                assert ti.getLocalSubtileLinearId(s0, s1) == \
                    _ref_local_subtile_linear_id(ti, s0, s1), f"{ctx}.linearId"
                assert ti.grLoadIndexForSubtile(s0, s1) == \
                    _ref_gr_load_index(ti, s0, s1), f"{ctx}.grLoadIndex"
                assert ti.grLoadIndexForSubtile(s0, s1, 1) == \
                    _ref_gr_load_index(ti, s0, s1, 1), f"{ctx}.grLoadIndex(1)"
                for mfmaId in range(lrTiles):
                    assert ti.lrTileIndexForSubtile(s0, s1, mfmaId) == \
                        _ref_lr_tile_index(ti, s0, s1, mfmaId), \
                        f"{ctx}.lrTileIndex({mfmaId})"
                assert [tuple(t) for t in ti.waveMmaTilesForSubtile(s0, s1)] == \
                    _ref_wave_mma_tiles(ti, s0, s1), f"{ctx}.waveMmaTiles"
                # globalMmaTilesForSubtile uses subtileForMmaTile internally;
                # assert it is well-formed (coordinates within the MMA grid).
                glob = [tuple(t) for t in ti.globalMmaTilesForSubtile(s0, s1)]
                assert glob, f"{ctx}.globalMmaTiles non-empty"
                for m, k in glob:
                    assert 0 <= m < ti.globalMMATileGrid[0], f"{ctx}.globalMmaTiles row"
                    assert 0 <= k < ti.globalMMATileGrid[1], f"{ctx}.globalMmaTiles col"
            assert ti.grRegGroupForSubtileRow(s0) == _ref_gr_reg_group(ti, s0), \
                f"{name}.{tc}.grRegGroup({s0})"


# ---------------------------------------------------------------------------
# The C++ snapshot's derived properties must match the Python TileInfo state
# (TileInfo.__init__ still computes the derived grids in Python; the C++
# snapshot that backs every query recomputes them and must agree).
# ---------------------------------------------------------------------------
class TestTileInfoSnapshotProperties:
    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_derived_state(self, name, tc):
        pair, kernel = AB_PAIRS[name]
        ti = _make_tileinfo(pair, tc, kernel)
        q = ti._cppQuery()
        assert tuple(q.globalMMATileGrid) == tuple(ti.globalMMATileGrid)
        assert tuple(q.localMMATileGrid) == tuple(ti.localMMATileGrid)
        assert tuple(q.subtileShape) == tuple(ti.subtileShape)
        assert q.subtileCount == ti.subtileCount
        assert q.subtileStride == ti.subtileStride
        assert tuple(q.globalSubtileGrid) == pytest.approx(tuple(ti.globalSubtileGrid))
        assert tuple(q.localSubtileGrid) == tuple(ti.localSubtileGrid)
        assert q.subtileSize == pytest.approx(ti.subtileSize)
        assert q.loadRatioGR == pytest.approx(ti.loadRatioGR)
        assert tuple(q.lrSubtileShape) == tuple(ti.lrSubtileShape)
        assert q.lrSubtileSize == pytest.approx(ti.lrSubtileSize)
        assert tuple(q.lrGlobalSubtileGrid) == tuple(ti.lrGlobalSubtileGrid)
        assert tuple(q.lrLocalSubtileGrid) == tuple(ti.lrLocalSubtileGrid)
        assert q.loadRatioLR == pytest.approx(ti.loadRatioLR)
        # Count properties mirror TileInfo convenience accessors.
        assert q.numMFMATiles == ti.mmaTileLocalTotalCount
        assert q.numGlobalSubtiles == ti.grSubtileTotalCount
        assert q.numLocalSubtiles == int(ti.localSubtileGrid[0] * ti.localSubtileGrid[1])

    def test_fractional_lr_global_subtile_grid(self):
        """lrGlobalSubtileGrid is a raw (possibly fractional) float grid; the
        C++ snapshot must preserve the fraction rather than truncating to int.

        MacroTileB=128 with the 16-row TLU1 LR subtile yields a half-tile in
        the M dimension (8 MMA tiles / 16 rows = 0.5), which previously exposed
        a C++ narrowing-to-long divergence (0.5 -> 0)."""
        ti = _make_tileinfo(AB_B16_TLU1_16x1, "B", _kernel(512, 128, 128))
        assert tuple(ti.lrGlobalSubtileGrid) == (0.5, 4.0)
        q = ti._cppQuery()
        assert tuple(q.lrGlobalSubtileGrid) == pytest.approx(tuple(ti.lrGlobalSubtileGrid))


# ---------------------------------------------------------------------------
# Absolute-value pins for an AB_B16 gfx950 case (lock correctness, not just
# self-consistency with the reference formulas).
# ---------------------------------------------------------------------------
class TestAbsoluteValues:
    def test_ab_b16_a_known_values(self):
        ti = _make_tileinfo(AB_B16, "A")
        # MacroTileA=256, depthU=128, MMA tile (16,32), subtileShape (1,2),
        # waveGroupSize=4 -> localMMATileGrid=(4,4), localSubtileGrid=(4,2).
        assert ti.localMMATileGrid == [4, 4]
        assert ti.localSubtileGrid == [4, 2]
        assert ti.loadRatioGR == pytest.approx(0.5)
        # getLocalSubtileLinearId: sId1*localSubtileGrid[0] + sId0
        assert ti.getLocalSubtileLinearId(3, 1) == 7
        # loadRatioGR=0.5 < 2 -> grRegGroup is identity.
        assert ti.grRegGroupForSubtileRow(3) == 3
        # baseGR = floor(linearId / 0.5) = linearId*2.
        assert ti.grLoadIndexForSubtile(3, 1) == 14
        assert ti.grLoadIndexForSubtile(3, 1, 1) == 15
        # globalMmaTilesForSubtile is well-formed and covers the subtile.
        tiles = [tuple(t) for t in ti.globalMmaTilesForSubtile(1, 0)]
        assert tiles
