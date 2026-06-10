# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the optional C++ (nanobind) TileInfo query layer.

These tests compare the pure-Python read-only TileInfo grid/index query methods
in ``Tensile.Components.Subtile.Kernel.TileInfo`` (ABTilePair case) against the
compiled ``tensile_writer.subtile.tile_info.ABTileInfoQuery`` value object. They
run only when the extension is importable; otherwise they skip, so the default
(Python-only) TensileLite build is unaffected.

Delegation is gated on ``Kernel._USE_CPP`` (flipped at call time by the
``cpp_delegation`` context manager), so the *same* TileInfo objects are
exercised through both the Python and the C++ code paths and asserted to
produce identical results.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import contextlib

import pytest

# Both the ISA layer (rocisa) and the tile_info extension must be present.
pytest.importorskip("rocisa")
cppti = pytest.importorskip("tensile_writer.subtile.tile_info")

from Tensile.Components.Subtile import Kernel as krn
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


@contextlib.contextmanager
def cpp_delegation():
    """Temporarily enable the C++ TileInfo query layer.

    The geometry layer is always C++; this only flips ``Kernel._USE_CPP`` so the
    read-only TileInfo queries route through ``ABTileInfoQuery``.
    """
    saved_use = krn._USE_CPP
    krn._USE_CPP = True
    try:
        yield
    finally:
        krn._USE_CPP = saved_use


def _assert_same(py, cpp, ctx=""):
    """Recursively assert Python and C++ results match (float-tolerant)."""
    if isinstance(py, (tuple, list)):
        assert isinstance(cpp, (tuple, list)), f"{ctx}: type mismatch {py!r} vs {cpp!r}"
        assert len(py) == len(cpp), f"{ctx}: length {len(py)} vs {len(cpp)}"
        for i, (a, b) in enumerate(zip(py, cpp)):
            _assert_same(a, b, f"{ctx}[{i}]")
    elif isinstance(py, float) or isinstance(cpp, float):
        assert py == pytest.approx(cpp), f"{ctx}: {py!r} != {cpp!r}"
    else:
        assert py == cpp, f"{ctx}: {py!r} != {cpp!r}"


def _make_tileinfo(pair, tc, kernel=BASE_KERNEL):
    """Build a TileInfo for a pair/tc. Query methods never touch the writer,
    so a None writer is sufficient for the read-only query layer."""
    return TileInfo(pair, tc, writer=None, kernel=kernel)


def _both(call):
    """Run ``call()`` with the Python path and the C++ path; return (py, cpp)."""
    py = call()
    with cpp_delegation():
        cpp = call()
    return py, cpp


def _parity(call, ctx):
    py, cpp = _both(call)
    _assert_same(py, cpp, ctx)
    return py


# ---------------------------------------------------------------------------
# Read-only query-method parity over the local subtile grid
# ---------------------------------------------------------------------------
class TestTileInfoQueryParity:
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
                _parity(lambda s0=s0, s1=s1: ti.getLocalSubtileLinearId(s0, s1),
                        f"{ctx}.getLocalSubtileLinearId")
                _parity(lambda s0=s0, s1=s1: ti.grLoadIndexForSubtile(s0, s1),
                        f"{ctx}.grLoadIndexForSubtile")
                _parity(lambda s0=s0, s1=s1: ti.grLoadIndexForSubtile(s0, s1, 1),
                        f"{ctx}.grLoadIndexForSubtile(loadIdx=1)")
                for mfmaId in range(lrTiles):
                    _parity(lambda s0=s0, s1=s1, m=mfmaId:
                            ti.lrTileIndexForSubtile(s0, s1, m),
                            f"{ctx}.lrTileIndexForSubtile(mfmaId={mfmaId})")
                _parity(lambda s0=s0, s1=s1: ti.globalMmaTilesForSubtile(s0, s1),
                        f"{ctx}.globalMmaTilesForSubtile")
                _parity(lambda s0=s0, s1=s1: ti.waveMmaTilesForSubtile(s0, s1),
                        f"{ctx}.waveMmaTilesForSubtile")
            _parity(lambda s0=s0: ti.grRegGroupForSubtileRow(s0),
                    f"{name}.{tc}.grRegGroupForSubtileRow({s0})")


# ---------------------------------------------------------------------------
# The C++ snapshot's derived properties must match the Python TileInfo state.
# ---------------------------------------------------------------------------
class TestTileInfoSnapshotProperties:
    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_derived_state(self, name, tc):
        pair, kernel = AB_PAIRS[name]
        ti = _make_tileinfo(pair, tc, kernel)
        with cpp_delegation():
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

    def test_fractional_lr_global_subtile_grid_parity(self):
        """lrGlobalSubtileGrid is a raw (possibly fractional) float grid; the
        C++ snapshot must preserve the fraction rather than truncating to int.

        MacroTileB=128 with the 16-row TLU1 LR subtile yields a half-tile in
        the M dimension (8 MMA tiles / 16 rows = 0.5), which previously exposed
        a C++ narrowing-to-long divergence (0.5 -> 0)."""
        ti = _make_tileinfo(AB_B16_TLU1_16x1, "B", _kernel(512, 128, 128))
        assert tuple(ti.lrGlobalSubtileGrid) == (0.5, 4.0)
        with cpp_delegation():
            q = ti._cppQuery()
        assert tuple(q.lrGlobalSubtileGrid) == pytest.approx(tuple(ti.lrGlobalSubtileGrid))


# ---------------------------------------------------------------------------
# Absolute-value pins for an AB_B16 gfx950 case (lock correctness, not just
# self-consistency between the two implementations).
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

    def test_ab_b16_a_values_match_under_cpp(self):
        ti = _make_tileinfo(AB_B16, "A")
        py_tiles = [tuple(t) for t in ti.globalMmaTilesForSubtile(1, 0)]
        with cpp_delegation():
            assert ti.getLocalSubtileLinearId(3, 1) == 7
            assert ti.grRegGroupForSubtileRow(3) == 3
            assert ti.grLoadIndexForSubtile(3, 1) == 14
            assert ti.grLoadIndexForSubtile(3, 1, 1) == 15
            cpp_tiles = [tuple(t) for t in ti.globalMmaTilesForSubtile(1, 0)]
        assert cpp_tiles == py_tiles


def test_default_path_is_python_only():
    """With the env flag unset, delegation must be disabled by default."""
    import os
    if os.environ.get("TENSILE_WRITER_CPP", "").strip().lower() not in (
            "", "0", "false", "no", "off"):
        pytest.skip("TENSILE_WRITER_CPP is set; default-off behavior not under test")
    assert krn._USE_CPP is False
    ti = _make_tileinfo(AB_B16, "A")
    assert ti._useCppQuery() is False
