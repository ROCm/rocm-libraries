################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R5 — Subtile/Kernel.py TLU1-arm + CDTileGeometry characterization (CPU-only).

Direct-instantiation tests for two uncovered regions of Kernel.py:

1. **TLU1 tile branches** (lines 151-253):
   ABGRTile and ABLRTile constructors have a branch on GRTag_TLU1 / LRTag_TLU1
   that sets contiguousDim='M' (column-major geometry).  This branch is selected
   when the kernel uses TLUA=True (TransposeA=0/NN layout), which triggers
   Solution.py to choose _ABTilePairA="AB_B16_TLU1".  ABGRTile and ABLRTile
   are instantiated during TileInfo construction — before any emit call — so we
   can cover these lines by constructing TileInfo with AB_B16_TLU1 directly.

   Note: the SubtileGREmit TLU1 emit functions are registered as stubs
   (SubtileGREmit.py lines 78-84), so full kernel emit for NN+UseSubtileImpl
   is not yet implemented.  TileInfo construction succeeds; it is the subsequent
   graTileAssignment() call that would crash.  This test covers the construction
   path only, matching the scope of the uncovered lines.

2. **CDTileGeometry TileInfo branches** (lines 402, 500-551):
   TileInfo.__init__ has three isinstance branches: ABTilePair, MXScaleTilePair,
   and CDTileGeometry.  The CDTileGeometry branch (tc='C'/'D') is exercised by
   storeD roundtrip tests but those are @pytest.mark.gpu.  Here we construct
   TileInfo(CD_F32, 'C', ...) directly to cover the CDTile init path.

3. **Grid utility methods** (lines 572-603):
   TileInfo.getLocalSubtileLinearId, getLocalSubtileIdFromLinearId,
   getLocalMMATileLinearId, getLocalSubtileIdFromMMATile, grLoadIndexForSubtile
   are defined on TileInfo but not called from the codegen path.  We call them
   directly here.

pytestmark = pytest.mark.unit.  CPU-only; no GPU, no compile, no hardware.
"""

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _mock_writer():
    """Minimal writer stub for TileInfo construction (no emit calls)."""
    class _MockPool:
        def size(self):
            return 0
        def checkOut(self, n, align=None):
            return 0
        def checkIn(self, v):
            pass

    class _MockCaps(dict):
        pass

    class _MockStates:
        regCaps = _MockCaps({"PhysicalMaxVgpr": 512, "MaxVgpr": 256})
        agprPool = _MockPool()
        vgprPool = _MockPool()

    class _MockWriter:
        states = _MockStates()
        agprPool = _MockPool()
        vgprPool = _MockPool()

    return _MockWriter()


def _kernel_ab(macroTileA=128, macroTileB=128, depthUA=64, depthUB=64,
               miwavegroup=(2, 2)):
    """Minimal kernel dict for ABTilePair TileInfo construction (tc='A' or 'B')."""
    return {
        "MacroTileA":  macroTileA,
        "MacroTileB":  macroTileB,
        "_DepthUA":    depthUA,
        "_DepthUB":    depthUB,
        "MIWaveGroup": list(miwavegroup),
        "WavefrontSize": 64,
    }


def _kernel_cd(macroTile0=128, macroTile1=128, miwavegroup=(2, 2)):
    """Minimal kernel dict for CDTileGeometry TileInfo construction (tc='C'/'D')."""
    return {
        "MacroTile0":  macroTile0,
        "MacroTile1":  macroTile1,
        "MIWaveGroup": list(miwavegroup),
        "WavefrontSize": 64,
    }


# ---------------------------------------------------------------------------
# 1. TLU1 path — ABGRTile/ABLRTile with GRTag_TLU1 / LRTag_TLU1
#    Targets Kernel.py lines 151-152, 159, 163, 167, 175, 188, 191, 194, 197,
#    222-223, 230, 234, 247, 250, 253.
# ---------------------------------------------------------------------------

def test_r5_abgrtile_tlu1_branch():
    """ABGRTile init: GRTag_TLU1 sets contiguousDim='M', not 'K'.

    Covers lines 151-152: the ``if isinstance(config.tag, GRTag_TLU1):`` branch
    in ABGRTile.__init__.  TileInfo constructs ABGRTile automatically when
    geometry is an ABTilePair with a TLU1-tagged GR config.
    """
    from Tensile.Components.Subtile.Kernel import (
        TileInfo, AB_B16_TLU1,
    )
    from Tensile.Components.Subtile.SubtileGeometry import GRTag_TLU1

    # Use macroTileA=256 so localMMATileGrid[0] = (256//16)/2 = 8 >= subtileShape[0]=8.
    kernel = _kernel_ab(macroTileA=256, macroTileB=128, depthUA=64, depthUB=64)
    writer = _mock_writer()

    ti = TileInfo(AB_B16_TLU1, "A", writer, kernel)

    # ABGRTile was constructed with GRTag_TLU1 config — hit line 151.
    gr_tile = ti.gr
    assert gr_tile is not None, "gr tile should be set for ABTilePair"
    assert isinstance(gr_tile.config.tag, GRTag_TLU1), (
        "AB_B16_TLU1 should produce GRTag_TLU1-tagged GR config"
    )
    # Line 151-152: contiguousDim='M' for TLU1 (not 'K').
    assert gr_tile.contiguousDim == "M", (
        f"TLU1 GR should be M-contiguous, got {gr_tile.contiguousDim!r}"
    )
    assert gr_tile.contiguousElements == gr_tile.config.loadShape.m, (
        "TLU1 GR contiguousElements should equal loadShape.m"
    )

    # Line 159: subtileShape property (returns config.subtileShape).
    assert gr_tile.subtileShape == gr_tile.config.subtileShape

    # Line 163: subtileCount property.
    assert gr_tile.subtileCount == gr_tile.config.subtileCount

    # Line 167: subtileStride property.
    assert gr_tile.subtileStride is not None  # may be 0

    # Line 175: loadShape property.
    assert gr_tile.loadShape == gr_tile.config.loadShape

    # Lines 188, 191, 194, 197: emit-dispatch methods invoked via TLU1 stubs
    # (SubtileGREmit.py registers GRTag_TLU1 as stubs that return None).
    # Calling them covers the dispatch lines without a real register/module.
    assert gr_tile.emitGlobalReadOffset(ti, writer, kernel) is None  # line 188
    assert gr_tile.emitGlobalRead(ti, writer, kernel) is None        # line 191
    assert gr_tile.emitLocalWrite(ti, writer, kernel) is None        # line 194
    assert gr_tile.emitDTLInit(ti, writer, kernel) is None           # line 197


def test_r5_ablrtile_tlu1_branch():
    """ABLRTile init: LRTag_TLU1 sets contiguousDim='M', not 'K'.

    Covers lines 222-223: the ``if isinstance(config.tag, LRTag_TLU1):`` branch
    in ABLRTile.__init__.
    """
    from Tensile.Components.Subtile.Kernel import (
        TileInfo, AB_B16_TLU1,
    )
    from Tensile.Components.Subtile.SubtileGeometry import LRTag_TLU1

    # Use macroTileA=256 to ensure valid localSubtileGrid for TLU1 subtileShape=(8,1).
    kernel = _kernel_ab(macroTileA=256, macroTileB=128, depthUA=64, depthUB=64)
    writer = _mock_writer()

    ti = TileInfo(AB_B16_TLU1, "A", writer, kernel)

    # ABLRTile was constructed with LRTag_TLU1 config — hit line 221-223.
    lr_tile = ti.lr
    assert lr_tile is not None, "lr tile should be set for ABTilePair"
    assert isinstance(lr_tile.config.tag, LRTag_TLU1), (
        "AB_B16_TLU1 should produce LRTag_TLU1-tagged LR config"
    )
    # Line 222-223: contiguousDim='M' for TLU1 (not 'K').
    assert lr_tile.contiguousDim == "M", (
        f"TLU1 LR should be M-contiguous, got {lr_tile.contiguousDim!r}"
    )
    assert lr_tile.contiguousElements == lr_tile.config.loadShape.m, (
        "TLU1 LR contiguousElements should equal loadShape.m"
    )

    # Line 230: subtileShape property.
    assert lr_tile.subtileShape == lr_tile.config.subtileShape

    # Line 234: loadShape property.
    assert lr_tile.loadShape == lr_tile.config.loadShape

    # Lines 247, 250, 253: LR emit-dispatch methods invoked via TLU1 stubs.
    assert lr_tile.emitLocalReadOffset(ti, writer, kernel) is None  # line 247
    assert lr_tile.emitLocalRead(ti, writer, kernel) is None        # line 250
    assert lr_tile.emitDTLInit(ti, writer, kernel) is None          # line 253


def test_r5_tileinfo_tlu1_grids():
    """TileInfo with AB_B16_TLU1 computes correct grid dimensions.

    Verifies that TileInfo.__init__ grid computation for ABTilePair works with
    the TLU1 geometry: all subtile/MMA grid fields are populated and positive.

    AB_B16_TLU1 has subtileShape=(8, 1): each subtile covers 8 MMA-M tiles x 1
    MMA-K tile.  With waveGroupSize=2 and mmaM=16, we need at least
    8 * 16 * 2 = 256 elements in M to get a localMMATileGrid[0] >= 8.
    Using macroTileA=256, depthUA=64:
      globalMMATileGrid = (256//16, 64//32) = (16, 2)
      localMMATileGrid  = (16/2, 2) = (8, 2)
      localSubtileGrid  = [8//8, 2//1] = [1, 2]
    """
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16_TLU1

    kernel = _kernel_ab(macroTileA=256, macroTileB=128, depthUA=64, depthUB=64,
                        miwavegroup=(2, 2))
    writer = _mock_writer()
    ti = TileInfo(AB_B16_TLU1, "A", writer, kernel)

    assert ti.globalMMATileGrid[0] > 0
    assert ti.globalMMATileGrid[1] > 0
    assert ti.localMMATileGrid[0] > 0
    assert ti.localMMATileGrid[1] > 0
    assert ti.globalSubtileGrid[0] > 0
    assert ti.globalSubtileGrid[1] > 0
    assert ti.localSubtileGrid[0] >= 1
    assert ti.localSubtileGrid[1] >= 1
    assert ti.waveGroupSize == 2


def test_r5_tileinfo_tlu1_16x1():
    """TileInfo with AB_B16_TLU1_16x1 (256-bit GR along M) hits same TLU1 path."""
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16_TLU1_16x1
    from Tensile.Components.Subtile.SubtileGeometry import GRTag_TLU1, LRTag_TLU1

    kernel = _kernel_ab(macroTileA=128, macroTileB=128, depthUA=64, depthUB=64,
                        miwavegroup=(2, 2))
    writer = _mock_writer()
    ti = TileInfo(AB_B16_TLU1_16x1, "A", writer, kernel)

    # TLU1 branch again for the wider (16-element) variant.
    assert ti.gr.contiguousDim == "M"
    assert ti.lr.contiguousDim == "M"
    assert ti.gr.contiguousElements == 16  # loadShape.m=16 for TLU1_16x1


# ---------------------------------------------------------------------------
# 2. CDTileGeometry TileInfo branches (lines 402, 500-551)
# ---------------------------------------------------------------------------

def test_r5_cdtilegeometry_tileinfo_init():
    """TileInfo(CD_F32, 'C', ...) exercises the CDTileGeometry __init__ branch.

    Covers:
      402-408   elif isinstance(geometry, CDTileGeometry) in TileInfo.__init__
      500-510   CDTileGeometry grid computation (localMMATileGrid / subtileShape)
      522-524   CDTileGeometry loadWidth accessor
      546-551   CDTileGeometry _check_dim consistency
    """
    from Tensile.Components.Subtile.Kernel import TileInfo, CD_F32

    # CD_F32 has subtileShape=(1,1), mmaLayout=MFMA_16x16_1B_4N_4V
    # macroTile0=128, macroTile1=128, waveGroup=(2,2):
    #   globalMMATileGrid = (128//(1*16), 128//(1*16)) = (8, 8)
    #   localMMATileGrid  = globalMMATileGrid / wg = (4, 4)
    #   globalSubtileGrid = globalMMATileGrid / subtileShape = (8, 8)
    #   localSubtileGrid  = localMMATileGrid / subtileShape = (4, 4)
    kernel = _kernel_cd(macroTile0=128, macroTile1=128, miwavegroup=(2, 2))
    writer = _mock_writer()

    ti = TileInfo(CD_F32, "C", writer, kernel)

    # Line 402: CDTileGeometry branch was taken.
    assert ti.macroTile is None
    assert ti.macroTile0 == 128
    assert ti.macroTile1 == 128
    assert ti.waveGroup == (2, 2)
    assert ti.depthU is None
    assert ti.isSwizzled is False

    # Lines 500-510: CDTileGeometry grid fields populated.
    assert hasattr(ti, "globalMMATileGrid")
    assert hasattr(ti, "localMMATileGrid")
    assert hasattr(ti, "subtileShape")
    assert hasattr(ti, "globalSubtileGrid")
    assert hasattr(ti, "localSubtileGrid")
    assert hasattr(ti, "subtileSize")
    assert hasattr(ti, "subtileLocalTotalCount")

    # Sanity: localSubtileGrid correctly computed.
    assert ti.subtileLocalTotalCount == ti.localSubtileGrid[0] * ti.localSubtileGrid[1]

    # Lines 522-524: loadWidth fields.
    assert hasattr(ti, "loadWidthGR")
    assert hasattr(ti, "loadWidthLR")
    assert ti.loadWidthGR > 0


# ---------------------------------------------------------------------------
# 3. Grid utility methods (lines 572-603)
# ---------------------------------------------------------------------------

def test_r5_tileinfo_grid_methods_via_cdtile():
    """TileInfo grid utility methods via CDTileGeometry instance.

    Exercises:
      572-573  getLocalSubtileLinearId
      575-578  getLocalSubtileIdFromLinearId
      580-581  getLocalMMATileLinearId
      583-585  getLocalSubtileIdFromMMATile
    """
    from Tensile.Components.Subtile.Kernel import TileInfo, CD_F32

    kernel = _kernel_cd(macroTile0=128, macroTile1=128, miwavegroup=(2, 2))
    writer = _mock_writer()
    ti = TileInfo(CD_F32, "C", writer, kernel)

    # CDTileGeometry may store localSubtileGrid values as floats; cast for range().
    g0 = int(ti.localSubtileGrid[0])
    g1 = int(ti.localSubtileGrid[1])

    # getLocalSubtileLinearId — row-major linearization (line 572-573)
    for sId0 in range(g0):
        for sId1 in range(g1):
            lin = ti.getLocalSubtileLinearId(sId0, sId1)
            assert lin == sId1 * g0 + sId0, (
                f"linearId mismatch at ({sId0},{sId1}): got {lin}"
            )

    # getLocalSubtileIdFromLinearId — round-trip (lines 575-578)
    for lin_id in range(g0 * g1):
        sId0, sId1 = ti.getLocalSubtileIdFromLinearId(lin_id)
        assert sId0 == lin_id % g0
        assert sId1 == lin_id // g0

    # getLocalMMATileLinearId (lines 580-581)
    m0 = int(ti.localMMATileGrid[0])
    m1 = int(ti.localMMATileGrid[1])
    for mId0 in range(m0):
        for mId1 in range(m1):
            lin = ti.getLocalMMATileLinearId(mId0, mId1)
            assert lin == mId1 * m0 + mId0

    # getLocalSubtileIdFromMMATile (lines 583-585)
    st = ti.subtileShape
    st0 = int(st[0])
    st1 = int(st[1])
    for mId0 in range(m0):
        for mId1 in range(m1):
            s0, s1 = ti.getLocalSubtileIdFromMMATile(mId0, mId1)
            assert s0 == mId0 // st0
            assert s1 == mId1 // st1


def test_r5_tileinfo_grid_methods_via_abtile():
    """TileInfo grid utility methods via ABTilePair (AB_B16) instance.

    Uses the standard TN TilePair (not TLU1) to exercise grid methods.
    Covers same lines as CDTile variant but from a different TileInfo path.
    """
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16

    kernel = _kernel_ab(macroTileA=128, macroTileB=128, depthUA=64, depthUB=64,
                        miwavegroup=(2, 2))
    writer = _mock_writer()
    ti = TileInfo(AB_B16, "A", writer, kernel)

    g0 = ti.localSubtileGrid[0]
    g1 = ti.localSubtileGrid[1]

    # getLocalSubtileLinearId (line 572-573)
    lin = ti.getLocalSubtileLinearId(0, 0)
    assert lin == 0
    if g0 > 1 or g1 > 1:
        lin2 = ti.getLocalSubtileLinearId(g0 - 1, g1 - 1)
        assert lin2 == (g1 - 1) * g0 + (g0 - 1)

    # getLocalSubtileIdFromLinearId (lines 575-578)
    for lid in range(g0 * g1):
        s0, s1 = ti.getLocalSubtileIdFromLinearId(lid)
        assert s0 == lid % g0
        assert s1 == lid // g0

    # getLocalMMATileLinearId (lines 580-581)
    m0 = ti.localMMATileGrid[0]
    m1 = ti.localMMATileGrid[1]
    lin = ti.getLocalMMATileLinearId(0, 0)
    assert lin == 0

    # getLocalSubtileIdFromMMATile (lines 583-585)
    st = ti.subtileShape
    s0, s1 = ti.getLocalSubtileIdFromMMATile(0, 0)
    assert s0 == 0
    assert s1 == 0
