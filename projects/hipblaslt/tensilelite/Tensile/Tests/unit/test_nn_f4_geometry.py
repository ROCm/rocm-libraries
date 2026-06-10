#!/usr/bin/env python3
################################################################################
# Step 1 validation: NN-FP4 column-major (TLU=1) GR geometry.
#
# Pure-CPU test (no GPU, no hip). Builds TileInfo for AB_B4_TLU1 with the
# tr_sample_f4.yaml config and asserts the derived load accounting covers the
# ENTIRE A macro tile (numGRTotal == 8).
#
#   pytest test_nn_f4_geometry.py -v
#   python  test_nn_f4_geometry.py
################################################################################
import os
import sys

# Make the tensilelite package importable when run standalone.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from types import SimpleNamespace
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1
from Tensile.Components.Subtile.SubtileGeometry import (
    ABGRGeometry,
    ABLRGeometry,
    ABTilePair,
    LoadShape,
    GRTag_TLU1,
    LRTag_TLU1,
    MFMA_16x16_1B_4K_4V,
)

_BPE = 0.5  # fp4


# tr_sample_f4.yaml: MT=256x256, DepthU=256, MI=16x16x128, MIWaveGroup=[2,2].
def _tr_sample_kernel():
    return {
        "DepthU": 256,
        "_DepthU": 256,
        "_DepthUA": 256,
        "_DepthUB": 256,
        "MacroTileA": 256,
        "MacroTileB": 256,
        "MacroTile0": 256,
        "MacroTile1": 256,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 128,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "UseSubtileImpl": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "ProblemType": {"DataTypeA": None, "DataTypeB": None, "ComputeDataType": None},
    }


def _build_tileinfo_A(geometry):
    kernel = _tr_sample_kernel()
    writer = SimpleNamespace()  # TileInfo.__init__ derives from kernel only
    return TileInfo(geometry, "A", writer, kernel), kernel


def test_ab_b4_tlu1_geometry_numbers():
    """Derived geometry matches the hand-audited values."""
    tiA, _ = _build_tileinfo_A(AB_B4_TLU1)
    assert list(tiA.mmaTileShape) == [16, 128]
    assert tiA.mmaTileSize == 1024  # 16*128*0.5
    assert list(tiA.globalMMATileGrid) == [16, 2]
    assert list(tiA.localMMATileGrid) == [8, 2]
    assert list(tiA.subtileShape) == [2, 1]
    assert tiA.subtileCount == 2  # DERIVED = MIWaveGroup[0]
    assert tiA.subtileStride == 8  # DERIVED = (256/16)/2
    assert [int(x) for x in tiA.globalSubtileGrid] == [8, 2]
    assert list(tiA.localSubtileGrid) == [4, 2]
    assert tiA.subtileSize == 2048  # 2*1*1024
    assert tiA.loadRatioGR == 1.0
    assert tiA.numGRPerSubtile == 1
    assert tiA.numGRTotal == 8  # == 4 * subtileCount


def test_ab_b4_tlu1_covers_all_of_A():
    """loads x lanes x bytes x waves == every byte of the A macro tile."""
    tiA, kernel = _build_tileinfo_A(AB_B4_TLU1)
    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    bytes_loaded = tiA.numGRTotal * kernel["WavefrontSize"] * tiA.loadWidthGR * numWaves
    bytes_in_A = int(kernel["MacroTileA"] * kernel["DepthU"] * _BPE)
    assert bytes_loaded == bytes_in_A == 32768


def test_pinned_subtilecount_would_undercover():
    """Regression guard: pinning subtileCount=1 (bf16-TLU1 style) loads HALF of A.
    Documents exactly why AB_B4_TLU1 must leave count/stride derived."""
    _B4 = dict(
        mmaLayout=MFMA_16x16_1B_4K_4V, instK=128, bpe=0.5, supportedTypes=("fp4",)
    )
    bad = ABTilePair(
        gr=ABGRGeometry(
            tag=GRTag_TLU1(),
            **_B4,
            tlu=True,
            subtileShape=(8, 1),
            subtileCount=1,
            subtileStride=0,
            loadShape=LoadShape(m=32, k=1)
        ),
        lr=ABLRGeometry(
            tag=LRTag_TLU1(),
            **_B4,
            tlu=True,
            subtileShape=(8, 1),
            loadShape=LoadShape(m=32, k=1)
        ),
    )
    tiBad, kernel = _build_tileinfo_A(bad)
    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    bytes_loaded = (
        tiBad.numGRTotal * kernel["WavefrontSize"] * tiBad.loadWidthGR * numWaves
    )
    assert tiBad.numGRTotal == 4
    assert bytes_loaded == 16384  # only HALF of A
    assert bytes_loaded != int(kernel["MacroTileA"] * kernel["DepthU"] * _BPE)


if __name__ == "__main__":
    test_ab_b4_tlu1_geometry_numbers()
    test_ab_b4_tlu1_covers_all_of_A()
    test_pinned_subtilecount_would_undercover()
    print(
        "OK: AB_B4_TLU1 covers all of A (numGRTotal=8); pinned count would load half."
    )
