#!/usr/bin/env python3
################################################################################
# Step 1 LDS→MFMA validation: LR geometry accounting for AB_B4_TLU1.
#
# Pure-CPU test (no GPU, no hip). Verifies that the LR load width and derived
# numLRPerSubtile match the ds_read_b64_tr_b4 instruction (8 bytes per lane,
# 4 reads per subtile).
#
#   pytest test_nn_f4_lr_geometry.py -v
#   python  test_nn_f4_lr_geometry.py
################################################################################
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from types import SimpleNamespace
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1


def _tr_sample_kernel():
    """MT=256×256, DepthU=256, MI=16×16×128, MIWaveGroup=[2,2]."""
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
        "ProblemType": {
            "DataTypeA": None,
            "DataTypeB": None,
            "ComputeDataType": None,
        },
    }


def _build_tileinfo_A():
    kernel = _tr_sample_kernel()
    writer = SimpleNamespace()
    return TileInfo(AB_B4_TLU1, "A", writer, kernel), kernel


# ── Test 1: LR loadWidth matches ds_read_b64_tr_b4 ─────────────────────────
def test_lr_load_width():
    """ds_read_b64_tr_b4 reads 8 bytes per lane → loadWidthLR must be 8."""
    tiA, _ = _build_tileinfo_A()
    assert (
        tiA.loadWidthLR == 8
    ), f"loadWidthLR={tiA.loadWidthLR}, expected 8 (ds_read_b64 = 64 bits = 8 bytes)"


# ── Test 2: numLRPerSubtile matches 4 reads per subtile ────────────────────
def test_num_lr_per_subtile():
    """Each subtile is 2048 B; each ds_read covers 512 B → 4 reads needed."""
    tiA, _ = _build_tileinfo_A()
    assert tiA.numLRPerSubtile == 4, (
        f"numLRPerSubtile={tiA.numLRPerSubtile}, expected 4 "
        f"(subtile=2048 B / read=512 B = 4)"
    )


# ── Test 3: loadRatioLR is 0.25 ────────────────────────────────────────────
def test_load_ratio_lr():
    """loadRatioLR = bytesPerRead / subtileSize = 512/2048 = 0.25."""
    tiA, _ = _build_tileinfo_A()
    assert tiA.loadRatioLR == 0.25, f"loadRatioLR={tiA.loadRatioLR}, expected 0.25"


# ── Test 4: 4 reads cover the full subtile exactly ─────────────────────────
def test_lr_covers_full_subtile():
    """numLRPerSubtile × loadWidthLR × waveSize == lrSubtileSize."""
    tiA, kernel = _build_tileinfo_A()
    ws = kernel["WavefrontSize"]
    bytes_read = tiA.numLRPerSubtile * tiA.loadWidthLR * ws
    assert (
        bytes_read == tiA.lrSubtileSize == 2048
    ), f"LR bytes read={bytes_read}, subtileSize={tiA.lrSubtileSize}, expected 2048"


# ── Test 5: GR geometry is NOT affected ─────────────────────────────────────
def test_gr_unchanged():
    """Changing LR loadWidth must not affect any GR-derived values."""
    tiA, _ = _build_tileinfo_A()
    assert tiA.loadWidthGR == 16, f"loadWidthGR={tiA.loadWidthGR}, expected 16"
    assert tiA.numGRPerSubtile == 1
    assert tiA.numGRTotal == 8
    assert tiA.loadRatioGR == 1.0


# ── Test 6: VGPR budget cross-check ────────────────────────────────────────
def test_vgpr_budget():
    """4 ds_read × 2 VGPRs/read = 8 VGPRs = 2 tiles × 4 VGPRs/tile."""
    tiA, _ = _build_tileinfo_A()
    vgprs_per_read = 2  # ds_read_b64 → 64 bits → 2 VGPRs
    total_vgprs = tiA.numLRPerSubtile * vgprs_per_read
    expected = int(
        tiA.lrSubtileShape[0] * tiA.lrSubtileShape[1] * tiA.mmaTileRegCount
    )  # 2 * 1 * 4.0 = 8
    assert (
        total_vgprs == expected == 8
    ), f"total VGPRs={total_vgprs}, expected={expected}"


if __name__ == "__main__":
    test_lr_load_width()
    test_num_lr_per_subtile()
    test_load_ratio_lr()
    test_lr_covers_full_subtile()
    test_gr_unchanged()
    test_vgpr_budget()
    print(
        "OK: LR geometry for AB_B4_TLU1 is correct "
        "(loadWidth=8, numLRPerSubtile=4, 8 VGPRs/subtile)."
    )
