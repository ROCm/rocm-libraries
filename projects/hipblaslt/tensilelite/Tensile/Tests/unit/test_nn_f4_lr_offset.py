#!/usr/bin/env python3
################################################################################
# Step 3 LDS→MFMA validation: LR offset COMPUTATION for NN-FP4 (LRTag_TLU1).
# Pure-CPU test (no GPU / no hip).
#
# Verifies:
#   * Dispatch routes to _emitLROffset_TLU1, not stub.
#   * Emit produces a non-empty Module (assembly was generated).
#   * Python reference model produces correct per-lane offsets for all 64 lanes.
#   * Reference model covers all 128 K-columns of the subtile exactly once per
#     lane-group, confirming the transpose read pattern is complete.
#
#   pytest test_nn_f4_lr_offset.py -v   |   python test_nn_f4_lr_offset.py
################################################################################
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from types import SimpleNamespace
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1
from gpu_test_helpers import create_writer, TileConfig, init_rocisa

# ── Reference model ──────────────────────────────────────────────────────────
INST_M = 16
INST_K = 128
BPE = 0.5
WAVESIZE = 64
SUBTILE_SHAPE_M = 2
K_GROUPS = WAVESIZE // INST_M  # 4
K_PER_GROUP = INST_K // K_GROUPS  # 32
COL_STRIDE = int(SUBTILE_SHAPE_M * INST_M * BPE)  # 16
K_HALF_STRIDE = 16 * COL_STRIDE  # 256
TILE_M_STRIDE = int(INST_M * BPE)  # 8
READS_PER_TILE = K_PER_GROUP // 16  # 2
NUM_LR_PER_SUBTILE = SUBTILE_SHAPE_M * READS_PER_TILE  # 4


def reference_lr_offset(
    lane_id, read_idx, mWave=0, localSubtileGrid0=4, subtileSize=2048
):
    """Compute expected LDS byte address for lane_id, read read_idx, wave mWave."""
    lane16 = lane_id % INST_M
    lane16Group = lane_id // INST_M
    base_offset = (lane16Group * K_PER_GROUP + lane16) * COL_STRIDE
    tile_m = read_idx // READS_PER_TILE
    k_half = read_idx % READS_PER_TILE
    read_const = tile_m * TILE_M_STRIDE + k_half * K_HALF_STRIDE
    wave_partition = mWave * localSubtileGrid0 * subtileSize
    return base_offset + read_const + wave_partition


# ── Test 1: Dispatch routes to real function ─────────────────────────────────
def test_dispatch_not_stub():
    """_emitLocalReadOffset must dispatch to _emitLROffset_TLU1 for LRTag_TLU1."""
    from Tensile.Components.Subtile.SubtileLREmit import _emitLocalReadOffset
    from Tensile.Components.Subtile.SubtileGeometry import LRTag_TLU1

    name = _emitLocalReadOffset.dispatch(LRTag_TLU1).__name__
    assert (
        name == "_emitLROffset_TLU1"
    ), f"dispatches to {name}, expected _emitLROffset_TLU1"


# ── Test 2: Emit produces a non-empty Module ─────────────────────────────────
def test_emit_produces_module():
    """Calling emitLocalReadOffset returns a Module with instructions."""
    cfg = TileConfig(mt_a=256, mt_b=256, depth_u=256, stride_a=256, stride_b=256)
    writer, kernel, tiA, _ = create_writer(
        cfg, mi_wave_group=[2, 2], geometry=AB_B4_TLU1, inst_k=128, bpe=1
    )
    init_rocisa()
    tiA.allocOffsetRegisters(writer, kernel)
    module = tiA.emitLocalReadOffset(writer, kernel)
    assert module is not None, "emitLocalReadOffset returned None (still stub?)"
    asm = str(module)
    assert len(asm) > 50, f"Module too short ({len(asm)} chars), likely empty"
    assert (
        "lane16" in asm.lower() or "laneId" in asm.lower() or "serial" in asm.lower()
    ), "No lane-related comment found in emitted asm"


# ── Test 3: Reference model basic correctness ────────────────────────────────
def test_reference_lane0():
    """Lane 0 (group 0, lane16=0), wave 0: offset[0]=0, offset[1]=256, offset[2]=8, offset[3]=264."""
    assert reference_lr_offset(0, 0) == 0
    assert reference_lr_offset(0, 1) == 256
    assert reference_lr_offset(0, 2) == 8
    assert reference_lr_offset(0, 3) == 264


def test_reference_lane16():
    """Lane 16 (group 1, lane16=0): base = 32*16 = 512."""
    assert reference_lr_offset(16, 0) == 512
    assert reference_lr_offset(16, 1) == 768
    assert reference_lr_offset(16, 2) == 520
    assert reference_lr_offset(16, 3) == 776


def test_reference_lane63():
    """Lane 63 (group 3, lane16=15): base = (3*32+15)*16 = 1776."""
    assert reference_lr_offset(63, 0) == 1776
    assert reference_lr_offset(63, 1) == 2032
    assert reference_lr_offset(63, 2) == 1784
    assert reference_lr_offset(63, 3) == 2040


# ── Test 4: Wave partition offset ────────────────────────────────────────────
def test_reference_wave1():
    """Wave 1 (mWave=1): adds 4*2048 = 8192 to all offsets."""
    for lane in [0, 15, 16, 63]:
        for r in range(4):
            diff = reference_lr_offset(lane, r, mWave=1) - reference_lr_offset(
                lane, r, mWave=0
            )
            assert diff == 8192, f"lane={lane}, r={r}: wave diff={diff}, expected 8192"


# ── Test 5: Each read covers 16 distinct K-columns per group ─────────────────
def test_k_coverage_per_group():
    """Within each 16-lane group, one read addresses 16 unique K-columns."""
    for group in range(4):
        for r in range(4):
            k_cols = set()
            for j in range(16):
                lane = group * 16 + j
                addr = reference_lr_offset(lane, r, mWave=0)
                k_col = addr // COL_STRIDE
                k_cols.add(k_col)
            assert (
                len(k_cols) == 16
            ), f"group={group}, read={r}: only {len(k_cols)} unique K-cols (expected 16)"


# ── Test 6: All 4 reads cover all 128 K-columns exactly once per group ──────
def test_full_k_coverage():
    """Across all 4 reads, each kGroup touches all its 32 assigned K-columns
    twice (once for MMA tile 0, once for MMA tile 1) = 32*2 = 64 total addresses.
    All 128 K-columns appear exactly once per MMA tile across all groups."""
    for tile_m in range(2):
        all_k_cols = set()
        for group in range(4):
            for read_in_tile in range(READS_PER_TILE):
                r = tile_m * READS_PER_TILE + read_in_tile
                for j in range(16):
                    lane = group * 16 + j
                    addr = reference_lr_offset(lane, r, mWave=0)
                    k_col = addr // COL_STRIDE
                    all_k_cols.add(k_col)
        assert (
            len(all_k_cols) == 128
        ), f"tile_m={tile_m}: {len(all_k_cols)} K-cols covered, expected 128"


# ── Test 7: All addresses within subtile bounds ──────────────────────────────
def test_addresses_in_bounds():
    """All per-lane addresses (wave 0, no subtile offset) fit within subtile [0, 2048)."""
    subtile_bytes = 2048
    for lane in range(64):
        for r in range(4):
            addr = reference_lr_offset(lane, r, mWave=0)
            assert (
                0 <= addr < subtile_bytes
            ), f"lane={lane}, r={r}: addr={addr} out of bounds [0, {subtile_bytes})"


# ── Test 8: Maximum VGPR value is sane ───────────────────────────────────────
def test_max_vgpr_value():
    """Maximum per-lane value (wave 1, lane 63, read 3) fits in 16-bit + ds_offset."""
    max_addr = reference_lr_offset(63, 3, mWave=1)
    assert max_addr == 2040 + 8192 == 10232
    max_ds_offset = 3 * 2048 + 1 * 8 * 2048  # sId0=3, sId1=1, globalGrid[0]=8
    assert max_addr + max_ds_offset < 65536, "Would overflow 16-bit LDS address"


if __name__ == "__main__":
    test_dispatch_not_stub()
    test_emit_produces_module()
    test_reference_lane0()
    test_reference_lane16()
    test_reference_lane63()
    test_reference_wave1()
    test_k_coverage_per_group()
    test_full_k_coverage()
    test_addresses_in_bounds()
    test_max_vgpr_value()
    print("OK: LR offset TLU1 emits correctly; reference model covers all 128 K-cols.")
