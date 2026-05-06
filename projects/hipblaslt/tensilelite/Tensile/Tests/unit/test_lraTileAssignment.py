#!/usr/bin/env python3
################################################################################
# GPU functional test for lraTileAssignment with parameterized tile configs
#
# Usage:
#   pytest test_lraTileAssignment.py -v -s
################################################################################

import os
import struct
import sys
import tempfile

import pytest
from types import SimpleNamespace

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    BPE, LOAD_WIDTH, WAVESIZE, NUM_THREADS, NUM_WAVES,
    create_writer,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
    generate_export_epilogue,
    print_offset_grid,
)
from Tensile.Components.Subtile.SubtileLREmit import lraTileAssignment

EXPORT_LOAD_PARAMS = (
    (4, 2, 0x00, "output_ptr"),
    ("StrideA0I", 1, 0x08, "strideA"),
    ("StrideB1J", 1, 0x0c, "strideB"),
)

EXPORT_ARGS = (
    ("output_ptr", 8, "global_buffer", "u32"),
    ("strideA",    4, "by_value",      "u32"),
    ("strideB",    4, "by_value",      "u32"),
)


def generate_lra_asm(cfg):
    """Run lraTileAssignment and return (lra_asm, writer, tileInfoA, tileInfoB, kernel)."""
    writer, kernel, tileInfoA, tileInfoB = create_writer(cfg)
    init_rocisa()

    # Reserve s0-s11 for hardware regs + kernarg loads
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    prologue = generate_load_params(EXPORT_LOAD_PARAMS)
    module = lraTileAssignment(writer, kernel)
    lra_asm = f"{prologue}\n{module}"
    return lra_asm, writer, tileInfoA, tileInfoB, kernel


def export_register(writer, test_asm, export_reg, is_sgpr, cfg, tmp_path, label):
    """Generate export kernel, assemble, run, return per-thread u32 results."""
    epilogue, allocated = generate_export_epilogue(writer, export_reg, is_sgpr)
    kernel_asm = generate_kernel_asm(f"{test_asm}\n{epilogue}", writer, EXPORT_ARGS)
    for v in allocated:
        writer.vgprPool.checkIn(v)

    raw = assemble_and_run(kernel_asm, tmp_path, label, NUM_THREADS * 4,
                           scalars=(cfg.stride_a, cfg.stride_b))
    return struct.unpack(f"{NUM_THREADS}I", raw)


# ---- Reference implementations ----

def compute_expected_lr_offset(thread_id, cfg, tileInfo, ldsStartOffsetB=None):
    """Python reference implementation matching lraTileAssignment kernel logic.

    Mirrors the kernel's exact steps:
      1. lane16, lane16Group from Serial
      2. Swizzling: rotation + VPermlane16SwapB32 on colOffset (always on)
      3. rowOffset = lane16 * depthUBytes (no splitOffset — commented out in kernel)
      4. _computeLROffset: colOffset stepping by numMFMACols
      5. Wave partitioning via _applyWavePartitionLROffset
      6. B matrix LDS base offset
    """
    depthUBytes = cfg.depth_u * BPE
    blockSize = depthUBytes // LOAD_WIDTH
    numRowsPerLDSBanks = (WAVESIZE * 4) // depthUBytes
    numMFMACols = tileInfo.mmaTileShape[1] * tileInfo.bpe // LOAD_WIDTH
    laneId = thread_id % WAVESIZE

    # --- Step 1: lane16, lane16Group ---
    lane16 = laneId & 15
    lane16Group = laneId >> 4

    # --- Step 2: Swizzling (always on in kernel) ---
    # Rotation
    lds_row_id = lane16 >> (numRowsPerLDSBanks.bit_length() - 1)
    rotation = (lds_row_id // 2) * 2
    colOffset = (rotation + lane16Group) % blockSize

    # VPermlane16SwapB32 with exec mask 0x33333333
    # Active lanes: those where (lane_pos % 4) < 2, i.e. lanes 0,1,4,5,8,9,12,13,...
    # For active lanes, swap colOffset value with lane (laneId XOR 16).
    # We need to compute colOffset for the partner lane too.
    partnerLaneId = laneId ^ 16
    partnerLane16 = partnerLaneId & 15  # same as lane16 (XOR 16 doesn't change lower 4 bits)
    partnerLane16Group = partnerLaneId >> 4
    partnerLdsRowId = partnerLane16 >> (numRowsPerLDSBanks.bit_length() - 1)
    partnerRotation = (partnerLdsRowId // 2) * 2
    partnerColOffset = (partnerRotation + partnerLane16Group) % blockSize

    # Check if this lane is active under exec mask 0x33333333
    isActive = (laneId % 4) < 2
    if isActive:
        colOffset = partnerColOffset  # swap: we get partner's value

    # --- Step 3: rowOffset (no splitOffset) ---
    rowOffset = lane16 * depthUBytes

    # --- Step 4: _computeLROffset ---
    offsets = []
    for lr_idx in range(tileInfo.numLRPerSubtile):
        if lr_idx == 0:
            newColOffset = colOffset
        else:
            newColOffset = (colOffset + numMFMACols * lr_idx) % blockSize
        offsets.append(newColOffset * LOAD_WIDTH + rowOffset)

    # --- Step 5: Wave partitioning (non-interleaved layout) ---
    waveId = thread_id // WAVESIZE
    partitionOffset = 0
    MT = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
    subIterKBytes = depthUBytes

    if tileInfo.loadRatioGR >= 2.0:
        pass  # no partition needed
    elif tileInfo.loadRatioGR == 1.0:
        # 2x2 config, non-interleaved: sInterval = MT * subIterKBytes // 2
        if tileInfo.tc == 'A':
            wavePartId = waveId & 1
        else:
            wavePartId = waveId >> 1
        partitionOffset = wavePartId * (MT * subIterKBytes // 2)
    elif tileInfo.loadRatioGR == 0.5:
        # 4x1 / 1x4 config, non-interleaved: sInterval = MT * subIterKBytes // 4
        partitionOffset = waveId * (MT * subIterKBytes // 4)
    else:
        raise NotImplementedError(f"Unsupported loadRatioGR: {tileInfo.loadRatioGR}")

    # --- Step 6: B matrix LDS base offset ---
    if tileInfo.tc == 'B':
        if ldsStartOffsetB is not None:
            partitionOffset += ldsStartOffsetB
        else:
            partitionOffset += cfg.mt_a * depthUBytes

    for i in range(len(offsets)):
        offsets[i] += partitionOffset

    return offsets


def compute_expected_lr_subtile(subtileId0, cfg, tileInfo):
    """Compute expected LR subtile register value.

    """
    subtile_rows = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    depthU_bytes = cfg.depth_u * BPE
    return subtile_rows * depthU_bytes * subtileId0


# Tile configs to test
TILE_CONFIGS = [
    # 2x2 configs
    TileConfig(mt_a=256, mt_b=256, depth_u=64),
    TileConfig(mt_a=96, mt_b=256, depth_u=64),
    # 1x4 configs
    TileConfig(mt_a=80, mt_b=64, depth_u=64),
    # 4x1 configs
    TileConfig(mt_a=64, mt_b=80, depth_u=64),
    TileConfig(mt_a=128, mt_b=240, depth_u=64),
]


# ---- Pytest tests ----

class TestLraTileAssignmentUnit:
    """Non-GPU unit tests for lraTileAssignment."""

    @pytest.fixture(params=TILE_CONFIGS, ids=lambda c: c.label)
    def lra_env(self, request):
        """Generate lraTileAssignment output once per tile config."""
        cfg = request.param
        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg)
        return SimpleNamespace(
            cfg=cfg,
            lra_asm=lra_asm,
            writer=writer,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            kernel=kernel,
        )

    def test_returns_valid_module(self, lra_env):
        """Verify lraTileAssignment returns a non-empty assembly string."""
        assert lra_env.lra_asm is not None
        assert len(lra_env.lra_asm) > 0

    def test_lr_offset_registers_allocated(self, lra_env):
        """Verify LR offset registers are allocated for both A and B."""
        for tileInfo in [lra_env.tileInfoA, lra_env.tileInfoB]:
            assert hasattr(tileInfo, 'sharedVgprLROffset'), \
                f"tileInfo{tileInfo.tc} missing sharedVgprLROffset"
            assert len(tileInfo.sharedVgprLROffset) == tileInfo.numLRPerSubtile, \
                f"tileInfo{tileInfo.tc}: expected {tileInfo.numLRPerSubtile} LR offset regs, " \
                f"got {len(tileInfo.sharedVgprLROffset)}"

    def test_tile_info_lr_consistency(self, lra_env):
        """Verify LR-related TileInfo fields are consistent."""
        for tileInfo in [lra_env.tileInfoA, lra_env.tileInfoB]:
            assert tileInfo.numLRPerSubtile >= 1, \
                f"tileInfo{tileInfo.tc}: numLRPerSubtile should be >= 1"
            assert tileInfo.numLRTotal >= 1, \
                f"tileInfo{tileInfo.tc}: numLRTotal should be >= 1"
            assert tileInfo.loadRatioLR > 0, \
                f"tileInfo{tileInfo.tc}: loadRatioLR should be > 0"


@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestLraTileAssignmentGPU:

    @pytest.fixture(params=TILE_CONFIGS, ids=lambda c: c.label)
    def lra_env(self, request, tmp_path):
        """Generate lraTileAssignment asm once per tile config."""
        cfg = request.param
        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg)
        return SimpleNamespace(
            cfg=cfg,
            lra_asm=lra_asm,
            writer=writer,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            kernel=kernel,
            tmp_path=tmp_path,
        )

    def test_offset_a(self, lra_env):
        """Validate all sharedVgprLROffset vgprs for matrix A across all threads."""
        cfg = lra_env.cfg
        for idx, reg in enumerate(lra_env.tileInfoA.sharedVgprLROffset):
            results = export_register(lra_env.writer, lra_env.lra_asm, reg, False,
                                      cfg, lra_env.tmp_path, f"lr_offsetA_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset(tid, cfg, lra_env.tileInfoA,
                                                      lra_env.writer.ldsStartOffsetB)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] A LR offset[{idx}] v{reg} mismatch at tid={tid}: " \
                    f"got {results[tid]}, expected {expected[idx]}"

    def test_offset_b(self, lra_env):
        """Validate all sharedVgprLROffset vgprs for matrix B across all threads."""
        cfg = lra_env.cfg
        for idx, reg in enumerate(lra_env.tileInfoB.sharedVgprLROffset):
            results = export_register(lra_env.writer, lra_env.lra_asm, reg, False,
                                      cfg, lra_env.tmp_path, f"lr_offsetB_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset(tid, cfg, lra_env.tileInfoB,
                                                      lra_env.writer.ldsStartOffsetB)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] B LR offset[{idx}] v{reg} mismatch at tid={tid}: " \
                    f"got {results[tid]}, expected {expected[idx]}"




if __name__ == "__main__":
    """Run standalone without pytest."""
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for lraTileAssignment")
    parser.add_argument("--grid", action="store_true",
                        help="Display offsets as 2D grid (waves x lanes) for A and B")
    parser.add_argument("--debug", action="store_true",
                        help="Display expected matrix in grid mode (implies --grid)")
    args = parser.parse_args()
    if args.debug:
        args.grid = True

    for cfg in TILE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Tile Config: {cfg.label}")
        print(f"{'='*60}")

        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            # Print the generated assembly for inspection
            print("\n--- Generated Assembly (lraTileAssignment section) ---")
            for line in lra_asm.split('\n'):
                print(line)
            print("--- End ---\n")

            print(f"  TileInfoA: numLRPerSubtile={tileInfoA.numLRPerSubtile}, "
                  f"loadRatioLR={tileInfoA.loadRatioLR}, "
                  f"sharedVgprLROffset={tileInfoA.sharedVgprLROffset}")
            print(f"  TileInfoB: numLRPerSubtile={tileInfoB.numLRPerSubtile}, "
                  f"loadRatioLR={tileInfoB.loadRatioLR}, "
                  f"sharedVgprLROffset={tileInfoB.sharedVgprLROffset}")

            if HAS_HIP:
                # Test all sharedVgprLROffset vgprs for both matrices
                for tc, tileInfo in [("A", tileInfoA), ("B", tileInfoB)]:
                    for idx, reg in enumerate(tileInfo.sharedVgprLROffset):
                        results = export_register(writer, lra_asm, reg, False, cfg, tmp_path,
                                                  f"lr_offset{tc}_v{reg}_{cfg.label}")

                        if args.grid:
                            print_offset_grid(f"Matrix {tc} LR GPU offset[{idx}] v{reg} ({cfg.label})",
                                              results, WAVESIZE, NUM_WAVES)

                            if args.debug:
                                expected = [compute_expected_lr_offset(tid, cfg, tileInfo, writer.ldsStartOffsetB)[idx]
                                            for tid in range(NUM_THREADS)]
                                print_offset_grid(f"Matrix {tc} LR EXPECTED offset[{idx}] ({cfg.label})",
                                                  expected, WAVESIZE, NUM_WAVES)

                                mismatches = sum(1 for t in range(NUM_THREADS)
                                                 if results[t] != expected[t])
                                if mismatches:
                                    print(f"\n--- Matrix {tc} LR offset[{idx}] DIFF ({mismatches} mismatches) ---")
                                    for w in range(NUM_WAVES):
                                        print(f"  w{w}: ", end="")
                                        for lane in range(WAVESIZE):
                                            tid = w * WAVESIZE + lane
                                            if results[tid] != expected[tid]:
                                                print(f" t{tid}:{results[tid]}!={expected[tid]}", end="")
                                        print()
                                else:
                                    print(f"\n  Matrix {tc} LR offset[{idx}]: all match.")

                        errors = 0
                        for tid in range(NUM_THREADS):
                            exp = compute_expected_lr_offset(tid, cfg, tileInfo, writer.ldsStartOffsetB)[idx]
                            if results[tid] != exp:
                                errors += 1
                                if not args.grid:
                                    print(f"  FAIL {tc} LR offset[{idx}] v{reg} tid={tid}: "
                                          f"got {results[tid]}, expected {exp}")
                            elif not args.grid and tid < 64:
                                print(f"  OK   {tc} LR offset[{idx}] v{reg} tid={tid}: {results[tid]}")

                        print(f"  Matrix {tc} LR offset[{idx}] v{reg}: "
                              f"{NUM_THREADS} threads, {errors} errors")
            else:
                print("HIP not available - assembly generated but not executed")
