#!/usr/bin/env python3
################################################################################
# GPU functional test for lraTileAssignment with parameterized tile configs
#
# Usage:
#   pytest test_lraTileAssignment.py -v -s
################################################################################

import os
import sys
import tempfile

import pytest
from types import SimpleNamespace

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    BPE, LOAD_WIDTH, WAVESIZE, NUM_THREADS, NUM_WAVES,
    create_writer_for_gpu,
    init_rocisa,
    build_and_run,
    print_offset_grid,
)
from Tensile.Components.SubtileBasedKernel import lraTileAssignment


def generate_lra_asm(cfg):
    """Run lraTileAssignment and return (lra_asm, tileInfoA, tileInfoB, kernel)."""
    writer, kernel, tileInfoA, tileInfoB = create_writer_for_gpu(cfg)
    init_rocisa()

    module = lraTileAssignment(writer, kernel)
    lra_asm = str(module)
    return lra_asm, tileInfoA, tileInfoB, kernel


# ---- Reference implementations ----

def compute_expected_lr_offset(thread_id, cfg, tileInfo):
    """Python reference implementation for LR (Local Read) offset computation.
    """
    depthUBytes = cfg.depth_u * BPE
    MT = cfg.mt_a if tileInfo.tc == 'A' else cfg.mt_b
    blockSize = depthUBytes // LOAD_WIDTH
    numRowsPerLDSBanks = (WAVESIZE*4) // depthUBytes

    waveReadSize = WAVESIZE*LOAD_WIDTH
    numRowsPerHalfWave = WAVESIZE // blockSize // 2 # split wave load
    numMFMACols = tileInfo.mmaTileShape[1]*tileInfo.bpe // LOAD_WIDTH
    laneId = thread_id % WAVESIZE

    # Contiguous rows for loadRatioGR == 2.0, interleaved rows for loadRatioGR <= 1.0
    if tileInfo.loadRatioGR == 2.0: 
        splitOffset = 0
    else:
        splitOffset = ((laneId % 16) // numRowsPerHalfWave)*(waveReadSize//2)

    enableSwizzling = True
    if enableSwizzling:
        enableRotation = True
        # Swap lanes by 16 (stride of consecutive cols for MFMA layout)
        if (laneId % 4)//2 == 0:
            if (laneId // 16) % 2 == 0:
                laneId = (laneId + 16) % WAVESIZE
            else:
                laneId = (laneId - 16) % WAVESIZE
    else:
        enableRotation = False

    lane16 = laneId % 16
    lane16Group = laneId // 16

    colOffset = lane16Group
    if enableRotation:
        # rotate by 2 rows every 2 lds_row_id
        lds_row_id = lane16 // numRowsPerLDSBanks
        rotation = (lds_row_id//2)*2 
        colOffset = (colOffset+rotation) % blockSize

    rowOffset = lane16 * depthUBytes + splitOffset
        
    offsets = []
    # offset by numMFMACols read along K (TN)
    for lr_idx in range(tileInfo.numLRPerSubtile):
        newColOffset = (numMFMACols*lr_idx+ colOffset) % blockSize
        offsets.append(rowOffset+ newColOffset*LOAD_WIDTH)


    # Wave partitioning.
    waveId = thread_id // WAVESIZE
    partitionOffset = 0

    # 2x2 config. Partitionning depends on tc.
    if tileInfo.loadRatioGR == 1.0:
        # W0 W2
        # W1 W3
        if tileInfo.tc == 'A':
            #W1/W3 get offset for rows 16-31
            if waveId % 2 == 1: 
                partitionOffset = numRowsPerHalfWave*depthUBytes
        elif tileInfo.tc == 'B':
            #W2/W3 get offset for rows 16-31
            if (waveId //2)%2 == 1:
                partitionOffset = numRowsPerHalfWave*depthUBytes
    # 1x4 config:
    # W0
    # W1 
    # W2
    # W3
    # 1st buffer load (i, i + MT/2)
    # 2nd buffer load (i+MT/4, i + MT/2 + MT/4)
    # offset W2/W3 by numRowsPerHalfWave*depthUBytes to get i + MT/2
    # offset W1/W3 by MT*depthUBytes//4 to get i + MT/4
    elif tileInfo.loadRatioGR == 0.5:
        if (waveId // 2) % 2 == 1:
            partitionOffset += numRowsPerHalfWave*depthUBytes
        if waveId % 2 == 1:
            partitionOffset += MT * depthUBytes // 4 
    elif tileInfo.loadRatioGR > 2.0:
        raise NotImplementedError("Unsupported loadRatioGR > 2.0 in reference implementation")
          
    if tileInfo.tc == 'B':
        partitionOffset+= cfg.mt_a * depthUBytes # B is after A in memory
    for id in range(len(offsets)):
        offsets[id] += partitionOffset


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
        lra_asm, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg)
        return SimpleNamespace(
            cfg=cfg,
            lra_asm=lra_asm,
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
        lra_asm, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg)
        return SimpleNamespace(
            cfg=cfg,
            lra_asm=lra_asm,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            kernel=kernel,
            tmp_path=tmp_path,
        )

    def test_offset_a(self, lra_env):
        """Validate all sharedVgprLROffset vgprs for matrix A across all threads."""
        cfg = lra_env.cfg
        for idx, reg in enumerate(lra_env.tileInfoA.sharedVgprLROffset):
            results = build_and_run(lra_env.lra_asm, reg, False, cfg, lra_env.tmp_path,
                                    f"lr_offsetA_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset(tid, cfg, lra_env.tileInfoA)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] A LR offset[{idx}] v{reg} mismatch at tid={tid}: " \
                    f"got {results[tid]}, expected {expected[idx]}"

    def test_offset_b(self, lra_env):
        """Validate all sharedVgprLROffset vgprs for matrix B across all threads."""
        cfg = lra_env.cfg
        for idx, reg in enumerate(lra_env.tileInfoB.sharedVgprLROffset):
            results = build_and_run(lra_env.lra_asm, reg, False, cfg, lra_env.tmp_path,
                                    f"lr_offsetB_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset(tid, cfg, lra_env.tileInfoB)
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

        lra_asm, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg)

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
                        results = build_and_run(lra_asm, reg, False, cfg, tmp_path,
                                                f"lr_offset{tc}_v{reg}_{cfg.label}")

                        if args.grid:
                            print_offset_grid(f"Matrix {tc} LR GPU offset[{idx}] v{reg} ({cfg.label})",
                                              results, WAVESIZE, NUM_WAVES)

                            if args.debug:
                                expected = [compute_expected_lr_offset(tid, cfg, tileInfo)[idx]
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
                            exp = compute_expected_lr_offset(tid, cfg, tileInfo)[idx]
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
