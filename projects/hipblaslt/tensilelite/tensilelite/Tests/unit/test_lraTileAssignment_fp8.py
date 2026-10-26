#!/usr/bin/env python3
################################################################################
# GPU functional test for lraTileAssignment with FP8 tile configs
#
# Config: MT=256x256, DU=128, 2x2 wave group
#   bpe=1, instK=128, loadWidth=16, numLRPerSubtile=2, loadRatioGR=1.0
#
# FP8 LR swizzle = block-swap + wave de-rotation (zero LDS bank conflicts):
#   lane16      = laneId % 16          (M-row within MFMA tile)
#   lane16Group = laneId // 16         (K-group, range 0-3)
#   rotation    = 2 * (lane16 >> 3)   (undo GR wave rotation for M-rows 8..15)
#   finalColId  = (lane16Group + rotation) % 4
#   swap_bit    = (lane16 >> 1) & 1
#   colOffset_0 = finalColId + swap_bit * 4
#   colOffset_1 = colOffset_0 ^ 4
#   offset[i]   = colOffset_i * 16 + lane16 * 128
#
# Usage:
#   pytest test_lraTileAssignment_fp8.py -v -s
################################################################################

import os
import tempfile

import pytest
from types import SimpleNamespace

from gpu_test_helpers import (
    HAS_GFX950,
    GFX_TARGET,
    TileConfig,
    LOAD_WIDTH, WAVESIZE, NUM_THREADS, NUM_WAVES,
    AB_B8,
    generate_lra_asm,
    export_register,
    print_offset_grid,
)

BPE = 1  # fp8: 1 byte per element


# ---- Reference implementation ----

def compute_expected_lr_offset_fp8(thread_id, cfg, tileInfo, ldsStartOffsetB=None):
    """FP8 reference for lraTileAssignment LR byte offset.

    Two ds_read_b128 per MFMA (load_idx=0 and load_idx=1) use complementary
    block assignments:
      rotation    = 2 * (lane16 >> 3)        [undo GR wave rotation for M-rows >= 8]
      finalColId  = (lane16Group + rotation) % 4
      colOffset_0 = finalColId + swap_bit * 4
      colOffset_1 = colOffset_0 ^ 4
    where swap_bit = (lane16 >> 1) & 1
    """
    depthUBytes = cfg.depth_u * BPE   # 128
    mi_m = 16

    laneId = thread_id % WAVESIZE
    waveId = thread_id // WAVESIZE

    lane16      = laneId & (mi_m - 1)        # laneId % 16
    lane16Group = laneId >> 4                 # laneId // 16

    rotation   = 2 * (lane16 >> 3)
    finalColId = (lane16Group + rotation) & 3
    swap_bit   = (lane16 >> 1) & 1
    swap_val   = swap_bit * 4

    colOffset0 = finalColId + swap_val
    colOffset1 = colOffset0 ^ 4

    rowOffset = lane16 * depthUBytes

    offset0 = colOffset0 * LOAD_WIDTH + rowOffset
    offset1 = colOffset1 * LOAD_WIDTH + rowOffset

    # Wave partitioning
    MT = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
    if tileInfo.loadRatioGR >= 2.0:
        partitionOffset = 0
    elif tileInfo.loadRatioGR == 1.0:
        wavePartId = waveId & 1 if tileInfo.tc == 'A' else waveId >> 1
        partitionOffset = wavePartId * (MT * depthUBytes // 2)
    elif tileInfo.loadRatioGR == 0.5:
        partitionOffset = waveId * (MT * depthUBytes // 4)
    else:
        raise NotImplementedError(f"Unsupported loadRatioGR: {tileInfo.loadRatioGR}")

    if tileInfo.tc == 'B':
        partitionOffset += ldsStartOffsetB if ldsStartOffsetB is not None else cfg.mt_a * depthUBytes

    return [offset0 + partitionOffset, offset1 + partitionOffset]


TILE_CONFIGS_FP8 = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=512,  stride_b=512),
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=4096, stride_b=1024),
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=256,  stride_b=4096),
]


# ---- Pytest tests ----

@pytest.mark.skipif(not HAS_GFX950, reason=f"GPU tests require gfx950, found {GFX_TARGET}")
class TestLraTileAssignmentFP8GPU:

    @pytest.fixture(params=TILE_CONFIGS_FP8, ids=lambda c: c.label)
    def lra_env(self, request, tmp_path):
        cfg = request.param
        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg, geometry=AB_B8, inst_k=128, bpe=BPE)
        return SimpleNamespace(
            cfg=cfg, lra_asm=lra_asm, writer=writer,
            tileInfoA=tileInfoA, tileInfoB=tileInfoB,
            kernel=kernel, tmp_path=tmp_path,
        )

    def test_offset_a(self, lra_env):
        cfg = lra_env.cfg
        for idx, reg in enumerate(lra_env.tileInfoA.sharedVgprLROffset):
            results = export_register(
                lra_env.writer, lra_env.lra_asm, reg, False,
                cfg, lra_env.tmp_path, f"lr_offsetA_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset_fp8(
                    tid, cfg, lra_env.tileInfoA, lra_env.writer.ldsStartOffsetB)
                assert results[tid] == expected[idx], (
                    f"[{cfg.label}] A LR offset[{idx}] v{reg} mismatch at tid={tid}: "
                    f"got {results[tid]}, expected {expected[idx]}"
                )

    def test_offset_b(self, lra_env):
        cfg = lra_env.cfg
        for idx, reg in enumerate(lra_env.tileInfoB.sharedVgprLROffset):
            results = export_register(
                lra_env.writer, lra_env.lra_asm, reg, False,
                cfg, lra_env.tmp_path, f"lr_offsetB_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset_fp8(
                    tid, cfg, lra_env.tileInfoB, lra_env.writer.ldsStartOffsetB)
                assert results[tid] == expected[idx], (
                    f"[{cfg.label}] B LR offset[{idx}] v{reg} mismatch at tid={tid}: "
                    f"got {results[tid]}, expected {expected[idx]}"
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for FP8 lraTileAssignment")
    parser.add_argument("--grid",  action="store_true", help="Print offset grid")
    parser.add_argument("--debug", action="store_true", help="Print expected grid + diffs")
    args = parser.parse_args()
    if args.debug:
        args.grid = True

    for cfg in TILE_CONFIGS_FP8:
        print(f"\n{'='*60}")
        print(f"  FP8 LRA Config: {cfg.label}")
        print(f"{'='*60}")
        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm(cfg, geometry=AB_B8, inst_k=128, bpe=BPE)

        print(f"  numLRPerSubtile A={tileInfoA.numLRPerSubtile}, B={tileInfoB.numLRPerSubtile}")
        print(f"  loadRatioGR A={tileInfoA.loadRatioGR}, B={tileInfoB.loadRatioGR}")
        print(f"  ldsStartOffsetB={writer.ldsStartOffsetB}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            if HAS_GFX950:
                for tc, tileInfo in [("A", tileInfoA), ("B", tileInfoB)]:
                    for idx, reg in enumerate(tileInfo.sharedVgprLROffset):
                        results = export_register(writer, lra_asm, reg, False, cfg,
                                                      tmp_path, f"lr_offset{tc}_v{reg}_{cfg.label}")
                        expected_all = [compute_expected_lr_offset_fp8(
                                            tid, cfg, tileInfo, writer.ldsStartOffsetB)
                                        for tid in range(NUM_THREADS)]
                        if args.grid:
                            print_offset_grid(
                                f"Matrix {tc} LR GPU offset[{idx}] v{reg} ({cfg.label})",
                                results, WAVESIZE, NUM_WAVES)
                        if args.debug:
                            expected = [e[idx] for e in expected_all]
                            print_offset_grid(
                                f"Matrix {tc} LR EXPECTED offset[{idx}] ({cfg.label})",
                                expected, WAVESIZE, NUM_WAVES)
                            mismatches = [tid for tid in range(NUM_THREADS)
                                          if results[tid] != expected[tid]]
                            if mismatches:
                                print(f"  {len(mismatches)} mismatches:")
                                for tid in mismatches[:20]:
                                    print(f"    tid={tid}: got {results[tid]}, "
                                          f"expected {expected[tid]}")
                            else:
                                print(f"  All {NUM_THREADS} threads match.")
                        errors = sum(1 for tid in range(NUM_THREADS)
                                     if results[tid] != expected_all[tid][idx])
                        print(f"  {tc} LR offset[{idx}] v{reg}: {errors} errors / {NUM_THREADS}")
            else:
                print(f"GPU tests require gfx950, found {GFX_TARGET} — assembly only")
