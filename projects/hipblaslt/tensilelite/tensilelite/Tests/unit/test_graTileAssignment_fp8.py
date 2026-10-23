#!/usr/bin/env python3
################################################################################
# GPU functional test for graTileAssignment with FP8 tile configs
#
# Config: MT=256x256, DU=128, 2x2 wave group
#   bpe=1 (bytes per element, fp8), instM=16, instK=128, waveSize=64, numWaves=4, subtileShape=(1,1),
#   loadWidth=16 bytes (128-bit load)
#
#   loadRatioGR: fraction of the global GR tile covered by one cooperative buffer_load
#     (i.e., all waves issuing buffer_load together: loadWidth bytes per lane across all waves)
#     loadRatioGR      = bytesPerLoad / globalGRTileSize
#     bytesPerLoad    = loadWidth * waveSize * numWaves
#     globalGRTileSize = subtileSize * subtileCount
#       mmaTileSize  = bytes in one MMA tile (instM * instK elements) = instM * instK * bpe
#       subtileSize  = bytes in one subtile (subtileShape[0]*subtileShape[1] MMA tiles)
#                    = subtileShape[0] * subtileShape[1] * mmaTileSize
#       subtileCount = wg_m  (number of M-dimension wave partners, derived from wave group)
#     For 2x2 wave group: wg_m=2
#       mmaTileSize  = 16 * 128 * 1 = 2048
#       subtileSize  = 1 * 1 * 2048 = 2048
#       bytesPerLoad = 16 * 64 * 4 = 4096
#       globalGRTileSize = 2048 * 2 = 4096
#       loadRatioGR  = 4096 / 4096 = 1.0
#
#   blockSize: number of lanes needed to cover the full K depth (depthU * bpe bytes),
#     also defines the number of colIds in the K direction
#     blockSize = depthU * bpe / loadWidth = 128 * 1 / 16 = 8
#
#   numRowsPerLDSBanks: number of consecutive M rows that span one LDS bank period
#     (1 bank period = LDS_BANK_COUNT * LDS_BANK_WIDTH = 256 bytes)
#     after numRowsPerLDSBanks rows the bank assignment repeats
#     (e.g. row 2 maps to same banks as row 0, row 3 same as row 1)
#     numRowsPerLDSBanks = LDS_BANK_COUNT * LDS_BANK_WIDTH / (depthU * bpe)
#                        = 64 * 4 / 128 = 2
#
# FP8 GR swizzle = two-step zero-conflict scheme:
#   Step 1 (block-swap):
#     ldsRowId = (laneId >> 3) >> 1    # = laneId // 16, range 0..3
#     swap_bit = ldsRowId & 1          # 1 for ldsRowId in {1, 3}
#     colId ^= swap_bit * 4            # XOR with blockSize//2=4 for odd ldsRowId
#   Step 2 (wave K_group rotation, for loadRatioGR != 0.5):
#     rotation = (waveId & 1) * 2
#     colId = ((colId & 3) + rotation) % 4 + (colId & 4)
#
# Usage:
#   pytest test_graTileAssignment_fp8.py -v -s
################################################################################

import math
import os
import struct
import sys
import tempfile

import pytest
from types import SimpleNamespace
from gpu_test_helpers import (
    HAS_GFX950,
    GFX_TARGET,
    TileConfig,
    LOAD_WIDTH, WAVESIZE, NUM_THREADS, NUM_WAVES,
    AB_B8,
    generate_gra_asm,
    export_register,
    compute_expected_subtile,
    print_offset_grid,
)

# FP8 constants
BPE = 1        # fp8: 1 byte per element (shadows gpu_test_helpers.BPE=2)
LDS_BANK_COUNT = 64
LDS_BANK_WIDTH = 4  # bytes


# ---- Reference implementations ----

def compute_expected_offset_fp8(thread_id, cfg, tileInfo):
    """FP8 reference for graTileAssignment GR byte offset.

    Two-step zero-conflict scheme:
      Step 1 (block-swap):
        ldsRowId = (laneId >> 3) >> 1    (= laneId // 16)
        if ldsRowId & 1:  colId ^= 4     (XOR blockSize//2 = 4)
      Step 2 (wave K_group rotation, for loadRatioGR != 0.5):
        rotation = (waveId & 1) * 2
        colId = ((colId & 3) + rotation) % 4 + (colId & 4)
    """
    stride = cfg.stride_a if tileInfo.tc == 'A' else cfg.stride_b
    bpe = BPE                                # 1 for fp8
    depthUBytes = cfg.depth_u * bpe          # 128
    blockSize = depthUBytes // LOAD_WIDTH    # 8
    numRowsPerLDSBanks = (LDS_BANK_COUNT * LDS_BANK_WIDTH) // depthUBytes  # 64 * 4 = 256
    blockHalf = blockSize // 2               # 4: XOR mask to swap lower/upper block halves

    waveId = thread_id // WAVESIZE
    laneId = thread_id % WAVESIZE

    colId    = thread_id & (blockSize - 1)   # thread_id % 8
    rowInWave = laneId >> (blockSize.bit_length() - 1)   # laneId // 8
    ldsRowId  = rowInWave >> (numRowsPerLDSBanks.bit_length() - 1)  # rowInWave // 2

    # Step 1: block-swap (XOR colId with 4 for odd ldsRowId)
    if ldsRowId & 1:
        colId ^= blockHalf
    # Step 2: K_group rotation for M-rows 8..15 (odd waveId half)
    if tileInfo.loadRatioGR != 0.5:
        rotation = (waveId & 1) * 2
        colId = ((colId & 3) + rotation) % 4 + (colId & 4)

    colId_bytes = colId * LOAD_WIDTH

    # Wave row partition
    numRowsPerWave = WAVESIZE // blockSize   # 8
    partitionOffset = tileInfo.mmaTileShape[0] * tileInfo.localSubtileGrid[0]

    if tileInfo.loadRatioGR == 1.0:
        localRow    = waveId & 1
        partitionRow = waveId >> 1
    elif tileInfo.loadRatioGR == 0.5:
        localRow    = 0
        partitionRow = waveId
    elif tileInfo.loadRatioGR == 2.0:
        localRow    = waveId
        partitionRow = 0
    else:
        raise NotImplementedError(f"Unsupported loadRatioGR: {tileInfo.loadRatioGR}")

    localRow     = localRow << (numRowsPerWave.bit_length() - 1)
    waveRowOffset = localRow + partitionOffset * partitionRow

    rowId    = laneId >> (blockSize.bit_length() - 1)
    totalRow = rowId + waveRowOffset

    base = totalRow * stride * bpe + colId_bytes

    if tileInfo.numGRPerSubtile == 1:
        return [base]

    # Second GR offset: intra-block K_group +2 rotation (preserving block bit)
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    rowAdvance  = math.ceil(subtileSize * tileInfo.loadRatioGR)
    totalRow2   = totalRow + rowAdvance
    colId2      = ((colId & 3) + 2) % 4 + (colId & 4)
    offset2     = totalRow2 * stride * bpe + colId2 * LOAD_WIDTH
    return [base, offset2]




# ---- Tile configs: MT=256x256, DU=128, 2x2 WG ----

TILE_CONFIGS_FP8 = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=512,  stride_b=512,  use_swizzling=True),
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=4096, stride_b=1024, use_swizzling=True),
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=256,  stride_b=4096, use_swizzling=True),
]


# ---- Pytest tests ----

@pytest.mark.skipif(not HAS_GFX950, reason=f"GPU tests require gfx950, found {GFX_TARGET}")
class TestGraTileAssignmentFP8GPU:

    @pytest.fixture(params=TILE_CONFIGS_FP8, ids=lambda c: c.label)
    def gra_env(self, request, tmp_path):
        cfg = request.param
        gra_asm, writer, tileInfoA, tileInfoB, kernel = generate_gra_asm(cfg, geometry=AB_B8, inst_k=128, bpe=BPE)
        return SimpleNamespace(
            cfg=cfg, gra_asm=gra_asm, writer=writer,
            tileInfoA=tileInfoA, tileInfoB=tileInfoB,
            kernel=kernel, tmp_path=tmp_path,
        )

    def test_offset_a(self, gra_env):
        """Validate sharedVgprGROffset vgprs for matrix A across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoA.sharedVgprGROffset):
            results = export_register(
                gra_env.writer, gra_env.gra_asm, reg, False,
                cfg, gra_env.tmp_path, f"offsetA_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_offset_fp8(tid, cfg, gra_env.tileInfoA)
                assert results[tid] == expected[idx], (
                    f"[{cfg.label}] A offset[{idx}] v{reg} mismatch at tid={tid}: "
                    f"got {results[tid]}, expected {expected[idx]}"
                )

    def test_offset_b(self, gra_env):
        """Validate sharedVgprGROffset vgprs for matrix B across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoB.sharedVgprGROffset):
            results = export_register(
                gra_env.writer, gra_env.gra_asm, reg, False,
                cfg, gra_env.tmp_path, f"offsetB_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_offset_fp8(tid, cfg, gra_env.tileInfoB)
                assert results[tid] == expected[idx], (
                    f"[{cfg.label}] B offset[{idx}] v{reg} mismatch at tid={tid}: "
                    f"got {results[tid]}, expected {expected[idx]}"
                )

    def _test_subtile_registers(self, gra_env, tc):
        """Validate M-direction soffset registers (localSubtilesRegister) for tensor component tc ('A' or 'B').

        For this config (16x128 subtile covering full K), all subtile movement is in M.
        Each register holds the soffset for buffer_load to reach subtile row regId:
          soffset = regId * rowsPerSubtile * stride * BPE
        Row 0 has soffset=0 and holds an empty RegList (no register allocated).
        """
        cfg = gra_env.cfg
        tileInfo = gra_env.tileInfoA if tc == 'A' else gra_env.tileInfoB
        stride = cfg.stride_a if tc == 'A' else cfg.stride_b
        seen = set()
        for st in tileInfo.localSubtiles:
            regId = st.regListId
            if regId in seen:
                continue
            seen.add(regId)
            for reg in tileInfo.localSubtilesRegister[regId]:
                results = export_register(
                    gra_env.writer, gra_env.gra_asm, reg, st.useSgpr,
                    cfg, gra_env.tmp_path, f"subtile{tc}_s{reg}_{cfg.label}")
                expected = compute_expected_subtile(regId, stride, tileInfo, BPE)
                assert results[0] == expected, (
                    f"[{cfg.label}] {tc} subtile s{reg} (regId={regId}): "
                    f"got {results[0]}, expected {expected}"
                )

    def test_subtile_registers_a(self, gra_env):
        self._test_subtile_registers(gra_env, 'A')

    def test_subtile_registers_b(self, gra_env):
        self._test_subtile_registers(gra_env, 'B')


if __name__ == "__main__":
    """Run standalone without pytest."""
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for FP8 graTileAssignment")
    parser.add_argument("--grid",  action="store_true", help="Print offset grid")
    parser.add_argument("--debug", action="store_true", help="Print expected grid + diffs")
    args = parser.parse_args()
    if args.debug:
        args.grid = True

    for cfg in TILE_CONFIGS_FP8:
        print(f"\n{'='*60}")
        print(f"  FP8 Config: {cfg.label}")
        print(f"{'='*60}")
        gra_asm, writer, tileInfoA, tileInfoB, kernel = generate_gra_asm(cfg, geometry=AB_B8, inst_k=128, bpe=BPE)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            print("\n--- graTileAssignment assembly ---")
            in_gra = False
            for line in gra_asm.split('\n'):
                if 'GR Offset' in line or in_gra:
                    in_gra = True
                    print(line)
            print("--- end ---\n")

            if HAS_GFX950:
                for tc, tileInfo, stride in [("A", tileInfoA, cfg.stride_a),
                                              ("B", tileInfoB, cfg.stride_b)]:
                    for idx, reg in enumerate(tileInfo.sharedVgprGROffset):
                        results = export_register(writer, gra_asm, reg, False, cfg,
                                                      tmp_path, f"offset{tc}_v{reg}_{cfg.label}")
                        if args.grid:
                            print_offset_grid(
                                f"Matrix {tc} GPU offset[{idx}] v{reg} ({cfg.label})",
                                results, WAVESIZE, NUM_WAVES)
                        if args.debug:
                            expected = [compute_expected_offset_fp8(tid, cfg, tileInfo)[idx]
                                        for tid in range(NUM_THREADS)]
                            print_offset_grid(
                                f"Matrix {tc} EXPECTED offset[{idx}] ({cfg.label})",
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
                                     if results[tid] !=
                                     compute_expected_offset_fp8(tid, cfg, tileInfo)[idx])
                        print(f"  {tc} offset[{idx}] v{reg}: {errors} errors / {NUM_THREADS}")

                    # Subtile soffset registers
                    seen = set()
                    for st in tileInfo.localSubtiles:
                        regId = st.regListId
                        if regId in seen:
                            continue
                        seen.add(regId)
                        for reg in tileInfo.localSubtilesRegister[regId]:
                            results = export_register(
                                writer, gra_asm, reg, st.useSgpr,
                                cfg, tmp_path, f"subtile{tc}_s{reg}_{cfg.label}")
                            expected = compute_expected_subtile(regId, stride, tileInfo, BPE)
                            status = "OK" if results[0] == expected else "FAIL"
                            print(f"  {tc} subtile s{reg} (regId={regId}): "
                                  f"{results[0]} (expected {expected}) {status}")
            else:
                print(f"GPU tests require gfx950, found {GFX_TARGET} — assembly only")
