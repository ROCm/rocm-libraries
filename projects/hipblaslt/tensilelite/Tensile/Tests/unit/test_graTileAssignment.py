#!/usr/bin/env python3
################################################################################
# GPU functional test for graTileAssignment with parameterized tile configs
#
# Usage:
#   pytest test_graTileAssignment.py -v -s
################################################################################

import math
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
from Tensile.Components.SubtileBasedKernel import graTileAssignment


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


def generate_gra_asm(cfg):
    """Run graTileAssignment and return (gra_asm, writer, tileInfoA, tileInfoB, kernel)."""
    writer, kernel, tileInfoA, tileInfoB = create_writer(cfg)
    init_rocisa()

    # Reserve s0-s11 for hardware regs + kernarg loads
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    prologue = generate_load_params(EXPORT_LOAD_PARAMS)
    module = graTileAssignment(writer, kernel, useSwizzling=cfg.use_swizzling)
    gra_asm = f"{prologue}\n{module}"
    return gra_asm, writer, tileInfoA, tileInfoB, kernel


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

def compute_expected_offset(thread_id, cfg, tileInfo):
    """Python reference implementation matching graTileAssignment / _grComputeOffset.

    Mirrors the kernel's exact logic:
      1. colId from Serial, swizzled via DPP quad_perm + rotation (always on)
      2. _grComputeRowOffset for wave partitioning
      3. _grComputeOffset: (rowId + rowOffset) * stride * bpe + colId_bytes
    """
    stride = cfg.stride_a if tileInfo.tc == 'A' else cfg.stride_b
    bpe = BPE
    depthUBytes = cfg.depth_u * bpe
    blockSize = depthUBytes // LOAD_WIDTH
    numRowsPerLDSBanks = (64 * 4) // depthUBytes  # ldsRowBankSize / depthUBytes

    waveId = thread_id // WAVESIZE
    laneId = thread_id % WAVESIZE

    # --- colId computation (common for A and B) ---
    colId = thread_id & (blockSize - 1)

    # Swizzling is always on in the kernel (hardcodes True)
    # Step 1: DPP quad_perm[1,0,3,2] applied only when ldsRowId is even
    # Kernel uses laneId (not Serial) for ldsRowId computation
    rowInWave = laneId >> (blockSize.bit_length() - 1)
    ldsRowId = rowInWave >> (numRowsPerLDSBanks.bit_length() - 1)
    if ldsRowId % 2 == 0:
        # quad_perm[1,0,3,2] swaps pairs within each quad
        if colId % 2 == 0:
            colId = colId + 1
        else:
            colId = colId - 1

    # Step 2: Rotation with wave-specific component
    rotation = (ldsRowId // 2) * 2
    rotationOffset = blockSize - rotation

    if tileInfo.loadRatioGR != 0.5:
        # Wave-specific rotation: subtract (waveId & 1) << log2(2*numRowsPerLDSBanks)
        waveRotation = (waveId & 1) << ((2 * numRowsPerLDSBanks).bit_length() - 1)
        colId = (colId + rotationOffset - waveRotation) & (blockSize - 1)
    else:
        colId = (colId + rotationOffset) & (blockSize - 1)

    # Scale colId by loadWidth
    colId_bytes = colId * LOAD_WIDTH

    # --- _grComputeRowOffset ---
    numRowsPerWave = WAVESIZE // blockSize
    partitionOffset = tileInfo.mmaTileShape[0] * tileInfo.localSubtileGrid[0]

    if tileInfo.loadRatioGR == 1.0:
        localRow = waveId & 1
        partitionRow = waveId >> 1
    elif tileInfo.loadRatioGR == 0.5:
        localRow = 0
        partitionRow = waveId
    elif tileInfo.loadRatioGR == 2.0:
        localRow = waveId
        partitionRow = 0
    else:
        raise NotImplementedError(f"Unsupported loadRatioGR: {tileInfo.loadRatioGR}")

    localRow = localRow << (numRowsPerWave.bit_length() - 1)
    partitionRow = partitionOffset * partitionRow
    waveRowOffset = localRow + partitionRow

    # --- _grComputeOffset ---
    rowId = laneId >> (blockSize.bit_length() - 1)
    totalRow = rowId + waveRowOffset

    base = totalRow * stride * bpe + colId_bytes

    # Second GR offset if numGRPerSubtile > 1
    # Kernel advances row by ceil(subtileSize * loadRatioGR) and rotates colId by +4
    if tileInfo.numGRPerSubtile == 1:
        return [base]
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    rowAdvance = math.ceil(subtileSize * tileInfo.loadRatioGR)
    totalRow2 = totalRow + rowAdvance
    colId2 = (colId + 4) & (blockSize - 1)
    colId2_bytes = colId2 * LOAD_WIDTH
    offset2 = totalRow2 * stride * bpe + colId2_bytes
    return [base, offset2]

def compute_expected_subtile(regId, stride, tileInfo):
    """Compute expected subtile register value: rowOffset * bpe * regId * stride.

    The kernel uses regId (index into localSubtilesRegister) and rowOffset
    which is 2*subtileSize when loadRatioGR==2.0, otherwise subtileSize.
    """
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    rowOffset = 2 * subtileSize if tileInfo.loadRatioGR == 2.0 else subtileSize
    return rowOffset * BPE * regId * stride


# Tile configs to test
TILE_CONFIGS = [
    # 2x2 configs
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=512, stride_b=512, use_swizzling=False),
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=4096, stride_b=1024, use_swizzling=True),
    TileConfig(mt_a=96, mt_b=256, depth_u=64, stride_a=1024, stride_b=256, use_swizzling=True),
    # # 1x4 configs
    TileConfig(mt_a=80, mt_b=64, depth_u=64, stride_a=64, stride_b=256, use_swizzling=True),
    TileConfig(mt_a=80, mt_b=64, depth_u=64, stride_a=64, stride_b=64, use_swizzling=True),
    # # 4x1 configs
    TileConfig(mt_a=64, mt_b=48, depth_u=64, stride_a=64, stride_b=256, use_swizzling=True),
    # # mt0<32 (read size)
    TileConfig(mt_a=16, mt_b=64, depth_u=64, stride_a=64, stride_b=64, use_swizzling=True),
]


# ---- Pytest tests ----

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraTileAssignmentGPU:

    @pytest.fixture(params=TILE_CONFIGS, ids=lambda c: c.label)
    def gra_env(self, request, tmp_path):
        """Generate graTileAssignment asm once per tile config."""
        cfg = request.param
        gra_asm, writer, tileInfoA, tileInfoB, kernel = generate_gra_asm(cfg)
        return SimpleNamespace(
            cfg=cfg,
            gra_asm=gra_asm,
            writer=writer,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            kernel=kernel,
            tmp_path=tmp_path,
        )

    def test_offset_a(self, gra_env):
        """Validate all sharedVgprGROffset vgprs for matrix A across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoA.sharedVgprGROffset):
            results = export_register(gra_env.writer, gra_env.gra_asm, reg, False,
                                      cfg, gra_env.tmp_path, f"offsetA_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_offset(tid, cfg, gra_env.tileInfoA)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] A offset[{idx}] v{reg} mismatch at tid={tid}: got {results[tid]}, expected {expected[idx]}"

    def test_offset_b(self, gra_env):
        """Validate all sharedVgprGROffset vgprs for matrix B across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoB.sharedVgprGROffset):
            results = export_register(gra_env.writer, gra_env.gra_asm, reg, False,
                                      cfg, gra_env.tmp_path, f"offsetB_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_offset(tid, cfg, gra_env.tileInfoB)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] B offset[{idx}] v{reg} mismatch at tid={tid}: got {results[tid]}, expected {expected[idx]}"

    def _test_subtile_registers(self, gra_env, tc):
        """Validate localSubtilesRegister values for matrix tc."""
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
                results = export_register(gra_env.writer, gra_env.gra_asm, reg, st.useSgpr,
                                          cfg, gra_env.tmp_path, f"subtile{tc}_s{reg}_{cfg.label}")
                expected = compute_expected_subtile(regId, stride, tileInfo)
                actual = results[0]
                assert actual == expected, \
                    f"[{cfg.label}] {tc} subtile s{reg} (regId={regId}): " \
                    f"got {actual}, expected {expected}"

    def test_subtile_registers_a(self, gra_env):
        """Validate localSubtilesRegister values for matrix A."""
        self._test_subtile_registers(gra_env, 'A')

    def test_subtile_registers_b(self, gra_env):
        """Validate localSubtilesRegister values for matrix B."""
        self._test_subtile_registers(gra_env, 'B')


if __name__ == "__main__":
    """Run standalone without pytest."""
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for graTileAssignment")
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

        gra_asm, writer, tileInfoA, tileInfoB, kernel = generate_gra_asm(cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            # Print the generated assembly for inspection
            print("\n--- Generated Assembly (graTileAssignment section) ---")
            in_gra = False
            for line in gra_asm.split('\n'):
                if 'GR Offset' in line or in_gra:
                    in_gra = True
                    print(line)
            print("--- End ---\n")

            if HAS_HIP:
                # Test all sharedVgprGROffset vgprs for both matrices
                for tc, tileInfo, stride, mt in [("A", tileInfoA, cfg.stride_a, cfg.mt_a),
                                                  ("B", tileInfoB, cfg.stride_b, cfg.mt_b)]:
                    for idx, reg in enumerate(tileInfo.sharedVgprGROffset):
                        results = export_register(writer, gra_asm, reg, False, cfg, tmp_path,
                                                  f"offset{tc}_v{reg}_{cfg.label}")

                        if args.grid:
                            print_offset_grid(f"Matrix {tc} GPU offset[{idx}] v{reg} ({cfg.label})",
                                              results, WAVESIZE, NUM_WAVES)

                            if args.debug:
                                expected = [compute_expected_offset(tid, cfg,
                                                                     tileInfo)[idx]
                                            for tid in range(NUM_THREADS)]
                                print_offset_grid(f"Matrix {tc} EXPECTED offset[{idx}] ({cfg.label})",
                                                  expected, WAVESIZE, NUM_WAVES)

                                mismatches = sum(1 for t in range(NUM_THREADS) if results[t] != expected[t])
                                if mismatches:
                                    print(f"\n--- Matrix {tc} offset[{idx}] DIFF ({mismatches} mismatches) ---")
                                    for w in range(NUM_WAVES):
                                        print(f"  w{w}: ", end="")
                                        for lane in range(WAVESIZE):
                                            tid = w * WAVESIZE + lane
                                            if results[tid] != expected[tid]:
                                                print(f" t{tid}:{results[tid]}!={expected[tid]}", end="")
                                        print()
                                else:
                                    print(f"\n  Matrix {tc} offset[{idx}]: all match.")

                        errors = 0
                        for tid in range(NUM_THREADS):
                            exp = compute_expected_offset(tid, cfg, tileInfo)[idx]
                            if results[tid] != exp:
                                errors += 1
                                if not args.grid:
                                    print(f"  FAIL {tc} offset[{idx}] v{reg} tid={tid}: got {results[tid]}, expected {exp}")
                            elif not args.grid and tid < 64:
                                print(f"  OK   {tc} offset[{idx}] v{reg} tid={tid}: {results[tid]}")

                        print(f"  Matrix {tc} offset[{idx}] v{reg}: {NUM_THREADS} threads, {errors} errors")

                # Subtile registers
                for tc, tileInfo, stride in [("A", tileInfoA, cfg.stride_a),
                                              ("B", tileInfoB, cfg.stride_b)]:
                    seen = set()
                    for st in tileInfo.localSubtiles:
                        regId = st.regListId
                        if regId in seen:
                            continue
                        seen.add(regId)
                        for reg in tileInfo.localSubtilesRegister[regId]:
                            print("Regl",reg)
                            results = export_register(writer, gra_asm, reg, st.useSgpr, cfg,
                                                      tmp_path, f"subtile{tc}_s{reg}_{cfg.label}")
                            expected = compute_expected_subtile(regId, stride, tileInfo)
                            actual = results[0]
                            status = "OK" if actual == expected else "FAIL"
                            print(f"  Subtile {tc} s{reg} (regId={regId}): {actual} (expected {expected}) {status}")
            else:
                print("HIP not available - assembly generated but not executed")