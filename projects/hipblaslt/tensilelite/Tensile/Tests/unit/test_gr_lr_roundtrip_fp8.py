#!/usr/bin/env python3
################################################################################
# End-to-end GPU roundtrip test for FP8: Uses production globalReadDoSubtile /
# localReadDoSubtile code paths.
#
# Config: MT=256x256, DU=128, 2x2 wave group
#   bpe=1, instK=128, vgprsPerTile=8, numLRPerSubtile=2
#
# Expected MFMA FP8 register layout after GR->LDS->LR roundtrip
# (GR block-swap + LR block-swap cancel):
#   lane16 = laneId % 16  (M-row within MFMA tile)
#   lane16Group = laneId // 16  (K-group, 0-3)
#   vgprs[0:3] = input[M_row, K_group*16       : K_group*16+16]  (K_lo, 16 bytes)
#   vgprs[4:7] = input[M_row, K_group*16 + 64  : K_group*16+80]  (K_hi, 16 bytes)
#
# Usage:
#   pytest test_gr_lr_roundtrip_fp8.py -v -s
#   python test_gr_lr_roundtrip_fp8.py --debug --wave all
################################################################################

import os
import sys
import tempfile

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_GFX950,
    GFX_TARGET,
    TileConfig,
    WAVESIZE, NUM_THREADS, NUM_WAVES,
    AB_B8,
    assemble_and_run,
    generate_kernel_asm,
    setup_roundtrip_writer,
    build_roundtrip_inner_asm,
    alloc_export_vgprs,
)

BPE = 1  # fp8: 1 byte per element


assert NUM_THREADS == 256, f"Roundtrip test requires exactly 256 threads (4 waves × 64), got {NUM_THREADS}"
assert NUM_WAVES == 4,    f"Roundtrip test requires exactly 4 waves, got {NUM_WAVES}"

VGPRS_PER_TILE = 8  # MFMA_16x16_1B_4K_8V: 8 vgprs per lane per tile


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
TILE_CONFIGS_FP8 = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=512,  stride_b=512),
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=4096, stride_b=1024),
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=256,  stride_b=4096),
]


# ---------------------------------------------------------------------------
# Export ASM (8 vgprs per tile → 2 × flat_store_dwordx4)
# ---------------------------------------------------------------------------

def generate_export_asm_fp8(wave_id, tileInfoA, tileInfoB):
    """Generate assembly to export vgprTile registers from a selected wave.

    Each FP8 vgprTile is 8 vgprs (32 bytes per lane):
      - vgprs[0:3]: Load 1 data (K_lo block, 16 bytes)
      - vgprs[4:7]: Load 2 data (K_hi block, 16 bytes)
    Output layout: [A tiles][B tiles], each tile = WAVESIZE * 32 bytes.
    """
    lines = []
    lines.append(f"  // ---- Wave-gated export (wave {wave_id}, FP8 8-vgpr tiles) ----")

    tmp, addr_lo, addr_hi, next_v = alloc_export_vgprs(tileInfoA, tileInfoB)

    # Wave masking: only the selected wave writes output
    lines.append(f"  v_lshrrev_b32 v{tmp}, 6, v0                // waveId = tid >> 6")
    lines.append(f"  v_cmp_eq_u32 vcc, {wave_id}, v{tmp}")
    lines.append(f"  s_and_saveexec_b64 s[2:3], vcc              // gate to wave {wave_id}")
    lines.append(f"  v_and_b32 v{tmp}, 0x3F, v0                  // laneId = tid & 63")

    tile_index = 0
    all_tiles = list(tileInfoA.vgprTiles) + list(tileInfoB.vgprTiles)

    for tile in all_tiles:
        regs = tile.regList.indices
        num_regs = len(regs)
        assert num_regs == VGPRS_PER_TILE, \
            f"Expected {VGPRS_PER_TILE} regs per FP8 tile, got {num_regs}"
        vstart = regs[0]

        # Each tile: 32 bytes per lane (lane offset = laneId * 32)
        base_offset = tile_index * WAVESIZE * 32
        lines.append(f"  // Tile {tile_index}: v[{vstart}:{vstart+7}]")

        # Export first half (vgprs[0:3], Load 1 = K_lo, bytes 0..15 of lane slot)
        lo_offset = base_offset
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 5, v{tmp}       // laneId * 32")
        if lo_offset > 0:
            lines.append(f"  v_add_u32 v{addr_lo}, {lo_offset}, v{addr_lo}  // + tile base")
        lines.append(f"  v_mov_b32 v{addr_hi}, s9                   // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{vstart}:{vstart+3}]")
        lines.append(f"  s_waitcnt vmcnt(0)")

        # Export second half (vgprs[4:7], Load 2 = K_hi, bytes 16..31 of lane slot)
        hi_offset = base_offset + 16
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 5, v{tmp}       // laneId * 32")
        lines.append(f"  v_add_u32 v{addr_lo}, {hi_offset}, v{addr_lo}  // + 16")
        lines.append(f"  v_mov_b32 v{addr_hi}, s9                   // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{vstart+4}:{vstart+7}]")
        lines.append(f"  s_waitcnt vmcnt(0)")

        tile_index += 1

    lines.append(f"  s_or_b64 exec, exec, s[2:3]                // restore exec")
    return "\n".join(lines), next_v


# ---------------------------------------------------------------------------
# Assembly generation using production code paths
# ---------------------------------------------------------------------------

def generate_roundtrip_kernel_fp8(cfg, wave_id=0):
    """Generate a complete FP8 kernel using production GR/LR code paths."""
    writer, kernel, tileInfoA, tileInfoB, lds_size = setup_roundtrip_writer(
        cfg, geometry=AB_B8, inst_k=128, bpe=BPE)

    export_asm, _next_v = generate_export_asm_fp8(wave_id, tileInfoA, tileInfoB)
    inner_asm = build_roundtrip_inner_asm(writer, kernel, export_asm)

    args = (
        ("input_A_ptr", 8, "global_buffer", "u8"),   # s[4:5]
        ("input_B_ptr", 8, "global_buffer", "u8"),   # s[6:7]
        ("output_ptr",  8, "global_buffer", "u32"),  # s[8:9]
        ("strideA",     4, "by_value",      "u32"),  # s10
        ("strideB",     4, "by_value",      "u32"),  # s11
    )
    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_size, num_threads=NUM_THREADS)

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b
    output_size = total_tiles * WAVESIZE * 32  # 32 bytes (8 vgprs × 4B) per lane per tile

    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size


# ---------------------------------------------------------------------------
# Host-side expected value computation
# ---------------------------------------------------------------------------

def _build_tile_to_mma(tileInfo):
    """Build map from vgprTile index to (mmaId0, mmaId1) using the LR subtile grid.

    Uses lrLocalSubtileGrid + waveMmaTilesForSubtile (avoids localSubtiles which is []).
    """
    tile_to_mma = {}
    for sId0 in range(tileInfo.lrLocalSubtileGrid[0]):
        for sId1 in range(tileInfo.lrLocalSubtileGrid[1]):
            mma_tiles = tileInfo.waveMmaTilesForSubtile(sId0, sId1)
            for mfmaId, (mmaId0, mmaId1) in enumerate(mma_tiles):
                tileIdx = tileInfo.lrTileIndexForSubtile(sId0, sId1, mfmaId)
                tile_to_mma[tileIdx] = (mmaId0, mmaId1)
    return tile_to_mma


def compute_expected_output_fp8(cfg, tileInfoA, tileInfoB, kernel, input_A, input_B, wave_id):
    """Compute expected vgprTile contents for a given wave after GR→LDS→LR roundtrip.

    After the GR block-swap write and LR block-swap read cancel each other,
    each lane holds 32 FP8 bytes split into two non-contiguous K blocks:
      lane16 = laneId % 16  (M-row within MMA tile)
      K_group = laneId // 16
      vgprs[0:3] (16 bytes) = input[M_row, K_group*16       : K_group*16 + 16]  (K_lo)
      vgprs[4:7] (16 bytes) = input[M_row, K_group*16 + 64  : K_group*16 + 80]  (K_hi)

    Returns list of uint8 numpy arrays, one per vgprTile (A tiles first, then B).
    Each array is WAVESIZE * 32 bytes (32 bytes per lane).
    """
    mi_wave_group = kernel["MIWaveGroup"]
    results = []

    for tc, tileInfo, input_data in [('A', tileInfoA, input_A), ('B', tileInfoB, input_B)]:
        stride = cfg.stride_a if tc == 'A' else cfg.stride_b
        input_matrix = input_data.reshape(-1, stride)

        # Wave position within the macro tile (M dimension)
        if tc == 'A':
            wave_offset_factor = wave_id % mi_wave_group[0]
        else:
            wave_offset_factor = wave_id // mi_wave_group[0]
        wave_row_offset = wave_offset_factor * tileInfo.localMMATileGrid[0] * 16

        tile_to_mma = _build_tile_to_mma(tileInfo)

        for tileIdx in range(len(tileInfo.vgprTiles)):
            if tileIdx not in tile_to_mma:
                results.append(np.zeros(WAVESIZE * 32, dtype=np.uint8))
                continue

            mmaId0, mmaId1 = tile_to_mma[tileIdx]
            tile_data = np.zeros(WAVESIZE * 32, dtype=np.uint8)

            for lane in range(WAVESIZE):
                lane16 = lane % 16        # M-row within MMA tile
                K_group = lane // 16      # K-group (0-3)

                actual_M_row = wave_row_offset + mmaId0 * 16 + lane16
                # K base for this MMA tile's K-column (mmaId1 * depth_u)
                k_base = mmaId1 * cfg.depth_u
                k_lo_start = k_base + K_group * 16        # K_lo: 16 bytes
                k_hi_start = k_base + K_group * 16 + 64   # K_hi: 16 bytes

                if (actual_M_row < input_matrix.shape[0] and
                        k_hi_start + 16 <= input_matrix.shape[1]):
                    # vgprs[0:3]: Load 1 (K_lo block)
                    tile_data[lane * 32 : lane * 32 + 16] = \
                        input_matrix[actual_M_row, k_lo_start : k_lo_start + 16]
                    # vgprs[4:7]: Load 2 (K_hi block)
                    tile_data[lane * 32 + 16 : lane * 32 + 32] = \
                        input_matrix[actual_M_row, k_hi_start : k_hi_start + 16]

            results.append(tile_data)

    return results


def compare_tiles_fp8(actual_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id, debug=False):
    """Compare GPU output against expected FP8 tile data. Returns number of tile mismatches."""
    tile_errors = 0
    tile_size = WAVESIZE * 32  # 32 bytes per lane per tile

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b

    for tile_idx in range(total_tiles):
        tc = 'A' if tile_idx < num_tiles_a else 'B'
        local_idx = tile_idx if tile_idx < num_tiles_a else tile_idx - num_tiles_a

        offset = tile_idx * tile_size
        actual = np.frombuffer(actual_bytes[offset : offset + tile_size], dtype=np.uint8).copy()
        expected = expected_tiles[tile_idx]

        if not np.array_equal(actual, expected):
            tile_errors += 1
            if tile_errors <= 8 or debug:
                for lane in range(WAVESIZE):
                    a_lo = actual[lane * 32 : lane * 32 + 16]
                    a_hi = actual[lane * 32 + 16 : lane * 32 + 32]
                    e_lo = expected[lane * 32 : lane * 32 + 16]
                    e_hi = expected[lane * 32 + 16 : lane * 32 + 32]
                    if not (np.array_equal(a_lo, e_lo) and np.array_equal(a_hi, e_hi)):
                        print(f"  MISMATCH wave {wave_id} {tc} tile {local_idx} lane {lane}:")
                        print(f"    expected lo: {list(e_lo)}")
                        print(f"    actual   lo: {list(a_lo)}")
                        print(f"    expected hi: {list(e_hi)}")
                        print(f"    actual   hi: {list(a_hi)}")
                        if not debug:
                            break

    return tile_errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GFX950, reason=f"GPU tests require gfx950, found {GFX_TARGET}")
class TestGrLrRoundtripFP8:

    @pytest.fixture(params=TILE_CONFIGS_FP8, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    @pytest.fixture(params=list(range(NUM_WAVES)), ids=lambda w: f"wave{w}")
    def wave_id(self, request):
        return request.param

    def test_gr_lr_roundtrip_fp8(self, cfg, wave_id, tmp_path):
        """Verify FP8 GR -> LDS -> LR roundtrip using production code paths."""
        sys.stdout.flush()

        kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size = \
            generate_roundtrip_kernel_fp8(cfg, wave_id=wave_id)

        # Unique uint8 input data (values 1..255 cycling, A positive, B inverted)
        input_A = (np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.uint64) % 251 + 1).astype(np.uint8)
        input_B = (200 - np.arange(cfg.mt_b * cfg.stride_b, dtype=np.uint64) % 199).astype(np.uint8)

        label = f"roundtrip_fp8_{cfg.label}_wave{wave_id}"
        output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                        inputs=(input_A, input_B),
                                        scalars=(cfg.stride_a, cfg.stride_b),
                                        lds_size=lds_size,
                                        num_threads=NUM_THREADS)

        expected_tiles = compute_expected_output_fp8(cfg, tileInfoA, tileInfoB, kernel,
                                                     input_A, input_B, wave_id)
        tile_errors = compare_tiles_fp8(output_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id)

        assert tile_errors == 0, \
            f"Wave {wave_id}, config {cfg.label}: {tile_errors} tile mismatches"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GR/LR FP8 roundtrip GPU test")
    parser.add_argument("--debug",  action="store_true", help="Print detailed mismatch info")
    parser.add_argument("--wave",   default="all",       help="Wave to test: 0-3 or 'all'")
    parser.add_argument("--config", type=int, default=None, help="Config index (default: all)")
    args = parser.parse_args()

    if not HAS_GFX950:
        print(f"GPU tests require gfx950, found {GFX_TARGET} — cannot run GPU test")
        sys.exit(1)

    wave_list   = list(range(NUM_WAVES)) if args.wave == "all" else [int(args.wave)]
    config_list = TILE_CONFIGS_FP8 if args.config is None else [TILE_CONFIGS_FP8[args.config]]

    total_errors = 0
    total_tests  = 0

    for cfg in config_list:
        print(f"\n{'='*60}")
        print(f"Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")

        for wave_id in wave_list:
            total_tests += 1
            print(f"\n  --- Wave {wave_id} ---")

            kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size = \
                generate_roundtrip_kernel_fp8(cfg, wave_id=wave_id)

            num_tiles_a = len(tileInfoA.vgprTiles)
            num_tiles_b = len(tileInfoB.vgprTiles)
            print(f"  A tiles: {num_tiles_a}, B tiles: {num_tiles_b}, "
                  f"output: {output_size} bytes, lds: {lds_size} bytes")
            print(f"  numLRPerSubtile A={tileInfoA.numLRPerSubtile}, "
                  f"B={tileInfoB.numLRPerSubtile}")

            if args.debug:
                print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = type('P', (), {
                    '__truediv__': lambda s, n: os.path.join(tmp_dir, n)
                })()

                input_A = (np.arange(1, cfg.mt_a * cfg.stride_a + 1,
                                     dtype=np.uint64) % 251 + 1).astype(np.uint8)
                input_B = (200 - np.arange(cfg.mt_b * cfg.stride_b,
                                           dtype=np.uint64) % 199).astype(np.uint8)

                label = f"roundtrip_fp8_{cfg.label}_wave{wave_id}"
                output_bytes = assemble_and_run(
                    kernel_asm, tmp_path, label, output_size,
                    inputs=(input_A, input_B),
                    scalars=(cfg.stride_a, cfg.stride_b),
                    lds_size=lds_size,
                    num_threads=NUM_THREADS)

                expected_tiles = compute_expected_output_fp8(
                    cfg, tileInfoA, tileInfoB, kernel,
                    input_A, input_B, wave_id)
                tile_errors = compare_tiles_fp8(
                    output_bytes, expected_tiles,
                    tileInfoA, tileInfoB, wave_id, debug=args.debug)

                if tile_errors == 0:
                    print(f"  PASS")
                else:
                    print(f"  FAIL: {tile_errors} tile mismatches")
                    total_errors += tile_errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    sys.exit(0 if total_errors == 0 else 1)
