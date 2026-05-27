#!/usr/bin/env python3
################################################################################
# End-to-end GPU roundtrip test: Uses production globalReadDoSubtile /
# localReadDoSubtile code paths.
#
#
# Usage:
#   pytest test_gr_lr_roundtrip.py -v -s
#   python test_gr_lr_roundtrip.py --debug --wave all
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
    BPE, WAVESIZE, NUM_THREADS,
    assemble_and_run,
    generate_kernel_asm,
    setup_roundtrip_writer,
    build_roundtrip_inner_asm,
    alloc_export_vgprs,
)


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    # 2x2 configs (both mt_a//16 and mt_b//16 even)
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=512,  stride_b=64),
    TileConfig(mt_a=96,  mt_b=128, depth_u=64, stride_a=64,  stride_b=1024),
    # 1x4 config (mt_a//16 odd, mt_b//16 div by 4)
    TileConfig(mt_a=48,  mt_b=64, depth_u=64, stride_a=512,  stride_b=64),
    TileConfig(mt_a=48,  mt_b=128, depth_u=64, stride_a=64,  stride_b=1024),
    # 4x1 config (mt_a//16 div by 4, mt_b//16 odd)
    TileConfig(mt_a=128, mt_b=48,  depth_u=64, stride_a=512,  stride_b=64),
    TileConfig(mt_a=64,  mt_b=48, depth_u=64, stride_a=64,  stride_b=1024),
    # Stride > depthU variants
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=128, stride_b=128),
    TileConfig(mt_a=96,  mt_b=128, depth_u=64, stride_a=128, stride_b=128),
    # Larger MT
    TileConfig(mt_a=256, mt_b=240,  depth_u=64, stride_a=64,  stride_b=64),
    TileConfig(mt_a=240, mt_b=256,  depth_u=64, stride_a=64,  stride_b=64),

]


# ---------------------------------------------------------------------------
# Assembly generation using production code paths
# ---------------------------------------------------------------------------


def generate_export_asm(wave_id, tileInfoA, tileInfoB):
    """Generate assembly to export vgprTile registers from a selected wave.

    Each vgprTile is 4 consecutive vgprs (one dwordx4 = 16 bytes per lane).
    Output layout: [A tiles][B tiles], each tile = WAVESIZE * 16 bytes.
    """
    lines = []
    lines.append(f"  // ---- Wave-gated export (wave {wave_id}) ----")

    tmp, addr_lo, addr_hi, next_v = alloc_export_vgprs(tileInfoA, tileInfoB)

    # Wave masking
    lines.append(f"  v_lshrrev_b32 v{tmp}, 6, v0                // waveId")
    lines.append(f"  v_cmp_eq_u32 vcc, {wave_id}, v{tmp}")
    lines.append(f"  s_and_saveexec_b64 s[2:3], vcc              // gate to wave {wave_id}")

    # Compute laneId for offset
    lines.append(f"  v_and_b32 v{tmp}, 0x3F, v0                  // laneId")

    tile_index = 0
    all_tiles = list(tileInfoA.vgprTiles) + list(tileInfoB.vgprTiles)

    for tile in all_tiles:
        vgpr_start = tile.regList.indices[0]
        num_regs = len(tile.regList.indices)
        assert num_regs == 4, f"Expected 4 regs per tile, got {num_regs}"

        # Output offset = tile_index * WAVESIZE * 16 + laneId * 16
        base_offset = tile_index * WAVESIZE * 16
        lines.append(f"  // Export tile {tile_index}: v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 4, v{tmp}       // laneId * 16")
        if base_offset > 0:
            lines.append(f"  v_add_u32 v{addr_lo}, {base_offset}, v{addr_lo}  // + tile base")
        lines.append(f"  v_mov_b32 v{addr_hi}, s9                   // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  s_waitcnt vmcnt(0)")
        tile_index += 1

    lines.append(f"  s_or_b64 exec, exec, s[2:3]                // restore exec")
    return "\n".join(lines), next_v


def generate_roundtrip_kernel(cfg, wave_id=0):
    """Generate a complete kernel using production GR/LR code paths."""
    writer, kernel, tileInfoA, tileInfoB, lds_size = setup_roundtrip_writer(cfg)
    print(tileInfoA)

    export_asm, _next_v = generate_export_asm(wave_id, tileInfoA, tileInfoB)
    inner_asm = build_roundtrip_inner_asm(writer, kernel, export_asm)
    print(inner_asm)

    args = (
        ("input_A_ptr", 8, "global_buffer", "f16"),  # s[4:5]
        ("input_B_ptr", 8, "global_buffer", "f16"),  # s[6:7]
        ("output_ptr",  8, "global_buffer", "u32"),  # s[8:9]
        ("strideA",     4, "by_value",      "u32"),  # s10
        ("strideB",     4, "by_value",      "u32"),  # s11
    )
    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_size)

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b
    output_size = total_tiles * WAVESIZE * 16  # 16 bytes (4 vgprs x 4B) per lane per tile

    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size


# ---------------------------------------------------------------------------
# Host-side expected value computation
# ---------------------------------------------------------------------------

def compute_expected_output(cfg, tileInfoA, tileInfoB, kernel, input_A, input_B, wave_id):
    """Compute expected vgprTile contents for a given wave.

    After the GR->LDS->LR roundtrip (with swizzle/de-swizzle cancelling out),
    each vgprTile holds data in MFMA register format:
      - lane16 (lane % 16) maps to row lane16 within the MMA tile
      - lane16Group (lane // 16) maps to column group (8 fp16 per group)

    Returns a list of numpy arrays, one per vgprTile (A tiles first, then B tiles).
    Each array is WAVESIZE * 8 fp16 values (4 dwords = 8 fp16 per lane).
    """
    mi_wave_group = kernel["MIWaveGroup"]
    results = []

    for tc, tileInfo, input_data in [('A', tileInfoA, input_A), ('B', tileInfoB, input_B)]:
        stride = cfg.stride_a if tc == 'A' else cfg.stride_b
        input_matrix = input_data.reshape(-1, stride)

        # Wave offset in the tile dimension
        if tc == 'A':
            wave_offset_factor = wave_id % mi_wave_group[0]
        else:
            wave_offset_factor = wave_id // mi_wave_group[0]

        # Non-interleaved layout: each wave partition covers a contiguous
        # block of localMMATileGrid[0] MMA tiles (each 16 rows).
        wave_row_offset = wave_offset_factor * tileInfo.localMMATileGrid[0] * 16

        # Build reverse map: vgprTile index -> (mmaId0, mmaId1)
        # Non-interleaved layout: interleave_factor = 1 for all configs.
        tile_to_mma = _build_tile_to_mma(tileInfo)

        for tileIdx in range(len(tileInfo.vgprTiles)):
            if tileIdx not in tile_to_mma:
                results.append(np.zeros(WAVESIZE * 8, dtype=np.float16))
                continue

            mmaId0, mmaId1 = tile_to_mma[tileIdx]
            tile_data = np.zeros(WAVESIZE * 8, dtype=np.float16)

            for lane in range(WAVESIZE):
                lane16 = lane % 16
                lane16Group = lane // 16

                row_in_input = wave_row_offset + mmaId0 * 16 + lane16
                col_in_input = mmaId1 * 32 + lane16Group * 8

                if row_in_input < input_matrix.shape[0] and col_in_input + 8 <= input_matrix.shape[1]:
                    tile_data[lane * 8 : lane * 8 + 8] = input_matrix[row_in_input, col_in_input : col_in_input + 8]

            results.append(tile_data)

    return results


def compare_tiles(actual_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id, debug=False):
    """Compare GPU output against expected tile data. Returns number of errors."""
    errors = 0
    tile_size = WAVESIZE * 16  # bytes per tile

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b

    for tile_idx in range(total_tiles):
        tc = 'A' if tile_idx < num_tiles_a else 'B'
        local_idx = tile_idx if tile_idx < num_tiles_a else tile_idx - num_tiles_a

        offset = tile_idx * tile_size
        actual = np.frombuffer(actual_bytes[offset:offset + tile_size], dtype=np.float16)
        expected = expected_tiles[tile_idx]

        if not np.array_equal(actual, expected):
            errors += 1
            if errors <= 8 or debug:
                # Find first mismatching lane
                for lane in range(WAVESIZE):
                    a_slice = actual[lane * 8 : lane * 8 + 8]
                    e_slice = expected[lane * 8 : lane * 8 + 8]
                    if not np.array_equal(a_slice, e_slice):
                        print(f"  MISMATCH wave {wave_id} {tc} tile {local_idx} lane {lane}:")
                        print(f"    expected: {e_slice}")
                        print(f"    actual:   {a_slice}")
                        if not debug:
                            break

    return errors


def _build_tile_to_mma(tileInfo):
    """Build map from vgprTile index to (mmaId0, mmaId1).

    Uses the same indexing formula as emitMfmaInstructions (Kernel.py):
      tileId = (sId1 * lrLocalGrid0 + sId0) * numMmaTilePerSubtile + _mmak
    """
    lrSubtileShape = tileInfo.lr.subtileShape
    lrLocalGrid0 = tileInfo.localMMATileGrid[0] // lrSubtileShape[0]
    numMmaTilePerSubtile = lrSubtileShape[0] * lrSubtileShape[1]

    tile_to_mma = {}
    for mmak in range(tileInfo.localMMATileGrid[1]):
        for mma0 in range(tileInfo.localMMATileGrid[0]):
            sId0 = mma0 // lrSubtileShape[0]
            sId1 = mmak // lrSubtileShape[1]
            _mmak = mmak % lrSubtileShape[1]
            tileId = (sId1 * lrLocalGrid0 + sId0) * numMmaTilePerSubtile + _mmak
            tile_to_mma[tileId] = (mma0, mmak)
    return tile_to_mma


def reconstruct_matrix(tiles, tileInfo, wave_id, kernel, mt, depth_u):
    """Reconstruct 2D matrix (mt x depth_u) from MFMA register tile data."""
    mi = kernel["MIWaveGroup"]
    if tileInfo.tc == 'A':
        wf = wave_id % mi[0]
    else:
        wf = wave_id // mi[0]
    wave_row_off = wf * tileInfo.localMMATileGrid[0] * 16
    tile_to_mma = _build_tile_to_mma(tileInfo)
    mat = np.full((mt, depth_u), np.nan, dtype=np.float64)

    for idx, tile in enumerate(tiles):
        if idx not in tile_to_mma:
            continue
        m0, m1 = tile_to_mma[idx]
        for lane in range(WAVESIZE):
            r = wave_row_off + m0 * 16 + (lane % 16)
            c = m1 * 32 + (lane // 16) * 8
            if r < mt and c + 8 <= depth_u:
                mat[r, c:c+8] = tile[lane*8:lane*8+8].astype(np.float64)
    return mat


def print_matrix_grid(label, mat):
    """Print matrix grid showing first value of each 8-col group."""
    rows, cols = mat.shape
    ng = cols // 8
    print(f"\n  --- {label} ({rows}r x {cols}c, val[0] per 8-col group) ---")
    print(f"  {'':>4}", end="")
    for g in range(ng):
        print(f" c{g*8:<5}", end="")
    print()
    for r in range(rows):
        if all(np.isnan(mat[r, g*8]) for g in range(ng)):
            continue
        print(f"  {r:>4}", end="")
        for g in range(ng):
            v = mat[r, g*8]
            print(f" {'---':>6}" if np.isnan(v) else f" {int(v):>6}", end="")
        print()


def print_grid_diff(label, actual, expected):
    """Print only mismatch rows: actual!=expected per 8-col group."""
    rows, cols = actual.shape
    ng = cols // 8
    n_mis = 0
    print(f"\n  --- {label} DIFF (*=mismatch, actual!=expected) ---")
    print(f"  {'':>4}", end="")
    for g in range(ng):
        print(f"   c{g*8:<10}", end="")
    print()
    for r in range(rows):
        diffs = []
        for g in range(ng):
            a, e = actual[r, g*8], expected[r, g*8]
            if (np.isnan(a) != np.isnan(e)) or (not np.isnan(a) and a != e):
                diffs.append(g)
        if not diffs:
            continue
        print(f"  {r:>4}", end="")
        for g in range(ng):
            a, e = actual[r, g*8], expected[r, g*8]
            if g in diffs:
                n_mis += 1
                av = "nan" if np.isnan(a) else str(int(a))
                ev = "nan" if np.isnan(e) else str(int(e))
                print(f" *{av:>5}!={ev:<5}", end="")
            elif np.isnan(a):
                print(f"      ---      ", end="")
            else:
                print(f"  {int(a):>5}       ", end="")
        print()
    print(f"  {'All match.' if n_mis == 0 else f'{n_mis} group mismatches.'}")

# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GFX950, reason=f"GPU tests require gfx950, found {GFX_TARGET}")
class TestGrLrRoundtrip:

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2, 3], ids=lambda w: f"wave{w}")
    def wave_id(self, request):
        return request.param

    def test_gr_lr_roundtrip(self, cfg, wave_id, tmp_path):
        """Verify GR -> LDS -> LR roundtrip using production code paths."""
        sys.stdout.flush()

        kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size = \
            generate_roundtrip_kernel(cfg, wave_id=wave_id)

        # Create input data
        input_A = np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.float16)
        input_B = -np.arange(1, cfg.mt_b * cfg.stride_b + 1, dtype=np.float16)

        label = f"roundtrip_{cfg.label}_wave{wave_id}"
        output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                        inputs=(input_A, input_B),
                                        scalars=(cfg.stride_a, cfg.stride_b),
                                        lds_size=lds_size)

        expected_tiles = compute_expected_output(cfg, tileInfoA, tileInfoB, kernel,
                                                 input_A, input_B, wave_id)
        errors = compare_tiles(output_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id)

        assert errors == 0, f"Wave {wave_id}, config {cfg.label}: {errors} tile mismatches"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GR/LR roundtrip GPU test")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed output and asm")
    parser.add_argument("--grid", action="store_true",
                        help="Display actual/expected as 2D matrix grids")
    parser.add_argument("--wave", default="all",
                        help="Which wave to test: 0-3 or 'all' (default: all)")
    parser.add_argument("--config", type=int, default=None,
                        help="Config index to test (default: all)")
    args = parser.parse_args()

    wave_list = [0, 1, 2, 3] if args.wave == "all" else [int(args.wave)]
    config_list = CONFIGS if args.config is None else [CONFIGS[args.config]]

    total_errors = 0
    total_tests = 0

    for cfg_idx, cfg in enumerate(config_list):
        print(f"\n{'='*60}")
        print(f"Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")
        print(f"  stride_a={cfg.stride_a}, stride_b={cfg.stride_b}")

        for wave_id in wave_list:
            total_tests += 1
            print(f"\n  --- Wave {wave_id} ---")

            kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size = \
                generate_roundtrip_kernel(cfg, wave_id=wave_id)

            num_tiles_a = len(tileInfoA.vgprTiles)
            num_tiles_b = len(tileInfoB.vgprTiles)
            print(f"  A tiles: {num_tiles_a}, B tiles: {num_tiles_b}, output: {output_size} bytes")
            print(f"  MIWaveGroup: {kernel['MIWaveGroup']}")

            if args.debug:
                print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

                input_A = np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.float16)
                input_B = -np.arange(1, cfg.mt_b * cfg.stride_b + 1, dtype=np.float16)

                label = f"roundtrip_{cfg.label}_wave{wave_id}"
                output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                                inputs=(input_A, input_B),
                                                scalars=(cfg.stride_a, cfg.stride_b),
                                                lds_size=lds_size)

                expected_tiles = compute_expected_output(cfg, tileInfoA, tileInfoB, kernel,
                                                        input_A, input_B, wave_id)
                errors = compare_tiles(output_bytes, expected_tiles, tileInfoA, tileInfoB,
                                       wave_id, debug=args.debug)

                if args.grid:
                    tile_size = WAVESIZE * 16
                    actual_a = [np.frombuffer(output_bytes[i*tile_size:(i+1)*tile_size],
                                dtype=np.float16).copy() for i in range(num_tiles_a)]
                    actual_b = [np.frombuffer(output_bytes[(num_tiles_a+i)*tile_size:(num_tiles_a+i+1)*tile_size],
                                dtype=np.float16).copy() for i in range(num_tiles_b)]
                    expected_a = expected_tiles[:num_tiles_a]
                    expected_b = expected_tiles[num_tiles_a:]

                    for tc, t_act, t_exp, ti, mt in [
                        ('A', actual_a, expected_a, tileInfoA, cfg.mt_a),
                        ('B', actual_b, expected_b, tileInfoB, cfg.mt_b)]:
                        mat_exp = reconstruct_matrix(t_exp, ti, wave_id, kernel, mt, cfg.depth_u)
                        mat_act = reconstruct_matrix(t_act, ti, wave_id, kernel, mt, cfg.depth_u)
                        print_matrix_grid(f"Matrix {tc} EXPECTED (wave {wave_id})", mat_exp)
                        print_matrix_grid(f"Matrix {tc} ACTUAL (wave {wave_id})", mat_act)
                        print_grid_diff(f"Matrix {tc} (wave {wave_id})", mat_act, mat_exp)

                if errors == 0:
                    print(f"  PASS")
                else:
                    print(f"  FAIL: {errors} tile mismatches")
                    total_errors += errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    if total_errors > 0:
        print("FAILED")
        sys.exit(1)
    else:
        print("PASSED")