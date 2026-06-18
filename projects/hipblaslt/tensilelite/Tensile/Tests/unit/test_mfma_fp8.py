#!/usr/bin/env python3
################################################################################
# End-to-end GPU MFMA test for FP8:
#   GR -> LDS -> LR -> v_mfma_f32_16x16x128_f8f6f4 -> export AGPRs -> verify
#
# Uses OCP FP8 E4M3FN format (cbsz:0 blgp:0) for both A and B operands.
#
# MFMA instruction:
#   v_mfma_f32_16x16x128_f8f6f4 a[acc:acc+3], v[B:B+7], v[A:A+7], a[acc:acc+3]
#   cbsz:0 blgp:0
#
# The 8-VGPR operand for A (or B) consists of two consecutive 4-VGPR tiles:
#   v[a_lo:a_lo+3] = K_lo tile (FP8 elements at K[K_group*16 : K_group*16+16])
#   v[a_lo+4:a_lo+7] = K_hi tile (FP8 elements at K[K_group*16+64 : K_group*16+80])
#
# Output layout per lane:
#   Lane l: M_row = l % 16, N_col_group = l // 16
#   4 float32 accumulators: C[M_row, N_col_group*4 : N_col_group*4+4]
#
# Usage:
#   pytest test_mfma_fp8.py -v -s
#   python test_mfma_fp8.py --debug
################################################################################

import os
import sys
import tempfile

import pytest
import numpy as np

import math

from gpu_test_helpers import (
    HAS_GFX950,
    GFX_TARGET,
    TileConfig,
    WAVESIZE, NUM_WAVES, NUM_THREADS,
    AB_B8,
    assemble_and_run,
    generate_kernel_asm,
    setup_roundtrip_writer,
    build_roundtrip_inner_asm,
    collect_tile_vgprs,
    requires_gpu,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BPE_FP8       = 1    # 1 byte per OCP FP8 element
MATRIX_INST_K = 128  # v_mfma_f32_16x16x128_f8f6f4
ACC_SIZE      = 4    # 4 AGPRs per MFMA output tile (4 float32 per lane)
MFMA_WAIT_NOPS = 16  # s_nop 7 repeated 16 times (128 cycles) after MFMAs

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=128, stride_b=128),
]

# ---------------------------------------------------------------------------
# FP8 E4M3FN helpers
# ---------------------------------------------------------------------------

def fp8_e4m3fn_to_float32(byte_val: int) -> float:
    """Decode OCP FP8 E4M3FN byte to Python float.

    Format: sign(1) | exponent(4, bias=7) | mantissa(3)
    Special: 0x7F and 0xFF are NaN; no infinity.
    """
    byte_val = int(byte_val)
    if byte_val & 0x7F == 0x7F:
        return float('nan')
    sign = -1.0 if (byte_val >> 7) else 1.0
    exp_stored = (byte_val >> 3) & 0xF
    mantissa   = byte_val & 0x7
    if exp_stored == 0:
        return sign * (mantissa / 8.0) * (2.0 ** -6)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exp_stored - 7))


def make_fp8_input(num_rows: int, num_cols: int, seed: int = 42) -> np.ndarray:
    """Build a random uint8 FP8 E4M3FN matrix.

    NaN bytes (0x7F, 0xFF) are replaced with 0x00 to avoid NaN propagation
    in the expected float32 matmul.
    """
    rng  = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=num_rows * num_cols, dtype=np.uint8)
    data[data == 0x7F] = 0x00
    data[data == 0xFF] = 0x00
    return data


# ---------------------------------------------------------------------------
# MFMA assembly generation
# ---------------------------------------------------------------------------


def generate_mfma_pairs(tileInfoA, tileInfoB, writer):
    """Enumerate MFMA pairs in (mma1 × mma0) order and allocate 4-AGPR accumulators.

    Matches the loop order in emitMfmaCode:
      for mmak in range(localMMATileGrid[1]):   # K-loop (=1 for FP8 DU128)
        for mma1 in range(localMMATileGrid[0]): # N tiles (B dimension)
          for mma0 in range(localMMATileGrid[0]): # M tiles (A dimension)

    For FP8 DU=128: localMMATileGrid = [8, 1]; atileId = mma0; btileId = mma1.

    Returns list of dicts: {mma0, mma1, a_lo, b_lo, acc}
    """
    pairs = []
    lrSubtileShapeA = tileInfoA.lr.subtileShape
    lrSubtileShapeB = tileInfoB.lr.subtileShape

    for mmak in range(tileInfoA.localMMATileGrid[1]):
        for mma1 in range(tileInfoB.localMMATileGrid[0]):
            for mma0 in range(tileInfoA.localMMATileGrid[0]):
                # A tile index (matches emitMfmaCode's atileId formula)
                lrGridA0 = tileInfoA.localMMATileGrid[0] // lrSubtileShapeA[0]
                numPerA  = int(lrSubtileShapeA[0]) * int(lrSubtileShapeA[1])
                aSId0    = mma0 // lrSubtileShapeA[0]
                aSId1    = mmak // lrSubtileShapeA[1]
                _mmak_A  = mmak % lrSubtileShapeA[1]
                atileId  = (aSId1 * lrGridA0 + aSId0) * numPerA + _mmak_A

                # B tile index
                lrGridB0 = tileInfoB.localMMATileGrid[0] // lrSubtileShapeB[0]
                numPerB  = int(lrSubtileShapeB[0]) * int(lrSubtileShapeB[1])
                bSId0    = mma1 // lrSubtileShapeB[0]
                bSId1    = mmak // lrSubtileShapeB[1]
                _mmak_B  = mmak % lrSubtileShapeB[1]
                btileId  = (bSId1 * lrGridB0 + bSId0) * numPerB + _mmak_B

                a_lo = tileInfoA.vgprTiles[atileId].regList.indices[0]
                b_lo = tileInfoB.vgprTiles[btileId].regList.indices[0]

                acc = writer.agprPool.checkOutAligned(4, 4, "mfma_acc",
                                                      preventOverflow=False)
                pairs.append({'mma0': mma0, 'mma1': mma1,
                               'a_lo': a_lo, 'b_lo': b_lo, 'acc': acc})
    return pairs


def generate_mfma_asm(mfma_pairs):
    """Generate AGPR zero + MFMA instruction assembly for all pairs.

    Returns assembly string (no export).
    """
    lines = [" // === Zero all MFMA accumulators ==="]
    for p in mfma_pairs:
        acc = p['acc']
        for k in range(ACC_SIZE):
            lines.append(f" v_accvgpr_write a{acc + k}, 0")
    lines.append(f" s_nop 7 // wait for accvgpr_write to settle")

    lines.append(" // === FP8 MFMA: v_mfma_f32_16x16x128_f8f6f4 cbsz:0 blgp:0 ===")
    lines.append(" // SRC0=B (N-dim, 8 VGPRs), SRC1=A (M-dim, 8 VGPRs), acc=4 AGPRs")
    for p in mfma_pairs:
        a   = p['a_lo']
        b   = p['b_lo']
        acc = p['acc']
        lines.append(
            f" v_mfma_f32_16x16x128_f8f6f4 a[{acc}:{acc + 3}],"
            f" v[{b}:{b + 7}], v[{a}:{a + 7}], a[{acc}:{acc + 3}]"
            f" cbsz:0 blgp:0"
            f" // C[mma0={p['mma0']}, mma1={p['mma1']}] += A @ B.T"
        )

    lines.append(f" // Wait {MFMA_WAIT_NOPS} × 8 = {MFMA_WAIT_NOPS * 8} cycles for MFMA results")
    lines.extend([f" s_nop 7"] * MFMA_WAIT_NOPS)
    return "\n".join(lines)


def generate_mfma_export_asm(mfma_pairs, tileInfoA, tileInfoB):
    """Generate assembly to export AGPR results from all waves in parallel.

    Each wave writes to its own region:
      output_ptr + wave_id * wave_region_size + pair_base + laneId * 16

    wave_region_size = num_pairs * WAVESIZE * ACC_SIZE * 4 bytes

    Returns: (asm_string, next_free_vgpr)
    """
    wave_region_size = len(mfma_pairs) * WAVESIZE * ACC_SIZE * 4

    used   = collect_tile_vgprs(tileInfoA, tileInfoB)
    next_v = max(used | {0}) + 1

    wave_off = next_v; next_v += 1
    lane     = next_v; next_v += 1
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1
    while next_v % 4 != 0:
        next_v += 1
    r0, r1, r2, r3 = next_v, next_v+1, next_v+2, next_v+3
    next_v += 4

    assert wave_region_size & (wave_region_size - 1) == 0, \
        "wave_region_size must be power of 2"
    shift = int(math.log2(wave_region_size))

    lines = [" // ---- MFMA export: all waves write in parallel ----"]
    lines.append(f" v_lshrrev_b32 v{wave_off}, 6, v0           // wave_id = tid / 64")
    lines.append(f" v_lshlrev_b32 v{wave_off}, {shift}, v{wave_off}"
                 f"  // wave_id * {wave_region_size}")
    lines.append(f" v_and_b32 v{lane}, 0x3F, v0               // lane_id = tid % 64")

    for pair_base, pair in enumerate(mfma_pairs):
        acc         = pair['acc']
        pair_offset = pair_base * WAVESIZE * 16

        lines.append(f" // Pair mma0={pair['mma0']} mma1={pair['mma1']}:"
                     f" a[{acc}:{acc+3}]")
        lines.append(f" v_accvgpr_read v{r0}, a{acc}")
        lines.append(f" v_accvgpr_read v{r1}, a{acc + 1}")
        lines.append(f" v_accvgpr_read v{r2}, a{acc + 2}")
        lines.append(f" v_accvgpr_read v{r3}, a{acc + 3}")
        lines.append(f" s_nop 1 // wait for accvgpr_read")
        lines.append(f" v_lshlrev_b32 v{addr_lo}, 4, v{lane}  // lane_id * 16 bytes")
        lines.append(f" v_add_u32 v{addr_lo}, v{wave_off}, v{addr_lo}"
                     f"  // + wave_offset")
        if pair_offset > 0:
            lines.append(f" v_add_u32 v{addr_lo}, {pair_offset}, v{addr_lo}"
                         f"  // + pair base")
        lines.append(f" v_mov_b32 v{addr_hi}, s9              // output_ptr hi")
        lines.append(f" v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f" v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f" flat_store_dwordx4 v[{addr_lo}:{addr_hi}],"
                     f" v[{r0}:{r3}]")
        lines.append(f" s_waitcnt vmcnt(0)")

    return "\n".join(lines), next_v


# ---------------------------------------------------------------------------
# Full kernel generator
# ---------------------------------------------------------------------------

def generate_mfma_kernel_fp8(cfg):
    """Generate a complete FP8 GR->LDS->LR->MFMA kernel.

    All 4 waves export in parallel; output buffer layout:
      wave_id * wave_region_size + pair_idx * WAVESIZE*16 + lane_id*16

    Returns:
        (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
         mfma_pairs, output_size, lds_size)
    """
    writer, kernel, tileInfoA, tileInfoB, lds_size = setup_roundtrip_writer(
        cfg, geometry=AB_B8, inst_k=MATRIX_INST_K, bpe=BPE_FP8)

    mfma_pairs  = generate_mfma_pairs(tileInfoA, tileInfoB, writer)
    mfma_asm    = generate_mfma_asm(mfma_pairs)
    export_asm, _ = generate_mfma_export_asm(mfma_pairs, tileInfoA, tileInfoB)

    # GR->LDS->LR pipeline + MFMA + export
    inner_asm = build_roundtrip_inner_asm(writer, kernel, mfma_asm + "\n" + export_asm)

    args = (
        ("input_A_ptr", 8, "global_buffer", "u8"),
        ("input_B_ptr", 8, "global_buffer", "u8"),
        ("output_ptr",  8, "global_buffer", "f32"),
        ("strideA",     4, "by_value",      "u32"),
        ("strideB",     4, "by_value",      "u32"),
    )
    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_size)

    wave_region_size = len(mfma_pairs) * WAVESIZE * ACC_SIZE * 4
    output_size      = NUM_WAVES * wave_region_size
    return (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
            mfma_pairs, output_size, lds_size)


# ---------------------------------------------------------------------------
# Expected MFMA output computation
# ---------------------------------------------------------------------------

def compute_expected_mfma_fp8(cfg, tileInfoA, tileInfoB, kernel,
                               input_A, input_B, wave_id):
    """Compute expected float32 MFMA accumulator values for each pair.

    The MFMA instruction computes (in AMD ISA notation):
      D[M, N] = SRC1[M, K] × SRC0[N, K] + C[M, N]
    where SRC0 = B (N-indexed) and SRC1 = A (M-indexed), with zero C.
    Equivalently: C_block = A_block @ B_block.T (float32)

    Output layout per MFMA pair, per lane:
      Lane l: M_row = l % 16, N_col_group = l // 16
      4 values: C_block[M_row, N_col_group*4 : N_col_group*4 + 4]

    Returns:
        list of np.ndarray (float32, shape WAVESIZE*ACC_SIZE), one per pair
        in (mma1, mma0) enumeration order.
    """
    mi_wave_group = kernel["MIWaveGroup"]

    A_f32 = np.array(
        [fp8_e4m3fn_to_float32(b) for b in input_A], dtype=np.float32
    ).reshape(cfg.mt_a, cfg.stride_a)
    B_f32 = np.array(
        [fp8_e4m3fn_to_float32(b) for b in input_B], dtype=np.float32
    ).reshape(cfg.mt_b, cfg.stride_b)

    wave_a_off = (wave_id % mi_wave_group[0]) * tileInfoA.localMMATileGrid[0] * 16
    wave_b_off = (wave_id // mi_wave_group[0]) * tileInfoB.localMMATileGrid[0] * 16

    results = []
    for mma1 in range(tileInfoB.localMMATileGrid[0]):
        for mma0 in range(tileInfoA.localMMATileGrid[0]):
            a_row = wave_a_off + mma0 * 16
            b_row = wave_b_off + mma1 * 16

            A_blk = A_f32[a_row:a_row + 16, :cfg.depth_u]
            B_blk = B_f32[b_row:b_row + 16, :cfg.depth_u]
            C_blk = (A_blk @ B_blk.T).astype(np.float32)

            lane_data = np.zeros(WAVESIZE * ACC_SIZE, dtype=np.float32)
            for lane in range(WAVESIZE):
                m  = lane % 16
                ng = lane // 16
                lane_data[lane * ACC_SIZE:(lane + 1) * ACC_SIZE] = (
                    C_blk[m, ng * ACC_SIZE:(ng + 1) * ACC_SIZE]
                )
            results.append(lane_data)

    return results


def compute_expected_all_waves(cfg, tileInfoA, tileInfoB, kernel, input_A, input_B):
    """Compute expected output for all 4 waves, concatenated in wave order."""
    all_pairs = []
    for wave_id in range(NUM_WAVES):
        all_pairs.extend(
            compute_expected_mfma_fp8(cfg, tileInfoA, tileInfoB, kernel,
                                      input_A, input_B, wave_id)
        )
    return all_pairs


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_mfma_output(actual_bytes, expected_pairs, debug=False):
    """Compare GPU output float32 values against Python-computed expected.

    Returns error count.
    """
    errors    = 0
    pair_size = WAVESIZE * ACC_SIZE * 4

    for pair_idx, expected in enumerate(expected_pairs):
        offset = pair_idx * pair_size
        actual = np.frombuffer(actual_bytes[offset:offset + pair_size],
                               dtype=np.float32)

        if not np.allclose(actual, expected, rtol=1e-3, atol=50.0):
            errors += 1
            if errors <= 4 or debug:
                for lane in range(WAVESIZE):
                    a_sl = actual  [lane * ACC_SIZE:(lane + 1) * ACC_SIZE]
                    e_sl = expected[lane * ACC_SIZE:(lane + 1) * ACC_SIZE]
                    if not np.allclose(a_sl, e_sl, rtol=1e-4, atol=1.0):
                        wave      = pair_idx // (len(expected_pairs) // NUM_WAVES)
                        local_pair = pair_idx  % (len(expected_pairs) // NUM_WAVES)
                        print(f"  MISMATCH wave={wave} pair={local_pair} lane={lane}:")
                        print(f"    expected: {e_sl.tolist()}")
                        print(f"    actual:   {a_sl.tolist()}")
                        if not debug:
                            break

    return errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@requires_gpu
class TestMfmaFP8:
    """GPU tests for FP8 GR->LDS->LR->MFMA roundtrip."""

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    def test_mfma_fp8(self, cfg, tmp_path):
        """Verify FP8 MFMA accumulator values after full GR->LDS->LR->MFMA pipeline.

        All 4 waves export in parallel; output is verified for each wave.
        """
        sys.stdout.flush()

        (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
         mfma_pairs, output_size, lds_size) = generate_mfma_kernel_fp8(cfg)

        input_A = make_fp8_input(cfg.mt_a, cfg.stride_a, seed=42)
        input_B = make_fp8_input(cfg.mt_b, cfg.stride_b, seed=43)

        label = f"mfma_fp8_{cfg.label}"
        output_bytes = assemble_and_run(
            kernel_asm, tmp_path, label, output_size,
            inputs=(input_A, input_B),
            scalars=(cfg.stride_a, cfg.stride_b),
            lds_size=lds_size,
        )

        expected_pairs = compute_expected_all_waves(
            cfg, tileInfoA, tileInfoB, kernel, input_A, input_B)
        errors = compare_mfma_output(output_bytes, expected_pairs)

        assert errors == 0, (
            f"Config {cfg.label}: {errors} MFMA pair mismatches"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FP8 MFMA GPU test")
    parser.add_argument("--debug",  action="store_true",
                        help="Print detailed mismatch output and kernel ASM")
    parser.add_argument("--config", type=int, default=None,
                        help="Config index to test (default: all)")
    args = parser.parse_args()

    if not HAS_GFX950:
        print(f"GPU tests require gfx950, found {GFX_TARGET} — cannot run GPU test")
        sys.exit(1)

    config_list  = CONFIGS if args.config is None else [CONFIGS[args.config]]
    total_errors = 0
    total_tests  = 0

    for cfg in config_list:
        total_tests += 1
        print(f"\n{'='*60}")
        print(f"FP8 MFMA Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")
        print(f"  stride_a={cfg.stride_a}, stride_b={cfg.stride_b}")

        (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
         mfma_pairs, output_size, lds_size) = generate_mfma_kernel_fp8(cfg)

        print(f"  MIWaveGroup: {kernel['MIWaveGroup']}")
        print(f"  A tiles: {len(tileInfoA.vgprTiles)}, "
              f"B tiles: {len(tileInfoB.vgprTiles)}")
        print(f"  MFMA pairs/wave: {len(mfma_pairs)}, "
              f"output: {output_size} bytes (all waves)")
        print(f"  LDS: {lds_size} bytes")

        if args.debug:
            for p in mfma_pairs[:4]:
                print(f"  pair mma0={p['mma0']} mma1={p['mma1']}: "
                      f"A v[{p['a_lo']}:{p['a_lo']+7}], "
                      f"B v[{p['b_lo']}:{p['b_lo']+7}], "
                      f"acc a[{p['acc']}:{p['acc']+3}]")
            print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {
                '__truediv__': lambda s, n: os.path.join(tmp_dir, n)
            })()

            input_A = make_fp8_input(cfg.mt_a, cfg.stride_a, seed=42)
            input_B = make_fp8_input(cfg.mt_b, cfg.stride_b, seed=43)

            output_bytes = assemble_and_run(
                kernel_asm, tmp_path, f"mfma_fp8_{cfg.label}", output_size,
                inputs=(input_A, input_B),
                scalars=(cfg.stride_a, cfg.stride_b),
                lds_size=lds_size,
            )

            expected_pairs = compute_expected_all_waves(
                cfg, tileInfoA, tileInfoB, kernel, input_A, input_B)
            errors = compare_mfma_output(output_bytes, expected_pairs,
                                         debug=args.debug)

            if errors == 0:
                print("  PASS")
            else:
                print(f"  FAIL: {errors} pair mismatches")
                total_errors += errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    if total_errors > 0:
        print("FAILED")
        sys.exit(1)
    else:
        print("PASSED")
