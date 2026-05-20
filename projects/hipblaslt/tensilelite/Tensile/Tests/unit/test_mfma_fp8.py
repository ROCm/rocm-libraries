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

from types import SimpleNamespace

from gpu_test_helpers import (
    HAS_GFX950,
    GFX_TARGET,
    TileConfig,
    WAVESIZE, NUM_WAVES, NUM_THREADS,
    AB_B8,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
    generate_srd_setup,
    setup_roundtrip_writer,
    build_roundtrip_inner_asm,
    collect_tile_vgprs,
    init_rocisa,
)

from rocisa.code import Module
from rocisa.container import sgpr, vgpr, accvgpr, VOP3PModifiers
from rocisa.enum import InstType
from rocisa.instruction import SBarrier, SMovB32, MXMFMAInstruction

from Tensile.Components.Subtile.Kernel import TileInfo, MXSA_E8M0, MXSB_E8M0
from Tensile.Components.Subtile.SubtileGREmit import (
    graTileAssignment, globalReadDTLInitCommonSgpr, globalReadDoSubtile,
)
from Tensile.Components.Subtile.SubtileLREmit import lraTileAssignment, localReadDoSubtile
from Tensile.Components.Subtile.SubtileScaleEmit import (
    globalReadScaleSwizzledDTLInitCommonSgpr,
    graTileAssignmentScaleSwizzled,
    lraTileAssignmentScaleSwizzled,
    emitScaleGRLoad,
    emitScaleLRLoad,
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
    TileConfig(mt_a=128, mt_b=128, depth_u=256, stride_a=256, stride_b=256),
]

# ---------------------------------------------------------------------------
# MX scale constants (for 128×128×256 FP8 config with MXBlock=32)
# ---------------------------------------------------------------------------

_MXBLOCK        = 32
_LR_SUBTILE_SZ  = 256   # bytes per LR subtile (2×2 MMA tiles × 64 bytes)
_SCALE_GROUPS   = 2     # lrLocalSubtileGrid[0] × [1] = 2 × 1
_SCALE_LDS_PER_WAVE = WAVESIZE * 16          # 1024 bytes (loadWidth * waveSize)
_SCALE_LDS_TOTAL    = NUM_WAVES * _SCALE_LDS_PER_WAVE   # 4096 bytes per tensor


# ---------------------------------------------------------------------------
# MX scale helpers (pre-swizzle + expected VGPR computation)
# ---------------------------------------------------------------------------

def pre_swizzle_scales_gfx950(canonical: np.ndarray) -> np.ndarray:
    """GFX950 32×8 pre-swizzle for a canonical (sm × sn) E8M0 scale matrix."""
    sm_orig, sn_orig = canonical.shape
    sm = int(np.ceil(sm_orig / 32)) * 32
    sn = int(np.ceil(sn_orig / 8)) * 8
    padded = np.zeros((sm, sn), dtype=canonical.dtype)
    padded[:sm_orig, :sn_orig] = canonical
    s = padded.reshape(sm // 32, 2, 16, sn // 8, 2, 4)
    s = s.transpose(0, 3, 5, 2, 4, 1)
    return s.reshape(sm, sn)


def make_scale_input(rows: int, cols: int, seed: int = 100) -> np.ndarray:
    """Random E8M0 uint8 scale matrix.

    Valid range is 0..254 (byte 255 = NaN is excluded). Byte 0 represents
    2^(-127) and is a valid scale value (E8M0 HasZero=false).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(rows, cols), dtype=np.uint8)


def compute_expected_scale_vgprs(flat_buf: np.ndarray, wave_id: int,
                                  is_a: bool) -> np.ndarray:
    """Expected scale VGPR bytes for each lane of wave_id (MT=128×128, DU=256).

    Returns uint8 array of shape (_SCALE_GROUPS, WAVESIZE, 4).
    """
    partition_idx    = (wave_id % 2) if is_a else (wave_id // 2)
    partition_offset = partition_idx * 512   # 512 bytes per partition
    expected = np.zeros((_SCALE_GROUPS, WAVESIZE, 4), dtype=np.uint8)
    for g in range(_SCALE_GROUPS):
        group_offset = g * _LR_SUBTILE_SZ   # 0 or 256
        for lane in range(WAVESIZE):
            lr_off = partition_offset + lane * 4 + group_offset
            expected[g, lane] = flat_buf[lr_off : lr_off + 4]
    return expected


# ---------------------------------------------------------------------------
# Combined writer setup (A/B MFMA + MXSA/MXSB scale)
# ---------------------------------------------------------------------------

def setup_roundtrip_writer_with_scale(cfg):
    """Extend a roundtrip writer with MXSA/MXSB scale TileInfo.

    Calls setup_roundtrip_writer for A/B, then allocates scale SGPRs and
    creates scale TileInfo objects.  ldsTotalSize is set to _SCALE_LDS_TOTAL
    (= 4096, the scale LDS swap mask); the A/B swap value will be incorrect
    but is never used in single-iteration unit tests.

    Returns:
        (writer, kernel, tileInfoA, tileInfoB, tiMXSA, tiMXSB,
         lds_size_ab, lds_declared)
    """
    writer, kernel, tileInfoA, tileInfoB, lds_size_ab = setup_roundtrip_writer(
        cfg, geometry=AB_B8, inst_k=MATRIX_INST_K, bpe=BPE_FP8)

    # Add MX fields to kernel dict
    k_scale = cfg.depth_u // _MXBLOCK
    kernel["_DepthUMXSA"]  = k_scale
    kernel["_DepthUMXSB"]  = k_scale
    kernel["NonTemporalMXSA"] = 0
    kernel["NonTemporalMXSB"] = 0
    kernel["ProblemType"]["MXBlockA"] = _MXBLOCK
    kernel["ProblemType"]["MXBlockB"] = _MXBLOCK

    # Allocate consecutive pair for strides (enables single s_load_dwordx2)
    strides_base = writer.sgprPool.checkOut(2, preventOverflow=False)
    writer.sgprs["StridesMXSA"] = strides_base
    writer.sgprs["StridesMXSB"] = strides_base + 1

    # 4-dword-aligned SRD descriptors for scale (base loaded directly from kernarg)
    writer.sgprs["SrdMXSA"] = writer.sgprPool.checkOutAligned(
        4, 4, "SrdMXSA", preventOverflow=False)
    writer.sgprs["SrdMXSB"] = writer.sgprPool.checkOutAligned(
        4, 4, "SrdMXSB", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrMXSA"] = writer.sgprPool.checkOut(
        1, preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrMXSB"] = writer.sgprPool.checkOut(
        1, preventOverflow=False)
    writer.sgprs["SwapMXSA"] = writer.sgprPool.checkOut(1, preventOverflow=False)
    writer.sgprs["SwapMXSB"] = writer.sgprPool.checkOut(1, preventOverflow=False)

    # Create scale TileInfo — states.mxsa/mxsb must exist before TileInfo init
    # (globalReadScaleSwizzledDTLInitCommonSgpr reads them at emit time, not init)
    tiMXSA = TileInfo(MXSA_E8M0, 'MXSA', writer, kernel)
    tiMXSB = TileInfo(MXSB_E8M0, 'MXSB', writer, kernel)
    writer.states.mxsa = SimpleNamespace(tileInfo=tiMXSA)
    writer.states.mxsb = SimpleNamespace(tileInfo=tiMXSB)

    tiMXSA.allocOffsetRegisters(writer, kernel)
    tiMXSB.allocOffsetRegisters(writer, kernel)
    tiMXSA.allocVgprTileRegisters_legacy(writer, kernel)
    tiMXSB.allocVgprTileRegisters_legacy(writer, kernel)

    # Scale LDS after A/B LDS.  ldsTotalSize = scale swap mask (4096).
    # XOR property: ldsStartOffsetMXSA XOR (ldsStartOffsetMXSA + 4096) = 4096
    # holds whenever ldsStartOffsetMXSA has bit-13 = 0 (i.e. is a multiple of 8192,
    # or < 8192).  For lds_size_ab = 65536 = 0x10000: 0x10000 XOR 0x11000 = 0x1000. ✓
    writer.ldsStartOffsetMXSA = lds_size_ab
    writer.ldsStartOffsetMXSB = lds_size_ab + _SCALE_LDS_TOTAL
    writer.ldsTotalSize       = _SCALE_LDS_TOTAL   # scale swap mask

    # Declared LDS: A/B + MXSA double-buffer + MXSB double-buffer
    lds_declared = lds_size_ab + 4 * _SCALE_LDS_TOTAL   # ab + 4×4096
    return (writer, kernel, tileInfoA, tileInfoB, tiMXSA, tiMXSB,
            lds_size_ab, lds_declared)


def _generate_scale_srd_setup():
    """Set SrdMXSA/B descriptor fields 2 and 3 (base already in fields 0:1)."""
    module = Module("SRD scale")
    module.add(SMovB32(dst=sgpr("SrdMXSA+2"), src="0xFFFFFFFF",
                       comment="SrdMXSA NumRecords"))
    module.add(SMovB32(dst=sgpr("SrdMXSA+3"), src="0x20000",
                       comment="SrdMXSA OOB_SELECT=2"))
    module.add(SMovB32(dst=sgpr("SrdMXSB+2"), src="0xFFFFFFFF",
                       comment="SrdMXSB NumRecords"))
    module.add(SMovB32(dst=sgpr("SrdMXSB+3"), src="0x20000",
                       comment="SrdMXSB OOB_SELECT=2"))
    return module


# ---------------------------------------------------------------------------
# Scale VGPR export (all 4 waves in parallel)
# ---------------------------------------------------------------------------

def _generate_scale_export_asm(tileInfoA, tileInfoB, tiMXSA, tiMXSB,
                                mfma_output_size):
    """Generate assembly to export all scale VGPRs for all 4 waves in parallel.

    Output layout (appended after mfma_output_size bytes):
      wave_id * scale_wave_region  +  [MXSA_g0 | MXSA_g1 | MXSB_g0 | MXSB_g1]
    Each group = WAVESIZE × 4 = 256 bytes.  scale_wave_region = 4 × 256 = 1024.
    """
    scale_wave_region = _SCALE_GROUPS * 2 * WAVESIZE * 4  # 1024 bytes per wave
    shift = int(math.log2(scale_wave_region))               # 10

    # Collect all used VGPRs to find free temporaries
    used = collect_tile_vgprs(tileInfoA, tileInfoB)
    for ti in (tiMXSA, tiMXSB):
        for t in ti.vgprTiles:
            used.update(t.regList.indices)
        used.update(ti.sharedVgprGROffset)
        used.update(ti.sharedVgprLROffset)
        used.update(ti.sharedVgprLROffsetSwap)

    next_v = max(used | {0}) + 1
    wave_off = next_v; next_v += 1
    lane     = next_v; next_v += 1
    if next_v % 2 != 0:
        next_v += 1
    alo = next_v; next_v += 1
    ahi = next_v; next_v += 1

    lines = [" // ---- Scale VGPR export: all waves in parallel ----"]
    lines.append(f" v_lshrrev_b32 v{wave_off}, 6, v0"
                 f"          // wave_id = tid >> 6")
    lines.append(f" v_lshlrev_b32 v{wave_off}, {shift}, v{wave_off}"
                 f"  // wave_id * {scale_wave_region}")
    # Fold mfma_output_size into wave_off (both are per-wave constants)
    lines.append(f" v_add_u32 v{wave_off}, {mfma_output_size}, v{wave_off}"
                 f"  // + mfma_output_size")
    lines.append(f" v_and_b32 v{lane}, 0x3F, v0             // lane_id")
    lines.append(f" v_lshlrev_b32 v{lane}, 2, v{lane}       // lane_id * 4")

    group_idx = 0
    for tc_label, ti in [("MXSA", tiMXSA), ("MXSB", tiMXSB)]:
        seen_vgpr = []
        for tile in ti.vgprTiles:
            vstart = tile.regList.indices[0] if tile.regList.indices else None
            if vstart is None or vstart in seen_vgpr:
                continue
            seen_vgpr.append(vstart)
            within_wave_off = group_idx * WAVESIZE * 4   # byte offset within wave region

            lines.append(f" // {tc_label} group {len(seen_vgpr)-1}: v{vstart}")
            lines.append(f" v_add_u32 v{alo}, v{wave_off}, v{lane}  // wave_off + lane*4")
            if within_wave_off:
                lines.append(f" v_add_u32 v{alo}, {within_wave_off}, v{alo}"
                             f"  // + group offset")
            lines.append(f" v_mov_b32 v{ahi}, s9")
            lines.append(f" v_add_co_u32 v{alo}, vcc, s8, v{alo}")
            lines.append(f" v_addc_co_u32 v{ahi}, vcc, v{ahi}, 0, vcc")
            lines.append(f" flat_store_dword v[{alo}:{ahi}], v{vstart}")
            lines.append(f" s_waitcnt vmcnt(0)")
            group_idx += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined FP8 MFMA + scale GR/LR kernel
# ---------------------------------------------------------------------------

def generate_mfma_kernel_fp8_with_scale(cfg):
    """Generate a combined FP8 GR→LDS→LR→MFMA + scale GR→LDS→LR kernel.

    Kernarg layout (inputs in this order: A, B, scale_A, scale_B):
      0x00 input_A_ptr  (8B) → s[4:5]
      0x08 input_B_ptr  (8B) → s[6:7]
      0x10 scale_A_ptr  (8B) → SrdMXSA[0:1]
      0x18 scale_B_ptr  (8B) → SrdMXSB[0:1]
      0x20 output_ptr   (8B) → s[8:9]
      0x28 strideA      (4B) → s10
      0x2C strideB      (4B) → s11
      0x30 strideMXSA   (4B) → StridesMXSA
      0x34 strideMXSB   (4B) → StridesMXSB

    Output layout:
      [mfma_output_size bytes]  then
      [wave 0 scale: MXSA_g0, MXSA_g1, MXSB_g0, MXSB_g1] ... [wave 3 scale]

    Returns:
        (kernel_asm, writer, kernel, tileInfoA, tileInfoB, tiMXSA, tiMXSB,
         mfma_pairs, mfma_output_size, output_size, lds_declared)
    """
    (writer, kernel, tileInfoA, tileInfoB, tiMXSA, tiMXSB,
     lds_size_ab, lds_declared) = setup_roundtrip_writer_with_scale(cfg)

    # MFMA pairs and assembly — use MX-scale MFMA to test actual byte selection
    mfma_pairs = generate_mfma_pairs(tileInfoA, tileInfoB, writer)
    mfma_asm   = generate_mfma_asm_with_scale(mfma_pairs, tiMXSA, tiMXSB)

    wave_region_mfma = len(mfma_pairs) * WAVESIZE * ACC_SIZE * 4
    mfma_output_size = NUM_WAVES * wave_region_mfma

    # Collect scale VGPRs so MFMA export temps don't overlap with them.
    # (Scale tile VGPRs hold ds_read results that must survive past MFMA export.)
    scale_used = set()
    for ti in (tiMXSA, tiMXSB):
        for t in ti.vgprTiles:
            scale_used.update(t.regList.indices)
        scale_used.update(ti.sharedVgprGROffset)
        scale_used.update(ti.sharedVgprLROffset)
        scale_used.update(ti.sharedVgprLROffsetSwap)

    mfma_export_asm, _ = generate_mfma_export_asm(mfma_pairs, tileInfoA, tileInfoB,
                                                    extra_used=scale_used)
    scale_export_asm   = _generate_scale_export_asm(
        tileInfoA, tileInfoB, tiMXSA, tiMXSB, mfma_output_size)

    # A/B pipeline modules
    gra_ab    = graTileAssignment(writer, kernel, useSwizzling=True)
    lra_ab    = lraTileAssignment(writer, kernel)
    dtl_ab    = globalReadDTLInitCommonSgpr(writer, kernel)
    gr_a      = globalReadDoSubtile('A', writer, kernel)
    gr_b      = globalReadDoSubtile('B', writer, kernel)
    wait_gr   = SBarrier()   # use barrier; actual GR wait is implicit after vmcnt
    lr_a      = localReadDoSubtile('A', writer, kernel)
    lr_b      = localReadDoSubtile('B', writer, kernel)

    # Scale pipeline modules
    dtl_sc    = globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel)
    gra_sc    = graTileAssignmentScaleSwizzled(writer, kernel)
    lra_sc    = lraTileAssignmentScaleSwizzled(writer, kernel)
    gr_mxsa   = emitScaleGRLoad(tiMXSA, writer, kernel)
    gr_mxsb   = emitScaleGRLoad(tiMXSB, writer, kernel)
    lr_mxsa   = emitScaleLRLoad(tiMXSA, writer, kernel)
    lr_mxsb   = emitScaleLRLoad(tiMXSB, writer, kernel)

    from rocisa.instruction import SWaitCnt
    wait_vmcnt = SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="wait GR")
    barrier    = SBarrier(comment="sync LDS")
    wait_lgkm  = SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait LR")

    # Extended prologue (covers all 9 kernel args)
    prologue = generate_load_params([
        (4,                            4, 0x00, "input_A_ptr + input_B_ptr"),
        (writer.sgprs["SrdMXSA"],      2, 0x10, "scale_A_ptr -> SrdMXSA[0:1]"),
        (writer.sgprs["SrdMXSB"],      2, 0x18, "scale_B_ptr -> SrdMXSB[0:1]"),
        (8,                            4, 0x20, "output_ptr + strideA + strideB"),
        (writer.sgprs["StridesMXSA"],  2, 0x30, "strideMXSA + strideMXSB"),
    ])

    inner_asm = "\n".join([
        str(prologue),
        str(generate_srd_setup()),         # A/B SRD: s[4:5] → SrdA, s[6:7] → SrdB
        str(_generate_scale_srd_setup()),  # Scale SRD: complete fields 2 and 3
        # A/B pipeline
        str(gra_ab),
        str(lra_ab),
        str(dtl_ab),
        str(gr_a),
        str(gr_b),
        str(wait_vmcnt),
        str(barrier),
        str(lr_a),
        str(lr_b),
        str(wait_lgkm),
        # Scale pipeline
        str(dtl_sc),
        str(gra_sc),
        str(lra_sc),
        str(gr_mxsa),
        str(gr_mxsb),
        str(SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="wait scale GR")),
        str(SBarrier(comment="sync scale LDS")),
        str(lr_mxsa),
        str(lr_mxsb),
        str(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait scale LR")),
        # MFMA + export
        mfma_asm,
        mfma_export_asm,
        scale_export_asm,
    ])

    args = (
        ("input_A_ptr",  8, "global_buffer", "u8"),
        ("input_B_ptr",  8, "global_buffer", "u8"),
        ("scale_A_ptr",  8, "global_buffer", "u8"),
        ("scale_B_ptr",  8, "global_buffer", "u8"),
        ("output_ptr",   8, "global_buffer", "f32"),
        ("strideA",      4, "by_value",      "u32"),
        ("strideB",      4, "by_value",      "u32"),
        ("strideMXSA",   4, "by_value",      "u32"),
        ("strideMXSB",   4, "by_value",      "u32"),
    )
    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_declared)

    scale_output_size = NUM_WAVES * _SCALE_GROUPS * 2 * WAVESIZE * 4
    output_size = mfma_output_size + scale_output_size
    return (kernel_asm, writer, kernel, tileInfoA, tileInfoB, tiMXSA, tiMXSB,
            mfma_pairs, mfma_output_size, output_size, lds_declared)


# ---------------------------------------------------------------------------
# Scale output comparison
# ---------------------------------------------------------------------------

def compare_scale_output(scale_bytes, flat_a, flat_b, debug=False):
    """Verify scale VGPRs for all 4 waves.

    scale_bytes: raw bytes starting right after the MFMA output section.
    Layout: wave_id * (4 groups × WAVESIZE × 4 bytes) + group_idx * WAVESIZE * 4 + lane * 4.
    Returns error count.
    """
    errors = 0
    scale_wave_region = _SCALE_GROUPS * 2 * WAVESIZE * 4   # 1024 bytes per wave

    for wave_id in range(NUM_WAVES):
        wave_data = scale_bytes[wave_id * scale_wave_region :
                                (wave_id + 1) * scale_wave_region]
        exp_a = compute_expected_scale_vgprs(flat_a, wave_id, is_a=True)
        exp_b = compute_expected_scale_vgprs(flat_b, wave_id, is_a=False)

        off = 0
        for tc_label, expected in [("MXSA", exp_a), ("MXSB", exp_b)]:
            for g in range(_SCALE_GROUPS):
                chunk = np.frombuffer(wave_data[off : off + WAVESIZE * 4],
                                      dtype=np.uint8).reshape(WAVESIZE, 4)
                for lane in range(WAVESIZE):
                    if not np.array_equal(chunk[lane], expected[g, lane]):
                        errors += 1
                        if errors <= 8 or debug:
                            print(f"  SCALE MISMATCH wave={wave_id} {tc_label}"
                                  f" g={g} lane={lane}:"
                                  f" got {list(chunk[lane])}"
                                  f" expected {list(expected[g, lane])}")
                off += WAVESIZE * 4

    return errors


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


def e8m0_to_float(v: int) -> float:
    """Decode E8M0 byte to float32. Value = 2^(v - 127).

    v=0 → 2^(-127) (valid minimum scale; E8M0 HasZero=false).
    v=255 → NaN (excluded by make_scale_input).
    """
    v = int(v) & 0xFF
    if v == 255:
        return float('nan')
    return 2.0 ** (v - 127)


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
    """Enumerate MFMA pairs in (mma1 × mma0) order, each accumulating over all K-groups.

    One accumulator is allocated per (mma0, mma1) pair. For each pair, k_tiles
    lists the (a_lo, b_lo) vgpr bases for every K-group (mmak). The MFMA
    instructions for a pair are issued in K order, all targeting the same AGPR
    accumulator, matching emitMfmaCode's accumulation semantics.

    Returns list of dicts: {mma0, mma1, acc, k_tiles: [{a_lo, b_lo, mmak}]}
    """
    pairs = []
    lrSubtileShapeA = tileInfoA.lr.subtileShape
    lrSubtileShapeB = tileInfoB.lr.subtileShape
    lrGridA0 = tileInfoA.localMMATileGrid[0] // lrSubtileShapeA[0]
    lrGridB0 = tileInfoB.localMMATileGrid[0] // lrSubtileShapeB[0]
    numPerA  = int(lrSubtileShapeA[0]) * int(lrSubtileShapeA[1])
    numPerB  = int(lrSubtileShapeB[0]) * int(lrSubtileShapeB[1])

    for mma1 in range(tileInfoB.localMMATileGrid[0]):
        for mma0 in range(tileInfoA.localMMATileGrid[0]):
            acc = writer.agprPool.checkOutAligned(4, 4, "mfma_acc",
                                                  preventOverflow=False)
            k_tiles = []
            for mmak in range(tileInfoA.localMMATileGrid[1]):
                # A tile index (matches emitMfmaCode's atileId formula)
                aSId0   = mma0 // lrSubtileShapeA[0]
                aSId1   = mmak // lrSubtileShapeA[1]
                _mmak_A = mmak % lrSubtileShapeA[1]
                atileId = (aSId1 * lrGridA0 + aSId0) * numPerA + _mmak_A

                # B tile index
                bSId0   = mma1 // lrSubtileShapeB[0]
                bSId1   = mmak // lrSubtileShapeB[1]
                _mmak_B = mmak % lrSubtileShapeB[1]
                btileId = (bSId1 * lrGridB0 + bSId0) * numPerB + _mmak_B

                a_lo = tileInfoA.vgprTiles[atileId].regList.indices[0]
                b_lo = tileInfoB.vgprTiles[btileId].regList.indices[0]
                k_tiles.append({'a_lo': a_lo, 'b_lo': b_lo, 'mmak': mmak})

            pairs.append({'mma0': mma0, 'mma1': mma1, 'acc': acc, 'k_tiles': k_tiles})
    return pairs


def generate_mfma_asm(mfma_pairs):
    """Generate AGPR zero + MFMA instruction assembly for all pairs.

    Each pair accumulates over all K-groups (k_tiles) into a single AGPR
    accumulator, matching emitMfmaCode semantics.

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
        acc = p['acc']
        for kt in p['k_tiles']:
            a = kt['a_lo']
            b = kt['b_lo']
            lines.append(
                f" v_mfma_f32_16x16x128_f8f6f4 a[{acc}:{acc + 3}],"
                f" v[{b}:{b + 7}], v[{a}:{a + 7}], a[{acc}:{acc + 3}]"
                f" cbsz:0 blgp:0"
                f" // C[mma0={p['mma0']}, mma1={p['mma1']}, mmak={kt['mmak']}] += A @ B.T"
            )

    lines.append(f" // Wait {MFMA_WAIT_NOPS} × 8 = {MFMA_WAIT_NOPS * 8} cycles for MFMA results")
    lines.extend([f" s_nop 7"] * MFMA_WAIT_NOPS)
    return "\n".join(lines)


def generate_mfma_asm_with_scale(mfma_pairs, tiMXSA, tiMXSB):
    """Generate v_mfma_scale_f32_16x16x128_f8f6f4 instructions using loaded scale VGPRs.

    Operand order matches generate_mfma_asm: SRC0=B (a param), SRC1=A (b param).
    op_sel/op_sel_hi select the correct E8M0 byte from the 4-byte scale VGPR.
    Requires rocIsa initialized for gfx950 (done by init_rocisa()).

    Scale byte selection formula (mirrors emitMfmaCode):
      scaleGroupA = (mma0 // scaleMShapeA) * scaleKGridA + mmak // scaleKShapeA
      sAsel       = (mma0 % scaleMShapeA) + scaleMShapeA * (mmak % scaleKShapeA)
    """
    scaleMShapeA = tiMXSA.lrSubtileShape[0]
    scaleMShapeB = tiMXSB.lrSubtileShape[0]
    scaleKShapeA = tiMXSA.lrSubtileShape[1]
    scaleKShapeB = tiMXSB.lrSubtileShape[1]
    scaleKGridA  = tiMXSA.lrLocalSubtileGrid[1]
    scaleKGridB  = tiMXSB.lrLocalSubtileGrid[1]

    lines = [" // === Zero all MFMA accumulators ==="]
    for p in mfma_pairs:
        acc = p['acc']
        for k in range(ACC_SIZE):
            lines.append(f" v_accvgpr_write a{acc + k}, 0")
    lines.append(" s_nop 7 // wait for accvgpr_write to settle")

    lines.append(" // === MX FP8 MFMA: v_mfma_scale_f32_16x16x128_f8f6f4 ===")
    for p in mfma_pairs:
        acc  = p['acc']
        mma0 = p['mma0']
        mma1 = p['mma1']
        for kt in p['k_tiles']:
            a_lo = kt['a_lo']
            b_lo = kt['b_lo']
            mmak = kt['mmak']

            scaleGroupA = (mma0 // scaleMShapeA) * scaleKGridA + mmak // scaleKShapeA
            scaleGroupB = (mma1 // scaleMShapeB) * scaleKGridB + mmak // scaleKShapeB
            sAsel = (mma0 % scaleMShapeA) + scaleMShapeA * (mmak % scaleKShapeA)
            sBsel = (mma1 % scaleMShapeB) + scaleMShapeB * (mmak % scaleKShapeB)
            scaleAVgpr = tiMXSA.vgprTiles[4 * scaleGroupA].regList.indices[0]
            scaleBVgpr = tiMXSB.vgprTiles[4 * scaleGroupB].regList.indices[0]

            instr = MXMFMAInstruction(
                instType=InstType.INST_F8, accType=InstType.INST_F32,
                variant=[16, 16, MATRIX_INST_K, 1],
                acc=accvgpr(acc, ACC_SIZE),
                a=vgpr(b_lo, 8),            # SRC0 = B (N-indexed)
                b=vgpr(a_lo, 8),            # SRC1 = A (M-indexed)
                acc2=accvgpr(acc, ACC_SIZE),
                mxsa=vgpr(scaleAVgpr),      # scale for A (SRC1)
                mxsb=vgpr(scaleBVgpr),      # scale for B (SRC0)
                vop3=VOP3PModifiers(
                    op_sel=[sAsel % 2, sBsel % 2],
                    op_sel_hi=[(sAsel >> 1) % 2, (sBsel >> 1) % 2],
                ),
                comment=f"MX C[{mma0},{mma1}] += scA*A*scB*B K={mmak}",
            )
            lines.append(f" {str(instr)}")

    lines.append(f" // Wait {MFMA_WAIT_NOPS} × 8 = {MFMA_WAIT_NOPS * 8} cycles for MFMA results")
    lines.extend([" s_nop 7"] * MFMA_WAIT_NOPS)
    return "\n".join(lines)


def generate_mfma_export_asm(mfma_pairs, tileInfoA, tileInfoB, extra_used=None):
    """Generate assembly to export AGPR results from all waves in parallel.

    Each wave writes to its own region:
      output_ptr + wave_id * wave_region_size + pair_base + laneId * 16

    wave_region_size = num_pairs * WAVESIZE * ACC_SIZE * 4 bytes

    Args:
        extra_used: optional set of additional VGPR indices to reserve
                    (e.g. scale tile VGPRs) so temp registers don't overlap.

    Returns: (asm_string, next_free_vgpr)
    """
    wave_region_size = len(mfma_pairs) * WAVESIZE * ACC_SIZE * 4

    used = collect_tile_vgprs(tileInfoA, tileInfoB)
    if extra_used:
        used.update(extra_used)
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


def compute_expected_mfma_fp8_with_scale(cfg, tileInfoA, tileInfoB,
                                          tiMXSA, tiMXSB, kernel,
                                          input_A, input_B,
                                          flat_a, flat_b, wave_id):
    """Compute expected float32 MFMA values with MX scale byte selection.

    The 16×16×128 MFMA with mxBlock=32 uses 4 K-sub-groups of 32 elements
    each.  For output lane l = m + ng*16 (M-row m, N-group ng):

      D[l, acc] = sum_{q=0}^{3}
                    scale_a(m, q) * scale_b(n_col, q)
                    * dot(A[row_a+m, k_start+q*32 : +32],
                          B[row_b+n_col, k_start+q*32 : +32])

    where n_col = ng*4 + acc, and the scale for K-sub-group q comes from
    the lane encoding that sub-group:
      scale_a(m, q) = flat_a[partition_a*psSzA + (m + q*16)*4 + gA*lrSzA + sAsel]
      scale_b(n, q) = flat_b[partition_b*psSzB + (n + q*16)*4 + gB*lrSzB + sBsel]

    This matches the pre_swizzle_scales_gfx950 layout, where lane l = M + q*16
    carries scale for (M-row=l%16, K-sub-group=l//16).

    partition_a = wave_id % 2   (M-wave index for MXSA)
    partition_b = wave_id // 2  (N-wave index for MXSB)

    Returns list of np.ndarray (float32, shape WAVESIZE*ACC_SIZE), one per pair
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

    # Scale geometry (mirrors emitMfmaCode)
    scaleMShapeA = tiMXSA.lrSubtileShape[0]
    scaleMShapeB = tiMXSB.lrSubtileShape[0]
    scaleKShapeA = tiMXSA.lrSubtileShape[1]
    scaleKShapeB = tiMXSB.lrSubtileShape[1]
    scaleKGridA  = tiMXSA.lrLocalSubtileGrid[1]
    scaleKGridB  = tiMXSB.lrLocalSubtileGrid[1]

    lrSzA = int(tiMXSA.lrSubtileSize)
    lrSzB = int(tiMXSB.lrSubtileSize)
    psSzA = int(tiMXSA.lrLocalSubtileGrid[0]) * lrSzA   # bytes per M-partition
    psSzB = int(tiMXSB.lrLocalSubtileGrid[0]) * lrSzB

    partition_a = wave_id % 2    # M-wave index
    partition_b = wave_id // 2   # N-wave index

    # Number of mxBlock-sized K-sub-groups per MFMA invocation
    n_sub = MATRIX_INST_K // _MXBLOCK   # 128 // 32 = 4

    results = []
    for mma1 in range(tileInfoB.localMMATileGrid[0]):
        for mma0 in range(tileInfoA.localMMATileGrid[0]):
            lane_data = np.zeros(WAVESIZE * ACC_SIZE, dtype=np.float32)

            for mmak in range(tileInfoA.localMMATileGrid[1]):
                scaleGroupA = (mma0 // scaleMShapeA) * scaleKGridA + mmak // scaleKShapeA
                scaleGroupB = (mma1 // scaleMShapeB) * scaleKGridB + mmak // scaleKShapeB
                sAsel = (mma0 % scaleMShapeA) + scaleMShapeA * (mmak % scaleKShapeA)
                sBsel = (mma1 % scaleMShapeB) + scaleMShapeB * (mmak % scaleKShapeB)

                row_a   = wave_a_off + mma0 * 16
                row_b   = wave_b_off + mma1 * 16
                k_start = mmak * MATRIX_INST_K

                # Each output lane l = m + ng*16; accumulate over 4 K-sub-groups.
                # For K-sub-group q: A-scale comes from lane (m + q*16),
                # B-scale from lane (n_col + q*16).
                for m in range(16):
                    for q in range(n_sub):
                        a_lane = m + q * 16
                        sa_off = partition_a * psSzA + a_lane * 4 + scaleGroupA * lrSzA + sAsel
                        scale_a = e8m0_to_float(flat_a[sa_off])

                        k_sub = k_start + q * _MXBLOCK
                        A_vec = A_f32[row_a + m, k_sub : k_sub + _MXBLOCK].astype(np.float64)

                        for ng in range(ACC_SIZE):   # ng = lane // 16 in {0,1,2,3}
                            lane = m + ng * 16
                            for acc_idx in range(ACC_SIZE):
                                n_col  = ng * ACC_SIZE + acc_idx
                                b_lane = n_col + q * 16
                                sb_off = (partition_b * psSzB + b_lane * 4
                                          + scaleGroupB * lrSzB + sBsel)
                                scale_b = e8m0_to_float(flat_b[sb_off])

                                B_vec = B_f32[row_b + n_col, k_sub : k_sub + _MXBLOCK].astype(np.float64)
                                dp = float(np.dot(A_vec, B_vec))
                                lane_data[lane * ACC_SIZE + acc_idx] += float(scale_a * scale_b) * dp

            results.append(lane_data)

    return results


def compute_expected_all_waves_with_scale(cfg, tileInfoA, tileInfoB,
                                           tiMXSA, tiMXSB, kernel,
                                           input_A, input_B, flat_a, flat_b):
    """Compute scaled expected output for all 4 waves, concatenated in wave order."""
    all_pairs = []
    for wave_id in range(NUM_WAVES):
        all_pairs.extend(
            compute_expected_mfma_fp8_with_scale(
                cfg, tileInfoA, tileInfoB, tiMXSA, tiMXSB, kernel,
                input_A, input_B, flat_a, flat_b, wave_id)
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

@pytest.mark.skipif(not HAS_GFX950, reason=f"GPU tests require gfx950, found {GFX_TARGET}")
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
# FP8 MFMA + scale GR/LR combined test (128×128×256 only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GFX950, reason=f"GPU tests require gfx950, found {GFX_TARGET}")
class TestMfmaFP8WithScale:
    """GPU test: FP8 MFMA + scale GR→LDS→LR verification for MT=128×128, DU=256."""

    def test_mfma_and_scale(self, tmp_path):
        """Verify MX-scaled MFMA accumulators and scale VGPRs after the combined pipeline.

        Uses v_mfma_scale_f32_16x16x128_f8f6f4 with actual scale VGPRs and op_sel
        byte selection (sAsel/sBsel), exercising the same logic as emitMfmaCode.

        Pipeline:
          A/B GR→LDS→LR → v_mfma_scale (with scale byte selection via op_sel)
          scale_A/B GR(DTL)→LDS→LR (exported separately for independent verification)
        Output:
          [MX-scaled MFMA results for all 4 waves] + [scale VGPRs for all 4 waves]
        """
        cfg = CONFIGS[1]   # TileConfig(mt_a=128, mt_b=128, depth_u=256, ...)

        (kernel_asm, writer, kernel, tileInfoA, tileInfoB, tiMXSA, tiMXSB,
         mfma_pairs, mfma_output_size, output_size, lds_declared) = \
            generate_mfma_kernel_fp8_with_scale(cfg)

        # A/B FP8 data inputs
        input_A = make_fp8_input(cfg.mt_a, cfg.stride_a, seed=42)
        input_B = make_fp8_input(cfg.mt_b, cfg.stride_b, seed=43)

        # Scale inputs: canonical → pre-swizzled
        k_scale = cfg.depth_u // _MXBLOCK   # 8 K-scale cols
        canon_a = make_scale_input(cfg.mt_a, k_scale, seed=10)
        canon_b = make_scale_input(cfg.mt_b, k_scale, seed=20)
        flat_a  = pre_swizzle_scales_gfx950(canon_a).flatten()
        flat_b  = pre_swizzle_scales_gfx950(canon_b).flatten()

        scale_stride = _LR_SUBTILE_SZ   # 256 (elements stride between M-groups)

        output_bytes = assemble_and_run(
            kernel_asm, tmp_path, "mfma_fp8_with_scale", output_size,
            inputs=(input_A, input_B, flat_a, flat_b),
            scalars=(cfg.stride_a, cfg.stride_b, scale_stride, scale_stride),
            lds_size=lds_declared,
        )

        # Verify MFMA accumulators with MX scale byte selection
        expected_pairs = compute_expected_all_waves_with_scale(
            cfg, tileInfoA, tileInfoB, tiMXSA, tiMXSB, kernel,
            input_A, input_B, flat_a, flat_b)
        mfma_errors = compare_mfma_output(output_bytes[:mfma_output_size],
                                          expected_pairs)

        # Verify scale VGPRs for all 4 waves (data path independent check)
        scale_errors = compare_scale_output(output_bytes[mfma_output_size:],
                                            flat_a, flat_b)

        assert mfma_errors  == 0, f"{cfg.label}: {mfma_errors} MX-MFMA mismatches"
        assert scale_errors == 0, f"{cfg.label}: {scale_errors} scale VGPR mismatches"


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
