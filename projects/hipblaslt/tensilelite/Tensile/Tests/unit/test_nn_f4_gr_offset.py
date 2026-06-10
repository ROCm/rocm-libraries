#!/usr/bin/env python3
################################################################################
# Step 3 validation (GPU, gfx950): column-major (TLU=1) FP4 A per-lane GR offset.
#
# A[M,K] is column-major. Each lane issues ONE buffer_load_dwordx4 (16 B = 32 FP4)
# gathering 32 contiguous M elements at a single K column. We fill A so each byte
# encodes its K column (bits 0..6) and the M-group bit (bit 7), load the BASE
# subtile (i=0,j=0) to VGPRs across all 4 waves, and verify:
#   (a) each lane's 16 bytes are identical  -> column-major shape (one column),
#   (b) the (M-group, column) pairs tile {0,1} x [0,128) exactly once each.
################################################################################
import os, sys, tempfile
import pytest
import numpy as np
from gpu_test_helpers import (
    HAS_GFX950,
    GFX_TARGET,
    TileConfig,
    WAVESIZE,
    create_writer,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
)
from Tensile.Components.Subtile.Kernel import AB_B4_TLU1
from Tensile.Components.Subtile.SubtileGREmit import _grComputeColMajorOffsetsA_legacy
from rocisa.code import Module, TextBlock
from rocisa.container import sgpr
from rocisa.instruction import SMovB32, SMovB64, SWaitCnt

LDA = 256  # column-major A leading dim (== M) in elements
NK = 256  # number of K columns
COL_BYTES = LDA // 2  # FP4 (0.5 B/elem): 128 bytes per K column


def fill_colmajor_A():
    """byte p -> low 7 bits = column k=(p//COL_BYTES); bit 7 = M-group (M_base 0 vs 128)."""
    buf = np.zeros(NK * COL_BYTES, dtype=np.uint8)
    for p in range(buf.size):
        k = (p // COL_BYTES) & 0x7F
        mbit = (
            1 if (p % COL_BYTES) >= 64 else 0
        )  # M_base=128 -> bytes 64..79 within the column
        buf[p] = k | (mbit << 7)
    return buf


def _srd_setup():
    m = Module("SRD setup")
    m.add(
        SMovB64(
            dst=sgpr("SrdA+0", 2),
            src=sgpr(4, 2),
            comment="SrdA base = input_A_ptr (s[4:5])",
        )
    )
    m.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF", comment="numRecords = max"))
    m.add(
        SMovB32(dst=sgpr("SrdA+3"), src="0x20000", comment="OOB_SELECT=2 (raw buffer)")
    )
    return m


def _base_subtile_load(writer, ti):
    """One buffer_load_dwordx4 per lane for the BASE subtile (i=0, j=0)."""
    m = Module("base subtile GR->VGPR")
    voff = ti.gr.sharedVgprGROffset[0]
    dst = writer.vgprPool.checkOutAligned(4, 2, preventOverflow=False)
    m.add(
        TextBlock(
            f"  buffer_load_dwordx4 v[{dst}:{dst+3}], v{voff}, "
            f"s[sgprSrdA:sgprSrdA+3], 0 offen offset:0\n"
        )
    )
    m.add(SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1))
    return m, dst


def _export_all_waves(ti, dst):
    """Store each lane's 16 B: wave w, lane Q -> output[w*64*16 + Q*16].
    The address MUST be a consecutive even-aligned VGPR pair v[alo:alo+1].
    Scratch indices are taken above every VGPR already in use (Serial, the data
    regs, and the offset reg) so nothing collides.
    """
    used = {0, dst, dst + 1, dst + 2, dst + 3}
    used.update(ti.gr.sharedVgprGROffset)
    nxt = max(used) + 1
    wave_v, lane_v, woff_v = nxt, nxt + 1, nxt + 2
    nxt += 3
    if nxt % 2:  # align the address pair to an even index
        nxt += 1
    alo, ahi = nxt, nxt + 1
    L = [
        "  // ---- export base subtile (all waves) ----",
        f"  v_lshrrev_b32 v{wave_v}, 6, v0           // waveId",
        f"  v_and_b32     v{lane_v}, 0x3F, v0        // laneId",
        f"  s_mov_b32     s2, {WAVESIZE*16}          // per-wave bytes (1 load)",
        f"  v_mul_lo_u32  v{woff_v}, s2, v{wave_v}   // wave byte offset",
        f"  v_lshlrev_b32 v{alo}, 4, v{lane_v}       // laneId*16",
        f"  v_add_u32     v{alo}, v{woff_v}, v{alo}  // + wave offset",
        f"  v_mov_b32     v{ahi}, s7                 // output_ptr hi",
        f"  v_add_co_u32  v{alo}, vcc, s6, v{alo}    // + output_ptr lo",
        f"  v_addc_co_u32 v{ahi}, vcc, v{ahi}, 0, vcc",
        f"  flat_store_dwordx4 v[{alo}:{ahi}], v[{dst}:{dst+3}]",
        "  s_waitcnt vmcnt(0)",
    ]
    return "\n".join(L)


def _gen_kernel():
    cfg = TileConfig(mt_a=256, mt_b=256, depth_u=256, stride_a=LDA, stride_b=LDA)
    writer, kernel, tiA, _ = create_writer(
        cfg, mi_wave_group=[2, 2], geometry=AB_B4_TLU1, inst_k=128, bpe=1
    )
    init_rocisa()
    writer.sgprPool.checkOut(12)  # s0..s11 (HW regs + pointers)
    writer.sgprs["StrideAL"] = 10  # A unroll (K) stride == lda
    tiA.allocOffsetRegisters(writer, kernel)
    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(
        4, 4, "SrdA", preventOverflow=False
    )
    off = _grComputeColMajorOffsetsA_legacy(kernel, writer, tiA)
    load, dst = _base_subtile_load(writer, tiA)
    prologue = generate_load_params(
        [
            (4, 2, 0x00, "input_A_ptr"),
            (6, 2, 0x08, "output_ptr"),
            (10, 1, 0x10, "strideA_unroll"),
        ]
    )
    inner = "\n".join(
        [
            str(prologue),
            str(_srd_setup()),
            str(off),
            str(load),
            _export_all_waves(tiA, dst),
        ]
    )
    args = (
        ("input_A_ptr", 8, "global_buffer", "u8"),
        ("output_ptr", 8, "global_buffer", "u8"),
        ("strideA_unroll", 4, "by_value", "u32"),
    )
    out_size = 4 * WAVESIZE * 16
    return generate_kernel_asm(inner, writer, args, lds_size=0), out_size


def _verify(out):
    waves_coop, kCoopCols = 2, 64
    seen, errs = set(), 0
    for w in range(4):
        for q in range(WAVESIZE):
            b = out[(w * WAVESIZE + q) * 16 : (w * WAVESIZE + q) * 16 + 16]
            if len(set(b)) != 1:  # 16 identical bytes -> exactly one column
                errs += 1
                continue
            v = b[0]
            col = v & 0x7F
            g = (v >> 7) & 1
            if col != (w // waves_coop) * kCoopCols + q or g != (w % waves_coop):
                errs += 1
            seen.add((g, col))
    if seen != {(g, c) for g in range(2) for c in range(128)}:
        errs += 1  # base subtile must cover {0,1} x [0,128)
    return errs


@pytest.mark.skipif(not HAS_GFX950, reason=f"requires gfx950, found {GFX_TARGET}")
def test_nn_f4_colmajor_gr_offset(tmp_path):
    asm, out_size = _gen_kernel()
    out = assemble_and_run(
        asm,
        tmp_path,
        "nn_f4_gr_offset",
        out_size,
        inputs=(fill_colmajor_A(),),
        scalars=(LDA,),
    )
    assert _verify(out) == 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        P = type("P", (), {"__truediv__": lambda s, n: os.path.join(d, n)})()
        asm, out_size = _gen_kernel()
        out = assemble_and_run(
            asm,
            P,
            "nn_f4_gr_offset",
            out_size,
            inputs=(fill_colmajor_A(),),
            scalars=(LDA,),
        )
        e = _verify(out)
        print("PASS" if e == 0 else f"FAIL ({e} mismatches)")
        sys.exit(1 if e else 0)
