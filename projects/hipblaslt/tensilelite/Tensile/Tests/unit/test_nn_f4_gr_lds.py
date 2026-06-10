#!/usr/bin/env python3
################################################################################
# Step 4 validation (GPU, gfx950): column-major (TLU=1) FP4 A global read -> LDS.
#
# Runs the production column-major offset + DTL-load emit for ALL subtiles, dumps
# the entire A-LDS region linearly, and checks every byte equals the HBM byte the
# layout in design-note §1 says should be there.
#
# Run:
#   cd tensilelite && python -m pytest Tensile/Tests/unit/test_nn_f4_gr_lds.py -v -s
#   cd tensilelite/Tensile/Tests/unit && python test_nn_f4_gr_lds.py
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
from Tensile.Components.Subtile.SubtileGREmit import (
    _grComputeColMajorOffsetsA_legacy,
    _grComputeColMajorSoffsetsA_legacy,
    _emitColMajorBufferLoad,
)
from rocisa.code import Module, TextBlock
from rocisa.container import sgpr
from rocisa.instruction import SMovB32, SMovB64, SWaitCnt, SBarrier

LDA, MTA, DU = 256, 256, 256
WAVES_COOP = 2  # MIWaveGroup = [2,2] -> 2 waves cooperate on K
A_BYTES = int(MTA * DU // 2)  # fp4: 0.5 B/elem -> 32768


def fill_A():
    """Column-major HBM A, 1 byte per 2 fp4 elements; mixed value per byte index."""
    idx = np.arange(A_BYTES, dtype=np.uint64)
    return (((idx * np.uint64(2654435761)) >> np.uint64(8)) & np.uint64(0xFF)).astype(
        np.uint8
    )


def expected_lds(ti):
    """Build the expected A-LDS image from the §1 layout + column-major HBM mapping."""
    fill = fill_A()
    lsg0, lsg1 = int(ti.localSubtileGrid[0]), int(ti.localSubtileGrid[1])
    COOP = WAVESIZE * 16
    SUB = int(ti.subtileSize)
    GROUP = lsg0 * lsg1 * SUB
    exp = np.zeros(2 * GROUP, dtype=np.uint8)  # 2 M-groups (MIWaveGroup[0]=2)
    mTile = int(ti.subtileShape[0] * ti.mmaTileShape[0])  # M elems per subtile (=32)
    for g in range(2):
        for c in range(WAVES_COOP):
            base = g * GROUP + c * COOP
            for sId1 in range(lsg1):
                for sId0 in range(lsg0):
                    sub = (sId0 + sId1 * lsg0) * SUB
                    M_start = g * (MTA // 2) + sId0 * mTile
                    for Q in range(WAVESIZE):
                        K = sId1 * 128 + c * WAVESIZE + Q
                        hbm0 = (K * LDA + M_start) // 2
                        for b in range(16):
                            exp[base + sub + Q * 16 + b] = fill[hbm0 + b]
    return exp


def _srd_A():
    m = Module("SRD A")
    m.add(
        SMovB64(
            dst=sgpr("SrdA+0", 2), src=sgpr(4, 2), comment="SrdA base = input_A_ptr"
        )
    )
    m.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF"))
    m.add(SMovB32(dst=sgpr("SrdA+3"), src="0x20000"))
    return m


def _lds_base_asm(ti):
    """Inline LocalWriteBaseAddrA = g*GROUP + c*COOP (uniform per wave)."""
    lsg0, lsg1 = int(ti.localSubtileGrid[0]), int(ti.localSubtileGrid[1])
    COOP = WAVESIZE * 16
    GROUP = lsg0 * lsg1 * int(ti.subtileSize)
    return "\n".join(
        [
            "  // ---- LocalWriteBaseAddrA = g*%d + c*%d ----" % (GROUP, COOP),
            "  v_lshrrev_b32 v60, 6, v0",
            "  v_and_b32     v61, %d, v60   // g" % (WAVES_COOP - 1),
            "  v_lshrrev_b32 v62, %d, v60   // c" % (WAVES_COOP.bit_length() - 1),
            "  s_mov_b32 s2, %d" % GROUP,
            "  v_mul_lo_u32 v61, s2, v61",
            "  s_mov_b32 s2, %d" % COOP,
            "  v_mul_lo_u32 v62, s2, v62",
            "  v_add_u32 v61, v61, v62",
            "  s_nop 0",
            "  v_readfirstlane_b32 s[sgprLocalWriteBaseAddrA], v61",
        ]
    )


def _dump_lds_asm(total_bytes):
    """Linear LDS dump: thread t copies LDS[pass*4096 + t*16 .. +16] -> output[same].
    output_ptr is in s[6:7] for THIS kernel. run_on_gpu packs kernargs as
    [inputs...][output][scalars...]; with ONE input (A) the output pointer is the
    2nd u64 (kernarg 0x08), loaded by the prologue into s[6:7]. (s[8:9] is the
    TWO-input A/B/output convention -> here it is undefined garbage and the
    flat_store faults -> "Aborted (core dumped)".)
    Also: flat_store address is an EVEN-aligned VGPR pair (v[56:57]); the flat
    'offset:' immediate is only 12-bit, so we advance the pointer each pass.
    """
    passes = total_bytes // (256 * 16)  # 256 threads * 16 B = 4096 B/pass
    L = [
        "  // ---- raw LDS dump ----",
        "  v_lshlrev_b32 v50, 4, v0            // v50 = tid*16  (LDS read byte addr)",
        "  v_lshlrev_b32 v54, 4, v0            // v54 = tid*16  (output byte offset)",
        "  v_mov_b32     v57, s7               // out ptr hi = output_ptr hi  (s[6:7])",
        "  v_add_co_u32  v56, vcc, s6, v54     // out ptr lo = output_ptr lo + tid*16",
        "  v_addc_co_u32 v57, vcc, v57, 0, vcc // carry -> hi   (pair v[56:57], EVEN)",
    ]
    for p in range(passes):
        if p > 0:  # step to the next 4096-byte slab
            L += [
                "  v_add_u32     v50, 4096, v50            // LDS read addr += 4096",
                "  v_add_co_u32  v56, vcc, 4096, v56       // out ptr lo += 4096",
                "  v_addc_co_u32 v57, vcc, v57, 0, vcc     // carry -> hi",
            ]
        L += [
            "  ds_read_b128 v[40:43], v50",  # offset 0; addr advanced via v50
            "  s_waitcnt lgkmcnt(0)",
            "  flat_store_dwordx4 v[56:57], v[40:43]",  # no 12-bit offset needed
            "  s_waitcnt vmcnt(0)",
        ]
    return "\n".join(L)


def _gen_kernel():
    cfg = TileConfig(mt_a=MTA, mt_b=MTA, depth_u=DU, stride_a=LDA, stride_b=LDA)
    writer, kernel, tiA, _ = create_writer(
        cfg, mi_wave_group=[2, 2], geometry=AB_B4_TLU1, inst_k=128, bpe=1
    )
    init_rocisa()
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideAL"] = 10
    tiA.allocOffsetRegisters(writer, kernel)
    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(
        4, 4, "SrdA", preventOverflow=False
    )
    writer.sgprs["LocalWriteBaseAddrA"] = writer.sgprPool.checkOut(
        1, "LWB", preventOverflow=False
    )
    off = Module("offsets")
    off.add(_grComputeColMajorOffsetsA_legacy(kernel, writer, tiA))
    _grComputeColMajorSoffsetsA_legacy(writer, off, tiA)
    loads = Module("loads")
    for sId1 in range(int(tiA.localSubtileGrid[1])):
        for sId0 in range(int(tiA.localSubtileGrid[0])):
            loads.add(_emitColMajorBufferLoad(tiA, kernel, sId0, sId1))
    lsg0, lsg1 = int(tiA.localSubtileGrid[0]), int(tiA.localSubtileGrid[1])
    total = 2 * lsg0 * lsg1 * int(tiA.subtileSize)
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
            str(_srd_A()),
            _lds_base_asm(tiA),
            str(off),
            str(loads),
            str(SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1)),
            str(SBarrier()),
            _dump_lds_asm(total),
        ]
    )
    args = (
        ("input_A_ptr", 8, "global_buffer", "u8"),
        ("output_ptr", 8, "global_buffer", "u8"),
        ("strideA_unroll", 4, "by_value", "u32"),
    )
    return generate_kernel_asm(inner, writer, args, lds_size=total), total, tiA


@pytest.mark.skipif(not HAS_GFX950, reason=f"requires gfx950, found {GFX_TARGET}")
def test_nn_f4_colmajor_gr_lds(tmp_path):
    asm, total, tiA = _gen_kernel()
    out = assemble_and_run(
        asm, tmp_path, "nn_f4_gr_lds", total, inputs=(fill_A(),), scalars=(LDA,)
    )
    exp = expected_lds(tiA)
    assert np.array_equal(
        np.frombuffer(out, dtype=np.uint8), exp
    ), f"{int((np.frombuffer(out, np.uint8) != exp).sum())} LDS byte mismatches"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        P = type("P", (), {"__truediv__": lambda s, n: os.path.join(d, n)})()
        asm, total, tiA = _gen_kernel()
        out = np.frombuffer(
            assemble_and_run(
                asm, P, "nn_f4_gr_lds", total, inputs=(fill_A(),), scalars=(LDA,)
            ),
            dtype=np.uint8,
        )
        e = expected_lds(tiA)
        n = int((out != e).sum())
        print("PASS" if n == 0 else f"FAIL ({n} mismatches)")
        sys.exit(1 if n else 0)
