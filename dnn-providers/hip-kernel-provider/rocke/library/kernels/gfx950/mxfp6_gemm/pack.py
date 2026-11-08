# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU packing kernels of the gfx950 MXFP6 GEMM: canonical MXFP6 operands -> the GEMM's pre-tiled device layouts.

* :func:`build_pack_c0c1`: canonical codes (dense little-endian 6-bit stream, ``rows x K``) -> the c0c1 image of
  ``layout.pack_operand_c0c1`` (per 256-row block and 128-K tile: slot ``o = 64 rb + 16 kg + r16`` holds the 24-byte
  chunk of row ``16 rb + r16``, K group ``kg``; its first 16 bytes at ``16 o``, the last 8 at ``64 R + 8 o``).
  One thread per slot: three 8-byte loads, a 16-byte and an 8-byte store (byte-identical to the host packer).
* :func:`build_pack_scales_tile`: raw E8M0 scales (``rows x K/32`` bytes) -> ``layout.pack_scales_tile`` (16x16x128
  atom). One thread per output dword: four byte loads (rows ``128 q2 + 64 d + 16 i + l % 16``, block ``4 s + l / 16``).

Kernel signatures: ``(src: i8*, dst: i8*)``; grids from :func:`pack_grids`. Python engine, any compiler.

v2 (bandwidth-oriented, see the second half of the file):

* :func:`build_pack_operand`: codes AND scales in one launch, a work-group stages ``wg_rows x wg_tiles`` K tiles of
  canonical rows in LDS (coalesced 16-byte loads of whole 128-byte lines), then writes whole c0 / c1 runs.
* :func:`build_quant_mx`: bf16 / f32 activations -> E2M3 / E3M2 codes in c0c1 (or canonical) + tile (or raw)
  scales; the codes from the gfx950 hardware conversion ``v_cvt_scalef32_pk32_*`` (or the software encoder).
* :func:`build_copy`: the device-copy roofline probe. Host references: :func:`quantize_ref`, :func:`pack_ref`.
"""

from __future__ import annotations

from typing import Tuple

from rocke.core.ir import I8, I32, IRBuilder, KernelDef, PtrType

__all__ = [
    "build_copy",
    "build_pack_c0c1",
    "build_pack_operand",
    "build_pack_scales_tile",
    "build_quant_e2m3",
    "build_quant_mx",
    "copy_grid",
    "pack_grids",
    "pack_operand_grid",
    "pack_ref",
    "quant_grid",
    "quant_mx_grid",
    "quantize_ref",
]

THREADS = 256


def pack_grids(rows: int, k: int, block_rows: int = 256) -> Tuple[int, int]:
    """(c0c1 pack blocks, scale pack blocks) of ``THREADS`` threads."""
    slots = (rows // block_rows) * (k // 128) * (4 * block_rows)
    dwords = (rows // 128) * (k // 128) * 128
    return slots // THREADS, dwords // THREADS


def build_pack_c0c1(rows: int, k: int, block_rows: int = 256) -> KernelDef:
    if rows % block_rows or block_rows % 16 or k % 128:
        raise ValueError("rows must be a multiple of block_rows (a multiple of 16), K of 128")
    R, nk, row_bytes = block_rows, k // 128, k * 3 // 4
    b = IRBuilder(f"rocke_mxfp6_pack_c0c1_{rows}x{k}_r{R}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    src = b.param("src", PtrType(I8, "global"))
    dst = b.param("dst", PtrType(I8, "global"))
    c = b.const_i32
    gid = b.add(b.mul(b.block_id_x(), c(THREADS)), b.thread_id_x())
    o = b.mod(gid, c(4 * R))
    ts = b.div(gid, c(4 * R))
    s, t = b.mod(ts, c(nk)), b.div(ts, c(nk))
    rb, ln = b.div(o, c(64)), b.mod(o, c(64))
    kg, r16 = b.div(ln, c(16)), b.mod(ln, c(16))
    row = b.add(b.add(b.mul(t, c(R)), b.mul(rb, c(16))), r16)
    src_off = b.add(b.mul(row, c(row_bytes)), b.add(b.mul(s, c(96)), b.mul(kg, c(24))))
    rs = b.buffer_rsrc(src, c(rows * row_bytes))
    rd = b.buffer_rsrc(dst, c(rows * row_bytes))
    q = [b.buffer_load_vN(rs, b.add(src_off, c(8 * j)), c(0), I32, 2) for j in range(3)]
    base = b.mul(b.add(b.mul(t, c(nk)), s), c(R * 96))
    c0 = b.vec_pack([b.vec_extract(q[0], 0), b.vec_extract(q[0], 1), b.vec_extract(q[1], 0), b.vec_extract(q[1], 1)], I32)
    b.buffer_store_vN(rd, b.add(base, b.mul(o, c(16))), c(0), c0)
    b.buffer_store_vN(rd, b.add(base, b.add(c(64 * R), b.mul(o, c(8)))), c(0), q[2])
    return b.kernel


def build_pack_scales_tile(rows: int, k: int) -> KernelDef:
    if rows % 128 or k % 128:
        raise ValueError("rows must be a multiple of 128 and K of 128")
    nb, nq2 = k // 32, rows // 128
    b = IRBuilder(f"rocke_mxfp6_pack_scales_tile_{rows}x{k}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    src = b.param("src", PtrType(I8, "global"))
    dst = b.param("dst", PtrType(I8, "global"))
    c = b.const_i32
    gid = b.add(b.mul(b.block_id_x(), c(THREADS)), b.thread_id_x())
    d = b.mod(gid, c(2))
    lane = b.mod(b.div(gid, c(2)), c(64))
    q2 = b.mod(b.div(gid, c(128)), c(nq2))
    s = b.div(gid, c(128 * nq2))
    rs = b.buffer_rsrc(src, c(rows * nb))
    rd = b.buffer_rsrc(dst, c(rows * nb))
    col = b.add(b.mul(s, c(4)), b.div(lane, c(16)))
    word = None
    for i in range(4):
        row = b.add(b.add(b.mul(q2, c(128)), b.mul(d, c(64))), b.add(c(16 * i), b.mod(lane, c(16))))
        v = b.buffer_load_vN(rs, b.add(b.mul(row, c(nb)), col), c(0), I8, 1)
        byte = b.zext(b.vec_extract(v, 0), I32)
        part = b.shl(byte, c(8 * i)) if i else byte
        word = part if word is None else b.lor(word, part)
    b.buffer_store_vN(rd, b.mul(gid, c(4)), c(0), b.vec_pack([word], I32))
    return b.kernel


def _quant_block(b, words, F32):
    """32 bf16 elements (16 i32 words, element 2 i in the low half of word i) -> (6 packed code words, E8M0 byte).

    OCP MX quantization (``layout.quantize_bf16`` is the reference): shared exponent from the integer amax of the
    bf16 bits (exact), per-element RNE to E2M3 with fp32 arithmetic (power-of-two scaling, round-to-nearest-even
    via the 2^23 add), saturation at 7.5; a zero block -> byte 0, a NaN / inf block -> byte 0xFF with zero codes."""
    c = b.const_i32
    bits = []  # 32 bf16 bit patterns (i32, low 16 bits)
    for w in words:
        bits.append(b.land(w, c(0xFFFF)))
        bits.append(b.lshr(w, c(16)))
    amax = b.land(bits[0], c(0x7FFF))
    for h in bits[1:]:
        amax = b.smax(amax, b.land(h, c(0x7FFF)))
    e_field = b.lshr(amax, c(7))
    m7 = b.land(amax, c(0x7F))
    msb = c(0)
    for p_ in range(1, 7):
        msb = b.select(b.cmp_ge(m7, c(1 << p_)), c(p_), msb)
    flog2 = b.select(b.cmp_lt(c(0), e_field), b.sub(e_field, c(127)), b.sub(msb, c(133)))
    byte = b.smax(c(0), b.smin(c(254), b.add(flog2, c(127 - 2))))
    zero = b.cmp_eq(amax, c(0))
    bad = b.cmp_eq(e_field, c(255))
    byte = b.select(zero, c(0), byte)
    byte = b.select(bad, c(255), byte)
    se_c = b.select(b.lor(zero, bad), c(0), b.sub(byte, c(127)))
    mult = b.bitcast(b.shl(b.sub(c(127), se_c), c(23)), F32)  # 2^-se_c (normal: se_c in -126..125)
    magic = b.const_f32(8388608.0)  # 2^23: (y + 2^23) - 2^23 = RNE(y) for 0 <= y < 2^22

    def rne_int(y):  # integer RNE of 0 <= y < 2^22 (the add rounds to nearest even)
        return b.sub(b.bitcast(b.fadd(y, magic), I32), c(0x4B000000))

    codes = []
    for h in bits:
        xf = b.bitcast(b.shl(h, c(16)), F32)
        a = b.fabs(b.fmul(xf, mult))
        sub = rne_int(b.fmul(a, b.const_f32(8.0)))  # a < 1: code = RNE(8 a) (8 -> code 8 = 1.0)
        abits = b.bitcast(a, I32)
        tt = b.sub(b.lshr(abits, c(23)), c(127))  # floor(log2 a) for a >= 1
        tt = b.smax(c(0), b.smin(tt, c(3)))
        unit = b.bitcast(b.shl(b.sub(c(127), tt), c(23)), F32)  # 2^-t
        m = rne_int(b.fmul(b.fsub(b.fmul(a, unit), b.const_f32(1.0)), b.const_f32(8.0)))
        nrm = b.add(b.mul(b.add(tt, c(1)), c(8)), m)
        code = b.select(b.cmp_lt(abits, c(0x3F800000)), sub, nrm)
        code = b.smin(code, c(31))  # saturate at 7.5 (also 8 <= a: t clamps to 3, code >= 32)
        code = b.lor(code, b.land(b.lshr(h, c(10)), c(32)))  # sign bit 15 -> code bit 5
        codes.append(b.select(bad, c(0), code))
    out_w = []
    for d in range(6):  # 192 bits: code i at bits [6 i, 6 i + 6)
        w = c(0)
        for i in range(32):
            lo, hi = 6 * i, 6 * i + 6
            if hi <= 32 * d or lo >= 32 * d + 32:
                continue
            sh = lo - 32 * d
            part = b.shl(codes[i], c(sh)) if sh >= 0 else b.lshr(codes[i], c(-sh))
            w = b.lor(w, part)
        out_w.append(w)
    return out_w, byte


def _scale_tile_off(b, row, s, kg, nq2):
    """Byte offset of (row, block 4 s + kg) in ``layout.pack_scales_tile``: dword ((s nq2 + q2) 64 + l) 2 + d,
    byte i, with row = 128 q2 + 64 d + 16 i + r16 and lane l = 16 kg + r16."""
    c = b.const_i32
    q2, rem = b.div(row, c(128)), b.mod(row, c(128))
    dd, ii = b.div(rem, c(64)), b.div(b.mod(rem, c(64)), c(16))
    lane = b.add(b.mul(kg, c(16)), b.mod(row, c(16)))
    dw = b.add(b.mul(b.add(b.mul(b.add(b.mul(s, c(nq2)), q2), c(64)), lane), c(2)), dd)
    return b.add(b.mul(dw, c(4)), ii)


QUANT_STAGES = ("lds", "slot")


def build_quant_e2m3(rows: int, k: int, out: str = "c0c1", scale_out: str = "tile", block_rows: int = 256,
                     stage: str = "lds", wg_rows: int = 256) -> KernelDef:
    """Fused bf16 -> MXFP6 E2M3 quantize (+ pack): ``(x: bf16 bits [rows, K], codes_out: i8*, scales_out: i8*)``.

    Each (row, 32-block) = one c0c1 slot is quantized by :func:`_quant_block`; its 24 bytes go out in ``out`` layout
    ("c0c1": slot 16 B + 8 B of the tile image; "canonical": the dense row stream), its scale byte in ``scale_out``
    layout ("tile": layout.pack_scales_tile position; "raw": [rows, K/32]). Same bytes as a canonical quantizer,
    only the addresses differ. ``stage``:

    * "lds" (default): a work-group owns ``wg_rows`` rows x one 128-K tile; it reads the bf16 tile with fully
      coalesced 16-byte loads (16 lanes per 256-byte row segment) into LDS (row pitch 272 B: conflict-free slot
      reads), then each thread quantizes ``wg_rows / 64`` slots from LDS.
    * "slot": one thread per slot reading its 64 bytes straight from global memory (lanes one row apart).

    Grid: :func:`quant_grid` (same ``stage`` / ``wg_rows``).
    """
    if out not in ("c0c1", "canonical") or scale_out not in ("tile", "raw"):
        raise ValueError("out must be c0c1 / canonical and scale_out tile / raw")
    if stage not in QUANT_STAGES:
        raise ValueError(f"stage must be one of {QUANT_STAGES}")
    if rows % block_rows or k % 128 or rows % 128:
        raise ValueError("rows must be a multiple of 256 and K of 128")
    if stage == "lds" and (wg_rows not in (64, 128, 256) or block_rows % wg_rows):
        raise ValueError("wg_rows must be 64, 128 or 256 and divide block_rows")
    from rocke.core.ir import F32

    R, nk, nb, nq2 = block_rows, k // 128, k // 32, rows // 128
    row_bytes = k * 3 // 4
    sfx = "" if stage == "slot" else f"_lds{wg_rows}"
    b = IRBuilder(f"rocke_mxfp6_quant_e2m3_{out}_{scale_out}_{rows}x{k}{sfx}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    x = b.param("x", PtrType(I8, "global"))
    co = b.param("codes_out", PtrType(I8, "global"))
    so = b.param("scales_out", PtrType(I8, "global"))
    c = b.const_i32
    rx = b.buffer_rsrc(x, c(rows * k * 2))
    rc = b.buffer_rsrc(co, c(rows * row_bytes))
    rsx = b.buffer_rsrc(so, c(rows * nb))

    def store(row, s, kg, out_w, byte, o_in_block):
        """Write one slot's 24 code bytes and its scale byte."""
        t = b.div(row, c(R))
        blk = b.add(b.mul(s, c(4)), kg)
        if out == "c0c1":
            base = b.mul(b.add(b.mul(t, c(nk)), s), c(R * 96))
            b.buffer_store_vN(rc, b.add(base, b.mul(o_in_block, c(16))), c(0), b.vec_pack(out_w[:4], I32))
            b.buffer_store_vN(rc, b.add(base, b.add(c(64 * R), b.mul(o_in_block, c(8)))), c(0),
                              b.vec_pack(out_w[4:], I32))
        else:
            coff = b.add(b.mul(row, c(row_bytes)), b.mul(blk, c(24)))
            b.buffer_store_vN(rc, coff, c(0), b.vec_pack(out_w[:2], I32))
            b.buffer_store_vN(rc, b.add(coff, c(8)), c(0), b.vec_pack(out_w[2:4], I32))
            b.buffer_store_vN(rc, b.add(coff, c(16)), c(0), b.vec_pack(out_w[4:], I32))
        soff = _scale_tile_off(b, row, s, kg, nq2) if scale_out == "tile" else b.add(b.mul(row, c(nb)), blk)
        b.buffer_store_vN(rsx, soff, c(0), b.vec_pack([b.trunc(byte, I8)], I8))

    if stage == "slot":
        gid = b.add(b.mul(b.block_id_x(), c(THREADS)), b.thread_id_x())
        o = b.mod(gid, c(4 * R))
        ts = b.div(gid, c(4 * R))
        s, t = b.mod(ts, c(nk)), b.div(ts, c(nk))
        rb, ln = b.div(o, c(64)), b.mod(o, c(64))
        kg, r16 = b.div(ln, c(16)), b.mod(ln, c(16))
        row = b.add(b.add(b.mul(t, c(R)), b.mul(rb, c(16))), r16)
        xoff = b.add(b.mul(row, c(k * 2)), b.mul(b.add(b.mul(s, c(4)), kg), c(64)))
        words = []
        for j in range(4):
            v = b.buffer_load_vN(rx, b.add(xoff, c(16 * j)), c(0), I32, 4)
            words += [b.vec_extract(v, e) for e in range(4)]
        out_w, byte = _quant_block(b, words, F32)
        store(row, s, kg, out_w, byte, o)
        return b.kernel

    WR, PITCH = wg_rows, 68  # LDS row pitch in dwords (256 B + 16 B pad)
    tid = b.thread_id_x()
    wg = b.block_id_x()
    s, rblk = b.mod(wg, c(nk)), b.div(wg, c(nk))  # consecutive work-groups walk K (row-segment locality)
    row0 = b.mul(rblk, c(WR))
    smem = b.smem_alloc(I32, [WR * PITCH], name_hint="qx")
    for it in range(WR * 16 // THREADS):  # 16 B per thread per step; 16 lanes per 256-byte row segment
        idx = b.add(c(it * THREADS), tid)
        r, ch = b.div(idx, c(16)), b.mod(idx, c(16))
        goff = b.add(b.mul(b.add(row0, r), c(k * 2)), b.add(b.mul(s, c(256)), b.mul(ch, c(16))))
        v = b.buffer_load_vN(rx, goff, c(0), I32, 4)
        b.smem_store_vN(smem, [b.add(b.mul(r, c(PITCH)), b.mul(ch, c(4)))], v, 4)
    b.sync()
    for j in range(WR * 4 // THREADS):
        o = b.add(c(j * THREADS), tid)  # slot within the work-group's rows
        if out == "c0c1":  # lanes r16-adjacent: the c0 / c1 stores are contiguous across lanes
            rb, ln = b.div(o, c(64)), b.mod(o, c(64))
            kg, r16 = b.div(ln, c(16)), b.mod(ln, c(16))
            lrow = b.add(b.mul(rb, c(16)), r16)
        else:  # lanes kg-adjacent: 4 lanes write a row's 96 contiguous bytes
            lrow, kg = b.div(o, c(4)), b.mod(o, c(4))
        words = []
        for q in range(4):
            li = b.add(b.mul(lrow, c(PITCH)), b.add(b.mul(kg, c(16)), c(4 * q)))
            v = b.smem_load_vN(smem, li, dtype=I32, n=4)
            words += [b.vec_extract(v, e) for e in range(4)]
        out_w, byte = _quant_block(b, words, F32)
        row = b.add(row0, lrow)
        o_blk = b.add(b.mul(b.div(b.mod(row, c(R)), c(16)), c(64)), b.add(b.mul(kg, c(16)), b.mod(row, c(16))))
        store(row, s, kg, out_w, byte, o_blk)
    return b.kernel


def quant_grid(rows: int, k: int, stage: str = "lds", wg_rows: int = 256) -> int:
    """Work-groups of :func:`build_quant_e2m3` (``THREADS`` threads each) for the same ``stage`` / ``wg_rows``."""
    if stage == "slot":
        return rows * (k // 32) // THREADS
    return (rows // wg_rows) * (k // 128)


# ---------------------------------------------------------------------------------------------------------------------
# v2 kernels (pack agent): bandwidth-oriented pack / quantize + the copy roofline probe.
#
# All are LLVM-built Python-engine kernels of THREADS threads using global (flat) memory ops, so the streaming (nt)
# hints are available; i32 element indices (operands < 2 GiB).

_FMT_GEOM = {"e2m3": (1, 3, 2), "e3m2": (3, 2, 4)}  # (exponent bias, mantissa bits, OCP emax)
#: gfx950 hardware conversions (``v_cvt_scalef32_*``: 32 values / scale -> 32 6-bit codes in 6 dwords). "fp6" = E2M3,
#: "bf6" = E3M2 in the ISA's names; the f32 form takes the even and the odd 16 values as two operands.
_HW_CVT = {
    ("bf16", "e2m3"): "v_cvt_scalef32_pk32_fp6_bf16 $0, $1, $2",
    ("bf16", "e3m2"): "v_cvt_scalef32_pk32_bf6_bf16 $0, $1, $2",
    ("f32", "e2m3"): "v_cvt_scalef32_2xpk16_fp6_f32 $0, $1, $2, $3",
    ("f32", "e3m2"): "v_cvt_scalef32_2xpk16_bf6_f32 $0, $1, $2, $3",
}
PACK_ORDERS = ("k", "row", "xcd")
PACK_MUTANTS = ("none", "kg", "scale")
QUANT_MUTANTS = ("none", "emax", "nanmask", "word", "scalepos")


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _udiv(b, x, n: int):
    return b.lshr(x, b.const_i32(n.bit_length() - 1)) if _pow2(n) else b.div(x, b.const_i32(n))


def _umod(b, x, n: int):
    return b.land(x, b.const_i32(n - 1)) if _pow2(n) else b.mod(x, b.const_i32(n))


def _hint(nt: bool):
    from rocke.core.ir import TemporalHint

    return TemporalHint.STREAMING if nt else TemporalHint.DEFAULT


def _wg_order(b, order: str, n_wg: int):
    """Logical work-group index. "xcd": the hardware deals work-groups round-robin to the 8 XCDs; remap so each XCD
    walks one contiguous eighth of the logical order (its L2 sees neighbouring blocks). A grid that is not a multiple
    of 8 work-groups keeps the identity walk."""
    g = b.block_id_x()
    if order != "xcd" or n_wg % 8:
        return g
    return b.add(b.mul(_umod(b, g, 8), b.const_i32(n_wg // 8)), _udiv(b, g, 8))


def _guarded(b, total: int, body) -> None:
    """``body(j)`` for j = it * THREADS + tid over ``range(total)`` (a runtime guard on a last partial step)."""
    tid = b.thread_id_x()
    for it in range((total + THREADS - 1) // THREADS):
        j = b.add(b.const_i32(it * THREADS), tid)
        if (it + 1) * THREADS <= total:
            body(j)
        else:
            with b.scf_if(b.cmp_lt(j, b.const_i32(total))):
                body(j)


# ----------------------------------------------------------------------------------------------------- copy roofline


def copy_grid(nbytes: int, unroll: int = 4) -> int:
    return nbytes // (THREADS * 16 * unroll)


def build_copy(nbytes: int, *, unroll: int = 4, nt_load: bool = False, nt_store: bool = False,
               order: str = "k") -> KernelDef:
    """Device copy ``(src, dst)`` of ``nbytes``: each work-group copies ``THREADS * 16 * unroll`` contiguous bytes with
    16-byte loads (all issued first) then 16-byte stores. The copy roofline the pack kernels are judged against.
    ``order="xcd"``: each XCD copies one contiguous eighth (as the packs' "xcd" walk)."""
    per = THREADS * 16 * unroll
    if nbytes % per or order not in ("k", "xcd"):
        raise ValueError(f"nbytes must be a multiple of {per}; order k / xcd")
    sfx = ("_ntl" if nt_load else "") + ("_nts" if nt_store else "") + ("_xcd" if order == "xcd" else "")
    b = IRBuilder(f"rocke_copy_{nbytes}_u{unroll}{sfx}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    src = b.param("src", PtrType(I8, "global"))
    dst = b.param("dst", PtrType(I8, "global"))
    c = b.const_i32
    base = b.add(b.mul(_wg_order(b, order, nbytes // per), c(per // 4)), b.mul(b.thread_id_x(), c(4)))
    idx = [b.add(base, c(j * THREADS * 4)) for j in range(unroll)]
    vals = [b.global_load_vN(src, i, I32, 4, temporal_hint=_hint(nt_load)) for i in idx]
    for i, v in zip(idx, vals):
        b.global_store_vN(dst, i, v, 4, temporal_hint=_hint(nt_store))
    return b.kernel


# ------------------------------------------------------------------------------------- pack (canonical -> c0c1)


def pack_operand_grid(rows: int, k: int, wg_rows: int = 128, wg_tiles: int = 4) -> int:
    return (rows // wg_rows) * (k // 128 // wg_tiles)


def build_pack_operand(rows: int, k: int, *, wg_rows: int = 128, wg_tiles: int = 4, c1_wide: bool = False,
                       order: str = "xcd", nt_load: bool = False, nt_store: bool = True, block_rows: int = 256,
                       scales: bool = True, mutant: str = "none") -> KernelDef:
    """One-launch operand pack: canonical codes (dense 6-bit rows) + raw E8M0 scales -> the c0c1 image
    (``layout.pack_operand_c0c1``, ``block_rows`` = GEMM tile rows) + ``layout.pack_scales_tile`` (16x16x128).

    Signature ``(codes_src, scales_src, codes_dst, scales_dst)``; grid :func:`pack_operand_grid`.

    A work-group owns ``wg_rows`` rows x ``wg_tiles`` 128-K tiles. It reads that row segment
    (``wg_tiles * 96`` bytes per row; 4 tiles = 384 B = three whole 128-byte lines, so no line is shared with a
    neighbouring work-group) with coalesced 16-byte loads into LDS (row pitch ``96 * wg_tiles + 16`` bytes: the
    24-byte slot reads of 16 consecutive rows hit distinct banks), plus the rows' scale dwords; after one barrier
    every lane writes its c0c1 slot (16-byte c0 store, 8-byte c1 store, or with ``c1_wide`` 16-byte c1 stores of slot
    pairs) and the tile-layout scale dwords. A wave's stores cover one contiguous 1 KiB (c0) / 512 B (c1) run.
    ``order``: work-group walk ("k": K tiles fastest, "row": row blocks fastest, "xcd": "k" with each XCD on one
    contiguous eighth). ``nt_load`` / ``nt_store``: streaming (nt) cache hints. ``mutant`` (tests only): "kg" swaps
    two 32-K groups, "scale" picks the wrong scale byte. Defaults = the measured best on MI355X (128 rows x 4 tiles,
    XCD walk, nt stores: 96 % of the best device-copy bandwidth).
    """
    R, WR, T = block_rows, wg_rows, wg_tiles
    if order not in PACK_ORDERS or mutant not in PACK_MUTANTS:
        raise ValueError(f"order must be one of {PACK_ORDERS}, mutant one of {PACK_MUTANTS}")
    if R % 16 or rows % R or k % 128 or rows % 128:
        raise ValueError("rows must be a multiple of block_rows (multiple of 16) and of 128, K of 128")
    if WR % 64 or R % WR or T not in (1, 2, 4, 8) or (k // 128) % T or (WR * T) % 128:
        raise ValueError("wg_rows: multiple of 64 dividing block_rows; wg_tiles in 1/2/4/8 dividing K/128; "
                         "wg_rows * wg_tiles a multiple of 128")
    nk, nq2 = k // 128, rows // 128
    ng, nrb = nk // T, rows // WR
    n_wg = ng * nrb
    row_dw, nbd = k * 3 // 16, k // 128  # dwords per canonical code row / per raw scale row
    PW = (T * 96 + 16) // 4  # LDS row pitch (dwords)
    sfx = (f"_w{WR}t{T}_{order}" + ("_c1w" if c1_wide else "") + ("_ntl" if nt_load else "")
           + ("_nts" if nt_store else "") + ("" if scales else "_nosc") + ("" if mutant == "none" else f"_mut{mutant}"))
    b = IRBuilder(f"rocke_mxfp6_pack_op_{rows}x{k}_r{R}{sfx}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    csrc = b.param("codes_src", PtrType(I8, "global"))
    ssrc = b.param("scales_src", PtrType(I8, "global"))
    cdst = b.param("codes_dst", PtrType(I8, "global"))
    sdst = b.param("scales_dst", PtrType(I8, "global"))
    c = b.const_i32
    g = _wg_order(b, order, n_wg)
    if order == "row":
        rblk, sg = _umod(b, g, nrb), _udiv(b, g, nrb)
    else:
        sg, rblk = _umod(b, g, ng), _udiv(b, g, ng)
    row0 = b.mul(rblk, c(WR))
    smem = b.smem_alloc(I32, [WR * PW], name_hint="pk")
    CH = 6 * T  # 16-byte chunks per row segment

    def load_chunk(j):
        r = _udiv(b, j, CH)
        ch = b.sub(j, b.mul(r, c(CH)))
        gi = b.add(b.add(b.mul(b.add(row0, r), c(row_dw)), b.mul(sg, c(24 * T))), b.mul(ch, c(4)))
        v = b.global_load_vN(csrc, gi, I32, 4, temporal_hint=_hint(nt_load))
        b.smem_store_vN(smem, [b.add(b.mul(r, c(PW)), b.mul(ch, c(4)))], v, 4)

    _guarded(b, WR * CH, load_chunk)
    ssm = None
    if scales:
        ssm = b.smem_alloc(I32, [WR * T], name_hint="pks")

        def load_scale(j):
            r, w = _udiv(b, j, T), _umod(b, j, T)
            v = b.global_load_i32(ssrc, b.add(b.add(b.mul(b.add(row0, r), c(nbd)), b.mul(sg, c(T))), w))
            b.smem_store_vN(ssm, [j], v, 1)

        _guarded(b, WR * T, load_scale)
    b.sync()
    t_blk = _udiv(b, row0, R)
    o_base = b.mul(_umod(b, row0, R), c(4))  # c0c1 slot of this work-group's first row within its row block

    def block_dw(sl):  # dword index of the (row block, K tile) image
        return b.mul(b.add(b.mul(t_blk, c(nk)), b.add(b.mul(sg, c(T)), sl)), c(R * 24))

    def slot_li(o, sl):  # LDS dword of slot o (within the work-group's rows) of local tile sl
        rb_l, kg, r16 = _udiv(b, o, 64), _umod(b, _udiv(b, o, 16), 4), _umod(b, o, 16)
        if mutant == "kg":
            kg = b.xor(kg, c(1))
        lrow = b.add(b.mul(rb_l, c(16)), r16)
        return b.add(b.add(b.mul(lrow, c(PW)), b.mul(sl, c(24))), b.mul(kg, c(6)))

    def store_slot(j):
        sl, o = _udiv(b, j, 4 * WR), _umod(b, j, 4 * WR)
        li = slot_li(o, sl)
        q = [b.smem_load_vN(smem, b.add(li, c(2 * e)), dtype=I32, n=2) for e in range(2 if c1_wide else 3)]
        base = block_dw(sl)
        ob = b.add(o_base, o)
        c0 = b.vec_pack([b.vec_extract(q[0], 0), b.vec_extract(q[0], 1), b.vec_extract(q[1], 0),
                         b.vec_extract(q[1], 1)], I32)
        b.global_store_vN(cdst, b.add(base, b.mul(ob, c(4))), c0, 4, temporal_hint=_hint(nt_store))
        if not c1_wide:
            b.global_store_vN(cdst, b.add(base, b.add(c(16 * R), b.mul(ob, c(2)))), q[2], 2,
                              temporal_hint=_hint(nt_store))

    _guarded(b, 4 * WR * T, store_slot)
    if c1_wide:
        def store_pair(j):
            sl, p = _udiv(b, j, 2 * WR), _umod(b, j, 2 * WR)
            o = b.mul(p, c(2))
            q0 = b.smem_load_vN(smem, b.add(slot_li(o, sl), c(4)), dtype=I32, n=2)
            q1 = b.smem_load_vN(smem, b.add(slot_li(b.add(o, c(1)), sl), c(4)), dtype=I32, n=2)
            v = b.vec_pack([b.vec_extract(q0, 0), b.vec_extract(q0, 1), b.vec_extract(q1, 0), b.vec_extract(q1, 1)],
                           I32)
            b.global_store_vN(cdst, b.add(block_dw(sl), b.add(c(16 * R), b.mul(b.add(o_base, o), c(2)))), v, 4,
                              temporal_hint=_hint(nt_store))

        _guarded(b, 2 * WR * T, store_pair)
    if scales:
        def store_scale(j):  # tile dword ((s nq2 + q2) 64 + l) 2 + d; byte i = scale[64 d + 16 i + l % 16, 4 s + l / 16]
            sl, rem = _udiv(b, j, WR), _umod(b, j, WR)
            if WR >= 128:  # d fastest: the stores of 128 consecutive j are one contiguous 512 B run
                g64 = b.add(b.mul(_udiv(b, rem, 128), c(2)), _umod(b, rem, 2))
                lane = _umod(b, _udiv(b, rem, 2), 64)
            else:
                g64, lane = _udiv(b, rem, 64), _umod(b, rem, 64)
            sel = _udiv(b, lane, 16)
            if mutant == "scale":
                sel = _umod(b, b.add(sel, c(1)), 4)
            word = None
            for i in range(4):
                lr = b.add(b.add(b.mul(g64, c(64)), c(16 * i)), _umod(b, lane, 16))
                v = b.vec_extract(b.smem_load_vN(ssm, b.add(b.mul(lr, c(T)), sl), dtype=I32, n=1), 0)
                byte = b.land(b.lshr(v, b.mul(sel, c(8))), c(0xFF))
                part = b.shl(byte, c(8 * i)) if i else byte
                word = part if word is None else b.lor(word, part)
            row64 = b.add(row0, b.mul(g64, c(64)))
            q2, d = _udiv(b, row64, 128), _umod(b, _udiv(b, row64, 64), 2)
            s = b.add(b.mul(sg, c(T)), sl)
            dw = b.add(b.mul(b.add(b.mul(b.add(b.mul(s, c(nq2)), q2), c(64)), lane), c(2)), d)
            b.global_store_vN(sdst, dw, b.vec_pack([word], I32), 1, temporal_hint=_hint(nt_store))

        _guarded(b, WR * T, store_scale)
    return b.kernel


# --------------------------------------------------------------------------------- quantize (bf16 / f32 -> MXFP6)


def _pk_max_u16(b, x, y):
    return b.inline_asm("v_pk_max_u16 $0, $1, $2", "=v,v,v", [x, y], I32, sideeffect=False)


def _block_amax(b, words, src: str):
    """Integer |x| maximum of one 32-value block (bf16: 16 packed words, f32: 32 words) -> amax bits (i32)."""
    c = b.const_i32
    if src == "bf16":
        m = [b.land(w, c(0x7FFF7FFF)) for w in words]
        while len(m) > 1:
            m = [_pk_max_u16(b, m[i], m[i + 1]) for i in range(0, len(m), 2)]
        return b.smax(b.land(m[0], c(0xFFFF)), b.lshr(m[0], c(16)))
    m = [b.land(w, c(0x7FFFFFFF)) for w in words]
    while len(m) > 1:
        m = [b.smax(m[i], m[i + 1]) for i in range(0, len(m), 2)]  # non-negative: signed max
    return m[0]


def _block_scale(b, amax, src: str, emax: int):
    """OCP shared exponent of a block: (E8M0 byte, zero block, bad block). floor(log2 amax) exact from the exponent
    field (a subnormal amax via the exponent of its mantissa converted to f32), byte clamped to 0..254; zero block
    -> 0; an inf / NaN anywhere -> 0xFF."""
    c = b.const_i32
    mb_src, ebias_sub = (7, 133) if src == "bf16" else (23, 149)
    e_field = b.lshr(amax, c(mb_src))
    mant = b.land(amax, c((1 << mb_src) - 1))
    msb = b.sub(b.lshr(b.bitcast(b.sitofp_f32(mant), I32), c(23)), c(127))
    flog2 = b.select(b.cmp_lt(c(0), e_field), b.sub(e_field, c(127)), b.sub(msb, c(ebias_sub)))
    byte = b.smax(c(0), b.smin(c(254), b.add(flog2, c(127 - emax))))
    zero = b.cmp_eq(amax, c(0))
    bad = b.cmp_eq(e_field, c(255))
    byte = b.select(zero, c(0), byte)
    byte = b.select(bad, c(255), byte)
    return byte, zero, bad


def _codes_sw(b, vals_f32bits, se_mult, fmt: str):
    """Software OCP encode of 32 values (f32 bit patterns, i32) scaled by ``se_mult`` (f32 2^-se, exact) -> 6 packed
    code words. RNE onto the element grid with one f32 add: C = 2^(23 + max(binade, 1 - bias) - mbits) has an ulp
    equal to the grid step of |a|'s binade, so bits(|a| + C) - bits(C) = RNE(|a| / step); code = that + the
    binade's code offset, saturated at 31; sign -> bit 5."""
    from rocke.core.ir import F32

    c = b.const_i32
    bias, mbits, _ = _FMT_GEOM[fmt]
    lo = 128 - bias  # biased exponent of the first normal binade
    codes = []
    for xb in vals_f32bits:
        a = b.fmul(b.bitcast(xb, F32), se_mult)
        abits = b.land(b.bitcast(a, I32), c(0x7FFFFFFF))
        jc = b.smax(b.lshr(abits, c(23)), c(lo))
        cbits = b.shl(b.add(jc, c(23 - mbits)), c(23))
        s = b.fadd(b.bitcast(abits, F32), b.bitcast(cbits, F32))
        code = b.add(b.sub(b.bitcast(s, I32), cbits), b.shl(b.sub(jc, c(lo)), c(mbits)))
        code = b.smin(code, c(31))
        codes.append(b.lor(code, b.land(b.lshr(xb, c(26)), c(32))))
    out = []
    for d in range(6):  # 192 bits: code i at bits [6 i, 6 i + 6)
        w = None
        for i in range(32):
            lo_b, hi_b = 6 * i, 6 * i + 6
            if hi_b <= 32 * d or lo_b >= 32 * d + 32:
                continue
            sh = lo_b - 32 * d
            part = b.shl(codes[i], c(sh)) if sh > 0 else (b.lshr(codes[i], c(-sh)) if sh < 0 else codes[i])
            w = part if w is None else b.lor(w, part)
        out.append(w)
    return out


def quant_mx_grid(rows: int, k: int, wg_rows: int = 64) -> int:
    return (rows // wg_rows) * (k // 128)


def build_quant_mx(rows: int, k: int, *, fmt: str = "e2m3", src: str = "bf16", cvt: str = "hw",
                   out: str = "c0c1", scale_out: str = "tile", wg_rows: int = 64, stage: str = "lds",
                   lane_order: str = "kg", fallback: int = -1, nt_load: bool = True, nt_store: bool = False,
                   block_rows: int = 256, mutant: str = "none") -> KernelDef:
    """Fused OCP MX quantize (+ pack): activations (bf16 bits or f32, ``rows x K`` row-major) -> MXFP6 codes in the
    GEMM's ``out`` layout ("c0c1" or the canonical dense rows) + E8M0 scales (``scale_out`` "tile" / "raw").

    Signature ``(x, codes_out, scales_out)``; grid :func:`quant_mx_grid`. Reference: :func:`quantize_ref`
    (= ``layout.quantize_bf16`` for bf16) followed by the host packers.

    A work-group owns ``wg_rows`` rows x one 128-K tile = ``4 wg_rows`` blocks (c0c1 slots); one lane per block:
    amax (packed u16 max for bf16), OCP scale byte, then the codes -- ``cvt="hw"``: one gfx950
    ``v_cvt_scalef32_pk32_{fp6,bf6}_bf16`` / ``_2xpk16_..._f32`` per block with the scale 2^se as f32 (E8M0 byte << 23;
    measured on MI355X: byte-identical to the OCP reference on every corner, including byte-0 scales -- the f32
    scale +0.0 / subnormal -- subnormal inputs, ties, saturation and -0; ``fallback`` >= 0 routes blocks with a scale
    byte <= ``fallback`` through the software encoder in a divergent branch, default -1 = off); ``cvt="sw"``: the
    software encoder.
    NaN / inf blocks get byte 0xFF and zero codes. ``stage`` "direct": each lane loads its block's 64 / 128 bytes
    (16-byte loads; ``lane_order`` "kg" = the 4 blocks of a row on adjacent lanes, "row" = 16 rows on adjacent
    lanes); "lds": coalesced row-segment loads into LDS first. Scale bytes are gathered in LDS so every scale store
    is a whole dword. ``mutant`` (tests only): "emax" (wrong emax), "nanmask" (no NaN-block masking), "word" (two
    code words swapped), "scalepos" (scale dword of the wrong lane). Defaults = the measured best on MI355X (LDS stage,
    64 rows, nt loads: whole-line loads, read once).
    """
    if fmt not in _FMT_GEOM or src not in ("bf16", "f32") or cvt not in ("hw", "sw"):
        raise ValueError("fmt e2m3/e3m2, src bf16/f32, cvt hw/sw")
    if out not in ("c0c1", "canonical") or scale_out not in ("tile", "raw") or stage not in ("direct", "lds"):
        raise ValueError("out c0c1/canonical, scale_out tile/raw, stage direct/lds")
    if lane_order not in ("kg", "row") or mutant not in QUANT_MUTANTS:
        raise ValueError(f"lane_order kg/row, mutant one of {QUANT_MUTANTS}")
    R, WR = block_rows, wg_rows
    if rows % R or R % WR or WR % 64 or k % 128 or rows % 128 or R % 16:
        raise ValueError("rows: multiple of block_rows and 128; wg_rows: multiple of 64 dividing block_rows; K % 128")
    from rocke.core.ir import F32, VectorType

    bias, mbits, emax = _FMT_GEOM[fmt]
    if mutant == "emax":
        emax += 1
    esz = 2 if src == "bf16" else 4
    nk, nb, nq2 = k // 128, k // 32, rows // 128
    wpb = 16 if src == "bf16" else 32  # i32 words per block
    sfx = (f"_{cvt}_w{WR}_{stage}_{lane_order}" + (f"_fb{fallback}" if cvt == "hw" else "")
           + ("_ntl" if nt_load else "") + ("_nts" if nt_store else "") + ("" if mutant == "none" else f"_mut{mutant}"))
    b = IRBuilder(f"rocke_mxfp6_quant_{fmt}_{src}_{out}_{scale_out}_{rows}x{k}{sfx}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    x = b.param("x", PtrType(I8, "global"))
    co = b.param("codes_out", PtrType(I8, "global"))
    so = b.param("scales_out", PtrType(I8, "global"))
    c = b.const_i32
    wg = b.block_id_x()
    s, rblk = _umod(b, wg, nk), _udiv(b, wg, nk)
    row0 = b.mul(rblk, c(WR))
    xrow_dw = k * esz // 4
    seg_dw = 128 * esz // 4  # dwords of one row's 128-K segment
    PITCH = seg_dw + 4
    xs = None
    if stage == "lds":
        xs = b.smem_alloc(I32, [WR * PITCH], name_hint="qx")
        cpr = seg_dw // 4  # 16-byte chunks per row segment

        def ld(j):
            r, ch = _udiv(b, j, cpr), _umod(b, j, cpr)
            gi = b.add(b.add(b.mul(b.add(row0, r), c(xrow_dw)), b.mul(s, c(seg_dw))), b.mul(ch, c(4)))
            v = b.global_load_vN(x, gi, I32, 4, temporal_hint=_hint(nt_load))
            b.smem_store_vN(xs, [b.add(b.mul(r, c(PITCH)), b.mul(ch, c(4)))], v, 4)

        _guarded(b, WR * cpr, ld)
        b.sync()
    # scale bytes gathered in LDS: tile -> dword (64-row group, lane 16 kg + row % 16), byte (row % 64) / 16;
    # raw -> dword = local row, byte = K group
    ssm = b.smem_alloc(I8, [WR * 4], name_hint="qs")

    def slot_rc(j):
        """(local row, K group) of slot j; a wave's 64 slots = 16 rows x 4 groups (one 16-row block)."""
        grp, ln = _udiv(b, j, 64), _umod(b, j, 64)
        if lane_order == "kg":
            r16, kg = _udiv(b, ln, 4), _umod(b, ln, 4)
        else:
            kg, r16 = _udiv(b, ln, 16), _umod(b, ln, 16)
        return b.add(b.mul(grp, c(16)), r16), kg

    def load_slot(j):
        lrow, kg = slot_rc(j)
        if stage == "lds":
            base = b.add(b.mul(lrow, c(PITCH)), b.mul(kg, c(wpb)))
            vs = [b.smem_load_vN(xs, b.add(base, c(4 * q)), dtype=I32, n=4) for q in range(wpb // 4)]
        else:
            gi = b.add(b.add(b.mul(b.add(row0, lrow), c(xrow_dw)), b.mul(s, c(seg_dw))), b.mul(kg, c(wpb)))
            vs = [b.global_load_vN(x, b.add(gi, c(4 * q)), I32, 4, temporal_hint=_hint(nt_load))
                  for q in range(wpb // 4)]
        return lrow, kg, vs

    def quant_slot(lrow, kg, vs):
        words = [b.vec_extract(v, e) for v in vs for e in range(4)]
        amax = _block_amax(b, words, src)
        byte, zero, bad = _block_scale(b, amax, src, emax)
        se_c = b.select(b.lor(zero, bad), c(0), b.sub(byte, c(127)))
        if src == "bf16":
            vals = []
            for w in words:
                vals += [b.shl(w, c(16)), b.land(w, c(0xFFFF0000 - (1 << 32)))]
        else:
            vals = words

        def sw():
            mult = b.bitcast(b.shl(b.sub(c(127), se_c), c(23)), F32)  # 2^-se_c (se_c in -127..125: normal)
            return _codes_sw(b, vals, mult, fmt)

        def hw():
            scale = b.bitcast(b.shl(byte, c(23)), F32)  # 2^(byte - 127) as f32 (E8M0 -> f32 bits)
            vt = VectorType(I32, 6)
            if src == "bf16":
                r = b.inline_asm(_HW_CVT[(src, fmt)], "=&v,v,v", [b.vec_pack(words, I32), scale], vt,
                                 sideeffect=False)
            else:
                ev = b.vec_pack(words[0::2], I32)
                od = b.vec_pack(words[1::2], I32)
                r = b.inline_asm(_HW_CVT[(src, fmt)], "=&v,v,v,v", [ev, od, scale], vt, sideeffect=False)
            return [b.vec_extract(r, e) for e in range(6)]

        if cvt == "sw":
            ow = sw()
        elif fallback < 0:
            ow = hw()
        else:
            br = b.scf_if_else(b.cmp_le(byte, c(fallback)), result_types=[I32] * 6)
            with br as (then_ctx, else_ctx):
                with then_ctx:
                    b.scf_yield(*sw())
                with else_ctx:
                    b.scf_yield(*hw())
            ow = list(br.results)
        if mutant != "nanmask":
            keep = b.select(bad, c(0), c(-1))
            ow = [b.land(w, keep) for w in ow]
        if mutant == "word":
            ow[1], ow[2] = ow[2], ow[1]
        row = b.add(row0, lrow)
        if out == "c0c1":
            t = _udiv(b, row, R)
            rr = _umod(b, row, R)
            o_blk = b.add(b.mul(_udiv(b, rr, 16), c(64)), b.add(b.mul(kg, c(16)), _umod(b, rr, 16)))
            base = b.mul(b.add(b.mul(t, c(nk)), s), c(R * 24))
            b.global_store_vN(co, b.add(base, b.mul(o_blk, c(4))), b.vec_pack(ow[:4], I32), 4,
                              temporal_hint=_hint(nt_store))
            b.global_store_vN(co, b.add(base, b.add(c(16 * R), b.mul(o_blk, c(2)))), b.vec_pack(ow[4:], I32), 2,
                              temporal_hint=_hint(nt_store))
        else:
            cw = b.add(b.mul(row, c(k * 3 // 16)), b.mul(b.add(b.mul(s, c(4)), kg), c(6)))
            for e in range(3):
                b.global_store_vN(co, b.add(cw, c(2 * e)), b.vec_pack(ow[2 * e:2 * e + 2], I32), 2,
                                  temporal_hint=_hint(nt_store))
        if scale_out == "tile":
            dw = b.add(b.mul(_udiv(b, lrow, 64), c(64)), b.add(b.mul(kg, c(16)), _umod(b, lrow, 16)))
            bi = _umod(b, _udiv(b, lrow, 16), 4)
        else:
            dw, bi = lrow, kg
        b.smem_store_vN(ssm, [b.add(b.mul(dw, c(4)), bi)], b.trunc(byte, I8), 1)

    tid = b.thread_id_x()
    # every slot's loads first (the stores may alias the input as far as LLVM knows: loads after a store would wait)
    loaded = [load_slot(b.add(c(it * THREADS), tid)) for it in range(4 * WR // THREADS)]
    for args in loaded:
        quant_slot(*args)
    b.sync()

    def store_scale(j):
        if scale_out == "tile":
            if WR >= 128:  # d fastest: 128 consecutive j -> one contiguous 512 B run
                g64 = b.add(b.mul(_udiv(b, j, 128), c(2)), _umod(b, j, 2))
                lane = _umod(b, _udiv(b, j, 2), 64)
            else:
                g64, lane = _udiv(b, j, 64), _umod(b, j, 64)
            src_lane = b.xor(lane, c(1)) if mutant == "scalepos" else lane
            li = b.mul(b.add(b.mul(g64, c(64)), src_lane), c(4))
            row64 = b.add(row0, b.mul(g64, c(64)))
            q2, d = _udiv(b, row64, 128), _umod(b, _udiv(b, row64, 64), 2)
            gi = b.add(b.mul(b.add(b.mul(b.add(b.mul(s, c(nq2)), q2), c(64)), lane), c(2)), d)
        else:
            li = b.mul(j, c(4))
            gi = b.add(b.mul(b.add(row0, j), c(nb // 4)), s)
        v4 = b.smem_load_vN(ssm, li, dtype=I8, n=4)
        word = b.vec_bitcast(v4, VectorType(I32, 1))
        b.global_store_vN(so, gi, word, 1, temporal_hint=_hint(nt_store))

    _guarded(b, WR, store_scale)
    return b.kernel


# ------------------------------------------------------------------------------------------- host references


def quantize_ref(x, fmt: str = "e2m3", src: str = "bf16"):
    """Host reference of :func:`build_quant_mx` -> ``layout.Operand`` (canonical codes + raw scales).

    ``src="bf16"``: ``x`` = uint16 bf16 bit patterns, exactly ``layout.quantize_bf16``. ``src="f32"``: ``x`` = float32
    (or its uint32 bits); the same OCP rule with the f32 exponent / 23-bit mantissa: shared exponent
    ``floor(log2 amax) - emax`` (exact, subnormal amax via its leading mantissa bit), byte clamped to 0..254, codes =
    RNE of ``x * 2^-se`` (float32) onto the element grid saturating at the format max, zero block -> byte 0, a block
    with an inf / NaN -> byte 0xFF and zero codes.
    """
    import numpy as np

    from . import layout as L

    if src == "bf16":
        return L.quantize_bf16(x, fmt)
    if src != "f32":
        raise ValueError("src must be bf16 or f32")
    bits = np.ascontiguousarray(x).view(np.uint32) if np.asarray(x).dtype != np.uint32 else np.ascontiguousarray(x)
    rows, k = bits.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    xb = bits.reshape(rows, k // 32, 32)
    amax = (xb & np.uint32(0x7FFFFFFF)).max(axis=-1).astype(np.int64)
    e_field = amax >> 23
    m23 = amax & 0x7FFFFF
    msb = np.frexp(m23.astype(np.float64))[1] - 1  # floor(log2 m23) for m23 > 0 (exact below 2^53)
    flog2 = np.where(e_field > 0, e_field - 127, msb - 149)
    byte = np.clip(flog2 - L.FMT_EMAX[fmt] + 127, 0, 254)
    byte = np.where(amax == 0, 0, byte)
    bad = e_field == 255
    byte = np.where(bad, 255, byte).astype(np.uint8)
    se_c = np.where(bad | (amax == 0), 0, byte.astype(np.int64) - 127)
    xf = xb.view(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        v = (xf * np.ldexp(np.float32(1.0), -se_c)[..., None].astype(np.float32)).astype(np.float32)
    codes = L.fmt_encode(v, fmt)
    codes = np.where(bad[..., None], 0, codes).astype(np.uint8)
    return L.Operand(rows, k, L.pack_codes(codes.reshape(rows, k)), byte, fmt)


def pack_ref(op, block_rows: int = 256):
    """Host reference of :func:`build_pack_operand`: (c0c1 code image, tile scale bytes), both flat uint8."""
    from . import layout as L

    return (L.pack_operand_c0c1(op, block_rows),
            L.pack_scales_tile(op.scales, "16x16x128").view("uint8").reshape(-1))


def fmt_levels(fmt: str):
    """The 32 non-negative magnitudes of ``fmt`` from its definition (bias / mantissa bits), float64, by code."""
    import numpy as np

    bias, mbits, _ = _FMT_GEOM[fmt]
    out = []
    for code in range(32):
        e, m = code >> mbits, code & ((1 << mbits) - 1)
        out.append(m * 2.0 ** (1 - bias - mbits) if e == 0 else 2.0 ** (e - bias) * (1 + m / 2.0 ** mbits))
    return np.array(out)


def edge_activations(rows: int, k: int, *, fmt: str = "e2m3", src: str = "bf16", seed: int = 3):
    """Deterministic activations with every OCP corner in fixed blocks (rows >= 32, K >= 512), the rest Gaussian with
    per-row magnitudes 2^-140 (bf16) / 2^-150 (f32) .. 2^120. Returns uint16 bf16 bits or float32.

    Corners: zero / -0 / NaN / +inf / -inf blocks; RNE midpoints of every level pair (and +-1 ulp) at several
    binades and at both ends of the exponent range; amax at the format max, between max and the next power of two
    (saturation), exactly a power of two; subnormal amax (byte clamps to 0); max finite and min normal everywhere;
    -0 next to subnormals; tiny negatives in a large-amax block (sign of an underflowed code); for f32 values with
    mantissa bits below bf16 precision.
    """
    import numpy as np

    from . import layout as L

    if rows < 32 or k < 512 or k % 32:
        raise ValueError("edge_activations needs rows >= 32 and K >= 512 (multiple of 32)")
    rng = np.random.default_rng(seed)
    lo = -140 if src == "bf16" else -150
    x = rng.standard_normal((rows, k)).astype(np.float32)
    with np.errstate(over="ignore", under="ignore"):
        x = (x * np.ldexp(np.float32(1.0), rng.integers(lo, 121, size=(rows, 1))).astype(np.float32)).astype(
            np.float32)
    lv = fmt_levels(fmt)
    mx = lv[-1]
    mid = (lv[:-1] + lv[1:]) / 2  # 31 midpoints (RNE ties)
    up = np.nextafter(mid.astype(np.float32), np.float32(np.inf)) if src == "f32" else None

    def blk(r, j, vals):
        v = np.zeros(32, np.float32)
        v[: len(vals)] = np.asarray(vals, np.float32)[:32]
        x[r, 32 * j: 32 * j + 32] = v

    def sgn(v):
        v = np.asarray(v, np.float64).copy()
        v[1::2] *= -1
        return v

    inf, nan = np.float32(np.inf), np.float32(np.nan)
    blk(0, 0, [0.0] * 32)
    blk(1, 0, [-0.0] * 32)
    blk(2, 1, [1.0, nan] + [0.5] * 30)
    blk(3, 2, [inf] + [1.0] * 31)
    blk(4, 3, [-inf] + [1.0] * 31)
    # ties at several binades (amax = the format max -> se = 0 relative), signs alternating
    for j, e in enumerate((0, 7, -9, 100, -100, -120)):
        blk(5, j, sgn(np.concatenate([[mx], mid])) * 2.0 ** e)
        blk(6, j, sgn(np.concatenate([[mx], lv[1:-1:1][:31]])) * 2.0 ** e)  # exact levels
    for j, e in enumerate((0, 30, -60)):  # just above / below the midpoints
        if src == "f32":
            blk(7, j, sgn(np.concatenate([[mx], up])) * 2.0 ** e)
            blk(8, j, sgn(np.concatenate([[mx], np.nextafter(mid.astype(np.float32), np.float32(0))])) * 2.0 ** e)
        else:
            bump = mid * (1 + 2.0 ** -6)  # one bf16 ulp scale: still bf16-representable after rounding
            blk(7, j, sgn(np.concatenate([[mx], bump])) * 2.0 ** e)
            blk(8, j, sgn(np.concatenate([[mx], mid * (1 - 2.0 ** -7)])) * 2.0 ** e)
    # amax relative to the format max: = max, in (max, 2^(emax+1)) (saturation), = 2^(emax+1), = 2^emax
    emax = _FMT_GEOM[fmt][2]
    for j, amax in enumerate((mx, (mx + 2.0 ** (emax + 1)) / 2, 2.0 ** (emax + 1), 2.0 ** emax, mx * 0.999)):
        for jj, e in enumerate((0, -126, 120)):
            blk(9, 3 * j + jj, sgn(np.linspace(amax, amax / 64, 32)) * 2.0 ** e)
    # subnormal amax (the byte clamps to 0), min normal, max finite
    if src == "bf16":
        sub = (np.arange(1, 33) * 2.0 ** -133)
        tiny = 2.0 ** -133
    else:
        sub = (np.arange(1, 33) * 2.0 ** -149)
        tiny = 2.0 ** -149
    blk(10, 0, sgn(sub))
    blk(10, 1, [2.0 ** -126] * 32)
    maxfin = float(np.finfo(np.float32).max) if src == "f32" else (2 - 2.0 ** -7) * 2.0 ** 127  # bf16 0x7F7F
    blk(10, 2, sgn([maxfin] * 32))
    blk(10, 3, np.where(np.arange(32) % 2, -0.0, tiny))  # -0 next to a subnormal
    blk(10, 4, sgn(np.arange(32) * 2.0 ** -128))  # subnormal values with a normal-ish amax
    # tiny negatives in a large-amax block: codes 0 with the sign bit (32)
    blk(11, 0, [2.0 ** 20] + [-2.0 ** -130] * 15 + [-1e-3] * 16)
    blk(11, 1, [-(2.0 ** 100)] + [-(2.0 ** -100)] * 31)
    if src == "f32":  # mantissa bits below bf16 precision
        blk(12, 0, sgn(np.float32(1.0) + np.arange(32, dtype=np.float32) * np.float32(2.0 ** -20)) * 4)
        blk(12, 1, sgn(rng.standard_normal(32)) * 2.0 ** 60)
    if src == "bf16":
        return L.f32_to_bf16_bits(x)
    return x


# ------------------------------------------------------------------------- scale pack (raw E8M0 -> tile layout)

SCALE_MUTANTS = ("none", "transpose", "d")


def _scale_tiles_per_wg(k: int) -> int:
    s = 32
    while (k // 128) % s:
        s //= 2
    return s


def pack_scales_grid(rows_a: int, k: int, rows_b: int = 0) -> int:
    return (rows_a // 128 + rows_b // 128) * ((k // 128) // _scale_tiles_per_wg(k))


def _transpose4x4(b, d):
    """4x4 byte transpose of 4 dwords: out[j] byte i = d[i] byte j (8 v_perm_b32)."""
    c = b.const_i32
    lo, hi = c(0x05010400), c(0x07030602)  # [a0 b0 a1 b1] / [a2 b2 a3 b3] of perm(b, a)
    t0, t1 = b.perm_b32(d[1], d[0], lo), b.perm_b32(d[1], d[0], hi)
    u0, u1 = b.perm_b32(d[3], d[2], lo), b.perm_b32(d[3], d[2], hi)
    s0, s1 = c(0x05040100), c(0x07060302)  # [x0 x1 y0 y1] / [x2 x3 y2 y3] of perm(y, x)
    return [b.perm_b32(u0, t0, s0), b.perm_b32(u0, t0, s1), b.perm_b32(u1, t1, s0), b.perm_b32(u1, t1, s1)]


def build_pack_scales(rows_a: int, k: int, rows_b: int = 0, *, atom: str = "16x16x128", order: str = "xcd",
                      nt_load: bool = False, nt_store: bool = True, mutant: str = "none") -> KernelDef:
    """Raw E8M0 scales ``[rows][K/32]`` -> ``layout.pack_scales_tile(scales, atom)`` for operand A and (``rows_b`` >
    0) operand B in ONE launch. Signature ``(sa_src, sb_src, sa_dst, sb_dst)``; grid :func:`pack_scales_grid`
    (work-groups ``< rows_a / 128 * ...`` are A's, the rest B's; ``rows_b = 0``: the B pointers are unused).

    A work-group owns one 128-row chunk (``q2``) x ``S`` = 32 K tiles (fewer when K / 128 is not a multiple of 32):
    coalesced 16-byte loads of the 128 rows x 4 S bytes (one 128-byte line per row at S = 32) into LDS (row pitch
    S + 4 dwords: the dword reads below are conflict-free for 16x16x128), then each lane reads the dword (4 K blocks
    of one tile) of 4 rows (``16 i`` / ``32 i`` apart), transposes the 4x4 bytes in registers (8 ``v_perm_b32``) and
    writes whole 16-byte runs of the tile layout (a wave's stores = 128-byte lines). NaN (0xFF) and every other byte
    are moved, never interpreted. ``order``: "k" (K tiles fastest) or "xcd" (each XCD a contiguous eighth).
    ``mutant`` (tests only): "transpose" (wrong byte selector), "d" (the two dwords of a lane swapped).
    """
    if atom not in ("16x16x128", "32x32x64") or order not in ("k", "xcd") or mutant not in SCALE_MUTANTS:
        raise ValueError(f"atom 16x16x128 / 32x32x64, order k / xcd, mutant one of {SCALE_MUTANTS}")
    if rows_a <= 0 or rows_a % 128 or rows_b % 128 or rows_b < 0 or k % 128:
        raise ValueError("rows must be multiples of 128 (rows_b may be 0), K of 128")
    nk, nbd = k // 128, k // 32
    S = _scale_tiles_per_wg(k)
    ng = nk // S
    nwa, nwb = (rows_a // 128) * ng, (rows_b // 128) * ng
    n_wg = nwa + nwb
    PW = S + 4  # LDS row pitch (dwords); keeps 16-byte stores aligned
    tps = 8 if atom == "16x16x128" else 16  # lanes per K tile
    sfx = (f"_{atom}_{order}" + ("_ntl" if nt_load else "") + ("_nts" if nt_store else "")
           + ("" if mutant == "none" else f"_mut{mutant}"))
    b = IRBuilder(f"rocke_mxfp6_pack_scales_{rows_a}_{rows_b}x{k}{sfx}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    sa = b.param("sa_src", PtrType(I8, "global"))
    sb = b.param("sb_src", PtrType(I8, "global"))
    da = b.param("sa_dst", PtrType(I8, "global"))
    db = b.param("sb_dst", PtrType(I8, "global"))
    c = b.const_i32
    g = _wg_order(b, order, n_wg)
    if nwb:
        is_b = b.cmp_ge(g, c(nwa))
        src, dst = b.select(is_b, sb, sa), b.select(is_b, db, da)
        g = b.select(is_b, b.sub(g, c(nwa)), g)
        nq2 = b.select(is_b, c(rows_b // 128), c(rows_a // 128))
    else:
        src, dst, nq2 = sa, da, c(rows_a // 128)
    sg, q2 = _umod(b, g, ng), _udiv(b, g, ng)
    row0 = b.mul(q2, c(128))
    smem = b.smem_alloc(I32, [128 * PW], name_hint="psc")
    cb = min(16, 4 * S)  # bytes per load
    per_row = 4 * S // cb

    def ld(j):
        r, ch = _udiv(b, j, per_row), _umod(b, j, per_row)
        gi = b.add(b.add(b.mul(b.add(row0, r), c(nbd // 4)), b.mul(sg, c(S))), b.mul(ch, c(cb // 4)))
        li = b.add(b.mul(r, c(PW)), b.mul(ch, c(cb // 4)))
        if cb == 4:
            v = b.global_load_i32(src, gi)
            b.smem_store_vN(smem, [li], v, 1)
        else:
            v = b.global_load_vN(src, gi, I32, cb // 4, temporal_hint=_hint(nt_load))
            b.smem_store_vN(smem, [li], v, cb // 4)

    _guarded(b, 128 * per_row, ld)
    b.sync()

    def rd(row, sl):
        return b.vec_extract(b.smem_load_vN(smem, b.add(b.mul(row, c(PW)), sl), dtype=I32, n=1), 0)

    def tr(d4):
        out = _transpose4x4(b, d4)
        if mutant == "transpose":
            out[1], out[2] = out[2], out[1]
        return out

    def store16(sl, lane0, words):
        s = b.add(b.mul(sg, c(S)), sl)
        dw = b.mul(b.add(b.mul(b.add(b.mul(s, nq2), q2), c(64)), lane0), c(2))
        if mutant == "d":
            words = [words[1], words[0], words[3], words[2]]
        b.global_store_vN(dst, dw, b.vec_pack(words, I32), 4, temporal_hint=_hint(nt_store))

    def unit(j):
        sl, p = _udiv(b, j, tps), _umod(b, j, tps)
        if atom == "16x16x128":  # rows 64 d + 16 i + r16, lanes 16 kg + r16 (r16 = 2 p + e)
            t = {(d, e): tr([rd(b.add(c(64 * d + 16 * i + e), b.mul(p, c(2))), sl) for i in range(4)])
                 for d in range(2) for e in range(2)}
            for kg in range(4):
                store16(sl, b.add(c(16 * kg), b.mul(p, c(2))),
                        [t[(0, 0)][kg], t[(1, 0)][kg], t[(0, 1)][kg], t[(1, 1)][kg]])
        else:  # rows 32 i + r32, lanes 32 h + r32 (r32 = 2 p + e), byte 2 d + h of the tile's dword
            t = {e: tr([rd(b.add(c(32 * i + e), b.mul(p, c(2))), sl) for i in range(4)]) for e in range(2)}
            for h in range(2):
                store16(sl, b.add(c(32 * h), b.mul(p, c(2))), [t[0][h], t[0][2 + h], t[1][h], t[1][2 + h]])

    _guarded(b, S * tps, unit)
    return b.kernel
