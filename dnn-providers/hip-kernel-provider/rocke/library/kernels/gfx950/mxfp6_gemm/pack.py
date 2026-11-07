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
"""

from __future__ import annotations

from typing import Tuple

from rocke.core.ir import I8, I32, IRBuilder, KernelDef, PtrType

__all__ = ["build_pack_c0c1", "build_pack_scales_tile", "pack_grids"]

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


def build_quant_e2m3(rows: int, k: int, out: str = "c0c1", scale_out: str = "tile", block_rows: int = 256) -> KernelDef:
    """Fused bf16 -> MXFP6 E2M3 quantize (+ pack): ``(x: bf16 bits [rows, K], codes_out: i8*, scales_out: i8*)``.

    One thread per (row, 32-block) = one c0c1 slot: reads the block's 64 bytes, computes the OCP shared exponent
    from the bf16 bits (exact; ``layout.quantize_bf16`` is the reference), RNE-encodes the 32 elements with fp32
    arithmetic (power-of-two scaling, round-to-nearest-even via the 2^23 add), packs 24 bytes and writes them in
    ``out`` layout ("c0c1": slot 16 B + 8 B of the tile image; "canonical": the dense row stream) and the scale byte
    in ``scale_out`` layout ("tile": layout.pack_scales_tile position; "raw": [rows, K/32]). Same bytes as a
    canonical quantizer, only the addresses differ. Grid: rows * K / 32 / THREADS blocks.
    """
    if out not in ("c0c1", "canonical") or scale_out not in ("tile", "raw"):
        raise ValueError("out must be c0c1 / canonical and scale_out tile / raw")
    if rows % block_rows or k % 128 or rows % 128:
        raise ValueError("rows must be a multiple of 256 and K of 128")
    from rocke.core.ir import F32

    R, nk, nb, nq2 = block_rows, k // 128, k // 32, rows // 128
    row_bytes = k * 3 // 4
    b = IRBuilder(f"rocke_mxfp6_quant_e2m3_{out}_{scale_out}_{rows}x{k}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    x = b.param("x", PtrType(I8, "global"))
    co = b.param("codes_out", PtrType(I8, "global"))
    so = b.param("scales_out", PtrType(I8, "global"))
    c = b.const_i32
    gid = b.add(b.mul(b.block_id_x(), c(THREADS)), b.thread_id_x())
    o = b.mod(gid, c(4 * R))
    ts = b.div(gid, c(4 * R))
    s, t = b.mod(ts, c(nk)), b.div(ts, c(nk))
    rb, ln = b.div(o, c(64)), b.mod(o, c(64))
    kg, r16 = b.div(ln, c(16)), b.mod(ln, c(16))
    row = b.add(b.add(b.mul(t, c(R)), b.mul(rb, c(16))), r16)
    blk = b.add(b.mul(s, c(4)), kg)
    rx = b.buffer_rsrc(x, c(rows * k * 2))
    xoff = b.add(b.mul(row, c(k * 2)), b.mul(blk, c(64)))
    words = []
    for j in range(4):
        v = b.buffer_load_vN(rx, b.add(xoff, c(16 * j)), c(0), I32, 4)
        words += [b.vec_extract(v, e) for e in range(4)]
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
    rc = b.buffer_rsrc(co, c(rows * row_bytes))
    if out == "c0c1":
        base = b.mul(b.add(b.mul(t, c(nk)), s), c(R * 96))
        b.buffer_store_vN(rc, b.add(base, b.mul(o, c(16))), c(0), b.vec_pack(out_w[:4], I32))
        b.buffer_store_vN(rc, b.add(base, b.add(c(64 * R), b.mul(o, c(8)))), c(0), b.vec_pack(out_w[4:], I32))
    else:
        coff = b.add(b.mul(row, c(row_bytes)), b.mul(blk, c(24)))
        b.buffer_store_vN(rc, coff, c(0), b.vec_pack(out_w[:2], I32))
        b.buffer_store_vN(rc, b.add(coff, c(8)), c(0), b.vec_pack(out_w[2:4], I32))
        b.buffer_store_vN(rc, b.add(coff, c(16)), c(0), b.vec_pack(out_w[4:], I32))
    rsx = b.buffer_rsrc(so, c(rows * nb))
    if scale_out == "tile":  # layout.pack_scales_tile: dword ((s nq2 + q2) 64 + l) 2 + d, byte i
        q2 = b.div(row, c(128))
        rem = b.mod(row, c(128))
        dd, ii = b.div(rem, c(64)), b.div(b.mod(rem, c(64)), c(16))
        lane = b.add(b.mul(kg, c(16)), r16)
        dw = b.add(b.mul(b.add(b.mul(b.add(b.mul(s, c(nq2)), q2), c(64)), lane), c(2)), dd)
        soff = b.add(b.mul(dw, c(4)), ii)
    else:
        soff = b.add(b.mul(row, c(nb)), blk)
    b.buffer_store_vN(rsx, soff, c(0), b.vec_pack([b.trunc(byte, I8)], I8))
    return b.kernel


def quant_grid(rows: int, k: int) -> int:
    return rows * (k // 32) // THREADS
