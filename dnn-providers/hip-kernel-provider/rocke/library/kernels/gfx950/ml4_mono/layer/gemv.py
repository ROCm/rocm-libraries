# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MFMA GEMV pieces: weight units against bf16 activations staged in LDS.

A *unit* is one issued weight tile of a 16-row group: an FP8 128-k block
(``fp8x2``, two packed 64-k chunks and one fp32 block scale) or two packed
bf16 32-k chunks (``bf16``). ``mma_units`` multiplies each unit against the
activation words at its LDS offset with ``mfma_f32_16x16x32_bf16`` (A = 16
weight rows, B = up to 16 activation columns) and adds ``scale * tile`` into
the wave's accumulator. ``reduce_rows`` then sums the waves' tiles in wave
order. Weight tiles are the packed layout the host prepares: lane ``l`` of a
64-k chunk loads 16 bytes at ``((rg * NKC + kc) * 64 + l) * 16``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from rocke.core.ir import BF16, F32, I32, Value, VectorType

from .ctx import EPS, THREADS, WAVES, Ml4Ctx

SCALE_BM = 128


@dataclass
class Unit:
    fmt: str
    words: list
    coef: Optional[Value]
    b_word: Value


def _log2(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def _lane_b_word(cx: Ml4Ctx, b_word: Value) -> Value:
    b = cx.b
    return b.add(b_word, b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(4)))


def unit_fp8x2(cx: Ml4Ctx, w_rsrc, s_rsrc, rg: Value, kc: Value, nkc: int, k_dim: int,
               b_word: Value, scale_rows: int = SCALE_BM) -> Unit:  # fmt: skip
    """Both 64-k halves (chunks ``kc``, ``kc + 1``) of one 128-k FP8 block of
    row group ``rg``, and its block scale."""
    b = cx.b
    words = []
    for h in range(2):
        tile = b.add(b.mul(rg, cx.i(nkc)), b.add(kc, cx.i(h)))
        idx = b.add(b.mul(tile, cx.i(64)), cx.lane)
        words.append(b.buffer_load_vN(w_rsrc, b.mul(idx, cx.i(16)), cx.i(0), I32, 4))
    srow = b.lshr(b.mul(rg, cx.i(16)), cx.i(_log2(scale_rows)))
    s_idx = b.add(b.mul(srow, cx.i(k_dim // 128)), b.lshr(kc, cx.i(1)))
    s = cx.ld_f32(s_rsrc, s_idx)
    return Unit("fp8x2", words, s, _lane_b_word(cx, b_word))


def unit_fp8(cx: Ml4Ctx, w_rsrc, s_rsrc, rg: Value, kc: Value, nkc: int, k_dim: int,
             b_word: Value, bk: int = 64) -> Unit:  # fmt: skip
    """One packed 64-k FP8 chunk ``kc`` of row group ``rg`` and the scale of
    its (128 x ``bk``) block."""
    b = cx.b
    idx = b.add(b.mul(b.add(b.mul(rg, cx.i(nkc)), kc), cx.i(64)), cx.lane)
    words = [b.buffer_load_vN(w_rsrc, b.mul(idx, cx.i(16)), cx.i(0), I32, 4)]
    srow = b.lshr(b.mul(rg, cx.i(16)), cx.i(_log2(SCALE_BM)))
    col = b.lshr(b.mul(kc, cx.i(64)), cx.i(_log2(bk)))
    s = cx.ld_f32(s_rsrc, b.add(b.mul(srow, cx.i(k_dim // bk)), col))
    return Unit("fp8", words, s, _lane_b_word(cx, b_word))


def unit_bf16(cx: Ml4Ctx, w_rsrc, rg: Value, kc: Value, nkc: int, b_word: Value,
              ln: Optional[Value] = None) -> Unit:  # fmt: skip
    """Two packed bf16 32-k chunks of row group ``rg``, 64-k chunk ``kc``;
    ``ln``: the lane whose weights this lane loads (default its own)."""
    b = cx.b
    ln = cx.lane if ln is None else ln
    words = []
    for sp in range(2):
        tile = b.add(b.mul(b.add(b.mul(rg, cx.i(nkc)), kc), cx.i(2)), cx.i(sp))
        idx = b.add(b.mul(tile, cx.i(64)), ln)
        words.append(b.buffer_load_vN(w_rsrc, b.mul(idx, cx.i(16)), cx.i(0), I32, 4))
    return Unit("bf16", words, None, _lane_b_word(cx, b_word))


def mxfp4_scale_index(cx: Ml4Ctx, row: Value, col: Value, cols: int) -> Value:
    """Byte index of the E8M0 scale of (row, 32-k block col) in the ATOM
    preshuffled scale plane of a ``cols``-block-wide expert matrix."""
    b = cx.b
    r32 = b.lshr(row, cx.i(5))
    a = b.land(b.lshr(row, cx.i(4)), cx.i(1))
    rb = b.land(row, cx.i(15))
    c8 = b.lshr(col, cx.i(3))
    d = b.land(b.lshr(col, cx.i(2)), cx.i(1))
    e = b.land(col, cx.i(3))
    x = b.add(b.mul(r32, cx.i((cols + 7) // 8)), c8)
    x = b.add(b.mul(b.add(b.mul(x, cx.i(4)), e), cx.i(16)), rb)
    return b.add(b.mul(b.add(b.mul(x, cx.i(2)), d), cx.i(2)), a)


def mxfp4_scale_pair(cx: Ml4Ctx, row: Value, col: Value, cols: int) -> tuple[Value, Value]:
    """(dword index, byte offset) holding the scales of ``col`` and ``col + 4``
    (``col`` with ``(col // 4) % 2 == 0``)."""
    b = cx.b
    r32 = b.lshr(row, cx.i(5))
    a = b.land(b.lshr(row, cx.i(4)), cx.i(1))
    rb = b.land(row, cx.i(15))
    c8 = b.lshr(col, cx.i(3))
    e = b.land(col, cx.i(3))
    x = b.add(b.mul(r32, cx.i((cols + 7) // 8)), c8)
    return b.add(b.mul(b.add(b.mul(x, cx.i(4)), e), cx.i(16)), rb), a


def ld_byte(cx: Ml4Ctx, rsrc, idx: Value) -> Value:
    """Byte ``idx`` of ``rsrc`` (zero-extended), from its aligned dword."""
    b = cx.b
    w = cx.ld_i32(rsrc, b.lshr(idx, cx.i(2)))
    return b.land(b.lshr(w, b.mul(b.land(idx, cx.i(3)), cx.i(8))), cx.i(0xFF))


def scale_byte(cx: Ml4Ctx, word: Value, shift: Value, late: bool = False):
    """Byte ``shift / 8`` of a loaded scale word; ``late``: a callable the
    MFMA evaluates, so the extract (which waits on the load, and with it, in
    order, on every load issued before it) moves to where the tile is needed."""
    b = cx.b
    if not late:
        return b.land(b.lshr(word, shift), cx.i(0xFF))
    return lambda: b.land(b.lshr(word, shift), cx.i(0xFF))


def unit_native_mxfp4(cx: Ml4Ctx, w_rsrc, s_rsrc, rg: Value, kc: Value, k_dim: int,
                      b_word: Value, b_scale_word: Value, coef=None,
                      ln: Optional[Value] = None, scale=None,
                      late_scale: bool = False) -> Unit:  # fmt: skip
    """One ATOM 16-byte-per-lane MXFP4 tile (128-k chunk ``kc`` of row group
    ``rg``) for the scaled MFMA against the MXFP8 activation at LDS word
    ``b_word``; ``b_scale_word``: this lane's activation scale word in ``misc``;
    ``scale``: the weight scale byte (or a callable giving it) if the caller
    already loaded it; ``late_scale``: else extract the loaded byte at the MFMA."""
    b = cx.b
    ln = cx.lane if ln is None else ln
    tile = b.add(b.mul(rg, cx.i(k_dim // 64)), b.mul(kc, cx.i(2)))
    idx = b.add(b.mul(tile, cx.i(128)), b.mul(ln, cx.i(4)))
    raw = b.buffer_load_vN(w_rsrc, b.mul(idx, cx.i(4)), cx.i(0), I32, 4)
    if scale is None:
        row = b.add(b.mul(rg, cx.i(16)), b.land(ln, cx.i(15)))
        col = b.add(b.mul(kc, cx.i(4)), b.lshr(ln, cx.i(4)))
        s_idx = mxfp4_scale_index(cx, row, col, k_dim // 32)
        if late_scale:
            w = cx.ld_i32(s_rsrc, b.lshr(s_idx, cx.i(2)))
            scale = scale_byte(cx, w, b.mul(b.land(s_idx, cx.i(3)), cx.i(8)), True)
        else:
            scale = ld_byte(cx, s_rsrc, s_idx)
    return Unit("native_mxfp4", [raw, scale, b_scale_word], coef, b_word)


def lds_mxfp8(cx: Ml4Ctx, chunk_word: Value) -> Value:
    """One scaled-MFMA B operand (``<8 x i32>``) from a row-major 128-value
    MXFP8 tile at LDS word ``chunk_word``."""
    b = cx.b
    q = b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(4))
    halves = [cx.lds_ld4(b.add(b.add(chunk_word, cx.i(cx.xs + 16 * h)), q)) for h in range(2)]
    return b.vec_pack([b.vec_extract(halves[h], e) for h in range(2) for e in range(4)], I32)


def fp8_to_bf16x8(cx: Ml4Ctx, w0: Value, w1: Value) -> Value:
    """Eight E4M3 bytes (two dwords, byte order) as ``<8 x bf16>``, exactly."""
    b = cx.b
    one = cx.f(1.0)
    parts = []
    for w in (w0, w1):
        for hi in (False, True):
            pair = b.cvt_scalef32_pk_bf16_fp8(w, one, hi)
            parts.append(b.bitcast(pair, I32))
    return b.bitcast(b.vec_pack(parts, I32), VectorType(BF16, 8))


def _b_operand(cx: Ml4Ctx, word: Value) -> Value:
    return cx.b.bitcast(cx.lds_ld4(word), VectorType(BF16, 8))


def zero4(cx: Ml4Ctx) -> list[Value]:
    return [cx.f(0.0) for _ in range(4)]


def mma_units(cx: Ml4Ctx, acc: Sequence[Value], units: Sequence[Unit]) -> list[Value]:
    """acc[e] += coef * (W_unit @ X_unit)[e] for every unit, in order."""
    b = cx.b
    acc = list(acc)
    for u in units:
        coef = u.coef() if callable(u.coef) else u.coef
        c = b.vec_pack(zero4(cx), F32)
        if u.fmt in ("fp8", "fp8x2"):
            for sp in range(4 if u.fmt == "fp8x2" else 2):
                wv = u.words[sp // 2] if u.fmt == "fp8x2" else u.words[0]
                ws = sp % 2
                a = fp8_to_bf16x8(cx, b.vec_extract(wv, ws * 2), b.vec_extract(wv, ws * 2 + 1))
                bop = _b_operand(cx, b.add(u.b_word, cx.i(sp * 16)))
                c = b.mfma_f32_16x16x32_bf16(a, bop, c)
        elif u.fmt == "bf16":
            for sp in range(2):
                a = b.bitcast(u.words[sp], VectorType(BF16, 8))
                bop = _b_operand(cx, b.add(u.b_word, cx.i(sp * 16)))
                c = b.mfma_f32_16x16x32_bf16(a, bop, c)
        elif u.fmt == "native_mxfp4":
            raw, w_scale, scale_word = u.words
            if callable(w_scale):
                w_scale = w_scale()
            a_scale = cx.lds_ld(b.add(scale_word, cx.i(cx.misc)))
            c = b.mfma_scale_mx(raw, lds_mxfp8(cx, u.b_word), c, w_scale, a_scale,
                                fmt_a="fp4", fmt_b="fp8")  # fmt: skip
        else:
            raise ValueError(f"unknown unit format {u.fmt!r}")
        for e in range(4):
            ce = b.vec_extract(c, e)
            acc[e] = b.fadd(acc[e], ce if coef is None else b.fmul(ce, coef))
    return acc


def run_units(cx: Ml4Ctx, make_unit: Callable[[int], Unit], cpw: int, batch: int,
              pre: Optional[Sequence[Unit]] = None,
              acc0: Optional[Sequence[Value]] = None) -> list[Value]:  # fmt: skip
    """Software pipelined: batch ``i + 1``'s loads issue before batch ``i``
    computes. ``pre``: the first units, already issued (a batch or fewer; the
    next batch starts after them). ``acc0``: an accumulator to continue
    (default zeros)."""
    acc = zero4(cx) if acc0 is None else list(acc0)
    cur = list(pre) if pre else [make_unit(c) for c in range(min(batch, cpw))]
    starts = [0] + list(range(len(cur), cpw, batch))
    for bi in range(len(starts)):
        nxt = None
        if bi + 1 < len(starts):
            n0 = starts[bi + 1]
            nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
        acc = mma_units(cx, acc, cur)
        cur = nxt
    return acc


def reduce_rows(cx: Ml4Ctx, R: int, acc: Sequence[Value],
                emit: Callable[[Value, Value, Value], None], count: int) -> None:  # fmt: skip
    """Sum the waves' 16 x 16 tiles: ``R`` row groups of ``WAVES // R`` waves
    each; ``emit(row_local, column, value)`` for the first ``count`` columns."""
    b = cx.b
    wpr = WAVES // R
    base = b.mul(b.add(b.mul(cx.wave, cx.i(64)), cx.lane), cx.i(4))
    cx.lds_st4(b.add(base, cx.i(cx.red)), b.bitcast(b.vec_pack(list(acc), F32), VectorType(I32, 4)))
    cx.barrier()
    n_out = R * 16 * count
    rows = _log2(R * 16)
    for i in range((n_out + THREADS - 1) // THREADS):
        t = b.add(cx.tid, cx.i(i * THREADS))
        with b.scf_if(b.cmp_lt(t, cx.i(n_out))):
            rl = b.land(t, cx.i(R * 16 - 1))
            n = b.lshr(t, cx.i(rows))
            r = b.land(rl, cx.i(15))
            lane_of = b.add(n, b.mul(b.lshr(r, cx.i(2)), cx.i(16)))
            e = b.land(r, cx.i(3))
            tot = cx.f(0.0)
            for j in range(wpr):
                ww = b.add(b.mul(b.lshr(rl, cx.i(4)), cx.i(wpr)), cx.i(j))
                word = b.add(b.mul(b.add(b.mul(ww, cx.i(64)), lane_of), cx.i(4)), e)
                tot = b.fadd(tot, cx.lds_ldf(b.add(word, cx.i(cx.red))))
            emit(rl, n, tot)


def emit_out(cx: Ml4Ctx, stride: int) -> Callable[[Value, Value, Value], None]:
    def f(rl: Value, n: Value, v: Value) -> None:
        b = cx.b
        cx.lds_st(b.add(b.add(b.mul(n, cx.i(stride)), rl), cx.i(cx.outs)), v)

    return f


def n_sel(cx: Ml4Ctx, count: int) -> Value:
    """This lane's MFMA B column; columns past ``count`` repeat the last one."""
    b = cx.b
    return b.smin(b.land(cx.lane, cx.i(15)), cx.i(count - 1))


def stage_x_pairs(cx: Ml4Ctx, region: str, n_total: int,
                  src_of: Optional[Callable[[Value], Value]] = None) -> None:  # fmt: skip
    """LDS bf16 X[k] = element ``src_of(k)`` of the packed bf16 mailbox
    ``region`` for ``k < n_total`` (``src_of`` contiguous over aligned groups
    of 4; default identity): one 16-byte poll (two pairs) per 4 elements."""
    b = cx.b
    nq = n_total // 4
    full = nq // THREADS

    def word(i: int) -> Value:
        return b.add(cx.tid, cx.i(i * THREADS))

    def pair(w: Value) -> Value:
        k = b.mul(w, cx.i(4))
        return b.lshr(src_of(k) if src_of is not None else k, cx.i(1))

    vals = cx.poll([(region, pair(word(i)), 2) for i in range(full)])
    for i in range(full):
        for j in range(2):
            cx.lds_st(b.add(b.mul(word(i), cx.i(2)), cx.i(cx.xs + j)), vals[i][j])
    if nq % THREADS:
        w = word(full)
        with b.scf_if(b.cmp_lt(w, cx.i(nq))):
            v = cx.poll([(region, pair(w), 2)])[0]
            for j in range(2):
                cx.lds_st(b.add(b.mul(w, cx.i(2)), cx.i(cx.xs + j)), v[j])


# ---------------------------------------------------------------- RMSNorm
def load_x_rmsnorm(cx: Ml4Ctx, ld4s, n: int, gamma: str, count: int):
    """The gamma loads, then ``ld4s([(s, k)])`` -> (gammas, x values)."""
    b = cx.b
    r_g = cx.rsrc(gamma)
    ks = [b.mul(b.add(cx.tid, cx.i(i * THREADS)), cx.i(4)) for i in range(n // (4 * THREADS))]
    gs = [cx.ld_bf16x4(r_g, k) for k in ks]
    return gs, ld4s([(s, k) for s in range(count) for k in ks])


def stage_x_rmsnorm(cx: Ml4Ctx, loaded, n: int, count: int) -> list[Value]:
    """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for ``count`` rows from
    ``loaded`` = load_x_rmsnorm's (gammas, values); returns the rstds."""
    b = cx.b
    per = n // (4 * THREADS)
    ks = [b.mul(b.add(cx.tid, cx.i(i * THREADS)), cx.i(4)) for i in range(per)]
    gs, vals = loaded
    sss = []
    for s in range(count):
        ss = cx.f(0.0)
        for i in range(per):
            for a in vals[s * per + i]:
                ss = b.fadd(ss, b.fmul(a, a))
        sss.append(ss)
    rstds = [
        b.rsqrt(b.fadd(b.fmul(tot, cx.f(1.0 / n)), cx.f(EPS))) for tot in cx.block_sums(sss)
    ]
    for s in range(count):
        for i in range(per):
            a = vals[s * per + i]
            base = b.lshr(b.add(cx.i(s * n), ks[i]), cx.i(1))
            for j in range(2):
                lo = b.fmul(b.fmul(a[2 * j], rstds[s]), gs[i][2 * j])
                hi = b.fmul(b.fmul(a[2 * j + 1], rstds[s]), gs[i][2 * j + 1])
                cx.lds_st(b.add(base, cx.i(cx.xs + j)), cx.bf16_pair(lo, hi))
    return rstds
