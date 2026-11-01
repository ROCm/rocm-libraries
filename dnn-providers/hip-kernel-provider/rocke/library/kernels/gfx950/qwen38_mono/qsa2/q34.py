# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Q3 (index-block scores) and Q4 (top-512 selection, rank order) of the mono
QSA layer: ATOM ``kernels/stages/qsa_fast.py`` operation for operation
(``QsaKnobs()``), narrow builds (rows 1-4, plain and seq), every context up to
``MAX_CONTEXT`` (262144 tokens).

Q3: waves 0-3 build index query head ``w`` (zero-centred norm, RoPE at row
``L``) in LDS; the blocks the first token's context completes are scored from
their ``index_cmp`` rows (``_cached_scores``: one block per wave up to ``nw``
blocks, else ``q3_batch`` (UC) blocks per wave per batch, f64 products as one
multiply and an fma, the batch's head sums reduce-scattered); the rest, at
most one per wave, from their 4 raw index key rows (``_block_key``); the row
completing a block writes its key to ``index_cmp``. A score: per head the f64
wave sum of the lane products, relu, the four summed, over sqrt(128).

Q4: ``CHUNKS`` chunk tasks select the 512 best of their ``FAST_CHUNK`` blocks
(radix select over (score bits, ``0xFFFF - block``) keys in three rotating LDS
histograms, pass 0 counted into ``COPIES`` copies), the final task the 512
best of the candidates (or of the scores, or all of them up to 512) and ranks
them by a 512-wide bitonic sort, ties to the lower block, into ``sel``, then
``selh``.

Knobs (exact): ``q4_umax`` (``asm``: the mono's ``s_nop 1``-padded DPP max
chain | ``ir``: the same chain as ``update.dpp`` ops the scheduler can pad
less | ``readlane``: the unique hit lane via ballot + ``s_ff1`` + readlane,
no DPP chain), ``q4_scan`` (``dpp``: the mono's DPP inclusive scan | ``lds``),
``q3_batch`` (UC 4 / 8 / 16: the same sums in the same order).
"""

from __future__ import annotations

from rocke.core.ir import F32, F64, I32, I64, VectorType
from rocke.helpers import device_ops as D

from . import layout as QL
from . import lower as X
from .ctx import EPS, R128, SIGN, SQRT128, QCtx

# ------------------------------------------------------------------ helpers


def _f64_of(x: QCtx, w0, w1):
    b = x.b
    return b.vec_extract(b.vec_bitcast(b.vec_pack([w0, w1], I32), VectorType(F64, 1)), 0)


def _words(x: QCtx, v):
    b = x.b
    w = b.vec_bitcast(b.vec_pack([v], F64), VectorType(I32, 2))
    return b.vec_extract(w, 0), b.vec_extract(w, 1)


def _i64_of(x: QCtx, w0, w1):
    b = x.b
    return b.vec_extract(b.vec_bitcast(b.vec_pack([w0, w1], I32), VectorType(I64, 1)), 0)


def _i64_words(x: QCtx, v):
    b = x.b
    w = b.vec_bitcast(b.vec_pack([v], I64), VectorType(I32, 2))
    return b.vec_extract(w, 0), b.vec_extract(w, 1)


def _wave_sum64(x: QCtx, v):
    """``qsa_fast._wave_sum64``."""
    b = x.b
    for off in (32, 16, 8, 4, 2, 1):
        w0, w1 = _words(x, v)
        if off >= 16:
            h0 = x.swap(w0, w0, off)
            h1 = x.swap(w1, w1, off)
            v = b.fadd(_f64_of(x, h0[0], h1[0]), _f64_of(x, h0[1], h1[1]))
        else:
            v = b.fadd(v, _f64_of(x, x.xshfl(w0, off), x.xshfl(w1, off)))
    return v


_DPP_HALF = {8: (0x118, 0x108, 0xC), 4: (0x114, 0x104, 0xA)}


def _scatter_step(x: QCtx, lo, hi, off):
    b = x.b
    out = []
    upper = x.ne(x.land(x.lane, off), 0)
    for a_, b_ in zip(lo, hi):
        wa, wb = _words(x, a_), _words(x, b_)
        if off >= 16:
            pairs = [x.swap(wa[k], wb[k], off) for k in range(2)]
            out.append(b.fadd(_f64_of(x, pairs[0][0], pairs[1][0]), _f64_of(x, pairs[0][1], pairs[1][1])))
            continue
        if off in _DPP_HALF:
            shr, shl, banks = _DPP_HALF[off]
            got = []
            for k in range(2):
                r = X.update_dpp(b, wb[k], wb[k], shr, 0xF, banks)
                got.append(X.update_dpp(b, r, wa[k], shl, 0xF, 0xF ^ banks))
        else:
            send = _words(x, b.select(upper, a_, b_))
            got = [x.xshfl(send[k], off) for k in range(2)]
        out.append(b.fadd(b.select(upper, b_, a_), _f64_of(x, got[0], got[1])))
    return out


def _scatter_sums(x: QCtx, vals):
    b = x.b
    off = 32
    while len(vals) > 1:
        half = len(vals) // 2
        vals = _scatter_step(x, vals[:half], vals[half:], off)
        off //= 2
    v = vals[0]
    while off:
        w0, w1 = _words(x, v)
        v = b.fadd(v, _f64_of(x, x.xshfl(w0, off), x.xshfl(w1, off)))
        off //= 2
    return v


def _norm(x: QCtx, xs, w, r_bits, zero_centered):
    """``qsa.norm``."""
    b = x.b
    sq = [b.fmul(v, v) for v in xs]
    while len(sq) > 1:
        sq = [b.fadd(sq[2 * i], sq[2 * i + 1]) for i in range(len(sq) // 2)]
    stab = b.fadd(b.fmul(x.natural_sum(sq[0]), x.fbits(r_bits)), x.fbits(EPS))
    root = b.sqrt(b.cvt_f32_to_f64(stab))
    inv = b.cvt_f64_to_f32(b.fdiv(b.const_f64(1.0), root))
    if zero_centered:
        w = [b.fadd(x.f(1.0), v) for v in w]
    return [x.bf16r(b.fmul(b.fmul(v, inv), s)) for v, s in zip(xs, w)]


def _rope16(x: QCtx, y, cs, sn, active):
    b = x.b
    low = x.eq(x.land(x.lane, 16), 0)
    neg = x.sel(low, SIGN, 0)
    out = []
    for v, c_, s_ in zip(y, cs, sn):
        vi = b.bitcast(v, I32)
        lo, hi = x.swap(vi, vi, 16)
        partner = b.bitcast(b.xor(b.select(low, hi, lo), neg), F32)
        r = x.bf16r(b.fadd(x.bf16r(b.fmul(v, c_)), x.bf16r(b.fmul(partner, s_))))
        out.append(b.select(active, r, v))
    return out


def _block_key(x: QCtx, raw, wk, kc, ks):
    b = x.b
    pooled = [
        x.bf16r(b.fmul(b.fadd(b.fadd(raw[0][e], raw[1][e]), b.fadd(raw[2][e], raw[3][e])), x.f(0.25)))
        for e in range(2)
    ]
    y = _norm(x, pooled, wk, R128, True)
    return _rope16(x, y, x.unpack([kc]), x.unpack([ks]), x.lt(x.lane, x.lay.rope_words))


def _score(x: QCtx, q, kn):
    b = x.b
    relu = []
    for h in range(x.lay.index_heads):
        part = b.fadd(b.cvt_f32_to_f64(b.fmul(q[2 * h], kn[0])), b.cvt_f32_to_f64(b.fmul(q[2 * h + 1], kn[1])))
        r = x.fmax(b.cvt_f64_to_f32(_wave_sum64(x, part)), x.f(0.0))
        if x.k.mutate == "q3_relu" and h == 3:
            r = b.cvt_f64_to_f32(_wave_sum64(x, part))
        relu.append(r)
    return b.fdiv(b.fadd(b.fadd(relu[0], relu[1]), b.fadd(relu[2], relu[3])), x.fbits(SQRT128))


def _word(x: QCtx, a64, word, rs=None):
    """``qsa.word``: one i32 (vector load)."""
    return x.load(rs if rs is not None else x.rsrc(a64), word, 1)


# ============================================================ Q3


def _cached_scores(x: QCtx, rq, q, a_pages, rs_pg, rs_cmp, first, nc, nw):
    b, lay = x.b, x.lay
    IW = lay.index_words
    UC = x.k.q3_batch
    PB = lay.page_blocks
    SPAN = 64 * PB
    S = 6 - (4 * UC).bit_length() + 1
    assert 4 * UC == 64 >> S
    mb_isc = x.mb("isc", half=True)
    # knob q3_switch: the one-block-per-wave path up to nw / q3_switch blocks (1 = the mono's nw; 0 = never);
    # both paths give the same bits (the f32 products of bf16 values are exact), so the switch is exact (P8)
    sw = x.k.q3_switch
    small = x.le(nc, nw) if sw == 1 else (x.le(nc, x.div(nw, sw)) if sw else x.lt(nc, 0))
    with X.ifr(b, small, []) as (then_, else_, _):
        with then_:
            with b.scf_if(x.lt(first, nc)):
                g = x.page_block(a_pages, first, True)
                kv_ = _word(x, None, x.add(x.mul(g, IW), x.lane), rs_cmp)
                score = _score(x, q, x.unpack([kv_]))
                with b.scf_if(x.eq(x.lane, 0)):
                    x.put(mb_isc, x.mb_ix(first, rq), score)
            X.yield_(b)
        with else_:
            qd = [b.cvt_f32_to_f64(v) for v in q]
            nwuc = x.mul(nw, UC)
            nb = x.mul(b.div(b.sub(x.add(nc, nwuc), x.c(1)), nwuc), UC)
            start = x.mul(first, nb)
            end = x.smin(x.add(start, nb), nc)
            last = x.div(b.sub(end, x.c(1)), PB)
            with b.scf_for(start, end, x.c(SPAN), iv_name="q3s0") as s0:
                pg = x.load(rs_pg, x.smin(x.add(x.div(s0, PB), x.lane), last), 1)
                b.s_waitcnt(vmcnt=0)
                with b.scf_for(s0, x.smin(x.add(s0, SPAN), end), x.c(UC), iv_name="q3b0") as b0:
                    at = x.div(b.sub(b0, s0), PB)
                    pages = [b.readlane(pg, x.add(at, k)) for k in range(UC // PB)]
                    gs = [x.add(x.mul(pages[u // PB], PB), u % PB) for u in range(UC)]
                    kws = [_word(x, None, x.add(x.mul(g, IW), x.lane), rs_cmp) for g in gs]
                    parts = []
                    for u in range(UC):
                        kd = [b.cvt_f32_to_f64(v) for v in x.unpack([kws[u]])]
                        for h in range(lay.index_heads):
                            parts.append(X.fma_f64(b, qd[2 * h + 1], kd[1], b.fmul(qd[2 * h], kd[0])))
                    relu = x.fmax(b.cvt_f64_to_f32(_scatter_sums(x, parts)), x.f(0.0))
                    pair = b.fadd(relu, x.xshfl(relu, 1 << S))
                    score = b.fadd(pair, x.xshfl(pair, 2 << S))
                    blk = x.add(b0, b.lshr(x.lane, x.c(S + 2)))
                    put_here = x.band(x.eq(x.land(x.lane, (4 << S) - 1), 0), x.lt(blk, end))
                    with b.scf_if(put_here):
                        x.put(mb_isc, x.mb_ix(blk, rq), b.fdiv(score, x.fbits(SQRT128)))
            X.yield_(b)


def _q3_mono(x: QCtx, t, rq, tq, per, L, cb):
    b, lay = x.b, x.lay
    lane, wave = x.lane, x.wave
    iqn = lay.lds["iqn"]
    IW, RW = lay.index_words, lay.rope_words
    a_pages = x.qsa_entry(rq, "pages")
    a_ik = x.qsa_entry(rq, "index_key")
    a_cmp = x.qsa_entry(rq, "index_cmp")
    a_cos = x.qsa_entry(rq, "cos")
    a_sin = x.qsa_entry(rq, "sin")
    with b.scf_if(x.band(x.lt(t, lay.tasks["q3"]), x.lt(x.mul(tq, lay.waves), x.smax(cb, 1)))):
        rs_pg, rs_ik, rs_cmp = x.rsrc(a_pages), x.rsrc(a_ik), x.rsrc(a_cmp)
        rs_cos, rs_sin = x.rsrc(a_cos), x.rsrc(a_sin)
        rope_word = x.smin(lane, RW - 1)
        wq = x.unpack([_word(x, x.qsa_entry(0, "iq_norm"), lane)])
        wk = x.unpack([_word(x, x.qsa_entry(0, "ik_norm"), lane)])
        cs = x.unpack([_word(x, None, x.add(x.mul(L, RW), rope_word), rs_cos)])
        sn = x.unpack([_word(x, None, x.add(x.mul(L, RW), rope_word), rs_sin)])
        mb_qp = x.mb("qp")
        x.pre_poll(lambda m: (mb_qp, x.mb_ix(x.mul(m, 32), rq)), lay.q1_index_tasks)
        with b.scf_if(x.lt(wave, lay.index_heads)):
            xv = x.poll([(mb_qp, x.mb_ix(x.add(x.mul(wave, IW), lane), rq), 1)])[0]
            y = _norm(x, x.unpack(xv), wq, R128, True)
            y = _rope16(x, y, cs, sn, x.lt(lane, RW))
            for k in range(2):
                x.st(x.add(x.add(x.mul(wave, lay.index_dim), x.mul(lane, 2)), iqn + k), y[k])
        if x.seq:
            got = x.poll([(mb_qp, x.mb_ix(x.add(lay.p_ik, lane), q), 1) for q in range(x.rows)])
            iks = [g_[0] for g_ in got]
        else:
            ik = x.poll([(mb_qp, x.mb_ix(x.add(lay.p_ik, lane), rq), 1)])[0][0]
        x.barrier()
        q = [x.ldf(x.add(x.mul(lane, 2), iqn + h * lay.index_dim + e)) for h in range(4) for e in range(2)]

        nw = (lay.tasks["q3"] // x.rows) * lay.waves
        first = x.add(x.mul(tq, lay.waves), wave)
        nc = x.div(x.ls[0] if x.seq else L, 4)
        _cached_scores(x, rq, q, a_pages, rs_pg, rs_cmp, first, nc, x.c(nw))
        blk = x.add(nc, first)
        with b.scf_if(x.lt(blk, cb)):
            g = x.page_block(a_pages, blk, True)
            kws = [_word(x, None, x.add(x.mul(x.add(x.mul(g, 4), r), IW), lane), rs_ik) for r in range(4)]
            at = x.add(x.mul(x.mul(blk, 4), RW), rope_word)
            kc, ks = _word(x, None, at, rs_cos), _word(x, None, at, rs_sin)
            raw = []
            for r in range(4):
                row = x.add(x.mul(blk, 4), r)
                kw = kws[r]
                if x.seq:
                    for s_ in range(x.rows):
                        kw = b.select(x.eq(row, x.ls[s_]), iks[s_], kw)
                else:
                    kw = b.select(x.eq(row, L), ik, kw)
                raw.append(x.unpack([kw]))
            if x.k.mutate == "q3_key_row":
                raw = [raw[1], raw[1], raw[2], raw[3]]
            kn = _block_key(x, raw, wk, kc, ks)
            score = _score(x, q, kn)
            with b.scf_if(x.eq(lane, 0)):
                x.put(x.mb("isc", half=True), x.mb_ix(blk, rq), score)
            with b.scf_if(x.eq(x.add(x.mul(blk, 4), 3), L)):
                x.store(rs_cmp, x.add(x.mul(g, IW), lane), [x.bf16_pair(kn[0], kn[1])])
        x.barrier()
        with b.scf_if(x.eq(x.tid, 0)):
            x.put(x.mb("isc", fsc="fisch"), x.mb_ix(tq, rq), x.c(1))


def _q3_key(x: QCtx, t, rq, tq, L, cb):
    """``qsa_dense._q3_key`` (qdense 2 / 3, an under-budget row)."""
    b, lay = x.b, x.lay
    lane, wave = x.lane, x.wave
    IW, RW = lay.index_words, lay.rope_words
    a_pages = x.qsa_entry(rq, "pages")
    a_ik = x.qsa_entry(rq, "index_key")
    a_cmp = x.qsa_entry(rq, "index_cmp")
    a_cos = x.qsa_entry(rq, "cos")
    a_sin = x.qsa_entry(rq, "sin")
    first = x.add(x.mul(tq, lay.waves), wave)
    nc = x.div(x.ls[0] if x.seq else L, 4)
    blk = x.add(nc, first)
    cond = x.band(x.band(x.lt(t, lay.tasks["q3"]), x.lt(blk, cb)), x.eq(x.add(x.mul(blk, 4), 3), L))
    with b.scf_if(cond):
        rs_ik, rs_cmp = x.rsrc(a_ik), x.rsrc(a_cmp)
        rope_word = x.smin(lane, RW - 1)
        wk = x.unpack([_word(x, x.qsa_entry(0, "ik_norm"), lane)])
        g = x.page_block(a_pages, blk, True)
        kws = [_word(x, None, x.add(x.mul(x.add(x.mul(g, 4), r), IW), lane), rs_ik) for r in range(4)]
        at = x.add(x.mul(x.mul(blk, 4), RW), rope_word)
        kc, ks = _word(x, a_cos, at), _word(x, a_sin, at)
        mb_qp = x.mb("qp")
        if x.seq:
            got = x.poll([(mb_qp, x.mb_ix(x.add(lay.p_ik, lane), q), 1) for q in range(x.rows)])
            iks = [g_[0] for g_ in got]
        else:
            ik = x.poll([(mb_qp, x.mb_ix(x.add(lay.p_ik, lane), rq), 1)])[0][0]
        raw = []
        for r in range(4):
            row = x.add(x.mul(blk, 4), r)
            kw = kws[r]
            if x.seq:
                for s_ in range(x.rows):
                    kw = b.select(x.eq(row, x.ls[s_]), iks[s_], kw)
            else:
                kw = b.select(x.eq(row, L), ik, kw)
            raw.append(x.unpack([kw]))
        kn = _block_key(x, raw, wk, kc, ks)
        x.store(rs_cmp, x.add(x.mul(g, IW), lane), [x.bf16_pair(kn[0], kn[1])])


def q3(x: QCtx):
    lay = x.lay
    t = x.a.task
    per = lay.tasks["q3"] // x.rows
    rq, tq = x.request(t, per)
    L = x.row_len(rq)
    cb = x.div(x.add(L, 1), 4)
    if x.k.qdense in (2, 3):
        b = x.b
        under = x.le(cb, lay.sel)
        with b.scf_if(under):
            _q3_key(x, t, rq, tq, L, cb)
        with b.scf_if(b.cmp_eq(b.zext(under, I32), x.c(0))):
            _q3_mono(x, t, rq, tq, per, L, cb)
    else:
        _q3_mono(x, t, rq, tq, per, L, cb)


# ============================================================ Q4


class _Q4:
    def __init__(self, x: QCtx):
        self.x = x
        self.b = x.b
        lay = x.lay
        self.k0 = lay.lds["q4_keys"]
        self.L = QL.q4_lds(lay.sel, lay.threads)
        self.H = QL.H_WORDS
        self.T = lay.threads
        self.SEL = lay.sel
        # knob q4_chunks: chunk tasks per row (the mono's 8 of 8192 blocks); the final task follows them. The top
        # 512 of all blocks are among their chunk's top 512 at any chunking, and the final rank is of the same set:
        # exact. fcand grows to CH x 512 candidates (it ends below the row's scores: checked).
        self.CH = x.k.q4_chunks
        self.FC = lay.max_blocks // self.CH
        # knob q4_thr: two-level above this many blocks (0: the chunk size, the mono's rule); the single-level
        # final then holds up to THR blocks (THR / 512 keys per thread)
        self.THR = x.k.q4_thr or self.FC
        assert self.THR >= self.FC and self.THR % self.T == 0
        assert lay.fsc["fcand"] + self.CH * self.SEL * 2 * lay.pair <= lay.sc["isc"]
        assert x.rows * (self.CH + 1) <= lay.grid - 74

    def lo_of(self, blk):
        """The key's low word of block ``blk``: ``0xFFFF - blk`` (ties to the lower block; mutant ``q4_tie``:
        ``blk``, ties to the higher)."""
        x = self.x
        return blk if x.k.mutate == "q4_tie" else self.b.sub(x.c(0xFFFF), blk)

    def w(self, i):
        """Word ``k0 + i``."""
        return self.x.add(i, self.k0) if not isinstance(i, int) else self.x.c(self.k0 + i)

    def clear(self):
        x = self.x
        x.st(self.w(x.tid), x.c(0))
        for e in range(self.L["H0_CLEAR"] // self.T):
            x.st(self.w(x.add(x.tid, self.L["H0"] + self.T * e)), x.c(0))

    # ---- scan / max
    def scan(self, v):
        """``fx.coop.warp_inclusive_scan(v, ADD, 64)`` (the mono's DPP form)."""
        x, b = self.x, self.b
        if x.k.q4_scan == "lds":
            return self._scan_lds(v)
        for ctrl, rm in ((0x111, 0xF), (0x112, 0xF), (0x114, 0xF), (0x118, 0xF), (0x142, 0xA), (0x143, 0xC)):
            v = b.add(X.update_dpp(b, x.c(0), v, ctrl, rm, 0xF), v)
        return v

    def _scan_lds(self, v):
        """Inclusive scan by ``mbcnt`` of the lane's prefix through ds_bpermute
        steps (Kogge-Stone, 6 steps, no DPP row hazards)."""
        x, b = self.x, self.b
        for off in (1, 2, 4, 8, 16, 32):
            src = b.sub(x.lane, x.c(off))
            got = b.ds_bpermute(b.shl(x.smax(src, 0), x.c(2)), v)
            v = b.add(v, x.sel(x.ge(src, 0), got, 0))
        return v

    def umax(self, v):
        """``device_ops.wave_umax_dpp``."""
        x, b = self.x, self.b
        mode = x.k.q4_umax
        if mode == "asm":
            return D.wave_umax_dpp(b, v)
        if mode == "readlane":
            # exactly one lane holds a nonzero pick (or none: 0)
            m = b.wave_ballot(b.zext(x.ne(v, 0), I32))
            lo = b.trunc(m, I32)
            hi = b.trunc(b.lshr(m, b.const_i64(32)), I32)
            # first set lane: ctz over the 64-bit mask
            lz = x.sel(x.ne(lo, 0), _cttz(x, lo), x.add(_cttz(x, hi), 32))
            got = b.readlane(v, x.uni(x.smin(lz, 63)))
            return x.sel(b.cmp_eq(b.lor(lo, hi), x.c(0)), 0, got)
        # ir: the same DPP max chain as update.dpp ops (bound_ctrl:0 = keep old)
        for ctrl, rm in ((0x111, 0xF), (0x112, 0xF), (0x114, 0xF), (0x118, 0xF), (0x142, 0xA), (0x143, 0xC)):
            d = X.update_dpp(b, v, v, ctrl, rm, 0xF, bound=False)
            v = X.umax_i32(b, v, d)
        return b.readlane(v, x.c(63))

    # ---- radix select
    def digit(self, h, shift, prefix, remain, copies):
        x, b = self.x, self.b
        lane = x.lane
        base = b.sub(x.c(252), x.mul(lane, 4))
        v = x.ld4(x.add(base, h))
        for k in range(1, copies):
            w_ = x.ld4(x.add(base, h + self.L["H0_STRIDE"] * k))
            v = [b.add(p, q) for p, q in zip(v, w_)]
        cnt = [v[3], v[2], v[1], v[0]]
        tot = b.add(b.add(cnt[0], cnt[1]), b.add(cnt[2], cnt[3]))
        above = b.sub(self.scan(tot), tot)
        pick = x.c(0)
        for k in range(4):
            nxt = b.add(above, cnt[k])
            hit = x.band(x.lt(above, remain), x.ge(nxt, remain))
            whole = x.sel(x.eq(nxt, remain), 1 << 24, 0)
            val = b.lor(b.lor(whole, b.shl(b.sub(x.c(255 - k), x.mul(lane, 4)), x.c(16))), b.sub(remain, above))
            pick = b.select(hit, val, pick)
            above = nxt
        got = self.umax(pick)
        newp = b.lor(prefix, b.shl(b.land(_shr(x, got, 16), x.c(255)), x.c(shift)))
        return newp, b.land(got, x.c(0xFFFF)), _shr(x, got, 24)

    def count(self, hi, lo, k, prefix, mask):
        x, b = self.x, self.b
        which, shift = QL.RADIX[k]
        if k == 0:
            h = x.add(x.mul(x.land(x.lane, QL.COPIES - 1), self.L["H0_STRIDE"]), self.k0 + self.L["H0"])
        else:
            h = x.c(self.k0 + self.H * (k % 3))
        for e in range(len(hi)):
            if k == 0:
                m = x.ge(hi[e], 0)
            else:
                m = x.band(x.eq(x.land(hi[e], _i32(mask[0])), prefix[0]), x.eq(x.land(lo[e], _i32(mask[1])), prefix[1]))
            with b.scf_if(m):
                src = (hi[e], lo[e])[which]
                d = x.land(_shr(x, src, shift), 255)
                x.atomic_add(x.add(h, d), 1)
        x.barrier()
        if k + 2 < len(QL.RADIX):
            with b.scf_if(x.lt(x.tid, self.H)):
                x.st(x.add(x.tid, self.k0 + self.H * ((k + 2) % 3)), x.c(0))

    def select(self, hi, lo):
        x, b = self.x, self.b
        p = [x.c(0), x.c(0)]
        mask = [0, 0]
        remain = x.c(self.SEL)
        live = x.c(1)
        for k in range(len(QL.RADIX)):
            which, shift = QL.RADIX[k]
            if k == 0:
                h, copies = self.k0 + self.L["H0"], QL.COPIES
            else:
                h, copies = self.k0 + self.H * (k % 3), 1
            with X.ifr(b, x.ne(live, 0), [I32, I32, I32, I32]) as (then_, else_, res):
                with then_:
                    self.count(hi, lo, k, p, mask)
                    got, rem2, whole = self.digit(h, shift, p[which], remain, copies)
                    np_ = list(p)
                    np_[which] = got
                    X.yield_(b, np_[0], np_[1], rem2, b.xor(whole, x.c(1)))
                with else_:
                    X.yield_(b, p[0], p[1], remain, live)
            p = [res[0], res[1]]
            remain, live = res[2], res[3]
            mask[which] |= 0xFF << shift
        return p

    def slots(self, keep):
        x, b = self.x, self.b
        n = x.c(0)
        for kk in keep:
            n = b.add(n, x.sel(kk, 1, 0))
        incl = self.scan(n)
        with b.scf_if(x.eq(x.lane, 63)):
            x.st(x.add(x.wave, self.k0 + self.L["CNT"]), incl)
        x.barrier()
        before = x.c(0)
        for wv in range(x.lay.waves):
            before = b.add(before, x.sel(x.gt(x.wave, wv), x.ld(x.c(self.k0 + self.L["CNT"] + wv)), 0))
        return b.sub(b.add(before, incl), n)

    @staticmethod
    def kept(x, hi, lo, thr, every=None):
        b = x.b
        if every is not None:
            thr = [x.sel(every, 0, t) for t in thr]
        return [b.lor(x.gt(h, thr[0]), x.band(x.eq(h, thr[0]), x.ge(l_, thr[1]))) for h, l_ in zip(hi, lo)]

    def put_key(self, i, hi, lo, blk):
        x, b = self.x, self.b
        x.st(x.add(x.mul(i, 2), self.k0 + self.L["CK"]), b.lor(b.shl(hi, x.c(16)), lo))
        x.st(x.add(x.mul(i, 2), self.k0 + self.L["CK"] + 1), _shr(x, hi, 16))
        x.st(x.add(i, self.k0 + self.L["CB"]), blk)

    def scores(self, base, n, rq, span=None):
        x, b = self.x, self.b
        T = self.T
        mb_isc = x.mb("isc", half=True)
        hi, lo, blk = [], [], []
        keys = (span or self.FC) // T
        bat = min(QL.Q4_BATCH, keys)
        for e0 in range(0, keys, bat):
            at = [x.smin(x.add(x.tid, T * (e0 + k)), b.sub(n, x.c(1))) for k in range(bat)]
            got = x.poll([(mb_isc, x.mb_ix(x.add(base, a_), rq), 1) for a_ in at])
            for k in range(bat):
                j = x.add(x.tid, T * (e0 + k))
                hi.append(x.sel(x.lt(j, n), x.land(got[k][0], 0x7FFFFFFF), -1))
                lo.append(self.lo_of(x.add(base, j)))
                blk.append(x.add(base, j))
        return hi, lo, blk

    def candidates(self, cb, rq):
        x, b = self.x, self.b
        mb_fc = x.mb("isc", fsc="fcand")
        hi, lo, blk = [], [], []
        for e0 in range(0, self.CH, 4):
            specs, have = [], []
            for e in range(e0, e0 + 4):
                hv = x.lt(x.tid, x.smin(b.sub(cb, x.c(e * self.FC)), self.SEL))
                have.append(hv)
                specs.append((mb_fc, x.mb_ix(x.mul(x.sel(hv, x.add(x.tid, e * self.SEL), x.tid), 2), rq), 2))
            got = x.poll(specs)
            for k in range(4):
                hi.append(b.select(have[k], got[k][0], x.c(-1)))
                blk.append(got[k][1])
                lo.append(self.lo_of(got[k][1]))
        return hi, lo, blk

    def keep(self, hi, lo, blk):
        x, b = self.x, self.b
        thr = self.select(hi, lo)
        keep = self.kept(x, hi, lo, thr)
        slot = self.slots(keep)
        for e in range(len(hi)):
            with b.scf_if(keep[e]):
                self.put_key(slot, hi[e], lo[e], blk[e])
            slot = b.add(slot, x.sel(keep[e], 1, 0))

    def chunk(self, j, cb, busy, per, rq):
        x, b = self.x, self.b
        base = x.mul(j, self.FC)
        n = x.smin(b.sub(cb, base), self.FC)
        self.clear()
        mb_fisch = x.mb("isc", fsc="fisch")
        x.pre_poll(lambda m: (mb_fisch, x.mb_ix(x.smin(m, b.sub(busy, x.c(1))), rq)), per)
        hi, lo, blk = self.scores(base, n, rq)
        thr = self.select(hi, lo)
        keep = self.kept(x, hi, lo, thr, x.le(n, self.SEL))
        slot = self.slots(keep)
        mb_fc = x.mb("isc", fsc="fcand")
        for e in range(len(hi)):
            with b.scf_if(keep[e]):
                x.put2(mb_fc, x.mb_ix(x.mul(x.add(x.mul(j, self.SEL), slot), 2), rq), hi[e], blk[e])
            slot = b.add(slot, x.sel(keep[e], 1, 0))
        x.barrier()
        with b.scf_if(x.eq(x.tid, 0)):
            x.put(x.mb("isc", fsc="fch"), x.mb_ix(j, rq), x.c(1))

    def xchg64(self, key, stride, buf):
        x, b = self.x, self.b
        w0, w1 = _i64_words(x, key)
        if stride >= 64:
            at = x.add(x.mul(x.tid, 2), self.k0 + buf)
            x.st(at, w0)
            x.st(x.add(at, 1), w1)
            x.barrier()
            at2 = x.add(x.mul(b.xor(x.tid, x.c(stride)), 2), self.k0 + buf)
            got = [x.ld(at2), x.ld(x.add(at2, 1))]
        elif stride >= 16:
            upper = x.ne(x.land(x.lane, stride), 0)
            got = []
            for w in (w0, w1):
                lo_, hi_ = x.swap(w, w, stride)
                got.append(b.select(upper, lo_, hi_))
        else:
            got = [x.xshfl(w0, stride), x.xshfl(w1, stride)]
        return _i64_of(x, got[0], got[1])

    def sort(self, key):
        x, b = self.x, self.b
        bufs = (0, self.L["CK"])
        n = 0
        size = 2
        while size <= self.T:
            stride = size // 2
            while stride:
                other = self.xchg64(key, stride, bufs[n % 2])
                if stride >= 64:
                    n += 1
                down = x.land(b.xor(_shr(x, x.tid, stride.bit_length() - 1), _shr(x, x.tid, size.bit_length() - 1)), 1)
                up = x.sel(b.cmp_gt(other, key), 1, 0)
                key = b.select(x.eq(up, b.xor(down, x.c(1))), other, key)
                stride //= 2
            size *= 2
        return key

    def rank(self, n, rq):
        x, b = self.x, self.b
        x.barrier()
        have = x.lt(x.tid, n)
        w0 = x.sel(have, x.ld(x.add(x.mul(x.tid, 2), self.k0 + self.L["CK"])), -1)
        w1 = x.sel(have, x.ld(x.add(x.mul(x.tid, 2), self.k0 + self.L["CK"] + 1)), -1)
        key = self.sort(_i64_of(x, w0, w1))
        mb_sel = x.mb("sel")
        with b.scf_if(have):
            lo = x.land(_i64_words(x, key)[0], 0xFFFF)
            slot = x.tid
            if x.k.mutate == "q4_rank":
                slot = x.sel(x.lt(x.tid, 2), b.xor(x.tid, x.c(1)), x.tid)
            x.put(mb_sel, x.mb_ix(slot, rq), lo if x.k.mutate == "q4_tie" else b.sub(x.c(0xFFFF), lo))
        x.barrier()
        with b.scf_if(x.eq(x.tid, 0)):
            x.put(x.mb("selh"), x.mb_ix(0, rq), x.c(1))

    def final(self, cb, busy, per, nch, rq):
        x, b = self.x, self.b
        self.clear()
        big = x.gt(cb, self.THR)
        with b.scf_if(big):
            mb_fch = x.mb("isc", fsc="fch")
            x.pre_poll(lambda m: (mb_fch, x.mb_ix(x.smin(m, b.sub(nch, x.c(1))), rq)), self.CH)
            hi, lo, blk = self.candidates(cb, rq)
            self.keep(hi, lo, blk)
        with b.scf_if(_not(x, big)):
            mb_fisch = x.mb("isc", fsc="fisch")
            x.pre_poll(lambda m: (mb_fisch, x.mb_ix(x.smin(m, b.sub(busy, x.c(1))), rq)), per)
            over = x.gt(cb, self.SEL)
            with b.scf_if(over):
                hi, lo, blk = self.scores(x.c(0), cb, rq, self.THR)
                self.keep(hi, lo, blk)
            with b.scf_if(_not(x, over)):
                with b.scf_if(x.lt(x.tid, cb)):
                    sv = x.poll([(x.mb("isc", half=True), x.mb_ix(x.tid, rq), 1)])[0][0]
                    self.put_key(x.tid, x.land(sv, 0x7FFFFFFF), self.lo_of(x.tid), x.tid)
        self.rank(x.smin(cb, self.SEL), rq)

    def publish(self, cb, rq):
        """``qsa_dense._publish`` (qdense 1 / 2)."""
        x, b = self.x, self.b
        with b.scf_if(x.lt(x.tid, cb)):
            blk = x.tid
            if x.k.mutate == "qdense_drop":
                blk = x.add(x.tid, x.sel(x.band(x.eq(x.tid, 0), x.gt(cb, 1)), 1, 0))
            x.put(x.mb("sel"), x.mb_ix(x.tid, rq), blk)
        x.barrier()
        with b.scf_if(x.eq(x.tid, 0)):
            x.put(x.mb("selh"), x.mb_ix(0, rq), x.c(1))


def _not(x: QCtx, cond):
    return x.b.cmp_eq(x.b.zext(cond, I32), x.c(0))


def _i32(m: int) -> int:
    return m - (1 << 32) if m >= 1 << 31 else m


def _shr(x: QCtx, v, s: int):
    """FlyDSL ``>>`` on Int32: arithmetic shift."""
    if s == 0:
        return v
    return X.ashr(x.b, v, x.c(s))


def _cttz(x: QCtx, v):
    """Count of trailing zeros of a nonzero i32 (by bit tests; uniform)."""
    b = x.b
    r = x.c(0)
    for s in (16, 8, 4, 2, 1):
        low = x.eq(x.land(v, (1 << s) - 1), 0)
        r = b.add(r, x.sel(low, s, 0))
        v = b.select(low, b.lshr(v, x.c(s)), v)
    return r


def q4(x: QCtx):
    b, lay = x.b, x.lay
    t = x.a.task
    per = lay.tasks["q3"] // x.rows
    Q = _Q4(x)
    CH, FC = Q.CH, Q.FC
    rq, j = x.request(t, CH + 1)
    L = x.row_len(rq)
    with b.scf_if(x.lt(t, x.rows * (CH + 1))):
        cb = x.div(x.add(L, 1), 4)
        busy = x.smax(x.smin(x.div(x.add(cb, lay.waves - 1), lay.waves), per), 1)
        nch = x.div(x.add(cb, FC - 1), FC)

        def mono():
            with b.scf_if(x.lt(j, CH)):
                with b.scf_if(x.band(x.gt(cb, Q.THR), x.lt(j, nch))):
                    Q.chunk(j, cb, busy, per, rq)
            with b.scf_if(x.ge(j, CH)):
                Q.final(cb, busy, per, nch, rq)

        if x.k.qdense == 0:
            mono()
        else:
            under = x.le(cb, lay.sel)
            with b.scf_if(under):
                if x.k.qdense in (1, 2):
                    with b.scf_if(x.eq(j, CH)):
                        Q.publish(cb, rq)
            with b.scf_if(_not(x, under)):
                mono()
