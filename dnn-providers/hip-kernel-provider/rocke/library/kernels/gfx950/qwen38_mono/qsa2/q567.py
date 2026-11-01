# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Q5 (scores), Q6 (softmax), Q7 (attention output + MXFP8 pack) of the mono
QSA layer, ATOM ``kernels/stages/qsa.py`` ``_q5`` / ``q6`` / ``_q7``, operation
for operation (``QsaKnobs()``). Narrow builds: rows 1-4, plain and seq.

The selected list: position ``j < 4 n`` is token ``4 sel[j / 4] + j % 4``
(``n = min(cb, 512)``), then the incomplete block's tokens; ``ak`` positions
in all. Every stage takes the current token's key / value from the mailboxes
(Q2's ``qn`` / Q1's ``qp``), never from the cache rows this launch appends.

``qdense`` (N4a, tolerance class, P13): level 3 makes Q5 / Q7 take block
``j`` for list slot ``j`` of a row whose complete blocks fit the budget
(``cb <= 512``), without polling ``sel`` / ``selh``; rows above the budget run
the code above unchanged.
"""

from __future__ import annotations

from rocke.core.ir import F32, I32

from . import lower as X
from .ctx import NEG_MAX, SCALE_BITS, QCtx


def _cache_row(x: QCtx, g, j, n, ak, tail):
    """``qsa._cache_row``."""
    b = x.b
    row = x.sel(x.lt(j, x.mul(n, 4)), x.add(x.mul(g, 4), x.mod(j, 4)),
                x.add(x.mul(tail, 4), b.sub(j, x.mul(n, 4))))  # fmt: skip
    if x.k.mutate == "q7_tail" and x.stage == "q7":
        row = x.sel(x.lt(j, x.mul(n, 4)), row, b.sub(row, x.c(4)))
    return x.sel(x.lt(j, ak), row, 0)


def _now_rows(x: QCtx, a_pages, L):
    """``qsa._now_rows``: (tail, now) -- now a list in seq builds."""
    tail = x.page_block(a_pages, x.div(L, 4), True)
    if x.seq:
        return tail, [x.add(x.mul(x.page_block(a_pages, x.div(Lq, 4), True), 4), x.mod(Lq, 4)) for Lq in x.ls]
    return tail, x.add(x.mul(tail, 4), x.mod(L, 4))


def _dense(x: QCtx, cb):
    """qdense level 3: this row takes block j for list slot j (``cb <= SEL``)."""
    return x.le(cb, x.lay.sel)


def _not(x: QCtx, cond):
    return x.b.cmp_eq(x.b.zext(cond, I32), x.c(0))


def _blk(x: QCtx, j, cb):
    """``qsa_dense._blk``: block ``j`` (mutant: block 1 at slot 0 when ``cb > 1``)."""
    if x.k.mutate == "qdense_drop":
        return x.add(j, x.sel(x.band(x.eq(j, 0), x.gt(cb, 1)), 1, 0))
    return j


# ======================================================================= Q5


def _q5(x: QCtx, t, turn: int):
    b, lay = x.b, x.lay
    rows = x.rows
    G = lay.group
    HW = lay.head_words
    ST = lay.score_tile
    tid = x.tid
    qs, ks, sl = lay.lds["q5_q"], lay.lds["q5_k"], lay.lds["q5_sel"]
    rq = 0 if rows == 1 else x.mod(x.add(x.div(t, 2), turn), rows)
    L = x.row_len(rq)
    a_pages = x.qsa_entry(rq, "pages")
    a_key = x.qsa_entry(rq, "key")
    with b.scf_if(x.lt(t, lay.tasks["q5"])):
        kv = x.mod(t, 2)
        tile = x.div(t, 2) if rows == 1 else x.div(t, 2 * rows)
        cb, n, ak = x.list_len(L)
        with b.scf_if(x.lt(x.mul(tile, ST), ak)):
            tail, now_row = _now_rows(x, a_pages, L)
            mb_qn = x.mb("qn")
            mb_sel = x.mb("sel")
            dense = _dense(x, cb) if x.k.qdense >= 3 else None

            def hint(m):
                head = x.sel(x.lt(m, G), x.add(x.mul(kv, G), m), x.add(lay.q_heads, kv))
                return mb_qn, x.mb_ix(x.sel(x.le(m, G), x.mul(head, HW), lay.selh_from_qn), rq)

            def hint_seq(m):
                head = x.sel(x.lt(m, G), x.add(x.mul(kv, G), m), x.add(lay.q_heads, kv))
                kq = x.sel(x.lt(m, G), rq, x.smin(x.smax(b.sub(m, x.c(G)), 0), rows - 1))
                return mb_qn, x.sel(x.lt(m, G + rows), x.mb_ix(x.mul(HW, head), kq), x.mb_ix(lay.selh_from_qn, rq))

            def wait(extra):
                if x.seq:
                    x.pre_poll(hint_seq, G + rows + extra)
                else:
                    x.pre_poll(hint, G + 1 + extra)

            first = x.add(x.mul(tile, ST // 4), tid)
            want = x.band(x.lt(tid, ST // 4), x.lt(first, n))

            def from_sel():
                with b.scf_if(want):
                    blk = x.poll([(mb_sel, x.mb_ix(first, rq), 1)])[0][0]
                    x.st(x.add(tid, sl), x.page_block(a_pages, blk))

            pre_k = None
            if x.k.q5_kpre:
                if dense is None:
                    wait(1)
                else:
                    with b.scf_if(dense):
                        wait(0)
                    with b.scf_if(_not(x, dense)):
                        wait(1)
                pre_k = _q5_kpre(x, tile, kv, n, ak, tail, a_pages, a_key, rq, cb, dense)
            elif dense is None:
                wait(1)
                from_sel()
            else:
                # P13 ``_q5d``: an under-budget row waits on its q / k heads
                # only and takes block j for slot j; the others as the mono
                with b.scf_if(dense):
                    wait(0)
                    with b.scf_if(want):
                        x.st(x.add(tid, sl), x.page_block(a_pages, _blk(x, first, cb)))
                with b.scf_if(_not(x, dense)):
                    wait(1)
                    from_sel()
            specs, dst = [], []
            nspec = lay.qn_seq(rows) if x.seq else lay.qn_specs
            for r in range(2):
                s = x.smin(x.add(tid, lay.threads * r), nspec - 1)
                hh = x.div(s, HW // 2)
                w = x.mul(x.mod(s, HW // 2), 2)
                head = x.sel(x.lt(hh, G), x.add(x.mul(kv, G), hh), x.add(lay.q_heads, kv))
                if x.seq:
                    qn_at = x.mb_ix(x.add(x.mul(head, HW), w), x.sel(x.lt(hh, G), rq, b.sub(hh, x.c(G))))
                else:
                    qn_at = x.mb_ix(x.add(x.mul(head, HW), w), rq)
                specs.append((mb_qn, qn_at, 2))
                dst.append(x.add(x.mul(hh, HW), w))
            got = x.poll(specs)
            for r in range(2):
                x.stv(x.add(qs, dst[r]), got[r])
            x.barrier()

            rs_k = x.rsrc(a_key)
            quad = x.mod(tid, 32)
            if x.seq:
                curs = [x.ld4(x.add(x.c(qs + (G + q) * HW), x.mul(quad, 4))) for q in range(rows)]
            else:
                cur = x.ld4(x.add(x.c(qs + G * HW), x.mul(quad, 4)))
            aux = 2 if x.k.kv_nt else 0
            for r in range(2):
                row = x.add(x.div(tid, 32), 16 * r)
                if pre_k is not None:
                    kr, kw = pre_k[r]
                else:
                    g = x.ld(x.add(sl, x.div(row, 4)))
                    kr = _cache_row(x, g, x.add(x.mul(tile, ST), row), n, ak, tail)
                    at = x.add(x.mul(x.add(x.mul(kr, lay.kv_heads), kv), HW), x.mul(quad, 4))
                    kw = x.load_words(rs_k, at, 4, aux)
                if x.seq:
                    w4 = kw
                    for q in range(rows):
                        hit = x.eq(kr, now_row[q])
                        w4 = [b.select(hit, curs[q][e], w4[e]) for e in range(4)]
                else:
                    if x.k.mutate == "q5_cur":
                        w4 = kw
                    else:
                        now = x.eq(kr, now_row)
                        w4 = [b.select(now, cur[e], kw[e]) for e in range(4)]
                x.stv(x.add(x.add(x.mul(row, lay.k_stride), x.mul(quad, 4)), ks), w4)
            x.barrier()

            with b.scf_if(x.lt(tid, G * ST)):
                h = x.div(tid, ST)
                i = x.mod(tid, ST)
                acc = x.f(0.0)
                for d in range(HW // 4):
                    qf = x.bf16x8(x.ld4(x.add(x.mul(h, HW), qs + 4 * d)))
                    kf = x.bf16x8(x.ld4(x.add(x.mul(i, lay.k_stride), ks + 4 * d)))
                    for e in range(8):
                        acc = x.fma(qf[e], kf[e], acc)
                j = x.add(x.mul(tile, ST), i)
                scale = SCALE_BITS ^ (1 if x.k.mutate == "q5_scale" else 0)
                score = b.select(x.lt(j, ak), b.fmul(acc, x.fbits(scale)), x.f(NEG_MAX))
                if x.k.fuse == "tile":
                    _q5_tile_split(x, score, j, ak, h, i, kv, tile, rq)
                else:
                    other = x.xshfl(score, 1)
                    with b.scf_if(x.eq(x.mod(i, 2), 0)):
                        at = x.mb_ix(x.add(x.mul(x.add(x.mul(kv, G), h), lay.ss_row), j), rq)
                        x.put2(x.mb("ss"), at, score, other)
            x.barrier()
            with b.scf_if(x.eq(tid, 0)):
                ix = x.mb_ix(x.add(x.mul(tile, 2), kv), rq) if rows > 1 else t
                x.put(x.mb("ssh"), ix, x.c(1))


def _q5_tile_split(x: QCtx, score, j, ak, h, i, kv, tile, rq):
    """N4b ``fuse=tile`` (tolerance class): this task's tile is a split of every one of its 12 heads. The 32
    lanes holding a head's tile (half a wave) take its max ``m``, ``e = mexp(s - m)`` (0 past ``ak``) and
    ``l`` = the sum of ``e`` (lane butterflies 16, 8, 4, 2, 1: one order); ``e`` goes out where the score went
    (``ss``: f32, or with ``n4b_e=bf16`` as bf16 pairs at half the index), ``(m, l)`` as one pair of pairs in
    ``pp`` at ``2 (head 65 + tile)`` (Q6 idles: ``pp`` is free)."""
    b, lay = x.b, x.lay
    G = lay.group
    m = score
    for off in (16, 8, 4, 2, 1):
        m = x.xred(m, off, x.fmax)
    e = b.select(x.lt(j, ak), x.mexp_fma(b.fsub(score, m)), x.f(0.0))
    lsum = e
    for off in (16, 8, 4, 2, 1):
        lsum = x.xred(lsum, off, b.fadd)
    head = x.add(x.mul(kv, G), h)
    other = x.xshfl(e, 1)
    with b.scf_if(x.eq(x.mod(i, 2), 0)):
        if x.k.n4b_e == "f32":
            x.put2(x.mb("ss"), x.mb_ix(x.add(x.mul(head, lay.ss_row), j), rq), e, other)
        else:
            x.put(x.mb("ss"), x.mb_ix(x.add(x.mul(head, lay.pp_row), x.div(j, 2)), rq), x.bf16_pair(e, other))
    with b.scf_if(x.eq(i, 0)):
        x.put2(x.mb("pp"), x.mb_ix(x.mul(x.add(x.mul(head, lay.score_tiles), tile), 2), rq), m, lsum)


def _q7_pv_tile(x: QCtx, words, slot, qh0, kv, nt, ak, rq, own):
    """N4b ``fuse=tile``: Q7 waits on Q5's tiles (``ssh``), takes each head's tiles' ``(m, l)`` from ``pp``:
    ``M`` = their max, ``L`` = the sum of ``mexp(m_t - M) l`` (one wave per head, fixed order), the tile
    coefficient ``mexp(m_t - M) / L``; then ``e`` from ``ss`` and PV with ``p = e coef_t`` (f32)."""
    b, lay = x.b, x.lay
    tid, lane, wave = x.tid, x.lane, x.wave
    W, ST, SSR, NT = lay.waves, lay.score_tile, lay.ss_row, lay.score_tiles
    pb, part = lay.lds["q7_p"], lay.lds["q7_part"]
    RM = lay.lds["q6_e"]  # Q6's entry: the tiles' m, l, then the coefficients (Q6 idles)
    RL = RM + Q7_HEADS * NT
    RC = RL + Q7_HEADS * NT
    mb_ssh, mb_ss, mb_pp = x.mb("ssh"), x.mb("ss"), x.mb("pp")
    x.pre_poll(lambda m: (mb_ssh, x.mb_ix(x.add(x.mul(x.smin(m, b.sub(nt, x.c(1))), 2), kv), rq)), NT)
    # the tiles' (m, l): thread tid < 4 x 65 holds (head tid / 65, tile tid % 65)
    h4 = x.div(tid, NT)
    tl = x.mod(tid, NT)
    with b.scf_if(x.band(x.lt(tid, Q7_HEADS * NT), x.lt(tl, nt))):
        ml = x.poll([(mb_pp, x.mb_ix(x.mul(x.add(x.mul(x.add(qh0, h4), NT), tl), 2), rq), 2)])[0]
        x.st(x.add(tid, RM), ml[0])
        x.st(x.add(tid, RL), ml[1])
    # e of the 4 heads to LDS as Q7 takes the probabilities (f32: one per pair; bf16: two per pair)
    specs, dst = [], []
    if x.k.n4b_e == "f32":
        nr = -(-Q7_HEADS * SSR // (2 * lay.threads))
        for r in range(nr):
            sp = x.smin(x.add(tid, lay.threads * r), Q7_HEADS * SSR // 2 - 1)
            hh = x.div(sp, SSR // 2)
            w0 = x.mul(x.mod(sp, SSR // 2), 2)
            at = x.add(x.mul(x.add(qh0, hh), SSR), x.smin(w0, b.sub(x.mul(nt, ST), x.c(2))))
            specs.append((mb_ss, x.mb_ix(at, rq), 2))
            dst.append(x.add(x.mul(hh, SSR), w0))
        got = x.poll(specs)
        for r in range(len(dst)):
            x.stv(x.add(dst[r], pb), [b.bitcast(got[r][0], F32), b.bitcast(got[r][1], F32)])
    else:
        PPR = lay.pp_row
        for r in range(-(-Q7_HEADS * PPR // (2 * lay.threads))):
            sp = x.smin(x.add(tid, lay.threads * r), Q7_HEADS * PPR // 2 - 1)
            hh = x.div(sp, PPR // 2)
            w0 = x.mul(x.mod(sp, PPR // 2), 2)
            at = x.add(x.mul(x.add(qh0, hh), PPR), x.smin(w0, b.sub(x.mul(nt, ST // 2), x.c(2))))
            specs.append((mb_ss, x.mb_ix(at, rq), 2))
            dst.append(x.add(x.mul(hh, SSR), x.mul(w0, 2)))
        got = x.poll(specs)
        for r in range(len(dst)):
            x.stv(x.add(dst[r], pb), x.unpack(got[r]))
    x.barrier()
    # the merge: wave h (< 4) for head h
    with b.scf_if(x.lt(wave, Q7_HEADS)):
        t1 = x.add(lane, 64)
        hb = x.mul(wave, NT)
        m0 = b.select(x.lt(lane, nt), x.ldf(x.add(hb, x.add(lane, RM))), x.f(NEG_MAX))
        m1 = b.select(x.lt(t1, nt), x.ldf(x.add(hb, x.add(x.smin(t1, NT - 1), RM))), x.f(NEG_MAX))
        M = x.wave_max(x.fmax(m0, m1))
        s0 = b.select(x.lt(lane, nt), x.mexp_fma(b.fsub(m0, M)), x.f(0.0))
        s1 = b.select(x.lt(t1, nt), x.mexp_fma(b.fsub(m1, M)), x.f(0.0))
        l0 = b.select(x.lt(lane, nt), x.ldf(x.add(hb, x.add(lane, RL))), x.f(0.0))
        l1 = b.select(x.lt(t1, nt), x.ldf(x.add(hb, x.add(x.smin(t1, NT - 1), RL))), x.f(0.0))
        tot = b.fadd(b.fmul(s0, l0), b.fmul(s1, l1))
        if x.k.mutate == "tile_norescale":
            tot = b.fadd(l0, l1)
        Lsum = x.wave_sum(tot)
        inv = b.fdiv(x.f(1.0), Lsum)
        c0 = x.f(1.0) if x.k.mutate == "tile_norescale" else s0
        x.st(x.add(hb, x.add(lane, RC)), b.fmul(c0, inv))
        with b.scf_if(x.lt(t1, NT)):
            x.st(x.add(hb, x.add(t1, RC)), b.fmul(s1, inv))
    x.barrier()
    acc = [x.f(0.0)] * (8 * Q7_HEADS)
    for i in range(lay.q7_rounds):
        j = x.add(slot, 128 * i)
        vf = x.bf16x8(words[i])
        jt = x.div(x.smin(j, SSR - 1), ST)
        for hh in range(Q7_HEADS):
            pj = b.fmul(x.ldf(x.add(x.smin(j, SSR - 1), pb + hh * SSR)), x.ldf(x.add(jt, RC + hh * NT)))
            pj = b.select(x.lt(j, ak), pj, x.f(0.0))
            for e in range(8):
                acc[8 * hh + e] = x.fma(pj, vf[e], acc[8 * hh + e])
    for off in (4, 8, 16, 32):
        acc = [x.xred(v_, off, b.fadd) for v_ in acc]
    with b.scf_if(x.lt(lane, 4)):
        for k in range(2 * Q7_HEADS):
            at = x.add(x.mul(wave, 128), x.add(x.mul(lane, 8), part + (k // 2) * 32 + 4 * (k % 2)))
            x.stv(at, acc[4 * k : 4 * k + 4])
    x.barrier()
    av = x.ldf(x.add(own, part))
    for w in range(1, W):
        av = b.fadd(av, x.ldf(x.add(own, part + w * 128)))
    return av


def _q5_kpre(x: QCtx, tile, kv, n, ak, tail, a_pages, a_key, rq, cb, dense):
    """Knob ``q5_kpre``: each thread polls the selected blocks of its two key rows itself (``sel``, after the
    selh hint) and issues their key-row loads before the q / k head poll. Returns [(cache row, 4 key words)]."""
    b, lay = x.b, x.lay
    ST, HW = lay.score_tile, lay.head_words
    tid = x.tid
    quad = x.mod(tid, 32)
    rs_k = x.rsrc(a_key)
    aux = 2 if x.k.kv_nt else 0
    mb_sel = x.mb("sel")
    out = []
    for r in range(2):
        row = x.add(x.div(tid, 32), 16 * r)
        slot = x.add(x.mul(tile, ST // 4), x.div(row, 4))
        have = x.lt(slot, n)
        with X.ifr(b, have, [I32]) as (then_, else_, res):
            with then_:
                if dense is None:
                    blk = x.poll([(mb_sel, x.mb_ix(slot, rq), 1)])[0][0]
                    X.yield_(b, x.page_block(a_pages, blk))
                else:
                    with X.ifr(b, dense, [I32]) as (t2, e2, r2):
                        with t2:
                            X.yield_(b, x.page_block(a_pages, _blk(x, slot, cb)))
                        with e2:
                            blk = x.poll([(mb_sel, x.mb_ix(slot, rq), 1)])[0][0]
                            X.yield_(b, x.page_block(a_pages, blk))
                    X.yield_(b, r2[0])
            with else_:
                X.yield_(b, x.c(0))
        kr = _cache_row(x, res[0], x.add(x.mul(tile, ST), row), n, ak, tail)
        at = x.add(x.mul(x.add(x.mul(kr, lay.kv_heads), kv), HW), x.mul(quad, 4))
        out.append((kr, x.load_words(rs_k, at, 4, aux)))
    return out


def q5(x: QCtx):
    lay = x.lay
    t = x.a.task
    turn = 0 if x.seq or lay.grid % (2 * x.rows) else 1
    for k in range(lay.task_rounds["q5"]):
        _q5(x, t if k == 0 else x.add(t, k * lay.grid), k * turn % x.rows)


# ======================================================================= Q6


def q6(x: QCtx):
    if x.k.fuse != "none":
        return  # N4b: Q7 takes Q5's scores itself
    b, lay = x.b, x.lay
    tid, lane, wave = x.tid, x.lane, x.wave
    t = x.a.task
    W = lay.waves
    eb, red = lay.lds["q6_e"], lay.lds["q6_red"]
    rq, hq = x.request(t, lay.q_heads)
    L = x.row_len(rq)
    ST = lay.score_tile
    with b.scf_if(x.lt(t, lay.tasks["q6"])):
        kv = x.div(hq, lay.group)
        _cb, _n, ak = x.list_len(L)
        nt = x.div(x.add(ak, ST - 1), ST)
        mb_ssh, mb_ss = x.mb("ssh"), x.mb("ss")
        x.pre_poll(lambda m: (mb_ssh, x.mb_ix(x.add(x.mul(x.smin(m, b.sub(nt, x.c(1))), 2), kv), rq)),
                   lay.score_tiles)  # fmt: skip
        top = b.sub(x.mul(nt, ST), x.c(2))
        specs = []
        for r in range(3):
            pos = x.smin(x.mul(x.add(tid, lay.threads * r), 2), top)
            specs.append((mb_ss, x.mb_ix(x.add(x.mul(hq, lay.ss_row), pos), rq), 2))
        got = x.poll(specs)
        s = []
        for r in range(3):
            for e in range(2):
                j = x.add(x.mul(x.add(tid, lay.threads * r), 2), e)
                s.append(b.select(x.lt(j, ak), b.bitcast(got[r][e], F32), x.f(NEG_MAX)))
        m = s[0]
        for k in range(1, 6):
            m = x.fmax(m, s[k])
        m = x.wave_max(m)
        with b.scf_if(x.eq(lane, 0)):
            x.st(x.add(wave, red), m)
        x.barrier()
        m = x.ldf(x.c(red))
        for w in range(1, W):
            m = x.fmax(m, x.ldf(x.c(red + w)))
        e = [b.select(b.fcmp("ole", v, x.f(NEG_MAX)), x.f(0.0), x.mexp_fma(b.fsub(v, m))) for v in s]
        for r in range(2):
            x.stv(x.add(x.mul(x.add(tid, lay.threads * r), 2), eb), e[2 * r : 2 * r + 2])
        with b.scf_if(x.lt(tid, 128)):
            x.stv(x.add(x.mul(tid, 2), eb + 2 * lay.threads * 2), e[4:6])
        x.barrier()
        with b.scf_if(x.lt(wave, 4)):
            loc = x.f(0.0)
            for k in range(9):
                loc = b.fadd(loc, x.ldf(x.add(tid, eb + 256 * k)))
            loc = x.wave_sum(loc)
            with b.scf_if(x.eq(lane, 0)):
                x.st(x.add(wave, red + W), loc)
        x.barrier()
        den = x.f(0.0)
        nw = 3 if x.k.mutate == "q6_den" else 4
        for w in range(nw):
            den = b.fadd(den, x.ldf(x.c(red + W + w)))
        inv = b.fdiv(x.f(1.0), den)
        mb_pp = x.mb("pp")
        for r in range(3):
            j = x.mul(x.add(tid, lay.threads * r), 2)
            with b.scf_if(x.lt(j, x.mul(nt, ST))):
                pw = x.bf16_pair(b.fmul(e[2 * r], inv), b.fmul(e[2 * r + 1], inv))
                x.put(mb_pp, x.mb_ix(x.add(x.mul(hq, lay.pp_row), x.div(j, 2)), rq), pw)
        x.barrier()
        with b.scf_if(x.eq(tid, 0)):
            x.put(x.mb("pph"), x.mb_ix(hq, rq), x.c(1))


# ======================================================================= Q7

Q7_HEADS = 4


def _q7(x: QCtx, t):
    b, lay = x.b, x.lay
    tid, lane, wave = x.tid, x.lane, x.wave
    rows = x.rows
    W = lay.waves
    G, HW, ST = lay.group, lay.head_words, lay.score_tile
    sl, pb, cur, part = lay.lds["q7_sel"], lay.lds["q7_p"], lay.lds["q7_cur"], lay.lds["q7_part"]
    rq, xx = x.request(t, lay.tasks["q7"] // rows)
    L = x.row_len(rq)
    a_pages = x.qsa_entry(rq, "pages")
    a_value = x.qsa_entry(rq, "value")
    with b.scf_if(x.lt(t, lay.tasks["q7"])):
        dt = x.mod(xx, 8)
        kv = x.div(xx, 24)
        qh0 = x.add(x.mul(kv, G), x.mul(x.mod(x.div(xx, 8), 3), Q7_HEADS))
        cb, n, ak = x.list_len(L)
        tail, now_row = _now_rows(x, a_pages, L)
        nt = x.div(x.add(ak, ST - 1), ST)
        own = x.mod(tid, 128)
        hd = x.div(own, 32)
        dim = x.mod(own, 32)
        mb_sel, mb_qp, mb_selh = x.mb("sel"), x.mb("qp"), x.mb("selh")
        dense = _dense(x, cb) if x.k.qdense >= 3 else None

        def from_sel():
            x.pre_poll(lambda m: (mb_selh, x.mb_ix(0, rq)), 1)
            with b.scf_if(x.lt(tid, n)):
                blk = x.poll([(mb_sel, x.mb_ix(tid, rq), 1)])[0][0]
                x.st(x.add(tid, sl), x.page_block(a_pages, blk))

        if dense is None:
            from_sel()
        else:
            # P13 ``_q7d``: an under-budget row's slot j is block j (no selh, no sel)
            with b.scf_if(dense):
                with b.scf_if(x.lt(tid, n)):
                    x.st(x.add(tid, sl), x.page_block(a_pages, _blk(x, tid, cb)))
            with b.scf_if(_not(x, dense)):
                from_sel()
        if x.seq:
            ct = x.smin(tid, 8 * rows - 1)
            cur_at = x.add(x.c(lay.p_v), x.add(x.mul(kv, HW), x.add(x.mul(dt, 16), x.mul(x.mod(ct, 8), 2))))
            cur_row = x.div(ct, 8)
        else:
            cur_at = x.add(x.c(lay.p_v), x.add(x.mul(kv, HW), x.add(x.mul(dt, 16), x.mul(x.smin(tid, 7), 2))))
            cur_row = rq
        gd = x.div(dim, 2)
        if x.k.mutate == "q7_gate":
            gd = b.xor(gd, x.c(1))
        gate_at = x.add(x.c(lay.p_q + HW),
                        x.add(x.mul(x.add(qh0, hd), 2 * HW), x.add(x.mul(dt, 16), gd)))  # fmt: skip
        v, g = x.poll([(mb_qp, x.mb_ix(cur_at, cur_row), 2), (mb_qp, x.mb_ix(gate_at, rq), 1)])
        gw = g[0]
        with b.scf_if(x.lt(tid, 8 * (rows if x.seq else 1))):
            x.stv(x.add(x.mul(tid, 2), cur), v)
        x.barrier()

        slot = x.add(x.mul(wave, 16), x.div(lane, 4))
        dq = x.mod(lane, 4)
        rs_v = x.rsrc(a_value)
        aux = 2 if x.k.kv_nt else 0
        vq = []
        for i in range(lay.q7_rounds):
            j = x.add(slot, 128 * i)
            g_ = x.ld(x.add(x.smin(x.div(j, 4), lay.sel - 1), sl))
            vr = _cache_row(x, g_, j, n, ak, tail)
            at = x.add(x.mul(x.add(x.mul(vr, lay.kv_heads), kv), HW), x.add(x.mul(dt, 16), x.mul(dq, 4)))
            vw = x.load_words(rs_v, at, 4, aux)
            vq.append((vw, vr if x.seq else x.eq(vr, now_row)))
        words = []
        if x.seq:
            now_vs = [x.ld4(x.add(x.mul(dq, 4), cur + 16 * q)) for q in range(rows)]
            for i in range(lay.q7_rounds):
                w, tk = vq[i]
                live = x.lt(x.add(slot, 128 * i), ak)
                pick = list(w)
                for q in range(rows):
                    hit = x.eq(tk, now_row[q])
                    pick = [b.select(hit, now_vs[q][e], pick[e]) for e in range(4)]
                words.append([b.select(live, pick[e], x.c(0)) for e in range(4)])
        else:
            now_v = x.ld4(x.add(x.mul(dq, 4), cur))
            for i in range(lay.q7_rounds):
                w, now = vq[i]
                live = x.lt(x.add(slot, 128 * i), ak)
                words.append([b.select(live, b.select(now, now_v[e], w[e]), x.c(0)) for e in range(4)])

        if x.k.fuse == "none":
            av = _q7_pv_mono(x, words, slot, qh0, nt, ak, rq, own)
        elif x.k.fuse == "tile":
            av = _q7_pv_tile(x, words, slot, qh0, kv, nt, ak, rq, own)
        else:
            av = _q7_pv_fused(x, words, slot, qh0, kv, nt, ak, rq, own, hd)

        with b.scf_if(x.lt(wave, 2)):
            qh = x.add(qh0, hd)
            gl, gh = x.bf2(gw)
            sig = x.bf16r(x.sigmoid_acc(b.select(x.eq(x.mod(dim, 2), 0), gl, gh)))
            out = x.bf16r(b.fmul(x.bf16r(av), sig))
            amax = b.fabs(out)
            for off in (1, 2, 4, 8, 16):
                amax = x.xred(amax, off, x.fmax)
            ex, scale = x.e8m0_fp8(amax)
            x1 = x.xshfl(out, 1)
            wd = x.fp8_word(out, x1, x.xshfl(out, 2), x.xshfl(x1, 2), scale)
            with b.scf_if(x.eq(x.mod(dim, 4), 0)):
                x.put(x.mb("d8"), x.mb_ix(x.add(x.mul(qh, 64), x.add(x.mul(dt, 8), x.div(dim, 4))), rq), wd)
            with b.scf_if(x.eq(dim, 0)):
                x.put(x.mb("de"), x.mb_ix(x.add(x.mul(qh, 8), dt), rq), ex)


def _q7_pv_mono(x: QCtx, words, slot, qh0, nt, ak, rq, own):
    """Q7 after the value loads (the mono): Q6's probabilities polled to LDS, PV, wave / CTA sums."""
    b, lay = x.b, x.lay
    tid, lane, wave = x.tid, x.lane, x.wave
    W, ST = lay.waves, lay.score_tile
    pb, part = lay.lds["q7_p"], lay.lds["q7_part"]
    mb_pph, mb_pp = x.mb("pph"), x.mb("pp")
    x.pre_poll(lambda m: (mb_pph, x.mb_ix(x.add(qh0, x.smin(m, Q7_HEADS - 1)), rq)), Q7_HEADS)
    PPR = lay.pp_row
    specs, dst = [], []
    for r in range(-(-Q7_HEADS * PPR // (2 * lay.threads))):
        s = x.smin(x.add(tid, lay.threads * r), Q7_HEADS * PPR // 2 - 1)
        h4 = x.div(s, PPR // 2)
        w0 = x.mul(x.mod(s, PPR // 2), 2)
        at = x.add(x.mul(x.add(qh0, h4), PPR), x.smin(w0, b.sub(x.mul(nt, ST // 2), x.c(2))))
        specs.append((mb_pp, x.mb_ix(at, rq), 2))
        dst.append(x.add(x.mul(h4, lay.ss_row), x.mul(w0, 2)))
    if x.k.q7_pp == "batch":
        got = x.poll(specs)
    else:
        got = [x.poll([sp])[0] for sp in specs]
    for r in range(len(dst)):
        x.stv(x.add(dst[r], pb), x.unpack(got[r]))
    x.barrier()

    acc = [x.f(0.0)] * (8 * Q7_HEADS)
    for i in range(lay.q7_rounds):
        j = x.add(slot, 128 * i)
        vf = x.bf16x8(words[i])
        for h4 in range(Q7_HEADS):
            pj = x.ldf(x.add(x.smin(j, lay.ss_row - 1), pb + h4 * lay.ss_row))
            pj = b.select(x.lt(j, ak), pj, x.f(0.0))
            for e in range(8):
                acc[8 * h4 + e] = x.fma(pj, vf[e], acc[8 * h4 + e])
    for off in (4, 8, 16, 32):
        acc = [x.xred(v_, off, b.fadd) for v_ in acc]
    with b.scf_if(x.lt(lane, 4)):
        for k in range(2 * Q7_HEADS):
            at = x.add(x.mul(wave, 128), x.add(x.mul(lane, 8), part + (k // 2) * 32 + 4 * (k % 2)))
            x.stv(at, acc[4 * k : 4 * k + 4])
    x.barrier()
    av = x.ldf(x.add(own, part))
    for w in range(1, W):
        av = b.fadd(av, x.ldf(x.add(own, part + w * 128)))

    return av


def _q7_pv_fused(x: QCtx, words, slot, qh0, kv, nt, ak, rq, own, hd):
    """N4b (``fuse``, tolerance class): Q7 takes Q5's scores itself (Q6 idles: one hop fewer) with f32
    probabilities. ``single``: the head's max over every position (CTA-wide), ``e = mexp(s - M)`` (Q6's exp)
    in LDS, the denominator in fixed order (per thread, wave butterfly, waves in order), PV with the f32 ``e``,
    one multiply by ``1 / den`` at the end. ``splitkv``: each wave a split (its 17 x 16 positions): the
    wave's max, ``e = mexp(s - m_w)`` inside PV, ``l_w`` in the same butterfly as the outputs; the 8 splits
    merged in wave order with ``mexp(m_w - M)`` (no CTA-wide max before PV)."""
    b, lay = x.b, x.lay
    tid, lane, wave = x.tid, x.lane, x.wave
    W, ST, SSR = lay.waves, lay.score_tile, lay.ss_row
    pb, part = lay.lds["q7_p"], lay.lds["q7_part"]
    RB = lay.lds["q6_e"]  # Q6's arena entry: its stage idles in a fused build
    RL = RB + Q7_HEADS * W
    mb_ssh, mb_ss = x.mb("ssh"), x.mb("ss")
    x.pre_poll(lambda m: (mb_ssh, x.mb_ix(x.add(x.mul(x.smin(m, b.sub(nt, x.c(1))), 2), kv), rq)), lay.score_tiles)
    top = b.sub(x.mul(nt, ST), x.c(2))
    specs = []
    for h4 in range(Q7_HEADS):
        for r in range(3):
            pos = x.smin(x.mul(x.add(tid, lay.threads * r), 2), top)
            specs.append((mb_ss, x.mb_ix(x.add(x.mul(x.add(qh0, h4), SSR), pos), rq), 2))
    got = x.poll(specs)
    sv = [[None] * 6 for _ in range(Q7_HEADS)]
    for h4 in range(Q7_HEADS):
        for r in range(3):
            for e in range(2):
                j = x.add(x.mul(x.add(tid, lay.threads * r), 2), e)
                sv[h4][2 * r + e] = b.select(x.lt(j, ak), b.bitcast(got[3 * h4 + r][e], F32), x.f(NEG_MAX))

    def to_lds(vals_of):
        for h4 in range(Q7_HEADS):
            for r in range(3):
                j0 = x.mul(x.add(tid, lay.threads * r), 2)
                with b.scf_if(x.lt(j0, SSR)):
                    x.stv(x.add(j0, pb + h4 * SSR), vals_of(h4)[2 * r : 2 * r + 2])

    acc = [x.f(0.0)] * (8 * Q7_HEADS)
    if x.k.fuse == "single":
        for h4 in range(Q7_HEADS):
            m = sv[h4][0]
            for k in range(1, 6):
                m = x.fmax(m, sv[h4][k])
            m = x.wave_max(m)
            with b.scf_if(x.eq(lane, 0)):
                x.st(x.add(wave, RB + h4 * W), m)
        x.barrier()
        ev = []
        for h4 in range(Q7_HEADS):
            M = x.ldf(x.c(RB + h4 * W))
            for w in range(1, W):
                M = x.fmax(M, x.ldf(x.c(RB + h4 * W + w)))
            ev.append([b.select(b.fcmp("ole", v, x.f(NEG_MAX)), x.f(0.0), x.mexp_fma(b.fsub(v, M))) for v in sv[h4]])
        to_lds(lambda h4: ev[h4])
        for h4 in range(Q7_HEADS):
            loc = x.f(0.0)
            for v in ev[h4]:
                loc = b.fadd(loc, v)
            loc = x.wave_sum(loc)
            with b.scf_if(x.eq(lane, 0)):
                x.st(x.add(wave, RL + h4 * W), loc)
        x.barrier()
        for i in range(lay.q7_rounds):
            j = x.add(slot, 128 * i)
            vf = x.bf16x8(words[i])
            for h4 in range(Q7_HEADS):
                pj = x.ldf(x.add(x.smin(j, SSR - 1), pb + h4 * SSR))
                pj = b.select(x.lt(j, ak), pj, x.f(0.0))
                for e in range(8):
                    acc[8 * h4 + e] = x.fma(pj, vf[e], acc[8 * h4 + e])
    else:  # splitkv
        to_lds(lambda h4: sv[h4])
        x.barrier()
        sj = []
        for i in range(lay.q7_rounds):
            j = x.add(slot, 128 * i)
            row = []
            for h4 in range(Q7_HEADS):
                v = x.ldf(x.add(x.smin(j, SSR - 1), pb + h4 * SSR))
                row.append(b.select(x.lt(j, ak), v, x.f(NEG_MAX)))
            sj.append(row)
        mw = []
        for h4 in range(Q7_HEADS):
            m = sj[0][h4]
            for i in range(1, lay.q7_rounds):
                m = x.fmax(m, sj[i][h4])
            mw.append(x.wave_max(m))
        lsum = [x.f(0.0)] * Q7_HEADS
        for i in range(lay.q7_rounds):
            vf = x.bf16x8(words[i])
            for h4 in range(Q7_HEADS):
                v = sj[i][h4]
                pj = b.select(b.fcmp("ole", v, x.f(NEG_MAX)), x.f(0.0), x.mexp_fma(b.fsub(v, mw[h4])))
                lsum[h4] = b.fadd(lsum[h4], pj)
                for e in range(8):
                    acc[8 * h4 + e] = x.fma(pj, vf[e], acc[8 * h4 + e])
        for off in (4, 8, 16, 32):
            lsum = [x.xred(v_, off, b.fadd) for v_ in lsum]
        with b.scf_if(x.eq(lane, 0)):
            for h4 in range(Q7_HEADS):
                x.st(x.add(wave, RB + h4 * W), mw[h4])
                x.st(x.add(wave, RL + h4 * W), lsum[h4])
    for off in (4, 8, 16, 32):
        acc = [x.xred(v_, off, b.fadd) for v_ in acc]
    with b.scf_if(x.lt(lane, 4)):
        for k in range(2 * Q7_HEADS):
            at = x.add(x.mul(wave, 128), x.add(x.mul(lane, 8), part + (k // 2) * 32 + 4 * (k % 2)))
            x.stv(at, acc[4 * k : 4 * k + 4])
    x.barrier()
    hb = x.mul(hd, W)
    if x.k.fuse == "single":
        av = x.ldf(x.add(own, part))
        for w in range(1, W):
            av = b.fadd(av, x.ldf(x.add(own, part + w * 128)))
        den = x.f(0.0)
        nw = W - 1 if x.k.mutate == "fuse_den" else W
        for w in range(nw):
            den = b.fadd(den, x.ldf(x.add(hb, RL + w)))
        return b.fmul(av, b.fdiv(x.f(1.0), den))
    M = x.ldf(x.add(hb, RB))
    for w in range(1, W):
        M = x.fmax(M, x.ldf(x.add(hb, RB + w)))
    O, Ls = x.f(0.0), x.f(0.0)
    for w in range(W):
        if x.k.mutate == "splitkv_drop" and w == 0:
            continue
        mwv = x.ldf(x.add(hb, RB + w))
        sw = x.f(1.0) if x.k.mutate == "splitkv_norescale" else x.mexp_fma(b.fsub(mwv, M))
        O = b.fadd(O, b.fmul(sw, x.ldf(x.add(own, part + w * 128))))
        Ls = b.fadd(Ls, b.fmul(sw, x.ldf(x.add(hb, RL + w))))
    return b.fmul(O, b.fdiv(x.f(1.0), Ls))


def q7(x: QCtx):
    t = x.a.task
    for k in range(x.lay.task_rounds["q7"]):
        _q7(x, t if k == 0 else x.add(t, k * x.lay.grid))
