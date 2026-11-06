# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage ``split``: sparse MLA over one chunk of a row's selected keys.

Task ``(s, t)`` takes keys ``t * SK .. t * SK + SK - 1`` of row ``s``'s
top-k (SK = 32 or 64): wave 0 resolves them to cache slots, every wave
gathers its keys' E4M3 cache rows into bf16 LDS tiles (this launch's own
rows come from ``kvnew`` / ``penew``), and per head group the 8 waves score
``K Q^T`` on MFMA, take a split-local softmax (one head per wave) and
multiply ``P V`` (V = the latent rows). Publishes the unnormalized
``sp_acc`` rows and the softmax max / sum ``sp_m`` / ``sp_l`` per head.
"""

from __future__ import annotations

from rocke.core.ir import BF16, F32, I32, Value, VectorType

from .ctx import KS, OUT_WORDS, PS, QS, THREADS, WAVES, Ml4Ctx, split_keys
from .gemv import fp8_to_bf16x8
from .stage_index import single_list, tagged_index_word, tagged_indices

KV_LORA = 512
PE_DIM = 64
QK_DIM = KV_LORA + PE_DIM
TOPK = 2048
SOFTMAX_SCALE = 128**-0.5
NEG = -1.0e30
LOG2E = 1.4426950408889634


def emit_split(cx: Ml4Ctx) -> None:
    cx.tasks("split", lambda tt: _task(cx, tt))


def _geometry(cx: Ml4Ctx):
    sk = split_keys(cx.spec.rows, cx.spec.heads)
    groups = cx.spec.heads // WAVES
    merged = groups > 1
    return sk, TOPK // sk, groups, merged


def _resolve_keys(cx: Ml4Ctx, s: Value, t: Value, sk: int):
    """(nkeys, sparse) of row ``s``; wave 0 writes the chunk's cache slots to
    LDS ``keys``."""
    b = cx.b
    r = cx.rsrc("sparse_kv_indptr")
    begin = cx.ld_i32(r, s)
    end = cx.ld_i32(r, b.add(s, cx.i(1)))
    nkeys = b.sub(end, begin)
    sparse = b.cmp_lt(begin, end)
    slot_of = lambda logical: cx.page_slot(s, logical)  # noqa: E731
    if cx.spec.with_indexer and cx.spec.split_bt_lds:
        slot_of = _block_table_lds(cx, s)
        cx.barrier()
    # sel_tagged: a single-list row's indices arrive as tagged pairs
    tagged = None
    if tagged_indices(cx):
        bound = b.add(cx.row_position(s), cx.i(1))
        tagged = b.land(sparse, single_list(cx, bound))
        # the list's winners; past them the plain indices hold explicit zeros
        winners = b.smin(bound, cx.i(TOPK))
    if cx.spec.with_indexer:
        wait = b.land(sparse, b.cmp_eq(cx.wave, cx.i(0)))
        if tagged is not None:
            wait = b.land(wait, b.lnot(tagged))
        with b.scf_if(wait):
            with b.scf_if(b.cmp_eq(cx.lane, cx.i(0))):
                cx.poll([("indices_ready", s, 1)])
            if cx.spec.split_acquire == "fence":
                b.fence(scope="agent", ordering="acquire")
    with b.scf_if(b.cmp_eq(cx.wave, cx.i(0))):
        with b.scf_if(b.cmp_lt(cx.lane, cx.i(sk))):
            k_pos = b.add(b.mul(t, cx.i(sk)), cx.lane)
            k_cl = b.select(b.cmp_lt(k_pos, nkeys), k_pos, cx.i(0))
            at = b.add(b.mul(s, cx.i(TOPK)), k_cl)
            dst = b.add(cx.lane, cx.i(cx.keys))
            if tagged is not None:
                won = b.cmp_lt(k_cl, winners)
                with b.scf_if(b.land(tagged, won)):
                    got = cx.poll([("isel0", tagged_index_word(cx, s, k_cl), 1)])[0][0]
                    cx.lds_st(dst, slot_of(got))
                with b.scf_if(b.land(tagged, b.lnot(won))):
                    cx.lds_st(dst, slot_of(cx.i(0)))
                with b.scf_if(b.lnot(tagged)):
                    got = cx.ld_i32_cp(cx.mb_rsrc("indices"), at)
                    cx.lds_st(dst, slot_of(b.select(sparse, got, k_cl)))
            elif cx.spec.with_indexer:
                got = cx.ld_i32_cp(cx.mb_rsrc("indices"), at)
                cx.lds_st(dst, slot_of(b.select(sparse, got, k_cl)))
            else:
                # a row without keys reads no index (its entries may be stale)
                cx.lds_st(dst, cx.i(0))
                with b.scf_if(sparse):
                    cx.lds_st(dst, cx.page_slot(s, cx.ld_i32(cx.rsrc("indices"), at)))
    return nkeys, sparse


def _block_table_lds(cx: Ml4Ctx, s: Value):
    """Row ``s``'s request's block table (every page the index plane can
    hold) into the key-tile LDS, which the gather overwrites only after the
    lookups; returns the LDS-backed ``page_slot``. Needs a barrier before use."""
    b = cx.b
    ps = cx.spec.index_page_size
    pages = cx.spec.index_capacity // ps
    width = cx.spec.index_request_width or cx.spec.rows
    first = b.mul(b.div(s, cx.i(width)), cx.i(cx.spec.index_bt_stride))
    r_bt = cx.rsrc("cur_pos")
    for q in range((pages + THREADS - 1) // THREADS):
        p = b.add(cx.tid, cx.i(q * THREADS))
        with b.scf_if(b.cmp_lt(p, cx.i(pages))):
            cx.lds_st(b.add(p, cx.i(cx.kt_off)), cx.ld_i32(r_bt, b.add(first, p)))

    def slot_of(logical: Value) -> Value:
        page = cx.lds_ld(b.add(b.lshr(logical, cx.i(ps.bit_length() - 1)), cx.i(cx.kt_off)))
        return b.add(b.mul(page, cx.i(ps)), b.land(logical, cx.i(ps - 1)))

    return slot_of


def _gather_old_kv(cx: Ml4Ctx, kpw: int) -> None:
    """Each wave copies its keys' E4M3 latent (512 B) and rope (64 B) cache
    rows into the bf16 LDS tiles (the draft kind: bf16 rows, 1024 + 128 B)."""
    b = cx.b
    r_kv, r_pe = cx.rsrc("kv_cache"), cx.rsrc("pe_cache")
    krows = [cx.lds_ld(b.add(b.mul(cx.wave, cx.i(kpw)), cx.i(cx.keys + jj))) for jj in range(kpw)]
    if cx.spec.draft:
        for jj in range(kpw):
            j = b.add(b.mul(cx.wave, cx.i(kpw)), cx.i(jj))
            row = b.mul(krows[jj], cx.i(QK_DIM // 2))
            kv = b.buffer_load_vN(r_kv, b.mul(b.add(row, b.mul(cx.lane, cx.i(4))), cx.i(4)), cx.i(0), I32, 4)
            cx.lds_st4(b.add(b.add(b.mul(j, cx.i(KS)), b.mul(cx.lane, cx.i(4))), cx.i(cx.kt_off)), kv)
            with b.scf_if(b.cmp_lt(cx.lane, cx.i(PE_DIM // 8))):
                pe_w = b.add(row, b.add(cx.i(KV_LORA // 2), b.mul(cx.lane, cx.i(4))))
                pe = b.buffer_load_vN(r_pe, b.mul(pe_w, cx.i(4)), cx.i(0), I32, 4)
                cx.lds_st4(b.add(b.add(b.mul(j, cx.i(PS)), b.mul(cx.lane, cx.i(4))), cx.i(cx.pt_off)), pe)
        return
    for jj in range(kpw):
        j = b.add(b.mul(cx.wave, cx.i(kpw)), cx.i(jj))
        row = b.mul(krows[jj], cx.i(QK_DIM // 4))
        raw = b.buffer_load_vN(r_kv, b.mul(b.add(row, b.mul(cx.lane, cx.i(2))), cx.i(4)), cx.i(0), I32, 2)
        kv8 = fp8_to_bf16x8(cx, b.vec_extract(raw, 0), b.vec_extract(raw, 1))
        dst = b.add(b.add(b.mul(j, cx.i(KS)), b.mul(cx.lane, cx.i(4))), cx.i(cx.kt_off))
        cx.lds_st4(dst, b.bitcast(kv8, VectorType(I32, 4)))
        with b.scf_if(b.cmp_lt(cx.lane, cx.i(PE_DIM // 8))):
            pe_w = b.add(row, b.add(cx.i(KV_LORA // 4), b.mul(cx.lane, cx.i(2))))
            pe_raw = b.buffer_load_vN(r_pe, b.mul(pe_w, cx.i(4)), cx.i(0), I32, 2)
            pe8 = fp8_to_bf16x8(cx, b.vec_extract(pe_raw, 0), b.vec_extract(pe_raw, 1))
            dst = b.add(b.add(b.mul(j, cx.i(PS)), b.mul(cx.lane, cx.i(4))), cx.i(cx.pt_off))
            cx.lds_st4(dst, b.bitcast(pe8, VectorType(I32, 4)))


def newkv_lds(cx: Ml4Ctx) -> bool:
    """split_newkv_lds applies: the task waits on the top-k (which the
    cache's rows land well before; without the indexer the staging itself
    would wait on them), and every row's latent fits LDS ``outs``, its rope
    ``misc``."""
    S = cx.spec.rows
    return (cx.spec.split_newkv_lds and cx.spec.with_indexer and S * KV_LORA <= OUT_WORDS
            and S * PE_DIM <= cx.red - cx.misc)  # fmt: skip


def _stage_newkv(cx: Ml4Ctx) -> None:
    """This launch's rows' latent (``kvnew``, f32 words) into LDS ``outs`` and
    rope (``penew``) into ``misc``: patching then reads LDS, not mailboxes."""
    b = cx.b
    S = cx.spec.rows
    n_kv, n_pe = S * KV_LORA, S * PE_DIM
    for q in range((n_kv // 4 + THREADS - 1) // THREADS):
        w = b.add(cx.tid, cx.i(q * THREADS))
        with b.scf_if(b.cmp_lt(w, cx.i(n_kv // 4))):
            got = cx.poll([("kvnew", b.mul(w, cx.i(4)), 2), ("kvnew", b.add(b.mul(w, cx.i(4)), cx.i(2)), 2)])
            cx.lds_st4(b.add(b.mul(w, cx.i(4)), cx.i(cx.outs)), b.vec_pack(got[0] + got[1], I32))
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_pe // 2))):
        pv = cx.poll([("penew", b.mul(cx.tid, cx.i(2)), 2)])[0]
        cx.lds_st(b.add(b.mul(cx.tid, cx.i(2)), cx.i(cx.misc)), pv[0])
        cx.lds_st(b.add(b.add(b.mul(cx.tid, cx.i(2)), cx.i(1)), cx.i(cx.misc)), pv[1])


def _patch_new_kv(cx: Ml4Ctx, kpw: int) -> None:
    """Keys that are this launch's own rows take ``kvnew`` / ``penew``."""
    b = cx.b
    S = cx.spec.rows
    staged = newkv_lds(cx)
    slots = [cx.row_slot(s) for s in range(S)]
    active = [b.cmp_ge(sl, cx.i(0)) for sl in slots]
    for jj in range(kpw):
        j = b.add(b.mul(cx.wave, cx.i(kpw)), cx.i(jj))
        kr = cx.lds_ld(b.add(j, cx.i(cx.keys)))
        sn = cx.i(-1)
        for ns in range(S):
            match = b.land(active[ns], b.cmp_eq(kr, slots[ns]))
            sn = b.select(match, cx.i(ns), sn)
        with b.scf_if(b.cmp_ge(sn, cx.i(0))):
            base = b.add(b.mul(sn, cx.i(KV_LORA)), b.mul(cx.lane, cx.i(8)))
            if staged:
                v8 = [b.vec_extract(cx.lds_ld4(b.add(b.add(base, cx.i(h * 4)), cx.i(cx.outs))), e)
                      for h in range(2) for e in range(4)]  # fmt: skip
                kvp = [v8[2 * m : 2 * m + 2] for m in range(4)]
            else:
                kvp = cx.poll([("kvnew", b.add(base, cx.i(m * 2)), 2) for m in range(4)])
            w = [cx.bf16_pair(b.bitcast(v[0], F32), b.bitcast(v[1], F32)) for v in kvp]
            dst = b.add(b.add(b.mul(j, cx.i(KS)), b.mul(cx.lane, cx.i(4))), cx.i(cx.kt_off))
            cx.lds_st4(dst, b.vec_pack(w, I32))
            with b.scf_if(b.cmp_lt(cx.lane, cx.i(PE_DIM // 2))):
                pe = b.add(b.mul(sn, cx.i(PE_DIM)), b.mul(cx.lane, cx.i(2)))
                if staged:
                    pv = [cx.lds_ld(b.add(pe, cx.i(cx.misc))), cx.lds_ld(b.add(b.add(pe, cx.i(1)), cx.i(cx.misc)))]
                else:
                    pv = cx.poll([("penew", pe, 2)])[0]
                word = cx.bf16_pair(b.bitcast(pv[0], F32), b.bitcast(pv[1], F32))
                cx.lds_st(b.add(b.add(b.mul(j, cx.i(PS)), cx.lane), cx.i(cx.pt_off)), word)


def _stage_q(cx: Ml4Ctx, s: Value) -> None:
    """Q[h] (bf16 words h * QS + d / 2): latent 512 then rope 64, all heads."""
    b = cx.b
    H = cx.spec.heads
    nq = H * KV_LORA // 4 // 512
    tpe = b.smin(cx.tid, cx.i(H * PE_DIM // 4 - 1))
    specs = []
    for i in range(nq):
        w4 = b.add(cx.tid, cx.i(i * 512))
        specs.append(("q_lat", b.lshr(b.add(b.mul(s, cx.i(H * KV_LORA)), b.mul(w4, cx.i(4))), cx.i(1)), 2))
    specs.append(("q_pe", b.lshr(b.add(b.mul(s, cx.i(H * PE_DIM)), b.mul(tpe, cx.i(4))), cx.i(1)), 2))
    qv = cx.poll(specs)
    for i in range(nq):
        w4 = b.add(cx.tid, cx.i(i * 512))
        qw = b.add(b.mul(b.lshr(w4, cx.i(7)), cx.i(QS)), b.mul(b.land(w4, cx.i(127)), cx.i(2)))
        cx.lds_st(b.add(qw, cx.i(cx.xs)), qv[i][0])
        cx.lds_st(b.add(qw, cx.i(cx.xs + 1)), qv[i][1])
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(H * PE_DIM // 4))):
        hh = b.lshr(cx.tid, cx.i(4))
        qw = b.add(b.mul(hh, cx.i(QS)), b.add(cx.i(KV_LORA // 2), b.mul(b.land(cx.tid, cx.i(15)), cx.i(2))))
        cx.lds_st(b.add(qw, cx.i(cx.xs)), qv[nq][0])
        cx.lds_st(b.add(qw, cx.i(cx.xs + 1)), qv[nq][1])


def _b_bf16x8(cx: Ml4Ctx, word: Value) -> Value:
    return cx.b.bitcast(cx.lds_ld4(word), VectorType(BF16, 8))


def _task(cx: Ml4Ctx, tt: Value) -> None:
    b = cx.b
    H = cx.spec.heads
    sk, n_split, groups, merged = _geometry(cx)
    kpw = sk // WAVES
    if merged:
        s = b.div(tt, cx.i(n_split))
        hg0 = cx.i(0)
    else:
        s = b.div(tt, cx.i(groups * n_split))
        hg0 = b.mod(b.div(tt, cx.i(n_split)), cx.i(groups))
    t = b.mod(tt, cx.i(n_split))
    q_first = cx.spec.split_q_first
    if q_first:
        # the queries do not depend on the top-k: staged while it finishes
        _stage_q(cx, s)
    if newkv_lds(cx):
        # nor do this launch's own rows, which some keys will be
        _stage_newkv(cx)
    nkeys, _ = _resolve_keys(cx, s, t, sk)
    cx.barrier()
    _gather_old_kv(cx, kpw)
    cx.barrier()
    if not q_first:
        _stage_q(cx, s)
    _patch_new_kv(cx, kpw)
    cx.barrier()
    cx.stamp("split", tt, 2)
    lane16 = b.land(cx.lane, cx.i(15))
    quad = b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(4))
    for hgi in range(groups if merged else 1):
        head_group = cx.i(hgi) if merged else hg0
        h = b.add(b.mul(head_group, cx.i(WAVES)), cx.wave)
        if hgi:
            cx.barrier()
        hn = b.add(b.mul(head_group, cx.i(WAVES)), b.smin(lane16, cx.i(WAVES - 1)))
        rgk = b.land(cx.wave, cx.i(sk // 16 - 1))
        half = b.land(b.lshr(cx.wave, cx.i((sk // 16).bit_length() - 1)), cx.i(1))
        key = b.add(b.mul(rgk, cx.i(16)), lane16)
        c = b.vec_pack([cx.f(0.0)] * 4, F32)
        steps = QK_DIM // 32 // 2
        for st in range(steps):
            kst = b.add(b.mul(half, cx.i(steps)), cx.i(st))
            kw = b.select(
                b.cmp_lt(kst, cx.i(KV_LORA // 32)),
                b.add(cx.i(cx.kt_off), b.add(b.mul(key, cx.i(KS)), b.mul(kst, cx.i(16)))),
                b.add(cx.i(cx.pt_off), b.add(b.mul(key, cx.i(PS)), b.mul(b.sub(kst, cx.i(KV_LORA // 32)), cx.i(16)))),
            )
            a = _b_bf16x8(cx, b.add(kw, quad))
            qop = _b_bf16x8(cx, b.add(b.add(b.mul(hn, cx.i(QS)), b.mul(kst, cx.i(16))), b.add(quad, cx.i(cx.xs))))
            c = b.mfma_f32_16x16x32_bf16(a, qop, c)
        red_w = b.add(b.mul(b.add(b.mul(cx.wave, cx.i(64)), cx.lane), cx.i(4)), cx.i(cx.red))
        if sk == 64:
            cx.lds_st4(red_w, b.bitcast(c, VectorType(I32, 4)))
        else:
            with b.scf_if(b.cmp_lt(cx.wave, cx.i(4))):
                cx.lds_st4(red_w, b.bitcast(c, VectorType(I32, 4)))
        cx.barrier()
        # split-local softmax: wave = head, lane = key (score = both K halves)
        kidx = b.add(b.mul(t, cx.i(sk)), cx.lane)
        valid = b.land(b.cmp_lt(cx.lane, cx.i(sk)), b.cmp_lt(kidx, nkeys))
        cl = b.add(cx.wave, b.mul(b.lshr(lane16, cx.i(2)), cx.i(16)))
        key_rg = b.smin(b.lshr(cx.lane, cx.i(4)), cx.i(sk // 16 - 1))
        e = b.land(lane16, cx.i(3))

        def score_word(rg: Value) -> Value:
            return b.add(b.add(b.mul(b.add(b.mul(rg, cx.i(64)), cl), cx.i(4)), e), cx.i(cx.red))

        raw = b.fadd(cx.lds_ldf(score_word(key_rg)), cx.lds_ldf(score_word(b.add(key_rg, cx.i(sk // 16)))))
        sc_v = b.select(valid, b.fmul(raw, cx.f(SOFTMAX_SCALE)), cx.f(NEG))
        m = cx.wave_max(sc_v)
        p = b.select(valid, b.exp2_fast(b.fmul(b.fsub(sc_v, m), cx.f(LOG2E))), cx.f(0.0))
        lsum = cx.wave_sum(p)
        p_n = cx.xshfl(p, 1)
        even = b.cmp_eq(b.land(cx.lane, cx.i(1)), cx.i(0))
        with b.scf_if(b.land(b.cmp_lt(cx.lane, cx.i(sk)), even)):
            cx.lds_st(b.add(b.add(b.mul(h, cx.i(sk // 2)), b.lshr(cx.lane, cx.i(1))), cx.i(cx.xs)), cx.bf16_pair(p, p_n))
        cx.barrier()
        # O = P V: heads M, keys K, latent dim pairs N (even / odd dims)
        for g in range(KV_LORA // 32 // WAVES):
            dw = b.add(b.mul(b.add(b.mul(cx.wave, cx.i(KV_LORA // 32 // WAVES)), cx.i(g)), cx.i(16)), lane16)
            c0 = b.vec_pack([cx.f(0.0)] * 4, F32)
            c1 = b.vec_pack([cx.f(0.0)] * 4, F32)
            for js in range(sk // 32):
                a = _b_bf16x8(cx, b.add(b.add(b.mul(hn, cx.i(sk // 2)), cx.i(js * 16)), b.add(quad, cx.i(cx.xs))))
                k0 = b.add(cx.i(js * 32), b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(8)))
                ws = [cx.lds_ld(b.add(b.add(b.mul(b.add(k0, cx.i(i)), cx.i(KS)), dw), cx.i(cx.kt_off))) for i in range(8)]
                w_lo = [b.lor(b.land(ws[2 * i], cx.i(0xFFFF)), b.shl(ws[2 * i + 1], cx.i(16))) for i in range(4)]
                w_hi = [b.lor(b.lshr(ws[2 * i], cx.i(16)), b.land(ws[2 * i + 1], cx.i(-65536))) for i in range(4)]
                b0 = b.bitcast(b.vec_pack(w_lo, I32), VectorType(BF16, 8))
                b1 = b.bitcast(b.vec_pack(w_hi, I32), VectorType(BF16, 8))
                c0 = b.mfma_f32_16x16x32_bf16(a, b0, c0)
                c1 = b.mfma_f32_16x16x32_bf16(a, b1, c1)
            with b.scf_if(b.cmp_lt(cx.lane, cx.i(32))):
                for el in range(4):
                    hh = b.add(b.mul(head_group, cx.i(WAVES)), b.add(b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(4)), cx.i(el)))
                    row = b.add(b.mul(b.add(b.mul(s, cx.i(n_split)), t), cx.i(H)), hh)
                    cx.put_bf("sp_acc", b.add(b.mul(row, cx.i(KV_LORA)), b.mul(dw, cx.i(2))),
                              [b.vec_extract(c0, el), b.vec_extract(c1, el)])  # fmt: skip
        with b.scf_if(b.cmp_eq(cx.lane, cx.i(0))):
            row = b.add(b.mul(b.add(b.mul(s, cx.i(n_split)), t), cx.i(H)), h)
            cx.put("sp_m", row, m)
            cx.put("sp_l", row, lsum)
