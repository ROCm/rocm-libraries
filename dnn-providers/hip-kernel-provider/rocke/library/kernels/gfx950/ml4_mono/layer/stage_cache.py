# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage ``cache``: this launch's rows into the latent and index caches.

One task. Each verify row's kv latent (``kv_a``) is RMS-normed with ``g_kv``,
rounded through bf16 and E4M3 and written to the row's ATOM cache slot (512
E4M3 bytes), its rope key RoPE-rotated and written behind it (64 bytes); both
are published (``kvnew``, ``penew``) for this launch's attention. The
KV-sharing kind writes nothing and publishes the cached row of its source
layer instead. Indexer kinds LayerNorm the index key (``index_k``), rotate its
first 64 dims (RoPE kinds), quantize it to E4M3 with one power-of-two scale
per row, write it into the paged index plane, and publish the dequantized key
(``index_k_new``) and ``index_ready``.
"""

from __future__ import annotations

from rocke.core.ir import F32, I32, Value

from .ctx import EPS, Ml4Ctx

KV_LORA = 512
PE_DIM = 64
QK_DIM = KV_LORA + PE_DIM
INDEX_DIM = 128
INDEX_TILE = 16
# bytes per index-plane row: 128 key bytes, then (per page) an fp32 scale
INDEX_PAGE_ROW = INDEX_DIM + 4
FP8_MAX = 448.0
# rows of the parameter table (``table``)
T_G_INDEX_K, T_B_INDEX_K = 5, 6


def emit_cache(cx: Ml4Ctx, part: str = "all") -> None:
    """``part``: ``all``, or (``cache_kv_late``) ``index`` (the index keys and
    ``index_ready``) and later ``kv`` (the latent rows)."""
    cx.tasks("cache", lambda t: _task(cx, part))


def _sat(cx: Ml4Ctx, x: Value) -> Value:
    b = cx.b
    return b.fmin(b.fmax(x, cx.f(-FP8_MAX)), cx.f(FP8_MAX))


def _quad(cx: Ml4Ctx, pair: Value, upper_lane: Value) -> Value:
    """``pair`` (2 E4M3 bytes) | the pair of lane ``upper_lane`` << 16."""
    b = cx.b
    return b.lor(pair, b.shl(cx.bpermute(upper_lane, pair), cx.i(16)))


def _task(cx: Ml4Ctx, part: str) -> None:
    b = cx.b
    S, tid, lane = cx.spec.rows, cx.tid, cx.lane
    kvs = cx.spec.kind == "nope_kvs"
    rope = cx.spec.kind in ("rope", "rope_shared", "draft")
    r_kv, r_pe = cx.rsrc("kv_cache"), cx.rsrc("pe_cache")
    g = cx.ld_bf16(cx.rsrc("g_kv"), tid)
    tpe = b.land(tid, cx.i(PE_DIM // 2 - 1))
    pos = [cx.row_position(s) for s in range(S)]
    rope_i = [b.add(b.mul(p, cx.i(PE_DIM // 2)), tpe) for p in pos]
    cs = [cx.ld_rope("rope_cos", i) for i in rope_i]
    sns = [cx.ld_rope("rope_sin", i) for i in rope_i]
    cx.barrier()
    if cx.spec.with_indexer and part != "kv":
        # the index keys first: index scoring waits on index_ready, while the
        # latent rows' reader (attention) waits on the top-k selection
        _index_keys(cx, cs, sns, pos, rope)
        cx.barrier()
    if part == "index":
        return
    got = cx.poll([("kv_a", b.add(cx.i(s * QK_DIM), tid), 1) for s in range(S)])
    vs = [b.bitcast(w[0], F32) for w in got]
    pe_i = b.add(cx.i(KV_LORA), b.mul(tpe, cx.i(2)))
    got = cx.poll([("kv_a", b.add(cx.i(s * QK_DIM), pe_i), 2) for s in range(S)])
    pes = [(b.bitcast(w[0], F32), b.bitcast(w[1], F32)) for w in got]
    pe_lane = b.cmp_lt(tid, cx.i(PE_DIM // 2))
    if kvs:
        for s in range(S):
            raw = cx.row_slot(s)
            slot = b.select(b.cmp_ge(raw, cx.i(0)), raw, cx.i(0))
            row = b.mul(slot, cx.i(QK_DIM))
            cx.put("kvnew", b.add(cx.i(s * KV_LORA), tid), cx.ld_fp8(r_kv, b.add(row, tid)))
            with b.scf_if(pe_lane):
                p0, p1 = cx.ld_fp8_pair(r_pe, b.add(b.lshr(b.add(row, cx.i(KV_LORA)), cx.i(1)), tid))
                cx.put2("penew", b.add(cx.i(s * PE_DIM), b.mul(tid, cx.i(2))), p0, p1)
    elif cx.spec.draft:
        _draft_rows(cx, vs, pes, cs, sns, g, pe_lane)
    else:
        ssq = cx.block_sums([b.fmul(v, v) for v in vs])
        for s in range(S):
            slot = cx.row_slot(s)
            active = b.cmp_ge(slot, cx.i(0))
            rstd = b.rsqrt(b.fadd(b.fmul(ssq[s], cx.f(1.0 / KV_LORA)), cx.f(EPS)))
            value = _sat(cx, cx.bf16_round(b.fmul(b.fmul(vs[s], rstd), g)))
            kvn, _ = cx.fp8_roundtrip(value, value)
            with b.scf_if(active):
                quad = _quad(cx, cx.fp8_pair_bits(kvn, cx.xshfl(kvn, 1)),
                             b.add(b.land(lane, cx.i(-4)), cx.i(2)))  # fmt: skip
                with b.scf_if(b.cmp_eq(b.land(lane, cx.i(3)), cx.i(0))):
                    dword = b.add(b.mul(slot, cx.i(QK_DIM // 4)), b.lshr(tid, cx.i(2)))
                    cx.st_i32(r_kv, dword, quad)
            cx.put("kvnew", b.add(cx.i(s * KV_LORA), tid), kvn)
            with b.scf_if(pe_lane):
                x0, x1 = pes[s]
                c, sn = cs[s], sns[s]
                p0 = b.fsub(b.fmul(x0, c), b.fmul(x1, sn))
                p1 = b.fadd(b.fmul(x0, sn), b.fmul(x1, c))
                p0, p1 = cx.fp8_roundtrip(_sat(cx, cx.bf16_round(p0)), _sat(cx, cx.bf16_round(p1)))
                with b.scf_if(active):
                    quad = _quad(cx, cx.fp8_pair_bits(p0, p1), b.add(b.land(lane, cx.i(-2)), cx.i(1)))
                    with b.scf_if(b.cmp_eq(b.land(lane, cx.i(1)), cx.i(0))):
                        dword = b.add(b.mul(slot, cx.i(QK_DIM // 4)),
                                      b.add(cx.i(KV_LORA // 4), b.lshr(tid, cx.i(1))))  # fmt: skip
                        cx.st_i32(r_pe, dword, quad)
                cx.put2("penew", b.add(cx.i(s * PE_DIM), b.mul(tid, cx.i(2))), p0, p1)


def _draft_rows(cx: Ml4Ctx, vs, pes, cs, sns, g: Value, pe_lane: Value) -> None:
    """The draft kind's bf16 latent cache, in vLLM's draft-layer rounding: the
    projection rounded to bf16 (its GEMM's output), bf16(RMSNorm(kv_c) *
    g_kv), the rope key rotated with bf16-rounded products
    (``MistralRotaryEmbedding.rotate_``); 576 bf16 a slot, published as the
    other kinds publish their rows."""
    b = cx.b
    S, tid, lane = cx.spec.rows, cx.tid, cx.lane
    r_kv, r_pe = cx.rsrc("kv_cache"), cx.rsrc("pe_cache")
    r = cx.bf16_round
    vs = [r(v) for v in vs]
    ssq = cx.block_sums([b.fmul(v, v) for v in vs])
    even = b.cmp_eq(b.land(lane, cx.i(1)), cx.i(0))
    for s in range(S):
        slot = cx.row_slot(s)
        active = b.cmp_ge(slot, cx.i(0))
        row = b.mul(slot, cx.i(QK_DIM // 2))
        rstd = b.rsqrt(b.fadd(b.fmul(ssq[s], cx.f(1.0 / KV_LORA)), cx.f(EPS)))
        value = r(b.fmul(b.fmul(vs[s], rstd), g))
        word = cx.bf16_pair(value, cx.xshfl(value, 1))
        with b.scf_if(b.land(active, even)):
            cx.st_i32(r_kv, b.add(row, b.lshr(tid, cx.i(1))), word)
        cx.put("kvnew", b.add(cx.i(s * KV_LORA), tid), value)
        with b.scf_if(pe_lane):
            x0, x1 = r(pes[s][0]), r(pes[s][1])
            c, sn = cs[s], sns[s]
            p0 = r(b.fsub(r(b.fmul(x0, c)), r(b.fmul(x1, sn))))
            p1 = r(b.fadd(r(b.fmul(x1, c)), r(b.fmul(x0, sn))))
            with b.scf_if(active):
                cx.st_i32(r_pe, b.add(row, b.add(cx.i(KV_LORA // 2), tid)), cx.bf16_pair(p0, p1))
            cx.put2("penew", b.add(cx.i(s * PE_DIM), b.mul(tid, cx.i(2))), p0, p1)


def _index_keys(cx: Ml4Ctx, cs, sns, pos, rope: bool) -> None:
    b = cx.b
    S, tid, lane = cx.spec.rows, cx.tid, cx.lane
    r_gik = cx.rsrc_at(cx.table_addr(T_G_INDEX_K))
    r_bik = cx.rsrc_at(cx.table_addr(T_B_INDEX_K))
    r_ic = cx.rsrc("indices")
    live = b.cmp_lt(tid, cx.i(INDEX_DIM))
    half = b.mul(b.smin(tid, cx.i(INDEX_DIM // 2 - 1)), cx.i(2))
    one = b.smin(tid, cx.i(INDEX_DIM - 1))
    got = cx.poll(
        [("index_k", b.add(cx.i(s * INDEX_DIM), one), 1) for s in range(S)]
        + [("index_k", b.add(cx.i(s * INDEX_DIM), half), 2) for s in range(S)]
    )
    iks = [b.bitcast(w[0], F32) for w in got[:S]]
    pairs = [(b.bitcast(w[0], F32), b.bitcast(w[1], F32)) for w in got[S:]]
    zero = cx.f(0.0)
    means = [
        b.fmul(m, cx.f(1.0 / INDEX_DIM))
        for m in cx.block_sums([b.select(live, ik, zero) for ik in iks])
    ]
    cents = [b.select(live, b.fsub(ik, m), zero) for ik, m in zip(iks, means)]
    rstds = [
        b.rsqrt(b.fadd(b.fmul(q, cx.f(1.0 / INDEX_DIM)), cx.f(1.0e-6)))
        for q in cx.block_sums([b.fmul(c, c) for c in cents])
    ]
    page_size = cx.spec.index_page_size
    for s in range(S):
        active = b.cmp_ge(cx.row_slot(s), cx.i(0))
        with b.scf_if(b.cmp_lt(tid, cx.i(INDEX_DIM // 2))):
            i0 = b.mul(tid, cx.i(2))
            k0, k1 = pairs[s]
            mean, rstd = means[s], rstds[s]

            def norm(k: Value, i: Value) -> Value:
                y = b.fmul(b.fmul(b.fsub(k, mean), rstd), cx.ld_f32(r_gik, i))
                return b.fadd(y, cx.ld_f32(r_bik, i))

            v0 = norm(k0, i0)
            v1 = norm(k1, b.add(i0, cx.i(1)))
            if rope:
                c, sn = cs[s], sns[s]
                in_rope = b.cmp_lt(tid, cx.i(PE_DIM // 2))
                r0 = b.fsub(b.fmul(v0, c), b.fmul(v1, sn))
                r1 = b.fadd(b.fmul(v0, sn), b.fmul(v1, c))
                v0, v1 = b.select(in_rope, r0, v0), b.select(in_rope, r1, v1)
            v0, v1 = cx.bf16_round(v0), cx.bf16_round(v1)
            amax = b.fmax(cx.wave_max(b.fmax(b.fabs(v0), b.fabs(v1))), cx.f(1.0e-4))
            # exp2(ceil(log2(amax / 448))) read off the exponent field
            bits = b.bitcast(amax, I32)
            exponent = b.sub(b.land(b.lshr(bits, cx.i(23)), cx.i(0xFF)), cx.i(8))
            round_up = b.cmp_lt(cx.i(0x600000), b.land(bits, cx.i(0x7FFFFF)))
            exponent = b.add(exponent, b.select(round_up, cx.i(1), cx.i(0)))
            exponent = b.smax(b.smin(exponent, cx.i(254)), cx.i(1))
            qs = b.bitcast(b.shl(exponent, cx.i(23)), F32)
            inv = b.bitcast(b.shl(b.sub(cx.i(254), exponent), cx.i(23)), F32)
            q0 = _sat(cx, b.fmul(v0, inv))
            q1 = _sat(cx, b.fmul(v1, inv))
            page, row = cx.page_cell(s, pos[s])
            quad = _quad(cx, cx.fp8_pair_bits(q0, q1), b.add(b.land(lane, cx.i(-2)), cx.i(1)))
            page_base = b.mul(page, cx.i(page_size * INDEX_PAGE_ROW // 4))
            with b.scf_if(active):
                with b.scf_if(b.cmp_eq(b.land(lane, cx.i(1)), cx.i(0))):
                    tiled = b.add(
                        b.add(b.mul(b.lshr(row, cx.i(4)), cx.i(INDEX_TILE * INDEX_DIM)),
                              b.mul(b.lshr(i0, cx.i(4)), cx.i(INDEX_TILE * INDEX_TILE))),
                        b.add(b.mul(b.land(row, cx.i(15)), cx.i(INDEX_TILE)), b.land(i0, cx.i(15))),
                    )  # fmt: skip
                    cx.st_i32(r_ic, b.add(page_base, b.lshr(tiled, cx.i(2))), quad)
                with b.scf_if(b.cmp_eq(tid, cx.i(0))):
                    scale_dword = b.add(page_base, b.add(cx.i(page_size * (INDEX_DIM // 4)), row))
                    cx.st_f32(r_ic, scale_dword, qs)
            d0, d1 = cx.fp8_roundtrip(q0, q1)
            cx.put_bf("index_k_new", b.add(cx.i(s * INDEX_DIM), i0), [b.fmul(d0, qs), b.fmul(d1, qs)])
    cx.barrier()
    with b.scf_if(b.cmp_lt(tid, cx.i(S))):
        cx.put("index_ready", tid, cx.i(1))
