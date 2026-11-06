# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stages ``q_b`` and ``index_q``: the query projections of the normed input.

``q_b`` task ``t`` computes 16 of the ``heads * (nope + rope)`` query rows of
``q_an`` (FP8 GEMV, 8 waves split K) and publishes them with the ML4 SSMax
scale: nope rows to ``q_nope``, RoPE-rotated rope rows to ``q_pe`` (NoPE
kinds publish a zero ``q_pe``). With the indexer, a ``q_b`` task then reuses
its CTA for one 16-row index-query tile of ``iq_an`` (tile ``t``), and the
``index_q`` stage computes the remaining tiles on the CTAs after ``q_b``'s.
"""

from __future__ import annotations

from rocke.core.ir import Value

from .ctx import HIDDEN, WAVES, Ml4Ctx
from .gemv import emit_out, n_sel, reduce_rows, run_units, stage_x_pairs, unit_fp8x2
from .stage_qkv_a import INDEX_PROJ_TILES, index_projection_tile

TILE = 16
PE_DIM = 64
INDEX_Q_ROWS = 32 * 128
NKC = HIDDEN // 64
UNITS = NKC // (2 * WAVES)
# word offsets of the SSMax scale and bias rows in ``g_q``
P_SS_S = 2 * HIDDEN
P_SS_B = P_SS_S + 64
# rows of the parameter table (``table``)
T_W_INDEX_Q, T_S_INDEX_Q = 3, 4


def dims(cx: Ml4Ctx) -> tuple[int, int]:
    """(nope, rope) query dims per head of this kind."""
    nope = cx.spec.kind in ("nope", "nope_kvs")
    return (128, 0) if nope else (64, PE_DIM)


def n_qb(cx: Ml4Ctx) -> int:
    nope_dim, q_pe = dims(cx)
    return cx.spec.heads * (nope_dim + q_pe) // TILE


def _b_word(cx: Ml4Ctx, kc: Value) -> Value:
    b = cx.b
    col = b.mul(n_sel(cx, cx.spec.rows), cx.i(HIDDEN))
    return b.lshr(b.add(col, b.mul(kc, cx.i(64))), cx.i(1))


def _units(cx: Ml4Ctx, w: Value, s: Value, rg: Value):
    b = cx.b

    def unit(c: int):
        kc = b.mul(b.add(b.mul(cx.wave, cx.i(UNITS)), cx.i(c)), cx.i(2))
        return unit_fp8x2(cx, w, s, rg, kc, NKC, HIDDEN, _b_word(cx, kc))

    return unit


def emit_q_b(cx: Ml4Ctx) -> None:
    cx.tasks("q_b", lambda t: _q_b_task(cx, t))


def _q_b_task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    S, H = cx.spec.rows, cx.spec.heads
    nope_dim, q_pe = dims(cx)
    per_head = (nope_dim + q_pe) // TILE
    u_qb = _units(cx, cx.rsrc("w_q_b"), cx.rsrc("s_q_b"), t)
    early = cx.spec.with_indexer and cx.spec.index_q_first
    # iq_prefetch: the index-query tile's weights stream while iq_an lands,
    # and q_b's own after the index tiles (one set of tiles in flight either way)
    iq_pre = None
    if early and cx.spec.iq_prefetch:
        u_iq = _units(cx, cx.rsrc_at(cx.table_addr(T_W_INDEX_Q)), cx.rsrc_at(cx.table_addr(T_S_INDEX_Q)), t)
        iq_pre = [u_iq(c) for c in range(UNITS)]
        pre = None
    else:
        pre = [u_qb(c) for c in range(UNITS)]
    if cx.spec.with_indexer and cx.spec.idx_proj_on_qb:
        # the index key / head weights need only the layer input: computed
        # here while q_norm runs, which frees qkv_a's first CTAs for the
        # index-query tiles planned on them
        index_projection_tile(cx, t)

    def index_tiles() -> None:
        cx.barrier()
        stage_x_pairs(cx, "iq_an", S * HIDDEN)
        cx.barrier()
        _index_q_tile(cx, t, pre=iq_pre)
        first, on_free, extra = index_q_placement(cx)
        if extra > on_free and not cx.spec.iq_dual:
            # the extra tiles the free CTAs past q_b's could not take
            lo = leftover_q_b_task(cx)
            mine = b.land(b.cmp_ge(t, cx.i(lo)), b.cmp_lt(t, cx.i(lo + extra - on_free)))
            with b.scf_if(mine):
                cx.barrier()
                _index_q_tile(cx, b.add(t, cx.i(first + on_free - lo)), pre=None)

    if early:
        # index scoring waits on these, while q_b's consumers wait on the
        # top-k selection anyway
        index_tiles()
    if pre is None:
        pre = [u_qb(c) for c in range(UNITS)]
    cx.barrier()
    stage_x_pairs(cx, "q_an", S * HIDDEN)
    cx.barrier()
    acc = run_units(cx, u_qb, UNITS, UNITS, pre)
    reduce_rows(cx, 1, acc, emit_out(cx, TILE), S)
    cx.barrier()
    head = b.lshr(t, cx.i(per_head.bit_length() - 1))
    hoff = b.mul(b.land(t, cx.i(per_head - 1)), cx.i(TILE))
    if q_pe == 0:
        zero = cx.f(0.0)
        cond = b.land(b.cmp_eq(hoff, cx.i(0)), b.cmp_lt(cx.tid, cx.i(S * PE_DIM // 4)))
        with b.scf_if(cond):
            zs = b.lshr(cx.tid, cx.i(4))
            elem = b.add(b.mul(b.add(b.mul(zs, cx.i(H)), head), cx.i(PE_DIM)),
                         b.mul(b.land(cx.tid, cx.i(15)), cx.i(4)))  # fmt: skip
            cx.put_bf("q_pe", elem, [zero] * 4)
    with b.scf_if(b.cmp_lt(hoff, cx.i(nope_dim))):
        with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * TILE // 4))):
            s = b.lshr(cx.tid, cx.i(2))
            r = b.mul(b.land(cx.tid, cx.i(3)), cx.i(4))
            ss = cx.ssmax(cx.row_position(s), P_SS_S, P_SS_B, head)
            vals = []
            for j in range(4):
                v = cx.lds_ldf(b.add(b.add(b.mul(s, cx.i(TILE)), r), cx.i(cx.outs + j)))
                vals.append(cx.bf16_round(b.fmul(cx.bf16_round(v), ss)))
            elem = b.add(b.add(b.mul(b.add(b.mul(s, cx.i(H)), head), cx.i(nope_dim)), hoff), r)
            cx.put_bf("q_nope", elem, vals)
    if q_pe:
        with b.scf_if(b.cmp_ge(hoff, cx.i(nope_dim))):
            with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * TILE // 2))):
                s = b.lshr(cx.tid, cx.i(3))
                pr = b.land(cx.tid, cx.i(7))
                i = b.add(b.sub(hoff, cx.i(nope_dim)), b.mul(pr, cx.i(2)))
                o = b.add(b.mul(s, cx.i(TILE)), b.mul(pr, cx.i(2)))
                x0 = cx.bf16_round(cx.lds_ldf(b.add(o, cx.i(cx.outs))))
                x1 = cx.bf16_round(cx.lds_ldf(b.add(o, cx.i(cx.outs + 1))))
                pos = cx.row_position(s)
                ri = b.add(b.mul(pos, cx.i(PE_DIM // 2)), b.lshr(i, cx.i(1)))
                c = cx.ld_rope("rope_cos", ri)
                sn = cx.ld_rope("rope_sin", ri)
                ss = cx.ssmax(pos, P_SS_S, P_SS_B, head)
                p0 = cx.bf16_round(b.fsub(b.fmul(x0, c), b.fmul(x1, sn)))
                p1 = cx.bf16_round(b.fadd(b.fmul(x0, sn), b.fmul(x1, c)))
                elem = b.add(b.mul(b.add(b.mul(s, cx.i(H)), head), cx.i(PE_DIM)), i)
                cx.put_bf("q_pe", elem, [cx.bf16_round(b.fmul(p0, ss)), cx.bf16_round(b.fmul(p1, ss))])
    if cx.spec.with_indexer and not early:
        index_tiles()


def index_q_placement(cx: Ml4Ctx) -> tuple[int, int, int]:
    """(first extra tile, extras on the index_q CTAs, extras in all).

    The index-query tiles past the ``q_b`` tasks' own run on the stage's CTAs
    up to the end of the grid only: the stage's base follows q_b's CTAs, and
    the CTAs past the grid's end wrap onto the qkv_a / q_norm / cache CTAs,
    which would compute them last. The ones left over run, ahead of q_b, on
    the first q_b CTAs."""
    first = n_qb(cx)
    extra = INDEX_Q_ROWS // TILE - first
    base = cx.table.base["index_q"]
    if cx.spec.idx_proj_on_qb:
        # qkv_a's CTAs are free early then: the extra tiles run where the plan
        # puts them, the wrapped ones included, up to the cache CTA (index
        # scoring waits on the cache task's index keys)
        to_cache = (cx.table.base["cache"] - base) % cx.spec.grid
        return first, min(extra, to_cache), extra
    return first, min(extra, cx.spec.grid - base), extra


def leftover_q_b_task(cx: Ml4Ctx) -> int:
    """First q_b task that takes a leftover index-query tile (the ones with
    an index projection tile are skipped)."""
    return INDEX_PROJ_TILES if cx.spec.idx_proj_on_qb else 0


def emit_index_q(cx: Ml4Ctx) -> None:
    """The index-query tiles after the ``q_b`` tasks' own (see
    ``index_q_placement``)."""
    first, on_free, extra = index_q_placement(cx)
    # iq_dual: the tiles left over past the free CTAs ride, as a second tile
    # loaded together with the first, on the first index_q CTAs
    n_left = extra - on_free if cx.spec.iq_dual else 0
    b = cx.b

    def task(tt: Value) -> None:
        with b.scf_if(b.cmp_lt(tt, cx.i(on_free))):
            if n_left:
                with b.scf_if(b.cmp_lt(tt, cx.i(n_left))):
                    _extra_tiles(tt, [b.add(tt, cx.i(first)), b.add(tt, cx.i(first + on_free))])
                with b.scf_if(b.cmp_ge(tt, cx.i(n_left))):
                    _extra_tiles(tt, [b.add(tt, cx.i(first))])
            else:
                _extra_tiles(tt, [b.add(tt, cx.i(first))])

    def _extra_tiles(tt: Value, tiles) -> None:
        w, s = cx.rsrc_at(cx.table_addr(T_W_INDEX_Q)), cx.rsrc_at(cx.table_addr(T_S_INDEX_Q))
        pres = []
        for iq_t in tiles:
            unit = _units(cx, w, s, iq_t)
            pres.append([unit(c) for c in range(UNITS)])
        cx.barrier()
        stage_x_pairs(cx, "iq_an", cx.spec.rows * HIDDEN)
        cx.barrier()
        for iq_t, pre in zip(tiles, pres):
            _index_q_tile(cx, iq_t, pre=pre)

    cx.tasks("index_q", task)


def _index_q_tile(cx: Ml4Ctx, iq_t: Value, pre) -> None:
    """One 16-row index-query tile of the staged ``iq_an`` -> ``index_q``."""
    b = cx.b
    S = cx.spec.rows
    unit = _units(cx, cx.rsrc_at(cx.table_addr(T_W_INDEX_Q)),
                  cx.rsrc_at(cx.table_addr(T_S_INDEX_Q)), iq_t)  # fmt: skip
    acc = run_units(cx, unit, UNITS, UNITS, pre)
    reduce_rows(cx, 1, acc, emit_out(cx, TILE), S)
    cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * TILE // 4))):
        s = b.lshr(cx.tid, cx.i(2))
        r = b.mul(b.land(cx.tid, cx.i(3)), cx.i(4))
        o = b.add(b.mul(s, cx.i(TILE)), r)
        vals = [cx.lds_ldf(b.add(o, cx.i(cx.outs + j))) for j in range(4)]
        elem = b.add(b.add(b.mul(s, cx.i(INDEX_Q_ROWS)), b.mul(iq_t, cx.i(TILE))), r)
        cx.put_bf("index_q", elem, vals)
