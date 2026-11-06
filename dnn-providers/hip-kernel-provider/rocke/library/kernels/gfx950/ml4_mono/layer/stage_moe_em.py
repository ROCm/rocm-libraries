# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Expert-major MoE (``ug_em``, ``moe_em``).

The work items are the distinct experts of this launch's routing: the routed
experts the rows picked, in ascending id, and the shared expert. Every CTA
that routes derives the same item list. An item's group is a run of CTAs
consecutive in XCD-blocked order (CTA b on XCD b % 8), so one expert's tiles
sit on one or two XCDs. ``em_shared_ctas`` gives the shared expert a fixed
group, the last positions: it needs no routing, so its first up/gate loads go
out at stage entry while the routed groups still wait on the router (as
``ug_shared_first`` does for the slot-major ``ug``). The routed experts split
the other positions. Without it the shared expert is one more item.

``ug_em``: a group splits its expert's 8-column up/gate chunks. A chunk is
loaded once for every row that routes to the expert: the scaled MFMA's B
columns are the verify rows, where the slot-major ``ug`` loads a chunk once
per (row, slot). Each row's column is the same dot product, reduced over the
waves in the same order, so ``mid`` (and ``sel`` / ``prob``) get the same
bits in the same layout, and the slot-major ``down`` can follow unchanged.

``moe_em``: the group then splits the expert's 16-row down tiles. Its down
needs only its own expert's ``mid``, so it does not wait for the other
experts' up/gate; with ``em_dn_pre`` the first tile's weights stream during
that wait. Each wave multiplies whole tiles (no cross-wave reduction) and
writes the route-weighted partial of every row that routes to the expert to
``dn_part``. The ``down`` stage (``emit_down_em``) sums a row's five partials
in slot order, then the TP sum and the residual into ``x_out``. The sum over
a row's experts is re-associated (per-expert partials, then their sum,
instead of one accumulator over all slots), so ``x_out`` matches the
slot-major build to rounding, not bit for bit.
"""

from __future__ import annotations

from typing import Callable, Optional

from rocke.core.ir import F32, I32, Value, VectorType

from .ctx import HIDDEN, MOE_SLOTS, WAVES, XQ_S_STRIDE, XQ_X_STRIDE, Ml4Ctx
from .gemv import mma_units, mxfp4_scale_pair, n_sel, scale_byte, unit_native_mxfp4, zero4
from .stage_moe import (
    SHARED_EXPERT,
    T_SHARED,
    TOP_K,
    UG8,
    UG_RED_OFF,
    _expert_rsrc,
    _silu_mul,
    _uniform_f32,
    _uniform_i32,
    dn_route,
    dn_tile,
    expert_addr,
    load_bias,
    quant_mxfp8,
    routed_addr,
    st_mxfp8,
    stage_xq,
)
from .stage_uv import peer_reduce

# LDS words past the routing (``keys[s * MOE_SLOTS + q]``): the item tables
EM_ITEMS_OFF = 64  # items[k]: the expert of item k
EM_SLOT_OFF = 96  # islot[k * S + s]: row s's slot of item k, -1 if none
EM_WEIGHT_OFF = 192  # iw[k * S + s]: row s's route weight of item k
EM_XCDS = 8
# timeline rows: routed item k's CTAs stamp k * 16 + index, the shared group's
# CTAs this + index
EM_SHARED_STAMP = 256


def _popcount(cx: Ml4Ctx, x: Value) -> Value:
    b = cx.b
    x = b.sub(x, b.land(b.lshr(x, cx.i(1)), cx.i(0x55555555)))
    x = b.add(b.land(x, cx.i(0x33333333)), b.land(b.lshr(x, cx.i(2)), cx.i(0x33333333)))
    x = b.land(b.add(x, b.lshr(x, cx.i(4))), cx.i(0x0F0F0F0F))
    return b.lshr(b.mul(x, cx.i(0x01010101)), cx.i(24))


def em_items(cx: Ml4Ctx) -> Value:
    """The item tables into LDS (after ``dn_route``); returns the item count
    (uniform): the distinct routed experts, plus 1 for the shared one."""
    b = cx.b
    S = cx.spec.rows
    n_max = S * TOP_K + 1
    # the routed experts as a 128-bit set, every lane alike
    words = [cx.i(0) for _ in range(4)]
    for s in range(S):
        for q in range(1, MOE_SLOTS):
            e = cx.lds_ld(cx.i(cx.keys + s * MOE_SLOTS + q))
            hi = b.lshr(e, cx.i(5))
            bit = b.shl(cx.i(1), b.land(e, cx.i(31)))
            for j in range(4):
                words[j] = b.lor(words[j], b.select(b.cmp_eq(hi, cx.i(j)), bit, cx.i(0)))
    words = [b.readfirstlane(w) for w in words]
    counts = [_popcount(cx, w) for w in words]
    n_routed = b.add(b.add(counts[0], counts[1]), b.add(counts[2], counts[3]))
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_max * S))):
        cx.lds_st(b.add(cx.tid, cx.i(cx.keys + EM_SLOT_OFF)), cx.i(-1))
    cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * MOE_SLOTS))):
        s = b.div(cx.tid, cx.i(MOE_SLOTS))
        q = b.mod(cx.tid, cx.i(MOE_SLOTS))
        e = cx.lds_ld(b.add(cx.tid, cx.i(cx.keys)))
        hi = b.lshr(e, cx.i(5))
        low = b.sub(b.shl(cx.i(1), b.land(e, cx.i(31))), cx.i(1))
        below = cx.i(0)
        for j in range(4):
            whole = b.cmp_lt(cx.i(j), hi)
            part = _popcount(cx, b.land(words[j], low))
            below = b.add(below, b.select(whole, counts[j], b.select(b.cmp_eq(hi, cx.i(j)), part, cx.i(0))))
        rank = b.select(b.cmp_eq(q, cx.i(0)), n_routed, below)
        cx.lds_st(b.add(rank, cx.i(cx.keys + EM_ITEMS_OFF)), e)
        at = b.add(b.mul(rank, cx.i(S)), s)
        cx.lds_st(b.add(at, cx.i(cx.keys + EM_SLOT_OFF)), q)
        cx.lds_st(b.add(at, cx.i(cx.keys + EM_WEIGHT_OFF)), cx.lds_ld(b.add(cx.tid, cx.i(cx.dnw))))
    cx.barrier()
    return b.add(n_routed, cx.i(1))


def em_pos(cx: Ml4Ctx) -> Value:
    """This CTA's position in XCD-blocked order (uniform)."""
    b = cx.b
    per = cx.spec.grid // EM_XCDS
    return b.add(b.mul(b.land(cx.bid, cx.i(EM_XCDS - 1)), cx.i(per)), b.lshr(cx.bid, cx.i(EM_XCDS.bit_length() - 1)))


def em_group(cx: Ml4Ctx, pos: Value, n_items: Value, span: int) -> tuple[Value, Value, Value]:
    """(item, index in its group, group size), all uniform: item k's group is
    the positions [k span / n, (k + 1) span / n), for ``pos < span``."""
    b = cx.b
    # the k with floor(k span / n) <= pos < floor((k + 1) span / n)
    k = b.div(b.sub(b.mul(b.add(pos, cx.i(1)), n_items), cx.i(1)), cx.i(span))
    first = b.div(b.mul(k, cx.i(span)), n_items)
    last = b.div(b.mul(b.add(k, cx.i(1)), cx.i(span)), n_items)
    return b.readfirstlane(k), b.readfirstlane(b.sub(pos, first)), b.readfirstlane(b.sub(last, first))


def _kth_bit(cx: Ml4Ctx, w: Value, r: Value) -> Value:
    """Position of set bit ``r`` (from 0, low first) of ``w``: binary search on
    popcounts of the low half."""
    b = cx.b
    pos = cx.i(0)
    for half in (16, 8, 4, 2, 1):
        cnt = _popcount(cx, b.land(b.lshr(w, pos), cx.i((1 << half) - 1)))
        up = b.cmp_ge(r, cnt)
        r = b.select(up, b.sub(r, cnt), r)
        pos = b.select(up, b.add(pos, cx.i(half)), pos)
    return pos


def _scalar_items(cx: Ml4Ctx):
    """The item list without LDS tables or barriers (after ``dn_route`` and
    its barrier), all uniform: (routed item count, ``kth(k)`` -> the expert
    of routed item k, ``rows_of(e)`` -> the rows' slots (-1: none) and route
    weights for expert e; the shared expert is every row's slot 0)."""
    b = cx.b
    S = cx.spec.rows
    keys = [[_uniform_i32(cx, cx.i(cx.keys + s * MOE_SLOTS + q)) for q in range(MOE_SLOTS)] for s in range(S)]
    words = [cx.i(0) for _ in range(4)]
    for s in range(S):
        for q in range(1, MOE_SLOTS):
            e = keys[s][q]
            bit = b.shl(cx.i(1), b.land(e, cx.i(31)))
            hi = b.lshr(e, cx.i(5))
            for j in range(4):
                words[j] = b.lor(words[j], b.select(b.cmp_eq(hi, cx.i(j)), bit, cx.i(0)))
    counts = [_popcount(cx, w) for w in words]
    ends = [counts[0]]
    for c in counts[1:]:
        ends.append(b.add(ends[-1], c))

    def kth(k: Value) -> Value:
        j, before, w = cx.i(0), cx.i(0), words[0]
        for n in range(1, 4):
            past = b.cmp_ge(k, ends[n - 1])
            j = b.select(past, cx.i(n), j)
            before = b.select(past, ends[n - 1], before)
            w = b.select(past, words[n], w)
        return b.add(b.mul(j, cx.i(32)), _kth_bit(cx, w, b.sub(k, before)))

    def rows_of(e: Value):
        qs, ws = [], []
        for s in range(S):
            q, w = cx.i(-1), cx.f(0.0)
            for slot in range(MOE_SLOTS):
                hit = b.cmp_eq(keys[s][slot], e)
                q = b.select(hit, cx.i(slot), q)
                w = b.select(hit, _uniform_f32(cx, cx.i(cx.dnw + s * MOE_SLOTS + slot)), w)
            qs.append(q)
            ws.append(w)
        return qs, ws

    return ends[-1], kth, rows_of


def _routed_item(cx: Ml4Ctx, k: Value):
    """Item ``k``'s expert, and per row its slot (-1: none) and route weight
    (uniform; a weight is meaningful only where the slot is)."""
    b = cx.b
    S = cx.spec.rows

    def at(off: int, s: int) -> Value:
        return b.add(b.add(b.mul(k, cx.i(S)), cx.i(s)), cx.i(cx.keys + off))

    e = _uniform_i32(cx, b.add(k, cx.i(cx.keys + EM_ITEMS_OFF)))
    return e, [_uniform_i32(cx, at(EM_SLOT_OFF, s)) for s in range(S)], [_uniform_f32(cx, at(EM_WEIGHT_OFF, s)) for s in range(S)]


def emit_ug_em(cx: Ml4Ctx) -> None:
    b = cx.b
    S, grid = cx.spec.rows, cx.spec.grid
    if S < 2:
        raise NotImplementedError("the one-row up/gate schedule is not ported yet")
    g_sh = cx.spec.em_shared_ctas
    n_max = S * TOP_K
    if g_sh < 0 or (g_sh and grid - g_sh < n_max):
        raise ValueError(f"em_shared_ctas={g_sh}: the routed experts need at least {n_max} CTAs")
    cx.barrier()
    pos = em_pos(cx)

    def staging() -> None:
        stage_xq(cx)
        cx.barrier()

    def routed(span: int, shared_item: bool) -> None:
        dn_route(cx, load_bias(cx))
        cx.barrier()
        # em_early: the first chunks' loads go out before the activation is
        # staged (polled), as the shared group's do
        early = cx.spec.em_early
        if not early:
            staging()
        if cx.spec.em_scalar:
            n_r, kth, rows_of = _scalar_items(cx)
            k, idx, size = em_group(cx, pos, b.add(n_r, cx.i(1)) if shared_item else n_r, span)
            e = b.select(b.cmp_eq(k, n_r), cx.i(SHARED_EXPERT), kth(k)) if shared_item else kth(k)
            qs, ws = rows_of(e)
        else:
            n_items = em_items(cx)
            k, idx, size = em_group(cx, pos, n_items if shared_item else b.sub(n_items, cx.i(1)), span)
            e, qs, ws = _routed_item(cx, k)
        stamp = b.add(b.mul(k, cx.i(16)), b.smin(idx, cx.i(15)))
        _em_item(cx, stamp, idx, size, e, lambda: (qs, ws), span // (n_max + int(shared_item)),
                 staging if early else None)  # fmt: skip

    if not g_sh:
        routed(grid, True)
        return

    def shared() -> None:
        def prologue() -> None:
            if not cx.spec.moe_em:
                # the slot-major down reads this CTA's routing from LDS
                dn_route(cx, load_bias(cx))
                cx.barrier()
            staging()

        idx = b.readfirstlane(b.sub(pos, cx.i(grid - g_sh)))
        stamp = b.add(cx.i(EM_SHARED_STAMP), b.smin(idx, cx.i(15)))
        _em_item(cx, stamp, idx, cx.i(g_sh), cx.i(SHARED_EXPERT),
                 lambda: ([cx.i(0)] * S, [cx.f(1.0)] * S), g_sh, prologue)  # fmt: skip

    with b.scf_if_else(b.cmp_ge(pos, cx.i(grid - g_sh))) as (sh_ctx, rt_ctx):
        with sh_ctx:
            shared()
        with rt_ctx:
            routed(grid - g_sh, False)


def _em_item(cx: Ml4Ctx, stamp: Value, idx: Value, size: Value, e: Value,
             item: Callable[[], tuple[list, list]], min_group: int,
             prologue: Optional[Callable[[], None]] = None) -> None:  # fmt: skip
    """This CTA's share of one expert: up/gate chunks ``[idx C / size, (idx +
    1) C / size)`` and, under ``moe_em``, its down tiles. ``item()``: the
    rows' slots and weights for it, read after ``prologue`` (the shared
    group's activation staging), which runs once the first chunks' loads are
    out; ``min_group``: the smallest group size (sizes the static loops)."""
    b = cx.b
    S, inter = cx.spec.rows, cx.spec.inter
    ug_c = inter // UG8
    units_per_wave = (HIDDEN // 128) // WAVES
    w_bytes = 2 * inter * HIDDEN // 2
    s_bytes = 2 * inter * (HIDDEN // 32)
    tb = cx.spec.em_tiles
    ahead = min(cx.spec.em_prefetch, tb)
    max_tiles = -(-ug_c // min_group)
    if cx.xs + UG_RED_OFF + tb * 2048 > cx.keys or UG_RED_OFF < S * XQ_X_STRIDE:
        raise ValueError("em_tiles: the batched reduction does not fit beside the activation")
    c_lo = b.readfirstlane(b.div(b.mul(idx, cx.i(ug_c)), size))
    c_hi = b.readfirstlane(b.div(b.mul(b.add(idx, cx.i(1)), cx.i(ug_c)), size))
    shared = b.cmp_eq(e, cx.i(SHARED_EXPERT))
    aw = b.select(shared, cx.table_addr(T_SHARED), routed_addr(cx, "w_ug", e, w_bytes))
    as_ = b.select(shared, cx.table_addr(T_SHARED + 1), routed_addr(cx, "s_ug", e, s_bytes))
    sn = n_sel(cx, S)
    cx.stamp("ug", stamp, 0)

    def issue(n: int) -> dict:
        """Chunk ``c_lo + n``'s weight units (loads read zeros past ``c_hi``)."""
        c = b.add(c_lo, cx.i(n))
        live = b.cmp_lt(c, c_hi)
        rw = _expert_rsrc(cx, aw, w_bytes, live)
        rs_ = _expert_rsrc(cx, as_, s_bytes, live)
        w_rg = b.add(b.mul(b.lshr(b.land(cx.lane, cx.i(15)), cx.i(3)), cx.i(inter // 16)), b.lshr(c, cx.i(1)))
        w_ln = b.lor(b.land(cx.lane, cx.i(-16)),
                     b.add(b.mul(b.land(c, cx.i(1)), cx.i(8)), b.land(cx.lane, cx.i(7))))  # fmt: skip
        units = []
        pair_word = pair_byte = None
        for cc in range(units_per_wave):
            unit = b.add(b.mul(cx.wave, cx.i(units_per_wave)), cx.i(cc))
            if cc % 2 == 0:
                row = b.add(b.mul(w_rg, cx.i(16)), b.land(w_ln, cx.i(15)))
                col = b.add(b.mul(unit, cx.i(4)), b.lshr(w_ln, cx.i(4)))
                word, pair_byte = mxfp4_scale_pair(cx, row, col, HIDDEN // 32)
                pair_word = cx.ld_i32(rs_, word)
            shift = b.mul(b.add(pair_byte, cx.i((cc % 2) * 2)), cx.i(8))
            scale = scale_byte(cx, pair_word, shift, cx.spec.ug_scale_late)
            b_word = b.add(b.mul(sn, cx.i(XQ_X_STRIDE)), b.mul(unit, cx.i(32)))
            s_word = b.add(b.add(cx.i(8), b.mul(sn, cx.i(XQ_S_STRIDE))),
                           b.add(b.mul(unit, cx.i(4)), b.lshr(cx.lane, cx.i(4))))  # fmt: skip
            units.append(unit_native_mxfp4(cx, rw, rs_, w_rg, unit, HIDDEN, b_word, s_word,
                                           ln=w_ln, scale=scale))  # fmt: skip
        return dict(live=live, c=c, units=units)

    batches = [list(range(n, min(n + tb, max_tiles))) for n in range(0, max_tiles, tb)]
    pending = {n: issue(n) for n in batches[0]}
    if prologue is not None:
        prologue()
    qs, ws = item()
    def run(bi: int, batch: list[int]) -> None:
        nonlocal pending
        tiles = [pending.pop(n) if n in pending else issue(n) for n in batch]
        accs = [mma_units(cx, zero4(cx), t["units"]) for t in tiles]
        if bi + 1 < len(batches):
            # the next batch's first chunks stream during this one's reduction
            pending = {n: issue(n) for n in batches[bi + 1][:ahead]}
        _em_reduce_emit(cx, e, qs, ws, tiles, accs)
        cx.barrier()

    for bi, batch in enumerate(batches):
        if bi and bi + 1 == len(batches):
            # the static loop is sized for the smallest group: the last batch
            # is often all past this CTA's chunks (no reduction, no barriers)
            with b.scf_if(b.cmp_lt(b.add(c_lo, cx.i(batch[0])), c_hi)):
                run(bi, batch)
        else:
            run(bi, batch)
    cx.stamp("ug", stamp, 4)
    if cx.spec.moe_em:
        _em_down(cx, idx, size, e, qs, ws, min_group)


def _pick(cx: Ml4Ctx, j: Value, vals: list[Value]) -> Value:
    """``vals[j]`` for a lane-varying index ``j`` (a short select chain)."""
    b = cx.b
    out = vals[0]
    for n in range(1, len(vals)):
        out = b.select(b.cmp_eq(j, cx.i(n)), vals[n], out)
    return out


def _em_reduce_emit(cx: Ml4Ctx, e: Value, qs: list[Value], ws: list[Value], tiles: list[dict], accs) -> None:
    """The batch's chunks reduced over the waves (wave order, as ``ug``),
    then each routing row's SiLU(gate) * up pairs to ``mid`` and, with the
    expert's first chunk, its rows' ``sel`` / ``prob``."""
    b = cx.b
    S, inter = cx.spec.rows, cx.spec.inter
    n_t = len(tiles)
    base = b.mul(b.add(b.mul(cx.wave, cx.i(64)), cx.lane), cx.i(4))
    for n, acc in enumerate(accs):
        word = b.add(base, cx.i(cx.xs + UG_RED_OFF + n * 2048))
        cx.lds_st4(word, b.bitcast(b.vec_pack(list(acc), F32), VectorType(I32, 4)))
    cx.barrier()
    per_tile = 16 * S
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_t * per_tile))):
        j = b.div(cx.tid, cx.i(per_tile))
        col = b.div(b.mod(cx.tid, cx.i(per_tile)), cx.i(16))
        r = b.land(cx.tid, cx.i(15))
        lane_of = b.add(col, b.mul(b.lshr(r, cx.i(2)), cx.i(16)))
        red = b.add(b.mul(j, cx.i(2048)), cx.i(cx.xs + UG_RED_OFF))
        tot = cx.f(0.0)
        for w in range(WAVES):
            word = b.add(b.mul(b.add(cx.i(w * 64), lane_of), cx.i(4)), b.land(r, cx.i(3)))
            tot = b.fadd(tot, cx.lds_ldf(b.add(red, word)))
        cx.lds_st(b.add(cx.tid, cx.i(cx.outs)), tot)
    cx.barrier()
    cs = [t["c"] for t in tiles]
    lives = [b.select(t["live"], cx.i(1), cx.i(0)) for t in tiles]
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_t * S * 4))):
        j = b.div(cx.tid, cx.i(S * 4))
        s = b.mod(b.lshr(cx.tid, cx.i(2)), cx.i(S))
        r = b.mul(b.land(cx.tid, cx.i(3)), cx.i(2))
        c, live, q = _pick(cx, j, cs), _pick(cx, j, lives), _pick(cx, s, qs)
        with b.scf_if(b.land(b.cmp_ne(live, cx.i(0)), b.cmp_ge(q, cx.i(0)))):
            o = b.add(b.add(b.mul(j, cx.i(16 * S)), b.add(b.mul(s, cx.i(16)), r)), cx.i(cx.outs))
            g0, g1 = cx.lds_ldf(o), cx.lds_ldf(b.add(o, cx.i(1)))
            v0, v1 = cx.lds_ldf(b.add(o, cx.i(UG8))), cx.lds_ldf(b.add(o, cx.i(UG8 + 1)))
            at = b.add(b.add(b.mul(b.add(b.mul(s, cx.i(MOE_SLOTS)), q), cx.i(inter)),
                             b.mul(c, cx.i(UG8))), r)  # fmt: skip
            cx.put2("mid", at, _silu_mul(cx, g0, v0), _silu_mul(cx, g1, v1))
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_t * S))):
        j = b.div(cx.tid, cx.i(S))
        s = b.mod(cx.tid, cx.i(S))
        c, live, q = _pick(cx, j, cs), _pick(cx, j, lives), _pick(cx, s, qs)
        first = b.land(b.land(b.cmp_ne(live, cx.i(0)), b.cmp_eq(c, cx.i(0))), b.cmp_ge(q, cx.i(0)))
        with b.scf_if(first):
            p = b.add(b.mul(s, cx.i(MOE_SLOTS)), q)
            cx.put("sel", p, e)
            cx.put("prob", p, b.bitcast(_pick(cx, s, ws), I32))


def _em_down(cx: Ml4Ctx, idx: Value, size: Value, e: Value, qs: list[Value], ws: list[Value],
             min_group: int) -> None:  # fmt: skip
    """This CTA's share of the expert's 16-row down tiles: the expert's rows'
    ``mid`` re-quantized to MXFP8 in LDS, then per wave whole tiles (all of
    K), each row's route-weighted partial to ``dn_part``."""
    b = cx.b
    S, inter = cx.spec.rows, cx.spec.inter
    nbs = inter // 128
    n_units = inter // 128
    w_bytes = HIDDEN * inter // 2
    s_bytes = HIDDEN * (inter // 32)
    n_rg = HIDDEN // 16
    max_tiles = -(-n_rg // min_group)
    per_wave = -(-max_tiles // WAVES)
    lane16 = b.land(cx.lane, cx.i(15))
    # a row that routes to the expert (every item has one)
    anchor_s, anchor_q = cx.i(0), qs[0]
    for s in reversed(range(S)):
        has = b.cmp_ge(qs[s], cx.i(0))
        anchor_s = b.select(has, cx.i(s), anchor_s)
        anchor_q = b.select(has, qs[s], anchor_q)
    # mid MXFP8 blocks in LDS as ``down`` lays them out, block (s * MOE_SLOTS +
    # q) * nbs + j; one past them, zeroed, is the B data of the rows that skip
    # this expert (their scale is a staged block's, a valid E8M0)
    zero_blk = S * MOE_SLOTS * nbs
    if (zero_blk + 1) * 32 > cx.keys - cx.xs:
        raise ValueError("moe_em: the mid blocks do not fit in the activation LDS")
    # B column = verify row (lanes past the rows repeat the last); its slot
    # block, or the zero block, and its route weight (0: skips the expert)
    sn = n_sel(cx, S)
    q_sn = _pick(cx, sn, qs)
    has_sn = b.cmp_ge(q_sn, cx.i(0))
    coef = b.select(b.land(has_sn, b.cmp_lt(lane16, cx.i(S))), _pick(cx, sn, ws), cx.f(0.0))
    blk0 = b.mul(b.add(b.mul(sn, cx.i(MOE_SLOTS)), b.smax(q_sn, cx.i(0))), cx.i(nbs))
    anchor_blk0 = b.mul(b.add(b.mul(anchor_s, cx.i(MOE_SLOTS)), anchor_q), cx.i(nbs))
    wb = cx.rsrc_at(expert_addr(cx, "w_dn", e, w_bytes, 2))
    sb = cx.rsrc_at(expert_addr(cx, "s_dn", e, s_bytes, 3))
    r_lo = b.readfirstlane(b.div(b.mul(idx, cx.i(n_rg)), size))
    r_hi = b.readfirstlane(b.div(b.mul(b.add(idx, cx.i(1)), cx.i(n_rg)), size))

    def units_of(m: int):
        rg = b.add(r_lo, b.add(cx.wave, cx.i(m * WAVES)))
        live = b.cmp_lt(rg, r_hi)
        rgc = b.smin(rg, cx.i(n_rg - 1))
        out = []
        for u in range(n_units):
            blk = b.add(blk0, cx.i(u))
            b_blk = b.select(has_sn, blk, cx.i(zero_blk))
            s_blk = b.select(has_sn, blk, b.add(anchor_blk0, cx.i(u)))
            out.append(unit_native_mxfp4(cx, wb, sb, rgc, cx.i(u), inter, b.mul(b_blk, cx.i(32)),
                                         b.add(b.mul(s_blk, cx.i(4)), b.lshr(cx.lane, cx.i(4))),
                                         coef, late_scale=cx.spec.dn_scale_late))  # fmt: skip
        return live, rg, out

    # em_dn_pre: the first tile's weights stream during the mid wait
    cur = units_of(0) if cx.spec.em_dn_pre else None
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(32))):
        cx.lds_st(b.add(cx.tid, cx.i(cx.xs + zero_blk * 32)), cx.i(0))
    # stage this expert's rows' mid blocks: a row that skips it polls the
    # anchor row's blocks (ready with ours) and stores nothing
    per_row = -(-nbs // WAVES)
    specs, where = [], []
    for s in range(S):
        has = b.cmp_ge(qs[s], cx.i(0))
        ps = b.select(has, cx.i(s), anchor_s)
        pq = b.select(has, qs[s], anchor_q)
        for m in range(per_row):
            j = b.smin(b.add(cx.wave, cx.i(m * WAVES)), cx.i(nbs - 1))
            blk = b.add(b.mul(b.add(b.mul(ps, cx.i(MOE_SLOTS)), pq), cx.i(nbs)), j)
            specs.append(("mid", b.add(b.mul(blk, cx.i(128)), b.mul(cx.lane, cx.i(2))), 2))
            where.append((has, b.add(cx.wave, cx.i(m * WAVES)), blk))
    step = cx.spec.mid_chunk or len(specs)
    for c0 in range(0, len(specs), step):
        mids = cx.poll(specs[c0 : c0 + step])
        for (has, jv, blk), mid in zip(where[c0 : c0 + step], mids):
            with b.scf_if(b.land(has, b.cmp_lt(jv, cx.i(nbs)))):
                m0, m1, qsc = quant_mxfp8(cx, b.bitcast(mid[0], F32), b.bitcast(mid[1], F32))
                st_mxfp8(cx, b.add(b.mul(blk, cx.i(128)), b.mul(cx.lane, cx.i(2))), m0, m1)
                with b.scf_if(b.cmp_eq(lane16, cx.i(0))):
                    cx.lds_st(b.add(b.add(b.mul(blk, cx.i(4)), b.lshr(cx.lane, cx.i(4))), cx.i(cx.misc)), qsc)
    cx.barrier()
    if cur is None:
        cur = units_of(0)
    for m in range(per_wave):
        live, rg, units = cur
        if m + 1 < per_wave:
            cur = units_of(m + 1)
        acc = mma_units(cx, zero4(cx), units)
        # lane l holds hidden rows 4 (l >> 4) .. + 3 of column l & 15
        with b.scf_if(b.land(live, b.land(has_sn, b.cmp_lt(lane16, cx.i(S))))):
            h = b.add(b.mul(rg, cx.i(16)), b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(4)))
            p = b.add(b.mul(b.add(b.mul(sn, cx.i(MOE_SLOTS)), q_sn), cx.i(HIDDEN)), h)
            cx.put2("dn_part", p, acc[0], acc[1])
            cx.put2("dn_part", b.add(p, cx.i(2)), acc[2], acc[3])


def emit_down_em(cx: Ml4Ctx) -> None:
    cx.tasks("down", lambda t: _combine_task(cx, t))


def _combine_task(cx: Ml4Ctx, t: Value) -> None:
    """Task ``t``'s ``dn_tile`` hidden rows: each row's five partials (the
    shared expert, then slots 1-4) summed in that order, then the TP sum and
    the residual ``a`` into ``x_out`` (``peer_reduce``)."""
    b = cx.b
    S = cx.spec.rows
    tile = dn_tile(S)
    half = tile // 2
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * half))):
        s = b.div(cx.tid, cx.i(half))
        r = b.mul(b.mod(cx.tid, cx.i(half)), cx.i(2))
        h = b.add(b.mul(t, cx.i(tile)), r)
        got = cx.poll([("dn_part", b.add(b.mul(b.add(b.mul(s, cx.i(MOE_SLOTS)), cx.i(q)), cx.i(HIDDEN)), h), 2)
                       for q in range(MOE_SLOTS)])  # fmt: skip
        v0, v1 = cx.f(0.0), cx.f(0.0)
        for pair in got:
            v0 = b.fadd(v0, b.bitcast(pair[0], F32))
            v1 = b.fadd(v1, b.bitcast(pair[1], F32))
        o = b.add(b.mul(s, cx.i(tile)), r)
        cx.lds_st(b.add(o, cx.i(cx.outs)), v0)
        cx.lds_st(b.add(o, cx.i(cx.outs + 1)), v1)
    cx.barrier()
    r_x = cx.rsrc("x_out")

    def resid(s: Value, row: Value):
        w = cx.poll([("a", b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)), 1)])[0][0]
        return cx.bf2_f32(w)

    def store_x(s: Value, row: Value, v0: Value, v1: Value) -> None:
        cx.st_i32(r_x, b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)), cx.bf16_pair(v0, v1))

    peer_reduce(cx, t, tile, resid, store_x, "ffn")
