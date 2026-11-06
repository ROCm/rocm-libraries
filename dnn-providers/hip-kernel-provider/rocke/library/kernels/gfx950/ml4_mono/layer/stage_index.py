# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stages ``index_score`` and the selection tree: the fused sparse indexer.

``index_score`` task ``(s, tile)`` scores 64 index-cache positions of row
``s``: the SSMax-scaled, RoPE'd 32 x 128 index query against the E4M3 keys
of the paged index plane (this launch's own rows from ``index_k_new``) on
MFMA, ReLU, times the head weight, summed over heads, times the row scale.

The selection tree takes each row's exact top-2048 positions: level 0 ranks
one 16384-key segment of the score plane per task (an 8-bit radix select over
monotonic integer keys in LDS), each higher level re-ranks the concatenated
top lists of the level below; the final list goes to ``indices`` ascending by
position (ties: the lower position), zero-padded, then ``indices_ready``.
"""

from __future__ import annotations

from typing import Optional

from rocke.core.ir import BF16, CPOL_SC1, F32, I32, Value, VectorType
from rocke.helpers import device_ops as dops
from rocke.helpers.task_schedule import task_index

from .ctx import THREADS, WAVES, Ml4Ctx
from .gemv import fp8_to_bf16x8

INDEX_DIM = 128
INDEX_HEADS = 32
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
KEYS_PER_TASK = 64
INDEX_PAGE_ROW = INDEX_DIM + 4
PE_DIM = 64
TOPK = 2048
SEG = 16384
RADIX_BINS = 256
POLL_MAX = 12
# word offsets of the index SSMax scale and bias rows in ``g_q``
P_ISS_S = 2 * 6144 + 128
P_ISS_B = P_ISS_S + 64


# ------------------------------------------------------------------ tree shape
def seg(cap: int) -> int:
    return min(cap, SEG)


def fan(cap: int) -> int:
    return seg(cap) // TOPK


def lists(cap: int, level: int) -> int:
    n = (cap + seg(cap) - 1) // seg(cap)
    for _ in range(level):
        n = (n + fan(cap) - 1) // fan(cap)
    return n


def depth(cap: int) -> int:
    level = 0
    while lists(cap, level) > 1:
        level += 1
    return level


def stage_name(level: int, d: int) -> str:
    return "index_select" if level == d else f"index_sel{level}"


def _cdiv(cx: Ml4Ctx, a: Value, b_: int) -> Value:
    b = cx.b
    return b.div(b.add(a, cx.i(b_ - 1)), cx.i(b_))


def tree_lists(cx: Ml4Ctx, bound: Value, level: int, sg: int, fn: int) -> Value:
    n = _cdiv(cx, cx.b.smax(bound, cx.i(0)), sg)
    for _ in range(level):
        n = _cdiv(cx, n, fn)
    return n


def tree_inputs(cx: Ml4Ctx, bound: Value, level: int, j: Value, sg: int, fn: int) -> Value:
    b = cx.b
    if level == 0:
        return b.smax(b.smin(b.sub(bound, b.mul(j, cx.i(sg))), cx.i(sg)), cx.i(0))
    below = tree_lists(cx, bound, level - 1, sg, fn)
    first = b.mul(j, cx.i(fn))
    count = b.smin(b.sub(below, first), cx.i(fn))
    last = b.sub(b.add(first, count), cx.i(1))
    valid = b.smin(tree_inputs(cx, bound, level - 1, last, sg, fn), cx.i(TOPK))
    return b.add(b.mul(b.smax(b.sub(count, cx.i(1)), cx.i(0)), cx.i(TOPK)), valid)


def tagged_indices(cx: Ml4Ctx) -> bool:
    """sel_tagged applies: level 0 has an output plane (the tree has a level
    above it) where a row with a single list can carry its tagged indices."""
    return cx.spec.sel_tagged and cx.spec.with_indexer and depth(cx.spec.index_capacity) >= 1


def single_list(cx: Ml4Ctx, bound: Value) -> Value:
    """Row of ``bound`` positions selects in one level-0 list (final there)."""
    cap = cx.spec.index_capacity
    return cx.b.cmp_ge(cx.i(1), tree_lists(cx, bound, 0, seg(cap), fan(cap)))


def tagged_index_word(cx: Ml4Ctx, s: Value, pos: Value) -> Value:
    """Pair index, in ``isel0``'s identity plane, of selected position ``pos``
    of single-list row ``s`` (the plane's list-0 slot of that row)."""
    b = cx.b
    slots = lists(cx.spec.index_capacity, 0)
    ident = cx.spec.rows * slots * TOPK
    return b.add(b.add(b.mul(b.mul(s, cx.i(slots)), cx.i(TOPK)), pos), cx.i(ident))


def tree_count(cx: Ml4Ctx, bound: Value, level: int, sg: int, fn: int) -> Value:
    b = cx.b
    if level == 0:
        return b.smax(tree_lists(cx, bound, 0, sg, fn), cx.i(1))
    more = b.cmp_lt(cx.i(1), tree_lists(cx, bound, level - 1, sg, fn))
    return b.select(more, tree_lists(cx, bound, level, sg, fn), cx.i(0))


def order_key(cx: Ml4Ctx, bits: Value) -> Value:
    """An f32's bits as a key whose unsigned order is the float order."""
    b = cx.b
    return b.select(b.cmp_ge(bits, cx.i(0)), b.xor(bits, cx.i(-(2**31))), b.xor(bits, cx.i(-1)))


def _ugt(cx: Ml4Ctx, a: Value, c: Value) -> Value:
    b = cx.b
    return b.cmp_lt(b.xor(c, cx.i(-(2**31))), b.xor(a, cx.i(-(2**31))))


# ------------------------------------------------------------------ scoring
def emit_index_score(cx: Ml4Ctx) -> None:
    S, G = cx.spec.rows, cx.spec.grid
    if G % S:
        cx.tasks("index_score", lambda tt: _score_task(cx, tt))
        return
    # row-interleaved tasks: every task of a CTA scores the same row, so the
    # CTA stages that row's query once and then scores its tiles in turn
    b = cx.b
    tasks = cx.table.stage("index_score").tasks
    tt0 = task_index(b, cx.table, "index_score", cx.bid, 0)
    s = b.mod(tt0, cx.i(S))
    tile0 = b.div(tt0, cx.i(S))
    pos = cx.row_position(s)
    bound = b.add(pos, cx.i(1))
    cx.barrier()
    live0 = b.land(b.cmp_lt(tt0, cx.i(tasks)), b.cmp_lt(b.mul(tile0, cx.i(KEYS_PER_TASK)), bound))
    with b.scf_if(live0):
        n_k = (tasks + G - 1) // G
        # score_prefetch: the first tiles' block-table lookups and cached keys
        # need no query, so they load while the query lands
        pre = [_tile_loads(cx, s, b.add(tile0, cx.i(k * (G // S))), bound)
               for k in range(min(SCORE_PREFETCH, n_k))] if cx.spec.score_prefetch else []  # fmt: skip
        # score_hoist: the request's first position once per CTA (three
        # dependent loads), not once per tile
        first_pos = _first_position(cx, s) if cx.spec.score_hoist else None
        _stage_index_query(cx, s, pos)
        cx.barrier()
        held = None
        for k in range(n_k):
            tt = b.add(tt0, cx.i(k * G))
            tile = b.add(tile0, cx.i(k * (G // S)))
            live = b.land(b.cmp_lt(tt, cx.i(tasks)), b.cmp_lt(b.mul(tile, cx.i(KEYS_PER_TASK)), bound))
            holds = _holds_new(cx, s, tile, first_pos) if _lazy_newk(cx) else None
            with b.scf_if(live):
                if holds is not None:
                    # the CTA's tiles ascend: stage the new keys before the
                    # first one that holds any (the barrier below covers it)
                    with b.scf_if(holds if held is None else b.land(holds, b.cmp_eq(held, cx.i(0)))):
                        _stage_new_keys(cx)
                cx.barrier()
                cx.stamp("index_score", tt, 0)
                _score_tile(cx, s, tile, pos, bound, tt, pre[k] if k < len(pre) else None, first_pos)
                cx.stamp("index_score", tt, 4)
            if holds is not None:
                held = b.select(holds, cx.i(1), cx.i(0))


def _lazy_newk(cx: Ml4Ctx) -> bool:
    return cx.spec.score_lazy_newk and cx.spec.score_newk_lds


def _stage_new_keys(cx: Ml4Ctx) -> None:
    """This launch's index keys (every row, bf16 pairs) into LDS: polled once
    per CTA, where every tile used to poll them per 32-dim group."""
    b = cx.b
    n_words = cx.spec.rows * INDEX_DIM // 2
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_words))):
        kw = cx.poll([("index_k_new", cx.tid, 1)])[0][0]
        cx.lds_st(b.add(cx.tid, cx.i(cx.keys + NEWK_LDS)), kw)


def _first_position(cx: Ml4Ctx, s: Value) -> Value:
    """Position of the first verify row of row ``s``'s request."""
    b = cx.b
    width = cx.spec.index_request_width or cx.spec.rows
    return cx.row_position(b.mul(b.div(s, cx.i(width)), cx.i(width)))


def _holds_new(cx: Ml4Ctx, s: Value, tile: Value, first_position: Optional[Value] = None) -> Value:
    """Tile ``tile`` of row ``s`` covers one of this launch's own rows (its
    request's positions from the first verify row on)."""
    b = cx.b
    if first_position is None:
        first_position = _first_position(cx, s)
    return b.cmp_gt(b.mul(b.add(tile, cx.i(1)), cx.i(KEYS_PER_TASK)), first_position)


def _stage_index_query(cx: Ml4Ctx, s: Value, pos: Value) -> None:
    """Head weights of row ``s`` into LDS ``keys``, its SSMax-scaled, RoPE'd
    32 x 128 index query (bf16 pairs) into ``xs``; without ``score_lazy_newk``
    also this launch's index keys, after the cache task's ``index_ready``."""
    b = cx.b
    if not _lazy_newk(cx):
        cx.poll([("index_ready", s, 1)])
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(INDEX_HEADS))):
        w = cx.poll([("index_w", b.add(b.mul(s, cx.i(INDEX_HEADS)), cx.tid), 1)])[0][0]
        cx.lds_st(b.add(cx.tid, cx.i(cx.keys)), w)
    if cx.spec.score_newk_lds and not _lazy_newk(cx):
        _stage_new_keys(cx)
    n_pairs = (INDEX_Q_ROWS // 2) // THREADS
    q_pairs = [b.add(cx.tid, cx.i(i * THREADS)) for i in range(n_pairs)]
    q_words = cx.poll([("index_q", b.add(b.mul(s, cx.i(INDEX_Q_ROWS // 2)), p), 1) for p in q_pairs])
    rope = cx.spec.kind in ("rope", "rope_shared")
    for q_pair, q_word in zip(q_pairs, q_words):
        q_elem = b.mul(q_pair, cx.i(2))
        kq = b.land(q_elem, cx.i(INDEX_DIM - 1))
        q0, q1 = cx.bf2_f32(q_word[0])
        iss = cx.ssmax(pos, P_ISS_S, P_ISS_B, b.lshr(q_elem, cx.i(7)))
        q0, q1 = cx.bf16_round(b.fmul(q0, iss)), cx.bf16_round(b.fmul(q1, iss))
        if rope:
            ri = b.add(b.mul(pos, cx.i(PE_DIM // 2)), b.lshr(b.smin(kq, cx.i(PE_DIM - 1)), cx.i(1)))
            c = cx.ld_rope("rope_cos", ri)
            sn = cx.ld_rope("rope_sin", ri)
            in_rope = b.cmp_lt(kq, cx.i(PE_DIM))
            r0 = b.fsub(b.fmul(q0, c), b.fmul(q1, sn))
            r1 = b.fadd(b.fmul(q0, sn), b.fmul(q1, c))
            q0, q1 = b.select(in_rope, r0, q0), b.select(in_rope, r1, q1)
        cx.lds_st(b.add(q_pair, cx.i(cx.xs)), cx.bf16_pair(q0, q1))


def _score_task(cx: Ml4Ctx, tt: Value) -> None:
    b = cx.b
    S = cx.spec.rows
    s = b.mod(tt, cx.i(S))
    tile = b.div(tt, cx.i(S))
    pos = cx.row_position(s)
    bound = b.add(pos, cx.i(1))
    with b.scf_if(b.cmp_lt(b.mul(tile, cx.i(KEYS_PER_TASK)), bound)):
        _stage_index_query(cx, s, pos)
        if _lazy_newk(cx):
            with b.scf_if(_holds_new(cx, s, tile)):
                _stage_new_keys(cx)
        cx.barrier()
        _score_tile(cx, s, tile, pos, bound, tt)


# score_prefetch: tiles per CTA whose cached keys load before the query wait
SCORE_PREFETCH = 2
# score_newk_lds: LDS word (past the head weights at keys + 0 ..) of the
# staged index keys of this launch's rows
NEWK_LDS = 64


def _tile_loads(cx: Ml4Ctx, s: Value, tile: Value, bound: Value):
    """The query-independent loads of a score tile: (local key, row scale,
    the four 32-dim groups of cached E4M3 key words)."""
    b = cx.b
    lane16 = b.land(cx.lane, cx.i(15))
    key_group = b.lshr(cx.wave, cx.i(1))
    local_key = b.add(b.add(b.mul(tile, cx.i(KEYS_PER_TASK)), b.mul(key_group, cx.i(16))), lane16)
    safe_key = b.smin(local_key, b.sub(bound, cx.i(1)))
    page, row = cx.page_cell(s, safe_key)
    r_ic = cx.rsrc("indices")
    page_base = b.mul(page, cx.i(cx.spec.index_page_size * INDEX_PAGE_ROW // 4))
    kv_scale = cx.ld_f32(r_ic, b.add(page_base, b.add(cx.i(cx.spec.index_page_size * (INDEX_DIM // 4)), row)))
    words = []
    for k32 in range(INDEX_DIM // 32):
        k = b.add(cx.i(k32 * 32), b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(8)))
        tiled = b.add(
            b.add(b.mul(b.lshr(row, cx.i(4)), cx.i(INDEX_TILE * INDEX_DIM)),
                  b.mul(b.lshr(k, cx.i(4)), cx.i(INDEX_TILE * INDEX_TILE))),
            b.add(b.mul(b.land(row, cx.i(15)), cx.i(INDEX_TILE)), b.land(k, cx.i(15))),
        )  # fmt: skip
        words.append(b.buffer_load_vN(r_ic, b.mul(b.add(page_base, b.lshr(tiled, cx.i(2))), cx.i(4)),
                                      cx.i(0), I32, 2))  # fmt: skip
    return local_key, kv_scale, words


def _score_tile(cx: Ml4Ctx, s: Value, tile: Value, pos: Value, bound: Value, tt: Value, pre=None,
                first_position: Optional[Value] = None) -> None:  # fmt: skip
    """Scores of index positions ``64 tile ..`` of row ``s`` (query staged);
    ``pre``: the tile's ``_tile_loads``, already issued; ``first_position``:
    the request's first position, if the caller has it."""
    b = cx.b
    S = cx.spec.rows
    cap = cx.spec.index_capacity
    width = cx.spec.index_request_width or S
    request_start = b.mul(b.div(s, cx.i(width)), cx.i(width))
    if first_position is None:
        first_position = cx.row_position(request_start)
    cx.stamp("index_score", tt, 2)
    lane16 = b.land(cx.lane, cx.i(15))
    head_group = b.land(cx.wave, cx.i(1))
    local_key, kv_scale, words_k = pre if pre is not None else _tile_loads(cx, s, tile, bound)
    key_pos = local_key
    safe_key = b.smin(key_pos, b.sub(bound, cx.i(1)))
    head = b.add(b.mul(head_group, cx.i(16)), lane16)
    is_new = b.cmp_ge(safe_key, first_position)
    sn_row = b.select(is_new, b.sub(b.add(request_start, safe_key), first_position), cx.i(0))
    frag = b.vec_pack([cx.f(0.0)] * 4, F32)
    for k32 in range(INDEX_DIM // 32):
        k = b.add(cx.i(k32 * 32), b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(8)))
        qw = b.lshr(b.add(b.mul(head, cx.i(INDEX_DIM)), k), cx.i(1))
        qv = b.bitcast(b.vec_pack([cx.lds_ld(b.add(qw, cx.i(cx.xs + j))) for j in range(4)], I32),
                       VectorType(BF16, 8))  # fmt: skip
        words = words_k[k32]
        kv_cache = fp8_to_bf16x8(cx, b.vec_extract(words, 0), b.vec_extract(words, 1))
        # this launch's own rows: the cache task's mailbox (rows of others
        # poll row 0's, which is always written, and discard it)
        nwords = [b.lshr(b.add(b.add(b.mul(sn_row, cx.i(INDEX_DIM)), k), cx.i(2 * j)), cx.i(1)) for j in range(4)]
        if cx.spec.score_newk_lds:
            new = [cx.lds_ld(b.add(w, cx.i(cx.keys + NEWK_LDS))) for w in nwords]
        else:
            new = [v[0] for v in cx.poll([("index_k_new", w, 1) for w in nwords])]
        kv_new = b.bitcast(b.vec_pack(new, I32), VectorType(BF16, 8))
        kv_bits = b.select(is_new, b.bitcast(kv_new, VectorType(I32, 4)), b.bitcast(kv_cache, VectorType(I32, 4)))
        frag = b.mfma_f32_16x16x32_bf16(qv, b.bitcast(kv_bits, VectorType(BF16, 8)), frag)
    partial = cx.f(0.0)
    for e in range(4):
        index_head = b.add(b.add(b.mul(head_group, cx.i(16)), b.mul(b.lshr(cx.lane, cx.i(4)), cx.i(4))), cx.i(e))
        weight = cx.lds_ldf(b.add(index_head, cx.i(cx.keys)))
        partial = b.fadd(partial, b.fmul(b.fmax(b.vec_extract(frag, e), cx.f(0.0)), weight))
    partial = b.fmul(partial, b.select(is_new, cx.f(1.0), kv_scale))
    partial = cx.xred(partial, 16, "add")
    partial = cx.xred(partial, 32, "add")
    with b.scf_if(b.cmp_lt(cx.lane, cx.i(16))):
        cx.lds_st(b.add(b.add(b.mul(cx.wave, cx.i(16)), cx.lane), cx.i(cx.red)), partial)
    cx.barrier()
    cond = b.land(b.land(b.cmp_eq(head_group, cx.i(0)), b.cmp_lt(cx.lane, cx.i(16))), b.cmp_lt(key_pos, bound))
    with b.scf_if(cond):
        score = b.fadd(cx.lds_ldf(b.add(b.add(b.mul(cx.wave, cx.i(16)), cx.lane), cx.i(cx.red))),
                       cx.lds_ldf(b.add(b.add(b.mul(b.add(cx.wave, cx.i(1)), cx.i(16)), cx.lane), cx.i(cx.red))))  # fmt: skip
        cx.put("index_scores", b.add(b.mul(s, cx.i(cap)), local_key), score)


# ------------------------------------------------------------------ selection
def emit_index_select(cx: Ml4Ctx, name: str) -> None:
    cap = cx.spec.index_capacity
    d = depth(cap)
    level = d if name == "index_select" else int(name[len("index_sel"):])
    if stage_name(level, d) != name:
        raise ValueError(f"{name}: not a level of a depth-{d} tree")
    cx.tasks(name, lambda tt: _select_task(cx, tt, level, d))


def _select_digit(cx: Ml4Ctx, shift: int, prefix: Value, remain: Value):
    b = cx.b
    digit = b.sub(cx.i(255), b.smin(cx.tid, cx.i(255)))
    low = b.cmp_lt(cx.tid, cx.i(RADIX_BINS))
    count = b.select(low, cx.lds_ld(b.add(digit, cx.i(cx.keys))), cx.i(0))
    inclusive = cx.wave_inclusive_scan(count)
    total = b.readlane(inclusive, cx.i(63))
    with b.scf_if(b.land(b.cmp_lt(cx.wave, cx.i(4)), b.cmp_eq(cx.lane, cx.i(63)))):
        cx.lds_st(b.add(cx.wave, cx.i(cx.keys + 256)), total)
    cx.barrier()
    before = cx.i(0)
    for w in range(4):
        before = b.add(before, b.select(b.cmp_lt(cx.i(w), cx.wave), cx.lds_ld(cx.i(cx.keys + 256 + w)), cx.i(0)))
    above = b.sub(b.add(before, inclusive), count)
    hit = b.land(b.land(low, b.cmp_lt(above, remain)), b.cmp_ge(b.add(above, count), remain))
    cx.barrier()
    with b.scf_if(hit):
        cx.lds_st(cx.i(cx.keys + 256), b.lor(prefix, b.shl(digit, cx.i(shift))))
        cx.lds_st(cx.i(cx.keys + 257), b.sub(remain, above))
    cx.barrier()
    result = cx.lds_ld(cx.i(cx.keys + 256)), cx.lds_ld(cx.i(cx.keys + 257))
    cx.barrier()
    return result


def _select_digit_wide(cx: Ml4Ctx, hist: int, bins: int, shift: int, prefix: Value, remain: Value):
    """Radix step over a ``bins``-bin histogram at LDS word ``hist``: the digit
    holding the ``remain``-th largest key, and how many of it are still to
    take. Thread ``t`` scans digits ``bins - 1 - (t * per + j)``, descending."""
    b = cx.b
    per = max(1, bins // THREADS)
    counts, valid = [], []
    for j in range(per):
        k = b.add(b.mul(cx.tid, cx.i(per)), cx.i(j))
        ok = b.cmp_lt(k, cx.i(bins))
        d = b.sub(cx.i(bins - 1), b.smin(k, cx.i(bins - 1)))
        counts.append((d, b.select(ok, cx.lds_ld(b.add(d, cx.i(hist))), cx.i(0))))
        valid.append(ok)
    local = cx.i(0)
    for _, cnt in counts:
        local = b.add(local, cnt)
    inclusive = cx.wave_inclusive_scan(local)
    with b.scf_if(b.cmp_eq(cx.lane, cx.i(63))):
        cx.lds_st(b.add(cx.wave, cx.i(cx.keys + 256)), inclusive)
    cx.barrier()
    above = cx.i(0)
    for w in range(8):
        above = b.add(above, b.select(b.cmp_lt(cx.i(w), cx.wave), cx.lds_ld(cx.i(cx.keys + 256 + w)), cx.i(0)))
    above = b.sub(b.add(above, inclusive), local)
    for (d, cnt), ok in zip(counts, valid):
        hit = b.land(b.land(ok, b.cmp_lt(above, remain)), b.cmp_ge(b.add(above, cnt), remain))
        with b.scf_if(hit):
            cx.lds_st(cx.i(cx.keys + 264), b.lor(prefix, b.shl(d, cx.i(shift))))
            cx.lds_st(cx.i(cx.keys + 265), b.sub(remain, above))
        above = b.add(above, cnt)
    cx.barrier()
    return cx.lds_ld(cx.i(cx.keys + 264)), cx.lds_ld(cx.i(cx.keys + 265))


def _scan_flags(cx: Ml4Ctx, flags):
    """Thread-major exclusive offsets of ``flags`` and their CTA total."""
    b = cx.b
    local = cx.i(0)
    offsets = []
    for flag in flags:
        offsets.append(local)
        local = b.add(local, b.select(flag, cx.i(1), cx.i(0)))
    inclusive = cx.wave_inclusive_scan(local)
    total_w = b.readlane(inclusive, cx.i(63))
    with b.scf_if(b.cmp_eq(cx.lane, cx.i(63))):
        cx.lds_st(b.add(cx.wave, cx.i(cx.keys + 256)), total_w)
    cx.barrier()
    before, total = cx.i(0), cx.i(0)
    for w in range(8):
        cnt = cx.lds_ld(cx.i(cx.keys + 256 + w))
        before = b.add(before, b.select(b.cmp_lt(cx.i(w), cx.wave), cnt, cx.i(0)))
        total = b.add(total, cnt)
    base = b.sub(b.add(before, inclusive), local)
    return [b.add(base, off) for off in offsets], total


def _scan_counts(cx: Ml4Ctx, vals):
    """Thread-major exclusive prefix and CTA total of each per-thread count."""
    b = cx.b
    incl = [cx.wave_inclusive_scan(v) for v in vals]
    with b.scf_if(b.cmp_eq(cx.lane, cx.i(63))):
        for q, x in enumerate(incl):
            cx.lds_st(b.add(cx.wave, cx.i(cx.keys + 256 + 16 * q)), x)
    cx.barrier()
    out = []
    for q, (v, x) in enumerate(zip(vals, incl)):
        before, total = cx.i(0), cx.i(0)
        for w in range(8):
            cnt = cx.lds_ld(cx.i(cx.keys + 256 + 16 * q + w))
            before = b.add(before, b.select(b.cmp_lt(cx.i(w), cx.wave), cnt, cx.i(0)))
            total = b.add(total, cnt)
        out.append((b.sub(b.add(before, x), v), total))
    return out


# keys per thread per trip of the lean output loops
SEL_UNROLL = 4


def _emit_lean(cx: Ml4Ctx, n_in: Value, last: Value, threshold: Value, kw, ident_of, publish) -> None:
    """Level-0 output: thread ``t`` owns keys ``[t * per, (t + 1) * per)``,
    ``per`` sized to this list so every thread holds some. A counting pass,
    the CTA scan of the counts, then an emitting pass; both recompute their
    flags from the staged keys, so no per-item flags or offsets stay live."""
    b = cx.b
    one, zero = cx.i(1), cx.i(0)
    per = b.readfirstlane(b.lshr(b.add(n_in, cx.i(THREADS - 1)), cx.i(THREADS.bit_length() - 1)))
    trips = b.lshr(b.add(per, cx.i(SEL_UNROLL - 1)), cx.i(SEL_UNROLL.bit_length() - 1))
    first = b.mul(cx.tid, per)

    def trip(q: Value):
        out = []
        for u in range(SEL_UNROLL):
            dd = b.add(b.mul(q, cx.i(SEL_UNROLL)), cx.i(u))
            i = b.add(first, dd)
            live = b.land(b.cmp_lt(dd, per), b.cmp_lt(i, n_in))
            k = cx.lds_ld(kw(b.smin(i, last)))
            out.append((i, k, b.land(live, _ugt(cx, k, threshold)), b.land(live, b.cmp_eq(k, threshold))))
        return out

    def inc(v: Value, flag: Value) -> Value:
        return b.add(v, b.select(flag, one, zero))

    count = b.scf_for_iter(zero, trips, one, [("sel_cg", zero), ("sel_ce", zero)], iv_name="sel_c")
    with count as (q, (cg, ce)):
        for _, _, g, e in trip(q):
            cg, ce = inc(cg, g), inc(ce, e)
        b.scf_yield(cg, ce)
    (og0, out_gt), (oe0, _) = _scan_counts(cx, list(count.results))
    need_eq = b.sub(b.smin(cx.i(TOPK), n_in), out_gt)
    emit = b.scf_for_iter(zero, trips, one, [("sel_og", og0), ("sel_oe", oe0)], iv_name="sel_e")
    with emit as (q, (og, oe)):
        for i, k, g, e in trip(q):
            win = b.lor(g, b.land(e, b.cmp_lt(oe, need_eq)))
            pos = b.add(og, b.smin(oe, need_eq))
            with b.scf_if(win):
                publish(pos, k, ident_of(i))
            og, oe = inc(og, g), inc(oe, e)
        b.scf_yield(og, oe)


# sel_range: threshold-bucket keys ranked in one wave; more (ties) take the
# radix passes. LDS words past the select scratch at keys + 256 ..: the waves'
# key maxima and minima, and the threshold
RANK_CAP = 64
_KMAX, _KMIN, _THR = 320, 328, 336
RANGE_BINS = 4096


def _range_threshold(cx: Ml4Ctx, n_in: Value, last: Value, items: int, kw, rhist: int,
                     fallback, stamp) -> Value:  # fmt: skip
    """The TOPK-th largest staged key: one select over RANGE_BINS buckets
    ``(key - kmin) >> sh``, ``sh`` sized to the keys' span (order keys are
    about logarithmic in the score, so a few outliers widen it little), then
    the threshold bucket's keys ranked within a wave. A bucket of more than
    RANK_CAP keys (ties) takes ``fallback()``, the radix passes."""
    b = cx.b
    cand = rhist + RANGE_BINS
    if cand + RANK_CAP > cx.keys:
        raise ValueError("the range histogram does not fit past the staged keys")
    zero, one, ones = cx.i(0), cx.i(1), cx.i(-1)

    def key_at(bi: int):
        i = b.add(cx.tid, cx.i(bi * THREADS))
        return b.cmp_lt(i, n_in), cx.lds_ld(kw(b.smin(i, last)))

    # pass A: the keys' range; the bucket histogram zeroed meanwhile
    kmin, kmax = ones, zero
    for bi in range(items):
        live, key = key_at(bi)
        lo, hi = b.select(live, key, ones), b.select(live, key, zero)
        kmax = b.select(_ugt(cx, hi, kmax), hi, kmax)
        kmin = b.select(_ugt(cx, kmin, lo), lo, kmin)
    for q in range(RANGE_BINS // THREADS):
        cx.lds_st(b.add(cx.tid, cx.i(rhist + q * THREADS)), zero)
    wmax = dops.wave_umax_dpp(b, kmax)
    wmin = b.xor(dops.wave_umax_dpp(b, b.xor(kmin, ones)), ones)
    with b.scf_if(b.cmp_eq(cx.lane, zero)):
        cx.lds_st(b.add(cx.wave, cx.i(cx.keys + _KMAX)), wmax)
        cx.lds_st(b.add(cx.wave, cx.i(cx.keys + _KMIN)), wmin)
    cx.barrier()
    gmin, gmax = ones, zero
    for w in range(WAVES):
        hi = cx.lds_ld(cx.i(cx.keys + _KMAX + w))
        lo = cx.lds_ld(cx.i(cx.keys + _KMIN + w))
        gmax = b.select(_ugt(cx, hi, gmax), hi, gmax)
        gmin = b.select(_ugt(cx, gmin, lo), lo, gmin)
    gmin = b.readfirstlane(gmin)
    span = b.sub(b.readfirstlane(gmax), gmin)
    # sh = max(0, bit_length(span) - 12): the buckets cover the span
    sh = zero
    for k in range(RANGE_BINS.bit_length() - 1, 32):
        sh = b.add(sh, b.select(b.cmp_ne(b.lshr(span, cx.i(k)), zero), one, zero))

    def bucket(key: Value) -> Value:
        return b.lshr(b.sub(key, gmin), sh)

    # pass B: the bucket histogram, then the bucket holding the threshold
    for bi in range(items):
        live, key = key_at(bi)
        with b.scf_if(live):
            cx.lds_add(b.add(bucket(key), cx.i(rhist)), one)
    cx.barrier()
    stamp(2)
    tb, rem = _select_digit_wide(cx, rhist, RANGE_BINS, 0, zero, b.smin(cx.i(TOPK), n_in))
    # passes C, D: the bucket's keys, compacted (order does not matter)
    cnt = zero
    for bi in range(items):
        live, key = key_at(bi)
        cnt = b.add(cnt, b.select(b.land(live, b.cmp_eq(bucket(key), tb)), one, zero))
    ((run, total),) = _scan_counts(cx, [cnt])
    c = b.readfirstlane(total)
    for bi in range(items):
        live, key = key_at(bi)
        inb = b.land(live, b.cmp_eq(bucket(key), tb))
        with b.scf_if(b.land(inb, b.cmp_lt(run, cx.i(RANK_CAP)))):
            cx.lds_st(b.add(run, cx.i(cand)), key)
        run = b.add(run, b.select(inb, one, zero))
    cx.barrier()
    fast = b.cmp_le(c, cx.i(RANK_CAP))
    with b.scf_if(fast):
        # every wave ranks the candidates (lane l holds candidate l)
        k = cx.lds_ld(b.add(cx.lane, cx.i(cand)))
        gt_n, eq_n = zero, zero
        for jj in range(RANK_CAP):
            kj = b.readlane(k, cx.i(jj))
            vj = b.cmp_lt(cx.i(jj), c)
            gt_n = b.add(gt_n, b.select(b.land(vj, _ugt(cx, kj, k)), one, zero))
            eq_n = b.add(eq_n, b.select(b.land(vj, b.cmp_eq(kj, k)), one, zero))
        hit = b.land(b.cmp_lt(cx.lane, c), b.land(b.cmp_lt(gt_n, rem), b.cmp_le(rem, b.add(gt_n, eq_n))))
        thr = dops.wave_umax_dpp(b, b.select(hit, k, zero))
        with b.scf_if(b.cmp_eq(cx.tid, zero)):
            cx.lds_st(cx.i(cx.keys + _THR), thr)
    with b.scf_if(b.lnot(fast)):
        p = fallback()
        with b.scf_if(b.cmp_eq(cx.tid, zero)):
            cx.lds_st(cx.i(cx.keys + _THR), p)
    cx.barrier()
    stamp(3)
    return cx.lds_ld(cx.i(cx.keys + _THR))


def _hist_zero(cx: Ml4Ctx) -> None:
    b = cx.b
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(RADIX_BINS))):
        cx.lds_st(b.add(cx.tid, cx.i(cx.keys)), cx.i(0))


def _select_task(cx: Ml4Ctx, tt: Value, level: int, d: int) -> None:
    b = cx.b
    S = cx.spec.rows
    cap = cx.spec.index_capacity
    sg, fn = seg(cap), fan(cap)
    slots = lists(cap, level)
    below = lists(cap, level - 1) if level else 0
    items = (sg + THREADS - 1) // THREADS
    if slots == 1:
        s, j = tt, cx.i(0)
    else:
        s = b.div(tt, cx.i(slots))
        j = b.sub(tt, b.mul(s, cx.i(slots)))
    src = f"isel{level - 1}" if level else "index_scores"
    dst = f"isel{level}" if level < d else None
    # a level's region: the score plane, then the identity plane
    src_ident_pairs = S * below * TOPK
    dst_ident_pairs = S * slots * TOPK
    bound = b.add(cx.row_position(s), cx.i(1))
    with b.scf_if(b.cmp_lt(j, tree_count(cx, bound, level, sg, fn))):
        n_in = tree_inputs(cx, bound, level, j, sg, fn)
        at_final = None if level == d else b.cmp_ge(cx.i(1), tree_lists(cx, bound, level, sg, fn))

        def entry_pair(i: Value) -> Value:
            sub = b.div(i, cx.i(TOPK))
            return b.add(b.mul(b.add(b.add(b.mul(s, cx.i(below)), b.mul(j, cx.i(fn))), sub), cx.i(TOPK)),
                         b.sub(i, b.mul(sub, cx.i(TOPK))))  # fmt: skip

        # radix passes as (shift, bits), high digits first; the 12-bit first
        # pass spreads keys of similar magnitude over 16x the bins of an 8-bit
        # one, where most of them would land on a few LDS counters
        wide = cx.spec.select_radix == "12_12_8"
        passes = [(20, 12), (8, 12), (0, 8)] if wide else [(24, 8), (16, 8), (8, 8), (0, 8)]
        pad = cx.spec.sel_pad
        hist = cx.xs + sg + (sg // 32 if pad else 0) if wide else cx.keys
        if wide and hist + 4096 > cx.keys:
            raise ValueError("the 12-bit radix histogram does not fit past the staged keys")

        def kw(i: Value) -> Value:
            """LDS word of staged key ``i``."""
            return b.add(b.add(i, b.lshr(i, cx.i(5))) if pad else i, cx.i(cx.xs))

        def hist_zero(bins: int) -> None:
            for q in range((bins + THREADS - 1) // THREADS):
                k = b.add(cx.tid, cx.i(q * THREADS))
                with b.scf_if(b.cmp_lt(k, cx.i(bins))):
                    cx.lds_st(b.add(k, cx.i(hist)), cx.i(0))

        def digit_of(key: Value, shift: int, bits: int) -> Value:
            d = b.lshr(key, cx.i(shift))
            return d if shift + bits == 32 else b.land(d, cx.i((1 << bits) - 1))

        def select(shift: int, bits: int, prefix: Value, remain: Value):
            if wide:
                return _select_digit_wide(cx, hist, 1 << bits, shift, prefix, remain)
            return _select_digit(cx, shift, prefix, remain)

        def rescan_passes(prefix: Value, remain: Value, mask: int, todo):
            """Radix passes ``todo`` over the staged keys, each histogram built
            by a rescan of the keys that match the digits found so far."""
            for shift, bits in todo:
                hist_zero(1 << bits)
                cx.barrier()
                m = mask - (1 << 32) if mask >= 1 << 31 else mask
                for bi in range(items):
                    i = b.add(cx.tid, cx.i(bi * THREADS))
                    with b.scf_if(b.cmp_lt(i, n_in)):
                        key = cx.lds_ld(kw(i))
                        with b.scf_if(b.cmp_eq(b.land(key, cx.i(m)), prefix)):
                            cx.lds_add(b.add(digit_of(key, shift, bits), cx.i(hist)), cx.i(1))
                cx.barrier()
                prefix, remain = select(shift, bits, prefix, remain)
                mask |= ((1 << bits) - 1) << shift
            return prefix, remain

        # sel_range (level 0): one select over buckets sized to the keys' own
        # range, then the keys of the threshold bucket ranked directly
        by_range = cx.spec.sel_range and level == 0
        shift0, bits0 = passes[0]
        if not by_range:
            hist_zero(1 << bits0)
            cx.barrier()
        last = b.sub(n_in, cx.i(1))
        for chunk in range(0, items, POLL_MAX):
            ids = [b.add(cx.tid, cx.i(i * THREADS)) for i in range(chunk, min(chunk + POLL_MAX, items))]
            if level == 0:
                base = b.add(b.mul(s, cx.i(cap)), b.mul(j, cx.i(sg)))
                specs = [(src, b.add(base, b.smin(i, last)), 1) for i in ids]
            else:
                specs = [(src, entry_pair(b.smin(i, last)), 1) for i in ids]
            with b.scf_if(b.cmp_lt(cx.i(chunk * THREADS), n_in)):
                got = cx.poll(specs)
                for i, v in zip(ids, got):
                    with b.scf_if(b.cmp_lt(i, n_in)):
                        key = order_key(cx, v[0]) if level == 0 else v[0]
                        cx.lds_st(kw(i), key)
                        if not by_range:
                            cx.lds_add(b.add(digit_of(key, shift0, bits0), cx.i(hist)), cx.i(1))
        cx.barrier()
        if by_range:
            rhist = cx.xs + sg + (sg // 32 if pad else 0)
            threshold = _range_threshold(
                cx, n_in, last, items, kw, rhist, lambda: rescan_passes(cx.i(0), b.smin(cx.i(TOPK), n_in), 0, passes)[0],
                lambda col: cx.stamp(stage_name(level, d), tt, col),
            )  # fmt: skip
        else:
            cx.stamp(stage_name(level, d), tt, 2)
            prefix, remain = select(shift0, bits0, cx.i(0), b.smin(cx.i(TOPK), n_in))
            prefix, remain = rescan_passes(prefix, remain, ((1 << bits0) - 1) << shift0, passes[1:])
            cx.stamp(stage_name(level, d), tt, 3)
            threshold = prefix
        r_out = cx.mb_rsrc("indices")

        def store_index(at: Value, v: Value) -> None:
            b.buffer_store_cp(v, r_out, b.mul(at, cx.i(4)), cx.i(0), aux=CPOL_SC1)

        tagged = level == 0 and level < d and tagged_indices(cx)

        def publish(pos: Value, key: Value, ident: Value) -> None:
            """Winner ``pos`` (ascending position order) of this list."""
            at = b.add(b.mul(s, cx.i(TOPK)), pos)
            if level == d:
                store_index(at, ident)
                return
            with b.scf_if(at_final):
                store_index(at, ident)
                if tagged:
                    # split polls these directly (no ready flag, no reload)
                    cx.put(dst, tagged_index_word(cx, s, pos), ident)
            with b.scf_if(b.lnot(at_final)):
                at_d = b.add(b.mul(b.add(b.mul(s, cx.i(slots)), j), cx.i(TOPK)), pos)
                cx.put(dst, at_d, key)
                cx.put(dst, b.add(at_d, cx.i(dst_ident_pairs)), ident)

        def emit_outputs() -> None:
            """The winners in position order (the final list's to ``indices``,
            zero-padded, then ``indices_ready``; others to the next level)."""
            if cx.spec.sel_lean and level == 0:
                _emit_lean(cx, n_in, last, threshold, kw, lambda i: b.add(b.mul(j, cx.i(sg)), i), publish)
            else:
                idx = [b.add(b.mul(cx.tid, cx.i(items)), cx.i(dd)) for dd in range(items)]
                bits = [cx.lds_ld(kw(b.smin(i, last))) for i in idx]
                live = [b.cmp_lt(i, n_in) for i in idx]
                gt = [b.land(lv, _ugt(cx, k, threshold)) for lv, k in zip(live, bits)]
                eq = [b.land(lv, b.cmp_eq(k, threshold)) for lv, k in zip(live, bits)]
                if level == 0:
                    idents = [b.add(b.mul(j, cx.i(sg)), i) for i in idx]
                else:
                    idents = [v[0] for v in cx.poll([(src, b.add(cx.i(src_ident_pairs), entry_pair(b.smin(i, last))), 1)
                                                      for i in idx])]  # fmt: skip
                gt_off, out_gt = _scan_flags(cx, gt)
                cx.barrier()
                eq_off, _ = _scan_flags(cx, eq)
                cx.barrier()
                need_eq = b.sub(b.smin(cx.i(TOPK), n_in), out_gt)
                for dd in range(items):
                    win = b.lor(gt[dd], b.land(eq[dd], b.cmp_lt(eq_off[dd], need_eq)))
                    pos = b.add(gt_off[dd], b.smin(eq_off[dd], need_eq))
                    with b.scf_if(win):
                        publish(pos, bits[dd], idents[dd])
            final = cx.i(1) if level == d else b.select(at_final, cx.i(1), cx.i(0))
            is_final = b.cmp_eq(final, cx.i(1))
            with b.scf_if(is_final):
                # an explicit zero past the valid count, which every reader relies on
                for bi in range((TOPK + THREADS - 1) // THREADS):
                    dd = b.add(cx.tid, cx.i(bi * THREADS))
                    with b.scf_if(b.cmp_ge(dd, bound)):
                        store_index(b.add(b.mul(s, cx.i(TOPK)), dd), cx.i(0))
            if cx.spec.sel_release == "waitcnt":
                # every value above went out with agent-coherent (sc1) stores: only
                # their completion has to precede the ready flag
                b.s_waitcnt(vmcnt=0)
            else:
                b.fence(scope="agent", ordering="release")
            cx.barrier()
            with b.scf_if(b.land(is_final, b.cmp_eq(cx.tid, cx.i(0)))):
                cx.put("indices_ready", s, cx.i(1))

        emit_outputs()
