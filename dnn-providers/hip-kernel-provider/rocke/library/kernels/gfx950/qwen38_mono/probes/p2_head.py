# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P2: the MXFP4 lm_head inside a persistent launch (DESIGN.md 8.2, P2).

Question: what does the true monokernel gain or pay by running the head
(``mx_decode_gemv``, bf16 activations, fused argmax or logits) as a STAGE of a
persistent launch -- head CTAs that issue their first weight panels before
they poll for the head input, behind a producer stage in the same launch --
instead of as its own launch after the kernel that produced its input (the
hybrid)?

Pieces:

* :func:`emit_head`: the head body of
  :func:`kernels.gfx950.mx_decode_gemv.build_mx_decode_gemv` as an emitter on a
  caller's ``IRBuilder`` (CTA index, x staging and the tie rule injectable).
  :func:`build_head_clone` wraps it with the original's parameters; its LLVM IR
  equals the original's byte for byte (``test_qwen38_probe_p2``), so every
  difference a P2 build shows comes from the knobs below, not from the fork.
* :func:`build_p2` (``P2Spec``): ONE launch of ``grid`` CTAs x ``threads``:
  ``prod_ctas`` producer CTAs (by default the spare CTAs past the head's
  ``ctas``) publish the head input ``x [m, K]`` (copied from ``x``, after
  ``delay`` x ``s_sleep 8`` to stand in for the last layer's drain), and the
  head runs on CTAs ``0 .. head.ctas - 1`` (first ``head.waves`` waves).
  ``xmode``: ``"pairs"`` (x as tagged bf16-pair mailbox words, ``sc1``; the
  poll result is the data), ``"bulk"`` (the mono bulk rule: plain stores,
  ``vmcnt(0)``, ``buffer_wbl2 sc1`` + one ready pair per producer; consumers
  poll the ready pairs, ``buffer_inv sc1``, then load x), or ``"none"`` (no
  producer: x staged from ``x`` as the standalone head does; isolates the
  grid / threads change). ``pre``: ``"before"`` issues the first ``dg`` load
  groups of the first panel, then waits for x (the head's ``pro="w_first"``
  order with the wait inside the x staging); ``"after"`` waits and stages x
  first. The launch tag is derived from the head's never-reset 64-bit arrival
  counter (``ws[0:8]``, read at entry: launch ``e`` reads a count in
  ``[e G, e G + G)`` because no head CTA arrives before it has consumed x),
  so graph replays need no host-written step word; logits builds add the
  arrival the fused fold already does.
* :func:`build_p2_producer`: the hybrid arm's producer as its own launch
  (same producer CTAs, plain stores of x for the next launch).

Mutants (must fail the GPU gate): ``mutant="stale_tag"`` polls x with the
previous launch's tag (reads last launch's x); ``"tie_high"`` breaks argmax
ties toward the higher column.

Python engine only (the head's converts and swaps, ``buffer_*_cp``,
``tagged_poll``): every build carries ``_pyonly`` and compiles with
``backend="python"``. Not imported by any served build.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from kernels.gfx950 import mx_decode_gemv as MX
from kernels.gfx950.mx_decode_gemv import (
    _CHUNK_K,
    _FUSED_HDR,
    _GROUP_Q,
    _LANES,
    _OOB,
    _TILE_N,
    CPOLS,
    MxDecodeGemvSpec,
    _atomic_inc64,
    _b_frag,
    _bcast4,
    _bf16_key,
    _ci,
    _transpose4,
)
from rocke.core.ir import (
    BF16,
    CPOL_SC1,
    F32,
    I16,
    I32,
    I64,
    I8,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as MB

__all__ = [
    "MUTANTS",
    "PRES",
    "PWAITS",
    "NANRULES",
    "PIMPLS",
    "PLOOPS",
    "XMODES",
    "HeadIO",
    "P2Spec",
    "build_head_clone",
    "build_p2",
    "build_p2_producer",
    "emit_head",
    "is_valid_p2",
    "p2_signature",
    "producer_signature",
    "xbox_bytes",
    "XBOX_DATA_OFF",
]

XMODES = ("pairs", "bulk", "none")
PRES = ("before", "after")
#: ``pairs`` wait shape: payload polls only, or wave 0 first spins on one pair
#: per producer then the CTA barriers (the ``pre_poll`` throttle); ``auto`` =
#: the mono's ``payload_wait`` rule (hint above 4 rows).
PWAITS = ("none", "hint", "auto")
MUTANTS = ("none", "stale_tag", "tie_high")
#: ``pairs`` payload poll batches: unrolled over ``rows_max`` (rows past ``m`` re-poll the last valid
#: item) or a runtime loop over ``m`` rows only
PLOOPS = ("static", "runtime")
#: x polls: ``asm`` = ``rocke.helpers.mailbox.poll`` (one opaque asm loop, ``vmcnt(0)`` per retry, the
#: mono's default) or ``ir`` = ``tile.tagged_poll``. HAZARD (P2, gfx950, llvm23 comgr): the ``ir`` loop's
#: retry block holds only ``asm sideeffect "s_nop 0"`` without a memory clobber, so LLVM may hoist (and
#: narrow) the loop's readonly buffer loads out of it: the bulk build's ready-pair poll compiled to one
#: load + an endless ``s_nop`` / branch loop and hung whenever the pair was not ready at the first read.
PIMPLS = ("asm", "ir")
#: argmax order on non-finite logits: ``mono`` (mx_decode_gemv / the mono's _argmax_fold: -NaN ranks
#: lowest; the hybrid's draft heads) or ``sampler`` (ATOM's greedy sampler, the hybrid's target heads:
#: any NaN ranks highest)
NANRULES = ("mono", "sampler")
#: x mailbox layout: ready pairs (bulk) in the first 4 KiB, then the data
XBOX_DATA_OFF = 4096
#: bounds of the x mailbox views (the asm poll ignores them; the largest xbox)
P2_XB_MAX = 4096 + 16 * 8192 * 4


def xbox_bytes(rows_max: int, k: int) -> int:
    """Bytes of the x mailbox (both modes fit: pairs take 4 B per element)."""
    return XBOX_DATA_OFF + rows_max * k * 4


# ---------------------------------------------------------------- head emitter


@dataclass
class HeadIO:
    """The head's parameter values (``xs`` only for ``act="mxfp8"``, ``ws`` only
    for ``epilogue="fused"`` or a logits arrival)."""

    xp: Value
    wp: Value
    sp: Value
    yp: Value
    m_rows: Value
    sxm: Value
    sym: Value
    rev: Value
    xsp: Optional[Value] = None
    wsp: Optional[Value] = None


def _better_low(b: IRBuilder, k1, c1, k2, c2) -> Value:
    return MX._better(b, k1, c1, k2, c2)


def _better_high(b: IRBuilder, k1, c1, k2, c2) -> Value:
    """Mutant: ties go to the HIGHER column."""
    return b.lor(b.cmp_gt(k1, k2), b.land(b.cmp_eq(k1, k2), b.cmp_gt(c1, c2)))


def _bf16_key_sampler(b: IRBuilder, v: Value) -> Value:
    """ATOM's greedy sampler order (``aiter.topk_select(tie="low")``, measured by P2's rule probe): the
    mono key, except that a NaN of either sign ranks above everything (the mono key ranks -NaN below
    -inf). Ties, signed zeros and infinities are ordered as the mono key orders them."""
    key = _bf16_key(b, v)
    bits = b.shl(b.zext(b.bitcast(b.trunc_f32_to_bf16(v), I16), I32), _ci(b, 16))
    nan = b.cmp_gt(b.land(bits, _ci(b, 0x7FFFFFFF)), _ci(b, 0x7F800000))
    return b.select(nan, _ci(b, 0x7FFFFFFF), key)


def emit_head(
    b: IRBuilder,
    spec: MxDecodeGemvSpec,
    io: HeadIO,
    *,
    cta: Optional[Value] = None,
    x_wait: Optional[Callable[[Dict[str, Value]], None]] = None,
    x_stage: Optional[Callable[[Dict[str, Value]], Value]] = None,
    better: Callable = _better_low,
    logits_arrive: bool = False,
    key_fn: Callable = _bf16_key,
    hooks: Optional[Dict[str, Callable[[], None]]] = None,
    sync: Optional[Callable[[], None]] = None,
    on_fold: Optional[Callable[[Value, Value], None]] = None,
    fold_poll: Optional[Callable[[Value, List], List[Value]]] = None,
    on_keys: Optional[Callable[[Value, Value, Value, Value], None]] = None,
) -> None:
    """Emit the head of ``spec`` (``build_mx_decode_gemv``'s body) on ``b``.

    ``cta``: the head CTA index (default ``block_id_x``, at the original's
    position). ``x_wait(ctx)``: emitted at the start of the x staging (before
    its loads). ``x_stage(ctx)``: replaces the bf16 LDS staging; returns the
    LDS buffer ``[rows_max, k + xpad]`` (bf16), ending with a CTA barrier.
    ``better``: the argmax order (``(k1, c1)`` beats ``(k2, c2)``).
    ``logits_arrive``: logits builds also add the fused fold's arrival to the
    64-bit counter at ``ws`` (the launch epoch of the P2 builds).
    ``key_fn``: the argmax key of an f32 accumulator (default the mono rule).
    ``hooks``: optional emitters at ``after_issue`` (first weight groups issued),
    ``after_x`` (x staged), ``after_loop`` and ``end`` (stamped builds).
    ``sync``: replaces the CTA barrier of the fused argmax fold (default
    ``b.sync()``); a head that shares its CTA with other wave groups passes a
    wave-group barrier (probe P9). ``on_fold(cf, rr)``: emitted in the fold's
    last-arriver block after the result store, on every head wave, with the
    folded column ``cf`` of row ``rr`` (lane % 16; rows past ``m`` hold row
    ``m - 1``'s copy); P9's in-kernel accept / draft-token stage. Both default
    to None: the emitted IR is unchanged. ``fold_poll(tag, specs)``: replaces the fold's
    ``b.tagged_poll(tag, specs, aux=CPOL_SC1)`` (specs ``(rsrc, byte offset, 4)``; returns one
    ``<4 x i32>`` per spec), e.g. P9's poll with a timeout. ``on_keys(k, c, row, mine)``: replaces the
    fused fold (argmax builds): emitted on every head wave after the CTA's own argmax, with the CTA's key
    ``k`` and column ``c`` of row ``row`` (lane % 16; ``mine`` = wave 0, lane < 16, row < m); the
    caller delivers them (P9 ``fold=direct``: tagged slots its consumers poll and fold themselves).
    """
    _sync = sync if sync is not None else b.sync
    hk = hooks or {}

    def hook(name: str) -> None:
        if name in hk:
            hk[name]()

    s = spec
    K, N = s.k, s.n
    KB = K // 2
    KS = K // 32
    NT, W, G = s.nt, s.waves, s.ctas
    NG, NQ, DG = s.ngroups, s.nqk, s.dg
    S = s.ksplit
    SL = W // S
    P = s.panels
    TW = G * SL
    aux = CPOLS[s.cpol]

    def pick(k1, c1, k2, c2):
        w = better(b, k2, c2, k1, c1)
        return b.select(w, k2, k1), b.select(w, c2, c1)

    def reduce_lanes(k, c, masks):
        for m in masks:
            k2 = b.warp_shuffle_xor(k, m)
            c2 = b.warp_shuffle_xor(c, m)
            k, c = pick(k, c, k2, c2)
        return k, c

    q8 = s.act in ("mxfp8", "bf16q8")
    xin8 = s.act == "mxfp8"
    xp, wp, sp = io.xp, io.wp, io.sp
    xsp, yp, wsp = io.xsp, io.yp, io.wsp
    m_rows, sxm, sym, rev = io.m_rows, io.sxm, io.sym, io.rev

    c0 = _ci(b, 0)
    c1 = _ci(b, 1)
    tid = b.thread_id_x()
    lane = b.lane_id()
    wave = b.readfirstlane(b.lshr(tid, _ci(b, 6)))
    if cta is None:
        cta = b.block_id_x()
    g4 = b.lshr(lane, _ci(b, 4))
    c16 = b.land(lane, _ci(b, 15))
    m_last = b.sub(m_rows, c1)
    a_row = b.smin(c16, m_last)

    w_rsrc = b.buffer_rsrc(wp, _ci(b, N * KB))
    s_rsrc = b.buffer_rsrc(sp, _ci(b, N * KS))
    esz = 1 if xin8 else 2
    x_bytes = b.add(b.mul(b.mul(m_last, sxm), _ci(b, esz)), _ci(b, esz * K))
    x_rsrc = b.buffer_rsrc(xp, x_bytes)
    if xin8:
        xs_rsrc = b.buffer_rsrc(xsp, b.mul(m_rows, _ci(b, KS)))
    sh8g = b.mul(g4, _ci(b, 8))
    if S > 1:
        slot = b.readfirstlane(b.div(wave, _ci(b, S)))
        kp = b.readfirstlane(b.mod(wave, _ci(b, S)))
        qb = b.mul(kp, _ci(b, NQ))
        qb128 = b.mul(qb, _ci(b, 128))
    else:
        slot, kp, qb, qb128 = wave, None, None, None

    def kofs(v: Value) -> Value:
        return v if qb128 is None else b.add(v, qb128)

    X: Dict[str, Value] = {}
    ctx = {"tid": tid, "lane": lane, "wave": wave, "m_rows": m_rows, "m_last": m_last}

    def stage_x() -> None:
        if x_wait is not None:
            x_wait(ctx)
        if x_stage is not None:
            X["xs"] = x_stage(ctx)
            return
        if s.x_src == "lds" and s.act == "bf16q8":
            from rocke.helpers import mx_quant as _mq

            XS = K + s.xpad
            xs = b.smem_alloc(I8, [s.rows_max, XS], name_hint="xq")
            xss = b.smem_alloc(I8, [s.rows_max, KS], name_hint="xqs")
            per_row = K // 8
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                ok = b.cmp_lt(idx, _ci(b, total))
                idc = b.select(ok, idx, _ci(b, total - 1))
                row = b.div(idc, _ci(b, per_row))
                col = b.mul(b.mod(idc, _ci(b, per_row)), _ci(b, 8))
                off = b.mul(b.add(b.mul(b.smin(row, m_last), sxm), col), _ci(b, 2))
                v = b.buffer_load_vN(x_rsrc, off, c0, BF16, 8)
                vals = [b.cast_to_f32(b.vec_extract(v, i)) for i in range(8)]
                words, expo = _mq.quantize_mxfp8(b, vals, 4)
                with b.scf_if(ok):
                    pair = b.vec_pack(words, I32)
                    b.smem_store_vN(xs, [row, col], b.vec_bitcast(pair, VectorType(I8, 8)), 8)
                    with b.scf_if(b.cmp_eq(b.land(idx, _ci(b, 3)), c0)):
                        blk = b.div(col, _ci(b, 32))
                        b.smem_store_vN(xss, [row, blk], b.trunc(expo, I8), 1)
            b.sync()
        elif s.x_src == "lds" and q8:
            XS = K + s.xpad
            xs = b.smem_alloc(I8, [s.rows_max, XS], name_hint="xq")
            xss = b.smem_alloc(I8, [s.rows_max, KS], name_hint="xqs")
            per_row = K // 16
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                idx = b.select(b.cmp_lt(idx, _ci(b, total)), idx, _ci(b, total - 1))
                row = b.div(idx, _ci(b, per_row))
                col = b.mul(b.mod(idx, _ci(b, per_row)), _ci(b, 16))
                off = b.add(b.mul(b.smin(row, m_last), sxm), col)
                v = b.buffer_load_vN(x_rsrc, off, c0, I32, 4)
                b.smem_store_vN(xs, [row, col], b.vec_bitcast(v, VectorType(I8, 16)), 16)
            per_row = K // 128
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                idx = b.select(b.cmp_lt(idx, _ci(b, total)), idx, _ci(b, total - 1))
                row = b.div(idx, _ci(b, per_row))
                qq = b.mod(idx, _ci(b, per_row))
                off = b.add(b.mul(b.smin(row, m_last), _ci(b, KS)), b.mul(qq, _ci(b, 4)))
                v = b.buffer_load(xs_rsrc, off, c0, I32)
                b.smem_store_vN(xss, [row, b.mul(qq, _ci(b, 4))], b.vec_bitcast(b.vec_pack([v], I32), VectorType(I8, 4)), 4)
            b.sync()
        elif s.x_src == "lds":
            XS = K + s.xpad
            xs = b.smem_alloc(BF16, [s.rows_max, XS], name_hint="xs")
            per_row = K // 8
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                inb = b.cmp_lt(idx, _ci(b, total))
                idx = b.select(inb, idx, _ci(b, total - 1))
                row = b.div(idx, _ci(b, per_row))
                col = b.mul(b.mod(idx, _ci(b, per_row)), _ci(b, 8))
                src_row = b.smin(row, m_last)
                off = b.mul(b.add(b.mul(src_row, sxm), col), _ci(b, 2))
                v = b.buffer_load_vN(x_rsrc, off, c0, BF16, 8)
                b.smem_store_vN(xs, [row, col], v, 8)
            b.sync()
        if s.x_src == "lds":
            X["xs"] = xs
            if q8:
                X["xss"] = xss

    if s.pro == "x_first":
        stage_x()
        hook("after_x")

    def a_frag(row: Value, kx: Value) -> Value:
        if s.x_src == "lds":
            return b.smem_load_vN(X["xs"], row, kx, dtype=BF16, n=8)
        off = b.mul(b.add(b.mul(row, sxm), kx), _ci(b, 2))
        return b.buffer_load_vN(x_rsrc, off, c0, BF16, 8)

    def a_frag_q8(row: Value, q: int):
        k0 = b.add(sh8g, sh8g)
        halves = []
        for h in range(2):
            kx = kofs(b.add(k0, _ci(b, 128 * q + 64 * h)))
            if s.x_src == "lds":
                v = b.smem_load_vN(X["xs"], row, kx, dtype=I8, n=16)
                halves.append(b.vec_bitcast(v, VectorType(I32, 4)))
            else:
                halves.append(b.buffer_load_vN(x_rsrc, b.add(b.mul(row, sxm), kx), c0, I32, 4))
        qa = _ci(b, q) if qb is None else b.add(qb, _ci(b, q))
        if s.x_src == "lds":
            raw = b.smem_load_vN(X["xss"], row, b.mul(qa, _ci(b, 4)), dtype=I8, n=4)
            sw = b.vec_extract(b.vec_bitcast(raw, VectorType(I32, 1)), 0)
        else:
            sw = b.buffer_load(xs_rsrc, b.add(b.mul(row, _ci(b, KS)), b.mul(qa, _ci(b, 4))), c0, I32)
        return b.vec_concat(halves[0], halves[1]), b.lshr(sw, sh8g)

    gw = b.add(b.mul(cta, _ci(b, SL)), slot)
    if s.cta_map == "interleave":
        first = gw
        count = b.div(b.add(b.sub(_ci(b, P), gw), _ci(b, TW - 1)), _ci(b, TW))
        count = b.smax(count, c0)
        pstep = TW
        g0 = b.mul(cta, _ci(b, SL))
        trips = b.smax(b.div(b.add(b.sub(_ci(b, P), g0), _ci(b, TW - 1)), _ci(b, TW)), c0)
    else:
        first = b.div(b.mul(gw, _ci(b, P)), _ci(b, TW))
        nxt = b.div(b.mul(b.add(gw, c1), _ci(b, P)), _ci(b, TW))
        count = b.sub(nxt, first)
        pstep = 1
        trips = _ci(b, -(-P // TW))
    if S == 1:
        trips = count
    is_rev = b.cmp_ne(rev, c0)

    def panel_of(it: Value) -> Value:
        p = b.add(first, b.mul(it, _ci(b, pstep)))
        return b.select(is_rev, b.sub(_ci(b, P - 1), p), p)

    def row_base(it: Value, valid: Value):
        n0 = b.mul(panel_of(it), _ci(b, NT * _TILE_N))
        col = b.add(n0, c16)
        wb = b.add(b.mul(col, _ci(b, KB)), b.mul(g4, _ci(b, 16 if s.wload == "x4" else 4)))
        sb = b.mul(col, _ci(b, KS))
        if s.sload == "bcast":
            sb = b.add(sb, b.mul(g4, _ci(b, 4)))
        if qb is not None:
            wb = b.add(wb, b.mul(qb, _ci(b, 64)))
            sb = b.add(sb, b.mul(qb, _ci(b, 4)))
        wb = b.select(valid, wb, _ci(b, _OOB))
        sb = b.select(valid, sb, _ci(b, _OOB))
        return wb, sb

    def issue(wb: Value, sb: Value, gi: int) -> List[Value]:
        out = []
        q0 = gi * _GROUP_Q
        qn = min(_GROUP_Q, NQ - q0)
        for t in range(NT):
            soff = b.add(sb, _ci(b, t * _TILE_N * KS + q0 * 4))
            sdw = 1 if s.sload == "bcast" else 4
            out.append(b.buffer_load_cp(s_rsrc, soff, c0, dwords=sdw, aux=aux))
        for j in range(qn):
            for t in range(NT):
                off = b.add(wb, _ci(b, t * _TILE_N * KB + (q0 + j) * 64))
                if s.wload == "x4":
                    out.append(b.buffer_load_cp(w_rsrc, off, c0, dwords=4, aux=aux))
                else:
                    for d in range(4):
                        o2 = b.add(off, _ci(b, 16 * d))
                        out.append(b.buffer_load_cp(w_rsrc, o2, c0, dwords=1, aux=aux))
        return out

    def code_words(data: List[Value], j: int, t: int) -> List[Value]:
        if s.wload == "x4":
            v = data[NT + j * NT + t]
            return [b.vec_extract(v, d) for d in range(4)]
        base = NT + (j * NT + t) * 4
        return list(data[base : base + 4])

    def group_len(gi: int) -> int:
        per = 1 if s.wload == "x4" else 4
        return NT * (per * min(_GROUP_Q, NQ - gi * _GROUP_Q) + 1)

    argmax = s.epilogue != "logits"
    best0 = []
    if argmax:
        for _ in range(4):
            best0 += [_ci(b, -0x80000000), _ci(b, 0x7FFFFFFF)]

    valid0 = b.cmp_lt(c0, count)
    wb0, sb0 = row_base(c0, valid0)
    carry0: List[Value] = []
    if s.sched != "none":
        b.sched_barrier(0)
    for gi in range(DG):
        carry0 += issue(wb0, sb0, gi)
        if s.sched != "none":
            b.sched_barrier(0)

    hook("after_issue")
    if s.pro == "w_first":
        stage_x()
        hook("after_x")

    iters = [(f"ld{i}", v) for i, v in enumerate(carry0)]
    iters += [(f"best{i}", v) for i, v in enumerate(best0)]
    if S > 1:
        redk = b.smem_alloc(F32, [2, W, NT * 4, _LANES], name_hint="redk")
    loop = b.scf_for_iter(c0, trips, c1, iters, iv_name="it")
    with loop as (it, vals):
        carried = list(vals[: len(carry0)])
        best = list(vals[len(carry0) :])
        nxt_it = b.add(it, c1)
        valid_n = b.cmp_lt(nxt_it, count)
        wb, sb = row_base(it, b.cmp_lt(it, count))
        wbn, sbn = row_base(nxt_it, valid_n)
        n0 = b.mul(panel_of(it), _ci(b, NT * _TILE_N))

        groups: Dict[int, List[Value]] = {}
        pos = 0
        for gi in range(DG):
            ln = group_len(gi)
            groups[gi] = carried[pos : pos + ln]
            pos += ln
        next_carry: List[Value] = []
        acc = [b.zero_vec_f32(4) for _ in range(NT)]
        if s.setprio:
            b.s_setprio(s.setprio)
        for gi in range(NG):
            tgt = gi + DG
            if tgt < NG:
                groups[tgt] = issue(wb, sb, tgt)
            else:
                next_carry += issue(wbn, sbn, tgt - NG)
            data = groups.pop(gi)
            arow = dops.fresh(b, a_row)
            q0 = gi * _GROUP_Q
            qn = min(_GROUP_Q, NQ - q0)
            if s.sload == "bcast":
                bq = [_bcast4(b, data[t]) for t in range(NT)]
            else:
                bq = [[b.vec_extract(data[t], j) for j in range(4)] for t in range(NT)]
            for j in range(qn):
                q = q0 + j
                if q8:
                    a, sa = a_frag_q8(arow, q)
                    for t in range(NT):
                        bw = data[NT + j * NT + t]
                        sb_ = b.lshr(bq[t][j], sh8g)
                        acc[t] = b.mfma_scale_mx(
                            a, bw, acc[t], sa, sb_, shape="16x16x128", fmt_a="fp8", fmt_b="fp4"
                        )
                elif s.korder == "mono":
                    words = []
                    for t in range(NT):
                        w4 = code_words(data, j, t)
                        words.append(_transpose4(b, w4) if s.wload == "x4" else w4)
                    for d in range(4):
                        a = a_frag(arow, kofs(b.add(b.mul(g4, _ci(b, 8)), _ci(b, 128 * q + 32 * d))))
                        sh = 23 - 8 * d
                        for t in range(NT):
                            sw = bq[t][j]
                            sc = b.shl(sw, _ci(b, sh)) if sh > 0 else b.lshr(sw, _ci(b, 1))
                            if s.smask:
                                sc = b.land(sc, _ci(b, 0x7F800000))
                            bf = _b_frag(b, words[t][d], b.bitcast(sc, F32))
                            acc[t] = b.mfma_f32_16x16x32_bf16(a, bf, acc[t])
                else:
                    scs = []
                    for t in range(NT):
                        sw = b.shl(b.lshr(bq[t][j], b.mul(g4, _ci(b, 8))), _ci(b, 23))
                        if s.smask:
                            sw = b.land(sw, _ci(b, 0x7F800000))
                        scs.append(b.bitcast(sw, F32))
                    for d in range(4):
                        kx = b.add(b.mul(g4, _ci(b, 32)), _ci(b, 128 * q + 8 * d))
                        a = a_frag(arow, kofs(kx))
                        for t in range(NT):
                            bf = _b_frag(b, code_words(data, j, t)[d], scs[t])
                            acc[t] = b.mfma_f32_16x16x32_bf16(a, bf, acc[t])
                if s.sched == "q":
                    b.sched_barrier(0)
            if s.sched == "group":
                b.sched_barrier(0)
        if s.setprio:
            b.s_setprio(0)

        epi = None
        if S > 1:
            buf = b.land(it, c1)
            for t in range(NT):
                for i in range(4):
                    b.smem_store_vN(redk, [buf, wave, _ci(b, t * 4 + i), lane], b.vec_extract(acc[t], i), 1)
            b.sync_lds_only()
            epi = b.land(b.cmp_eq(kp, c0), b.cmp_lt(it, count))
            for t in range(NT):
                parts = [b.vec_extract(acc[t], i) for i in range(4)]
                for kk in range(1, S):
                    wk = b.add(wave, _ci(b, kk))
                    for i in range(4):
                        v = b.vec_extract(
                            b.smem_load_vN(redk, buf, wk, _ci(b, t * 4 + i), lane, dtype=F32, n=1), 0
                        )
                        parts[i] = b.fadd(parts[i], v)
                acc[t] = b.vec_pack(parts, F32)

        if s.epilogue == "logits":
            for t in range(NT):
                col = b.add(n0, _ci(b, t * _TILE_N))
                col = b.add(col, c16)
                for i in range(4):
                    row = b.add(b.mul(g4, _ci(b, 4)), _ci(b, i))
                    ok = b.cmp_lt(row, m_rows)
                    if epi is not None:
                        ok = b.land(ok, epi)
                    with b.scf_if(ok):
                        v = b.trunc_f32_to_bf16(b.vec_extract(acc[t], i))
                        b.global_store(yp, b.add(b.mul(row, sym), col), v)
            new_best = best
        else:
            new_best = list(best)
            for t in range(NT):
                col = b.add(b.add(n0, _ci(b, t * _TILE_N)), c16)
                for i in range(4):
                    key = key_fn(b, b.vec_extract(acc[t], i))
                    if epi is not None:
                        key = b.select(epi, key, _ci(b, -0x80000000))
                    kb, cb = new_best[2 * i], new_best[2 * i + 1]
                    new_best[2 * i], new_best[2 * i + 1] = pick(kb, cb, key, col)
        b.scf_yield(*next_carry, *new_best)
    hook("after_loop")

    if not argmax:
        if logits_arrive:
            # the launch epoch: one arrival per CTA after every wave consumed x
            # (the x staging ends with a CTA barrier, so every wave has read the
            # counter before thread 0 can add to it)
            with b.scf_if(b.cmp_eq(tid, c0)):
                ws64 = dops.uniform(b, b.global_ptr_to_i64(wsp))
                _atomic_inc64(b, ws64)
        hook("end")
        return

    res = list(loop.results[len(carry0) :])
    red = b.smem_alloc(I32, [W, 16, 2], name_hint="red", exclusive=True)
    for i in range(4):
        k, c = reduce_lanes(res[2 * i], res[2 * i + 1], (1, 2, 4, 8))
        row = b.add(b.mul(g4, _ci(b, 4)), _ci(b, i))
        with b.scf_if(b.cmp_eq(c16, c0)):
            b.smem_store_vN(red, [wave, row, c0], k, 1)
            b.smem_store_vN(red, [wave, row, c1], c, 1)
    _sync()
    r = c16
    kb = b.smem_load_vN(red, c0, r, c0, dtype=I32, n=1)
    cb = b.smem_load_vN(red, c0, r, c1, dtype=I32, n=1)
    kb = b.vec_extract(kb, 0)
    cb = b.vec_extract(cb, 0)
    for wv in range(1, W):
        k2 = b.vec_extract(b.smem_load_vN(red, _ci(b, wv), r, c0, dtype=I32, n=1), 0)
        c2 = b.vec_extract(b.smem_load_vN(red, _ci(b, wv), r, c1, dtype=I32, n=1), 0)
        kb, cb = pick(kb, cb, k2, c2)
    mine = b.land(b.cmp_eq(wave, c0), b.land(b.cmp_lt(lane, _ci(b, 16)), b.cmp_lt(r, m_rows)))

    if on_keys is not None:
        on_keys(kb, cb, r, mine)
        hook("end")
        return

    if s.epilogue == "keys":
        with b.scf_if(mine):
            hi = b.shl(b.sext(kb, I64), b.const_i64(32))
            lo = b.zext(b.sub(_ci(b, N - 1), cb), I64)
            b.global_store(yp, b.add(b.mul(r, _ci(b, G)), cta), b.lor(hi, lo))
        hook("end")
        return

    ws_rsrc = b.buffer_rsrc(wsp, _ci(b, _FUSED_HDR + s.rows_max * G * 16))
    epoch = s.fsync == "epoch"
    c64 = s.fsync == "counter64"
    flag = b.smem_alloc(I32, [1 if s.fsync == "counter" else 2], name_hint="flag")
    with b.scf_if(b.cmp_eq(tid, c0)):
        ws64 = dops.uniform(b, b.global_ptr_to_i64(wsp))
        if epoch:
            ep = dops.atomic_add_scoped(b, ws64, _ci(b, 4), c0, scope="system", returns=True)
            b.smem_store_vN(flag, [c1], ep, 1)
        if c64:
            cnt = _atomic_inc64(b, ws64)
            gq = b.const_i64(G)
            b.smem_store_vN(flag, [c0], b.trunc(b.mod(cnt, gq), I32), 1)
            b.smem_store_vN(flag, [c1], b.trunc(b.div(cnt, gq), I32), 1)
        else:
            old = dops.atomic_add_scoped(b, ws64, c0, c1, scope="system", returns=True)
            b.smem_store_vN(flag, [c0], old, 1)
    _sync()
    old = b.readfirstlane(b.vec_extract(b.smem_load_vN(flag, c0, dtype=I32, n=1), 0))
    if c64:
        q = b.readfirstlane(b.vec_extract(b.smem_load_vN(flag, c1, dtype=I32, n=1), 0))
        tag = b.add(q, c1)
        last = b.cmp_eq(old, _ci(b, G - 1))
    elif epoch:
        ep = b.readfirstlane(b.vec_extract(b.smem_load_vN(flag, c1, dtype=I32, n=1), 0))
        tag = b.add(ep, c1)
        last = b.cmp_eq(old, _ci(b, G - 1))
    else:
        lg = G.bit_length() - 1
        tag = b.add(b.lshr(old, _ci(b, lg)), c1)
        last = b.cmp_eq(b.land(old, _ci(b, G - 1)), _ci(b, G - 1))
    with b.scf_if(mine):
        slot_off = b.add(_ci(b, _FUSED_HDR), b.mul(b.add(b.mul(r, _ci(b, G)), cta), _ci(b, 16)))
        word = b.vec_pack([kb, tag, cb, tag], I32)
        b.buffer_store_cp(word, ws_rsrc, slot_off, c0, aux=CPOL_SC1)
    with b.scf_if(last):
        src = b.smin(tid, _ci(b, G - 1))
        specs = []
        for rr in range(s.rows_max):
            row = b.smin(_ci(b, rr), m_last)
            off = b.add(_ci(b, _FUSED_HDR), b.mul(b.add(b.mul(row, _ci(b, G)), src), _ci(b, 16)))
            specs.append((ws_rsrc, off, 4))
        if fold_poll is not None:
            got = fold_poll(tag, specs)
        else:
            got = b.tagged_poll(tag, specs, aux=CPOL_SC1)
        red2 = b.smem_alloc(I32, [W, s.rows_max, 2], name_hint="red2")
        for rr in range(s.rows_max):
            k = b.vec_extract(got[rr], 0)
            c = b.vec_extract(got[rr], 2)
            k, c = reduce_lanes(k, c, (1, 2, 4, 8, 16, 32))
            with b.scf_if(b.cmp_eq(lane, c0)):
                b.smem_store_vN(red2, [wave, _ci(b, rr), c0], k, 1)
                b.smem_store_vN(red2, [wave, _ci(b, rr), c1], c, 1)
        _sync()
        rr = c16
        rq = b.smin(rr, _ci(b, s.rows_max - 1))
        kf = b.vec_extract(b.smem_load_vN(red2, c0, rq, c0, dtype=I32, n=1), 0)
        cf = b.vec_extract(b.smem_load_vN(red2, c0, rq, c1, dtype=I32, n=1), 0)
        for wv in range(1, W):
            k2 = b.vec_extract(b.smem_load_vN(red2, _ci(b, wv), rq, c0, dtype=I32, n=1), 0)
            c2 = b.vec_extract(b.smem_load_vN(red2, _ci(b, wv), rq, c1, dtype=I32, n=1), 0)
            kf, cf = pick(kf, cf, k2, c2)
        with b.scf_if(mine):
            b.global_store(yp, rr, cf)
        if on_fold is not None:
            on_fold(cf, rr)
        if epoch:
            with b.scf_if(b.cmp_eq(tid, c0)):
                ws64 = dops.uniform(b, b.global_ptr_to_i64(wsp))
                dops.atomic_add_scoped(b, ws64, c0, _ci(b, -G), scope="system")
                dops.atomic_add_scoped(b, ws64, _ci(b, 4), c1, scope="system")
    hook("end")


def _head_params(b: IRBuilder, s: MxDecodeGemvSpec) -> HeadIO:
    """``build_mx_decode_gemv``'s parameters, in its order."""
    xin8 = s.act == "mxfp8"
    xp = b.param("x", PtrType(I8 if xin8 else BF16, "global"), noalias=True, readonly=True, align=16)
    wp = b.param("w", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    sp = b.param("s", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    xsp = None
    if xin8:
        xsp = b.param("xs", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    wsp = None
    if s.epilogue == "logits":
        yp = b.param("y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=2)
    elif s.epilogue == "keys":
        yp = b.param("y", PtrType(I64, "global"), noalias=True, writeonly=True, align=8)
    else:
        yp = b.param("y", PtrType(I32, "global"), noalias=True, writeonly=True, align=4)
        wsp = b.param("ws", PtrType(I32, "global"), noalias=True, align=16)
    m_rows = b.param("m", I32)
    sxm = b.param("sxm", I32)
    sym = b.param("sym", I32)
    rev = b.param("rev", I32)
    return HeadIO(xp=xp, wp=wp, sp=sp, yp=yp, m_rows=m_rows, sxm=sxm, sym=sym, rev=rev, xsp=xsp, wsp=wsp)


def build_head_clone(spec: MxDecodeGemvSpec) -> KernelDef:
    """``build_mx_decode_gemv(spec)`` through :func:`emit_head` (same name,
    same parameters): its LLVM IR must equal the original's byte for byte."""
    ok, why = MX.is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid MxDecodeGemvSpec: {why}")
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)
    io = _head_params(b, spec)
    emit_head(b, spec, io)
    return b.kernel


# ---------------------------------------------------------------- P2 builds


@dataclass(frozen=True)
class P2Spec:
    """One P2 build. ``head``: the head stage (``ctas`` = head CTAs ``G``,
    ``waves`` its waves; ``pro`` is set from ``pre``). ``grid`` x ``threads``:
    the launch (the true mono's is 256 x 512). Producers: ``prod_ctas`` CTAs
    from ``prod_first`` (-1: the last ``prod_ctas`` CTAs of the grid), each
    owning ``k / prod_ctas`` columns of every row. ``pbatch``: 8-element x
    items per thread and payload poll (``pairs``)."""

    head: MxDecodeGemvSpec
    grid: int = 256
    threads: int = 256
    xmode: str = "pairs"
    pre: str = "before"
    prod_ctas: int = 10
    prod_first: int = -1
    pwait: str = "auto"
    pbatch: int = 4
    ppause: str = "s_nop"
    ploop: str = "static"
    pimpl: str = "asm"
    nanrule: str = "mono"
    l2pf: int = 0
    stamps: bool = False
    mutant: str = "none"

    @property
    def pfirst(self) -> int:
        return self.grid - self.prod_ctas if self.prod_first < 0 else self.prod_first

    @property
    def hint(self) -> bool:
        if self.pwait == "auto":
            return self.head.rows_max > 4
        return self.pwait == "hint"

    def head_spec(self) -> MxDecodeGemvSpec:
        return replace(self.head, pro="w_first" if self.pre == "before" else "x_first")

    def kernel_name(self) -> str:
        h = self.head_spec()
        base = h.kernel_name()[: -len("_pyonly")].replace("mxgemv_gfx950_", "")
        d = P2Spec(head=self.head)
        parts = [f"p2head_{base}", f"G{self.grid}", f"T{self.threads}", f"x{self.xmode}"]
        if self.xmode != "none":
            parts += [f"pre{self.pre}", f"pc{self.prod_ctas}", f"pf{self.pfirst}"]
            if self.xmode == "pairs":
                parts.append(f"pw{'hint' if self.hint else 'none'}")
                if self.pbatch != d.pbatch:
                    parts.append(f"pb{self.pbatch}")
                if self.ppause != d.ppause:
                    parts.append(f"pp{self.ppause}")
                if self.ploop != d.ploop:
                    parts.append(f"pl{self.ploop}")
            if self.pimpl != d.pimpl:
                parts.append(f"pi{self.pimpl}")
        if self.nanrule != d.nanrule:
            parts.append(f"nan{self.nanrule}")
        if self.l2pf:
            parts.append(f"l2pf{self.l2pf}")
        if self.mutant != "none":
            parts.append(f"MUT{self.mutant}")
        if self.stamps:
            parts.append("STAMP")
        return "_".join(parts) + "_pyonly"


def is_valid_p2(spec: P2Spec) -> Tuple[bool, str]:
    h = spec.head_spec()
    ok, why = MX.is_valid_spec(h)
    if not ok:
        return False, why
    checks = [
        (spec.xmode in XMODES, f"xmode must be one of {XMODES}"),
        (spec.pre in PRES, f"pre must be one of {PRES}"),
        (spec.pwait in PWAITS, f"pwait must be one of {PWAITS}"),
        (spec.ploop in PLOOPS, f"ploop must be one of {PLOOPS}"),
        (spec.pimpl in PIMPLS, f"pimpl must be one of {PIMPLS}"),
        (spec.nanrule in NANRULES, f"nanrule must be one of {NANRULES}"),
        (spec.nanrule == "mono" or h.epilogue == "fused", "nanrule applies to the argmax"),
        (spec.l2pf == 0 or spec.threads >= 2 * h.threads, "l2pf needs a spare wave per head wave"),
        (spec.l2pf == 0 or (h.nt == 1 and h.ksplit == 1 and h.cta_map == "interleave" and h.wload == "x4"
                            and h.sload == "x4"), "l2pf follows the default panel schedule only"),
        (0 <= spec.l2pf <= 32, "l2pf must be 0..32 load groups"),
        (spec.mutant in MUTANTS, f"mutant must be one of {MUTANTS}"),
        (spec.ppause in dops.PAUSES, f"ppause must be one of {tuple(dops.PAUSES)}"),
        (h.act == "bf16" and h.x_src == "lds", "P2 heads take bf16 x staged in LDS"),
        (h.epilogue in ("fused", "logits"), "P2 heads end in fused argmax or logits"),
        (spec.threads in (64, 128, 256, 512, 1024) and h.threads <= spec.threads, "threads must hold the head's waves"),
        (spec.grid >= h.ctas, "grid must hold the head CTAs"),
        (spec.xmode == "none" or 1 <= spec.prod_ctas <= 64, "1..64 producer CTAs"),
        (spec.xmode == "none" or 0 <= spec.pfirst and spec.pfirst + spec.prod_ctas <= spec.grid, "producers inside the grid"),
        (spec.xmode == "none" or h.k % (4 * spec.prod_ctas) == 0, "k must split into 4-element items per producer"),
        (1 <= spec.pbatch <= 8, "pbatch must be 1..8"),
        (spec.mutant != "stale_tag" or spec.xmode != "none", "stale_tag needs a producer"),
        (spec.mutant != "tie_high" or h.epilogue == "fused", "tie_high needs the argmax"),
        (h.waves_per_eu == 0 or spec.threads <= 256, "waves_per_eu > 0 only for 256-thread launches"),
    ]
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ""


def p2_signature(spec: P2Spec) -> List[Tuple[str, str]]:
    sig = [
        ("x", "ptr"), ("w", "ptr"), ("s", "ptr"), ("y", "ptr"), ("ws", "ptr"), ("xb", "ptr"),
        ("m", "i32"), ("sxm", "i32"), ("sym", "i32"), ("rev", "i32"), ("delay", "i32"),
    ]
    if spec.stamps:
        sig.append(("st", "ptr"))
    return sig


def producer_signature(stamps: bool = False) -> List[Tuple[str, str]]:
    sig = [("x", "ptr"), ("xd", "ptr"), ("m", "i32"), ("sxm", "i32"), ("delay", "i32")]
    if stamps:
        sig.append(("st", "ptr"))
    return sig


def _delay(b: IRBuilder, delay: Value) -> None:
    """``delay`` x ``s_sleep 8`` (a stand-in for the last layer's drain)."""
    c0, c1 = _ci(b, 0), _ci(b, 1)
    with b.scf_for(c0, delay, c1, iv_name="dly"):
        dops.s_sleep(b, 8)


def _produce(b, *, k, rows_max, threads, P, p, m_rows, sxm, x_rsrc, store, before_store=None) -> None:
    """Producer ``p`` of ``P``: columns ``[p k / P, (p + 1) k / P)`` of rows
    ``0 .. m - 1``, four bf16 (two words) per item: ``store(row, c4, w0, w1)``.
    Every item's load is issued first (indices past the last valid item re-read
    it), then ``before_store()`` (e.g. the tag, whose load was issued earlier:
    one wait for all), then the stores. Lanes past the last valid item store
    that item again (the same bytes to the same address), so no branch lets the
    compiler sink the loads behind the tag's wait."""
    tid = b.thread_id_x()
    cols = k // P
    ipr = cols // 4
    total = b.mul(m_rows, _ci(b, ipr))
    items = []
    for it in range(-(-rows_max * ipr // threads)):
        idx = b.add(tid, _ci(b, it * threads))
        ok = b.cmp_lt(idx, total)
        idc = b.select(ok, idx, b.sub(total, _ci(b, 1)))
        row = b.div(idc, _ci(b, ipr))
        c4 = b.add(b.mul(b.mod(idc, _ci(b, ipr)), _ci(b, 4)), b.mul(p, _ci(b, cols)))
        off = b.mul(b.add(b.mul(row, sxm), c4), _ci(b, 2))
        items.append((ok, row, c4, b.buffer_load_vN(x_rsrc, off, _ci(b, 0), I32, 2)))
    if before_store is not None:
        b.sched_barrier(0)
        before_store()
    for ok, row, c4, v in items:
        store(row, c4, b.vec_extract(v, 0), b.vec_extract(v, 1))


#: prefetch loads in flight per spare-wave lane before one drain (x 4 VGPRs)
_PF_BATCH = 16


def _l2_prefetch(b: IRBuilder, h: MxDecodeGemvSpec, wp: Value, sp: Value, blk: Value, wave: Value, groups: int) -> None:
    """Spare wave ``W + j`` of a head CTA touches load groups ``dg .. dg + groups - 1`` of head wave
    ``j``'s panel stream (the groups the head wave issues after its prologue), so they are in L2 (and
    the CU's vL1D) when the head wave asks for them. The schedule is the head's default one (``nt=1``,
    ``ksplit=1``, interleave, ``x4`` loads). Asm loads with no consumer: every ``_PF_BATCH`` of them end
    in one ``s_waitcnt vmcnt(0)`` asm that keeps their destinations allocated (toy_chain idiom).
    Out-of-range panels are clamped onto the wave's first panel (a duplicate touch)."""
    W, G = h.waves, h.ctas
    K, N = h.k, h.n
    KB, KS = K // 2, K // 32
    NG, DG = h.ngroups, h.dg
    P = h.panels
    TW = G * W
    c0 = _ci(b, 0)
    is_pf = b.land(b.cmp_lt(blk, _ci(b, G)), b.cmp_ge(wave, _ci(b, W)))
    with b.scf_if(is_pf):
        lane = b.lane_id()
        g4 = b.lshr(lane, _ci(b, 4))
        c16 = b.land(lane, _ci(b, 15))
        gw = b.add(b.mul(blk, _ci(b, W)), b.sub(wave, _ci(b, W)))
        count = b.smax(b.div(b.add(b.sub(_ci(b, P), gw), _ci(b, TW - 1)), _ci(b, TW)), c0)
        w64 = dops.uniform(b, b.global_ptr_to_i64(wp))
        s64 = dops.uniform(b, b.global_ptr_to_i64(sp))
        v4 = VectorType(I32, 4)
        pending: List[Value] = []

        def load(base, off):
            pending.append(b.inline_asm("global_load_dwordx4 $0, $1, $2", "=&v,v,s", [off, base], v4))
            if len(pending) == _PF_BATCH:
                b.inline_asm("s_waitcnt vmcnt(0)", ",".join(["v"] * len(pending)), list(pending))
                pending.clear()

        for g in range(DG, DG + groups):
            it, gi = divmod(g, NG)
            ok = b.cmp_lt(_ci(b, it), count)
            panel = b.select(ok, b.add(gw, _ci(b, it * TW)), gw)
            col = b.add(b.mul(panel, _ci(b, _TILE_N)), c16)
            load(s64, b.add(b.mul(col, _ci(b, KS)), _ci(b, gi * 16)))
            wb = b.add(b.mul(col, _ci(b, KB)), b.mul(g4, _ci(b, 16)))
            for j in range(min(_GROUP_Q, h.nq - gi * _GROUP_Q)):
                load(w64, b.add(wb, _ci(b, (gi * _GROUP_Q + j) * 64)))
        if pending:
            b.inline_asm("s_waitcnt vmcnt(0)", ",".join(["v"] * len(pending)), list(pending))


def build_p2(spec: P2Spec) -> KernelDef:
    ok, why = is_valid_p2(spec)
    if not ok:
        raise ValueError(f"invalid P2Spec: {why}")
    h = spec.head_spec()
    K, G, W = h.k, h.ctas, h.waves
    P, PF = spec.prod_ctas, spec.pfirst

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    if h.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = int(h.waves_per_eu)
    xp = b.param("x", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    wp = b.param("w", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    sp = b.param("s", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    if h.epilogue == "logits":
        yp = b.param("y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=2)
    else:
        yp = b.param("y", PtrType(I32, "global"), noalias=True, writeonly=True, align=4)
    wsp = b.param("ws", PtrType(I32, "global"), noalias=True, align=16)
    xbp = b.param("xb", PtrType(I32, "global"), noalias=True, align=16)
    m_rows = b.param("m", I32)
    sxm = b.param("sxm", I32)
    sym = b.param("sym", I32)
    rev = b.param("rev", I32)
    delay = b.param("delay", I32)
    stp = b.param("st", PtrType(I64, "global"), noalias=True, align=8) if spec.stamps else None

    c0 = _ci(b, 0)
    c1 = _ci(b, 1)
    blk = b.block_id_x()
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.lshr(tid, _ci(b, 6)))
    ws_rsrc = b.buffer_rsrc(wsp, _ci(b, MX.fused_workspace_bytes(h)))
    xb_rsrc = b.buffer_rsrc(xbp, _ci(b, xbox_bytes(h.rows_max, K)))
    m_last = b.sub(m_rows, c1)
    x_rsrc = b.buffer_rsrc(xp, b.add(b.mul(b.mul(m_last, sxm), _ci(b, 2)), _ci(b, 2 * K)))

    # the launch epoch: loaded once at entry (before the delay and the weight prologue), as the true
    # mono loads its step at entry; its value is first needed at the producer's stores / the x poll
    ep_raw = b.buffer_load_cp(ws_rsrc, c0, c0, dwords=2, aux=CPOL_SC1)
    st_ptr = stp if spec.stamps else None

    def stamp(i: int) -> None:
        """thread 0: ``memrealtime`` into ``st[blk * 8 + i]`` (stamped builds only)"""
        if st_ptr is None:
            return
        with b.scf_if(b.cmp_eq(tid, c0)):
            b.global_store(st_ptr, b.add(b.mul(blk, _ci(b, 8)), _ci(b, i)), dops.memrealtime(b))

    stamp(0)

    def epoch_tag(stale: bool = False) -> Value:
        """``count / G + 1`` of the 64-bit arrival counter (``count / G`` for
        the stale-tag mutant): this launch's tag."""
        v = ep_raw
        lo = b.zext(b.vec_extract(v, 0), I64)
        hi = b.zext(b.vec_extract(v, 1), I64)
        cnt = b.lor(lo, b.shl(hi, b.const_i64(32)))
        e = b.trunc(b.div(cnt, b.const_i64(G)), I32)
        return b.readfirstlane(e if stale else b.add(e, c1))

    mbk = MB.MailboxKnobs(pause=spec.ppause)

    def poll_pairs(idxs, npairs, *, data):
        """Spin until the pairs ``idxs`` (npairs each; data region or ready pairs) carry this launch's
        tag; returns their payload words."""
        if spec.pimpl == "asm":
            base = b.global_ptr_add(xbp, _ci(b, XBOX_DATA_OFF)) if data else xbp
            mb = MB.mailbox_from_ptr(b, base, _ci(b, P2_XB_MAX), mbk)
            return MB.poll(b, [(mb, i, npairs) for i in idxs], poll_tag[0], mbk)
        specs = []
        for i in idxs:
            off = b.mul(i, _ci(b, 8))
            if data:
                off = b.add(off, _ci(b, XBOX_DATA_OFF))
            specs.append((xb_rsrc, off, 2 * npairs))
        got = b.tagged_poll(poll_tag[0], specs, aux=CPOL_SC1, pause=dops.PAUSES[spec.ppause])
        return [[b.vec_extract(v, 2 * q) for q in range(npairs)] for v in got]

    poll_tag = [None]

    # ---- producer stage
    if spec.xmode != "none":
        is_prod = b.land(b.cmp_ge(blk, _ci(b, PF)), b.cmp_lt(blk, _ci(b, PF + P)))
        with b.scf_if(is_prod):
            p = b.sub(blk, _ci(b, PF))
            _delay(b, delay)
            ptag = [None]

            def get_tag():
                ptag[0] = epoch_tag()

            if spec.xmode == "pairs":

                def store(row, c4, w0, w1):
                    # pairs (row K + c4) / 2 and + 1: one 16-byte sc1 store
                    tag = ptag[0]
                    off = b.add(_ci(b, XBOX_DATA_OFF), b.mul(b.add(b.mul(row, _ci(b, K)), c4), _ci(b, 4)))
                    b.buffer_store_cp(b.vec_pack([w0, tag, w1, tag], I32), xb_rsrc, off, c0, aux=CPOL_SC1)

            else:

                def store(row, c4, w0, w1):
                    off = b.add(_ci(b, XBOX_DATA_OFF), b.mul(b.add(b.mul(row, _ci(b, K)), c4), _ci(b, 2)))
                    b.buffer_store_cp(b.vec_pack([w0, w1], I32), xb_rsrc, off, c0, aux=0)

            _produce(
                b, k=K, rows_max=h.rows_max, threads=spec.threads, P=P, p=p,
                m_rows=m_rows, sxm=sxm, x_rsrc=x_rsrc, store=store, before_store=get_tag,
            )
            tag = ptag[0]
            if spec.xmode == "bulk":
                # the mono bulk rule: every thread drains, one thread releases + ready pair p
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare(fence="compiler")
                with b.scf_if(dops.is_thread0(b)):
                    dops.release_l2(b, "agent")
                    b.buffer_store_cp(b.vec_pack([c0, tag], I32), xb_rsrc, b.mul(p, _ci(b, 8)), c0, aux=CPOL_SC1)
            if spec.stamps:
                b.s_waitcnt(vmcnt=0)
                stamp(2)

    # ---- spare waves: L2 prefetch of the head waves' next load groups (l2pf)
    if spec.l2pf:
        _l2_prefetch(b, h, wp, sp, blk, wave, spec.l2pf)

    # ---- head stage
    stale = spec.mutant == "stale_tag"
    x_wait = x_stage = None
    if spec.xmode == "pairs":

        def x_stage(ctx):
            lane = ctx["lane"]
            XS = K + h.xpad
            xs = b.smem_alloc(BF16, [h.rows_max, XS], name_hint="xs")
            poll_tag[0] = epoch_tag(stale)
            if spec.hint:
                # wave 0: one pair per producer (its row-0 first pair), then the barrier
                with b.scf_if(b.cmp_eq(ctx["wave"], c0)):
                    pr = b.smin(lane, _ci(b, P - 1))
                    poll_pairs([b.mul(pr, _ci(b, K // P // 2))], 1, data=True)
                b.s_barrier_bare(fence="compiler")
            per_row = K // 8
            T = W * 64
            B = spec.pbatch
            nvalid = b.mul(m_rows, _ci(b, per_row))

            def batch(bt):
                specs, meta = [], []
                for j in range(B):
                    idx = b.add(ctx["tid"], b.mul(b.add(b.mul(bt, _ci(b, B)), _ci(b, j)), _ci(b, T)))
                    ok = b.cmp_lt(idx, nvalid)
                    idc = b.select(ok, idx, b.sub(nvalid, c1))
                    row = b.div(idc, _ci(b, per_row))
                    col = b.mul(b.mod(idc, _ci(b, per_row)), _ci(b, 8))
                    pi = b.lshr(b.add(b.mul(row, _ci(b, K)), col), c1)  # first of the item's 4 pairs
                    specs += [pi, b.add(pi, _ci(b, 2))]
                    meta.append((ok, row, col))
                got = poll_pairs(specs, 2, data=True)
                for j, (ok, row, col) in enumerate(meta):
                    words = [got[2 * j][0], got[2 * j][1], got[2 * j + 1][0], got[2 * j + 1][1]]
                    v = b.vec_bitcast(b.vec_pack(words, I32), VectorType(BF16, 8))
                    with b.scf_if(ok):
                        b.smem_store_vN(xs, [row, col], v, 8)

            if spec.ploop == "runtime":
                trips = b.div(b.add(nvalid, _ci(b, T * B - 1)), _ci(b, T * B))
                with b.scf_for(c0, trips, c1, iv_name="xbt") as bt:
                    batch(bt)
            else:
                for bt in range(-(-h.rows_max * per_row // (T * B))):
                    batch(_ci(b, bt))
            b.sync()
            return xs

    elif spec.xmode == "bulk":

        def x_wait(ctx):
            poll_tag[0] = epoch_tag(stale)
            with b.scf_if(b.cmp_eq(ctx["wave"], c0)):
                pr = b.smin(ctx["lane"], _ci(b, P - 1))
                poll_pairs([pr], 1, data=False)
                dops.buffer_inv(b, "agent")
            b.s_barrier_bare(fence="compiler")

    if spec.xmode == "bulk":
        hx = b.global_ptr_add(xbp, _ci(b, XBOX_DATA_OFF))
        hsxm = _ci(b, K)
    else:
        hx, hsxm = xp, sxm
    io = HeadIO(xp=hx, wp=wp, sp=sp, yp=yp, m_rows=m_rows, sxm=hsxm, sym=sym, rev=rev, wsp=wsp)
    better = _better_high if spec.mutant == "tie_high" else _better_low
    is_head = b.cmp_lt(blk, _ci(b, G))
    if h.threads < spec.threads:
        is_head = b.land(is_head, b.cmp_lt(wave, _ci(b, W)))
    with b.scf_if(is_head):
        emit_head(
            b, h, io, cta=blk, x_wait=x_wait, x_stage=x_stage, better=better,
            logits_arrive=h.epilogue == "logits",
            key_fn=_bf16_key_sampler if spec.nanrule == "sampler" else _bf16_key,
            hooks={"after_issue": lambda: stamp(1), "after_x": lambda: stamp(3),
                   "after_loop": lambda: stamp(4), "end": lambda: stamp(5)} if spec.stamps else None,
        )
    return b.kernel


def build_p2_producer(
    *, k: int, rows_max: int, grid: int = 256, threads: int = 256, prod_ctas: int = 10, prod_first: int = -1,
    stamps: bool = False,
) -> KernelDef:
    """The hybrid arm's producer launch: the same producer CTAs as
    :func:`build_p2`, plain stores of x ``[m, k]`` (row stride ``k``) for the
    next launch; every other CTA exits."""
    PF = grid - prod_ctas if prod_first < 0 else prod_first
    if k % (4 * prod_ctas) or PF < 0 or PF + prod_ctas > grid:
        raise ValueError("invalid producer geometry")
    b = IRBuilder(f"p2prod_k{k}_r{rows_max}_G{grid}_T{threads}_pc{prod_ctas}_pf{PF}" + ("_STAMP" if stamps else ""))
    b.kernel.attrs["max_workgroup_size"] = threads
    xp = b.param("x", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    xdp = b.param("xd", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    m_rows = b.param("m", I32)
    sxm = b.param("sxm", I32)
    delay = b.param("delay", I32)
    stp = b.param("st", PtrType(I64, "global"), noalias=True, align=8) if stamps else None
    c0 = _ci(b, 0)
    blk = b.block_id_x()

    def stamp(i: int) -> None:
        if stp is None:
            return
        with b.scf_if(b.cmp_eq(b.thread_id_x(), c0)):
            b.global_store(stp, b.add(b.mul(blk, _ci(b, 8)), _ci(b, i)), dops.memrealtime(b))

    stamp(0)
    m_last = b.sub(m_rows, _ci(b, 1))
    x_rsrc = b.buffer_rsrc(xp, b.add(b.mul(b.mul(m_last, sxm), _ci(b, 2)), _ci(b, 2 * k)))
    xd_rsrc = b.buffer_rsrc(xdp, _ci(b, rows_max * k * 2))
    is_prod = b.land(b.cmp_ge(blk, _ci(b, PF)), b.cmp_lt(blk, _ci(b, PF + prod_ctas)))
    with b.scf_if(is_prod):
        p = b.sub(blk, _ci(b, PF))
        _delay(b, delay)

        def store(row, c4, w0, w1):
            off = b.mul(b.add(b.mul(row, _ci(b, k)), c4), _ci(b, 2))
            b.buffer_store_cp(b.vec_pack([w0, w1], I32), xd_rsrc, off, c0, aux=0)

        _produce(
            b, k=k, rows_max=rows_max, threads=threads, P=prod_ctas, p=p,
            m_rows=m_rows, sxm=sxm, x_rsrc=x_rsrc, store=store,
        )
        if stamps:
            b.s_waitcnt(vmcnt=0)
            stamp(2)
    return b.kernel
