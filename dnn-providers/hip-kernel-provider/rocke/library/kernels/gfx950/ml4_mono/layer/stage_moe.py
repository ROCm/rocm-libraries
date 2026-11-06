# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stages ``router``, ``ug`` and ``down``: the MoE half of the layer.

``router`` task ``(s, t)``: RMSNorm of row ``s``'s post-attention state ``a``
with ``g_post``, the logits of experts ``8 t .. 8 t + 7`` (bf16 GEMV, K folded
over the MFMA's two halves) to ``scores``, and MXFP8 blocks of the normed
activation (one 128-block per wave of the first three) to ``xq`` / ``xqs``.

``ug`` (every CTA, three task rounds): each CTA routes every row (top-4 of
softmax over the selected logits, plus the shared expert in slot 0), stages
the MXFP8 activation, and per round computes 8 intermediate columns (gate and
up rows) of one routed slot for every row, the first 192 tasks also the
shared expert's for all rows at once, and publishes SiLU(gate) * up to
``mid``. MXFP4 weights on the scaled MFMA.

``down`` task ``t``: 24 hidden rows: the mid activations re-quantized to
MXFP8, the shared expert's rows (all rows in one MFMA) then the routed
slots', each scaled by its route weight, the TP sum, and the residual ``a``
into ``x_out``.
"""

from __future__ import annotations

import math

from rocke.core.ir import F32, I32, I64, PtrType, Value, VectorType
from rocke.helpers import device_ops as dops
from rocke.helpers.task_schedule import task_index

from .ctx import HIDDEN, MOE_SLOTS, WAVES, XQ_S_STRIDE, XQ_X_STRIDE, Ml4Ctx
from .gemv import (
    emit_out,
    load_x_rmsnorm,
    mma_units,
    mxfp4_scale_pair,
    n_sel,
    reduce_rows,
    run_units,
    scale_byte,
    stage_x_rmsnorm,
    unit_bf16,
    unit_native_mxfp4,
    zero4,
)
from .stage_uv import peer_reduce, resid_slot

N_EXPERTS = 128
TOP_K = 4
SHARED_EXPERT = N_EXPERTS
ROUTER_TILE = 8
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
UG8 = 8
FP8_MAX = 448.0
LOG2E = 1.4426950408889634
ROUTE_SCALE = 1.0
# parameter table rows of the shared expert's w_ug, s_ug, w_dn, s_dn
T_SHARED = 8


# ------------------------------------------------------------------ helpers
def _exp(cx: Ml4Ctx, x: Value) -> Value:
    b = cx.b
    return b.exp2_fast(b.fmul(x, cx.f(LOG2E)))


def _silu_mul(cx: Ml4Ctx, g: Value, u: Value) -> Value:
    b = cx.b
    e = _exp(cx, b.fmul(g, cx.f(-1.0)))
    return b.fmul(b.fmul(g, b.rcp_fast(b.fadd(cx.f(1.0), e))), u)


def _sat(cx: Ml4Ctx, x: Value) -> Value:
    b = cx.b
    return b.fmin(b.fmax(x, cx.f(-FP8_MAX)), cx.f(FP8_MAX))


def quant_mxfp8(cx: Ml4Ctx, a0: Value, a1: Value):
    """Per-32 E4M3 values and E8M0 exponent: 16 lanes hold one 32-block."""
    b = cx.b
    amax = b.fmax(b.fabs(a0), b.fabs(a1))
    for off in (1, 2, 4, 8):
        amax = cx.xred(amax, off, "max")
    bits = b.bitcast(b.fmul(amax, cx.f(1.0 / FP8_MAX)), I32)
    exponent = b.land(b.lshr(bits, cx.i(23)), cx.i(0xFF))
    frac = b.cmp_ne(b.land(bits, cx.i(0x7FFFFF)), cx.i(0))
    exponent = b.add(exponent, b.select(frac, cx.i(1), cx.i(0)))
    exponent = b.smax(b.smin(exponent, cx.i(254)), cx.i(1))
    inverse = b.rcp_fast(b.bitcast(b.shl(exponent, cx.i(23)), F32))
    return _sat(cx, b.fmul(a0, inverse)), _sat(cx, b.fmul(a1, inverse)), exponent


def st_mxfp8(cx: Ml4Ctx, k: Value, q0: Value, q1: Value) -> None:
    """LDS MXFP8 bytes k, k + 1 (k even; lane ^ 1 holds k ^ 2), row-major."""
    b = cx.b
    w = cx.fp8_pair_bits(q0, q1)
    nb = cx.xshfl(w, 1)
    with b.scf_if(b.cmp_eq(b.land(cx.lane, cx.i(1)), cx.i(0))):
        cx.lds_st(b.add(b.lshr(k, cx.i(2)), cx.i(cx.xs)), b.lor(w, b.shl(nb, cx.i(16))))


def _key_signed(cx: Ml4Ctx, k: Value) -> Value:
    """An unsigned key as a signed one of the same order."""
    return cx.b.xor(k, cx.i(-(2**31)))


def route_top8(cx: Ml4Ctx, s, bs):
    """Top-4 of row ``s`` (one whole wave, after the router logits landed).
    Ties take the lower expert id. Returns (expert id, softmax weight over the
    picks) of pick ``lane`` in score order, valid in lanes < TOP_K."""
    b = cx.b
    s = cx.i(s) if isinstance(s, int) else s
    nc = N_EXPERTS // 64
    got = cx.poll([("scores", b.add(b.mul(s, cx.i(N_EXPERTS)), b.add(cx.lane, cx.i(64 * i))), 1)
                   for i in range(nc)])  # fmt: skip
    raws = [b.bitcast(g[0], F32) for g in got]
    ks = []
    for i in range(nc):
        kb = b.bitcast(b.fadd(raws[i], bs[i]), I32)
        ks.append(b.select(b.cmp_ge(kb, cx.i(0)), b.xor(kb, cx.i(-(2**31))), b.xor(kb, cx.i(-1))))
    ids = [b.add(cx.lane, cx.i(64 * i)) for i in range(nc)]
    if nc != 2:
        raise NotImplementedError("route_top8 sorts two candidates per lane")
    first = b.lor(
        b.cmp_lt(_key_signed(cx, ks[1]), _key_signed(cx, ks[0])),
        b.land(b.cmp_eq(ks[0], ks[1]), b.cmp_lt(ids[0], ids[1])),
    )
    ka, kb_, ia, ib = ks[0], ks[1], ids[0], ids[1]
    ks = [b.select(first, ka, kb_), b.select(first, kb_, ka), cx.i(0)]
    ids = [b.select(first, ia, ib), b.select(first, ib, ia), cx.i(N_EXPERTS)]
    mv = cx.i(0)
    for k in range(TOP_K):
        m = dops.wave_umax_dpp(b, ks[0])
        winner = dops.wave_umax_dpp(b, b.select(b.cmp_eq(ks[0], m), b.sub(cx.i(255), ids[0]), cx.i(0)))
        expert = b.sub(cx.i(255), winner)
        hit = b.land(b.cmp_eq(ks[0], m), b.cmp_eq(ids[0], expert))
        ks = [b.select(hit, ks[i + 1], ks[i]) for i in range(nc)] + [ks[nc]]
        ids = [b.select(hit, ids[i + 1], ids[i]) for i in range(nc)] + [ids[nc]]
        mv = b.writelane(expert, cx.i(k), mv)
    e = mv
    src = b.land(e, cx.i(63))
    picked = [cx.bpermute(src, b.bitcast(r, I32)) for r in raws]
    raw = picked[0]
    for i in range(1, nc):
        raw = b.select(b.cmp_eq(b.lshr(e, cx.i(6)), cx.i(i)), picked[i], raw)
    valid = b.cmp_lt(cx.lane, cx.i(TOP_K))
    lg = b.select(valid, b.bitcast(raw, F32), cx.f(-1.0e30))
    mx = lg
    for off in (1, 2, 4):
        mx = cx.xred(mx, off, "max")
    ex = b.select(valid, _exp(cx, b.fsub(lg, mx)), cx.f(0.0))
    tot = ex
    for off in (1, 2, 4):
        tot = cx.xred(tot, off, "add")
    return e, b.fmul(ex, b.fmul(b.rcp_fast(tot), cx.f(ROUTE_SCALE)))


def load_bias(cx: Ml4Ctx) -> list[Value]:
    r = cx.rsrc("bias")
    return [cx.ld_f32(r, cx.b.add(cx.lane, cx.i(64 * i))) for i in range(N_EXPERTS // 64)]


def dn_route(cx: Ml4Ctx, bs) -> None:
    """Every row's slots into LDS: ``keys[s * 5 + q]`` the expert, ``dnw`` its
    weight; slot 0 the shared expert (weight 1), slots 1-4 the picks."""
    b = cx.b
    S = cx.spec.rows
    if S > WAVES:
        raise NotImplementedError("one routing wave per row")
    with b.scf_if(b.cmp_lt(cx.wave, cx.i(S))):
        e, w = route_top8(cx, cx.wave, bs)
        with b.scf_if(b.cmp_lt(cx.lane, cx.i(MOE_SLOTS))):
            q = b.add(b.mul(cx.wave, cx.i(MOE_SLOTS)), b.mod(b.add(cx.lane, cx.i(1)), cx.i(MOE_SLOTS)))
            shared = b.cmp_eq(cx.lane, cx.i(TOP_K))
            cx.lds_st(b.add(q, cx.i(cx.keys)), b.select(shared, cx.i(SHARED_EXPERT), e))
            cx.lds_st(b.add(q, cx.i(cx.dnw)), b.select(shared, cx.f(1.0), w))


def expert_addr(cx: Ml4Ctx, base: str, e: Value, stride: int, which: int) -> Value:
    """Address of expert ``e`` of expert tensor ``base`` (the shared expert's
    own tensor from the parameter table)."""
    b = cx.b
    return b.select(b.cmp_eq(e, cx.i(SHARED_EXPERT)), cx.table_addr(T_SHARED + which),
                    routed_addr(cx, base, e, stride))  # fmt: skip


def routed_addr(cx: Ml4Ctx, base: str, e: Value, stride: int) -> Value:
    """Address of routed expert ``e`` (a top-k pick: never the shared one, so
    no parameter-table load sits in front of its weight loads)."""
    b = cx.b
    return b.add(cx.arg[base], b.mul(b.zext(e, I64), b.const_i64(stride)))


def _uniform_i32(cx: Ml4Ctx, word: Value) -> Value:
    return cx.b.readfirstlane(cx.lds_ld(word))


def _uniform_f32(cx: Ml4Ctx, word: Value) -> Value:
    return cx.b.bitcast(cx.b.readfirstlane(cx.lds_ld(word)), F32)


# ------------------------------------------------------------------ router
def emit_router(cx: Ml4Ctx) -> None:
    cx.tasks("router", lambda tt: _router_task(cx, tt))


def _router_task(cx: Ml4Ctx, tt: Value) -> None:
    b = cx.b
    t = b.land(tt, cx.i(N_ROUTER - 1))
    rs = b.lshr(tt, cx.i(N_ROUTER.bit_length() - 1))
    nkc = HIDDEN // 64
    cpw = nkc // WAVES // 2
    r_sub = b.land(b.mul(t, cx.i(ROUTER_TILE)), cx.i(15))
    r_ln = b.lor(b.land(cx.lane, cx.i(-16)), b.add(r_sub, b.land(cx.lane, cx.i(ROUTER_TILE - 1))))
    r_fold = b.lshr(b.land(cx.lane, cx.i(15)), cx.i(3))
    r_w = cx.rsrc("w_r")
    rg = b.lshr(b.mul(t, cx.i(ROUTER_TILE)), cx.i(4))

    def u_r(c: int):
        kc = b.add(b.add(b.mul(cx.wave, cx.i(nkc // WAVES)), b.mul(r_fold, cx.i(cpw))), cx.i(c))
        return unit_bf16(cx, r_w, rg, kc, nkc, b.mul(kc, cx.i(32)), r_ln)

    pre = [u_r(c) for c in range(cpw)]
    cx.barrier()
    r_gp = cx.rsrc("g_post")
    x_blk = b.add(b.mul(cx.wave, cx.i(N_ROUTER)), t)
    x_ok = b.land(b.cmp_lt(cx.wave, cx.i(XQ_WAVES)), b.cmp_lt(x_blk, cx.i(XQ_BLOCKS)))
    xk = b.add(b.mul(b.smin(x_blk, cx.i(XQ_BLOCKS - 1)), cx.i(128)), b.mul(cx.lane, cx.i(2)))
    xg = (cx.ld_bf16(r_gp, xk), cx.ld_bf16(r_gp, b.add(xk, cx.i(1))))
    xa = []
    row0 = b.mul(rs, cx.i(HIDDEN))

    def ld_a(sks):
        specs = [("a", b.lshr(b.add(row0, k), cx.i(1)), 2) for _, k in sks]
        specs.append(("a", b.lshr(b.add(row0, xk), cx.i(1)), 1))
        v = cx.poll(specs)
        xa.append(cx.bf2_f32(v[-1][0]))
        return [list(cx.bf2_f32(w[0])) + list(cx.bf2_f32(w[1])) for w in v[:-1]]

    rstds = stage_x_rmsnorm(cx, load_x_rmsnorm(cx, ld_a, HIDDEN, "g_post", 1), HIDDEN, 1)
    with b.scf_if(x_ok):
        a0, a1 = xa[0]
        q0, q1, qs = quant_mxfp8(cx, b.fmul(b.fmul(a0, rstds[0]), xg[0]),
                                 b.fmul(b.fmul(a1, rstds[0]), xg[1]))  # fmt: skip
        w8 = cx.fp8_pair_bits(q0, q1)
        w8n = cx.xshfl(w8, 1)
        with b.scf_if(b.cmp_eq(b.land(cx.lane, cx.i(1)), cx.i(0))):
            cx.put("xq", b.lshr(b.add(row0, xk), cx.i(2)), b.lor(w8, b.shl(w8n, cx.i(16))))
        d0, d1 = cx.fp8_roundtrip(q0, q1)
        qscale = b.bitcast(b.shl(qs, cx.i(23)), F32)
        vec = b.vec_pack([b.fmul(d0, qscale), b.fmul(d1, qscale)], F32)
        b.buffer_store_vN_f32(cx.mb_rsrc("xqd"), b.mul(b.add(row0, xk), cx.i(4)), cx.i(0), vec, 2)
        with b.scf_if(b.cmp_eq(b.land(cx.lane, cx.i(15)), cx.i(0))):
            gi = b.add(b.add(b.mul(rs, cx.i(XQ_GROUPS)), b.mul(x_blk, cx.i(4))), b.lshr(cx.lane, cx.i(4)))
            cx.put("xqs", gi, qs)
    cx.barrier()
    acc = run_units(cx, u_r, cpw, cpw, pre)
    base = b.mul(b.add(b.mul(cx.wave, cx.i(64)), cx.lane), cx.i(4))
    cx.lds_st4(b.add(base, cx.i(cx.red)), b.bitcast(b.vec_pack(list(acc), F32), VectorType(I32, 4)))
    cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(ROUTER_TILE))):
        r = cx.tid
        logit = cx.f(0.0)
        for w in range(WAVES):
            for f in range(2):
                m = b.add(cx.i(f * ROUTER_TILE), r)
                lane_of = b.add(cx.i(f * ROUTER_TILE), b.mul(b.lshr(m, cx.i(2)), cx.i(16)))
                word = b.add(b.mul(b.add(cx.i(w * 64), lane_of), cx.i(4)), b.land(m, cx.i(3)))
                logit = b.fadd(logit, cx.lds_ldf(b.add(word, cx.i(cx.red))))
        cx.put("scores", b.add(b.add(b.mul(rs, cx.i(N_EXPERTS)), b.mul(t, cx.i(ROUTER_TILE))), r), logit)


# ------------------------------------------------------------------ up/gate
def _ug_slot(cx: Ml4Ctx, u: Value, has_sh: Value, ug_c: int) -> Value:
    """Routed slot of up/gate task ``u`` (``ug_c`` column tasks per slot; the
    first ``ug_c`` also take the shared expert)."""
    b = cx.b
    if cx.spec.ug_slot_order:
        return b.add(b.div(u, cx.i(ug_c)), cx.i(1))
    return b.select(has_sh, cx.i(MOE_SLOTS - 1), b.div(u, cx.i(ug_c)))


def stage_xq(cx: Ml4Ctx) -> None:
    """Every row's MXFP8 activation (``xq``) and exponents (``xqs``) into LDS:
    row ``j`` at words ``j * XQ_X_STRIDE``, its scales at ``misc + 8 + j *
    XQ_S_STRIDE``."""
    b = cx.b
    S = cx.spec.rows
    nxw = HIDDEN // 4 // 512
    specs = []
    for sx in range(S):
        for i in range(nxw):
            specs.append(("xq", b.add(cx.i(sx * (HIDDEN // 4) + i * 512), cx.tid), 1))
    for sx in range(S):
        specs.append(("xqs", b.add(cx.i(sx * XQ_GROUPS), b.smin(cx.tid, cx.i(XQ_GROUPS - 1))), 1))
    got = cx.poll(specs)
    for j in range(S):
        for i in range(nxw):
            cx.lds_st(b.add(cx.tid, cx.i(cx.xs + j * XQ_X_STRIDE + i * 512)), got[j * nxw + i][0])
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(XQ_GROUPS))):
        for j in range(S):
            cx.lds_st(b.add(cx.tid, cx.i(cx.misc + 8 + j * XQ_S_STRIDE)), got[S * nxw + j][0])


def emit_ug(cx: Ml4Ctx) -> None:
    b = cx.b
    S = cx.spec.rows
    inter = cx.spec.inter
    if S < 2:
        raise NotImplementedError("the one-row up/gate schedule is not ported yet")
    ug_c = inter // UG8
    tasks = TOP_K * ug_c
    if tasks % cx.spec.grid:
        raise ValueError("the up/gate schedule needs whole CTA rounds")
    units_per_wave = (HIDDEN // 128) // WAVES
    w_bytes = 2 * inter * HIDDEN // 2
    s_bytes = 2 * inter * (HIDDEN // 32)
    u0 = task_index(b, cx.table, "ug", cx.bid, 0)
    rounds = tasks // cx.spec.grid
    if cx.spec.ug_batch:
        _ug_batched(cx, u0, rounds, ug_c, units_per_wave, w_bytes, s_bytes)
        return
    cx.barrier()
    dn_route(cx, load_bias(cx))
    cx.barrier()
    stage_xq(cx)
    cx.barrier()
    for rnd in range(rounds):
        u = b.add(u0, cx.i(rnd * cx.spec.grid))
        cx.stamp("ug", u, 0)
        c = b.mod(u, cx.i(ug_c))
        has_sh = b.cmp_lt(u, cx.i(ug_c))
        slot = _ug_slot(cx, u, has_sh, ug_c)
        w_rg = b.add(b.mul(b.lshr(b.land(cx.lane, cx.i(15)), cx.i(3)), cx.i(inter // 16)), b.lshr(c, cx.i(1)))
        w_ln = b.lor(b.land(cx.lane, cx.i(-16)),
                     b.add(b.mul(b.land(c, cx.i(1)), cx.i(8)), b.land(cx.lane, cx.i(7))))  # fmt: skip

        def ug8_units(e, sample, live=None):
            """``e`` None: the shared expert (all row columns); else routed."""
            if e is None:
                aw, as_ = cx.table_addr(T_SHARED), cx.table_addr(T_SHARED + 1)
            else:
                aw = routed_addr(cx, "w_ug", e, w_bytes)
                as_ = routed_addr(cx, "s_ug", e, s_bytes)
            rw = _expert_rsrc(cx, aw, w_bytes, live)
            rs_ = _expert_rsrc(cx, as_, s_bytes, live)
            sn = n_sel(cx, S) if sample is None else cx.i(sample)
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
                ug_scale = scale_byte(cx, pair_word, shift, cx.spec.ug_scale_late)
                b_word = b.add(b.mul(sn, cx.i(XQ_X_STRIDE)), b.mul(unit, cx.i(32)))
                s_word = b.add(b.add(cx.i(8), b.mul(sn, cx.i(XQ_S_STRIDE))),
                               b.add(b.mul(unit, cx.i(4)), b.lshr(cx.lane, cx.i(4))))  # fmt: skip
                units.append(unit_native_mxfp4(cx, rw, rs_, w_rg, unit, HIDDEN, b_word, s_word,
                                               ln=w_ln, scale=ug_scale))  # fmt: skip
            return units

        def ug8_emit(sample: int, shared: bool) -> None:
            n_rows = S if shared else 1
            with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_rows * UG8 // 2))):
                n = b.lshr(cx.tid, cx.i(2))
                r = b.mul(b.land(cx.tid, cx.i(3)), cx.i(2))
                o = b.add(b.mul(n, cx.i(16)), r)
                g0 = cx.lds_ldf(b.add(o, cx.i(cx.outs)))
                g1 = cx.lds_ldf(b.add(o, cx.i(cx.outs + 1)))
                v0 = cx.lds_ldf(b.add(o, cx.i(cx.outs + UG8)))
                v1 = cx.lds_ldf(b.add(o, cx.i(cx.outs + UG8 + 1)))
                sn = n if shared else cx.i(sample)
                sl = cx.i(0) if shared else slot
                at = b.add(b.add(b.mul(b.add(b.mul(sn, cx.i(MOE_SLOTS)), sl), cx.i(inter)),
                                 b.mul(c, cx.i(UG8))), r)  # fmt: skip
                cx.put2("mid", at, _silu_mul(cx, g0, v0), _silu_mul(cx, g1, v1))
            lead = b.cmp_lt(cx.tid, cx.i(S)) if shared else b.cmp_eq(cx.tid, cx.i(0))
            with b.scf_if(b.land(b.cmp_eq(c, cx.i(0)), lead)):
                sn = cx.tid if shared else cx.i(sample)
                sl = cx.i(0) if shared else slot
                q = b.add(b.mul(sn, cx.i(MOE_SLOTS)), sl)
                cx.put("sel", q, cx.lds_ld(b.add(q, cx.i(cx.keys))))
                cx.put("prob", q, cx.lds_ld(b.add(q, cx.i(cx.dnw))))

        if rnd == 0:
            shared_pre = ug8_units(None, None, has_sh)

        def expert(sample: int) -> Value:
            return _uniform_i32(cx, b.add(slot, cx.i(cx.keys + sample * MOE_SLOTS)))

        queue = [ug8_units(expert(0), 0)]
        if rnd == 0:
            with b.scf_if(has_sh):
                reduce_rows(cx, 1, mma_units(cx, zero4(cx), shared_pre), emit_out(cx, 16), S)
                cx.barrier()
                ug8_emit(0, True)
        for sample in range(S):
            pre = queue.pop(0)
            if sample + 1 < S:
                queue.append(ug8_units(expert(sample + 1), sample + 1))
            acc = mma_units(cx, zero4(cx), pre)
            reduce_rows(cx, 1, acc, emit_out(cx, 16), S)
            cx.barrier()
            ug8_emit(sample, False)
        cx.stamp("ug", u, 4)
        if rnd + 1 < rounds:
            cx.barrier()


def _ug_batched(cx: Ml4Ctx, u0, rounds, ug_c, units_per_wave, w_bytes, s_bytes) -> None:
    """The up/gate rounds with every row's routed tile of a round issued,
    multiplied and reduced together, the shared expert folded into round 0's
    reduction, and round r + 1's weight loads issued as soon as round r's
    MFMAs are done (they stream during its reduction and emit)."""
    b = cx.b
    S = cx.spec.rows
    inter = cx.spec.inter

    def params(rnd: int) -> dict:
        u = b.add(u0, cx.i(rnd * cx.spec.grid))
        c = b.mod(u, cx.i(ug_c))
        has_sh = b.cmp_lt(u, cx.i(ug_c))
        slot = _ug_slot(cx, u, has_sh, ug_c)
        w_rg = b.add(b.mul(b.lshr(b.land(cx.lane, cx.i(15)), cx.i(3)), cx.i(inter // 16)), b.lshr(c, cx.i(1)))
        w_ln = b.lor(b.land(cx.lane, cx.i(-16)),
                     b.add(b.mul(b.land(c, cx.i(1)), cx.i(8)), b.land(cx.lane, cx.i(7))))  # fmt: skip
        return dict(u=u, c=c, has_sh=has_sh, slot=slot, w_rg=w_rg, w_ln=w_ln)

    def issue(p: dict, e, sample, live=None):
        """One row's (or, ``e`` None, the shared expert's) weight tile."""
        if e is None:
            aw, as_ = cx.table_addr(T_SHARED), cx.table_addr(T_SHARED + 1)
        else:
            aw = routed_addr(cx, "w_ug", e, w_bytes)
            as_ = routed_addr(cx, "s_ug", e, s_bytes)
        rw = _expert_rsrc(cx, aw, w_bytes, live)
        rs_ = _expert_rsrc(cx, as_, s_bytes, live)
        sn = n_sel(cx, S) if sample is None else cx.i(sample)
        units = []
        pair_word = pair_byte = None
        for cc in range(units_per_wave):
            unit = b.add(b.mul(cx.wave, cx.i(units_per_wave)), cx.i(cc))
            if cc % 2 == 0:
                row = b.add(b.mul(p["w_rg"], cx.i(16)), b.land(p["w_ln"], cx.i(15)))
                col = b.add(b.mul(unit, cx.i(4)), b.lshr(p["w_ln"], cx.i(4)))
                word, pair_byte = mxfp4_scale_pair(cx, row, col, HIDDEN // 32)
                pair_word = cx.ld_i32(rs_, word)
            shift = b.mul(b.add(pair_byte, cx.i((cc % 2) * 2)), cx.i(8))
            scale = scale_byte(cx, pair_word, shift, cx.spec.ug_scale_late)
            b_word = b.add(b.mul(sn, cx.i(XQ_X_STRIDE)), b.mul(unit, cx.i(32)))
            s_word = b.add(b.add(cx.i(8), b.mul(sn, cx.i(XQ_S_STRIDE))),
                           b.add(b.mul(unit, cx.i(4)), b.lshr(cx.lane, cx.i(4))))  # fmt: skip
            units.append(unit_native_mxfp4(cx, rw, rs_, p["w_rg"], unit, HIDDEN, b_word, s_word,
                                           ln=p["w_ln"], scale=scale))  # fmt: skip
        return units

    def routed_tiles(p: dict, rows=None):
        rows = range(S) if rows is None else rows
        experts = {s: _uniform_i32(cx, b.add(p["slot"], cx.i(cx.keys + s * MOE_SLOTS))) for s in rows}
        return [issue(p, experts[s], s) for s in rows]

    ps = [params(r) for r in range(rounds)]
    # rows of round r + 1 whose tiles are issued right after round r's MFMAs
    # (the others at the start of round r + 1)
    ahead = min(cx.spec.ug_prefetch_rows or S, S) if cx.spec.ug_prefetch else 0
    # round 0: the shared expert's tile joins the same reduction; its loads
    # read zeros on the CTAs without a shared task. It needs no routing, so
    # its loads can stream while this CTA routes and stages the activation.
    first = cx.spec.ug_shared_first
    shared_pre = issue(ps[0], None, None, ps[0]["has_sh"]) if first else None
    cx.barrier()
    dn_route(cx, load_bias(cx))
    cx.barrier()
    # ug_routed_early: round 0's routed tiles need the routing only, not the
    # staged activation, so they can stream while it stages
    routed = routed_tiles(ps[0]) if cx.spec.ug_routed_early else None
    stage_xq(cx)
    cx.barrier()
    if not first:
        shared_pre = issue(ps[0], None, None, ps[0]["has_sh"])
    if routed is None:
        routed = routed_tiles(ps[0])
    for rnd in range(rounds):
        p = ps[rnd]
        cx.stamp("ug", p["u"], 0)
        if rnd and ahead < S:
            routed = routed + routed_tiles(p, range(ahead, S))
        shared = (mma_units(cx, zero4(cx), shared_pre), p["has_sh"]) if rnd == 0 else None
        accs = [mma_units(cx, zero4(cx), u) for u in routed]
        routed = routed_tiles(ps[rnd + 1], range(ahead)) if rnd + 1 < rounds else []
        _ug_rows_batched(cx, accs, p["slot"], p["c"], shared)
        cx.stamp("ug", p["u"], 4)
        if rnd + 1 < rounds:
            cx.barrier()


# every row's routed up/gate tile is reduced at once in this LDS span past the
# staged activations (free during the up/gate rounds)
UG_RED_OFF = 8192


def _ug_rows_batched(cx: Ml4Ctx, accs, slot: Value, c: Value, shared=None) -> None:
    """Every row's routed tile of this round in one reduction and one emit:
    row ``s``'s 16 outputs sum its own column over waves 0..7, as the per-row
    reduction does, and its SiLU(gate) * up pairs go to ``mid``.
    ``shared = (acc, has_sh)``: the shared expert's tile (every row a column),
    reduced in the same pass and published where ``has_sh``."""
    b = cx.b
    S = cx.spec.rows
    inter = cx.spec.inter
    tiles = list(accs) + ([shared[0]] if shared is not None else [])
    if cx.xs + UG_RED_OFF + len(tiles) * 2048 > cx.keys or UG_RED_OFF < S * XQ_X_STRIDE:
        raise ValueError("the batched up/gate reduction does not fit beside the activation")
    base = b.mul(b.add(b.mul(cx.wave, cx.i(64)), cx.lane), cx.i(4))
    for i, acc in enumerate(tiles):
        word = b.add(base, cx.i(cx.xs + UG_RED_OFF + i * 2048))
        cx.lds_st4(word, b.bitcast(b.vec_pack(list(acc), F32), VectorType(I32, 4)))
    cx.barrier()
    # threads [0, 16 S): routed row s output r (column 0 of tile s); then the
    # shared tile's output r of column n
    n_out = 16 * S * (2 if shared is not None else 1)
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(n_out))):
        tile = b.smin(b.lshr(cx.tid, cx.i(4)), cx.i(S))
        col = b.select(b.cmp_lt(cx.tid, cx.i(16 * S)), cx.i(0), b.sub(b.lshr(cx.tid, cx.i(4)), cx.i(S)))
        r = b.land(cx.tid, cx.i(15))
        lane_of = b.add(col, b.mul(b.lshr(r, cx.i(2)), cx.i(16)))
        red = b.add(b.mul(tile, cx.i(2048)), cx.i(cx.xs + UG_RED_OFF))
        tot = cx.f(0.0)
        for w in range(WAVES):
            word = b.add(b.mul(b.add(cx.i(w * 64), lane_of), cx.i(4)), b.land(r, cx.i(3)))
            tot = b.fadd(tot, cx.lds_ldf(b.add(red, word)))
        cx.lds_st(b.add(cx.tid, cx.i(cx.outs)), tot)
    cx.barrier()

    def emit(t: Value, out0: int, sl: Value) -> None:
        s = b.lshr(t, cx.i(2))
        r = b.mul(b.land(t, cx.i(3)), cx.i(2))
        o = b.add(b.add(b.mul(s, cx.i(16)), r), cx.i(cx.outs + out0))
        g0, g1 = cx.lds_ldf(o), cx.lds_ldf(b.add(o, cx.i(1)))
        v0, v1 = cx.lds_ldf(b.add(o, cx.i(UG8))), cx.lds_ldf(b.add(o, cx.i(UG8 + 1)))
        at = b.add(b.add(b.mul(b.add(b.mul(s, cx.i(MOE_SLOTS)), sl), cx.i(inter)),
                         b.mul(c, cx.i(UG8))), r)  # fmt: skip
        cx.put2("mid", at, _silu_mul(cx, g0, v0), _silu_mul(cx, g1, v1))

    def record(sl: Value) -> None:
        q = b.add(b.mul(cx.tid, cx.i(MOE_SLOTS)), sl)
        cx.put("sel", q, cx.lds_ld(b.add(q, cx.i(cx.keys))))
        cx.put("prob", q, cx.lds_ld(b.add(q, cx.i(cx.dnw))))

    with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * UG8 // 2))):
        emit(cx.tid, 0, slot)
    first = b.land(b.cmp_eq(c, cx.i(0)), b.cmp_lt(cx.tid, cx.i(S)))
    with b.scf_if(first):
        record(slot)
    if shared is not None:
        has_sh = shared[1]
        lo, hi = S * UG8 // 2, 2 * S * UG8 // 2
        mine = b.land(b.cmp_ge(cx.tid, cx.i(lo)), b.cmp_lt(cx.tid, cx.i(hi)))
        with b.scf_if(b.land(has_sh, mine)):
            emit(b.sub(cx.tid, cx.i(lo)), 16 * S, cx.i(0))
        with b.scf_if(b.land(has_sh, first)):
            record(cx.i(0))


def _expert_rsrc(cx: Ml4Ctx, addr: Value, nbytes: int, live=None) -> Value:
    """Resource over one expert tensor; with ``live`` false its loads read 0."""
    if live is None:
        return cx.rsrc_at(addr)
    b = cx.b
    ptr = b.int_to_ptr(addr, PtrType(I32, "global"))
    return b.buffer_rsrc(ptr, b.select(live, cx.i(nbytes), cx.i(0)))


# ------------------------------------------------------------------ down
def dn_tile(rows: int) -> int:
    return 32 if rows == 1 else HIDDEN // 256


def emit_down(cx: Ml4Ctx) -> None:
    cx.tasks("down", lambda t: _down_task(cx, t))


def _down_task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    S = cx.spec.rows
    inter = cx.spec.inter
    tile = dn_tile(S)
    nkc = inter // 64
    r_groups = (tile + 15) // 16
    wpr = WAVES // r_groups
    w_bytes = HIDDEN * inter // 2
    s_bytes = HIDDEN * (inter // 32)
    shared = S > 1 and S % wpr == 0
    if not shared:
        raise NotImplementedError("down without the hoisted shared expert is not ported yet")
    spw = S // wpr
    batch = cx.spec.dn_batch or (9 if S <= 4 else 2)
    if cx.spec.dn_reroute or "ug" not in cx.spec.stages:
        # in a launch with up/gate, its prologue left this CTA's routing in LDS
        dn_route(cx, load_bias(cx))
        cx.barrier()
    gu = b.lshr(cx.wave, cx.i(wpr.bit_length() - 1))
    dn_rg = b.lshr(b.mul(t, cx.i(tile)), cx.i(4))
    dn_off = b.land(b.mul(t, cx.i(tile)), cx.i(15))
    dn_lr = b.sub(b.add(b.mul(gu, cx.i(16)), b.land(cx.lane, cx.i(15))), dn_off)
    in_tile = b.land(b.cmp_ge(dn_lr, cx.i(0)), b.cmp_lt(dn_lr, cx.i(tile)))
    dn_ln = b.select(in_tile, cx.lane, b.xor(cx.lane, cx.i(8)))
    rg = b.add(dn_rg, gu)
    slots = TOP_K
    n_units = S * slots * nkc // 2
    upw = (n_units + wpr - 1) // wpr
    blocks = S * MOE_SLOTS * inter // 128
    wv = b.land(cx.wave, cx.i(wpr - 1))
    lane16 = b.land(cx.lane, cx.i(15))
    # a (row, slot) owns nkc / 2 consecutive units of one expert: resolve its
    # expert, resources and route weight once per run of `hoist` units
    hoist = math.gcd(nkc // 2, upw) if n_units % wpr == 0 else 1
    runs: dict[int, tuple] = {}

    def run_of(cc: int):
        got = runs.get(cc // hoist)
        if got is None:
            qu = b.add(b.mul(wv, cx.i(upw)), cx.i((cc // hoist) * hoist))
            q = b.mul(b.smin(qu, cx.i(n_units - 1)), cx.i(2))
            s_q = b.div(q, cx.i(slots * nkc))
            slot_q = b.add(b.mod(b.div(q, cx.i(nkc)), cx.i(slots)), cx.i(1))
            qs = b.add(b.mul(s_q, cx.i(MOE_SLOTS)), slot_q)
            e = _uniform_i32(cx, b.add(qs, cx.i(cx.keys)))
            wb = cx.rsrc_at(routed_addr(cx, "w_dn", e, w_bytes))
            sb = cx.rsrc_at(routed_addr(cx, "s_dn", e, s_bytes))
            weight = b.select(b.cmp_eq(lane16, s_q), _uniform_f32(cx, b.add(qs, cx.i(cx.dnw))), cx.f(0.0))
            got = runs[cc // hoist] = (qs, wb, sb, weight)
        return got

    def u_dn(cc: int):
        qu = b.add(b.mul(wv, cx.i(upw)), cx.i(cc))
        q = b.mul(b.smin(qu, cx.i(n_units - 1)), cx.i(2))
        kc = b.mod(q, cx.i(nkc))
        qs, wb, sb, weight = run_of(cc)
        unit = b.add(b.mul(qs, cx.i(nkc // 2)), b.lshr(kc, cx.i(1)))
        return unit_native_mxfp4(cx, wb, sb, rg, b.mod(unit, cx.i(inter // 128)), inter,
                                 b.mul(unit, cx.i(32)),
                                 b.add(b.mul(unit, cx.i(4)), b.lshr(cx.lane, cx.i(4))),
                                 weight, dn_ln, late_scale=cx.spec.dn_scale_late)  # fmt: skip

    sh_wb = cx.rsrc_at(cx.table_addr(T_SHARED + 2))
    sh_sb = cx.rsrc_at(cx.table_addr(T_SHARED + 3))

    def u_dn_shared(cc: int):
        sn = n_sel(cx, S)
        base = b.mul(wv, cx.i(spw))
        mine = b.land(b.cmp_ge(lane16, base), b.cmp_lt(lane16, b.add(base, cx.i(spw))))
        unit = b.add(b.mul(sn, cx.i(MOE_SLOTS * (nkc // 2))), cx.i(cc))

        def coef() -> Value:
            return b.select(mine, cx.lds_ldf(b.add(b.mul(sn, cx.i(MOE_SLOTS)), cx.i(cx.dnw))), cx.f(0.0))

        return unit_native_mxfp4(cx, sh_wb, sh_sb, rg, cx.i(cc), inter,
                                 b.mul(unit, cx.i(32)),
                                 b.add(b.mul(unit, cx.i(4)), b.lshr(cx.lane, cx.i(4))),
                                 coef, dn_ln, late_scale=cx.spec.dn_scale_late)  # fmt: skip

    n_sh = nkc // 2
    pipe = cx.spec.dn_pipe
    if pipe:
        # one software pipeline over the shared expert's units, then the
        # routed ones (the accumulation order of the two-phase form)
        def u_any(cc: int):
            return u_dn_shared(cc) if cc < n_sh else u_dn(cc - n_sh)

        pre = [u_any(cc) for cc in range(min(batch, n_sh + upw))]
    else:
        pre = [u_dn(cc) for cc in range(min(batch, upw))]
    early = cx.spec.dn_shared_early and not pipe
    shared_pre = [u_dn_shared(cc) for cc in range(n_sh)] if early else None
    # dn_resid_early: this thread's residual pair of ``a`` (published by o long
    # before) now, not behind the reduction (peer_reduce's thread map)
    early_a = None
    resid_spec = None
    if cx.spec.dn_resid_early:
        s_r, row_r = resid_slot(cx, t, tile)
        resid_spec = ("a", b.lshr(b.add(b.mul(s_r, cx.i(HIDDEN)), row_r), cx.i(1)), 1)
        if not cx.spec.dn_resid_with_mid:
            early_a = cx.poll([resid_spec])[0][0]
    cx.barrier()
    if cx.spec.dn_slot_phases:
        if S != wpr or pipe or early:
            raise NotImplementedError("dn_slot_phases: one row per wave, without dn_pipe / dn_shared_early")
        # a wave's units run shared, then slots 1..4 (per_slot units each): a
        # group's units start once its mid blocks are polled, quantized and in
        # LDS, the same units in the same order as one pass over all of mid
        per_slot = upw // slots
        nbs = inter // 128
        groups = ((0, 1), (2, 2), (3, 4))
        acc = zero4(cx)
        cur = pre
        for gi, (q0, q1) in enumerate(groups):
            per_row = (q1 - q0 + 1) * nbs
            total = S * per_row
            n_bg = (total + WAVES - 1) // WAVES
            step = min(cx.spec.mid_chunk or n_bg, n_bg)
            for c0 in range(0, n_bg, step):
                chunk = range(c0, min(c0 + step, n_bg))
                items = [b.smin(b.add(cx.wave, cx.i(bi * WAVES)), cx.i(total - 1)) for bi in chunk]
                blks = [b.add(b.mul(b.add(b.mul(b.div(it, cx.i(per_row)), cx.i(MOE_SLOTS)), cx.i(q0)), cx.i(nbs)),
                              b.mod(it, cx.i(per_row))) for it in items]  # fmt: skip
                specs = [("mid", b.add(b.mul(blk, cx.i(128)), b.mul(cx.lane, cx.i(2))), 2) for blk in blks]
                with_resid = gi == 0 and c0 == 0 and resid_spec is not None and early_a is None
                mids = cx.poll(specs + ([resid_spec] if with_resid else []))
                if with_resid:
                    early_a = mids.pop()[0]
                for bi, blk, mid in zip(chunk, blks, mids):
                    with b.scf_if(b.cmp_lt(b.add(cx.wave, cx.i(bi * WAVES)), cx.i(total))):
                        m0, m1, qs = quant_mxfp8(cx, b.bitcast(mid[0], F32), b.bitcast(mid[1], F32))
                        st_mxfp8(cx, b.add(b.mul(blk, cx.i(128)), b.mul(cx.lane, cx.i(2))), m0, m1)
                        with b.scf_if(b.cmp_eq(lane16, cx.i(0))):
                            cx.lds_st(b.add(b.add(b.mul(blk, cx.i(4)), b.lshr(cx.lane, cx.i(4))), cx.i(cx.misc)), qs)
            cx.barrier()
            if gi == 0:
                cx.stamp("down", t, 2)
                for c0 in range(0, n_sh, batch):
                    acc = mma_units(cx, acc, [u_dn_shared(cc) for cc in range(c0, min(c0 + batch, n_sh))])
            lo, hi = (max(q0, 1) - 1) * per_slot, q1 * per_slot
            acc = run_units(cx, lambda c, lo=lo: u_dn(lo + c), hi - lo, batch, cur, acc0=acc)
            if gi + 1 < len(groups):
                # the next group's first weights stream during its mid poll
                nlo, nhi = (groups[gi + 1][0] - 1) * per_slot, groups[gi + 1][1] * per_slot
                npre = batch if cx.spec.dn_group_pre < 0 else cx.spec.dn_group_pre
                cur = [u_dn(nlo + c) for c in range(min(npre, nhi - nlo))] if npre else None
    else:
        # the mid blocks a chunk at a time: polled, quantized and stored before
        # the next poll, so the live poll results stay small
        n_b = (blocks + WAVES - 1) // WAVES
        step = cx.spec.mid_chunk or n_b
        for c0 in range(0, n_b, step):
            chunk = range(c0, min(c0 + step, n_b))
            specs = [("mid", b.add(b.mul(b.smin(b.add(cx.wave, cx.i(bi * WAVES)), cx.i(blocks - 1)), cx.i(128)),
                                   b.mul(cx.lane, cx.i(2))), 2) for bi in chunk]  # fmt: skip
            with_resid = c0 == 0 and resid_spec is not None and early_a is None
            mids = cx.poll(specs + ([resid_spec] if with_resid else []))
            if with_resid:
                early_a = mids.pop()[0]
            for bi, mid in zip(chunk, mids):
                blk = b.add(cx.wave, cx.i(bi * WAVES))
                with b.scf_if(b.cmp_lt(blk, cx.i(blocks))):
                    q0, q1, qs = quant_mxfp8(cx, b.bitcast(mid[0], F32), b.bitcast(mid[1], F32))
                    st_mxfp8(cx, b.add(b.mul(blk, cx.i(128)), b.mul(cx.lane, cx.i(2))), q0, q1)
                    with b.scf_if(b.cmp_eq(lane16, cx.i(0))):
                        cx.lds_st(b.add(b.add(b.mul(blk, cx.i(4)), b.lshr(cx.lane, cx.i(4))), cx.i(cx.misc)), qs)
        cx.barrier()
        cx.stamp("down", t, 2)
        if pipe:
            acc = run_units(cx, u_any, n_sh + upw, batch, pre)
        else:
            if early:
                acc0 = mma_units(cx, zero4(cx), shared_pre)
            else:
                acc0 = zero4(cx)
                for c0 in range(0, n_sh, batch):
                    acc0 = mma_units(cx, acc0, [u_dn_shared(cc) for cc in range(c0, min(c0 + batch, n_sh))])
            acc = run_units(cx, u_dn, upw, batch, pre, acc0=acc0)
    cx.stamp("down", t, 3)

    def emit_dn(rl: Value, n: Value, v: Value) -> None:
        inside = b.land(b.cmp_ge(rl, dn_off), b.cmp_lt(rl, b.add(dn_off, cx.i(tile))))
        with b.scf_if(inside):
            cx.lds_st(b.add(b.add(b.mul(n, cx.i(tile)), b.sub(rl, dn_off)), cx.i(cx.outs)), v)

    reduce_rows(cx, r_groups, acc, emit_dn, S)
    cx.barrier()
    r_x = cx.rsrc("x_out")

    def resid(s: Value, row: Value):
        if early_a is not None:
            return cx.bf2_f32(early_a)
        w = cx.poll([("a", b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)), 1)])[0][0]
        return cx.bf2_f32(w)

    def store_x(s: Value, row: Value, v0: Value, v1: Value) -> None:
        cx.st_i32(r_x, b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)), cx.bf16_pair(v0, v1))

    peer_reduce(cx, t, tile, resid, store_x, "ffn")
