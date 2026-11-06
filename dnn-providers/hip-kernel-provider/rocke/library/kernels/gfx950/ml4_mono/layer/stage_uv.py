# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stages ``uv`` and ``o``: the split merge, W_UV, W_o and the residual.

``uv`` task ``(s, t)``: merge head ``t // 2``'s split partials of row ``s``
(weights exp(m - M) / L over the splits) into the bf16 latent, then 64 of
the head's 128 value rows through W_UV (FP8, 4 row groups x 2 waves) into
``o``. ``o`` task ``t``: 32 hidden rows of W_o over every row's ``o`` (FP8,
2 row groups x 4 waves), plus the TP partial sum and the residual ``h`` into
``a``.
"""

from __future__ import annotations

from rocke.core.ir import F32, I32, I64, Value
from rocke.helpers import mailbox as mbx

from .ctx import HIDDEN, WAVES, Ml4Ctx, split_keys
from .gemv import emit_out, n_sel, reduce_rows, run_units, stage_x_pairs, unit_fp8x2

KV_LORA = 512
V_DIM = 128
UV_TILE = 64
ROW_TILE = 32
TOPK = 2048
NEG = -1.0e30
LOG2E = 1.4426950408889634


def emit_uv(cx: Ml4Ctx) -> None:
    cx.tasks("uv", lambda tt: _uv_task(cx, tt))


def _uv_task(cx: Ml4Ctx, tt: Value) -> None:
    b = cx.b
    S, H = cx.spec.rows, cx.spec.heads
    n_split = TOPK // split_keys(S, H)
    n_uv = H * V_DIM // UV_TILE
    o_k = H * V_DIM
    nkc = KV_LORA // 64
    r_tiles = UV_TILE // 16
    wpr = WAVES // r_tiles
    units = nkc // (2 * wpr)
    s = b.div(tt, cx.i(n_uv))
    t = b.mod(tt, cx.i(n_uv))
    head = b.lshr(t, cx.i(1))
    r_w, r_s = cx.rsrc("w_uv"), cx.rsrc("s_uv")
    rg = b.add(b.mul(t, cx.i(r_tiles)), b.lshr(cx.wave, cx.i(wpr.bit_length() - 1)))

    def u_uv(c: int):
        kc = b.mul(b.add(b.mul(b.land(cx.wave, cx.i(wpr - 1)), cx.i(units)), cx.i(c)), cx.i(2))
        return unit_fp8x2(cx, r_w, r_s, rg, kc, nkc, KV_LORA, b.mul(kc, cx.i(32)))

    pre = [u_uv(c) for c in range(units)]
    cx.barrier()
    sph = n_split // 4
    dp_lo = b.land(cx.tid, cx.i(KV_LORA // 4 - 1))
    hf = b.lshr(cx.tid, cx.i(7))
    spi = b.smin(cx.lane, cx.i(n_split - 1))
    ml = b.add(b.mul(b.add(b.mul(s, cx.i(n_split)), spi), cx.i(H)), head)
    # wave 0 alone waits for the splits (one pair each) so a waiting CTA does
    # not flood memory while the split tasks still stream their keys
    with b.scf_if(b.cmp_eq(cx.wave, cx.i(0))):
        cx.poll([("sp_l", ml, 1)])
    cx.barrier()
    cx.stamp("uv", tt, 2)

    def acc_specs(dh: int):
        dp = b.add(dp_lo, cx.i(dh * (KV_LORA // 4)))
        out = []
        for j in range(sph):
            sp = b.add(b.mul(s, cx.i(n_split)), b.add(b.mul(hf, cx.i(sph)), cx.i(j)))
            out.append(("sp_acc", b.add(b.mul(b.add(b.mul(sp, cx.i(H)), head), cx.i(KV_LORA // 2)), dp), 1))
        return out

    if cx.spec.uv_batch_poll:
        # the softmax stats and both halves of the partials in one poll
        allv = cx.poll([("sp_m", ml, 1), ("sp_l", ml, 1)] + acc_specs(0) + acc_specs(1))
        ml_got, acc_got = allv[:2], [allv[2 : 2 + sph], allv[2 + sph :]]
    else:
        ml_got, acc_got = cx.poll([("sp_m", ml, 1), ("sp_l", ml, 1)]), None
    with b.scf_if(b.cmp_eq(cx.wave, cx.i(0))):
        ok = b.cmp_lt(cx.lane, cx.i(n_split))
        m_sp = b.select(ok, b.bitcast(ml_got[0][0], F32), cx.f(NEG))
        l_sp = b.select(ok, b.bitcast(ml_got[1][0], F32), cx.f(0.0))
        m_all = cx.wave_max(m_sp)
        w_sp = b.exp2_fast(b.fmul(b.fsub(m_sp, m_all), cx.f(LOG2E)))
        den = cx.wave_sum(b.fmul(l_sp, w_sp))
        inv = b.select(b.fcmp("ogt", den, cx.f(0.0)), b.rcp_fast(den), cx.f(0.0))
        with b.scf_if(ok):
            cx.lds_st(b.add(cx.lane, cx.i(cx.misc)), b.fmul(w_sp, inv))
    cx.barrier()
    for dh in range(2):
        dp = b.add(dp_lo, cx.i(dh * (KV_LORA // 4)))
        got = acc_got[dh] if acc_got is not None else cx.poll(acc_specs(dh))
        o0, o1 = cx.f(0.0), cx.f(0.0)
        for j in range(sph):
            wj = cx.lds_ldf(b.add(b.mul(hf, cx.i(sph)), cx.i(cx.misc + j)))
            a0, a1 = cx.bf2_f32(got[j][0])
            o0 = b.fadd(o0, b.fmul(a0, wj))
            o1 = b.fadd(o1, b.fmul(a1, wj))
        base = b.mul(b.add(b.mul(hf, cx.i(KV_LORA // 2)), dp), cx.i(2))
        cx.lds_st(b.add(base, cx.i(cx.red)), o0)
        cx.lds_st(b.add(base, cx.i(cx.red + 1)), o1)
    cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(KV_LORA // 2))):
        o0, o1 = cx.f(0.0), cx.f(0.0)
        for q in range(4):
            base = b.mul(b.add(cx.i(q * (KV_LORA // 2)), cx.tid), cx.i(2))
            o0 = b.fadd(o0, cx.lds_ldf(b.add(base, cx.i(cx.red))))
            o1 = b.fadd(o1, cx.lds_ldf(b.add(base, cx.i(cx.red + 1))))
        cx.lds_st(b.add(cx.tid, cx.i(cx.xs)), cx.bf16_pair(cx.bf16_round(o0), cx.bf16_round(o1)))
    cx.barrier()
    acc = run_units(cx, u_uv, units, units, pre)
    reduce_rows(cx, r_tiles, acc, emit_out(cx, UV_TILE), S)
    cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(UV_TILE // 4))):
        r = b.mul(cx.tid, cx.i(4))
        vals = [cx.lds_ldf(b.add(r, cx.i(cx.outs + j))) for j in range(4)]
        cx.put_bf("o", b.add(b.add(b.mul(s, cx.i(o_k)), b.mul(t, cx.i(UV_TILE))), r), vals)


def emit_o(cx: Ml4Ctx) -> None:
    cx.tasks("o", lambda t: _o_task(cx, t))


def _o_task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    S, H = cx.spec.rows, cx.spec.heads
    o_k = H * V_DIM
    nkc = o_k // 64
    r_tiles = ROW_TILE // 16
    wpr = WAVES // r_tiles
    units = nkc // (2 * wpr)
    r_w, r_s = cx.rsrc("w_o"), cx.rsrc("s_o")
    rg = b.add(b.mul(t, cx.i(r_tiles)), b.lshr(cx.wave, cx.i(wpr.bit_length() - 1)))
    col = b.mul(n_sel(cx, S), cx.i(o_k))

    def u_o(c: int):
        kc = b.mul(b.add(b.mul(b.land(cx.wave, cx.i(wpr - 1)), cx.i(units)), cx.i(c)), cx.i(2))
        b_word = b.lshr(b.add(col, b.mul(kc, cx.i(64))), cx.i(1))
        return unit_fp8x2(cx, r_w, r_s, rg, kc, nkc, o_k, b_word)

    pre = [u_o(c) for c in range(units)]
    r_h = cx.rsrc("h_in")

    def resid_word(s: Value, row: Value) -> Value:
        return cx.ld_i32(r_h, b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)))

    # o_resid_early: this thread's residual word (a launch input) loads now,
    # not behind the reduction (peer_reduce's thread -> (row, pair) map)
    early = resid_word(*resid_slot(cx, t, ROW_TILE)) if cx.spec.o_resid_early else None
    cx.barrier()
    stage_x_pairs(cx, "o", S * o_k)
    cx.barrier()
    cx.stamp("o", t, 2)
    acc = run_units(cx, u_o, units, units, pre)
    reduce_rows(cx, r_tiles, acc, emit_out(cx, ROW_TILE), S)
    cx.barrier()

    def resid(s: Value, row: Value):
        return cx.bf2_f32(early if early is not None else resid_word(s, row))

    def out(s: Value, row: Value, v0: Value, v1: Value) -> None:
        cx.put_bf("a", b.add(b.mul(s, cx.i(HIDDEN)), row), [v0, v1])

    peer_reduce(cx, t, ROW_TILE, resid, out, "attn")


def resid_slot(cx: Ml4Ctx, t: Value, tile: int) -> tuple[Value, Value]:
    """(verify row, hidden row) whose residual pair thread ``tid`` adds in
    ``peer_reduce`` (clamped for the threads past the tile)."""
    b = cx.b
    half = tile // 2
    tid = b.smin(cx.tid, cx.i(cx.spec.rows * half - 1))
    s = b.div(tid, cx.i(half))
    return s, b.add(b.mul(t, cx.i(tile)), b.mul(b.mod(tid, cx.i(half)), cx.i(2)))


def peer_reduce(cx: Ml4Ctx, t: Value, tile: int, residual, out_fn, region: str) -> None:
    """``out_fn(s, row, residual + sum of the ranks' partials)`` for this
    task's ``tile`` rows of every verify row. With TP peers, wave ``w`` pushes
    this rank's bf16 partial tile to peer ``w``'s slot of the symmetric region
    ``region``, then every rank sums the W partials in rank order."""
    b = cx.b
    S, W = cx.spec.rows, cx.spec.npes
    half = tile // 2
    region_base = None
    if W > 1:
        stride = b.const_i64(cx.spec.sym_offset("_part_stride"))
        region_base = b.add(b.const_i64(cx.spec.sym_offset(region)), b.mul(b.zext(cx.peer_slot, I64), stride))
        with b.scf_if(b.cmp_lt(cx.wave, cx.i(W))):
            peers = cx.rsrc("peers")
            pw = b.buffer_load_vN(peers, b.mul(b.smin(cx.wave, cx.i(W - 1)), cx.i(8)), cx.i(0), I32, 2)
            lo = b.zext(b.readfirstlane(b.vec_extract(pw, 0)), I64)
            hi = b.zext(b.readfirstlane(b.vec_extract(pw, 1)), I64)
            dst = mbx.mailbox_from_i64(b, b.add(b.lor(b.shl(hi, b.const_i64(32)), lo), region_base),
                                       cx.i(0x7FFFFFF0), cx.sys_knobs)  # fmt: skip
            pair_count = S * half
            for batch in range((pair_count + 63) // 64):
                pair = b.add(cx.lane, cx.i(batch * 64))
                with b.scf_if(b.cmp_lt(pair, cx.i(pair_count))):
                    si = b.div(pair, cx.i(half))
                    ri = b.mul(b.mod(pair, cx.i(half)), cx.i(2))
                    o = b.add(b.mul(si, cx.i(tile)), ri)
                    vals = [cx.lds_ldf(b.add(o, cx.i(cx.outs))), cx.lds_ldf(b.add(o, cx.i(cx.outs + 1)))]
                    elem = b.add(b.add(b.mul(b.add(b.mul(cx.arg["rank"], cx.i(S)), si), cx.i(HIDDEN)),
                                       b.mul(t, cx.i(tile))), ri)  # fmt: skip
                    mbx.put_bf(b, dst, elem, vals, cx.tag, cx.sys_knobs)
        cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * half))):
        s = b.div(cx.tid, cx.i(half))
        r = b.mul(b.mod(cx.tid, cx.i(half)), cx.i(2))
        row = b.add(b.mul(t, cx.i(tile)), r)
        r0, r1 = residual(s, row)
        if W == 1:
            o = b.add(b.mul(s, cx.i(tile)), r)
            parts = [(cx.lds_ldf(b.add(o, cx.i(cx.outs))), cx.lds_ldf(b.add(o, cx.i(cx.outs + 1))))]
        else:
            own = mbx.mailbox_from_i64(b, b.add(cx.arg["sym"], region_base), cx.i(0x7FFFFFF0), cx.sys_knobs)
            specs = [(own, b.lshr(b.add(b.mul(b.add(cx.i(src * S), s), cx.i(HIDDEN)), row), cx.i(1)), 1)
                     for src in range(W)]  # fmt: skip
            got = mbx.poll(b, specs, cx.tag, cx.sys_knobs)
            parts = [cx.bf2_f32(v[0]) for v in got]
        t0, t1 = cx.f(0.0), cx.f(0.0)
        for p0, p1 in parts:
            t0 = b.fadd(t0, p0)
            t1 = b.fadd(t1, p1)
        out_fn(s, row, b.fadd(r0, t0), b.fadd(r1, t1))
