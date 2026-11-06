# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stages ``mlp_ug`` and ``mlp_down``: the draft kind's dense MLP.

``mlp_ug`` task ``t`` computes rows ``48 t .. 48 t + 47`` (three 16-row
tiles) of the stacked [gate; up] FP8 projection (``w_ug`` / ``s_ug``,
``2 * inter`` rows, 128 x 128 block scales) of bf16(RMSNorm(a) * g_post)
for every verify row, the 8 waves splitting K as in ``qkv_a``, and
publishes them bf16-rounded (the GEMM output vLLM's activation reads) to
``mid``: row ``s``'s gate at ``[0, inter)``, its up at ``[inter, 2 inter)``.
A task stages the normed input once for its three tiles, and a tile's
weights stream during the previous tile's reduction.

``mlp_down`` task ``t``: 32 hidden rows of the FP8 down projection
(``w_dn`` / ``s_dn``, 2 row groups x 4 waves as in ``o``) of
bf16(SiLU(gate) * up), computed as vLLM's ``SiluAndMul`` does (fp32
``x / (1 + exp(-x)) * y``), then the TP sum and the residual ``a`` into
``x_out``. With ``mlp_act`` the activation is computed in ``mlp_ug``'s
epilogue instead (its tiles pair 8 gate rows with the same 8 up rows) and
``mid`` carries it as bf16 pairs.
"""

from __future__ import annotations

from rocke.core.ir import F32, Value

from .ctx import HIDDEN, THREADS, WAVES, Ml4Ctx
from .gemv import (
    emit_out,
    load_x_rmsnorm,
    mma_units,
    n_sel,
    reduce_rows,
    run_units,
    stage_x_pairs,
    stage_x_rmsnorm,
    unit_fp8x2,
    zero4,
)
from .stage_uv import peer_reduce

TILE = 16
UG_TILES = 3
NKC = HIDDEN // 64
UG_UNITS = NKC // (2 * WAVES)
DN_TILE = 32


def emit_mlp_ug(cx: Ml4Ctx) -> None:
    rows = 2 * cx.spec.inter
    if rows % (TILE * UG_TILES):
        raise ValueError(f"mlp_ug: 2 * inter = {rows} is not a multiple of {TILE * UG_TILES}")
    cx.tasks("mlp_ug", lambda t: _ug_task(cx, t))


def _ug_task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    count, inter = cx.spec.rows, cx.spec.inter
    col = b.mul(n_sel(cx, count), cx.i(HIDDEN))
    r_w, r_s = cx.rsrc("w_ug"), cx.rsrc("s_ug")

    def unit(tile: int, c: int):
        rg = b.add(b.mul(t, cx.i(UG_TILES)), cx.i(tile))
        kc = b.mul(b.add(b.mul(cx.wave, cx.i(UG_UNITS)), cx.i(c)), cx.i(2))
        return unit_fp8x2(cx, r_w, r_s, rg, kc, NKC, HIDDEN, b.lshr(b.add(col, b.mul(kc, cx.i(64))), cx.i(1)))

    def publish(tile: int) -> None:
        if cx.spec.mlp_act:
            _publish_act(cx, b.add(b.mul(t, cx.i(UG_TILES)), cx.i(tile)), count, inter)
            return
        with b.scf_if(b.cmp_lt(cx.tid, cx.i(count * TILE))):
            s = b.lshr(cx.tid, cx.i(4))
            rg = b.add(b.mul(t, cx.i(UG_TILES)), cx.i(tile))
            row = b.add(b.mul(rg, cx.i(TILE)), b.land(cx.tid, cx.i(TILE - 1)))
            v = cx.bf16_round(cx.lds_ldf(b.add(cx.tid, cx.i(cx.outs))))
            cx.put("mid", b.add(b.mul(s, cx.i(2 * inter)), row), v)

    pipe = cx.spec.mlp_ug_pipe
    pre = [unit(0, c) for c in range(UG_UNITS)]
    nxt = [unit(1, c) for c in range(UG_UNITS)] if pipe else None

    def ld_a(sks):
        got = cx.poll([("a", b.lshr(b.add(cx.i(s * HIDDEN), k), cx.i(1)), 2) for s, k in sks])
        return [list(cx.bf2_f32(w[0])) + list(cx.bf2_f32(w[1])) for w in got]

    stage_x_rmsnorm(cx, load_x_rmsnorm(cx, ld_a, HIDDEN, "g_post", count), HIDDEN, count)
    cx.barrier()
    # timeline builds: 1 the normed input staged, 2 the first tile out, 3 the last
    cx.stamp("mlp_ug", t, 1)
    if pipe:
        accs = []
        for tile in range(UG_TILES):
            accs.append(mma_units(cx, zero4(cx), pre))
            pre, nxt = nxt, ([unit(tile + 2, c) for c in range(UG_UNITS)] if tile + 2 < UG_TILES else None)
        for tile, acc in enumerate(accs):
            reduce_rows(cx, 1, acc, emit_out(cx, TILE), count)
            cx.barrier()
            publish(tile)
            if tile == 0:
                cx.stamp("mlp_ug", t, 2)
        cx.stamp("mlp_ug", t, 3)
        return
    for tile in range(UG_TILES):
        acc = run_units(cx, lambda c, tile=tile: unit(tile, c), UG_UNITS, UG_UNITS, pre)
        pre = [unit(tile + 1, c) for c in range(UG_UNITS)] if tile + 1 < UG_TILES else None
        reduce_rows(cx, 1, acc, emit_out(cx, TILE), count)
        cx.barrier()
        publish(tile)
        if tile == 0:
            cx.stamp("mlp_ug", t, 2)
    cx.stamp("mlp_ug", t, 3)


def _silu_mul(cx: Ml4Ctx, g: Value, u: Value) -> Value:
    """vLLM's SiluAndMul in fp32: g / (1 + exp(-g)) * u."""
    b = cx.b
    return b.fmul(b.fdiv(g, b.fadd(cx.f(1.0), b.exp(b.fneg(g)))), u)


def _publish_act(cx: Ml4Ctx, rg: Value, count: int, inter: int) -> None:
    """``mlp_act``: tile ``rg``'s rows 0-7 are gate rows ``8 rg ..``, its rows
    8-15 the same up rows: the activations of those 8 intermediate rows, from
    the bf16-rounded GEMM outputs, to ``mid`` as bf16 pairs (row ``s`` at
    ``[s * inter, (s + 1) * inter)``)."""
    b = cx.b
    half = TILE // 2
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(count * half // 2))):
        s = b.lshr(cx.tid, cx.i(2))
        r = b.mul(b.land(cx.tid, cx.i(3)), cx.i(2))
        base = b.add(b.mul(s, cx.i(TILE)), r)
        acts = []
        for j in range(2):
            g = cx.bf16_round(cx.lds_ldf(b.add(base, cx.i(cx.outs + j))))
            u = cx.bf16_round(cx.lds_ldf(b.add(base, cx.i(cx.outs + half + j))))
            acts.append(_silu_mul(cx, g, u))
        cx.put_bf("mid", b.add(b.mul(s, cx.i(inter)), b.add(b.mul(rg, cx.i(half)), r)), acts)


def emit_mlp_down(cx: Ml4Ctx) -> None:
    inter = cx.spec.inter
    units = inter // 64 // (2 * (WAVES // (DN_TILE // 16)))
    batch = cx.spec.mlp_dn_batch
    if inter % (2 * THREADS) or not 0 < batch <= units or HIDDEN % DN_TILE:
        raise ValueError(f"mlp_down: inter {inter} (batch {batch}) does not tile")
    cx.tasks("mlp_down", lambda t: _down_task(cx, t))


def _down_task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    S, k_dim = cx.spec.rows, cx.spec.inter
    nkc = k_dim // 64
    r_tiles = DN_TILE // 16
    wpr = WAVES // r_tiles
    units = nkc // (2 * wpr)
    r_w, r_s = cx.rsrc("w_dn"), cx.rsrc("s_dn")
    rg = b.add(b.mul(t, cx.i(r_tiles)), b.lshr(cx.wave, cx.i(wpr.bit_length() - 1)))
    col = b.mul(n_sel(cx, S), cx.i(k_dim))

    def u_dn(c: int):
        kc = b.mul(b.add(b.mul(b.land(cx.wave, cx.i(wpr - 1)), cx.i(units)), cx.i(c)), cx.i(2))
        return unit_fp8x2(cx, r_w, r_s, rg, kc, nkc, k_dim, b.lshr(b.add(col, b.mul(kc, cx.i(64))), cx.i(1)))

    batch = cx.spec.mlp_dn_batch
    pre = [u_dn(c) for c in range(batch)]
    cx.barrier()
    if cx.spec.mlp_act:
        stage_x_pairs(cx, "mid", S * k_dim)
    else:
        _stage_act(cx, S, k_dim)
    cx.barrier()
    # timeline builds: 1 the activation staged, 2 the GEMV reduced, 3 the TP sum out
    cx.stamp("mlp_down", t, 1)
    acc = run_units(cx, u_dn, units, batch, pre)
    reduce_rows(cx, r_tiles, acc, emit_out(cx, DN_TILE), S)
    cx.barrier()
    cx.stamp("mlp_down", t, 2)
    r_x = cx.rsrc("x_out")

    def resid(s: Value, row: Value):
        w = cx.poll([("a", b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)), 1)])[0][0]
        return cx.bf2_f32(w)

    def store_x(s: Value, row: Value, v0: Value, v1: Value) -> None:
        cx.st_i32(r_x, b.lshr(b.add(b.mul(s, cx.i(HIDDEN)), row), cx.i(1)), cx.bf16_pair(v0, v1))

    peer_reduce(cx, t, DN_TILE, resid, store_x, "ffn")
    cx.stamp("mlp_down", t, 3)


def _stage_act(cx: Ml4Ctx, S: int, inter: int) -> None:
    """LDS bf16 X[s][k] = bf16(SiLU(gate) * up) from ``mid``, a row at a
    time: one 16-byte poll of gate and one of up per element pair."""
    b = cx.b
    per = inter // 2 // THREADS
    for s in range(S):
        ps = [b.add(cx.tid, cx.i(i * THREADS)) for i in range(per)]
        specs = []
        for p in ps:
            base = b.add(cx.i(s * 2 * inter), b.mul(p, cx.i(2)))
            specs += [("mid", base, 2), ("mid", b.add(base, cx.i(inter)), 2)]
        got = cx.poll(specs)
        for n, p in enumerate(ps):
            g, u = got[2 * n], got[2 * n + 1]
            act = [_silu_mul(cx, b.bitcast(g[j], F32), b.bitcast(u[j], F32)) for j in range(2)]
            cx.lds_st(b.add(cx.i(s * inter // 2 + cx.xs), p), cx.bf16_pair(act[0], act[1]))
