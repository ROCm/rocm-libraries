# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage ``q_norm``: one CTA per verify row normalizes the layer input.

Publishes ``q_an`` = bf16(RMSNorm(h) * g_in), the q_b GEMV input, and, with
the indexer, ``iq_an`` = bf16(LayerNorm(q_an, eps 1e-6) * w + b), the index-q
GEMV input. ``w`` and ``b`` are fp32 rows of the parameter table ``g_q``.
Each thread owns three 4-element chunks at ``(tid + i * 512) * 4``.
"""

from __future__ import annotations

from rocke.core.ir import Value

from .ctx import EPS, HIDDEN, THREADS, Ml4Ctx

# word offsets of the index-q LayerNorm weight and bias in ``g_q``
P_IQ_W = 0
P_IQ_B = HIDDEN
INDEX_EPS = 1e-6
CHUNKS = HIDDEN // (4 * THREADS)


def emit_q_norm(cx: Ml4Ctx) -> None:
    cx.tasks("q_norm", lambda s: _row(cx, s))


def _row(cx: Ml4Ctx, s: Value) -> None:
    b = cx.b
    ks = [b.mul(b.add(cx.tid, cx.i(i * THREADS)), cx.i(4)) for i in range(CHUNKS)]
    r_g = cx.rsrc("g_in")
    gs = [cx.ld_bf16x4(r_g, k) for k in ks]
    row = b.mul(s, cx.i(HIDDEN))
    r_h = cx.rsrc("h_in")
    xs = [cx.ld_bf16x4(r_h, b.add(row, k)) for k in ks]
    ss = cx.f(0.0)
    for v in xs:
        for a in v:
            ss = b.fadd(ss, b.fmul(a, a))
    tot = cx.block_sum(ss)
    rstd = b.rsqrt(b.fadd(b.fmul(tot, cx.f(1.0 / HIDDEN)), cx.f(EPS)))
    normed = []
    for k, v, g in zip(ks, xs, gs):
        a = [cx.bf16_round(b.fmul(b.fmul(v[j], rstd), g[j])) for j in range(4)]
        cx.put_bf("q_an", b.add(row, k), a)
        normed.append((k, a))
    if not cx.spec.with_indexer:
        return
    tot = cx.f(0.0)
    for _, v in normed:
        for a in v:
            tot = b.fadd(tot, a)
    mean = b.fmul(cx.block_sum(tot), cx.f(1.0 / HIDDEN))
    sq = cx.f(0.0)
    for _, v in normed:
        for a in v:
            d = b.fsub(a, mean)
            sq = b.fadd(sq, b.fmul(d, d))
    rstd = b.rsqrt(
        b.fadd(b.fmul(cx.block_sum(sq), cx.f(1.0 / HIDDEN)), cx.f(INDEX_EPS))
    )
    r_p = cx.rsrc("g_q")
    for k, v in normed:
        out = []
        for j in range(4):
            w = cx.ld_f32(r_p, b.add(k, cx.i(P_IQ_W + j)))
            bias = cx.ld_f32(r_p, b.add(k, cx.i(P_IQ_B + j)))
            y = b.fadd(b.fmul(b.fmul(b.fsub(v[j], mean), rstd), w), bias)
            out.append(cx.bf16_round(y))
        cx.put_bf("iq_an", b.add(row, k), out)
