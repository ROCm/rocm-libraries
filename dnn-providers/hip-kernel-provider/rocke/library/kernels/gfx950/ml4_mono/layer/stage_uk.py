# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage ``uk``: the absorbed query ``q_lat = W_UK^T q_nope`` of one head.

Task ``t`` computes 128 of head ``t // 4``'s 512 latent rows: one 16-row
group per wave over the head's nope dims (FP8, 64-k chunks), so no
cross-wave sum; the tiles go straight from the reduction area to ``q_lat``.
"""

from __future__ import annotations

from rocke.core.ir import F32, I32, Value, VectorType

from .ctx import WAVES, Ml4Ctx
from .gemv import n_sel, run_units, stage_x_pairs, unit_fp8
from .stage_q_b import dims

KV_LORA = 512
UK_TILE = 128
PER_HEAD = KV_LORA // UK_TILE


def emit_uk(cx: Ml4Ctx) -> None:
    cx.tasks("uk", lambda t: _task(cx, t))


def _task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    S, H = cx.spec.rows, cx.spec.heads
    nope_dim, _ = dims(cx)
    nkc = nope_dim // 64
    head = b.lshr(t, cx.i(PER_HEAD.bit_length() - 1))
    r_w, r_s = cx.rsrc("w_uk"), cx.rsrc("s_uk")
    rg = b.add(b.mul(t, cx.i(WAVES)), cx.wave)
    col = b.mul(n_sel(cx, S), cx.i(nope_dim))

    def u_uk(c: int):
        b_word = b.lshr(b.add(col, cx.i(c * 64)), cx.i(1))
        return unit_fp8(cx, r_w, r_s, rg, cx.i(c), nkc, nope_dim, b_word)

    pre = [u_uk(c) for c in range(nkc)]
    cx.barrier()

    def src_of(k: Value) -> Value:
        s = b.lshr(k, cx.i(nope_dim.bit_length() - 1))
        row = b.mul(b.add(b.mul(s, cx.i(H)), head), cx.i(nope_dim))
        return b.add(row, b.land(k, cx.i(nope_dim - 1)))

    stage_x_pairs(cx, "q_nope", S * nope_dim, src_of)
    cx.barrier()
    acc = run_units(cx, u_uk, nkc, nkc, pre)
    base = b.mul(b.add(b.mul(cx.wave, cx.i(64)), cx.lane), cx.i(4))
    cx.lds_st4(b.add(base, cx.i(cx.red)), b.bitcast(b.vec_pack(list(acc), F32), VectorType(I32, 4)))
    cx.barrier()
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(S * UK_TILE // 4))):
        k = b.mul(cx.tid, cx.i(4))
        s = b.lshr(k, cx.i(7))
        r0 = b.land(k, cx.i(UK_TILE - 1))
        vals = []
        for j in range(4):
            r = b.add(r0, cx.i(j))
            ww = b.lshr(r, cx.i(4))
            lane_of = b.add(s, b.mul(b.lshr(b.land(r, cx.i(15)), cx.i(2)), cx.i(16)))
            word = b.add(b.mul(b.add(b.mul(ww, cx.i(64)), lane_of), cx.i(4)), b.land(r, cx.i(3)))
            vals.append(cx.lds_ldf(b.add(word, cx.i(cx.red))))
        elem = b.add(
            b.mul(b.add(b.mul(s, cx.i(H)), head), cx.i(KV_LORA)),
            b.add(b.mul(b.land(t, cx.i(PER_HEAD - 1)), cx.i(UK_TILE)), r0),
        )
        cx.put_bf("q_lat", elem, vals)
