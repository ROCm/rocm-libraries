# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage ``qkv_a``: the kv latent and rope key projection of the normed input.

Task ``t`` computes rows ``16 t .. 16 t + 15`` of the 576-row FP8 projection
of bf16(RMSNorm(h) * g_in) for every verify row and publishes them to
``kv_a``. The 8 waves split K (6 FP8 128-k blocks each). The indexer kinds
also project the same staged activations to the index key (FP8, tasks
``t < 8``, 16 of its 128 rows each) and the index head weights (bf16, tasks
``t < 2``), published to ``index_k`` and ``index_w``.
"""

from __future__ import annotations

from rocke.core.ir import Value

from .ctx import HIDDEN, WAVES, Ml4Ctx
from .gemv import (
    emit_out,
    load_x_rmsnorm,
    n_sel,
    reduce_rows,
    run_units,
    stage_x_rmsnorm,
    unit_bf16,
    unit_fp8x2,
)

QKV_A_ROWS = 512 + 64
TILE = 16
INDEX_DIM = 128
INDEX_HEADS = 32
NKC = HIDDEN // 64
UNITS = NKC // (2 * WAVES)
IW_CPW = NKC // WAVES
# rows of the parameter table (``table``)
T_W_INDEX_K, T_S_INDEX_K, T_W_INDEX_W = 0, 1, 2
# one staged activation tile holds at most this many rows
MAX_TILE_ROWS = 6


def emit_qkv_a(cx: Ml4Ctx) -> None:
    if cx.spec.rows > MAX_TILE_ROWS:
        raise ValueError(f"qkv_a stages at most {MAX_TILE_ROWS} rows in one tile")
    cx.tasks("qkv_a", lambda t: _task(cx, t))


def _task(cx: Ml4Ctx, t: Value) -> None:
    b = cx.b
    count = cx.spec.rows
    col = b.mul(n_sel(cx, count), cx.i(HIDDEN))
    r_w, r_s = cx.rsrc("w_qkv_a"), cx.rsrc("s_qkv_a")

    def b_word(kc: Value) -> Value:
        return b.lshr(b.add(col, b.mul(kc, cx.i(64))), cx.i(1))

    def u_qa(c: int):
        kc = b.mul(b.add(b.mul(cx.wave, cx.i(UNITS)), cx.i(c)), cx.i(2))
        return unit_fp8x2(cx, r_w, r_s, t, kc, NKC, HIDDEN, b_word(kc))

    r_h = cx.rsrc("h_in")

    def ld_h(sks):
        return [cx.ld_bf16x4(r_h, b.add(cx.i(s * HIDDEN), k)) for s, k in sks]

    loaded = load_x_rmsnorm(cx, ld_h, HIDDEN, "g_in", count)
    pre = [u_qa(c) for c in range(UNITS)]
    stage_x_rmsnorm(cx, loaded, HIDDEN, count)
    cx.barrier()
    if cx.spec.with_indexer and not cx.spec.idx_proj_on_qb:
        # the index key and head weights first: the indexer chain waits on
        # them, the kv latent's consumer (attention) waits on the top-k
        _index_projections(cx, t, t, b_word, count)
    acc = run_units(cx, u_qa, UNITS, UNITS, pre)
    reduce_rows(cx, 1, acc, emit_out(cx, TILE), count)
    cx.barrier()
    _publish(cx, "kv_a", QKV_A_ROWS, t, count)


# index-key tiles, then index head-weight tiles
INDEX_PROJ_TILES = INDEX_DIM // TILE + INDEX_HEADS // TILE


def index_projection_tile(cx: Ml4Ctx, t: Value) -> None:
    """Task ``t`` < INDEX_PROJ_TILES of another stage (``idx_proj_on_qb``): its
    own staging of the normed input, then index-key tile ``t`` or head-weight
    tile ``t - 8``, the tiles qkv_a's first tasks would otherwise compute."""
    b = cx.b
    count = cx.spec.rows
    with b.scf_if(b.cmp_lt(t, cx.i(INDEX_PROJ_TILES))):
        col = b.mul(n_sel(cx, count), cx.i(HIDDEN))

        def b_word(kc: Value) -> Value:
            return b.lshr(b.add(col, b.mul(kc, cx.i(64))), cx.i(1))

        r_h = cx.rsrc("h_in")

        def ld_h(sks):
            return [cx.ld_bf16x4(r_h, b.add(cx.i(s * HIDDEN), k)) for s, k in sks]

        stage_x_rmsnorm(cx, load_x_rmsnorm(cx, ld_h, HIDDEN, "g_in", count), HIDDEN, count)
        cx.barrier()
        _index_projections(cx, t, b.sub(t, cx.i(INDEX_DIM // TILE)), b_word, count)


def _index_projections(cx: Ml4Ctx, kt: Value, wt: Value, b_word, count: int) -> None:
    """Index key tile ``kt`` (FP8, if < 8) and head-weight tile ``wt`` (bf16,
    if in [0, 2)) of the staged activations."""
    b = cx.b
    t = kt
    with b.scf_if(b.cmp_lt(kt, cx.i(INDEX_DIM // TILE))):
        r_wik = cx.rsrc_at(cx.table_addr(T_W_INDEX_K))
        r_sik = cx.rsrc_at(cx.table_addr(T_S_INDEX_K))

        def u_index_k(c: int):
            kc = b.mul(b.add(b.mul(cx.wave, cx.i(UNITS)), cx.i(c)), cx.i(2))
            return unit_fp8x2(cx, r_wik, r_sik, t, kc, NKC, HIDDEN, b_word(kc))

        ik_acc = run_units(cx, u_index_k, UNITS, UNITS)
        reduce_rows(cx, 1, ik_acc, emit_out(cx, TILE), count)
        cx.barrier()
        _publish(cx, "index_k", INDEX_DIM, t, count)
    w_ok = b.land(b.cmp_ge(wt, cx.i(0)), b.cmp_lt(wt, cx.i(INDEX_HEADS // TILE)))
    with b.scf_if(w_ok):
        r_wiw = cx.rsrc_at(cx.table_addr(T_W_INDEX_W))

        def u_index_w(c: int):
            kc = b.add(b.mul(cx.wave, cx.i(IW_CPW)), cx.i(c))
            return unit_bf16(cx, r_wiw, wt, kc, NKC, b_word(kc))

        iw_acc = run_units(cx, u_index_w, IW_CPW, IW_CPW)
        reduce_rows(cx, 1, iw_acc, emit_out(cx, TILE), count)
        cx.barrier()
        _publish(cx, "index_w", INDEX_HEADS, wt, count)
    cx.barrier()


def _publish(cx: Ml4Ctx, region: str, rows: int, t: Value, count: int) -> None:
    """Thread ``s * 16 + r`` publishes row ``16 t + r`` of sample ``s``."""
    b = cx.b
    with b.scf_if(b.cmp_lt(cx.tid, cx.i(count * TILE))):
        s = b.lshr(cx.tid, cx.i(4))
        row = b.add(b.mul(t, cx.i(TILE)), b.land(cx.tid, cx.i(TILE - 1)))
        v = cx.lds_ld(b.add(cx.tid, cx.i(cx.outs)))
        cx.put(region, b.add(b.mul(s, cx.i(rows)), row), v)
