# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""WMMA (RDNA wave32) FMHA-forward inner body.

The wave32 analogue of :func:`rocke.helpers.mfma_attention.
mfma_attention_fwd_inner_body`, which dispatches here for wave32 targets. The
names defined here are re-exported from :mod:`rocke.helpers.mfma_attention`.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..core.ir import IRBuilder, Value
from .attention_fwd_ext import AttnFwdExt, emit_rows_epilogue, last_index, score_stage
from .attention import (
    apply_attention_mask,
    wave_reduce_stages,
)
from ._attention_shared import (
    MFMA_ATTN_BLOCK_K,
    _ir_type_for_dtype,
    _softmax_row_reduce,
)

__all__ = ["_wmma_attn_op_id", "_WMMA_ATTN_OP_ID", "_wmma_attention_fwd_inner_body"]


# ---------------------------------------------------------------------------
# WMMA (RDNA wave32) FMHA-forward inner body -- the wave32 analogue of the
# MFMA body above.
# ---------------------------------------------------------------------------
#
# This is the *same* QK -> online-softmax -> PV pipeline as
# :func:`mfma_attention_fwd_inner_body`, but every physical fragment fact (which
# lane holds which (row, k) / (k, col) / (row, col) element, how many slots a
# lane owns) is read from the verified gfx1151 ``wmma_f32_16x16x16_f16``
# ``MmaOp`` layout maps, and the matmul is emitted through the target-neutral
# :meth:`IRBuilder.mma`. The wave32 online softmax reuses the arch-parameterized
# :func:`wave_reduce_max` / :func:`wave_reduce_sum` (wave_size=32), which lower
# the in-half XOR butterfly to ``ds_swizzle``.
#
# The fundamental difference from the wave64 MFMA body is the fragment
# distribution: a WMMA accumulator row spans the 16 lanes of one wave32 half and
# each lane owns ``c_frag_len`` (=8) q-rows of one k-column. RDNA3/3.5 (gfx11)
# and RDNA4 (gfx12) differ in the *operand* distribution: on gfx11 the A/B
# fragment carries the full ``a_frag_len`` (=16) K row in every lane (cross-half
# duplication); on gfx12 the duplication is gone -- the fragment is ``<8 x half>``
# per lane and the 16 K-elements of one WMMA step are split across the two lane
# halves (lanes 0-15 carry K 0..7, lanes 16-31 carry K 8..15). The body reads
# *all* of these facts off the per-arch ``MmaOp`` layout maps (``a_frag_len`` and
# the A-operand K coordinate of slot 0 give the per-lane K base), so it never
# hard-codes the wave32 magic numbers and one body serves both RDNA generations.


# Per-arch WMMA attention op_id. gfx11 (RDNA3/3.5) uses the cross-half-duplicated
# ``wmma_f32_16x16x16_*`` atom; gfx12 (RDNA4) uses the split-K
# ``wmma_gfx12_f32_16x16x16_*`` atom (mirrors ``_wmma_params`` in
# ``instances/common/_matmul_nbits_large_n.py``). The op_id also selects the f16
# vs bf16 intrinsic mangling, so it is keyed on the kernel dtype.
def _wmma_attn_op_id(arch: str, dtype: str) -> str:
    elem = "bf16" if dtype == "bf16" else "f16"
    if arch == "gfx1201":
        return f"wmma_gfx12_f32_16x16x16_{elem}"
    return f"wmma_f32_16x16x16_{elem}"


# Default op_id for the historical gfx1151 f16 path (kept for back-references in
# adapters / docs that import the module-level constant).
_WMMA_ATTN_OP_ID = "wmma_f32_16x16x16_f16"


def _wmma_attention_fwd_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    O: Value,  # noqa: E741 - standard attention notation (Q,K,V,O)
    head_size: int,
    seqlen_k: Value,
    q_tile_base: Value,
    head_idx: Value,
    kv_head_idx: Value,
    q_pos_base: Optional[Value],
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    scale_log2: Value,
    dtype: str,
    mask_mode: str,
    sliding_window: int,
    causal_ctx_offset: Optional[Value],
    k_token_offset_elems: Optional[Value],
    v_token_offset_elems: Optional[Value],
    k_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]],
    v_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]],
    k_tile_start: Optional[Value],
    k_tile_stop: Optional[Value],
    extra_score_transform: Optional[Callable[[IRBuilder, Value, Value, int], Value]],
    extra_mask_predicate: Optional[Callable[[IRBuilder, Value], Value]],
    extra_skip_predicate: Optional[Callable[[IRBuilder, Value], Value]],
    k_block_iter_fn: Optional[Callable[[IRBuilder, Value], Value]],
    v_scale: Optional[Value],
    v_lds_stage: bool = False,
    arch: str,
    target,
    ext: Optional[AttnFwdExt] = None,
) -> None:
    """One WMMA-tiled QK->softmax->PV pass for a ``BLOCK_M``-row Q tile (wave32).

    Drives the QK^T and PV matmuls through the ``wmma_f32_16x16x16_f16``
    ``MmaOp`` layout maps and :meth:`IRBuilder.mma`; the online softmax uses the
    wave32 row reduction. Parameter semantics match
    :func:`mfma_attention_fwd_inner_body`. The kernel must launch with
    ``block_size == wave_size`` (one wave32 per CTA).
    """
    op_id = _wmma_attn_op_id(arch, dtype)
    op = target.mma.by_op_id(op_id)
    if op is None or op.family != "wmma":
        raise ValueError(f"WMMA attention atom {op_id} absent on {arch}")
    wave = op.wave_size  # 32
    dtype_ir = _ir_type_for_dtype(dtype)

    # Lane/slot coordinate maps come straight from the contract for THIS arch's
    # atom, so the gfx11 (cross-half-duplicated, a_frag=16) and gfx12 (split-K,
    # a_frag=8) ABIs are both expressed through the same accessors.
    a_map = op.a_layout()  # (row, k): lane l -> (row l%16, k=lane-base+slot)
    c_map = (
        op.c_layout()
    )  # (row, col): gfx11 (2i+l//16, l%16); gfx12 ((l//16)*8+i, l%16)
    a_frag = op.a_frag_len  # 16 (gfx11) | 8 (gfx12) -- K elems per lane per step
    c_frag = op.c_frag_len  # 8  -- accumulator slots per lane (same both)

    # Number of WMMA steps along the head-dim axis (QK K-dim == PV N-dim).
    n_dk = head_size // 16

    # Row reduction across the 16 lanes that share one accumulator row. The
    # stage count is derived from the atom geometry (log2(16) = 4), not
    # hard-coded; the XOR masks stay inside the 32-lane half on wave32.
    reduce_stages = wave_reduce_stages(wave_size=wave, lanes_per_row=16)
    assert reduce_stages == 4  # 16x16 tile -> 4-stage butterfly

    lane = b.mod(b.thread_id_x(), b.const_i32(wave))
    c16 = b.const_i32(16)

    # A/B-operand row for this lane (== lane % 16 for both Q and K fragments).
    a_row = a_map.coord(b, lane, 0)[0]
    # gfx12 (RDNA4) split-K: the 16 K-elements of one WMMA step are split across
    # the two lane-halves, so each lane loads ``a_frag`` (=8) elements from K base
    # ``(lane // 16) * a_frag``. gfx11 (RDNA3/3.5) duplicates the full K row in
    # every lane (a_frag=16, base 0). ``split_k`` keeps the gfx11 emission
    # byte-identical (no half-offset add at all) while the gfx12 Q/K/V loads pick
    # up the per-half K offset; mirrors ``split_k_by_half`` in
    # ``instances/common/_matmul_nbits_large_n.py``.
    split_k = a_frag * 2 == 16  # a_frag==8 -> two halves cover K=16 (gfx12)
    k_half_off = b.mul(b.div(lane, c16), b.const_i32(a_frag)) if split_k else None
    # Accumulator column == this lane's k-position in the QK score tile.
    col = b.mod(lane, c16)

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    k_off = k_token_offset_elems if k_token_offset_elems is not None else b.const_i32(0)
    v_off = v_token_offset_elems if v_token_offset_elems is not None else b.const_i32(0)

    # ---- Pre-load Q fragments (constant across the K-loop) ----
    # Lane l holds Q row (q_tile_base + a_row), full <a_frag x half> d-slice for
    # each head-dim WMMA tile. Q is K-loop-invariant -> register-resident.
    q_row = b.add(q_tile_base, a_row)
    if ext is not None and ext.seqlen_q is not None:
        seq_row = b.add(q_pos_base if q_pos_base is not None else q_tile_base, a_row)
        overshoot = b.smax(b.sub(seq_row, last_index(b, ext.seqlen_q)), b.const_i32(0))
        q_row = b.sub(q_row, overshoot)
    q_addr_row_base = b.add(
        b.mul(q_row, stride_q_token),
        b.mul(head_idx, stride_q_head),
    )
    q_frags = []
    for d in range(n_dk):
        q_addr = b.add(q_addr_row_base, b.const_i32(d * 16))
        if k_half_off is not None:
            q_addr = b.add(q_addr, k_half_off)
        q_frags.append(b.global_load_vN(Q, q_addr, dtype_ir, a_frag, align=a_frag * 2))

    # ---- LDS staging tiles ----
    # P_lds transposes the score acc layout -> the PV A-operand layout.
    # V_lds stages the K-tile's V rows once per iteration so the PV B-operand
    # (V in d x k layout) is read from LDS instead of a per-(d,k) scalar
    # global gather.
    P_lds = b.smem_alloc(dtype_ir, [16, 16], name_hint="Pwmma")
    V_lds = (
        b.smem_alloc(dtype_ir, [16, head_size], name_hint="Vwmma")
        if v_lds_stage
        else None
    )

    # ---- Online-softmax + PV accumulator iter-args ----
    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", neg_inf))
        iter_args.append((f"l{r}", zero_f))
    for d in range(n_dk):
        iter_args.append((f"acc{d}", b.zero_vec_f32(c_frag)))

    c_block_k = b.const_i32(MFMA_ATTN_BLOCK_K)
    k_guard = ext is not None and ext.guard_seqlen_k
    if k_guard:
        k_last = last_index(b, seqlen_k)
    loop_start = k_tile_start if k_tile_start is not None else b.const_i32(0)
    if k_tile_stop is not None:
        loop_stop = k_tile_stop
    elif k_guard:
        loop_stop = b.div(
            b.add(seqlen_k, b.const_i32(MFMA_ATTN_BLOCK_K - 1)), c_block_k
        )
    else:
        loop_stop = b.div(seqlen_k, c_block_k)

    kloop = b.scf_for_iter(
        loop_start, loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms = [state[2 * r] for r in range(c_frag)]
        ls = [state[2 * r + 1] for r in range(c_frag)]
        accs = list(state[2 * c_frag :])

        if k_block_iter_fn is not None:
            effective_kt = k_block_iter_fn(b, kt)
        else:
            effective_kt = kt

        k_tile_base = b.mul(effective_kt, c_block_k)
        k_row_for_lane = b.add(k_tile_base, a_row)  # k position for THIS lane
        if k_guard:
            k_row_for_lane = b.smin(k_row_for_lane, k_last)

        # Per-K-tile keep / skip predicates (block-sparse). Same semantics as
        # the MFMA body: when false, the score collapses to -inf so the
        # softmax exponential is zero (loads are still issued).
        if extra_mask_predicate is not None:
            keep_tile = extra_mask_predicate(b, kt)
        else:
            keep_tile = None
        if extra_skip_predicate is not None:
            skip_mask = extra_skip_predicate(b, kt)
            keep_tile = (
                b.land(keep_tile, skip_mask) if keep_tile is not None else skip_mask
            )

        if k_row_base_fn is not None:
            k_addr_row_base = k_row_base_fn(b, k_row_for_lane)
        else:
            k_addr_row_base = b.add(
                b.add(
                    b.mul(k_row_for_lane, stride_k_token),
                    b.mul(kv_head_idx, stride_k_head),
                ),
                k_off,
            )

        # ---- QK^T WMMA chain: score = sum_d Q[d-tile] (x) K[d-tile] ----
        score = b.zero_vec_f32(c_frag)
        for d in range(n_dk):
            k_addr = b.add(k_addr_row_base, b.const_i32(d * 16))
            if k_half_off is not None:
                k_addr = b.add(k_addr, k_half_off)
            k_frag = b.global_load_vN(K, k_addr, dtype_ir, a_frag, align=a_frag * 2)
            score = b.mma(op, q_frags[d], k_frag, score)

        # ---- Scale + mask + per-row online softmax ----
        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = []  # per-slot scaled probabilities (acc layout)
        q_pos_for_mask = q_pos_base if q_pos_base is not None else q_tile_base
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)  # (q-row in tile, k-col)
            s_r = b.fmul(b.vec_extract(score, r), scale_log2)
            row_q_pos = b.add(q_pos_for_mask, row_rel)
            k_col_pos = b.add(k_tile_base, col_k)
            if extra_score_transform is not None:
                s_r = extra_score_transform(b, s_r, kt, r)
            if ext is not None:
                s_r = score_stage(
                    b,
                    ext,
                    s_r,
                    q_row=row_q_pos,
                    k_col=k_col_pos,
                    kt=kt,
                    slot=r,
                    seqlen_k=seqlen_k,
                )
            s_r = apply_attention_mask(
                b,
                s_r,
                mask_mode=mask_mode,
                k_idx=k_col_pos,
                query_pos=row_q_pos,
                sliding_window=sliding_window,
                context_len=causal_ctx_offset,
            )
            if keep_tile is not None:
                s_r = b.select(keep_tile, s_r, neg_inf)
            # Per-row reduce across the 16 k-columns of this wave32 half. The
            # distribution-driven ``block_tile_reduce_sync`` emits the same
            # 4-stage in-half XOR butterfly as the legacy wave32
            # ``wave_reduce_max(lanes_per_row=16)``.
            row_max = _softmax_row_reduce(b, s_r, combine="max")
            m_new = b.fmax(ms[r], row_max)
            alpha = b.exp2(b.fsub(ms[r], m_new))
            p_r = b.exp2(b.fsub(s_r, m_new))
            row_sum = _softmax_row_reduce(b, p_r, combine="sum")
            l_new = b.fadd(b.fmul(ls[r], alpha), row_sum)
            new_ms.append(m_new)
            new_ls.append(l_new)
            ps.append(p_r)
            for d in range(n_dk):
                old = b.vec_extract(new_accs[d], r)
                new_accs[d] = b.vec_insert(new_accs[d], b.fmul(old, alpha), r)

        # ---- V staging into LDS (vectorized load; transposed PV reads) ----
        # Each lane loads its own k-row's full head_size d-slice as 8-wide
        # vector global loads and writes it row-major into ``V_lds``. The PV
        # B-operand (V in d x k layout) is then a strided *LDS* read, replacing
        # the per-(d,k) scalar global gather the correctness-first version did
        # (it issued ``n_dk * a_frag`` scalar global loads per lane per K-tile).
        # Both wave32 halves map to the same 16 rows (a_row == lane % 16), so
        # the store is redundant across halves but writes identical data.
        if v_lds_stage:
            v_stage_row = b.add(k_tile_base, a_row)
            if k_guard:
                v_stage_row = b.smin(v_stage_row, k_last)
            if v_row_base_fn is not None:
                v_stage_base = v_row_base_fn(b, v_stage_row)
            else:
                v_stage_base = b.add(
                    b.add(
                        b.mul(v_stage_row, stride_v_token),
                        b.mul(kv_head_idx, stride_v_head),
                    ),
                    v_off,
                )
            for e in range(head_size // 8):
                v_g = b.global_load_vN(
                    V, b.add(v_stage_base, b.const_i32(e * 8)), dtype_ir, 8, align=16
                )
                b.smem_store_vN(V_lds, [a_row, b.const_i32(e * 8)], v_g, 8)

        # ---- P staging through LDS: acc layout -> A-operand layout ----
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            b.smem_store_vN(P_lds, [row_rel, col_k], b.cast_f32_to(ps[r], dtype_ir), 1)
        b.sync()

        # ---- V load + PV WMMA chain ----
        # PV computes O = P @ V. WMMA evaluates A @ B^T, so B must be V in
        # (d x k) = N x K layout: the B-operand for d-column c is the V *column*
        # c gathered over k = 0..a_frag-1. P A-operand: lane l holds q-row a_row,
        # fragment slot j = P[row, j].
        p_a = b.zero_vec(dtype_ir, a_frag)
        for j in range(a_frag):
            # A-operand layout map gives slot j's (row, k); the row is the
            # loop-invariant ``a_row`` (== lane % 16, hoisted above) so we only
            # take the K coordinate from the map -- the P column to read.
            a_k = a_map.coord(b, lane, j)[1]
            p_v = b.vec_extract(
                b.smem_load_vN(P_lds, a_row, a_k, dtype=dtype_ir, n=1), 0
            )
            p_a = b.vec_insert(p_a, p_v, j)

        for d in range(n_dk):
            d_col = b.add(b.const_i32(d * 16), col)  # this lane's V d-column
            v_b = b.zero_vec(dtype_ir, a_frag)
            for j in range(a_frag):
                # B-operand for d-column ``d_col`` is V[k, d_col]. The K row this
                # lane's slot j feeds is ``j`` on gfx11 (every lane covers the full
                # K, byte-identical to the historical literal) and
                # ``(lane // 16) * a_frag + j`` on gfx12 (split-K halves). The
                # gfx12 base is added via ``k_half_off`` so the gfx11 path emits
                # exactly the previous IR.
                b_k = (
                    b.add(k_half_off, b.const_i32(j))
                    if k_half_off is not None
                    else b.const_i32(j)
                )
                if v_lds_stage:
                    # Optimized: read from the staged LDS tile (V_lds[k, d_col]).
                    v_elem = b.vec_extract(
                        b.smem_load_vN(V_lds, b_k, d_col, dtype=dtype_ir, n=1),
                        0,
                    )
                else:
                    # Baseline: per-(d,k) scalar global gather of V[k, d_col].
                    v_row = b.add(k_tile_base, b_k)
                    if k_guard:
                        v_row = b.smin(v_row, k_last)
                    if v_row_base_fn is not None:
                        v_row_base = v_row_base_fn(b, v_row)
                    else:
                        v_row_base = b.add(
                            b.add(
                                b.mul(v_row, stride_v_token),
                                b.mul(kv_head_idx, stride_v_head),
                            ),
                            v_off,
                        )
                    v_elem = b.global_load(
                        V, b.add(v_row_base, d_col), dtype_ir, align=2
                    )
                v_b = b.vec_insert(v_b, v_elem, j)
            new_accs[d] = b.mma(op, p_a, v_b, new_accs[d])

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        b.scf_yield(*yields)

    final = kloop.results
    ls_final = [final[2 * r + 1] for r in range(c_frag)]
    accs_final = list(final[2 * c_frag :])

    if ext is not None and ext.wants_epilogue:
        emit_rows_epilogue(
            b,
            ext,
            n_slots=c_frag,
            n_tiles=n_dk,
            row_rel_fn=lambda r: c_map.coord(b, lane, r)[0],
            q_pos_base=q_pos_base if q_pos_base is not None else q_tile_base,
            ms=[final[2 * r] for r in range(c_frag)],
            ls=ls_final,
            accs=accs_final,
            is_row_leader=b.cmp_eq(col, b.const_i32(0)),
            out_addr_fn=lambda r, n, row_rel: b.add(
                b.add(
                    b.mul(b.add(q_tile_base, row_rel), stride_o_token),
                    b.mul(head_idx, stride_o_head),
                ),
                b.add(b.const_i32(n * 16), col),
            ),
            O=O,
            dtype_ir=dtype_ir,
            v_scale=v_scale,
        )
        return

    # ---- Epilogue: O[q,d] = acc[q,d] / l[q] (zero-denominator guarded) ----
    for d in range(n_dk):
        for r in range(c_frag):
            row_rel, col_n = c_map.coord(b, lane, r)  # (q-row in tile, d-col)
            l_safe = ls_final[r]
            zero_mask = b.fcmp("oeq", l_safe, zero_f)
            inv_l = b.select(zero_mask, zero_f, b.rcp(l_safe))
            v_f32 = b.fmul(b.vec_extract(accs_final[d], r), inv_l)
            if v_scale is not None:
                v_f32 = b.fmul(v_f32, v_scale)
            o_row = b.add(q_tile_base, row_rel)
            o_col = b.add(b.const_i32(d * 16), col_n)
            o_addr = b.add(
                b.add(
                    b.mul(o_row, stride_o_token),
                    b.mul(head_idx, stride_o_head),
                ),
                o_col,
            )
            b.global_store(O, o_addr, b.cast_f32_to(v_f32, dtype_ir), align=2)
