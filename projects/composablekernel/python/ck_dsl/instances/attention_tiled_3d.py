# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled MFMA implementation of AITER's split-KV ``kernel_unified_attention_3d``.

The 3D path runs many CTAs per (q-block, kv-head), each one covering a
segment of the KV sequence. After the segment kernel, ``reduce_segments``
combines the per-segment partial m / l / acc into the final attention
output. AITER selects this path whenever
``use_2d_kernel == False``: i.e. for long sequences with no sliding window
when 2D grid is too small for the device.

This module re-uses every MFMA / async DMA / softmax helper from the
2D tiled kernel and changes only what's structural to the 3D variant:

  - Grid is ``(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)``.
  - ``tile_start..tile_end`` is bounded by the segment range, not the full
    sequence.
  - Outputs are written to a workspace ``segm_output[total_q, num_qh,
    num_segments, head_size]`` plus ``segm_max[..., num_segments]`` and
    ``segm_expsum[..., num_segments]`` (all fp32).
  - The final ``acc /= L`` and fp16 cast happen in the reduce kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from ..core.ir import (
    BF16,
    F16,
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)

from .attention_tiled_2d import (
    _apply_softcap,
    _binary_search_seq_idx,
    _mfma_16x16x16,
    _mfma_16x16x32,
    _warp_xor_reduce_max,
    _warp_xor_reduce_sum,
)


MFMA_M = 16
MFMA_N = 16


@dataclass(frozen=True)
class UnifiedAttention3DTiledSpec:
    """Spec for the split-KV 3D segment kernel.

    Mirrors :class:`UnifiedAttention2DTiledSpec` and adds the segment-count
    knob exactly as AITER's ``select_3d_config`` derives it.
    """

    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    use_sinks: bool
    sliding_window: int
    has_softcap: bool
    num_segments: int
    use_alibi: bool = False
    use_qq_bias: bool = False
    num_seqs: int = 0

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_m(self) -> int:
        return 16

    @property
    def block_q(self) -> int:
        return self.block_m // self.num_queries_per_kv

    @property
    def tile_size(self) -> int:
        return self.block_size

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    @property
    def binary_search_iters(self) -> int:
        if self.num_seqs <= 0:
            return 32
        return max(1, int(math.ceil(math.log2(self.num_seqs + 1))))

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        return kernel_name_join(
            "ck_dsl_uattn3d_tiled",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            f"seg{self.num_segments}",
            self.dtype,
            "sinks" if self.use_sinks else "",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
        )


def supports_tiled_3d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_fp8: bool,
    q_dtype,
) -> Tuple[bool, str]:
    if dtype not in ("fp16", "bf16"):
        return False, f"tiled 3D kernel currently supports fp16/bf16 (got {dtype!r})"
    if head_size not in (128, 256):
        return (
            False,
            f"tiled 3D kernel only supports head_size in {{128,256}} (got {head_size})",
        )
    if block_size not in (16, 64):
        return (
            False,
            f"tiled 3D kernel only supports block_size in {{16,64}} (got {block_size})",
        )
    if num_queries_per_kv > 16 or num_queries_per_kv < 1:
        return (
            False,
            f"tiled 3D kernel needs 1<=num_queries_per_kv<=16 (got {num_queries_per_kv})",
        )
    if 16 % num_queries_per_kv != 0:
        return False, "tiled 3D kernel needs num_queries_per_kv to divide BLOCK_M=16"
    if use_fp8 or q_dtype is not None:
        return False, "tiled 3D kernel does not implement FP8 path yet"
    # ALiBi and QQ-bias are now supported by the tiled 3D kernel.
    return True, "supported"


def build_unified_attention_3d_tiled(spec: UnifiedAttention3DTiledSpec) -> KernelDef:
    """Emit the tiled split-KV 3D segment kernel.

    Each CTA writes its segment's partial state into ``segm_output``,
    ``segm_max``, and ``segm_expsum``. The companion reduce kernel
    (:func:`build_unified_attention_reduce_tiled`) combines those into the
    final output.
    """
    if spec.dtype not in ("fp16", "bf16"):
        raise NotImplementedError("tiled 3D kernel supports fp16/bf16")
    dtype = spec.dtype_ir

    HD = spec.head_size
    T = spec.tile_size
    BS = spec.block_size
    BLOCK_M = spec.block_m
    BLOCK_Q = spec.block_q
    NQK = spec.num_queries_per_kv
    NUM_KV = spec.num_kv_heads
    NUM_QH = spec.num_query_heads
    NUM_SEG = spec.num_segments
    SLIDING_WINDOW = spec.sliding_window
    USE_SOFTCAP = spec.has_softcap
    USE_SINKS = spec.use_sinks
    USE_ALIBI = spec.use_alibi
    USE_QQ_BIAS = spec.use_qq_bias

    QK_K_STEP = 32
    PV_K_STEP = 32 if T % 32 == 0 else 16
    QK_K_ITERS = HD // QK_K_STEP
    QK_N_TILES = T // MFMA_N
    PV_K_ITERS = T // PV_K_STEP
    PV_N_TILES = HD // MFMA_N

    THREADS = 64

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    # ---------------- parameter declarations ----------------
    # NOTE: the AITER 3D signature distinguishes between segm_* workspace
    # pointers and the regular K/V/cu/seq_lens inputs. We mirror that order.
    segm_output_ptr = b.param(
        "segm_output_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    segm_max_ptr = b.param(
        "segm_max_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    segm_expsum_ptr = b.param(
        "segm_expsum_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    alibi_slopes_ptr = b.param(
        "alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4
    )
    qq_bias_ptr = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale_p = b.param("scale", F32)
    _k_scale = b.param("k_scale", F32)
    _v_scale = b.param("v_scale", F32)
    softcap_p = b.param("softcap", F32)
    num_seqs_p = b.param("num_seqs", I32)
    bt_stride_p = b.param("block_table_stride", I32)
    qq_bias_stride0_p = b.param("qq_bias_stride_0", I32)

    q_block_global_idx = b.block_id_x()
    kv_head_idx = b.block_id_y()
    seg_idx = b.block_id_z()
    tid = b.thread_id_x()

    seq_idx = _binary_search_seq_idx(
        b, cu_q, q_block_global_idx, num_seqs_p, BLOCK_Q, spec.binary_search_iters
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    cur_batch_q_len = b.sub(cu_q_stop, cu_q_start)
    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(BLOCK_Q)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    seq_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(seq_len, cur_batch_q_len)

    qb_start_pos = b.mul(q_block_local_idx, b.const_i32(BLOCK_Q))
    with b.scf_if(b.cmp_ge(qb_start_pos, cur_batch_q_len)):
        b.ret()

    # tiles_per_segment = cdiv(seq_len, NUM_SEG * T)
    tps = b.div(b.add(seq_len, b.const_i32(NUM_SEG * T - 1)), b.const_i32(NUM_SEG * T))

    # If this segment is past seq_len, write a neutral entry to the workspace
    # (m=-inf zeros the reduce contribution; acc=0 and l=0 keep everything
    # finite even when other segments have finite m). AITER's Triton kernel
    # achieves the same effect through `tl.store` masks.
    seg_start_tile_pos = b.mul(b.mul(seg_idx, tps), b.const_i32(T))
    with b.scf_if(b.cmp_ge(seg_start_tile_pos, seq_len)):
        neg_inf_local = b.const_f32(float("-inf"))
        zero_local = b.const_f32(0.0)
        ml_stride_qtoken_e = NUM_QH * NUM_SEG
        ml_stride_qhead_e = NUM_SEG
        seg_acc_stride_qt = NUM_QH * NUM_SEG * HD
        seg_acc_stride_qh = NUM_SEG * HD
        seg_acc_stride_seg = HD
        lane_writes_ml_e = b.cmp_eq(b.mod(tid, b.const_i32(16)), b.const_i32(0))
        for reg in range(4):
            row = b.add(
                b.mul(b.div(tid, b.const_i32(16)), b.const_i32(4)), b.const_i32(reg)
            )
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            qp_r_safe = b.select(row_ok, qp_r, b.const_i32(0))
            qh_r_safe = b.select(row_ok, qh_r, b.const_i32(0))
            qtoken = b.add(cu_q_start, qp_r_safe)
            ml_idx = b.add(
                b.add(
                    b.mul(qtoken, b.const_i32(ml_stride_qtoken_e)),
                    b.mul(qh_r_safe, b.const_i32(ml_stride_qhead_e)),
                ),
                seg_idx,
            )
            with b.scf_if(lane_writes_ml_e):
                b.global_store(segm_max_ptr, ml_idx, neg_inf_local, align=4)
                b.global_store(segm_expsum_ptr, ml_idx, zero_local, align=4)
        # Zero acc for this segment across all (n, lane_col) entries
        # belonging to this CTA. Each lane writes its slot.
        lane_rg_e = b.div(tid, b.const_i32(16))
        lane_col_e = b.mod(tid, b.const_i32(16))
        for n in range(PV_N_TILES):
            for reg in range(4):
                row = b.add(b.mul(lane_rg_e, b.const_i32(4)), b.const_i32(reg))
                col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col_e)
                qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
                qh_r = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                row_ok = b.land(
                    b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
                )
                qp_r_safe = b.select(row_ok, qp_r, b.const_i32(0))
                qh_r_safe = b.select(row_ok, qh_r, b.const_i32(0))
                qtoken = b.add(cu_q_start, qp_r_safe)
                seg_acc_idx = b.add(
                    b.add(
                        b.mul(qtoken, b.const_i32(seg_acc_stride_qt)),
                        b.mul(qh_r_safe, b.const_i32(seg_acc_stride_qh)),
                    ),
                    b.add(b.mul(seg_idx, b.const_i32(seg_acc_stride_seg)), col),
                )
                b.global_store(segm_output_ptr, seg_acc_idx, zero_local, align=4)
        b.ret()

    # ---------------- LDS layout ----------------
    Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    K_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Vlds")
    P_lds = b.smem_alloc(dtype, [BLOCK_M, T], name_hint="Plds")

    tr_lane_div_16 = b.div(tid, b.const_i32(16))
    tr_lane_div_4_mod_4 = b.mod(b.div(tid, b.const_i32(4)), b.const_i32(4))
    tr_lane_mod_4 = b.mod(tid, b.const_i32(4))
    tr_col_lane = b.mul(tr_lane_mod_4, b.const_i32(4))

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    qk_scale = b.fmul(scale_p, rcp_ln2)
    sw_const = b.const_i32(int(SLIDING_WINDOW))
    z8 = b.zero_vec(dtype, 8)

    # ---------------- Q -> LDS ----------------
    Q_VECS_PER_ROW = HD // 8
    Q_VECS_PER_THREAD = (BLOCK_M * Q_VECS_PER_ROW) // THREADS
    for li in range(Q_VECS_PER_THREAD):
        q_vid = b.add(b.mul(b.const_i32(li), b.const_i32(THREADS)), tid)
        Q_row = b.div(q_vid, b.const_i32(Q_VECS_PER_ROW))
        Q_col = b.mul(b.mod(q_vid, b.const_i32(Q_VECS_PER_ROW)), b.const_i32(8))
        q_pos_t = b.add(qb_start_pos, b.div(Q_row, b.const_i32(NQK)))
        qh_t = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(Q_row, b.const_i32(NQK))
        )
        qmask_t = b.land(
            b.cmp_lt(q_pos_t, cur_batch_q_len), b.cmp_lt(qh_t, b.const_i32(NUM_QH))
        )
        q_pos_safe = b.select(qmask_t, q_pos_t, b.const_i32(0))
        qh_safe = b.select(qmask_t, qh_t, b.const_i32(0))
        q_off_base = b.add(
            b.mul(b.add(cu_q_start, q_pos_safe), b.const_i32(NUM_QH * HD)),
            b.mul(qh_safe, b.const_i32(HD)),
        )
        v8 = b.global_load_vN(query, b.add(q_off_base, Q_col), dtype, 8, align=16)
        b.smem_store_vN(
            Q_lds,
            [Q_row, Q_col],
            b.vector_select(b.vector_splat(qmask_t, 8), v8, z8),
            8,
        )
    b.sync()

    # ---------------- Per-segment tile range ----------------
    bm1_div_nqk = (BLOCK_M - 1) // NQK
    msp_raw = b.add(b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1))
    max_seq_prefix_len = b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)
    num_tiles = b.div(b.add(max_seq_prefix_len, b.const_i32(T - 1)), b.const_i32(T))

    # Segment bounds: [seg_idx * tps, min((seg_idx+1)*tps, num_tiles))
    tile_start = b.mul(seg_idx, tps)
    tile_end_raw = b.mul(b.add(seg_idx, b.const_i32(1)), tps)
    tile_end = b.select(b.cmp_lt(tile_end_raw, num_tiles), tile_end_raw, num_tiles)

    # The sliding-window path is not used by the AITER 3D selector
    # (use_2d_kernel returns True whenever sliding_window > 0), but we still
    # mirror the mask code in case a future selector wants it. We never
    # restrict the segment range by sliding window: the per-cell mask below
    # already handles that case.
    _ = sw_const  # documented; only used inside the per-cell mask code below

    # ---------------- online softmax registers ----------------
    lane_rg = b.div(tid, b.const_i32(16))
    lane_col = b.mod(tid, b.const_i32(16))

    if USE_SINKS:
        # Triton's 3D applies sinks only when segm_idx == 0.
        seg0 = b.cmp_eq(seg_idx, b.const_i32(0))
        m_inits = []
        for r in range(4):
            row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(r))
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            sink_with_mask = b.select(qh_in, sink_f, neg_inf)
            m_inits.append(b.select(seg0, sink_with_mask, neg_inf))
    else:
        m_inits = [neg_inf, neg_inf, neg_inf, neg_inf]
    l_inits = [one_f, one_f, one_f, one_f]

    acc_zero = b.zero_vec_f32(4)
    acc_inits = [acc_zero for _ in range(PV_N_TILES)]

    iter_args = []
    for r in range(4):
        iter_args.append((f"m{r}", m_inits[r]))
        iter_args.append((f"l{r}", l_inits[r]))
    for n in range(PV_N_TILES):
        iter_args.append((f"acc{n}", acc_inits[n]))

    # ---------------- async K/V infra (identical to 2D) ----------------
    big_bytes = b.const_i32(0x7FFF0000)
    key_rsrc = b.buffer_rsrc(key, big_bytes)
    value_rsrc = b.buffer_rsrc(value, big_bytes)

    KV_HALVES_PER_CALL = THREADS * 8
    assert (T * HD) % KV_HALVES_PER_CALL == 0
    kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL
    bytes_per_call = KV_HALVES_PER_CALL * 2
    kv_stride_blk_b = BS * NUM_KV * HD * 2
    kv_stride_tok_b = NUM_KV * HD * 2
    kv_stride_h_b = HD * 2
    bytes_per_buf = T * HD * 2

    lane_half_base = b.mul(tid, b.const_i32(8))
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    zero_soff = b.const_i32(0)

    def _kv_base_bytes(kv_tile_idx: Value) -> Value:
        physical_block = b.global_load_i32(
            block_tables,
            b.add(b.mul(seq_idx, bt_stride_p), kv_tile_idx),
        )
        return b.add(
            b.mul(physical_block, b.const_i32(kv_stride_blk_b)),
            b.mul(kv_head_idx, b.const_i32(kv_stride_h_b)),
        )

    def _issue_k_load(kv_tile_idx: Value, buf_idx: Value) -> None:
        kv_base = _kv_base_bytes(kv_tile_idx)
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            t_idx = b.div(linear_half, b.const_i32(HD))
            hd_idx_bytes = b.mul(b.mod(linear_half, b.const_i32(HD)), b.const_i32(2))
            voff = b.add(
                b.add(b.mul(t_idx, b.const_i32(kv_stride_tok_b)), hd_idx_bytes), kv_base
            )
            k_dst = b.smem_ptr_add(K_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(key_rsrc, k_dst, voff, zero_soff, 4)

    def _issue_v_load(kv_tile_idx: Value, buf_idx: Value) -> None:
        kv_base = _kv_base_bytes(kv_tile_idx)
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        V_buf_base = b.smem_ptr_add(V_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            t_idx = b.div(linear_half, b.const_i32(HD))
            hd_idx_bytes = b.mul(b.mod(linear_half, b.const_i32(HD)), b.const_i32(2))
            voff = b.add(
                b.add(b.mul(t_idx, b.const_i32(kv_stride_tok_b)), hd_idx_bytes), kv_base
            )
            v_dst = b.smem_ptr_add(V_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(value_rsrc, v_dst, voff, zero_soff, 4)

    _issue_k_load(tile_start, b.const_i32(0))

    cur_buf_init = b.const_i32(0)
    iter_args.append(("cur_buf", cur_buf_init))

    kvloop = b.scf_for_iter(
        tile_start, tile_end, b.const_i32(1), iter_args, iv_name="kv_tile"
    )
    with kvloop as (kv_tile_iv, carry):
        m_vals = [carry[2 * r] for r in range(4)]
        l_vals = [carry[2 * r + 1] for r in range(4)]
        acc_vals = [carry[8 + n] for n in range(PV_N_TILES)]
        cur_buf = carry[8 + PV_N_TILES]
        nxt_buf = b.sub(b.const_i32(1), cur_buf)
        tile_off = b.mul(kv_tile_iv, b.const_i32(T))

        next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
        in_range_next = b.cmp_lt(next_tile_iv_raw, tile_end)
        safe_next_tile = b.select(in_range_next, next_tile_iv_raw, kv_tile_iv)

        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.sync()

        # QK
        A_kits = []
        for k in range(QK_K_ITERS):
            q_col_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
            A_kits.append(b.smem_load_vN(Q_lds, lane_col, q_col_off, dtype=dtype, n=8))
        S_n = []
        for n in range(QK_N_TILES):
            acc_v = b.zero_vec_f32(4)
            for k in range(QK_K_ITERS):
                kc_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                B_v = b.smem_load_vN(K_lds, cur_buf, k_row, kc_off, dtype=dtype, n=8)
                acc_v = _mfma_16x16x32(b, dtype, A_kits[k], B_v, acc_v)
            S_n.append(acc_v)

        _issue_v_load(kv_tile_iv, cur_buf)
        _issue_k_load(safe_next_tile, nxt_buf)

        # See attention_tiled_2d.py for the rationale on applying ALiBi /
        # QQ-bias before the select-with-(-inf) (equivalent to Triton's
        # post-select add for finite biases via IEEE -inf semantics, with
        # better robustness against compiler reordering).
        if USE_ALIBI:
            alibi_per_row = []
            for reg in range(4):
                row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
                qh_r = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                qh_ok = b.cmp_lt(qh_r, b.const_i32(NUM_QH))
                slope = b.masked_global_load(
                    alibi_slopes_ptr, qh_r, qh_ok, b.const_f32(0.0), dtype=F32, align=4
                )
                alibi_per_row.append(slope)
        masked = {}
        for reg in range(4):
            row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            for n in range(QK_N_TILES):
                col_abs = b.add(
                    b.add(tile_off, b.mul(b.const_i32(n), b.const_i32(16))), lane_col
                )
                causal_lim = b.add(context_len, qp_r)
                causal_ok = b.cmp_le(col_abs, causal_lim)
                in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                m_ok = b.land(b.land(row_ok, causal_ok), in_prefix)
                if SLIDING_WINDOW > 0:
                    dist = b.sub(causal_lim, col_abs)
                    m_ok = b.land(m_ok, b.cmp_lt(dist, sw_const))
                s_raw = b.vec_extract(S_n[n], reg)
                s_scaled = b.fmul(s_raw, qk_scale)
                if USE_SOFTCAP:
                    s_scaled = b.fmul(_apply_softcap(b, s_scaled, softcap_p), rcp_ln2)
                if USE_ALIBI:
                    pos_off = b.sub(col_abs, context_len)
                    pos_f = b.sitofp_f32(pos_off)
                    add_term = b.fmul(b.fmul(alibi_per_row[reg], pos_f), rcp_ln2)
                    s_scaled = b.fadd(s_scaled, add_term)
                if USE_QQ_BIAS:
                    # See attention_tiled_2d.py for the rationale on row_ok.
                    krp = b.sub(col_abs, context_len)
                    krp_ok = b.land(
                        b.cmp_ge(krp, b.const_i32(0)), b.cmp_lt(krp, qq_bias_stride0_p)
                    )
                    qq_ok = b.land(row_ok, krp_ok)
                    qp_safe = b.select(row_ok, qp_r, b.const_i32(0))
                    qq_idx = b.add(b.mul(qp_safe, qq_bias_stride0_p), krp)
                    qq_v = b.masked_global_load(
                        qq_bias_ptr,
                        qq_idx,
                        qq_ok,
                        b.const_f32(0.0),
                        dtype=F32,
                        align=4,
                    )
                    s_scaled = b.fadd(s_scaled, b.fmul(qq_v, rcp_ln2))
                masked[(n, reg)] = b.select(m_ok, s_scaled, neg_inf)

        m_new = []
        s_local = {}
        for reg in range(4):
            local_max = neg_inf
            for n in range(QK_N_TILES):
                v = masked[(n, reg)]
                s_local[(reg, n)] = v
                local_max = b.fmax(local_max, v)
            full_max_raw = _warp_xor_reduce_max(b, local_max)
            ok = b.fcmp("ogt", full_max_raw, neg_inf)
            m_new.append(b.select(ok, full_max_raw, zero_f))

        l_local = []
        for reg in range(4):
            row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            sum_p = zero_f
            for n in range(QK_N_TILES):
                p = b.exp2(b.fsub(s_local[(reg, n)], m_new[reg]))
                col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
                b.smem_store_vN(P_lds, [row, col], b.cast_f32_to(p, dtype), 1)
                sum_p = b.fadd(sum_p, p)
            l_local.append(_warp_xor_reduce_sum(b, sum_p))

        alpha_regs = [b.exp2(b.fsub(m_vals[r], m_new[r])) for r in range(4)]
        new_l_vals = [
            b.fadd(b.fmul(l_vals[r], alpha_regs[r]), l_local[r]) for r in range(4)
        ]
        b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
        b.sync()

        new_acc = []
        for n in range(PV_N_TILES):
            scaled_comps = []
            for reg in range(4):
                e = b.vec_extract(acc_vals[n], reg)
                scaled_comps.append(b.fmul(e, alpha_regs[reg]))
            acc_v = b.vec_pack(scaled_comps, F32)

            K_L_pv = PV_K_STEP // 4
            tr_row_base = b.add(
                b.mul(tr_lane_div_16, b.const_i32(K_L_pv)), tr_lane_div_4_mod_4
            )
            n_col_base = b.add(b.mul(b.const_i32(n), b.const_i32(16)), tr_col_lane)

            for k in range(PV_K_ITERS):
                if PV_K_STEP == 32:
                    p_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                    A_p = b.smem_load_vN(P_lds, lane_col, p_off, dtype=dtype, n=8)
                    row_r0 = b.add(b.const_i32(k * 32), tr_row_base)
                    row_r1 = b.add(b.const_i32(k * 32 + 4), tr_row_base)
                    B_r0 = b.ds_read_tr16_b64(
                        V_lds, cur_buf, row_r0, n_col_base, dtype=dtype
                    )
                    B_r1 = b.ds_read_tr16_b64(
                        V_lds, cur_buf, row_r1, n_col_base, dtype=dtype
                    )
                    B_v = b.vec_concat(B_r0, B_r1)
                    acc_v = _mfma_16x16x32(b, dtype, A_p, B_v, acc_v)
                else:
                    p_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                    A_p = b.smem_load_vN(P_lds, lane_col, p_off, dtype=dtype, n=4)
                    row_lane = b.add(b.const_i32(k * 16), tr_row_base)
                    B_v = b.ds_read_tr16_b64(
                        V_lds, cur_buf, row_lane, n_col_base, dtype=dtype
                    )
                    acc_v = _mfma_16x16x16(b, dtype, A_p, B_v, acc_v)
            new_acc.append(acc_v)

        yields = []
        for r in range(4):
            yields.append(m_new[r])
            yields.append(new_l_vals[r])
        for n in range(PV_N_TILES):
            yields.append(new_acc[n])
        yields.append(nxt_buf)
        b.scf_yield(*yields)

    # ---------------- write segment workspace ----------------
    final = kvloop.results
    m_final = [final[2 * r] for r in range(4)]
    l_final = [final[2 * r + 1] for r in range(4)]
    acc_final = [final[8 + n] for n in range(PV_N_TILES)]

    # Per-thread: write own (row, col) of acc; only the lane in each
    # 4-lane row group (lane%4 == 0) writes m and l.
    seg_stride_qtoken = NUM_QH * NUM_SEG * HD
    seg_stride_qhead = NUM_SEG * HD
    seg_stride_seg = HD
    ml_stride_qtoken = NUM_QH * NUM_SEG
    ml_stride_qhead = NUM_SEG

    # Only valid rows write to the workspace. Padding rows (rows of the
    # BLOCK_M tile that fall outside `cur_batch_q_len` or `NUM_QH`) would
    # otherwise race onto valid row 0's address after the qh/qtoken clamp.
    for n in range(PV_N_TILES):
        for reg in range(4):
            row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            qtoken = b.add(cu_q_start, qp_r)
            seg_acc_idx = b.add(
                b.add(
                    b.mul(qtoken, b.const_i32(seg_stride_qtoken)),
                    b.mul(qh_r, b.const_i32(seg_stride_qhead)),
                ),
                b.add(
                    b.mul(seg_idx, b.const_i32(seg_stride_seg)),
                    col,
                ),
            )
            v_acc = b.vec_extract(acc_final[n], reg)
            with b.scf_if(row_ok):
                b.global_store(segm_output_ptr, seg_acc_idx, v_acc, align=4)

    lane_writes_ml = b.cmp_eq(b.mod(tid, b.const_i32(16)), b.const_i32(0))
    for reg in range(4):
        row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
        qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
        qh_r = b.add(b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK)))
        row_ok = b.land(
            b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
        )
        qtoken = b.add(cu_q_start, qp_r)
        ml_idx = b.add(
            b.add(
                b.mul(qtoken, b.const_i32(ml_stride_qtoken)),
                b.mul(qh_r, b.const_i32(ml_stride_qhead)),
            ),
            seg_idx,
        )
        do_write = b.land(lane_writes_ml, row_ok)
        with b.scf_if(do_write):
            b.global_store(segm_max_ptr, ml_idx, m_final[reg], align=4)
            b.global_store(segm_expsum_ptr, ml_idx, l_final[reg], align=4)

    return b.kernel


# ---------------------------------------------------------------------------
# Reduce kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedAttentionReduceTiledSpec:
    head_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    num_segments: int

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        return kernel_name_join(
            "ck_dsl_uattn_reduce_tiled",
            f"d{self.head_size}",
            f"h{self.num_query_heads}",
            f"seg{self.num_segments}",
            self.dtype,
        )


def build_unified_attention_reduce_tiled(
    spec: UnifiedAttentionReduceTiledSpec,
) -> KernelDef:
    """Combine the per-segment partial state into the final fp16/bf16 output.

    Grid: ``(total_q, num_query_heads, 1)``.  Each CTA reduces over the
    ``NUM_SEGMENTS`` segments for one (token, head):

      1. overall_max = max(segm_max)
      2. overall_expsum = sum(segm_expsum * exp2(segm_max - overall_max))
      3. acc[d] = sum_s segm_output[s,d] * exp2(segm_max[s] - overall_max)
      4. output[d] = acc[d] / overall_expsum  (with 0/0 -> 0)
    """
    HD = spec.head_size
    NUM_SEG = spec.num_segments
    NUM_QH = spec.num_query_heads
    dtype = spec.dtype_ir

    THREADS = 64
    HALFS_PER_THREAD = HD // THREADS
    assert HALFS_PER_THREAD * THREADS == HD

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    out = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    seg_out = b.param(
        "segm_output_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    seg_max = b.param("segm_max_ptr", PtrType(F32, "global"), readonly=True, align=4)
    seg_l = b.param("segm_expsum_ptr", PtrType(F32, "global"), readonly=True, align=4)
    _seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)

    q_token = b.block_id_x()
    q_head = b.block_id_y()
    tid = b.thread_id_x()

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)

    base_ml = b.add(
        b.mul(q_token, b.const_i32(NUM_QH * NUM_SEG)),
        b.mul(q_head, b.const_i32(NUM_SEG)),
    )

    # ---- pass 1: overall_max ----
    max_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(NUM_SEG),
        b.const_i32(1),
        [("mx", neg_inf)],
        iv_name="s_mx",
    )
    with max_loop as (sv, (mx,)):
        idx = b.add(base_ml, sv)
        ms = b.global_load_f32(seg_max, idx)
        b.scf_yield(b.fmax(mx, ms))
    overall_max = max_loop.results[0]

    # ---- pass 2: overall_expsum ----
    sum_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(NUM_SEG),
        b.const_i32(1),
        [("den", zero_f)],
        iv_name="s_sum",
    )
    with sum_loop as (sv, (den,)):
        idx = b.add(base_ml, sv)
        ms = b.global_load_f32(seg_max, idx)
        ls = b.global_load_f32(seg_l, idx)
        # NaN-safe factor: when both ms and overall_max are -inf, the
        # difference is NaN; force factor to 0 in that case.
        ms_finite = b.fcmp("ogt", ms, neg_inf)
        factor_raw = b.exp2(b.fsub(ms, overall_max))
        factor = b.select(ms_finite, factor_raw, zero_f)
        b.scf_yield(b.fadd(den, b.fmul(ls, factor)))
    overall_expsum = sum_loop.results[0]
    safe_expsum = b.fcmp("oeq", overall_expsum, zero_f)
    inv_l = b.select(safe_expsum, zero_f, b.rcp(overall_expsum))

    # ---- pass 3: per-element reduce + normalize + write ----
    base_acc = b.add(
        b.mul(q_token, b.const_i32(NUM_QH * NUM_SEG * HD)),
        b.mul(q_head, b.const_i32(NUM_SEG * HD)),
    )
    # Each thread handles HALFS_PER_THREAD output dims.
    for li in range(HALFS_PER_THREAD):
        d = b.add(b.mul(b.const_i32(li), b.const_i32(THREADS)), tid)
        acc_loop = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(NUM_SEG),
            b.const_i32(1),
            [(f"ac{li}", zero_f)],
            iv_name=f"s_acc{li}",
        )
        with acc_loop as (sv, (ac,)):
            idx_ml = b.add(base_ml, sv)
            ms = b.global_load_f32(seg_max, idx_ml)
            idx_acc = b.add(
                b.add(base_acc, b.mul(sv, b.const_i32(HD))),
                d,
            )
            ov = b.global_load_f32(seg_out, idx_acc)
            ms_finite = b.fcmp("ogt", ms, neg_inf)
            factor_raw = b.exp2(b.fsub(ms, overall_max))
            factor = b.select(ms_finite, factor_raw, zero_f)
            b.scf_yield(b.fadd(ac, b.fmul(ov, factor)))
        scalar_out_f32 = b.fmul(acc_loop.results[0], inv_l)
        scalar_out = b.cast_f32_to(scalar_out_f32, dtype)
        out_idx = b.add(
            b.add(
                b.mul(q_token, b.const_i32(NUM_QH * HD)),
                b.mul(q_head, b.const_i32(HD)),
            ),
            d,
        )
        b.global_store(out, out_idx, scalar_out, align=2)

    return b.kernel
