# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 WMMA tiled-2D unified-attention forward.

This is the gfx1250 counterpart of the gfx950 tiled 2D unified-attention
entry point, but it is deliberately wave32/WMMA-specific:

* one wave32 CTA per ``(kv_head, q_block)``;
* ``BLOCK_M = 16`` logical rows, ``BLOCK_Q = 2`` for the target GQA-8 cohort;
* one paged-KV block per iteration (``T = block_size = 32``);
* Q/O in bf16, K/V stored as fp8e4m3 and dequantized to bf16 before the
  ``wmma_gfx1250_f32_16x16x32_bf16`` atom.

The first supported feature slice is the aiter unified-attention 2D trace cohort:
head_size=64, block_size=32, GQA-8, bf16 query/output, fp8e4m3 paged K/V, sinks
and sliding-window masks.  Everything outside that slice is rejected by
``supports_tiled_2d`` / spec validation so the common scalar fp8 leg can evolve
independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from ...core.ir import BF16, F32, FP8E4M3, I32, IRBuilder, KernelDef, PtrType, Type
from ...helpers.attention import PagedKvDescriptor, binary_search_seq_idx
from ._wmma_attention_common import (
    BLOCK_M as _BLOCK_M,
    HEAD_SIZE as _HEAD_SIZE,
    WAVE as _WAVE,
    WMMA_N as _WMMA_N,
    check_wmma_arch,
    compute_pv,
    compute_pv_from_probs,
    compute_qk_scores,
    load_q_frags,
    resolve_wmma,
    softmax_row_update,
    stage_v_tile,
)

__all__ = [
    "UnifiedAttention2DTiledSpec",
    "build_unified_attention_2d_tiled",
    "supports_tiled_2d",
]

_BLOCK_SIZE = 32
_NUM_QUERY_HEADS = 64
_NUM_KV_HEADS = 8
_NQK = _NUM_QUERY_HEADS // _NUM_KV_HEADS
_BLOCK_Q = _BLOCK_M // _NQK


@dataclass(frozen=True)
class UnifiedAttention2DTiledSpec:
    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    use_sinks: bool
    sliding_window: int
    has_softcap: bool
    use_alibi: bool = False
    use_qq_bias: bool = False
    num_seqs: int = 0
    num_warps: int = 1
    waves_per_eu: Optional[int] = None
    kv_storage_dtype: Optional[str] = "fp8e4m3"
    tile_size: Optional[int] = None
    block_m_per_warp: int = 16
    # Register-P is available for experiments, but the LDS bridge is the
    # correctness/performance-safe production path on gfx1250 today.
    use_register_p: bool = False

    def __post_init__(self) -> None:
        ok, why = supports_tiled_2d(
            head_size=self.head_size,
            block_size=self.block_size,
            dtype=self.dtype,
            num_queries_per_kv=self.num_queries_per_kv,
            use_alibi=self.use_alibi,
            use_qq_bias=self.use_qq_bias,
            use_fp8=self.kv_storage_dtype == "fp8e4m3",
            q_dtype=None,
            num_warps=self.num_warps,
            block_m_per_warp=self.block_m_per_warp,
            kv_storage_dtype=self.kv_storage_dtype,
            tile_size=self.tile_size_eff,
            arch="gfx1250",
        )
        if not ok:
            raise ValueError(why)
        if self.has_softcap:
            raise ValueError("gfx1250 tiled 2D does not support softcap yet")
        if self.sliding_window < 0:
            raise ValueError("sliding_window must be non-negative")

    @property
    def num_queries_per_kv(self) -> int:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_m(self) -> int:
        return _BLOCK_M

    @property
    def block_q(self) -> int:
        return _BLOCK_Q

    @property
    def tile_size_eff(self) -> int:
        return self.tile_size if self.tile_size is not None else self.block_size

    @property
    def n_blocks_per_tile(self) -> int:
        return 1

    @property
    def dtype_ir(self) -> Type:
        return BF16

    @property
    def binary_search_iters(self) -> int:
        if self.num_seqs <= 0:
            return 32
        return max(1, int(math.ceil(math.log2(self.num_seqs + 1))))

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn2d_tiled_gfx1250",
            "wmma16x16x32",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            self.dtype,
            "kvfp8e4m3",
            "sinks" if self.use_sinks else "",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "regp" if self.use_register_p else "ldsP",
        )


def supports_tiled_2d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_fp8: bool,
    q_dtype,
    num_warps: int = 1,
    block_m_per_warp: int = 16,
    kv_storage_dtype: Optional[str] = None,
    tile_size: Optional[int] = None,
    arch: str = "gfx1250",
    use_mfma_32x32x8: bool = False,
    use_transposed_qk_32x32: bool = False,
    use_k_single_buffer: bool = False,
    use_conflict_free_v_store: bool = False,
    use_k_sliced_ring: bool = False,
) -> Tuple[bool, str]:
    del (
        use_mfma_32x32x8,
        use_transposed_qk_32x32,
        use_k_single_buffer,
        use_conflict_free_v_store,
        use_k_sliced_ring,
    )
    if arch != "gfx1250":
        return False, f"gfx1250 tiled 2D only supports arch='gfx1250' (got {arch!r})"
    if dtype != "bf16":
        return (
            False,
            f"gfx1250 tiled 2D currently supports bf16 Q/O only (got {dtype!r})",
        )
    if head_size != _HEAD_SIZE:
        return (
            False,
            f"gfx1250 tiled 2D currently supports head_size=64 (got {head_size})",
        )
    if block_size != _BLOCK_SIZE:
        return (
            False,
            f"gfx1250 tiled 2D currently supports block_size=32 (got {block_size})",
        )
    if num_queries_per_kv != _NQK:
        return False, (
            "gfx1250 tiled 2D currently supports GQA-8 "
            f"(got num_queries_per_kv={num_queries_per_kv})"
        )
    if use_alibi:
        return False, "gfx1250 tiled 2D does not support ALiBi yet"
    if use_qq_bias:
        return False, "gfx1250 tiled 2D does not support QQ bias yet"
    if q_dtype is not None and q_dtype != "bf16":
        return False, f"gfx1250 tiled 2D: unsupported q_dtype {q_dtype!r}"
    if not use_fp8 or kv_storage_dtype != "fp8e4m3":
        return False, "gfx1250 tiled 2D requires fp8e4m3 paged K/V cache"
    if num_warps != 1 or block_m_per_warp != _BLOCK_M:
        return False, (
            "gfx1250 tiled 2D v1 is one wave32 CTA "
            f"(num_warps={num_warps}, block_m_per_warp={block_m_per_warp})"
        )
    if tile_size is not None and tile_size != block_size:
        return False, (
            "gfx1250 tiled 2D v1 consumes exactly one paged block per iteration "
            f"(tile_size={tile_size}, block_size={block_size})"
        )

    ok, why = check_wmma_arch(arch)
    if not ok:
        return False, why
    return True, "supported by gfx1250 WMMA tiled 2D v1"


def _declare_params(b: IRBuilder):
    output = b.param(
        "output_ptr", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16
    )
    query = b.param(
        "query_ptr", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr",
        PtrType(FP8E4M3, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(FP8E4M3, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(BF16, "global"), readonly=True, align=16)
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
    scale = b.param("scale", F32)
    k_scale = b.param("k_scale", F32)
    v_scale = b.param("v_scale", F32)
    out_scale = b.param("out_scale", F32)
    softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)
    block_table_stride = b.param("block_table_stride", I32)
    qq_bias_stride_0 = b.param("qq_bias_stride_0", I32)
    return locals()


def build_unified_attention_2d_tiled(
    spec: UnifiedAttention2DTiledSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build a gfx1250 WMMA tiled-2D unified-attention kernel."""
    ok, why = supports_tiled_2d(
        head_size=spec.head_size,
        block_size=spec.block_size,
        dtype=spec.dtype,
        num_queries_per_kv=spec.num_queries_per_kv,
        use_alibi=spec.use_alibi,
        use_qq_bias=spec.use_qq_bias,
        use_fp8=True,
        q_dtype=None,
        num_warps=spec.num_warps,
        block_m_per_warp=spec.block_m_per_warp,
        kv_storage_dtype=spec.kv_storage_dtype,
        tile_size=spec.tile_size_eff,
        arch=arch,
    )
    if not ok:
        raise ValueError(why)

    op, a_map, c_map, a_frag, c_frag = resolve_wmma(arch)

    dtype = spec.dtype_ir
    HD = spec.head_size
    BS = spec.block_size
    NQK = spec.num_queries_per_kv
    NUM_QH = spec.num_query_heads
    SLIDING_WINDOW = spec.sliding_window

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    p = _declare_params(b)

    output = p["output"]
    query = p["query"]
    key = p["key"]
    value = p["value"]
    sinks = p["sinks"]
    block_tables = p["block_tables"]
    seq_lens = p["seq_lens"]
    cu_q = p["cu_q"]
    scale = p["scale"]
    k_scale = p["k_scale"]
    v_scale = p["v_scale"]
    num_seqs = p["num_seqs"]
    block_table_stride = p["block_table_stride"]

    kv_head_idx = b.block_id_x()
    q_block_global_idx = b.block_id_y()
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(_WAVE))

    seq_idx = binary_search_seq_idx(
        b,
        cu_q,
        q_block_global_idx,
        num_seqs,
        block_q=_BLOCK_Q,
        iterations=spec.binary_search_iters,
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    cur_batch_q_len = b.sub(cu_q_stop, cu_q_start)
    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(_BLOCK_Q)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    seq_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(seq_len, cur_batch_q_len)

    qb_start_pos = b.mul(q_block_local_idx, b.const_i32(_BLOCK_Q))
    with b.scf_if(b.cmp_ge(qb_start_pos, cur_batch_q_len)):
        b.ret()

    lane_row = b.mod(lane, b.const_i32(16))
    half_k = a_map.coord(b, lane, 0)[1]
    col = b.mod(lane, b.const_i32(16))
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    qk_scale = b.fmul(scale, rcp_ln2)

    q_row_for_a = lane_row
    q_pos_for_a = b.add(qb_start_pos, b.div(q_row_for_a, b.const_i32(NQK)))
    qh_for_a = b.add(
        b.mul(kv_head_idx, b.const_i32(NQK)),
        b.mod(q_row_for_a, b.const_i32(NQK)),
    )
    q_valid_for_a = b.land(
        b.cmp_lt(q_pos_for_a, cur_batch_q_len),
        b.cmp_lt(qh_for_a, b.const_i32(NUM_QH)),
    )
    q_pos_safe = b.select(q_valid_for_a, q_pos_for_a, b.const_i32(0))
    qh_safe = b.select(q_valid_for_a, qh_for_a, b.const_i32(0))

    # Q/O are packed as [total_q, num_query_heads, head_size].
    q_token = b.add(cu_q_start, q_pos_safe)
    q_addr_row_base = b.mul(
        b.add(b.mul(q_token, b.const_i32(NUM_QH)), qh_safe), b.const_i32(HD)
    )
    q_frags = load_q_frags(
        b,
        query,
        q_addr_row_base,
        half_k,
        q_valid_for_a,
        head_size=HD,
        a_frag=a_frag,
        dtype=dtype,
    )

    kv_desc = PagedKvDescriptor(
        block_size=BS,
        stride_0=BS * spec.num_kv_heads * HD,
        stride_1=spec.num_kv_heads * HD,
        stride_2=HD,
        stride_3=1,
    )

    P_lds = (
        None
        if spec.use_register_p
        else b.smem_alloc(dtype, [_BLOCK_M, BS], name_hint="Pgfx1250_uattn")
    )
    # Staging buffer for the V tile (one 32-token paged block x head_size),
    # dequantised fp8->bf16 once per iteration with vectorised loads so the
    # P*V WMMA reads its B fragments from LDS instead of a scattered per-element
    # global gather.
    V_lds = b.smem_alloc(dtype, [BS, HD], name_hint="Vgfx1250_uattn")

    m_inits = []
    for r in range(c_frag):
        row_rel, _ = c_map.coord(b, lane, r)
        qh = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)),
            b.mod(row_rel, b.const_i32(NQK)),
        )
        qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
        if spec.use_sinks:
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            m_inits.append(b.select(qh_in, sink_f, neg_inf))
        else:
            m_inits.append(neg_inf)
    l_inits = [one_f for _ in range(c_frag)]
    acc_inits = [b.zero_vec_f32(c_frag) for _ in range(HD // _WMMA_N)]

    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", m_inits[r]))
        iter_args.append((f"l{r}", l_inits[r]))
    for d in range(HD // _WMMA_N):
        iter_args.append((f"acc{d}", acc_inits[d]))

    bm1_div_nqk = (_BLOCK_M - 1) // NQK
    msp_raw = b.add(b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1))
    max_seq_prefix_len = b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)
    num_tiles = b.div(
        b.add(max_seq_prefix_len, b.const_i32(BS - 1)),
        b.const_i32(BS),
    )
    if SLIDING_WINDOW > 0:
        qpos_hi_raw = b.add(qb_start_pos, b.const_i32(bm1_div_nqk))
        cur_q_minus1 = b.sub(cur_batch_q_len, b.const_i32(1))
        qpos_hi = b.select(
            b.cmp_lt(qpos_hi_raw, cur_q_minus1), qpos_hi_raw, cur_q_minus1
        )
        first_allowed_key = b.add(
            b.sub(b.add(context_len, qb_start_pos), b.const_i32(SLIDING_WINDOW)),
            b.const_i32(1),
        )
        last_allowed_key = b.add(context_len, qpos_hi)
        tile_start_raw = b.div(first_allowed_key, b.const_i32(BS))
        tile_start = b.select(
            b.cmp_lt(tile_start_raw, b.const_i32(0)), b.const_i32(0), tile_start_raw
        )
        tile_end_raw = b.add(b.div(last_allowed_key, b.const_i32(BS)), b.const_i32(1))
        tile_end = b.select(b.cmp_lt(tile_end_raw, num_tiles), tile_end_raw, num_tiles)
    else:
        tile_start = b.const_i32(0)
        tile_end = num_tiles

    kloop = b.scf_for_iter(
        tile_start,
        tile_end,
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="kt",
    )
    with kloop as (kt, state):
        ms = [state[2 * r] for r in range(c_frag)]
        ls = [state[2 * r + 1] for r in range(c_frag)]
        accs = list(state[2 * c_frag :])
        tile_base = b.mul(kt, b.const_i32(BS))
        physical_block = b.global_load_i32(
            block_tables,
            b.add(b.mul(seq_idx, block_table_stride), kt),
        )
        # One paged block per 32-token tile (block_size == 32): the block id is
        # constant across the tile, so reuse the single cached load.
        phys_block = lambda _tok: physical_block  # noqa: E731

        scores = compute_qk_scores(
            b,
            q_frags,
            key,
            kv_desc,
            tile_base=tile_base,
            lane_row=lane_row,
            half_k=half_k,
            kv_head_idx=kv_head_idx,
            block_size=BS,
            head_size=HD,
            kv_dtype=FP8E4M3,
            k_scale=k_scale,
            dtype=dtype,
            c_frag=c_frag,
            phys_block=phys_block,
        )

        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = [[], []]
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            q_pos = b.add(qb_start_pos, b.div(row_rel, b.const_i32(NQK)))
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)),
                b.mod(row_rel, b.const_i32(NQK)),
            )
            row_valid = b.land(
                b.cmp_lt(q_pos, cur_batch_q_len),
                b.cmp_lt(qh, b.const_i32(NUM_QH)),
            )
            srs = []
            for nsub in range(2):
                key_pos = b.add(
                    b.add(tile_base, b.const_i32(nsub * _WMMA_N)),
                    col_k,
                )
                score_log2 = b.fmul(b.vec_extract(scores[nsub], r), qk_scale)
                causal_keep = b.cmp_le(key_pos, b.add(context_len, q_pos))
                in_seq = b.cmp_lt(key_pos, seq_len)
                keep = b.land(row_valid, b.land(in_seq, causal_keep))
                if SLIDING_WINDOW > 0:
                    dist = b.sub(b.add(context_len, q_pos), key_pos)
                    keep = b.land(keep, b.cmp_lt(dist, b.const_i32(SLIDING_WINDOW)))
                srs.append(b.select(keep, score_log2, neg_inf))

            m_new, l_new, alpha, p = softmax_row_update(
                b,
                ms[r],
                ls[r],
                srs,
                neg_inf=neg_inf,
                zero_f=zero_f,
            )
            new_ms.append(m_new)
            new_ls.append(l_new)
            ps[0].append(p[0])
            ps[1].append(p[1])
            for d in range(HD // _WMMA_N):
                old = b.vec_extract(new_accs[d], r)
                new_accs[d] = b.vec_insert(new_accs[d], b.fmul(old, alpha), r)

        if not spec.use_register_p:
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                b.smem_store_vN(
                    P_lds,
                    [row_rel, col_k],
                    b.cast_f32_to(ps[0][r], dtype),
                    1,
                )
                b.smem_store_vN(
                    P_lds,
                    [row_rel, b.add(col_k, b.const_i32(_WMMA_N))],
                    b.cast_f32_to(ps[1][r], dtype),
                    1,
                )
        stage_v_tile(
            b,
            V_lds,
            value,
            kv_desc,
            kv_head_idx=kv_head_idx,
            tile_base=tile_base,
            lane=lane,
            block_size=BS,
            head_size=HD,
            kv_dtype=FP8E4M3,
            v_scale=v_scale,
            dtype=dtype,
            phys_block=phys_block,
        )
        b.sync()

        if spec.use_register_p:
            new_accs = compute_pv_from_probs(
                b,
                ps[0],
                ps[1],
                V_lds,
                new_accs,
                a_map=a_map,
                c_map=c_map,
                lane=lane,
                col=col,
                a_frag=a_frag,
                c_frag=c_frag,
                head_size=HD,
                dtype=dtype,
            )
        else:
            new_accs = compute_pv(
                b,
                P_lds,
                V_lds,
                new_accs,
                a_map=a_map,
                c_map=c_map,
                lane=lane,
                lane_row=lane_row,
                col=col,
                a_frag=a_frag,
                c_frag=c_frag,
                head_size=HD,
                dtype=dtype,
            )

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        b.scf_yield(*yields)

    final = kloop.results
    ls_final = [final[2 * r + 1] for r in range(c_frag)]
    accs_final = list(final[2 * c_frag :])

    for d in range(HD // _WMMA_N):
        for r in range(c_frag):
            row_rel, col_n = c_map.coord(b, lane, r)
            q_pos = b.add(qb_start_pos, b.div(row_rel, b.const_i32(NQK)))
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)),
                b.mod(row_rel, b.const_i32(NQK)),
            )
            out_valid = b.land(
                b.cmp_lt(q_pos, cur_batch_q_len),
                b.cmp_lt(qh, b.const_i32(NUM_QH)),
            )
            l_safe = ls_final[r]
            inv_l = b.select(b.fcmp("oeq", l_safe, zero_f), zero_f, b.rcp(l_safe))
            v_f32 = b.fmul(b.vec_extract(accs_final[d], r), inv_l)
            out_token = b.add(cu_q_start, q_pos)
            o_col = b.add(b.const_i32(d * _WMMA_N), col_n)
            out_addr = b.add(
                b.mul(
                    b.add(b.mul(out_token, b.const_i32(NUM_QH)), qh), b.const_i32(HD)
                ),
                o_col,
            )
            with b.scf_if(out_valid):
                b.global_store(output, out_addr, b.cast_f32_to(v_f32, dtype), align=2)

    b.ret()
    return b.kernel
