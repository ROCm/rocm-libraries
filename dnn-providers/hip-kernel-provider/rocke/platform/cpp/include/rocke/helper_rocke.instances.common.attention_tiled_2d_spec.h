/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.attention_tiled_2d_spec.h -- the C spec
 * struct the gfx942 and gfx950 tiled-2D attention builders share, the C mirror
 * of their UnifiedAttention2DTiledSpec dataclasses. Every header that exposes a
 * tiled-2D entry includes it. It also declares the struct's default
 * constructor and derived @property accessors (defined in
 * instances/common/attention_tiled_2d_spec.cpp).
 *
 *   Python (UnifiedAttention2DTiledSpec) -> C99 (this header)
 *   the dataclass                        -> rocke_attention_tiled_2d_spec_t
 *   its field defaults                   -> rocke_attention_tiled_2d_spec_default
 *   each @property <name>                -> rocke_attention_tiled_2d_spec_<name>
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_SPEC_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_SPEC_H

#include <stdbool.h>

#include "rocke/ir.h" /* rocke_type_t */

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- spec struct *
 *
 * Faithful mirror of UnifiedAttention2DTiledSpec. The first block holds the
 * required (no-default) fields; the rest carry the Python field defaults
 * (materialised by rocke_attention_tiled_2d_spec_default).
 *
 * Optional[int] fields are (has_<x>, <x>): has_<x>==false encodes Python None.
 * Optional[str] fields (kv_storage_dtype) are a ``const char*`` that is NULL
 * when Python is None. ``kv_cache_policy`` is a non-optional string ("stream"
 * default). All strings are caller-owned static/long-lived literals; this
 * struct does not copy them. */
typedef struct rocke_attention_tiled_2d_spec
{
    /* ---- required ---- */
    int head_size;
    int block_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype; /* "fp16" or "bf16" */
    bool use_sinks;
    int sliding_window;
    bool has_softcap;

    /* ---- defaulted ---- */
    bool use_alibi; /* False */
    bool use_qq_bias; /* False */
    int num_seqs; /* 0 */
    int num_warps; /* 1 */

    bool has_waves_per_eu; /* Optional[int] None */
    int waves_per_eu;

    const char* kv_storage_dtype; /* Optional[str] None ("fp8e4m3") */

    bool use_fp8_mfma_qk; /* False */
    bool use_fp8_mfma_pv; /* False */
    bool use_register_pv; /* False */

    /* Deep K+V prefetch / V-double-buffer schedule controls (Python mirror).
     * The gfx950 C emitter currently builds only the depth-2 single/early-V
     * schedule; these fields are carried so the spec is byte-identical with the
     * Python dataclass and the provider/dispatcher can pass them through. A
     * builder that sees use_v_double_buffer / use_staggered_iter_wait / kv_ring
     * _depth!=2 / use_q_reread it does not yet emit must reject in validate(),
     * NOT silently ignore (mirrors the Python __post_init__). */
    bool use_v_double_buffer; /* False */
    int kv_ring_depth; /* 2 */
    bool use_staggered_iter_wait; /* False */
    bool use_q_reread; /* False */
    /* VGPR-frugal BLOCK_M=128 body -- gather the per-lane Q32 MFMA operand
     * straight from global into VGPRs (no Q_lds), freeing 32 KB LDS so
     * BLOCK_M=128/T=64 fits the 32 KB / 2-WG/CU budget with K+V single-buffer.
     * Carried for spec byte-identity with the Python dataclass; the gfx950 C
     * twin does not yet emit this body and rejects it in validate() (mirrors
     * Python __post_init__ -- reject, do not silently ignore). Default False. */
    bool use_q_direct_reg; /* False */
    /* Lever-3 sched_barrier fence after the QK MFMA cluster (CK-Tile-derived).
     * Emitted by the gfx950 C twin (independent of the V schedule). */
    bool use_sched_barrier; /* False */
    int sched_barrier_mask; /* 0 */

    /* softmax-window MFMA interleave (Python gfx950 lever). The default
     * (off) is byte-identical; when on it emits a single iglp_opt at the loop
     * top (mode 0/1) or sched_group_barrier groups (mode 2) to schedule the
     * QK/PV MFMAs into the softmax VALU window. Carried for spec parity with the
     * Python dataclass; the gfx950 C twin rejects it (the body it rides on,
     * use_q_direct_reg, is itself unported). */
    bool use_softmax_mfma_interleave; /* False */
    int softmax_interleave_mode; /* 1 */
    int softmax_interleave_groups; /* 4 */

    bool has_tile_size; /* Optional[int] None */
    int tile_size;

    int block_m_per_warp; /* 16 */
    bool use_mfma_32x32; /* False */
    bool use_mfma_32x32x8; /* False */
    bool use_transposed_qk_32x32; /* False */
    bool use_transposed_scalar_state; /* False */
    bool use_transposed_invariant_hoist; /* False */
    bool use_transposed_mask_once; /* False */
    bool use_transposed_half_local_pv; /* False */
    bool use_mfma32_skip_legacy_qreg; /* False */
    bool use_transposed_mask_limit; /* False */
    bool use_grouped_kv2_softmax; /* False */
    bool use_fast_paged_kv_desc; /* False */
    bool use_i64_kv_addr; /* False */
    bool use_early_v_schedule; /* False */
    bool use_agpr_alloc_zero; /* False */
    bool use_conflict_free_v; /* False */
    bool use_conflict_free_v_store; /* False */
    bool use_conflict_free_v_store_split; /* True  */
    bool use_conflict_free_v_ck_vlds; /* True  */
    bool use_k_single_buffer; /* False */
    bool use_k_sliced_ring; /* False */
    /* Sliced-K ring geometry, in Python dataclass field order. ring_depth is how
     * many LDS slots the ring stages K into; k_slice_hd is the head-dim width of
     * one slice, so k_groups = head_size / k_slice_hd and k_slice_hd is also the
     * K_lds row stride the QK read's LDS bank spread follows. Both are meaningful
     * only when use_k_sliced_ring is set, and both are validated there (mirrors
     * the Python __post_init__). */
    int ring_depth; /* 3  */
    int k_slice_hd; /* 32 */
    bool use_k_sliced_ldsseq; /* False */
    bool use_iglp_opt; /* False */
    bool use_qk_pv_sched_group_barrier; /* False */
    bool use_q_direct_global; /* False */
    /* Read V straight from global memory instead of V_lds. The gfx942 C twin
     * does not emit it yet and rejects it (mirrors the Python field). */
    bool use_v_hbm_direct; /* False */
    /* Read the QK operand K straight from global memory; K_lds shrinks to a
     * one-element stub, so Q is read straight from global memory too. */
    bool use_k_hbm_direct; /* False */
    const char* kv_cache_policy; /* "stream" */
    bool use_global_load_lds_k; /* False */
    bool use_q_major_grid; /* False */
    bool use_qblock_reverse; /* False (gfx950: longest-first query blocks) */
    /* Also write the natural-log softmax normalizer (LSE) as FP32
     * [total_q, num_query_heads] through a trailing lse_ptr param. Off emits
     * no new IR; on appends the "lse" kernel-name token. */
    bool emit_lse; /* False (Python kw_only) */
    /* gfx950 lse_layout == "head_major" (Python kw_only, default "token_major"):
     * each sequence's LSE is a [num_query_heads, q_len] block; the kernel name's
     * "lse" token becomes "lsehm". Needs emit_lse. */
    bool lse_head_major; /* False */
    /* gfx950 dropout on P: the dropout params and
     * the declared extents dropout_max_seqlen_q / _k become the last kernargs,
     * and the kernel name gains "do". The gfx942 twin refuses it. */
    bool dropout; /* False (Python kw_only) */
} rocke_attention_tiled_2d_spec_t;

/* Materialise every defaulted field (the dataclass defaults). The required
 * fields are zero/NULL-initialised; the caller must set them before use. */
rocke_attention_tiled_2d_spec_t rocke_attention_tiled_2d_spec_default(void);

/* ------------------------------------------------ derived @property accessors *
 *
 * Pure, IR-free; faithful ports of the dataclass @property bodies. */

/* num_queries_per_kv = num_query_heads // num_kv_heads */
int rocke_attention_tiled_2d_spec_num_queries_per_kv(const rocke_attention_tiled_2d_spec_t* s);
/* block_m = block_m_per_warp * num_warps */
int rocke_attention_tiled_2d_spec_block_m(const rocke_attention_tiled_2d_spec_t* s);
/* regs_per_lane: 16 for the 32x32 paths, else block_m_per_warp // 4 */
int rocke_attention_tiled_2d_spec_regs_per_lane(const rocke_attention_tiled_2d_spec_t* s);
/* block_q = block_m // num_queries_per_kv */
int rocke_attention_tiled_2d_spec_block_q(const rocke_attention_tiled_2d_spec_t* s);
/* tile_size_eff = tile_size if has_tile_size else block_size */
int rocke_attention_tiled_2d_spec_tile_size_eff(const rocke_attention_tiled_2d_spec_t* s);
/* n_blocks_per_tile = tile_size_eff // block_size */
int rocke_attention_tiled_2d_spec_n_blocks_per_tile(const rocke_attention_tiled_2d_spec_t* s);
/* dtype_ir: F16 for "fp16", BF16 otherwise (returns a rocke_type_t*). */
const rocke_type_t*
    rocke_attention_tiled_2d_spec_dtype_ir(const rocke_attention_tiled_2d_spec_t* s);
/* binary_search_iters: 32 if num_seqs<=0, else max(1, ceil(log2(num_seqs+1))).
 * Exported for existing callers; the kernels derive the search at runtime. */
int rocke_attention_tiled_2d_spec_binary_search_iters(const rocke_attention_tiled_2d_spec_t* s);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_SPEC_H */
