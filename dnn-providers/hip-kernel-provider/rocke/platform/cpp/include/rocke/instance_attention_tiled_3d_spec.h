/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_tiled_3d_spec.h -- the spec PODs of the tiled
 * split-KV 3D attention kernels, shared by the gfx942 and gfx950 ports.
 *
 *   Python (gfx942/gfx950 attention_tiled_3d.py)   C (this header)
 *   --------------------------------------------   -------------------------------------
 *   @dataclass UnifiedAttention3DTiledSpec          rocke_unified_attention_3d_tiled_spec_t
 *                                                     + rocke_unified_attention_3d_tiled_spec_default()
 *   @dataclass UnifiedAttentionReduceTiledSpec      rocke_unified_attention_reduce_tiled_spec_t
 *                                                     + rocke_unified_attention_reduce_tiled_spec_default()
 *
 * Each arch module declares its own dataclasses; the gfx942 segment spec is
 * field-compatible with the gfx950 one so the dispatcher and harnesses build
 * both the same way, and the two reduce specs have the same fields. One C struct
 * per spec therefore serves both ports. The per-arch validators, @property
 * accessors and build entries stay in instance_gfx942_attention_tiled_3d.h and
 * instance_gfx950_attention_tiled_3d.h, because their bodies differ by arch.
 *
 * LIFETIME. Plain by-value PODs the caller owns; the strings (dtype,
 * kv_storage_dtype, kv_cache_policy) are caller-owned long-lived literals, not
 * copied.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_TILED_3D_SPEC_H
#define ROCKE_INSTANCE_ATTENTION_TILED_3D_SPEC_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------- segment-kernel spec *
 *
 * The first block holds the required (no-default) fields; the rest carry the
 * Python field defaults (materialised by
 * rocke_unified_attention_3d_tiled_spec_default).
 *
 * Optional[int] fields are (has_<x>, <x>); has_<x>==false encodes Python None.
 * Optional[str] kv_storage_dtype is a const char* (NULL == Python None,
 * "fp8e4m3" otherwise). dtype is "fp16" or "bf16". */
typedef struct rocke_unified_attention_3d_tiled_spec
{
    /* ---- required (no Python default) ---- */
    int head_size; /* 64 / 128 / 256                                   */
    int block_size; /* 16 / 32 / 64                                     */
    int num_query_heads;
    int num_kv_heads;
    const char* dtype; /* "fp16" or "bf16"                                 */
    bool use_sinks;
    int sliding_window; /* 0 == disabled                                    */
    bool has_softcap;
    int num_segments;

    /* ---- defaulted ---- */
    bool use_alibi; /* False */
    bool use_qq_bias; /* False */
    int num_seqs; /* 0 */

    bool has_waves_per_eu; /* Optional[int] None */
    int waves_per_eu;

    const char* kv_storage_dtype; /* Optional[str] None ("fp8e4m3") */

    bool has_tile_size_override; /* Optional[int] None (KV tile in tokens) */
    int tile_size_override;

    bool use_invariant_hoist; /* False; inert, kept so saved specs still load */
    bool use_wide_kv_load; /* False (gfx942-only knob; ignored on gfx950) */
    /* 64-bit paged-KV addressing for caches > 2 GiB. When set, the per-block
     * byte base (physical_block * block_stride) is folded into a 64-bit buffer
     * base (a wave-uniform per-block make_buffer_rsrc) and only the within-block
     * byte offset stays in the i32 buffer voffset -- mirrors the 2D tiled
     * kernel's use_i64_kv_addr. Default false => byte-identical small-cache
     * build. */
    bool use_i64_kv_addr; /* False */
    /* All-decode launch grid: seq_idx = block_id_x, one query block per
     * sequence, no binary search (legal only when every sequence has one query
     * token; the launcher sizes grid x to num_seqs). */
    bool use_decode_grid; /* False */
    /* Cache policy of the K/V copies ("all", "global", "stream", "nt";
     * rocke_cache_policy_from_name). gfx942's wide K/V loader takes "all" or
     * "stream"; fp8 K/V keeps "all". */
    const char* kv_cache_policy; /* "all" */
    /* K/V LDS rows store their 16-byte chunks XOR-permuted by the row.
     * Implemented for fp16/bf16 K/V, which can be extended. */
    bool use_kv_lds_swizzle; /* False */
    /* The wave loads the segment's page ids 64 at a time, one per lane; tiles
     * read theirs with v_readlane. Paged K/V; implemented for fp16/bf16, which
     * can be extended. */
    bool use_wave_page_id_load; /* False */
    /* gfx950 emit_lse (False, kw_only): the packaged module's reduce writes LSE.
     * The segment IR is unchanged; its kernel name gains a trailing "lse". */
    bool emit_lse;
} rocke_unified_attention_3d_tiled_spec_t;

/* Materialise every defaulted field. Required fields are zero/NULL-init; the
 * caller must set them before use. */
rocke_unified_attention_3d_tiled_spec_t rocke_unified_attention_3d_tiled_spec_default(void);

/* -------------------------------------------------------- reduce-kernel spec */
typedef struct rocke_unified_attention_reduce_tiled_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype; /* "fp16" or "bf16" */
    int num_segments;

    bool has_waves_per_eu; /* Optional[int] None */
    int waves_per_eu;
    /* emit_lse (False): add a trailing lse_ptr param and write the FP32 natural-log
     * LSE [total_q, num_query_heads, 1]; kernel name gains a trailing "lse". */
    bool emit_lse;
} rocke_unified_attention_reduce_tiled_spec_t;

/* Materialise every defaulted field (required fields zero/NULL-init). */
rocke_unified_attention_reduce_tiled_spec_t rocke_unified_attention_reduce_tiled_spec_default(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_TILED_3D_SPEC_H */
