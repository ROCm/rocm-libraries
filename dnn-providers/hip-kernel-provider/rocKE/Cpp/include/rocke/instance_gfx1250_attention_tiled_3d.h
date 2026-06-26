/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_attention_tiled_3d.h -- C99 port of the gfx1250 WMMA
 * split-KV 3D decode attention (segment + reduce) kernels:
 *   Python/rocke/instances/gfx1250/attention_tiled_3d.py
 *
 *   Python (gfx1250/attention_tiled_3d.py)                 C99 (this header)
 *   ---------------------------------------------------    --------------------------------
 *   class UnifiedAttention3DTiledSpec                      rocke_uattn3d_seg_gfx1250_spec_t
 *   UnifiedAttention3DTiledSpec.kernel_name()              rocke_uattn3d_seg_gfx1250_kernel_name()
 *   supports_tiled_3d(...)                                 rocke_uattn3d_seg_gfx1250_supports()
 *   build_unified_attention_3d_tiled(spec,arch)            rocke_build_uattn3d_seg_gfx1250()
 *
 *   class UnifiedAttentionReduceTiledSpec                  rocke_uattn3d_reduce_gfx1250_spec_t
 *   UnifiedAttentionReduceTiledSpec.kernel_name()          rocke_uattn3d_reduce_gfx1250_kernel_name()
 *   build_unified_attention_reduce_tiled(spec,arch)        rocke_build_uattn3d_reduce_gfx1250()
 */
#ifndef ROCKE_INSTANCE_GFX1250_ATTENTION_TILED_3D_H
#define ROCKE_INSTANCE_GFX1250_ATTENTION_TILED_3D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Segment spec (UnifiedAttention3DTiledSpec) ========== */
typedef struct rocke_uattn3d_seg_gfx1250_spec
{
    /* Required (no Python default): */
    int head_size;
    int block_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype;
    bool use_sinks;
    int sliding_window;
    bool has_softcap;
    int num_segments;
    /* Defaulted: */
    bool use_alibi; /* False */
    bool use_qq_bias; /* False */
    int num_seqs; /* 0 */
    bool waves_per_eu_set; /* Optional[int] None => false */
    int waves_per_eu;
    const char* kv_storage_dtype; /* NULL => "bf16" */
    bool tile_size_override_set; /* Optional[int] None => false */
    int tile_size_override;
    bool use_invariant_hoist; /* False */
    bool use_wide_kv_load; /* False */
    bool use_register_p; /* False */
    int wmma_spacing; /* 0 */
    int num_waves; /* 1 */
    bool use_wide_lds_reads; /* True */
    bool use_dtla_prefetch; /* False */
    bool use_ds_tr_reads; /* False */
    bool use_sw_pipeline; /* False */
    bool ablate_softmax; /* False */
    bool ablate_pv; /* False */
    bool use_fused_reduce; /* False */
    bool use_dpp_softmax; /* True */
} rocke_uattn3d_seg_gfx1250_spec_t;

rocke_uattn3d_seg_gfx1250_spec_t rocke_uattn3d_seg_gfx1250_spec_default(void);

int rocke_uattn3d_seg_gfx1250_num_queries_per_kv(const rocke_uattn3d_seg_gfx1250_spec_t* spec);
int rocke_uattn3d_seg_gfx1250_block_q(const rocke_uattn3d_seg_gfx1250_spec_t* spec);
int rocke_uattn3d_seg_gfx1250_binary_search_iters(const rocke_uattn3d_seg_gfx1250_spec_t* spec);

rocke_status_t rocke_uattn3d_seg_gfx1250_kernel_name(const rocke_uattn3d_seg_gfx1250_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap);

bool rocke_uattn3d_seg_gfx1250_supports(const rocke_uattn3d_seg_gfx1250_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

rocke_kernel_def_t* rocke_build_uattn3d_seg_gfx1250(rocke_ir_builder_t* b,
                                                    const rocke_uattn3d_seg_gfx1250_spec_t* spec,
                                                    const char* arch);
rocke_kernel_def_t* rocke_build_uattn3d_seg_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_uattn3d_seg_gfx1250_spec_t* spec, const char* arch);

rocke_status_t rocke_uattn3d_seg_gfx1250_lower_to_llvm(const rocke_uattn3d_seg_gfx1250_spec_t* spec,
                                                       const char* arch,
                                                       rocke_llvm_flavor_t flavor,
                                                       char** out_ll,
                                                       char* err,
                                                       size_t err_cap);

/* ========== Reduce spec (UnifiedAttentionReduceTiledSpec) ========== */
typedef struct rocke_uattn3d_reduce_gfx1250_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype;
    int num_segments;
    bool waves_per_eu_set;
    int waves_per_eu;
} rocke_uattn3d_reduce_gfx1250_spec_t;

rocke_uattn3d_reduce_gfx1250_spec_t rocke_uattn3d_reduce_gfx1250_spec_default(void);

rocke_status_t rocke_uattn3d_reduce_gfx1250_kernel_name(
    const rocke_uattn3d_reduce_gfx1250_spec_t* spec, char* out, size_t out_cap);

rocke_kernel_def_t* rocke_build_uattn3d_reduce_gfx1250(
    rocke_ir_builder_t* b, const rocke_uattn3d_reduce_gfx1250_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_uattn3d_reduce_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_uattn3d_reduce_gfx1250_spec_t* spec, const char* arch);

rocke_status_t
    rocke_uattn3d_reduce_gfx1250_lower_to_llvm(const rocke_uattn3d_reduce_gfx1250_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_ATTENTION_TILED_3D_H */
