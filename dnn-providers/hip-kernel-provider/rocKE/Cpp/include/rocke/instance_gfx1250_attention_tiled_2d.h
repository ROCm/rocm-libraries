/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_attention_tiled_2d.h -- C99 port of the gfx1250 WMMA
 * tiled-2D unified-attention forward
 * rocke/instances/gfx1250/attention_tiled_2d.py.
 *
 *   Python (gfx1250/attention_tiled_2d.py)       C99 (this header)
 *   ------------------------------------------   ------------------------------------------
 *   class UnifiedAttention2DTiledSpec            rocke_uattn2d_tiled_gfx1250_spec_t
 *   UnifiedAttention2DTiledSpec.kernel_name()    rocke_uattn2d_tiled_gfx1250_kernel_name()
 *   supports_tiled_2d(...)                       rocke_uattn2d_tiled_gfx1250_supports()
 *   build_unified_attention_2d_tiled(spec,arch)  rocke_build_uattn2d_tiled_gfx1250()
 *
 * One wave32 CTA per (kv_head, q_block); BLOCK_M=16 rows, BLOCK_Q=2 (GQA-8);
 * one paged-KV block per iter (T=block_size=32); Q/O bf16, K/V fp8e4m3 paged
 * dequantized to bf16 before wmma_gfx1250_f32_16x16x32_bf16. Feature slice:
 * head_size=64, block_size=32, GQA-8, sinks + sliding-window. gfx1250-ONLY.
 */
#ifndef ROCKE_INSTANCE_GFX1250_ATTENTION_TILED_2D_H
#define ROCKE_INSTANCE_GFX1250_ATTENTION_TILED_2D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python UnifiedAttention2DTiledSpec (frozen dataclass). Required
 * fields (no Python default): head_size, block_size, num_query_heads,
 * num_kv_heads, dtype, use_sinks, sliding_window, has_softcap. */
typedef struct rocke_uattn2d_tiled_gfx1250_spec
{
    int head_size;
    int block_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype;
    bool use_sinks;
    int sliding_window;
    bool has_softcap;
    bool use_alibi; /* False */
    bool use_qq_bias; /* False */
    int num_seqs; /* 0 */
    int num_warps; /* 1 */
    bool waves_per_eu_set; /* Optional[int] None => false */
    int waves_per_eu; /* used only if waves_per_eu_set */
    const char* kv_storage_dtype; /* "fp8e4m3" */
    bool tile_size_set; /* Optional[int] None => false */
    int tile_size; /* used only if tile_size_set */
    int block_m_per_warp; /* 16 */
    bool use_register_p; /* False */
} rocke_uattn2d_tiled_gfx1250_spec_t;

/* Default-constructed spec: optional/defaulted fields set to their Python
 * dataclass defaults; the required fields are left zeroed for the caller. */
rocke_uattn2d_tiled_gfx1250_spec_t rocke_uattn2d_tiled_gfx1250_spec_default(void);

/* num_queries_per_kv = num_query_heads / num_kv_heads. */
int rocke_uattn2d_tiled_gfx1250_num_queries_per_kv(const rocke_uattn2d_tiled_gfx1250_spec_t* spec);

/* tile_size_eff = tile_size if tile_size_set else block_size. */
int rocke_uattn2d_tiled_gfx1250_tile_size_eff(const rocke_uattn2d_tiled_gfx1250_spec_t* spec);

/* binary_search_iters: 32 if num_seqs<=0 else max(1, ceil(log2(num_seqs+1))). */
int rocke_uattn2d_tiled_gfx1250_binary_search_iters(const rocke_uattn2d_tiled_gfx1250_spec_t* spec);

/* kernel_name(). */
rocke_status_t rocke_uattn2d_tiled_gfx1250_kernel_name(
    const rocke_uattn2d_tiled_gfx1250_spec_t* spec, char* out, size_t out_cap);

/* supports_tiled_2d gate (+ has_softcap / sliding_window post-init checks). */
bool rocke_uattn2d_tiled_gfx1250_supports(const rocke_uattn2d_tiled_gfx1250_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* build_unified_attention_2d_tiled(spec, arch). */
rocke_kernel_def_t* rocke_build_uattn2d_tiled_gfx1250(
    rocke_ir_builder_t* b, const rocke_uattn2d_tiled_gfx1250_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_uattn2d_tiled_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_uattn2d_tiled_gfx1250_spec_t* spec, const char* arch);

rocke_status_t
    rocke_uattn2d_tiled_gfx1250_lower_to_llvm(const rocke_uattn2d_tiled_gfx1250_spec_t* spec,
                                              const char* arch,
                                              rocke_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_ATTENTION_TILED_2D_H */
