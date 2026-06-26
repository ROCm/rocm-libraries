/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_wmma_attention_fwd.h -- C99 port of the dense WMMA FMHA
 * forward for gfx1250 rocke/instances/gfx1250/wmma_attention_fwd.py.
 *
 *   Python (gfx1250/wmma_attention_fwd.py)  C99 (this header)
 *   --------------------------------------  ----------------------------------------
 *   class WmmaAttentionFwdSpec              rocke_wmma_attention_fwd_gfx1250_spec_t
 *   WmmaAttentionFwdSpec.kv_heads           rocke_wmma_attention_fwd_gfx1250_kv_heads()
 *   WmmaAttentionFwdSpec.block_size         rocke_wmma_attention_fwd_gfx1250_block_size()
 *   WmmaAttentionFwdSpec.kernel_name()      rocke_wmma_attention_fwd_gfx1250_kernel_name()
 *   is_valid_spec(spec, arch)               rocke_wmma_attention_fwd_gfx1250_is_valid_spec()
 *   build_wmma_attention_fwd(spec, arch)    rocke_build_wmma_attention_fwd_gfx1250()
 *   wmma_attention_fwd_grid(spec,...)       rocke_wmma_attention_fwd_gfx1250_grid()
 *
 * Standalone gfx1250 K=32 WMMA FMHA: one wave32 per (q_tile, head, batch),
 * BLOCK_M=16 Q rows, BLOCK_K=32 K positions per K-loop iter, online softmax over
 * 32 k columns, PV via one K=32 WMMA per head_size/16 d-tile. head_size must be a
 * multiple of 32. No async DMA.
 */
#ifndef ROCKE_INSTANCE_GFX1250_WMMA_ATTENTION_FWD_H
#define ROCKE_INSTANCE_GFX1250_WMMA_ATTENTION_FWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python WmmaAttentionFwdSpec (frozen dataclass):
 *     head_size: int          (required)
 *     num_query_heads: int    (required)
 *     num_kv_heads: int = 0   # 0 -> equal to num_query_heads (MHA)
 *     dtype: str = "fp16"     # fp16 only
 *     mask_mode: str = "none" # "none" | "causal"
 *     sliding_window: int = 0
 *     name: str = "rocke_wmma_attention_fwd_gfx1250" */
typedef struct rocke_wmma_attention_fwd_gfx1250_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype;
    const char* mask_mode;
    int sliding_window;
    const char* name;
} rocke_wmma_attention_fwd_gfx1250_spec_t;

/* Default-constructed spec. head_size/num_query_heads have NO Python default. */
rocke_wmma_attention_fwd_gfx1250_spec_t rocke_wmma_attention_fwd_gfx1250_spec_default(void);

/* kv_heads property: num_kv_heads or num_query_heads. */
int rocke_wmma_attention_fwd_gfx1250_kv_heads(const rocke_wmma_attention_fwd_gfx1250_spec_t* spec);

/* block_size property: 32 (one wave32 per block). */
int rocke_wmma_attention_fwd_gfx1250_block_size(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec);

/* kernel_name():
 *   kernel_name_join(name, "wmma16x16x32", f"H{hs}", f"HQ{hq}", f"HK{kvh}",
 *                    "fp16", mask_mode). */
rocke_status_t rocke_wmma_attention_fwd_gfx1250_kernel_name(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, char* out, size_t out_cap);

bool rocke_wmma_attention_fwd_gfx1250_is_valid_spec(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* build_wmma_attention_fwd(spec, arch). Signature:
 *   (Q,K,V: ptr<f16>, O: ptr<f16>, scale_log2: f32, seqlen_q, seqlen_k: i32,
 *    stride_{q,k,v,o}_{token,head}: i32). */
rocke_kernel_def_t* rocke_build_wmma_attention_fwd_gfx1250(
    rocke_ir_builder_t* b, const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_wmma_attention_fwd_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, const char* arch);

/* wmma_attention_fwd_grid(spec, seqlen_q, batch) -> (seqlen_q//16, hq, batch). */
rocke_status_t rocke_wmma_attention_fwd_gfx1250_grid(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, int seqlen_q, int batch, int out[3]);

rocke_status_t rocke_wmma_attention_fwd_gfx1250_lower_to_llvm(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_WMMA_ATTENTION_FWD_H */
