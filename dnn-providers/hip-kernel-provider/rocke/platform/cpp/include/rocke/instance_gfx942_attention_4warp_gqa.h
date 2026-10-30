/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx942_attention_4warp_gqa.h -- PUBLIC C99 surface for the gfx942
 * 4-warp GQA natural-QK (S = Q @ K^T) paged-attention builder, the route for the
 * gfx942 D256 causal bf16 cohort and the D128 sliding-window bf16/fp16 cohort.
 *
 *   Python                                          C99 (this header)
 *   ---------------------------------------------   ---------------------------------------------
 *   gfx942/attention_tiled_2d.py
 *     build_gfx942_4warp_gqa(spec, arch)          -> rocke_build_gfx942_4warp_gqa
 *       (_build_gfx942_4warp_gqa_lean, D256,         (static in the .cpp, as is
 *        _online_softmax_rescale)                     _online_softmax_rescale)
 *     supports_tiled_2d_spec(spec, arch)          -> rocke_gfx942_attention_tiled_2d_supports_spec
 *     supports_gfx942_4warp_gqa_spec(spec, arch)  -> rocke_gfx942_attention_4warp_gqa_supports_spec
 *   common/attention_unified.py
 *     gfx942_gqa_fold_eligible(...)               -> rocke_gfx942_gqa_fold_eligible
 *     gfx942_4warp_launch_grid(problem)           -> rocke_gfx942_4warp_launch_grid
 *   (convenience: build -> lower .ll)             -> rocke_gfx942_attention_4warp_gqa_lower_to_llvm
 *
 * The builder takes the shared tiled-2D spec (rocke_attention_tiled_2d_spec_t).
 * It reads head_size, block_size, num_query_heads, num_kv_heads, dtype,
 * sliding_window, waves_per_eu, use_i64_kv_addr and the binary-search trip
 * count; the kernel name is spec.kernel_name() plus "_4wgqa" or, for the GQA
 * head-fold, "_4wgqa_fold".
 *
 * Error model: the builder routes Python's raises through b's sticky error
 * (NotImplementedError -> ROCKE_ERR_NOTIMPL, ValueError and the LDS-capacity
 * assert -> ROCKE_ERR_VALUE) with Python's message. The supports wrappers
 * return a bool plus Python's reason string. The convenience lower returns a
 * rocke_status_t. Every IR node is arena-owned by the builder.
 */
#ifndef ROCKE_INSTANCE_GFX942_ATTENTION_4WARP_GQA_H
#define ROCKE_INSTANCE_GFX942_ATTENTION_4WARP_GQA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_helper_rocke.instances.common.attention_unified.h" /* problem struct */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h" /* spec struct */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* gfx942_gqa_fold_eligible(head_size, num_queries_per_kv, sliding_window, dtype,
 * block_size): the GQA head-fold cohort (D128, 4:1 GQA, sliding window, bf16,
 * block_size <= 32). The builder and the launch grid both read it, so the kernel
 * and its grid cannot disagree. ``dtype`` NULL is never eligible. */
bool rocke_gfx942_gqa_fold_eligible(
    int head_size, int num_queries_per_kv, int sliding_window, const char* dtype, int block_size);

/* gfx942_4warp_launch_grid(problem): the grid the 4-warp builder launches.
 * Folded: (num_kv_heads, total_q / 32 + num_seqs, 1); otherwise
 * (num_query_heads, total_q / 128 + num_seqs, 1). Returns false (grid untouched)
 * on a NULL argument or when num_query_heads is not a multiple of num_kv_heads
 * (Python's ValueError from problem.num_queries_per_kv). */
bool rocke_gfx942_4warp_launch_grid(const rocke_unified_attention_problem_t* p, int grid[3]);

/* supports_tiled_2d_spec(spec, arch) / supports_gfx942_4warp_gqa_spec(spec, arch):
 * rocke_gfx942_attention_tiled_2d_supports over the fields read off the spec,
 * with use_d256_fast false / true. ``arch`` NULL == "gfx942". ``reason`` may be
 * NULL. A NULL spec is rejected with reason "null spec". */
bool rocke_gfx942_attention_tiled_2d_supports_spec(const rocke_attention_tiled_2d_spec_t* spec,
                                                   const char* arch,
                                                   char* reason,
                                                   size_t reason_cap);
bool rocke_gfx942_attention_4warp_gqa_supports_spec(const rocke_attention_tiled_2d_spec_t* spec,
                                                    const char* arch,
                                                    char* reason,
                                                    size_t reason_cap);

/* build_gfx942_4warp_gqa(spec, arch="gfx942").
 *
 * Unlike rocke_build_unified_attention_2d_tiled_scalar, this INITIALIZES ``b``
 * itself (the name suffix depends on the fold decision). The caller frees ``b``
 * with rocke_ir_builder_free() whatever the outcome, except when ``b`` or
 * ``spec`` is NULL (then ``b`` is untouched). Returns the kernel, or NULL with
 * b's sticky error set. ``arch`` NULL == "gfx942". */
rocke_kernel_def_t* rocke_build_gfx942_4warp_gqa(rocke_ir_builder_t* b,
                                                 const rocke_attention_tiled_2d_spec_t* spec,
                                                 const char* arch);

/* Convenience: build with an internally owned builder, then lower to LLVM .ll
 * text. ``arch`` NULL == "gfx942". On ROCKE_OK *out_ll receives a malloc'd
 * string the caller frees with free(); on failure it stays NULL and, when
 * ``err`` is non-NULL, a diagnostic is written (capacity err_cap). */
rocke_status_t
    rocke_gfx942_attention_4warp_gqa_lower_to_llvm(const rocke_attention_tiled_2d_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX942_ATTENTION_4WARP_GQA_H */
