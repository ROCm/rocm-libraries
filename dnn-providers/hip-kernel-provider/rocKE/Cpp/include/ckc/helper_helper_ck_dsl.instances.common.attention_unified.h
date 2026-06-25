/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_helper_ck_dsl.instances.common.attention_unified.h -- C99 port of
 * the scalar-correct reference kernels of the AITER unified-attention
 * dispatcher, ck_dsl/instances/common/attention_unified.py.
 *
 * SCOPE OF THIS PORT (the requested symbol set):
 *
 *   Python (attention_unified.py)            C99 (this header / .c)
 *   --------------------------------------   -----------------------------------
 *   UnifiedAttentionProblem                  ckc_unified_attention_problem_t
 *   build_unified_attention_2d               ckc_build_unified_attention_2d_scalar
 *   UnifiedAttention2DSpec.kernel_name       ckc_unified_attention_2d_scalar_kernel_name
 *   (block_id grid of build_..._2d)          ckc_unified_attention_2d_scalar_grid
 *   _attn_signature(include_bt_stride=False) ckc_unified_attention_2d_scalar_signature
 *   build_unified_attention_3d               ckc_build_unified_attention_3d_scalar
 *   UnifiedAttention3DSpec.kernel_name       ckc_unified_attention_3d_scalar_kernel_name
 *   (block_id grid of build_..._3d)          ckc_unified_attention_3d_scalar_grid
 *   build_unified_attention_reduce           ckc_build_unified_attention_reduce_scalar
 *   UnifiedAttentionReduceSpec.kernel_name   ckc_unified_attention_reduce_scalar_kernel_name
 *   (block_id grid of build_..._reduce)      ckc_unified_attention_reduce_scalar_grid
 *
 * These are deliberately CORRECTNESS kernels (one workgroup of 64 threads, only
 * thread 0 active) that implement the full paged online-softmax semantics for
 * fp16/bf16 without relying on Triton. The optimized MFMA/tiled kernels replace
 * the bodies once parity is locked; the scalar oracle is what this header
 * exposes.
 *
 * Binds to ckc/ir.h (the C IRBuilder) plus the sibling helper headers
 * (spec for kernel_name_join / SignatureBuilder; transforms for the magic
 * division + naive TensorDescriptor offsets).
 *
 * NOTES ON DEPENDENCIES NOT YET IN THE C HELPER LAYER:
 *
 *   - PagedKvDescriptor (helpers/attention.py) is not exported by the C
 *     attention helper; its element-unit paged-KV offset is a fixed
 *     stride formula, so the .c ports it inline (ckc__paged_kv_offset).
 *   - apply_softcap_log2 (helpers/activations.py) is used only on the 2D
 *     softcap>0 path; the .c ports it inline (ckc__apply_softcap).
 *   - binary_search_seq_idx (helpers/attention.py) backs the 3D kernel's
 *     seq-idx scan. The 2D kernel uses an inline linear cu_q scan (ported
 *     faithfully); the 3D kernel's binary-search scan body is bounded but
 *     TODO(port): it falls back to the same inline linear scan (numerically
 *     identical seq_idx) until ckc_binary_search_seq_idx is exported.
 */
#ifndef CKC_HELPER_HELPER_CK_DSL_INSTANCES_COMMON_ATTENTION_UNIFIED_H
#define CKC_HELPER_HELPER_CK_DSL_INSTANCES_COMMON_ATTENTION_UNIFIED_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h" /* ckc_arena_t (signature storage)                    */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t, name join     */
#include "ckc/ir.h" /* ckc_ir_builder_t, ckc_kernel_def_t, status         */

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------- UnifiedAttentionProblem */

/* Python: @dataclass(frozen=True) class UnifiedAttentionProblem.
 *
 * Only the fields the scalar reference kernels (and their name/grid/signature)
 * actually read are carried with semantic meaning; the remaining AITER selector
 * knobs (num_sms, waves_per_eu, compile_backend, ...) are kept for ABI/field
 * parity but unused by the scalar path. `q_dtype` mirrors the Optional[str]
 * (NULL == Python None). The Python @property helpers used by the scalar
 * builders are exposed as free functions below. */
typedef struct ckc_unified_attention_problem
{
    int total_q;
    int num_seqs;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    int block_size;
    int max_seqlen_q;
    int max_seqlen_k;
    const char* dtype; /* "fp16" / "bf16" (scalar path supports these)      */
    const char* q_dtype; /* Optional[str]; NULL == None                        */
    int sliding_window; /* default 0                                          */
    double softcap; /* default 0.0                                        */
    bool use_sinks; /* default false                                      */
    bool use_alibi; /* default false                                      */
    bool use_qq_bias; /* default false                                      */
    bool use_fp8; /* default false                                      */
    int num_sms; /* default 120 (selector only; unused by scalar)      */
    int waves_per_eu; /* Optional[int]; <0 == None                          */
    bool waves_per_eu_set;
    const char* compile_backend; /* Optional[str]; NULL == None               */
    int num_kv_blocks; /* default 0                                  */
} ckc_unified_attention_problem_t;

/* Default-constructed problem (Python field defaults). The required positional
 * fields are zeroed; the caller fills them before use. */
ckc_unified_attention_problem_t ckc_unified_attention_problem_default(void);

/* @property num_queries_per_kv = num_query_heads // num_kv_heads. On a
 * non-divisible ratio (the Python ValueError) returns -1. */
int ckc_unified_attention_problem_num_queries_per_kv(const ckc_unified_attention_problem_t* p);

/* @property all_decode = (max_seqlen_q == 1). */
bool ckc_unified_attention_problem_all_decode(const ckc_unified_attention_problem_t* p);

/* ------------------------------------------------------------- 2D scalar */

/* UnifiedAttention2DSpec.kernel_name(). The default `name` is
 * "ck_dsl_unified_attention_2d_scalar"; pass NULL to use it. Writes the
 * NUL-terminated kernel name into `out` (capacity `out_cap`). */
ckc_status_t ckc_unified_attention_2d_scalar_kernel_name(const ckc_unified_attention_problem_t* p,
                                                         const char* name,
                                                         char* out,
                                                         size_t out_cap);

/* build_unified_attention_2d(spec). Emits the scalar 2D unified-attention
 * kernel into `b` and returns its kernel def (== b->kernel). `name` NULL uses
 * the default spec name. On an unsupported dtype or any IR error returns NULL
 * with the builder's sticky error set. */
ckc_kernel_def_t* ckc_build_unified_attention_2d_scalar(ckc_ir_builder_t* b,
                                                        const ckc_unified_attention_problem_t* p,
                                                        const char* name);

/* Launch grid of the 2D scalar kernel = (block_id_x, block_id_y, block_id_z)
 * extents = (total_q, num_query_heads, head_size). Writes out[0..2]. */
void ckc_unified_attention_2d_scalar_grid(const ckc_unified_attention_problem_t* p, int out[3]);

/* _attn_signature(dtype, include_bt_stride=False): the 2D scalar kernel ABI.
 * On CKC_OK *out_items / *out_count hold the arena-owned array. */
ckc_status_t ckc_unified_attention_2d_scalar_signature(const ckc_unified_attention_problem_t* p,
                                                       ckc_arena_t* arena,
                                                       const ckc_sig_entry_t** out_items,
                                                       size_t* out_count);

/* ------------------------------------------------------------- 3D scalar */

/* UnifiedAttention3DSpec.kernel_name(). `name` NULL uses the default
 * "ck_dsl_unified_attention_3d_scalar"; `num_segments` is the seg<N> tag. */
ckc_status_t ckc_unified_attention_3d_scalar_kernel_name(const ckc_unified_attention_problem_t* p,
                                                         const char* name,
                                                         int num_segments,
                                                         char* out,
                                                         size_t out_cap);

/* build_unified_attention_3d(spec). `num_segments` defaults to 8 in Python;
 * pass the desired value. Returns the kernel def or NULL on error. */
ckc_kernel_def_t* ckc_build_unified_attention_3d_scalar(ckc_ir_builder_t* b,
                                                        const ckc_unified_attention_problem_t* p,
                                                        const char* name,
                                                        int num_segments);

/* Launch grid of the 3D scalar kernel = (total_q, num_query_heads,
 * num_segments * head_size) -- block_id_z is split into (segm_idx, dim). */
void ckc_unified_attention_3d_scalar_grid(const ckc_unified_attention_problem_t* p,
                                          int num_segments,
                                          int out[3]);

/* ----------------------------------------------------------- reduce scalar */

/* UnifiedAttentionReduceSpec.kernel_name(). `name` NULL uses the default
 * "ck_dsl_unified_attention_reduce_scalar". */
ckc_status_t
    ckc_unified_attention_reduce_scalar_kernel_name(const ckc_unified_attention_problem_t* p,
                                                    const char* name,
                                                    int num_segments,
                                                    char* out,
                                                    size_t out_cap);

/* build_unified_attention_reduce(spec). Returns the kernel def or NULL. */
ckc_kernel_def_t*
    ckc_build_unified_attention_reduce_scalar(ckc_ir_builder_t* b,
                                              const ckc_unified_attention_problem_t* p,
                                              int num_segments,
                                              const char* name);

/* Launch grid of the reduce scalar kernel = (total_q, num_query_heads,
 * head_size). Writes out[0..2]. */
void ckc_unified_attention_reduce_scalar_grid(const ckc_unified_attention_problem_t* p, int out[3]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_HELPER_CK_DSL_INSTANCES_COMMON_ATTENTION_UNIFIED_H */
