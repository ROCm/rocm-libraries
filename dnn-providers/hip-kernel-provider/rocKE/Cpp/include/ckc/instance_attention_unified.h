/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_attention_unified.h -- C99 port of the SCALAR-correct reference
 * kernel builders of the AITER unified-attention dispatcher
 * (ck_dsl/instances/common/attention_unified.py, 3270 LOC).
 *
 * SCOPE. This header is the PUBLIC entry/glue surface for the three scalar
 * reference kernel builders that the Python module exposes as a fallback path
 * (the optimized MFMA/tiled 2D and 3D kernels live in arch packages and are
 * dispatched lazily by run_unified_attention_torch; they are NOT part of this
 * port). The host-side selectors, descriptors, and IR emit helpers used by
 * these builders are ported in the two sibling helper headers:
 *
 *   ckc/helper_helper_ck_dsl.instances.common.attention_unified.h
 *     (UnifiedAttentionProblem struct + the public build/name/grid/signature
 *      entry points whose symbol set the port map enumerates)
 *   ckc/helper_helper_ck_dsl.instances.common.attention_unified_selectors.h
 *     (_select_2d_*, _kv_storage_dtype, _magic_div(_mod), _q_descriptor,
 *      _paged_kv_descriptor, _segm_descriptors, _emit_qk_score, _emit_v_load,
 *      _physical_block_and_token)
 *
 * THIS header re-exports the public problem struct + build entries (so callers
 * include exactly one header per instance, matching instance_*.h convention)
 * and adds a build->lower-to-.ll convenience per kernel that the helper headers
 * do not provide.
 *
 *   Python (attention_unified.py)                C99 (this header)
 *   ------------------------------------------   --------------------------------
 *   @dataclass UnifiedAttentionProblem           ckc_unified_attention_problem_t
 *   @dataclass UnifiedAttention2DSpec            ckc_unified_attention_2d_spec_t
 *   @dataclass UnifiedAttention3DSpec            ckc_unified_attention_3d_spec_t
 *   @dataclass UnifiedAttentionReduceSpec        ckc_unified_attention_reduce_spec_t
 *   build_unified_attention_2d(spec)             ckc_build_unified_attention_2d_scalar
 *   build_unified_attention_3d(spec)             ckc_build_unified_attention_3d_scalar
 *   build_unified_attention_reduce(spec)         ckc_build_unified_attention_reduce_scalar
 *   *.kernel_name()                              ckc_unified_attention_*_kernel_name
 *   (block_id grid extents)                      ckc_unified_attention_*_grid
 *   _attn_signature(include_bt_stride=False)     ckc_unified_attention_2d_scalar_signature
 *   supports_native_unified_attention(problem)   ckc_unified_attention_supports_scalar
 *   (build -> lower .ll convenience, NEW)        ckc_unified_attention_*_lower_to_llvm
 *
 * The scalar kernels are deliberately CORRECTNESS kernels: one workgroup of 64
 * threads with only thread 0 active, implementing the full paged online-softmax
 * semantics for fp16/bf16 (no Triton). 2D computes one output element
 * (query_token, query_head, dim) per CTA. 3D is the split-KV segment kernel
 * writing per-segment (acc, max, expsum); reduce combines the segments.
 *
 * ERROR MODEL. Mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (ckc_b_*); Python ValueError (unsupported dtype, etc.)
 * becomes a set sticky error + NULL return. The lower convenience returns a
 * ckc_status_t and writes a diagnostic into the caller's err buffer.
 *
 * LIFETIME. Every emitted IR node is arena-owned (ckc_ir_builder_t.arena);
 * nothing is freed individually. Signature entries are owned by the caller's
 * ckc_arena_t. The _new convenience builders own/init the supplied builder; the
 * caller frees it with ckc_ir_builder_free(). The lower convenience owns and
 * frees its IRBuilder internally and mallocs *out_ll for the caller to free().
 */
#ifndef CKC_INSTANCE_ATTENTION_UNIFIED_H
#define CKC_INSTANCE_ATTENTION_UNIFIED_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h"
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t */
/* The public problem struct + base build/name/grid/signature entry points are
 * authored in the helper header; re-export them so callers include one header. */
#include "ckc/helper_helper_ck_dsl.instances.common.attention_unified.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * Default spec names (Python dataclass `name` field defaults)
 * ============================================================ */
#define CKC_UNIFIED_ATTN_2D_SCALAR_NAME "ck_dsl_unified_attention_2d_scalar"
#define CKC_UNIFIED_ATTN_3D_SCALAR_NAME "ck_dsl_unified_attention_3d_scalar"
#define CKC_UNIFIED_ATTN_REDUCE_SCALAR_NAME "ck_dsl_unified_attention_reduce_scalar"
/* Python UnifiedAttention3DSpec.num_segments default. */
#define CKC_UNIFIED_ATTN_3D_DEFAULT_NUM_SEGMENTS 8
/* rcp(ln2) constant the scalar bodies fold into the score (exp2 domain). */
#define CKC_UNIFIED_ATTN_RCP_LN2 1.4426950408889634

/* ============================================================ *
 * Coverage gate -- supports_native_unified_attention(problem)
 * ============================================================ *
 *
 * Returns whether the SCALAR 2D backend can run this problem (head_size in
 * {64,128,256}, block_size in {16,32,64}, dtype in {fp16,bf16}, no fp8/alibi/
 * qq_bias). On false, *out_reason (if non-NULL) points to a static string
 * mirroring the Python reason. */
bool ckc_unified_attention_supports_scalar(const ckc_unified_attention_problem_t* p,
                                           const char** out_reason);

/* ============================================================ *
 * Spec value types (Python @dataclass(frozen=True))
 * ============================================================ *
 *
 * The Python specs are thin wrappers over a UnifiedAttentionProblem plus a
 * name (and num_segments for 3D/reduce). The build entries already accept the
 * problem + name + num_segments directly (matching the helper-header symbol
 * set), so these structs are provided for callers that prefer to model the
 * Python dataclass shape and pass it to the _spec convenience overloads. */
typedef struct ckc_unified_attention_2d_spec
{
    ckc_unified_attention_problem_t problem;
    const char* name; /* NULL => CKC_UNIFIED_ATTN_2D_SCALAR_NAME */
} ckc_unified_attention_2d_spec_t;

typedef struct ckc_unified_attention_3d_spec
{
    ckc_unified_attention_problem_t problem;
    const char* name; /* NULL => CKC_UNIFIED_ATTN_3D_SCALAR_NAME */
    int num_segments; /* Python default 8 */
} ckc_unified_attention_3d_spec_t;

typedef struct ckc_unified_attention_reduce_spec
{
    ckc_unified_attention_problem_t problem;
    const char* name; /* NULL => CKC_UNIFIED_ATTN_REDUCE_SCALAR_NAME */
    int num_segments; /* required (no Python default) */
} ckc_unified_attention_reduce_spec_t;

/* ============================================================ *
 * Build + lower convenience (per kernel)
 * ============================================================ *
 *
 * The base build/name/grid/signature entries are declared in the helper header
 * (ckc_build_unified_attention_2d_scalar etc). The following ADD the
 * builder-owning + lower-to-.ll convenience the helper header omits.
 *
 * _new: init `b` via the resolved kernel_name, then build. The caller owns `b`
 *       and frees it with ckc_ir_builder_free(). Returns the kernel or NULL.
 * _lower_to_llvm: build into an internally-owned builder, then lower to LLVM
 *       .ll text. On CKC_OK *out_ll is a malloc'd NUL-terminated string the
 *       caller frees with free(); on failure *out_ll is left NULL and (if
 *       err != NULL, capacity err_cap) a diagnostic is written. */

ckc_kernel_def_t* ckc_build_unified_attention_2d_scalar_new(
    ckc_ir_builder_t* b, const ckc_unified_attention_problem_t* p, const char* name);
ckc_status_t ckc_unified_attention_2d_scalar_lower_to_llvm(const ckc_unified_attention_problem_t* p,
                                                           const char* name,
                                                           ckc_llvm_flavor_t flavor,
                                                           char** out_ll,
                                                           char* err,
                                                           size_t err_cap);

ckc_kernel_def_t*
ckc_build_unified_attention_3d_scalar_new(ckc_ir_builder_t* b,
                                          const ckc_unified_attention_problem_t* p,
                                          const char* name,
                                          int num_segments);
ckc_status_t ckc_unified_attention_3d_scalar_lower_to_llvm(const ckc_unified_attention_problem_t* p,
                                                           const char* name,
                                                           int num_segments,
                                                           ckc_llvm_flavor_t flavor,
                                                           char** out_ll,
                                                           char* err,
                                                           size_t err_cap);

ckc_kernel_def_t*
ckc_build_unified_attention_reduce_scalar_new(ckc_ir_builder_t* b,
                                              const ckc_unified_attention_problem_t* p,
                                              int num_segments,
                                              const char* name);
ckc_status_t
ckc_unified_attention_reduce_scalar_lower_to_llvm(const ckc_unified_attention_problem_t* p,
                                                  int num_segments,
                                                  const char* name,
                                                  ckc_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_ATTENTION_UNIFIED_H */
