/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common._fmha_warp_body.h -- C99 port of the
 * warp-distributed FMHA forward inner-body from
 * ck_dsl/instances/common/_fmha_warp_body.py.
 *
 *   Python (_fmha_warp_body.py)            C99 (this header)
 *   ------------------------------------   ------------------------------------
 *   WARP_SIZE = 64                         CKC_FMHA_WARP_SIZE / ckc_fmha_warp_size()
 *   fmha_warp_fwd_inner_body(b, *, ...)    ckc_fmha_warp_fwd_inner_body(b, &opts)
 *
 * Only WARP_SIZE and fmha_warp_fwd_inner_body are ported here (the file's
 * module __all__). The body is a pure IR emitter: it threads the
 * online-softmax (m, l) scalars and the per-lane PV accumulator through an
 * scf.for K-loop as iter_args and writes O. It does NOT touch arch / specs.
 *
 * Closures -> function pointers + opaque `user`.
 * ----------------------------------------------
 * The Python body takes several optional Python callables. Each becomes a C
 * function-pointer typedef whose first arg is the builder and whose last arg is
 * the opaque `user` environment pointer (the convention used by the other
 * ported helpers, e.g. mfma_gemm_inner). A NULL pointer reproduces the Python
 * `is None` default. The two loaders that return Python *lists* of `ept` f32
 * Values write into a caller-supplied out-array of length `ept` instead.
 *
 *   extra_score_transform : (b, score_log2, k_idx) -> score_log2
 *   extra_mask_predicate  : (b, k_idx) -> i1
 *   k_row_base_fn         : (b, k_idx) -> i32
 *   v_row_base_fn         : (b, k_idx) -> i32
 *   kv_lane_loader        : (b, k_idx, k_row_base, v_row_base, lane_d_base, ept)
 *                              -> (k_f32_list, v_f32_list)   [two lists of ept]
 *   q_lane_loader         : (b, q_row_base, lane_d_base, ept) -> q_f32_list
 *
 * Peers (NOT ported here, resolved at link time):
 *   ckc_b_io_ir_type / ckc_b_store_scalar_from_f32  (helper_..._io.h, ported)
 *   ckc_load_scalar_as_f32 / ckc_load_vec_as_f32    (io helper, peer)
 *   ckc_warp_xor_reduce_sum / ckc_causal_mask /
 *   ckc_sliding_window_mask                          (attention helper, peer)
 *
 * Error model: the builder is sticky-failing, matching ir.h. On the Python
 * `raise ValueError` paths (unsupported dtype, head_size % WARP_SIZE != 0) the
 * builder sticky error (CKC_ERR_VALUE) is set with the Python-matching message
 * and the function returns without emitting. No-op when the builder is already
 * in an error state, like every ckc_b_* call.
 */
#ifndef CKC_HELPER_CK_DSL_INSTANCES_COMMON__FMHA_WARP_BODY_H
#define CKC_HELPER_CK_DSL_INSTANCES_COMMON__FMHA_WARP_BODY_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WARP_SIZE = 64. One warp = warp_size threads per CTA; lane t owns the
 * head-dim slice [t*EPT, (t+1)*EPT) where EPT = head_size / warp_size. */
#define CKC_FMHA_WARP_SIZE 64

/* Accessor mirroring the module-level constant (lets callers that bind by
 * symbol read WARP_SIZE without the macro). Returns 64. */
int ckc_fmha_warp_size(void);

/* ---------------------------------------------------------------- callbacks */

/* extra_score_transform: (b, score_log2, k_idx) -> score_log2.
 * Invoked after the QK reduction, before the softmax update. NULL => skipped
 * (Python `is not None` guard). Used by the sage attention path. */
typedef ckc_value_t* (*ckc_fmha_score_transform_fn)(ckc_ir_builder_t* b,
                                                    ckc_value_t* score_log2,
                                                    ckc_value_t* k_idx,
                                                    void* user);

/* extra_mask_predicate: (b, k_idx) -> i1. When false the score is forced to
 * -inf. NULL => skipped. Used by the sparse-attention paths. */
typedef ckc_value_t* (*ckc_fmha_mask_predicate_fn)(ckc_ir_builder_t* b,
                                                   ckc_value_t* k_idx,
                                                   void* user);

/* k_row_base_fn / v_row_base_fn: (b, k_idx) -> i32 row-base element offset
 * (everything except the head-dim slot). NULL => dense linear addressing.
 * Override for paged-KV. */
typedef ckc_value_t* (*ckc_fmha_row_base_fn)(ckc_ir_builder_t* b, ckc_value_t* k_idx, void* user);

/* kv_lane_loader: (b, k_idx, k_row_base, v_row_base, lane_d_base, ept)
 *                    -> (k_f32_list, v_f32_list).
 * Writes this lane's `ept` dequantised-to-f32 K slot Values into out_k[0..ept)
 * and the V slot Values into out_v[0..ept). NULL => default linear scalar load.
 * When set, the body uses the generalised scalar-per-slot list path regardless
 * of ept so the loader fully owns the K/V addressing. */
typedef void (*ckc_fmha_kv_lane_loader_fn)(ckc_ir_builder_t* b,
                                           ckc_value_t* k_idx,
                                           ckc_value_t* k_row_base,
                                           ckc_value_t* v_row_base,
                                           ckc_value_t* lane_d_base,
                                           int ept,
                                           ckc_value_t** out_k,
                                           ckc_value_t** out_v,
                                           void* user);

/* q_lane_loader: (b, q_row_base, lane_d_base, ept) -> q_f32_list.
 * Writes this lane's `ept` f32 Q slot Values into out_q[0..ept). Only consulted
 * when kv_lane_loader is also set (the list path). NULL => built-in Q load. */
typedef void (*ckc_fmha_q_lane_loader_fn)(ckc_ir_builder_t* b,
                                          ckc_value_t* q_row_base,
                                          ckc_value_t* lane_d_base,
                                          int ept,
                                          ckc_value_t** out_q,
                                          void* user);

/* ----------------------------------------------------------------- options */

/* The Python keyword-only parameter set. Required Values are the Q/K/V/O
 * tensors, the head/token indices and the strides; the trailing fields default
 * to the Python signature defaults:
 *   mask_mode      = "none"  (NULL => "none")
 *   sliding_window = 0
 *   causal_ctx_len = NULL    (Optional[Value])
 *   k_token_offset_elems / v_token_offset_elems = NULL (=> const_i32(0))
 *   all callbacks  = NULL
 * `user` is the single opaque environment pointer passed through to every
 * callback (the Python closures captured their own state). */
typedef struct ckc_fmha_warp_fwd_opts
{
    ckc_value_t* Q;
    ckc_value_t* K;
    ckc_value_t* V;
    ckc_value_t* O;

    int head_size;
    ckc_value_t* seqlen_k;
    ckc_value_t* q_token;
    ckc_value_t* head_idx;
    ckc_value_t* kv_head_idx;

    ckc_value_t* stride_q_token;
    ckc_value_t* stride_q_head;
    ckc_value_t* stride_k_token;
    ckc_value_t* stride_k_head;
    ckc_value_t* stride_v_token;
    ckc_value_t* stride_v_head;
    ckc_value_t* stride_o_token;
    ckc_value_t* stride_o_head;

    ckc_value_t* scale_log2;
    const char* dtype; /* "f16"/"fp16"/"bf16" */

    const char* mask_mode;             /* NULL => "none" */
    int sliding_window;                /* default 0 */
    ckc_value_t* causal_ctx_len;       /* NULL => unset (Optional) */
    ckc_value_t* k_token_offset_elems; /* NULL => const_i32(0) */
    ckc_value_t* v_token_offset_elems; /* NULL => const_i32(0) */

    ckc_fmha_score_transform_fn extra_score_transform;
    ckc_fmha_mask_predicate_fn extra_mask_predicate;
    ckc_fmha_row_base_fn k_row_base_fn;
    ckc_fmha_row_base_fn v_row_base_fn;
    ckc_fmha_kv_lane_loader_fn kv_lane_loader;
    ckc_fmha_q_lane_loader_fn q_lane_loader;

    void* user;
} ckc_fmha_warp_fwd_opts_t;

/* C99 port of fmha_warp_fwd_inner_body.
 *
 * One warp's worth of FMHA forward for one (q_token, head) row. Emits the IR
 * byte-faithfully to the Python builder-call sequence:
 *   - validates dtype in {f16,fp16,bf16} and head_size % WARP_SIZE == 0
 *     (raise ValueError -> sticky CKC_ERR_VALUE + return);
 *   - computes ept = head_size / WARP_SIZE, the lane head-dim base and the
 *     Q/O row bases;
 *   - dispatches the kv_lane_loader list path, the ept==1 scalar path, or the
 *     ept>=2 vector path exactly as Python.
 *
 * Returns nothing (the Python function returns None). Check
 * ckc_ir_builder_ok(b) for failures. */
void ckc_fmha_warp_fwd_inner_body(ckc_ir_builder_t* b, const ckc_fmha_warp_fwd_opts_t* opts);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_INSTANCES_COMMON__FMHA_WARP_BODY_H */
