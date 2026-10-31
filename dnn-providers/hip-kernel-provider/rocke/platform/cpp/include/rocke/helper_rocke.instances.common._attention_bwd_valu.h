/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_valu.h -- C port of
 * library/kernels/common/_attention_bwd_valu.py: the softmax-gradient VALU
 * levers (bf16_cast, bwd_exp2, stats_fma, stats_vec) of the attention
 * backward main kernel.
 *
 *   Python (_attention_bwd_valu.py)     C (this header)
 *   ----------------------------------  ----------------------------------------
 *   BF16_CASTS / BWD_EXP2S / STATS_VECS ROCKE_ATTN_BWD_BF16_CASTS(_N) / _BWD_EXP2S(_N)
 *                                       / _STATS_VECS(_N)
 *   NATIVE_BF16_CVT_ARCHS               ROCKE_ATTN_BWD_NATIVE_BF16_CVT_ARCHS(_N)
 *   rne_noguard_bf16                    rocke_attn_bwd_rne_noguard_bf16
 *   cast_c_frag                         rocke_attn_bwd_cast_c_frag
 *   softmax_p                           rocke_attn_bwd_softmax_p
 *   check_stats_vec_layout              rocke_attn_bwd_check_stats_vec_layout
 *   batched_stats_rows                  rocke_attn_bwd_batched_stats_rows
 *   stats_rows_neg                      rocke_attn_bwd_stats_rows_neg
 *
 * check_stats_vec_layout evaluates the atom's C lane map on host integers
 * (Python's eval_layout of _attention_bwd_frag, an IntLaneBuilder). The C lane
 * maps emit builder ops, so the map is run on a scratch builder with a
 * constant lane and its integer-arithmetic ops are interpreted; the caller's
 * builder is not touched. The scratch evaluation is private to this file: the
 * frag port (another unit) owns the public eval_layout, if it has one.
 *
 * The stats dict of stats_rows_neg ({key: (lse2, Dsum)}) is an array of
 * (lse2, Dsum) pairs in dict order; the keys are not needed.
 *
 * Errors (Python ValueError) are recorded on the builder's sticky error as
 * ROCKE_ERR_VALUE; the Value-returning entries return NULL, the bool ones
 * false. Builder calls are made in Python's order.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_VALU_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_VALU_H

#include <stdbool.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_ATTN_BWD_BF16_CASTS_N 2
extern const char* const ROCKE_ATTN_BWD_BF16_CASTS[ROCKE_ATTN_BWD_BF16_CASTS_N];
#define ROCKE_ATTN_BWD_BWD_EXP2S_N 2
extern const char* const ROCKE_ATTN_BWD_BWD_EXP2S[ROCKE_ATTN_BWD_BWD_EXP2S_N];
#define ROCKE_ATTN_BWD_STATS_VECS_N 3
extern const int ROCKE_ATTN_BWD_STATS_VECS[ROCKE_ATTN_BWD_STATS_VECS_N];
#define ROCKE_ATTN_BWD_NATIVE_BF16_CVT_ARCHS_N 1
extern const char* const
    ROCKE_ATTN_BWD_NATIVE_BF16_CVT_ARCHS[ROCKE_ATTN_BWD_NATIVE_BF16_CVT_ARCHS_N];

/* <n x f32> -> <n x bf16> (n even), round to nearest even, no NaN guard. */
rocke_value_t* rocke_attn_bwd_rne_noguard_bf16(rocke_ir_builder_t* b, rocke_value_t* frag);

/* Round an fp32 C fragment to the I/O dtype elem under bf16_cast ("generic" or
 * "rne_noguard"; NULL is the default "generic"). */
rocke_value_t* rocke_attn_bwd_cast_c_frag(rocke_ir_builder_t* b,
                                          rocke_value_t* frag,
                                          const rocke_type_t* elem,
                                          const char* bf16_cast);

/* P = exp2(S * scale_log2 - lse2) of one element. neg_lse2 (-lse2, formed once
 * per row by the caller) selects the fused fma form; NULL is Python's None.
 * bwd_exp2: "exact" or "fast" (NULL is the default "exact"). */
rocke_value_t* rocke_attn_bwd_softmax_p(rocke_ir_builder_t* b,
                                        rocke_value_t* sv,
                                        rocke_value_t* scale_log2,
                                        rocke_value_t* lse2,
                                        rocke_value_t* neg_lse2,
                                        const char* bwd_exp2);

/* Raise (record a ValueError and return false) unless every lane's C slots of
 * op form runs of width consecutive rows. A NULL c_layout of the op is the
 * Python NotImplementedError of MmaOp.c_layout(), recorded as ROCKE_ERR_NOTIMPL. */
bool rocke_attn_bwd_check_stats_vec_layout(rocke_ir_builder_t* b,
                                           const rocke_mma_op_t* op,
                                           int width);

/* batched_stats_rows: (lse2, Dsum) of q-step rows, one vector load per run.
 * q0 and n_heads are host ints or SSA values (Python's int | Value);
 * n_heads is the query-head count. out_lse2 / out_dsum receive n_rows entries each. */
bool rocke_attn_bwd_batched_stats_rows(rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_workspace_addr_t* lse2,
                                       const rocke_attn_bwd_workspace_addr_t* dsum,
                                       rocke_value_t* h,
                                       rocke_attn_bwd_iv_t q0,
                                       rocke_value_t* const* rows,
                                       int n_rows,
                                       int width,
                                       rocke_attn_bwd_iv_t n_heads,
                                       rocke_value_t** out_lse2,
                                       rocke_value_t** out_dsum);

/* stats_rows_neg: out[i] = -lse2[i] for n (lse2, Dsum) pairs in key order. */
void rocke_attn_bwd_stats_rows_neg(rocke_ir_builder_t* b,
                                   rocke_value_t* const* lse2,
                                   int n,
                                   rocke_value_t** out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_VALU_H */
