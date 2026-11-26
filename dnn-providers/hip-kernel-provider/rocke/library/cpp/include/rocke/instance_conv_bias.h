// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_bias.h -- per-output-channel bias of the convolution forward
 * kernels, C++ twin of the library's kernels/conv/conv_bias.py.
 *
 *   y[n, (d,) h, w, g*kpg + k] = round(acc_fp32 + float(bias[g*kpg + k]))
 *
 * One rounding: the bias is added to the fp32 accumulators before any scalar
 * accumulator epilogue and before the conversion to the output dtype. The
 * bias vector has K entries in the output dtype and arrives as the trailing
 * Bias / Bias_bytes kernarg pair (conv_abi), read through a buffer resource
 * sized by Bias_bytes (a channel past the end reads 0).
 *
 *   rocke_conv_bias_declare_params()      <->  declare_conv_bias_params()
 *   rocke_conv_bias_rsrc()                <->  conv_bias_rsrc()
 *   rocke_conv_bias_load_f32()            <->  load_conv_bias_f32()
 *   rocke_conv_bias_load_vec_f32()        <->  load_conv_bias_vec_f32()
 *   rocke_conv_bias_load_run_f32()        <->  load_conv_bias_run_f32()
 *   rocke_conv_bias_add_to_accs()         <->  add_conv_bias_to_accs()
 *   rocke_conv_bias_apply_implicit_gemm() <->  apply_implicit_gemm_conv_bias()
 *
 * Each emits the same builder-call sequence as its Python peer; the
 * byte-identity gate holds the two together.
 */
#ifndef ROCKE_INSTANCE_CONV_BIAS_H
#define ROCKE_INSTANCE_CONV_BIAS_H

#include "rocke/helper_rocke.core.arch.h" /* rocke_mmaop_t */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Declare the Bias pointer (elem = the output dtype; noalias, readonly, align
 * 16) and its Bias_bytes size. Call after every other kernarg. */
void rocke_conv_bias_declare_params(rocke_ir_builder_t* b,
                                    const rocke_type_t* elem,
                                    rocke_value_t** bias,
                                    rocke_value_t** bias_bytes);

/* The buffer resource the bias is read through. */
rocke_value_t*
    rocke_conv_bias_rsrc(rocke_ir_builder_t* b, rocke_value_t* bias, rocke_value_t* bias_bytes);

/* bias[k] (element index k) widened to fp32; c0 is the soffset (i32 zero). */
rocke_value_t* rocke_conv_bias_load_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* rsrc,
                                        rocke_value_t* k,
                                        const rocke_type_t* elem,
                                        rocke_value_t* c0);

/* bias[k .. k+n) widened to an <n x f32> vector, read in loads of at most 16
 * bytes (the direct kernels, which store runs of channels). n must be a whole
 * number of such loads. */
rocke_value_t* rocke_conv_bias_load_vec_f32(rocke_ir_builder_t* b,
                                            rocke_value_t* rsrc,
                                            rocke_value_t* k,
                                            int n,
                                            const rocke_type_t* elem,
                                            rocke_value_t* c0);

/* bias[k .. k+n) as an <n x f32> vector, one scalar load per element
 * (rocke_conv_bias_load_f32): a run that may start off the vector alignment. */
rocke_value_t* rocke_conv_bias_load_run_f32(rocke_ir_builder_t* b,
                                            rocke_value_t* rsrc,
                                            rocke_value_t* k,
                                            int n,
                                            const rocke_type_t* elem,
                                            rocke_value_t* c0);

/* out[mi*mfmas_n + ni] = accs[mi*mfmas_n + ni] + splat(bias_per_ni[ni]).
 * Valid when every slot of a lane's fragment lies in one output column (all
 * MFMA / WMMA fp32 accumulator layouts the conv kernels use). */
void rocke_conv_bias_add_to_accs(rocke_ir_builder_t* b,
                                 rocke_value_t* const* accs,
                                 rocke_value_t* const* bias_per_ni,
                                 int mfmas_m,
                                 int mfmas_n,
                                 rocke_value_t** out);

/* The implicit-GEMM forward's bias add on its accumulator fragments: fragment
 * column ni of a lane is output channel k_out_group_base + block_n_off +
 * warp_n_idx * mfmas_n * warp_tile_n + ni * warp_tile_n + col(lane), col from
 * the op's destination layout at slot 0. k_out_group_base is NULL when
 * ungrouped. Writes mfmas_m * mfmas_n results into out. */
void rocke_conv_bias_apply_implicit_gemm(rocke_ir_builder_t* b,
                                         rocke_value_t* const* accs,
                                         const rocke_mmaop_t* op,
                                         const rocke_type_t* elem,
                                         rocke_value_t* bias,
                                         rocke_value_t* bias_bytes,
                                         rocke_value_t* lane,
                                         rocke_value_t* warp_n_idx,
                                         rocke_value_t* block_n_off,
                                         rocke_value_t* k_out_group_base,
                                         int mfmas_m,
                                         int mfmas_n,
                                         int warp_tile_n,
                                         rocke_value_t** out);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_CONV_BIAS_H */
