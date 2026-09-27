// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sparse_attention_inner_body_emit.c -- the mfma_attention_fwd_inner_body
 * wiring + b.ret() + `return kb.kernel` tail of both sparse-attention builders,
 * ported from rocke/instances/common/sparse_attention.py:
 *
 *   - rocke_jenga_emit_body  mirrors lines 513-539 (the Jenga inner-body call).
 *   - rocke_vsa_emit_body    mirrors lines 615-641 (the VSA inner-body call).
 *
 * Both phases assemble a rocke_mfma_attn_params_t identical to the Python keyword
 * call (Q/K/V/O, head_size, seqlen_k, q_tile_base, head/kv_head idx, all eight
 * q/k/v/o stride_token/stride_head pulled from ctx->kb, scale_log2, dtype,
 * mask_mode="none", arch), wire extra_mask_predicate to the bucket's tile
 * predicate with user=ctx, run rocke_mfma_attention_fwd_inner_body, emit b.ret(),
 * and return ctx->kb.kernel.
 *
 * The two builders are byte-identical except for which predicate / ctx type is
 * threaded into extra_mask_predicate, so the param assembly is shared in a static
 * helper. This TU binds peers via instance_sparse_attention_internal.h only and
 * edits no headers.
 */
#include <string.h> /* memset */

#include "rocke/instance_sparse_attention_internal.h"

#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/ir.h"

/* -------------------------------------------------------------------------
 * Shared inner-body call.
 *
 * Mirrors the Python keyword call to mfma_attention_fwd_inner_body(...) shared by
 * both builders (sparse_attention.py lines 513-537 / 615-639). Every field is set
 * in Python keyword order; the only per-bucket varying bits are the dtype/arch/
 * geometry pulled from the caller and the (predicate, user) pair. Optional Value
 * args / callbacks the Python call leaves at their defaults are zero-initialised
 * (NULL == Python None / unset keyword), matching rocke_mfma_attn_params_t's
 * "NULL => default" contract.
 *
 * Returns true if the inner body emitted cleanly (ROCKE_OK), false otherwise. On
 * either path the builder sticky error already carries any diagnostic.
 * ------------------------------------------------------------------------- */
static bool rocke_sparse_attn_emit_inner_body(rocke_ir_builder_t* b,
                                              rocke_fmha_kernel_builder_t* kb,
                                              const rocke_fmha_common_spec_t* s,
                                              rocke_value_t* Q,
                                              rocke_value_t* K,
                                              rocke_value_t* V,
                                              rocke_value_t* O,
                                              rocke_value_t* seqlen_k_arg,
                                              rocke_value_t* q_tile_base,
                                              rocke_value_t* head_idx,
                                              rocke_value_t* kv_head_idx,
                                              rocke_value_t* scale_log2,
                                              const char* arch,
                                              rocke_attn_predicate_fn extra_mask_predicate,
                                              void* extra_mask_predicate_user)
{
    rocke_mfma_attn_params_t p;
    rocke_status_t st;

    /* Zero-init so every keyword the Python call omits maps to NULL / default
     * (q_pos_base, sliding_window, kv_dtype, callbacks, the bool flags, ...). */
    memset(&p, 0, sizeof(p));

    /* Q=Q, K=K, V=V, O=O */
    p.Q = Q;
    p.K = K;
    p.V = V;
    p.O = O;

    /* head_size=s.shape.head_size */
    p.head_size = rocke_fmha_common_spec_head_size(s);

    /* seqlen_k=seqlen_k_arg */
    p.seqlen_k = seqlen_k_arg;

    /* q_tile_base=q_tile_base */
    p.q_tile_base = q_tile_base;

    /* head_idx=head_idx, kv_head_idx=kv_head_idx */
    p.head_idx = head_idx;
    p.kv_head_idx = kv_head_idx;

    /* stride_{q,k,v,o}_{token,head}=kb.stride_{token,head}(...) -- the eight
     * stride Values, pulled from ctx->kb in Python keyword order. */
    p.stride_q_token = rocke_fmha_kernel_builder_stride_token(kb, "q");
    p.stride_q_head = rocke_fmha_kernel_builder_stride_head(kb, "q");
    p.stride_k_token = rocke_fmha_kernel_builder_stride_token(kb, "k");
    p.stride_k_head = rocke_fmha_kernel_builder_stride_head(kb, "k");
    p.stride_v_token = rocke_fmha_kernel_builder_stride_token(kb, "v");
    p.stride_v_head = rocke_fmha_kernel_builder_stride_head(kb, "v");
    p.stride_o_token = rocke_fmha_kernel_builder_stride_token(kb, "o");
    p.stride_o_head = rocke_fmha_kernel_builder_stride_head(kb, "o");

    /* scale_log2=scale_log2 */
    p.scale_log2 = scale_log2;

    /* dtype=s.dtype */
    p.dtype = s->dtype;

    /* mask_mode="none" */
    p.mask_mode = ROCKE_ATTN_MASK_NONE;

    /* extra_mask_predicate=_<bucket>_tile_predicate (user = ctx) */
    p.extra_mask_predicate = extra_mask_predicate;
    p.extra_mask_predicate_user = extra_mask_predicate_user;

    /* arch=arch */
    p.arch = arch;

    st = rocke_mfma_attention_fwd_inner_body(b, &p);
    return st == ROCKE_OK;
}

/* ------------------------------------------------- rocke_jenga_emit_body *
 *
 * sparse_attention.py lines 513-539. */
rocke_kernel_def_t* rocke_jenga_emit_body(rocke_jenga_sparse_ctx_t* ctx)
{
    if(!rocke_sparse_attn_emit_inner_body(ctx->b,
                                          &ctx->kb,
                                          &ctx->s,
                                          ctx->Q,
                                          ctx->K,
                                          ctx->V,
                                          ctx->O,
                                          ctx->seqlen_k_arg,
                                          ctx->q_tile_base,
                                          ctx->head_idx,
                                          ctx->kv_head_idx,
                                          ctx->scale_log2,
                                          ctx->arch,
                                          rocke_jenga_tile_predicate,
                                          ctx))
    {
        return NULL;
    }

    /* b.ret() */
    rocke_b_ret(ctx->b);

    /* return kb.kernel */
    return rocke_fmha_kernel_builder_kernel(&ctx->kb);
}

/* --------------------------------------------------- rocke_vsa_emit_body *
 *
 * sparse_attention.py lines 615-641. */
rocke_kernel_def_t* rocke_vsa_emit_body(rocke_vsa_sparse_ctx_t* ctx)
{
    if(!rocke_sparse_attn_emit_inner_body(ctx->b,
                                          &ctx->kb,
                                          &ctx->s,
                                          ctx->Q,
                                          ctx->K,
                                          ctx->V,
                                          ctx->O,
                                          ctx->seqlen_k_arg,
                                          ctx->q_tile_base,
                                          ctx->head_idx,
                                          ctx->kv_head_idx,
                                          ctx->scale_log2,
                                          ctx->arch,
                                          rocke_vsa_tile_predicate,
                                          ctx))
    {
        return NULL;
    }

    /* b.ret() */
    rocke_b_ret(ctx->b);

    /* return kb.kernel */
    return rocke_fmha_kernel_builder_kernel(&ctx->kb);
}
