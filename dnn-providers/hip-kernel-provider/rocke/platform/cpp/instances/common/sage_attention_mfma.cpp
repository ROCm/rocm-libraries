// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sage_attention_sage_attention_mfma.c -- C99 port of _build_sage_mfma
 * (rocke/instances/common/sage_attention.py, lines 464-591): the MFMA-tiled
 * physical Sage forward kernel for fp16 / fp8 K-V storage.
 *
 * This translation unit implements the MFMA-path phase functions declared in
 * rocke/instance_sage_attention_internal.h:
 *
 *   rocke_sage_mfma_prologue                 (Python lines 477-496)
 *   rocke_sage_mfma_fold_scales              (Python lines 498-555)
 *   rocke_sage_mfma_k_block_scale_transform  (Python lines 534-553, the
 *                                           extra_score_transform closure)
 *   rocke_sage_mfma_emit_body                (Python lines 557-591)
 *
 * Each phase reads / writes only the carried rocke_sage_mfma_ctx_t (+ the builder
 * it carries) and emits IR in byte-identical Python order. Peer phases /
 * primitives (rocke_sage_declare_params, rocke_sage_magic_div, the qk_scale loaders,
 * rocke_mfma_attention_fwd_inner_body) are called through the shared headers.
 */
#include <stddef.h>

#include "rocke/instance_sage_attention_internal.h"

/* ---------------------------------------------------------------------- *
 * mask-mode mapping (Python `s.mask_mode in ("causal","sliding_window")`).
 * The common-spec mask mode (rocke_fmha_mask_mode_t) lowers to the attention
 * helper's narrower enum (rocke_attn_mask_mode_t) used by the params struct.
 * ---------------------------------------------------------------------- */
static rocke_attn_mask_mode_t sage_mfma_to_attn_mask(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_CAUSAL:
        return ROCKE_ATTN_MASK_CAUSAL;
    case ROCKE_FMHA_MASK_SLIDING_WINDOW:
        return ROCKE_ATTN_MASK_SLIDING_WINDOW;
    default:
        return ROCKE_ATTN_MASK_NONE;
    }
}

/* ===================================================================== *
 *  Prologue (Python lines 477-496).
 *
 *      s = spec.common
 *      kb = FmhaKernelBuilder(spec.kernel_name(), s)
 *      kb.block_size(64)
 *      _declare_params(kb, spec)
 *      kb.decode_grid()
 *      b = kb.builder
 *
 *      Q = kb.tensor("Q"); K = kb.tensor("K"); V = kb.tensor("V"); O = kb.tensor("O")
 *      q_scale_ptr = kb.ptr("q_scale"); k_scale_ptr = kb.ptr("k_scale")
 *      scale_log2_raw = kb.scalar("scale_log2"); seqlen_k_arg = kb.scalar("seqlen_k")
 *      q_tile_idx = kb.q_token; head_idx = kb.head_idx; kv_head_idx = kb.kv_head_idx
 *      q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
 *      batch_idx = b.const_i32(0)
 *
 *  The caller has already initialised ctx->kb with spec.kernel_name() +
 *  spec.common (FmhaKernelBuilder(...)). ctx->s is set here from spec.common.
 * ===================================================================== */
void rocke_sage_mfma_prologue(rocke_sage_mfma_ctx_t* ctx)
{
    rocke_ir_builder_t* b;

    if(ctx == NULL || ctx->kb == NULL || ctx->spec == NULL)
    {
        return;
    }

    /* s = spec.common */
    ctx->s = ctx->spec->common;

    /* kb.block_size(64) */
    rocke_fmha_kernel_builder_block_size(ctx->kb, 64);

    /* _declare_params(kb, spec) */
    rocke_sage_declare_params(ctx->kb, ctx->spec);

    /* kb.decode_grid() */
    rocke_fmha_kernel_builder_decode_grid(ctx->kb,
                                          /*num_queries_per_kv*/ -1,
                                          /*has_batch_axis*/ false,
                                          NULL,
                                          NULL,
                                          NULL);

    /* b = kb.builder */
    b = rocke_fmha_kernel_builder_builder(ctx->kb);
    ctx->b = b;

    /* Q / K / V / O tensors */
    ctx->Q = rocke_fmha_kernel_builder_tensor(ctx->kb, "Q");
    ctx->K = rocke_fmha_kernel_builder_tensor(ctx->kb, "K");
    ctx->V = rocke_fmha_kernel_builder_tensor(ctx->kb, "V");
    ctx->O = rocke_fmha_kernel_builder_tensor(ctx->kb, "O");

    /* scale / scalar params */
    ctx->q_scale_ptr = rocke_fmha_kernel_builder_ptr(ctx->kb, "q_scale");
    ctx->k_scale_ptr = rocke_fmha_kernel_builder_ptr(ctx->kb, "k_scale");
    ctx->scale_log2_raw = rocke_fmha_kernel_builder_scalar(ctx->kb, "scale_log2");
    ctx->seqlen_k_arg = rocke_fmha_kernel_builder_scalar(ctx->kb, "seqlen_k");

    /* decoded grid coords */
    ctx->q_tile_idx = ctx->kb->q_token;
    ctx->head_idx = ctx->kb->head_idx;
    ctx->kv_head_idx = ctx->kb->kv_head_idx;

    /* q_tile_base = q_tile_idx * BLOCK_M */
    ctx->q_tile_base
        = rocke_b_mul(b, ctx->q_tile_idx, rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_M));

    /* batch_idx = const_i32(0) */
    ctx->batch_idx = rocke_b_const_i32(b, 0);
}

/* ===================================================================== *
 *  Q-scale preload + scale fold (Python lines 498-555).
 *
 *      if spec.q_scale.layout == "per_block":
 *          q_block_idx = _magic_div(b, q_tile_base, spec.q_scale.scale_block)
 *      else:
 *          q_block_idx = b.const_i32(0)
 *      q_scale_v = load_q_scale_for_block(b, q_scale_ptr, spec=spec.q_scale,
 *                                         batch_idx, head_idx, q_block_idx)
 *
 *      if spec.k_scale.layout == "per_head":
 *          k_scale_const = load_k_scale_for_block(b, k_scale_ptr, spec=spec.k_scale,
 *                                                 batch_idx, kv_head_idx, const_i32(0))
 *          scale_log2 = b.fmul(scale_log2_raw, b.fmul(q_scale_v, k_scale_const))
 *          extra_score_transform = None
 *      else:
 *          scale_log2 = b.fmul(scale_log2_raw, q_scale_v)
 *          c_block_k = b.const_i32(MFMA_ATTN_BLOCK_K)
 *          extra_score_transform = _k_block_scale_transform
 *
 *  The per_head path sets ctx->k_scale_const and leaves ctx->c_block_k NULL;
 *  the per_block path leaves ctx->k_scale_const NULL and sets ctx->c_block_k
 *  (the closure constant for rocke_sage_mfma_k_block_scale_transform).
 * ===================================================================== */
void rocke_sage_mfma_fold_scales(rocke_sage_mfma_ctx_t* ctx)
{
    rocke_ir_builder_t* b;

    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return;
    }
    b = ctx->b;

    /* q_block_idx */
    if(ctx->spec->q_scale.layout == ROCKE_QK_SCALE_PER_BLOCK)
    {
        ctx->q_block_idx
            = rocke_sage_magic_div(b, ctx->q_tile_base, ctx->spec->q_scale.scale_block);
    }
    else
    {
        ctx->q_block_idx = rocke_b_const_i32(b, 0);
    }

    /* q_scale_v = load_q_scale_for_block(...) */
    ctx->q_scale_v = rocke_b_load_q_scale_for_block(
        b, ctx->q_scale_ptr, &ctx->spec->q_scale, ctx->batch_idx, ctx->head_idx, ctx->q_block_idx);

    if(ctx->spec->k_scale.layout == ROCKE_QK_SCALE_PER_HEAD)
    {
        /* k_scale_const = load_k_scale_for_block(..., k_block_idx=const_i32(0)) */
        ctx->k_scale_const = rocke_b_load_k_scale_for_block(b,
                                                            ctx->k_scale_ptr,
                                                            &ctx->spec->k_scale,
                                                            ctx->batch_idx,
                                                            ctx->kv_head_idx,
                                                            rocke_b_const_i32(b, 0));

        /* scale_log2 = raw * (q_scale_v * k_scale_const) */
        ctx->scale_log2 = rocke_b_fmul(
            b, ctx->scale_log2_raw, rocke_b_fmul(b, ctx->q_scale_v, ctx->k_scale_const));

        /* extra_score_transform = None */
        ctx->c_block_k = NULL;
    }
    else
    {
        /* scale_log2 = raw * q_scale_v */
        ctx->scale_log2 = rocke_b_fmul(b, ctx->scale_log2_raw, ctx->q_scale_v);

        /* c_block_k = const_i32(MFMA_ATTN_BLOCK_K) -- closure constant. */
        ctx->c_block_k = rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_K);

        /* k_scale_const stays NULL (per_head-only fold). */
        ctx->k_scale_const = NULL;
    }
}

/* ===================================================================== *
 *  extra_score_transform closure _k_block_scale_transform (Python lines
 *  534-553):
 *
 *      def _k_block_scale_transform(b, score, kt, _row_in_atom):
 *          k_pos = b.mul(kt, c_block_k)
 *          k_block_idx = _magic_div(b, k_pos, spec.k_scale.scale_block)
 *          k_scale_v = load_k_scale_for_block(b, k_scale_ptr, spec=spec.k_scale,
 *                                             batch_idx, kv_head_idx, k_block_idx)
 *          return b.fmul(score, k_scale_v)
 *
 *  Used only on the per_block k_scale path. `user` is the rocke_sage_mfma_ctx_t*
 *  (it carries c_block_k / k_scale_ptr / batch_idx / kv_head_idx / spec).
 * ===================================================================== */
rocke_value_t* rocke_sage_mfma_k_block_scale_transform(rocke_ir_builder_t* b,
                                                       rocke_value_t* score_log2,
                                                       rocke_value_t* kt,
                                                       int row_in_atom,
                                                       void* user)
{
    rocke_sage_mfma_ctx_t* ctx = (rocke_sage_mfma_ctx_t*)user;
    rocke_value_t* k_pos;
    rocke_value_t* k_block_idx;
    rocke_value_t* k_scale_v;

    (void)row_in_atom; /* Python `_row_in_atom` -- unused */

    if(ctx == NULL)
    {
        return NULL;
    }

    /* k_pos = b.mul(kt, c_block_k) */
    k_pos = rocke_b_mul(b, kt, ctx->c_block_k);

    /* k_block_idx = _magic_div(b, k_pos, spec.k_scale.scale_block) */
    k_block_idx = rocke_sage_magic_div(b, k_pos, ctx->spec->k_scale.scale_block);

    /* k_scale_v = load_k_scale_for_block(...) */
    k_scale_v = rocke_b_load_k_scale_for_block(
        b, ctx->k_scale_ptr, &ctx->spec->k_scale, ctx->batch_idx, ctx->kv_head_idx, k_block_idx);

    /* return b.fmul(score, k_scale_v) */
    return rocke_b_fmul(b, score_log2, k_scale_v);
}

/* ===================================================================== *
 *  Body + epilogue (Python lines 557-591).
 *
 *      causal_ctx = b.const_i32(0) if s.mask_mode in ("causal","sliding_window") else None
 *      kv_dtype = "fp8e4m3" if spec.quant_mode == "fp8_bf16" else None
 *
 *      mfma_attention_fwd_inner_body(b, Q=Q, K=K, V=V, O=O,
 *          head_size=s.shape.head_size, seqlen_k=seqlen_k_arg,
 *          q_tile_base=q_tile_base, head_idx=head_idx, kv_head_idx=kv_head_idx,
 *          stride_q_token=kb.stride_token("q"), stride_q_head=kb.stride_head("q"),
 *          ... (k/v/o strides) ...,
 *          scale_log2=scale_log2, dtype=s.dtype, mask_mode=s.mask_mode,
 *          sliding_window=s.sliding_window, causal_ctx_offset=causal_ctx,
 *          kv_dtype=kv_dtype, extra_score_transform=extra_score_transform, arch=arch)
 *      b.ret()
 *      return kb.kernel
 * ===================================================================== */
rocke_kernel_def_t* rocke_sage_mfma_emit_body(rocke_sage_mfma_ctx_t* ctx)
{
    rocke_ir_builder_t* b;
    rocke_mfma_attn_params_t p;
    rocke_kernel_def_t* kernel;
    int i;
    bool masked;

    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return NULL;
    }
    b = ctx->b;

    /* causal_ctx = const_i32(0) for causal / sliding_window, else None. */
    masked = (ctx->s.mask_mode == ROCKE_FMHA_MASK_CAUSAL
              || ctx->s.mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW);
    ctx->causal_ctx = masked ? rocke_b_const_i32(b, 0) : NULL;

    /* kv_dtype = "fp8e4m3" for fp8_bf16, else None (fp16_bf16: no KV dequant). */
    ctx->kv_dtype = (ctx->spec->quant_mode == ROCKE_SAGE_QUANT_FP8_BF16) ? "fp8e4m3" : NULL;

    /* Populate rocke_mfma_attn_params_t (Python keyword-only argument list). */
    for(i = 0; i < (int)sizeof(p); ++i)
    {
        ((char*)&p)[i] = 0;
    }

    p.Q = ctx->Q;
    p.K = ctx->K;
    p.V = ctx->V;
    p.O = ctx->O;
    p.head_size = ctx->s.shape.head_size;
    p.seqlen_k = ctx->seqlen_k_arg;
    p.q_tile_base = ctx->q_tile_base;
    p.head_idx = ctx->head_idx;
    p.kv_head_idx = ctx->kv_head_idx;
    p.q_pos_base = NULL; /* default => q_tile_base */

    p.stride_q_token = rocke_fmha_kernel_builder_stride_token(ctx->kb, "q");
    p.stride_q_head = rocke_fmha_kernel_builder_stride_head(ctx->kb, "q");
    p.stride_k_token = rocke_fmha_kernel_builder_stride_token(ctx->kb, "k");
    p.stride_k_head = rocke_fmha_kernel_builder_stride_head(ctx->kb, "k");
    p.stride_v_token = rocke_fmha_kernel_builder_stride_token(ctx->kb, "v");
    p.stride_v_head = rocke_fmha_kernel_builder_stride_head(ctx->kb, "v");
    p.stride_o_token = rocke_fmha_kernel_builder_stride_token(ctx->kb, "o");
    p.stride_o_head = rocke_fmha_kernel_builder_stride_head(ctx->kb, "o");

    p.scale_log2 = ctx->scale_log2;
    p.dtype = ctx->s.dtype;
    p.mask_mode = sage_mfma_to_attn_mask(ctx->s.mask_mode);
    p.sliding_window = ctx->s.sliding_window;
    p.causal_ctx_offset = ctx->causal_ctx;
    p.kv_dtype = ctx->kv_dtype;

    /* extra_score_transform: per_block k_scale path only (c_block_k != NULL).
     * Wired with user=ctx so the closure can read its captured constants. */
    if(ctx->c_block_k != NULL)
    {
        p.extra_score_transform = rocke_sage_mfma_k_block_scale_transform;
        p.extra_score_transform_user = ctx;
    }
    else
    {
        p.extra_score_transform = NULL;
        p.extra_score_transform_user = NULL;
    }

    p.arch = ctx->arch;

    (void)rocke_mfma_attention_fwd_inner_body(b, &p);

    /* b.ret() */
    rocke_b_ret(b);

    /* return kb.kernel */
    kernel = rocke_fmha_kernel_builder_kernel(ctx->kb);
    if(rocke_ir_builder_status(b) != ROCKE_OK)
    {
        return NULL;
    }
    return kernel;
}
