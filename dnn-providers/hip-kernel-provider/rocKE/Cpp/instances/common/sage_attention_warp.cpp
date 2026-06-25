// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sage_attention_sage_attention_warp.c -- C99 port of the
 * warp-distributed physical Sage kernel (_build_sage_warp, lines 599-812 of
 * rocke/instances/common/sage_attention.py).
 *
 * SCOPE OF THIS TU.
 *   The warp-path phase functions over rocke_sage_warp_ctx_t:
 *     rocke_sage_warp_prologue          (Python lines 631-677, incl. the i4
 *                                      head_size geometry validation 638-655)
 *     rocke_sage_warp_stage_codebooks   (lines 679-696)
 *     rocke_sage_warp_preload_q_scale   (lines 698-712)
 *     rocke_sage_warp_emit_body         (lines 780-812)
 *   plus the three warp-body seam closures (ported as free rocke_*_fn-shaped
 *   functions taking user == ctx):
 *     rocke_sage_warp_q_lane_loader     (lines 716-718)
 *     rocke_sage_warp_kv_lane_loader    (lines 720-756)
 *     rocke_sage_warp_qk_scale_transform(lines 758-778)
 *
 * The shared codebook / lane-load primitives, _declare_params, and the
 * _kv_pointee_for_quant_mode mapping are peers resolved at link time (declared
 * in rocke/instance_sage_attention_internal.h).
 */

#include "rocke/instance_sage_attention_internal.h"

#include <string.h> /* memset */

#include "rocke/helper_rocke.helpers.io.h" /* rocke_io_ir_type */
#include "rocke/ir_internal.h" /* rocke_i_set_err (sticky-error setter) */

/* ===================================================================== *
 *  WARP-PATH PHASE FUNCTIONS
 * ===================================================================== */

/* Prologue (lines 631-677).
 *
 *     s        = spec.common
 *     dtype    = s.dtype
 *     H        = s.shape.head_size
 *     block_size = WARP_SIZE
 *     kv_ty    = _kv_pointee_for_quant_mode(spec.quant_mode, dtype)
 *     q_pointee = io_ir_type(dtype)
 *
 *     if spec.quant_mode == "i4_fp8_bf16": (lines 638-655)
 *         if H < 2*WARP_SIZE:               raise ValueError(...)
 *         if H % (2*WARP_SIZE) != 0:        raise ValueError(...)
 *         ept_pairs = H // (2*WARP_SIZE)
 *         if ept_pairs != 1:               raise ValueError(...)
 *
 *     kb = FmhaKernelBuilder(spec.kernel_name(), s)   (driver-supplied here)
 *     kb.block_size(block_size)
 *     _declare_params(kb, spec)
 *     kb.decode_grid()
 *     b = kb.builder
 *     Q/K/V/O = kb.tensor(...); q_scale_ptr/k_scale_ptr = kb.ptr(...)
 *     cb_k/cb_v = kb.ptr("codebook_*") if codebook mode else None
 *     scale_log2 = kb.scalar("scale_log2"); seqlen_k = kb.scalar("seqlen_k")
 *     q_token/head_idx/kv_head_idx = kb.q_token/head_idx/kv_head_idx
 *     batch_idx = const_i32(0); tid = thread_id_x()
 */
bool rocke_sage_warp_prologue(rocke_sage_warp_ctx_t* ctx)
{
    rocke_fmha_kernel_builder_t* kb = ctx->kb;
    rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(kb);

    const rocke_fmha_common_spec_t* s = &ctx->spec->common;
    ctx->s = *s;
    ctx->dtype = s->dtype;
    ctx->H = s->shape.head_size;
    ctx->block_size = ROCKE_FMHA_WARP_SIZE;
    ctx->kv_ty = rocke_sage_kv_pointee_for_quant_mode(ctx->spec->quant_mode, ctx->dtype);
    ctx->q_pointee = rocke_io_ir_type(ctx->dtype);
    ctx->is_i4 = (ctx->spec->quant_mode == ROCKE_SAGE_QUANT_I4_FP8_BF16);

    const int H = ctx->H;

    /* i4 head_size geometry validation (lines 638-655). Two head-dim slots per
     * lane (one byte = two nibbles); requires head_size = 2*warp_size i.e.
     * >= 128, head_size % (2*warp_size) == 0, and exactly one byte per lane. */
    if(ctx->spec->quant_mode == ROCKE_SAGE_QUANT_I4_FP8_BF16)
    {
        if(H < 2 * ROCKE_FMHA_WARP_SIZE)
        {
            /* raise ValueError(
             *   f"i4 sage requires head_size >= {2*WARP_SIZE} so each "
             *   f"lane owns one packed byte (two nibbles); got {H}") */
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "i4 sage requires head_size >= 128 so each lane owns one packed "
                            "byte (two nibbles)");
            return false;
        }
        if(H % (2 * ROCKE_FMHA_WARP_SIZE) != 0)
        {
            /* raise ValueError(
             *   f"i4 sage requires head_size % {2*WARP_SIZE} == 0; got {H}") */
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "i4 sage requires head_size % 128 == 0");
            return false;
        }
        ctx->ept_pairs = H / (2 * ROCKE_FMHA_WARP_SIZE);
        if(ctx->ept_pairs != 1)
        {
            /* raise ValueError(
             *   "i4 sage v1 supports head_size == 128 (one byte per lane); "
             *   f"got head_size={H} which would need {ept_pairs} bytes/lane") */
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "i4 sage v1 supports head_size == 128 (one byte per lane)");
            return false;
        }
    }

    /* kb.block_size(block_size); _declare_params(kb, spec); kb.decode_grid(). */
    rocke_fmha_kernel_builder_block_size(kb, ctx->block_size);
    rocke_sage_declare_params(kb, ctx->spec);
    rocke_fmha_kernel_builder_decode_grid(kb, -1, false, NULL, NULL, NULL);

    ctx->b = b;

    /* Param Values. */
    ctx->Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
    ctx->K = rocke_fmha_kernel_builder_tensor(kb, "K");
    ctx->V = rocke_fmha_kernel_builder_tensor(kb, "V");
    ctx->O = rocke_fmha_kernel_builder_tensor(kb, "O");
    ctx->q_scale_ptr = rocke_fmha_kernel_builder_ptr(kb, "q_scale");
    ctx->k_scale_ptr = rocke_fmha_kernel_builder_ptr(kb, "k_scale");

    /* cb_k = kb.ptr("codebook_k") if quant_mode in _CODEBOOK_QUANT_MODES else None */
    if(rocke_sage_quant_mode_is_codebook(ctx->spec->quant_mode))
    {
        ctx->cb_k = rocke_fmha_kernel_builder_ptr(kb, "codebook_k");
        ctx->cb_v = rocke_fmha_kernel_builder_ptr(kb, "codebook_v");
    }
    else
    {
        ctx->cb_k = NULL;
        ctx->cb_v = NULL;
    }

    ctx->scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
    ctx->seqlen_k = rocke_fmha_kernel_builder_scalar(kb, "seqlen_k");

    /* q_token = kb.q_token ; head_idx = kb.head_idx ; kv_head_idx = kb.kv_head_idx */
    ctx->q_token = kb->q_token;
    ctx->head_idx = kb->head_idx;
    ctx->kv_head_idx = kb->kv_head_idx;

    ctx->batch_idx = rocke_b_const_i32(b, 0);
    ctx->tid = rocke_b_thread_id_x(b);

    return rocke_ir_builder_ok(b);
}

/* Codebook staging (lines 679-696).
 *
 *     if cb_k is not None:
 *         cb_entries = (_CODEBOOK_I4_ENTRIES if quant_mode == "i4_fp8_bf16"
 *                       else _CODEBOOK_I8_ENTRIES)
 *         cb_k = _stage_codebook_to_lds(b, cb_k, n_entries=cb_entries,
 *                                       tid=tid, name_hint="sage_cb_k")
 *         cb_v = _stage_codebook_to_lds(b, cb_v, n_entries=cb_entries,
 *                                       tid=tid, name_hint="sage_cb_v")
 *         b.sync()
 */
void rocke_sage_warp_stage_codebooks(rocke_sage_warp_ctx_t* ctx)
{
    if(ctx->cb_k == NULL)
        return;

    rocke_ir_builder_t* b = ctx->b;

    int cb_entries = (ctx->spec->quant_mode == ROCKE_SAGE_QUANT_I4_FP8_BF16)
                         ? ROCKE_SAGE_CODEBOOK_I4_ENTRIES
                         : ROCKE_SAGE_CODEBOOK_I8_ENTRIES;

    ctx->cb_k = rocke_sage_stage_codebook_to_lds(b, ctx->cb_k, cb_entries, ctx->tid, "sage_cb_k");
    ctx->cb_v = rocke_sage_stage_codebook_to_lds(b, ctx->cb_v, cb_entries, ctx->tid, "sage_cb_v");
    rocke_b_sync(b);
}

/* Q-scale preload (lines 698-712).
 *
 *     q_block_idx = (_magic_div(b, q_token, spec.q_scale.scale_block)
 *                    if spec.q_scale.layout == "per_block" else const_i32(0))
 *     q_scale_v = load_q_scale_for_block(b, q_scale_ptr, spec=spec.q_scale,
 *                     batch_idx=batch_idx, head_idx=head_idx,
 *                     q_block_idx=q_block_idx)
 */
void rocke_sage_warp_preload_q_scale(rocke_sage_warp_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    if(ctx->spec->q_scale.layout == ROCKE_QK_SCALE_PER_BLOCK)
        ctx->q_block_idx = rocke_sage_magic_div(b, ctx->q_token, ctx->spec->q_scale.scale_block);
    else
        ctx->q_block_idx = rocke_b_const_i32(b, 0);

    ctx->q_scale_v = rocke_b_load_q_scale_for_block(
        b, ctx->q_scale_ptr, &ctx->spec->q_scale, ctx->batch_idx, ctx->head_idx, ctx->q_block_idx);
}

/* ===================================================================== *
 *  WARP-BODY SEAM CLOSURES (user == ctx)
 * ===================================================================== */

/* _q_lane_loader (lines 716-718).
 *
 *     def _q_lane_loader(b, q_row_base, lane_d_base, ept) -> list[Value]:
 *         return _load_q_lane_f32(b, Q, q_row_base, lane_d_base, ept,
 *                                 dtype, q_pointee)
 *
 * (the helper writes ept f32 Values into out_q[0..ept).)
 */
void rocke_sage_warp_q_lane_loader(rocke_ir_builder_t* b,
                                   rocke_value_t* q_row_base,
                                   rocke_value_t* lane_d_base,
                                   int ept,
                                   rocke_value_t** out_q,
                                   void* user)
{
    rocke_sage_warp_ctx_t* ctx = (rocke_sage_warp_ctx_t*)user;
    rocke_sage_load_q_lane_f32(
        b, ctx->Q, q_row_base, lane_d_base, ept, ctx->dtype, ctx->q_pointee, out_q);
}

/* _kv_lane_loader (lines 720-756).
 *
 *     def _kv_lane_loader(b, k_idx, k_row_base, v_row_base, lane_d_base, ept):
 *         if is_i4:
 *             byte_off = _magic_div(b, lane_d_base, 2)  # = tid
 *             packed_k = b.global_load(K, b.add(k_row_base, byte_off), I8)
 *             lo_k, hi_k = _codebook_i4_pair_to_f32(b, cb_k, packed_k)
 *             packed_v = b.global_load(V, b.add(v_row_base, byte_off), I8)
 *             v_lo, v_hi = _codebook_i4_pair_to_f32(b, cb_v, packed_v)
 *             return [lo_k, hi_k], [v_lo, v_hi]
 *         k_lane = _load_kv_lane_f32(b, KV=K, base=k_row_base,
 *                      lane_d_base=lane_d_base, ept=ept, quant_mode=quant_mode,
 *                      cb_ptr=cb_k, kv_ty=kv_ty, dtype=dtype)
 *         v_lane = _load_kv_lane_f32(b, KV=V, base=v_row_base, ...)
 *         return k_lane, v_lane
 */
void rocke_sage_warp_kv_lane_loader(rocke_ir_builder_t* b,
                                    rocke_value_t* k_idx,
                                    rocke_value_t* k_row_base,
                                    rocke_value_t* v_row_base,
                                    rocke_value_t* lane_d_base,
                                    int ept,
                                    rocke_value_t** out_k,
                                    rocke_value_t** out_v,
                                    void* user)
{
    rocke_sage_warp_ctx_t* ctx = (rocke_sage_warp_ctx_t*)user;
    (void)k_idx; /* unused by sage's loader (matches the Python closure) */

    if(ctx->is_i4)
    {
        /* one packed byte per lane -> two nibbles -> two f32 (direct codebook) */
        rocke_value_t* byte_off = rocke_sage_magic_div(b, lane_d_base, 2); /* = tid */

        rocke_value_t* packed_k
            = rocke_b_global_load(b, ctx->K, rocke_b_add(b, k_row_base, byte_off), rocke_i8(), 1);
        rocke_sage_codebook_i4_pair_to_f32(b, ctx->cb_k, packed_k, &out_k[0], &out_k[1]);

        rocke_value_t* packed_v
            = rocke_b_global_load(b, ctx->V, rocke_b_add(b, v_row_base, byte_off), rocke_i8(), 1);
        rocke_sage_codebook_i4_pair_to_f32(b, ctx->cb_v, packed_v, &out_v[0], &out_v[1]);
        return;
    }

    rocke_sage_load_kv_lane_f32(b,
                                ctx->K,
                                k_row_base,
                                lane_d_base,
                                ept,
                                ctx->spec->quant_mode,
                                ctx->cb_k,
                                ctx->kv_ty,
                                ctx->dtype,
                                out_k);
    rocke_sage_load_kv_lane_f32(b,
                                ctx->V,
                                v_row_base,
                                lane_d_base,
                                ept,
                                ctx->spec->quant_mode,
                                ctx->cb_v,
                                ctx->kv_ty,
                                ctx->dtype,
                                out_v);
}

/* _qk_scale_transform (lines 758-778).
 *
 *     def _qk_scale_transform(b, score_log2, k_idx) -> Value:
 *         k_block_idx = (_magic_div(b, k_idx, spec.k_scale.scale_block)
 *                        if spec.k_scale.layout == "per_block"
 *                        else const_i32(0))
 *         k_scale_v = load_k_scale_for_block(b, k_scale_ptr, spec=spec.k_scale,
 *                         batch_idx=batch_idx, head_idx=kv_head_idx,
 *                         k_block_idx=k_block_idx)
 *         return apply_qk_scales(b, score_log2, q_scale=q_scale_v,
 *                                k_scale=k_scale_v)
 */
rocke_value_t* rocke_sage_warp_qk_scale_transform(rocke_ir_builder_t* b,
                                                  rocke_value_t* score_log2,
                                                  rocke_value_t* k_idx,
                                                  void* user)
{
    rocke_sage_warp_ctx_t* ctx = (rocke_sage_warp_ctx_t*)user;

    rocke_value_t* k_block_idx;
    if(ctx->spec->k_scale.layout == ROCKE_QK_SCALE_PER_BLOCK)
        k_block_idx = rocke_sage_magic_div(b, k_idx, ctx->spec->k_scale.scale_block);
    else
        k_block_idx = rocke_b_const_i32(b, 0);

    rocke_value_t* k_scale_v = rocke_b_load_k_scale_for_block(
        b, ctx->k_scale_ptr, &ctx->spec->k_scale, ctx->batch_idx, ctx->kv_head_idx, k_block_idx);

    return rocke_b_apply_qk_scales(b, score_log2, ctx->q_scale_v, k_scale_v);
}

/* ===================================================================== *
 *  WARP-PATH BODY + EPILOGUE (lines 780-812)
 *
 *     causal_ctx = q_token if s.mask_mode in ("causal","sliding_window") else None
 *     fmha_warp_fwd_inner_body(b, Q=Q, K=K, V=V, O=O, head_size=H,
 *         seqlen_k=seqlen_k, q_token=q_token, head_idx=head_idx,
 *         kv_head_idx=kv_head_idx, stride_*=kb.stride_*(...),
 *         scale_log2=scale_log2, dtype=dtype, mask_mode=s.mask_mode,
 *         sliding_window=s.sliding_window, causal_ctx_len=causal_ctx,
 *         extra_score_transform=_qk_scale_transform,
 *         kv_lane_loader=_kv_lane_loader, q_lane_loader=_q_lane_loader)
 *     b.ret()
 *     return kb.kernel
 * ===================================================================== */
rocke_kernel_def_t* rocke_sage_warp_emit_body(rocke_sage_warp_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_fmha_kernel_builder_t* kb = ctx->kb;

    /* causal_ctx = q_token if mask in {causal, sliding_window} else None */
    if(ctx->s.mask_mode == ROCKE_FMHA_MASK_CAUSAL
       || ctx->s.mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW)
        ctx->causal_ctx = ctx->q_token;
    else
        ctx->causal_ctx = NULL;

    rocke_fmha_warp_fwd_opts_t opts;
    memset(&opts, 0, sizeof(opts));

    opts.Q = ctx->Q;
    opts.K = ctx->K;
    opts.V = ctx->V;
    opts.O = ctx->O;

    opts.head_size = ctx->H;
    opts.seqlen_k = ctx->seqlen_k;
    opts.q_token = ctx->q_token;
    opts.head_idx = ctx->head_idx;
    opts.kv_head_idx = ctx->kv_head_idx;

    opts.stride_q_token = rocke_fmha_kernel_builder_stride_token(kb, "q");
    opts.stride_q_head = rocke_fmha_kernel_builder_stride_head(kb, "q");
    opts.stride_k_token = rocke_fmha_kernel_builder_stride_token(kb, "k");
    opts.stride_k_head = rocke_fmha_kernel_builder_stride_head(kb, "k");
    opts.stride_v_token = rocke_fmha_kernel_builder_stride_token(kb, "v");
    opts.stride_v_head = rocke_fmha_kernel_builder_stride_head(kb, "v");
    opts.stride_o_token = rocke_fmha_kernel_builder_stride_token(kb, "o");
    opts.stride_o_head = rocke_fmha_kernel_builder_stride_head(kb, "o");

    opts.scale_log2 = ctx->scale_log2;
    opts.dtype = ctx->dtype;

    opts.mask_mode = rocke_fmha_mask_mode_name(ctx->s.mask_mode);
    opts.sliding_window = ctx->s.sliding_window;
    opts.causal_ctx_len = ctx->causal_ctx;

    opts.extra_score_transform = rocke_sage_warp_qk_scale_transform;
    opts.kv_lane_loader = rocke_sage_warp_kv_lane_loader;
    opts.q_lane_loader = rocke_sage_warp_q_lane_loader;
    opts.user = ctx;

    rocke_fmha_warp_fwd_inner_body(b, &opts);

    rocke_b_ret(b);

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_fmha_kernel_builder_kernel(kb);
}
