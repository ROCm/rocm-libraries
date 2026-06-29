// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_attention_unified_attention_unified_3d_reduce.c -- C99 port of the
 * 3D-segment + reduce kernel-body bucket of
 * rocke/instances/common/attention_unified.py:
 *   build_unified_attention_3d      lines 2798-2914 (loop body + epilogue)
 *   build_unified_attention_reduce  lines 2941-3009 (params + max/combine + epilogue)
 *
 * Implements the phase functions:
 *   rocke_attn_unified_emit_3d_segment_loop
 *   rocke_attn_unified_emit_3d_epilogue
 *   rocke_attn_unified_declare_reduce_params
 *   rocke_attn_unified_emit_reduce_max_loop
 *   rocke_attn_unified_emit_reduce_combine_loop
 *   rocke_attn_unified_emit_reduce_epilogue
 *
 * Peers (prologue/ABI/descriptors/2D phases) live in sibling translation units
 * and are called only via the internal header. The builder-call sequence is a
 * byte-identical translation of the referenced Python spans.
 */

#include <math.h> /* INFINITY */
#include <string.h>

#include "rocke/instance_attention_unified_internal.h"

/* ============================================================ *
 * Local helpers (descriptor offset -> i32 element offset only).
 * Python `idx, _ = desc.offset(b, ...)` discards the validity. On a sticky
 * builder error these return NULL, matching the no-op propagation model.
 * ============================================================ */

static rocke_value_t* rocke__ml_offset(rocke_attn_unified_build_ctx_t* ctx,
                                       rocke_value_t* token,
                                       rocke_value_t* head,
                                       rocke_value_t* seg)
{
    const char* in_names[3] = {"token", "head", "seg"};
    rocke_value_t* in_values[3];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    if(!rocke_transforms_descriptor_offset(
           ctx->b, ctx->ml_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static rocke_value_t* rocke__out_offset(rocke_attn_unified_build_ctx_t* ctx,
                                        rocke_value_t* token,
                                        rocke_value_t* head,
                                        rocke_value_t* seg,
                                        rocke_value_t* dim)
{
    const char* in_names[4] = {"token", "head", "seg", "dim"};
    rocke_value_t* in_values[4];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    in_values[3] = dim;
    if(!rocke_transforms_descriptor_offset(
           ctx->b, ctx->out_desc, in_names, in_values, 4, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static rocke_value_t* rocke__q_offset(rocke_attn_unified_build_ctx_t* ctx,
                                      rocke_value_t* token,
                                      rocke_value_t* head,
                                      rocke_value_t* dim)
{
    const char* in_names[3] = {"token", "head", "dim"};
    rocke_value_t* in_values[3];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = dim;
    if(!rocke_transforms_descriptor_offset(
           ctx->b, ctx->q_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* ============================================================ *
 * 3D segment online-softmax loop (Python lines 2867-2898).
 *
 *   loop = b.scf_for_iter(seg_start, seg_stop_i, const_i32(1),
 *                         [("m", init_m), ("l", init_l), ("acc", init_acc)],
 *                         iv_name="kpos")
 *   with loop as (kpos, (m_val, l_val, acc_val)):
 *       score = _emit_qk_score(...)
 *       causal_ok = cmp_le(kpos, context_len + query_pos)
 *       score = select(causal_ok, score, neg_inf)
 *       new_m = fmax(m_val, score)
 *       alpha = exp2(m_val - new_m)
 *       prob  = exp2(score - new_m)
 *       new_l = fadd(fmul(l_val, alpha), prob)
 *       vv    = _emit_v_load(...)
 *       new_acc = fadd(fmul(acc_val, alpha), fmul(prob, vv))
 *       scf_yield(new_m, new_l, new_acc)
 *
 * Stashes loop.results[0..2] in ctx->loop_m / loop_l / loop_acc.
 * ============================================================ */
void rocke_attn_unified_emit_3d_segment_loop(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_iter_arg_t iter_args[3];
    rocke_for_t loop;
    rocke_value_t *kpos, *m_val, *l_val, *acc_val;
    rocke_value_t *score, *causal_ok, *new_m, *alpha, *prob, *new_l, *vv, *new_acc;
    rocke_value_t* yields[3];

    iter_args[0].name = "m";
    iter_args[0].init = ctx->init_m;
    iter_args[1].name = "l";
    iter_args[1].init = ctx->init_l;
    iter_args[2].name = "acc";
    iter_args[2].init = ctx->init_acc;

    loop = rocke_b_scf_for_iter(b,
                                ctx->seg_start,
                                ctx->seg_stop_i,
                                rocke_b_const_i32(b, 1),
                                iter_args,
                                3,
                                "kpos",
                                false,
                                true);

    rocke_b_region_enter(b, loop.body);

    kpos = loop.iv;
    m_val = loop.iter_vars ? loop.iter_vars[0] : NULL;
    l_val = loop.iter_vars ? loop.iter_vars[1] : NULL;
    acc_val = loop.iter_vars ? loop.iter_vars[2] : NULL;

    score = rocke_unified_attn_emit_qk_score(b,
                                             &ctx->sel_p,
                                             ctx->dtype,
                                             ctx->abi_query,
                                             ctx->abi_key,
                                             ctx->abi_block_tables,
                                             ctx->seq_idx,
                                             ctx->q_tok,
                                             ctx->q_head,
                                             ctx->kv_head,
                                             kpos,
                                             ctx->abi_scale,
                                             ctx->rcp_ln2);

    /* causal_ok = cmp_le(kpos, context_len + query_pos) */
    causal_ok = rocke_b_cmp_le(b, kpos, rocke_b_add(b, ctx->context_len, ctx->query_pos));
    score = rocke_b_select(b, causal_ok, score, ctx->neg_inf);

    new_m = rocke_b_fmax(b, m_val, score);
    alpha = rocke_b_exp2(b, rocke_b_fsub(b, m_val, new_m));
    prob = rocke_b_exp2(b, rocke_b_fsub(b, score, new_m));
    new_l = rocke_b_fadd(b, rocke_b_fmul(b, l_val, alpha), prob);

    vv = rocke_unified_attn_emit_v_load(b,
                                        &ctx->sel_p,
                                        ctx->dtype,
                                        ctx->abi_value,
                                        ctx->abi_block_tables,
                                        ctx->seq_idx,
                                        ctx->kv_head,
                                        kpos,
                                        ctx->dim);

    new_acc = rocke_b_fadd(b, rocke_b_fmul(b, acc_val, alpha), rocke_b_fmul(b, prob, vv));

    yields[0] = new_m;
    yields[1] = new_l;
    yields[2] = new_acc;
    rocke_b_scf_yield(b, yields, 3);

    rocke_b_region_leave(b);

    if(loop.op != NULL && loop.op->num_results >= 3)
    {
        ctx->loop_m = loop.op->results[0];
        ctx->loop_l = loop.op->results[1];
        ctx->loop_acc = loop.op->results[2];
    }
}

/* ============================================================ *
 * 3D guarded epilogue (Python lines 2900-2913).
 *
 *   ml_desc, out_desc = _segm_descriptors(p, num_segments)   [done in prologue]
 *   base, _ = out_desc.offset(b, token=q_tok, head=q_head, seg=segm_idx, dim=dim)
 *   with b.scf_if(active):
 *       b.global_store(segm_output, base, loop.results[2], align=4)
 *       is_dim0 = cmp_eq(dim, const_i32(0))
 *       with b.scf_if(is_dim0):
 *           segm_base, _ = ml_desc.offset(b, token=q_tok, head=q_head, seg=segm_idx)
 *           b.global_store(segm_max,    segm_base, loop.results[0], align=4)
 *           b.global_store(segm_expsum, segm_base, loop.results[1], align=4)
 * ============================================================ */
void rocke_attn_unified_emit_3d_epilogue(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* base;
    rocke_if_t outer;

    base = rocke__out_offset(ctx, ctx->q_tok, ctx->q_head, ctx->segm_idx, ctx->dim);

    outer = rocke_b_scf_if(b, ctx->active);
    rocke_b_region_enter(b, outer.then_region);
    {
        rocke_value_t* is_dim0;
        rocke_if_t inner;

        rocke_b_global_store(b, ctx->segm_output, base, ctx->loop_acc, 4);

        is_dim0 = rocke_b_cmp_eq(b, ctx->dim, rocke_b_const_i32(b, 0));
        inner = rocke_b_scf_if(b, is_dim0);
        rocke_b_region_enter(b, inner.then_region);
        {
            rocke_value_t* segm_base
                = rocke__ml_offset(ctx, ctx->q_tok, ctx->q_head, ctx->segm_idx);
            rocke_b_global_store(b, ctx->segm_max, segm_base, ctx->loop_m, 4);
            rocke_b_global_store(b, ctx->segm_expsum, segm_base, ctx->loop_l, 4);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_region_leave(b);
}

/* ============================================================ *
 * Reduce narrow ABI (Python lines 2946-2964).
 *
 *   out         = param("output_ptr",        ptr<dtype>,  noalias, writeonly, align=16)
 *   segm_output = param("segm_output_ptr",   ptr<f32>,    readonly, align=16)
 *   segm_max    = param("segm_max_ptr",      ptr<f32>,    readonly, align=16)
 *   segm_expsum = param("segm_expsum_ptr",   ptr<f32>,    readonly, align=16)
 *   _seq_lens   = param("seq_lens_ptr",      ptr<i32>,    readonly, align=4)
 *   _cu_q       = param("query_start_len_ptr", ptr<i32>,  readonly, align=4)
 *   q_tok = block_id_x(); q_head = block_id_y(); dim = block_id_z();
 *   tid = thread_id_x(); active = cmp_eq(tid, const_i32(0));
 *   neg_inf = const_f32(-inf); zero = const_f32(0.0)
 *
 * Param declaration order is the load-bearing kernel arg ABI.
 * ============================================================ */
void rocke_attn_unified_declare_reduce_params(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_param_opts_t opts;

    /* output_ptr : ptr<dtype, global>, noalias, writeonly, align=16 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    ctx->output = rocke_b_param(b, "output_ptr", rocke_ptr_type(b, ctx->dtype, "global"), &opts);

    /* segm_output_ptr : ptr<f32, global>, readonly, align=16 */
    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    ctx->segm_output
        = rocke_b_param(b, "segm_output_ptr", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    /* segm_max_ptr : ptr<f32, global>, readonly, align=16 */
    ctx->segm_max
        = rocke_b_param(b, "segm_max_ptr", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    /* segm_expsum_ptr : ptr<f32, global>, readonly, align=16 */
    ctx->segm_expsum
        = rocke_b_param(b, "segm_expsum_ptr", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    /* seq_lens_ptr : ptr<i32, global>, readonly, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->abi_seq_lens
        = rocke_b_param(b, "seq_lens_ptr", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* query_start_len_ptr : ptr<i32, global>, readonly, align=4 */
    ctx->abi_cu_q
        = rocke_b_param(b, "query_start_len_ptr", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* grid ids + thread predicate */
    ctx->q_tok = rocke_b_block_id_x(b);
    ctx->q_head = rocke_b_block_id_y(b);
    ctx->dim = rocke_b_block_id_z(b);
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->active = rocke_b_cmp_eq(b, ctx->tid, rocke_b_const_i32(b, 0));

    /* SSA constants used by the reduce passes */
    ctx->neg_inf = rocke_b_const_f32(b, -INFINITY);
    ctx->zero_f = rocke_b_const_f32(b, 0.0);
}

/* ============================================================ *
 * Reduce max pass (Python lines 2969-2980).
 *
 *   max_loop = scf_for_iter(0, num_segments, 1, [("mx", neg_inf)], iv_name="seg")
 *   with max_loop as (seg, (mx,)):
 *       idx, _ = ml_desc.offset(b, token=q_tok, head=q_head, seg=seg)
 *       mv = global_load_f32(segm_max, idx)
 *       scf_yield(fmax(mx, mv))
 *   overall = max_loop.results[0]
 * ============================================================ */
void rocke_attn_unified_emit_reduce_max_loop(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_iter_arg_t iter_args[1];
    rocke_for_t max_loop;
    rocke_value_t *seg, *mx, *idx, *mv;
    rocke_value_t* yields[1];

    iter_args[0].name = "mx";
    iter_args[0].init = ctx->neg_inf;

    max_loop = rocke_b_scf_for_iter(b,
                                    rocke_b_const_i32(b, 0),
                                    rocke_b_const_i32(b, ctx->num_segments),
                                    rocke_b_const_i32(b, 1),
                                    iter_args,
                                    1,
                                    "seg",
                                    false,
                                    true);

    rocke_b_region_enter(b, max_loop.body);

    seg = max_loop.iv;
    mx = max_loop.iter_vars ? max_loop.iter_vars[0] : NULL;

    idx = rocke__ml_offset(ctx, ctx->q_tok, ctx->q_head, seg);
    mv = rocke_b_global_load_f32(b, ctx->segm_max, idx, 0);

    yields[0] = rocke_b_fmax(b, mx, mv);
    rocke_b_scf_yield(b, yields, 1);

    rocke_b_region_leave(b);

    if(max_loop.op != NULL && max_loop.op->num_results >= 1)
    {
        ctx->overall = max_loop.op->results[0];
    }
}

/* ============================================================ *
 * Reduce combine pass (Python lines 2981-3003).
 *
 *   red = scf_for_iter(0, num_segments, 1, [("den", zero), ("acc", zero)],
 *                      iv_name="seg2")
 *   with red as (seg, (den, acc)):
 *       idx, _ = ml_desc.offset(b, token=q_tok, head=q_head, seg=seg)
 *       mv = global_load_f32(segm_max, idx)
 *       lv = global_load_f32(segm_expsum, idx)
 *       factor = exp2(mv - overall)
 *       den2 = fadd(den, fmul(lv, factor))
 *       out_idx, _ = seg_out_desc.offset(b, token=q_tok, head=q_head, seg=seg, dim=dim)
 *       ov = global_load_f32(segm_output, out_idx)
 *       acc2 = fadd(acc, fmul(ov, factor))
 *       scf_yield(den2, acc2)
 * ============================================================ */
void rocke_attn_unified_emit_reduce_combine_loop(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_iter_arg_t iter_args[2];
    rocke_for_t red;
    rocke_value_t *seg, *den, *acc, *idx, *mv, *lv, *factor, *den2, *out_idx, *ov, *acc2;
    rocke_value_t* yields[2];

    iter_args[0].name = "den";
    iter_args[0].init = ctx->zero_f;
    iter_args[1].name = "acc";
    iter_args[1].init = ctx->zero_f;

    red = rocke_b_scf_for_iter(b,
                               rocke_b_const_i32(b, 0),
                               rocke_b_const_i32(b, ctx->num_segments),
                               rocke_b_const_i32(b, 1),
                               iter_args,
                               2,
                               "seg2",
                               false,
                               true);

    rocke_b_region_enter(b, red.body);

    seg = red.iv;
    den = red.iter_vars ? red.iter_vars[0] : NULL;
    acc = red.iter_vars ? red.iter_vars[1] : NULL;

    idx = rocke__ml_offset(ctx, ctx->q_tok, ctx->q_head, seg);
    mv = rocke_b_global_load_f32(b, ctx->segm_max, idx, 0);
    lv = rocke_b_global_load_f32(b, ctx->segm_expsum, idx, 0);
    factor = rocke_b_exp2(b, rocke_b_fsub(b, mv, ctx->overall));
    den2 = rocke_b_fadd(b, den, rocke_b_fmul(b, lv, factor));

    out_idx = rocke__out_offset(ctx, ctx->q_tok, ctx->q_head, seg, ctx->dim);
    ov = rocke_b_global_load_f32(b, ctx->segm_output, out_idx, 0);
    acc2 = rocke_b_fadd(b, acc, rocke_b_fmul(b, ov, factor));

    yields[0] = den2;
    yields[1] = acc2;
    rocke_b_scf_yield(b, yields, 2);

    rocke_b_region_leave(b);

    if(red.op != NULL && red.op->num_results >= 2)
    {
        ctx->red_den = red.op->results[0];
        ctx->red_acc = red.op->results[1];
    }
}

/* ============================================================ *
 * Reduce guarded epilogue (Python lines 3004-3008).
 *
 *   result = fmul(red.results[1], rcp(red.results[0]))
 *   cast = cast_f32_to(result, dtype)
 *   out_idx, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=dim)
 *   with b.scf_if(active):
 *       b.global_store(out, out_idx, cast, align=2)
 * ============================================================ */
void rocke_attn_unified_emit_reduce_epilogue(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t *result, *cast, *out_idx;
    rocke_if_t guard;

    result = rocke_b_fmul(b, ctx->red_acc, rocke_b_rcp(b, ctx->red_den));
    cast = rocke_b_cast_f32_to(b, result, ctx->dtype);

    out_idx = rocke__q_offset(ctx, ctx->q_tok, ctx->q_head, ctx->dim);

    guard = rocke_b_scf_if(b, ctx->active);
    rocke_b_region_enter(b, guard.then_region);
    {
        rocke_b_global_store(b, ctx->output, out_idx, cast, 2);
    }
    rocke_b_region_leave(b);
}
