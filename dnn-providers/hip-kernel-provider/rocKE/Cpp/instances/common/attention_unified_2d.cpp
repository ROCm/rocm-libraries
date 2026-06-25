// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_attention_unified_attention_unified_2d.c --
 *   C99 port of the 2D SCALAR online-softmax kernel BODY bucket of
 *   ck_dsl/instances/common/attention_unified.py:build_unified_attention_2d
 *   (lines 2589-2774).
 *
 * SCOPE (this translation unit).
 *   ckc_attn_unified_emit_2d_softmax_loop  -- the scf.for_iter over [0,kv_len)
 *     with per-iter physical/token resolve, the INLINE vec8 QK dot product +
 *     scalar tail fold, score*scale*rcp_ln2, optional softcap, causal +
 *     sliding-window mask, online (m,l,acc) update with the -inf masked-row
 *     guard select, and the inline single-V-element fold via kv_desc.offset.
 *   ckc_attn_unified_emit_2d_epilogue      -- out_val = acc*rcp(l), cast to
 *     dtype, guarded store at q_desc.offset under (active && dim < head_size).
 *
 *   apply_softcap_log2 (helpers/attention.py) is not yet exported by a helper
 *   header, so it is ported inline here as the static ckc__apply_softcap.
 *
 * The builder-call sequence is byte-identical to the Python body: same ops, same
 * order, same operands. The prologue (grid ids, seq-idx scan, geometry, SSA
 * constants, descriptors, loop init) is populated by the glue driver into the
 * shared ctx before these phase functions run; this TU only reads those ctx
 * fields and emits the loop body + epilogue.
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#include "ckc/instance_attention_unified_internal.h"

/* --------------------------------------------------------------------------
 * apply_softcap_log2 (= Python _apply_softcap), ported inline.
 *
 *   Sdiv = score_log2 / softcap
 *   p1   = exp2(Sdiv)
 *   p2   = exp2(-Sdiv)
 *   out  = softcap * ((p1 - p2) * rcp(p1 + p2))
 *
 * Returns the natural-domain softcapped value.
 * -------------------------------------------------------------------------- */
static ckc_value_t*
    ckc__apply_softcap(ckc_ir_builder_t* b, ckc_value_t* score_log2, ckc_value_t* softcap)
{
    ckc_value_t* sdiv = ckc_b_fdiv(b, score_log2, softcap);
    ckc_value_t* p1 = ckc_b_exp2(b, sdiv);
    ckc_value_t* p2 = ckc_b_exp2(b, ckc_b_fneg(b, sdiv));
    /* Python evaluates fmul operands left-to-right: fsub(p1,p2) is emitted
     * before rcp(fadd(p1,p2)), so the fsub takes the lower SSA number. C
     * leaves sibling argument evaluation unspecified (right-to-left here),
     * which would emit the rcp first and renumber both. Sequence them. */
    ckc_value_t* num = ckc_b_fsub(b, p1, p2);
    ckc_value_t* den = ckc_b_rcp(b, ckc_b_fadd(b, p1, p2));
    return ckc_b_fmul(b, softcap, ckc_b_fmul(b, num, den));
}

/* Convenience: q_desc.offset(b, token=..., head=..., dim=...) -> offset only.
 * The Python discards the validity (`q_off, _ = q_desc.offset(...)`). */
static ckc_value_t* ckc__q_offset(ckc_attn_unified_build_ctx_t* ctx,
                                  ckc_value_t* token,
                                  ckc_value_t* head,
                                  ckc_value_t* dim)
{
    static const char* names[3] = {"token", "head", "dim"};
    ckc_value_t* in_vals[3];
    ckc_value_t* off = NULL;
    ckc_value_t* vld = NULL;
    in_vals[0] = token;
    in_vals[1] = head;
    in_vals[2] = dim;
    (void)ckc_transforms_descriptor_offset(ctx->b, ctx->q_desc, names, in_vals, 3, &off, &vld);
    return off;
}

/* ==========================================================================
 * ckc_attn_unified_emit_2d_softmax_loop
 *
 * Python: build_unified_attention_2d lines 2667-2766.
 * ========================================================================== */
void ckc_attn_unified_emit_2d_softmax_loop(ckc_attn_unified_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_unified_attention_problem_t* p = ctx->p;
    const ckc_type_t* dtype = ctx->dtype;
    const int VEC = 8;
    int d8;
    int d;

    /* loop = b.scf_for_iter(0, kv_len, 1,
     *           [("m", init_m), ("l", init_l), ("acc", init_acc)], iv_name="kpos") */
    ckc_iter_arg_t iter_args[3];
    iter_args[0].name = "m";
    iter_args[0].init = ctx->init_m;
    iter_args[1].name = "l";
    iter_args[1].init = ctx->init_l;
    iter_args[2].name = "acc";
    iter_args[2].init = ctx->init_acc;

    ckc_value_t* kpos_lb = ckc_b_const_i32(b, 0);
    ckc_value_t* kpos_step = ckc_b_const_i32(b, 1);
    ckc_for_t loop = ckc_b_scf_for_iter(b,
                                        kpos_lb,
                                        ctx->kv_len,
                                        kpos_step,
                                        iter_args,
                                        3,
                                        "kpos",
                                        /*unroll=*/false,
                                        /*elide_trailing_barrier=*/true);

    ckc_b_region_enter(b, loop.body);
    {
        ckc_value_t* kpos = loop.iv;
        ckc_value_t* m_val = loop.iter_vars[0];
        ckc_value_t* l_val = loop.iter_vars[1];
        ckc_value_t* acc_val = loop.iter_vars[2];

        /* block_idx, token_in_block = _magic_div_mod(b, kpos, p.block_size) */
        ckc_value_t* block_idx = NULL;
        ckc_value_t* token_in_block = NULL;
        (void)ckc_unified_attn_magic_div_mod(b, kpos, p->block_size, &block_idx, &token_in_block);

        /* physical = global_load_i32(block_tables,
         *     seq_idx * ceil(max_seqlen_k/block_size) + block_idx) */
        int max_blocks_2d = (p->max_seqlen_k + p->block_size - 1) / p->block_size;
        ckc_value_t* physical = ckc_b_global_load_i32(
            b,
            ctx->abi_block_tables,
            ckc_b_add(b, ckc_b_mul(b, ctx->seq_idx, ckc_b_const_i32(b, max_blocks_2d)), block_idx),
            /*align=*/4);

        /* Vectorised QK dot product (vec8 main fold + scalar tail). */
        ckc_value_t* score = ctx->zero_f;
        ckc_value_t* q_off_base
            = ckc__q_offset(ctx, ctx->q_tok, ctx->q_head, ckc_b_const_i32(b, 0));
        ckc_value_t* k_off_base = ckc_unified_attn_paged_kv_offset(
            b, &ctx->kv_desc, physical, token_in_block, ctx->kv_head, ckc_b_const_i32(b, 0));

        for(d8 = 0; d8 < p->head_size / VEC; ++d8)
        {
            ckc_value_t* d_base = ckc_b_const_i32(b, (int64_t)d8 * VEC);
            ckc_value_t* qv_vec = ckc_b_global_load_vN(
                b, ctx->abi_query, ckc_b_add(b, q_off_base, d_base), dtype, VEC, /*align=*/16);
            ckc_value_t* kv_vec = ckc_b_global_load_vN(
                b, ctx->abi_key, ckc_b_add(b, k_off_base, d_base), dtype, VEC, /*align=*/16);
            int i;
            for(i = 0; i < VEC; ++i)
            {
                /* Python evaluates the fmul operands left-to-right: the Q
                 * extractelement/cast is emitted before the K one, so it takes
                 * the lower SSA number. C leaves sibling argument evaluation
                 * order unspecified (right-to-left on this toolchain), which
                 * would emit K first and swap the operands. Force the Python
                 * order with explicit sequenced locals. */
                ckc_value_t* qv_f = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, qv_vec, i));
                ckc_value_t* kv_f = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, kv_vec, i));
                score = ckc_b_fadd(b, score, ckc_b_fmul(b, qv_f, kv_f));
            }
        }
        /* Defensive scalar tail for head_size % 8 != 0 (empty in production). */
        for(d = (p->head_size / VEC) * VEC; d < p->head_size; ++d)
        {
            ckc_value_t* d_v = ckc_b_const_i32(b, d);
            ckc_value_t* q_off = ckc__q_offset(ctx, ctx->q_tok, ctx->q_head, d_v);
            ckc_value_t* k_off = ckc_unified_attn_paged_kv_offset(
                b, &ctx->kv_desc, physical, token_in_block, ctx->kv_head, d_v);
            ckc_value_t* qv_s = ckc_b_cast_to_f32(
                b, ckc_b_global_load(b, ctx->abi_query, q_off, dtype, /*align=*/2));
            ckc_value_t* kv_s = ckc_b_cast_to_f32(
                b, ckc_b_global_load(b, ctx->abi_key, k_off, dtype, /*align=*/2));
            score = ckc_b_fadd(b, score, ckc_b_fmul(b, qv_s, kv_s));
        }

        /* score = score * scale * rcp_ln2 */
        score = ckc_b_fmul(b, ckc_b_fmul(b, score, ctx->abi_scale), ctx->rcp_ln2);
        if(p->softcap > 0)
        {
            score = ckc_b_fmul(b, ckc__apply_softcap(b, score, ctx->softcap), ctx->rcp_ln2);
        }

        /* causal_ok = kpos <= context_len + query_pos */
        ckc_value_t* causal_ok
            = ckc_b_cmp_le(b, kpos, ckc_b_add(b, ctx->context_len, ctx->query_pos));
        if(p->sliding_window > 0)
        {
            ckc_value_t* dist = ckc_b_sub(b, ckc_b_add(b, ctx->context_len, ctx->query_pos), kpos);
            ckc_value_t* sw_ok = ckc_b_cmp_lt(b, dist, ckc_b_const_i32(b, p->sliding_window));
            causal_ok = ckc_b_land(b, causal_ok, sw_ok);
        }
        score = ckc_b_select(b, causal_ok, score, ctx->neg_inf);

        ckc_value_t* new_m_raw = ckc_b_fmax(b, m_val, score);
        /* masked-row guard: force m to 0 when fully masked (m_j > -inf ? m_j : 0). */
        ckc_value_t* is_finite = ckc_b_fcmp(b, "ogt", new_m_raw, ctx->neg_inf);
        ckc_value_t* new_m = ckc_b_select(b, is_finite, new_m_raw, ctx->zero_f);
        ckc_value_t* alpha = ckc_b_exp2(b, ckc_b_fsub(b, m_val, new_m));
        ckc_value_t* prob = ckc_b_exp2(b, ckc_b_fsub(b, score, new_m));
        ckc_value_t* new_l = ckc_b_fadd(b, ckc_b_fmul(b, l_val, alpha), prob);

        /* single V element fold via kv_desc.offset(..., dim=dim) */
        ckc_value_t* v_off = ckc_unified_attn_paged_kv_offset(
            b, &ctx->kv_desc, physical, token_in_block, ctx->kv_head, ctx->dim);
        ckc_value_t* vv
            = ckc_b_cast_to_f32(b, ckc_b_global_load(b, ctx->abi_value, v_off, dtype, /*align=*/2));
        /* Left-to-right operand order to match Python: fmul(acc,alpha) before
         * fmul(prob,vv). Sequence the two nested products explicitly. */
        ckc_value_t* acc_scaled = ckc_b_fmul(b, acc_val, alpha);
        ckc_value_t* prob_v = ckc_b_fmul(b, prob, vv);
        ckc_value_t* new_acc = ckc_b_fadd(b, acc_scaled, prob_v);

        ckc_value_t* yield_vals[3];
        yield_vals[0] = new_m;
        yield_vals[1] = new_l;
        yield_vals[2] = new_acc;
        ckc_b_scf_yield(b, yield_vals, 3);
    }
    ckc_b_region_leave(b);

    /* Stash loop results (m, l, acc) for the epilogue. */
    ctx->loop_m = loop.op->results[0];
    ctx->loop_l = loop.op->results[1];
    ctx->loop_acc = loop.op->results[2];
}

/* ==========================================================================
 * ckc_attn_unified_emit_2d_epilogue
 *
 * Python: build_unified_attention_2d lines 2768-2773.
 * ========================================================================== */
void ckc_attn_unified_emit_2d_epilogue(ckc_attn_unified_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_unified_attention_problem_t* p = ctx->p;

    /* out_val = loop.results[2] * rcp(loop.results[1]) */
    ckc_value_t* out_val = ckc_b_fmul(b, ctx->loop_acc, ckc_b_rcp(b, ctx->loop_l));
    ckc_value_t* out_cast = ckc_b_cast_f32_to(b, out_val, ctx->dtype);

    /* out_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=dim) */
    ckc_value_t* out_off = ckc__q_offset(ctx, ctx->q_tok, ctx->q_head, ctx->dim);

    /* valid = active && (dim < head_size) */
    ckc_value_t* valid
        = ckc_b_land(b, ctx->active, ckc_b_cmp_lt(b, ctx->dim, ckc_b_const_i32(b, p->head_size)));

    ckc_if_t guard = ckc_b_scf_if(b, valid);
    ckc_b_region_enter(b, guard.then_region);
    {
        ckc_b_global_store(b, ctx->output, out_off, out_cast, /*align=*/2);
    }
    ckc_b_region_leave(b);
}
