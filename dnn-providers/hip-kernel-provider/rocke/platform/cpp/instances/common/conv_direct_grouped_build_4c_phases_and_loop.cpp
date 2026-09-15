// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_build_4c_phases_and_loop.c -- the four 4c
 * IR-emitting phase functions of the C99 port of build_direct_conv_4c
 * (rocke/instances/common/conv_direct_grouped.py, lines 833-1033).
 *
 * SCOPE (this TU):
 *   rocke_dconv4c_prologue          Python lines 833-876 (validate / spec gate,
 *                                 param decls, SSA constants, thread/wave/lane
 *                                 + grid/group decode, buffer rsrcs, the two
 *                                 register-zero vectors).
 *   rocke_dconv4c_load_weights      lines 878-901 (b_desc, k_out_val, the KH*KW
 *                                 per-lane weight loads).
 *   rocke_dconv4c_build_descriptors lines 903-965 (a_desc + 2 embeds, d_desc,
 *                                 acc_tiles zero seed, c_val_groupc, s_consts).
 *   rocke_dconv4c_stream_h_loop     lines 967-1033 (the unrolled H-row loop:
 *                                 OOB-safe A loads, the 4x4x4 MFMA chain into
 *                                 the circular acc slot, the conditional flush
 *                                 to D and the unconditional slot reset).
 *
 * The 4c builder is self-contained: no named closures. The Python prologue's
 * shared locals live in rocke_dconv_4c_ctx_t (see the internal header); the driver
 * (a peer TU) populates ctx->b/spec/arch/p and calls these in Python order.
 *
 * Builder-call sequence is byte-identical to the Python so the emitted IR op
 * stream matches exactly.
 */
#include "rocke/instance_conv_direct_grouped_internal.h"

#include <stdio.h> /* snprintf (error messages) */

/* ===================================================================== *
 *  rocke_dconv4c_prologue -- Python lines 833-876.
 *
 *  NOTE on line 838 (`b = IRBuilder(spec.kernel_name())`): in the C port the
 *  builder is created/initialised by the public entry (rocke_build_direct_conv_4c)
 *  before any phase runs, exactly as the public header documents ("Does NOT
 *  re-init the builder"). So this phase does NOT construct the builder; it only
 *  sets the kernel attr (line 839) and proceeds with param decls onward.
 * ===================================================================== */
bool rocke_dconv4c_prologue(rocke_dconv_4c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_4c_spec_t* spec = ctx->spec;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    char reason[ROCKE_ERR_MSG_CAP];
    rocke_status_t vst;
    bool ok;

    /* Line 833: spec.validate(). */
    vst = rocke_direct_conv_4c_validate(spec, reason, sizeof reason);
    if(vst != ROCKE_OK)
    {
        snprintf(b->err, sizeof b->err, "%s", reason);
        b->status = vst;
        return false;
    }

    /* Lines 834-836: is_valid_spec_4c(spec, arch); raise on reject. */
    ok = rocke_direct_conv_4c_is_valid_spec(spec, ctx->arch, reason, sizeof reason);
    if(!ok)
    {
        ROCKE_ERR_SNPRINTF(b->err,
                           sizeof b->err,
                           "invalid direct_conv_4c spec for %s: %s",
                           ctx->arch ? ctx->arch : "gfx950",
                           reason);
        b->status = ROCKE_ERR_VALUE;
        return false;
    }

    /* Line 837: p = spec.problem (already copied into ctx->p by the driver). */

    /* Line 839: b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block. */
    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", rocke_direct_conv_4c_threads_per_block(spec));

    /* Lines 841-846: kernel params. */
    {
        rocke_param_opts_t po;
        const rocke_type_t* f16_global = rocke_ptr_type(b, rocke_f16(), "global");

        /* A: noalias, readonly, align 16. */
        po = (rocke_param_opts_t){0};
        po.noalias = true;
        po.noalias_set = true;
        po.readonly = true;
        po.readonly_set = true;
        po.align = 16;
        po.align_set = true;
        ctx->A = rocke_b_param(b, "A", f16_global, &po);

        /* B: noalias, readonly, align 16. */
        po = (rocke_param_opts_t){0};
        po.noalias = true;
        po.noalias_set = true;
        po.readonly = true;
        po.readonly_set = true;
        po.align = 16;
        po.align_set = true;
        ctx->Bp = rocke_b_param(b, "B", f16_global, &po);

        /* D: noalias, writeonly, align 16. */
        po = (rocke_param_opts_t){0};
        po.noalias = true;
        po.noalias_set = true;
        po.writeonly = true;
        po.writeonly_set = true;
        po.align = 16;
        po.align_set = true;
        ctx->D = rocke_b_param(b, "D", f16_global, &po);

        ctx->A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), NULL);
        ctx->B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), NULL);
        ctx->D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), NULL);
    }

    /* Lines 848-857: common SSA constants. */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_W = rocke_b_const_i32(b, p->W);
    ctx->c_cpg = rocke_b_const_i32(b, p->cpg);
    ctx->c_kpg = rocke_b_const_i32(b, p->kpg);
    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);

    /* Lines 859-863: thread/wave/lane decode. */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, rocke_b_const_i32(b, spec->wave_size));
    ctx->lane = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, spec->wave_size));
    ctx->batch = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 4));
    ctx->lane_q = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 4));

    /* Lines 865-870: grid/group decode. */
    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, rocke_b_const_i32(b, spec->block_q));
    ctx->group_in_wg
        = rocke_b_add(b, rocke_b_mul(b, ctx->wave_id, rocke_b_const_i32(b, 16)), ctx->batch);
    ctx->g = rocke_b_add(
        b, rocke_b_mul(b, ctx->by, rocke_b_const_i32(b, spec->block_groups)), ctx->group_in_wg);

    /* Lines 872-876: buffer rsrcs + register-zero vectors. */
    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);
    ctx->fp16x4_zero = rocke_b_zero_vec_f16(b, 4);
    ctx->zero_acc = rocke_b_zero_vec_f32(b, 4);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  rocke_dconv4c_load_weights -- Python lines 878-901.
 *
 *  Weights: per (r, s), per lane: B[g*kpg + lane_q, r, s, 0:4].
 * ===================================================================== */
void rocke_dconv4c_load_weights(rocke_dconv_4c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int r_const, s_const;

    /* Lines 883-887: b_desc = TensorDescriptor.naive("B", ...). */
    {
        int lengths[4];
        static const char* const coord_names[4] = {"k_out", "r", "s", "c"};
        lengths[0] = rocke_direct_conv_problem_total_k(p);
        lengths[1] = p->KH;
        lengths[2] = p->KW;
        lengths[3] = p->cpg;
        ctx->b_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, coord_names, 4);
    }

    /* Line 888: k_out_val = b.add(b.mul(g, c_kpg), lane_q). */
    ctx->k_out_val = rocke_b_add(b, rocke_b_mul(b, ctx->g, ctx->c_kpg), ctx->lane_q);

    /* Lines 889-901: per (r, s) weight loads. */
    ctx->n_weights = 0;
    for(r_const = 0; r_const < p->KH; ++r_const)
    {
        for(s_const = 0; s_const < p->KW; ++s_const)
        {
            const char* in_names[4] = {"k_out", "r", "s", "c"};
            rocke_value_t* in_values[4];
            rocke_value_t* w_off = NULL;
            rocke_value_t* w_valid = NULL;
            rocke_value_t* w;

            in_values[0] = ctx->k_out_val;
            in_values[1] = rocke_b_const_i32(b, r_const);
            in_values[2] = rocke_b_const_i32(b, s_const);
            in_values[3] = ctx->c0;

            /* b_desc.offset(b, k_out=..., r=..., s=..., c=c0). */
            rocke_transforms_descriptor_offset(
                b, ctx->b_desc, in_names, in_values, 4, &w_off, &w_valid);

            /* b.buffer_load_vN_f16(b_rsrc, b.mul(w_off, c_half_bytes), c0, 2). */
            w = rocke_b_buffer_load_vN_f16(
                b, ctx->b_rsrc, rocke_b_mul(b, w_off, ctx->c_half_bytes), ctx->c0, 2);
            ctx->weights[ctx->n_weights++] = w;
        }
    }
}

/* ===================================================================== *
 *  rocke_dconv4c_build_descriptors -- Python lines 903-965.
 *
 *  a_desc (naive + 2 embeds), d_desc (naive), acc_tiles zero seed,
 *  c_val_groupc, s_consts. Also derives q_tiles_per_wave / n_iters.
 * ===================================================================== */
void rocke_dconv4c_build_descriptors(rocke_dconv_4c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int qt, slot, s;

    /* Line 903: q_tiles_per_wave = spec.block_q // 4. */
    ctx->q_tiles_per_wave = ctx->spec->block_q / 4;

    /* Lines 904-906: acc_tiles[qt] = [zero_acc, zero_acc, zero_acc]. The Python
     * seeds exactly p.KH (=3) slots per qt; the literal triple is KH-wide. */
    for(qt = 0; qt < ctx->q_tiles_per_wave; ++qt)
    {
        for(slot = 0; slot < p->KH; ++slot)
        {
            ctx->acc_tiles[qt][slot] = ctx->zero_acc;
        }
    }

    /* Line 907: n_iters = p.H + p.KH - 1. */
    ctx->n_iters = p->H + p->KH - 1;

    /* Lines 925-946: a_desc = naive("A", ...).transform(embed, embed). */
    {
        int lengths[4];
        static const char* const coord_names[4] = {"n", "h", "w", "c"};
        const rocke_tensor_descriptor_t* a_naive;
        const rocke_transform_t* xforms[2];
        static const char* const up_h[1] = {"y_iter"};
        static const char* const up_w[2] = {"wo", "s"};
        int strides_h[1] = {1};
        int strides_w[2] = {1, 1};

        lengths[0] = p->N;
        lengths[1] = p->H;
        lengths[2] = p->W;
        lengths[3] = rocke_direct_conv_problem_total_c(p);
        a_naive = rocke_tensor_descriptor_naive(b, "A", lengths, 4, NULL, coord_names, 4);

        /* embed(upper=("y_iter",), into="h", strides=(1,), offset=-PAD,
         *       lo=0, hi=H). */
        xforms[0] = rocke_embed_bounded(b, up_h, 1, "h", strides_h, -p->PAD, 0, p->H);
        /* embed(upper=("wo","s"), into="w", strides=(1,1), offset=-PAD,
         *       lo=0, hi=W). */
        xforms[1] = rocke_embed_bounded(b, up_w, 2, "w", strides_w, -p->PAD, 0, p->W);

        ctx->a_desc = rocke_tensor_descriptor_transform(b, a_naive, xforms, 2);
    }

    /* Lines 953-957: d_desc = naive("D", [N,H,W,total_k], ...). */
    {
        int lengths[4];
        static const char* const coord_names[4] = {"n", "h", "w", "k"};
        lengths[0] = p->N;
        lengths[1] = p->H;
        lengths[2] = p->W;
        lengths[3] = rocke_direct_conv_problem_total_k(p);
        ctx->d_desc = rocke_tensor_descriptor_naive(b, "D", lengths, 4, NULL, coord_names, 4);
    }

    /* Line 959: c_val_groupc = b.mul(g, c_cpg). */
    ctx->c_val_groupc = rocke_b_mul(b, ctx->g, ctx->c_cpg);

    /* Line 965: s_consts = [b.const_i32(s) for s in range(p.KW)]. */
    ctx->n_s_consts = 0;
    for(s = 0; s < p->KW; ++s)
    {
        ctx->s_consts[ctx->n_s_consts++] = rocke_b_const_i32(b, s);
    }
}

/* ===================================================================== *
 *  rocke_dconv4c_stream_h_loop -- Python lines 967-1033.
 *
 *  The unrolled H-row loop. For each of n_iters rows: gather per-(qt,s) OOB-safe
 *  A inputs, run the per-(qt,r,s) 4x4x4 MFMA chain into the circular acc slot,
 *  conditionally flush the oldest slot to D, then unconditionally reset it.
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv4c_stream_h_loop(rocke_dconv_4c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int y, qt, s_idx, r_const, s_const;

    for(y = 0; y < ctx->n_iters; ++y)
    {
        /* Line 968: y_iter = b.const_i32(y). */
        rocke_value_t* y_iter = rocke_b_const_i32(b, y);

        /* inputs_by_qtile[qt][s]; q_tiles_per_wave x KW. */
        rocke_value_t* inputs_by_qtile[ROCKE_DCONV_MAX_QTILES][16];
        int p_flush, P_FLUSH;

        /* Lines 970-988: gather A inputs per (qt, s). */
        for(qt = 0; qt < ctx->q_tiles_per_wave; ++qt)
        {
            /* Line 972-973: q_base = q_tile_start + qt*4; q_pos = q_base+lane_q. */
            rocke_value_t* q_base = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 4));
            rocke_value_t* q_pos = rocke_b_add(b, q_base, ctx->lane_q);

            for(s_idx = 0; s_idx < ctx->n_s_consts; ++s_idx)
            {
                rocke_value_t* s_val = ctx->s_consts[s_idx];
                const char* in_names[5] = {"n", "y_iter", "wo", "s", "c"};
                rocke_value_t* in_values[5];
                rocke_value_t* a_off = NULL;
                rocke_value_t* valid = NULL;
                rocke_value_t* safe_a;
                rocke_value_t* vec;

                in_values[0] = ctx->n;
                in_values[1] = y_iter;
                in_values[2] = q_pos;
                in_values[3] = s_val;
                in_values[4] = ctx->c_val_groupc;

                /* Lines 976-983: a_desc.offset(...). */
                rocke_transforms_descriptor_offset(
                    b, ctx->a_desc, in_names, in_values, 5, &a_off, &valid);

                /* Line 984: safe_a = select(valid, a_off*2, oob_sentinel). */
                safe_a = rocke_b_select(
                    b, valid, rocke_b_mul(b, a_off, ctx->c_half_bytes), ctx->oob_sentinel);
                /* Line 985: vec = buffer_load_vN_f16(a_rsrc, safe_a, c0, 2). */
                vec = rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_a, ctx->c0, 2);
                /* Line 986: vec = select(valid, vec, fp16x4_zero). */
                vec = rocke_b_select(b, valid, vec, ctx->fp16x4_zero);
                inputs_by_qtile[qt][s_idx] = vec;
            }
        }

        /* Lines 990-1000: the per-(qt, r, s) 4x4x4 MFMA chain. */
        for(qt = 0; qt < ctx->q_tiles_per_wave; ++qt)
        {
            rocke_value_t** accs = ctx->acc_tiles[qt];
            rocke_value_t** inputs = inputs_by_qtile[qt];
            for(r_const = 0; r_const < p->KH; ++r_const)
            {
                /* p_idx = (y - r_const) % p.KH (Python floor-mod; y,r_const>=0
                 * and r_const < KH so (y - r_const) % KH matches C for the
                 * non-negative case; when y < r_const the dividend is negative
                 * and Python floor-mod differs from C truncation, so normalise). */
                int p_idx = ((y - r_const) % p->KH + p->KH) % p->KH;
                rocke_value_t* acc = accs[p_idx];
                for(s_const = 0; s_const < p->KW; ++s_const)
                {
                    acc = rocke_b_mfma_f32_4x4x4_f16(
                        b, ctx->weights[r_const * p->KW + s_const], inputs[s_const], acc);
                }
                accs[p_idx] = acc;
            }
        }

        /* Lines 1002-1003: p_flush = y - (KH-1); P_FLUSH = p_flush % KH. */
        p_flush = y - (p->KH - 1);
        P_FLUSH = ((p_flush % p->KH) + p->KH) % p->KH;

        /* Lines 1004-1029: flush the oldest slot to D when in range. */
        if(0 <= p_flush && p_flush < p->H)
        {
            /* Line 1007: k_out_base = b.mul(g, c_kpg). */
            rocke_value_t* k_out_base = rocke_b_mul(b, ctx->g, ctx->c_kpg);
            for(qt = 0; qt < ctx->q_tiles_per_wave; ++qt)
            {
                rocke_value_t* acc = ctx->acc_tiles[qt][P_FLUSH];
                rocke_value_t* q_base
                    = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 4));
                rocke_value_t* out_q = rocke_b_add(b, q_base, ctx->lane_q);
                rocke_value_t* out_q_ok = rocke_b_cmp_lt(b, out_q, ctx->c_W);
                const char* in_names[4] = {"n", "h", "w", "k"};
                rocke_value_t* in_values[4];
                rocke_value_t* d_base = NULL;
                rocke_value_t* d_valid = NULL;
                rocke_value_t* safe_d;
                rocke_value_t* acc_h;

                in_values[0] = ctx->n;
                in_values[1] = rocke_b_const_i32(b, p_flush);
                in_values[2] = out_q;
                in_values[3] = k_out_base;

                /* Lines 1013-1019: d_desc.offset(n=, h=, w=, k=). */
                rocke_transforms_descriptor_offset(
                    b, ctx->d_desc, in_names, in_values, 4, &d_base, &d_valid);

                /* Line 1020: safe_d = select(out_q_ok, d_base*2, oob_sentinel). */
                safe_d = rocke_b_select(
                    b, out_q_ok, rocke_b_mul(b, d_base, ctx->c_half_bytes), ctx->oob_sentinel);
                /* Line 1028: acc_h = vec_trunc_f32_to_f16(acc). */
                acc_h = rocke_b_vec_trunc_f32_to_f16(b, acc);
                /* Line 1029: buffer_store_vN_f16(d_rsrc, safe_d, c0, acc_h, 2). */
                rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d, ctx->c0, acc_h, 2);
            }
        }

        /* Lines 1030-1031: reset the flushed slot to zero_acc. */
        for(qt = 0; qt < ctx->q_tiles_per_wave; ++qt)
        {
            ctx->acc_tiles[qt][P_FLUSH] = ctx->zero_acc;
        }
    }

    /* Line 1033: return b.kernel. */
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}
