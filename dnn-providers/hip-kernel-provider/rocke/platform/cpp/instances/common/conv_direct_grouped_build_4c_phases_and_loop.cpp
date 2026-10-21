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

#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */

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

    /* Lines 841-846: kernel params (AOT kernarg block, conv_abi order). */
    rocke_dconv_emit_params(
        b, &ctx->params, "fwd", rocke_b_io_ir_type(b, ctx->p.dtype ? ctx->p.dtype : "fp16"));
    ctx->A = ctx->params.A;
    ctx->Bp = ctx->params.Bp;
    ctx->D = ctx->params.D;
    ctx->A_bytes = ctx->params.A_bytes;
    ctx->B_bytes = ctx->params.B_bytes;
    ctx->D_bytes = ctx->params.D_bytes;

    /* Lines 848-857: common SSA constants. */
    ctx->c0 = rocke_b_const_i32(b, 0);
    /* AOT: the store guard bounds against the runtime output width. */
    ctx->c_W = ctx->params.p_Wo;
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
    ctx->io_vec4_zero = rocke_b_zero_vec_f16(b, 4);
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

    /* Lines 925-946: a_desc with runtime extents. Only PAD and stride stay
     * build-time -- they shape the filter-tap offsets, so they are kernel
     * capabilities rather than shape. */
    {
        rocke_dynamic_tensor_descriptor_t* a_dyn
            = rocke_dconv_a_descriptor_dynamic(b, &ctx->params, p->PAD, p->stride, "wo", "s");
        if(!a_dyn)
            return;
        ctx->a_desc = &a_dyn->base;
    }

    /* Output descriptor D[N, Ho, Wo, total_k] in NHWK. */
    {
        rocke_dynamic_tensor_descriptor_t* d_dyn
            = rocke_dconv_d_descriptor_dynamic(b, &ctx->params);
        if(!d_dyn)
            return;
        ctx->d_desc = &d_dyn->base;
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
    int j, qt, s_idx, r_const, s_const;
    int KH = p->KH;
    int q_tiles = ctx->q_tiles_per_wave;
    int num_accs = q_tiles * KH;

    rocke_iter_arg_t iter_args[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS];
    char acc_names[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS][32];
    rocke_value_t* accs_flat[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS];
    rocke_for_t for_op;
    rocke_value_t* y_base;
    int UNROLL;
    rocke_value_t* c_unroll;
    rocke_value_t* n_iters_v;

    /* AOT: Hi is a kernarg, so the trip count is runtime. The only
     * build-time periodicity in this variant is the accumulator slot rotation
     * (period KH) -- 4c reads straight from DRAM, with no LDS ping-pong -- so
     * the body is unrolled KH times. The step is then a multiple of KH, which
     * keeps the loop variable at 0 mod KH and every slot index a constant.
     *
     * The trip count rounds up; rows past the image zero-fill through the A
     * descriptor's 0 <= h < Hi bound and the flush predicate drops them. */
    UNROLL = rocke_dconv_row_loop_unroll(KH, /*lds_ping_pong=*/false);
    c_unroll = rocke_b_const_i32(b, UNROLL);
    n_iters_v = rocke_b_add(b, ctx->params.p_Hi, rocke_b_const_i32(b, KH - 1));

    if(num_accs > (int)(sizeof(iter_args) / sizeof(iter_args[0])))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE; /* too many accumulator slots */
        return NULL;
    }

    for(qt = 0; qt < q_tiles; ++qt)
    {
        int pi;
        for(pi = 0; pi < KH; ++pi)
        {
            int idx = qt * KH + pi;
            snprintf(acc_names[idx], sizeof(acc_names[0]), "acc_q%d_p%d", qt, pi);
            iter_args[idx].name = acc_names[idx];
            iter_args[idx].init = ctx->zero_acc;
        }
    }

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->c0,
                                  n_iters_v,
                                  c_unroll,
                                  iter_args,
                                  num_accs,
                                  "y_row",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/false);
    y_base = for_op.iv;
    {
        int i;
        for(i = 0; i < for_op.num_iter_vars; ++i)
            accs_flat[i] = for_op.iter_vars[i];
    }

    rocke_b_region_enter(b, for_op.body);
    for(j = 0; j < UNROLL; ++j)
    {
        rocke_value_t* y_iter = rocke_b_add(b, y_base, rocke_b_const_i32(b, j));
        rocke_value_t* inputs_by_qtile[ROCKE_DCONV_MAX_QTILES][16];
        int P_FLUSH;
        rocke_value_t* p_flush_v;
        rocke_value_t* row_ok;
        rocke_value_t* ho_row_v;
        rocke_value_t* k_out_base;

        /* Gather A inputs per (qt, s). */
        for(qt = 0; qt < q_tiles; ++qt)
        {
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

                if(!rocke_transforms_descriptor_offset(
                       b, ctx->a_desc, in_names, in_values, 5, &a_off, &valid))
                {
                    return NULL;
                }
                safe_a = rocke_b_select(
                    b, valid, rocke_b_mul(b, a_off, ctx->c_half_bytes), ctx->oob_sentinel);
                vec = rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_a, ctx->c0, 2);
                vec = rocke_b_select(b, valid, vec, ctx->io_vec4_zero);
                inputs_by_qtile[qt][s_idx] = vec;
            }
        }

        /* The per-(qt, r, s) 4x4x4 MFMA chain. */
        for(qt = 0; qt < q_tiles; ++qt)
        {
            rocke_value_t** inputs = inputs_by_qtile[qt];
            for(r_const = 0; r_const < KH; ++r_const)
            {
                /* y_base is 0 mod KH, so the rotating slot collapses to j. */
                int p_idx = (((j - r_const) % KH) + KH) % KH;
                int flat = qt * KH + p_idx;
                rocke_value_t* acc = accs_flat[flat];
                for(s_const = 0; s_const < p->KW; ++s_const)
                {
                    acc = rocke_b_mfma_f32_4x4x4_f16(
                        b, ctx->weights[r_const * p->KW + s_const], inputs[s_const], acc);
                }
                accs_flat[flat] = acc;
            }
        }

        /* Flush the slot this row completed, then ALWAYS reset it: the first
         * KH-1 rows have a negative flush index and would otherwise leak
         * their r = KH-1 term into a slot a later real output row flushes. */
        P_FLUSH = (((j - (KH - 1)) % KH) + KH) % KH;
        p_flush_v = rocke_b_add(b, y_base, rocke_b_const_i32(b, j - (KH - 1)));
        {
            rocke_value_t* ge = rocke_b_cmp_ge(b, p_flush_v, ctx->c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, p_flush_v, ctx->params.p_Hi);
            row_ok = rocke_b_land(b, ge, lt);
        }
        if(p->stride > 1)
        {
            rocke_value_t* c_stride_v = rocke_b_const_i32(b, p->stride);
            rocke_value_t* md = rocke_b_mod(b, p_flush_v, c_stride_v);
            rocke_value_t* eq = rocke_b_cmp_eq(b, md, ctx->c0);
            row_ok = rocke_b_land(b, row_ok, eq);
            ho_row_v = rocke_b_div(b, p_flush_v, c_stride_v);
        }
        else
        {
            ho_row_v = p_flush_v;
        }
        ho_row_v = rocke_b_select(b, row_ok, ho_row_v, ctx->c0);

        k_out_base = rocke_b_mul(b, ctx->g, ctx->c_kpg);
        for(qt = 0; qt < q_tiles; ++qt)
        {
            int flat = qt * KH + P_FLUSH;
            rocke_value_t* acc = accs_flat[flat];
            rocke_value_t* q_base = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 4));
            rocke_value_t* out_q = rocke_b_add(b, q_base, ctx->lane_q);
            rocke_value_t* store_ok = rocke_b_land(b, row_ok, rocke_b_cmp_lt(b, out_q, ctx->c_W));
            const char* in_names[4] = {"n", "h", "w", "k"};
            rocke_value_t* in_values[4];
            rocke_value_t* d_base = NULL;
            rocke_value_t* d_valid = NULL;
            rocke_value_t* safe_d;
            rocke_value_t* acc_h;

            in_values[0] = ctx->n;
            in_values[1] = ho_row_v;
            in_values[2] = out_q;
            in_values[3] = k_out_base;

            if(!rocke_transforms_descriptor_offset(
                   b, ctx->d_desc, in_names, in_values, 4, &d_base, &d_valid))
            {
                return NULL;
            }
            safe_d = rocke_b_select(
                b, store_ok, rocke_b_mul(b, d_base, ctx->c_half_bytes), ctx->oob_sentinel);
            /* MFMA 4x4x4 wave64 per-lane output layout:
             *   acc[i] -> D[n, ho_row, out_q, g*kpg + i]  for i in 0..3 */
            acc_h = rocke_b_vec_trunc_f32_to_f16(b, acc);
            rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d, ctx->c0, acc_h, 2);
            accs_flat[flat] = ctx->zero_acc;
        }
    }
    rocke_b_scf_yield(b, accs_flat, num_accs);
    rocke_b_region_leave(b);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}
