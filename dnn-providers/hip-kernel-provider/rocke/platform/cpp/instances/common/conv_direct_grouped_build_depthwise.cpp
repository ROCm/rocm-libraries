// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_build_depthwise.c -- C99 port of
 * build_direct_depthwise (rocke/instances/common/conv_direct_grouped.py,
 * lines 2559-2713).
 *
 * The depthwise kernel uses scalar FMA only — no MFMA, no LDS.  Each lane
 * owns one absolute channel for the duration of the kernel.
 *
 * Key differences from grouped variants:
 *   - No LDS allocation; each lane loads A scalars directly via buffer_load_f16.
 *   - Weights (KH×KW f16 per lane/channel) are preloaded into f32 registers.
 *   - Accumulator: BLOCK_W × KH scalar f32 values (not a vector per lane).
 *   - H-streaming loop: for each y, for each (r, w_out, s): one fma.
 *   - Store: scalar buffer_store_f16 per (w_out, flush_row).
 *   - No sync() calls (no shared LDS).
 *
 * Phase functions: prologue, load_weights, build_descriptors, stream_h_loop.
 * Byte-identical to the Python source.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  Prologue  (Python lines 2572-2611)
 * ===================================================================== */
bool rocke_dconv_dw_prologue(rocke_dconv_dw_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_depthwise_spec_t* spec = ctx->spec;
    char reason[ROCKE_ERR_MSG_CAP];

    if(rocke_direct_depthwise_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    if(!rocke_direct_depthwise_is_valid_spec(spec, ctx->arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->p = spec->problem;

    ctx->BLOCK_W = spec->block_w;
    ctx->BLOCK_WAVES = spec->block_waves;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_depthwise_threads_per_block(spec);
    ctx->BLOCK_CH = rocke_direct_depthwise_block_ch(spec);
    ctx->n_iters = ctx->p.H + ctx->p.KH - 1;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    {
        const rocke_type_t* f16ptr = rocke_ptr_type(b, rocke_f16(), "global");
        rocke_param_opts_t ro;
        rocke_param_opts_t wo;
        rocke_param_opts_t none;

        ro = (rocke_param_opts_t){0};
        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;
        ctx->A = rocke_b_param(b, "A", f16ptr, &ro);
        ctx->Bp = rocke_b_param(b, "B", f16ptr, &ro);

        wo = (rocke_param_opts_t){0};
        wo.noalias = true;
        wo.noalias_set = true;
        wo.writeonly = true;
        wo.writeonly_set = true;
        wo.align = 16;
        wo.align_set = true;
        ctx->D = rocke_b_param(b, "D", f16ptr, &wo);

        none = (rocke_param_opts_t){0};
        ctx->A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
        ctx->B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
        ctx->D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);
    }

    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_W = rocke_b_const_i32(b, ctx->p.W);
    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->zero_f32 = rocke_b_const_f32(b, 0.0);

    /* ---- thread / wave / lane decode (Python lines 2597-2608) ---- */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);

    /* Grid: bx=W-tile, by=channel-tile, bz=batch. */
    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, rocke_b_const_i32(b, ctx->BLOCK_W));

    /* ch = by*BLOCK_CH + wave_id*WAVE + lane
     * Python: b.add(b.mul(by, b.const_i32(BLOCK_CH)),
     *               b.add(b.mul(wave_id, c_wave), lane))
     * Force Python left-to-right SSA. */
    {
        rocke_value_t* mul_by = rocke_b_mul(b, ctx->by, rocke_b_const_i32(b, ctx->BLOCK_CH));
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_wave);
        rocke_value_t* inner = rocke_b_add(b, mul_wave, ctx->lane);
        ctx->ch = rocke_b_add(b, mul_by, inner);
    }

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Descriptor phase  (Python lines 2614-2646)
 *
 *  a_desc: A[N,H,W,total_c] with two embeds (y boundary, w boundary).
 *  b_desc: B[total_k,KH,KW,1] naive.
 *  d_desc: D[N,H,W,total_k] naive.
 *
 *  NOTE: In the Python the descriptors are built BEFORE weight loads but AFTER
 *  the prologue.  The phase ordering in the driver (load_weights before
 *  build_descriptors) matches the Python source order: weight loads reference
 *  b_desc which is built in build_descriptors, so descriptors come FIRST here.
 *  We reverse the phase call order in the driver to match:
 *    prologue → build_descriptors → load_weights → stream_h_loop.
 *  The internal header declares them separately so the driver can call them in
 *  the right order; the public glue entry calls build_descriptors before
 *  load_weights (unlike the grouped variants where it is the reverse).
 * ===================================================================== */
void rocke_dconv_dw_build_descriptors(rocke_dconv_dw_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_c = rocke_direct_conv_problem_total_c(&ctx->p);
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

    /* a_desc = naive("A", [N,H,W,total_c]) + 2 embeds (Python lines 2614-2634). */
    {
        static const char* const a_coords[4] = {"n", "h", "w", "c"};
        int a_lengths[4];
        rocke_tensor_descriptor_t* a_naive;
        const rocke_transform_t* xforms[2];

        a_lengths[0] = ctx->p.N;
        a_lengths[1] = ctx->p.H;
        a_lengths[2] = ctx->p.W;
        a_lengths[3] = total_c;
        a_naive = rocke_tensor_descriptor_naive(b, "A", a_lengths, 4, NULL, a_coords, 4);

        /* embed(upper=("y_iter",), into="h", strides=(1,), offset=-PAD, lo=0, hi=H) */
        {
            static const char* const h_upper[1] = {"y_iter"};
            int h_strides[1] = {1};
            xforms[0]
                = rocke_embed_bounded(b, h_upper, 1, "h", h_strides, -ctx->p.PAD, 0, ctx->p.H);
        }
        /* embed(upper=("wo","s_off"), into="w", strides=(1,1), offset=-PAD, lo=0, hi=W) */
        {
            static const char* const w_upper[2] = {"wo", "s_off"};
            int w_strides[2] = {1, 1};
            xforms[1]
                = rocke_embed_bounded(b, w_upper, 2, "w", w_strides, -ctx->p.PAD, 0, ctx->p.W);
        }
        ctx->a_desc = rocke_tensor_descriptor_transform(b, a_naive, xforms, 2);
    }

    /* b_desc = naive("B", [total_k, KH, KW, 1], coord_names=(k,r,s,c)) */
    {
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_lengths[4];
        b_lengths[0] = total_k;
        b_lengths[1] = ctx->p.KH;
        b_lengths[2] = ctx->p.KW;
        b_lengths[3] = 1;
        ctx->b_desc = rocke_tensor_descriptor_naive(b, "B", b_lengths, 4, NULL, b_coords, 4);
    }

    /* d_desc = naive("D", [N, H, W, total_k]) */
    {
        static const char* const d_coords[4] = {"n", "h", "w", "k"};
        int d_lengths[4];
        d_lengths[0] = ctx->p.N;
        d_lengths[1] = ctx->p.H;
        d_lengths[2] = ctx->p.W;
        d_lengths[3] = total_k;
        ctx->d_desc = rocke_tensor_descriptor_naive(b, "D", d_lengths, 4, NULL, d_coords, 4);
    }
}

/* ===================================================================== *
 *  Weight-load phase  (Python lines 2648-2662)
 *
 *  Preload KH × KW fp16 weights per lane into f32 registers.
 *  Each lane owns channel `ch`; weight[ch, r, s, 0] is a scalar.
 * ===================================================================== */
void rocke_dconv_dw_load_weights(rocke_dconv_dw_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int r_const, s_const;

    for(r_const = 0; r_const < ctx->p.KH; ++r_const)
    {
        for(s_const = 0; s_const < ctx->p.KW; ++s_const)
        {
            rocke_value_t* w_off = NULL;
            rocke_value_t* valid = NULL;
            rocke_value_t* w_h;
            const char* in_names[4] = {"k", "r", "s", "c"};
            rocke_value_t* in_values[4];

            in_values[0] = ctx->ch;
            in_values[1] = rocke_b_const_i32(b, r_const);
            in_values[2] = rocke_b_const_i32(b, s_const);
            in_values[3] = ctx->c0;
            rocke_transforms_descriptor_offset(
                b, ctx->b_desc, in_names, in_values, 4, &w_off, &valid);
            w_h = rocke_b_buffer_load_f16(
                b, ctx->b_rsrc, rocke_b_mul(b, w_off, ctx->c_half_bytes), ctx->c0);
            ctx->weights_f32[r_const][s_const] = rocke_b_cast_to_f32(b, w_h);
        }
    }
}

/* ===================================================================== *
 *  H-row streaming loop  (Python lines 2664-2712)
 *
 *  Accumulator: acc[w_out][slot] — BLOCK_W × KH scalar f32 values.
 *  For each y iteration:
 *    for each (r, w_out, s): load A scalar via a_desc, fma into acc slot.
 *    Flush: write the oldest slot for each w_out if in valid output range.
 *    Reset: acc[w_out][P_FLUSH] = 0.
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv_dw_stream_h_loop(rocke_dconv_dw_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int KW = p->KW;
    int BLOCK_W = ctx->BLOCK_W;
    int y;
    int n_iters = ctx->n_iters;
    int w_out, slot;

    /* Seed accumulators: acc[w_out][slot] = zero_f32. */
    for(w_out = 0; w_out < BLOCK_W; ++w_out)
    {
        for(slot = 0; slot < KH; ++slot)
        {
            ctx->acc[w_out][slot] = ctx->zero_f32;
        }
    }

    for(y = 0; y < n_iters; ++y)
    {
        rocke_value_t* y_i = rocke_b_const_i32(b, y);
        int r_const;
        int p_flush_val;
        int P_FLUSH;

        /* FMA phase (Python lines 2671-2691):
         * for r: for w_out: for s: load A → fma into acc[w_out][p_idx]. */
        for(r_const = 0; r_const < KH; ++r_const)
        {
            int p_idx = (((y - r_const) % KH) + KH) % KH;

            for(w_out = 0; w_out < BLOCK_W; ++w_out)
            {
                /* w_pos = q_tile_start + w_out */
                rocke_value_t* w_pos
                    = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, w_out));
                int s_const;

                for(s_const = 0; s_const < KW; ++s_const)
                {
                    rocke_value_t* a_off = NULL;
                    rocke_value_t* valid = NULL;
                    rocke_value_t* safe_off;
                    rocke_value_t* a_h;
                    rocke_value_t* a_f32;
                    const char* off_names[5];
                    rocke_value_t* off_vals[5];

                    /* a_off, valid = a_desc.offset(b, n=n, y_iter=y_i,
                     *                              wo=w_pos, s_off=const(s), c=ch) */
                    off_names[0] = "n";
                    off_vals[0] = ctx->n;
                    off_names[1] = "y_iter";
                    off_vals[1] = y_i;
                    off_names[2] = "wo";
                    off_vals[2] = w_pos;
                    off_names[3] = "s_off";
                    off_vals[3] = rocke_b_const_i32(b, s_const);
                    off_names[4] = "c";
                    off_vals[4] = ctx->ch;
                    rocke_transforms_descriptor_offset(
                        b, ctx->a_desc, off_names, off_vals, 5, &a_off, &valid);

                    /* safe_off = b.select(valid, b.mul(a_off, c_half_bytes), oob_sentinel) */
                    safe_off = rocke_b_select(
                        b, valid, rocke_b_mul(b, a_off, ctx->c_half_bytes), ctx->oob_sentinel);
                    a_h = rocke_b_buffer_load_f16(b, ctx->a_rsrc, safe_off, ctx->c0);
                    /* a_f32 = b.select(valid, b.cast_to_f32(a_h), zero_f32) */
                    a_f32 = rocke_b_select(b, valid, rocke_b_cast_to_f32(b, a_h), ctx->zero_f32);

                    /* acc[w_out][p_idx] = fma(weight[r][s], a_f32, acc[w_out][p_idx]) */
                    ctx->acc[w_out][p_idx] = rocke_b_fma(
                        b, ctx->weights_f32[r_const][s_const], a_f32, ctx->acc[w_out][p_idx]);
                }
            }
        }

        /* Flush phase (Python lines 2693-2708). */
        p_flush_val = y - (KH - 1);
        P_FLUSH = ((p_flush_val % KH) + KH) % KH;

        if(0 <= p_flush_val && p_flush_val < p->H)
        {
            for(w_out = 0; w_out < BLOCK_W; ++w_out)
            {
                rocke_value_t* out_q
                    = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, w_out));
                rocke_value_t* out_q_ok = rocke_b_cmp_lt(b, out_q, ctx->c_W);
                rocke_value_t* d_off = NULL;
                rocke_value_t* d_valid = NULL;
                rocke_value_t* safe_d;
                rocke_value_t* acc_h;
                const char* off_names[4];
                rocke_value_t* off_vals[4];

                off_names[0] = "n";
                off_vals[0] = ctx->n;
                off_names[1] = "h";
                off_vals[1] = rocke_b_const_i32(b, p_flush_val);
                off_names[2] = "w";
                off_vals[2] = out_q;
                off_names[3] = "k";
                off_vals[3] = ctx->ch;
                rocke_transforms_descriptor_offset(
                    b, ctx->d_desc, off_names, off_vals, 4, &d_off, &d_valid);

                safe_d = rocke_b_select(
                    b, out_q_ok, rocke_b_mul(b, d_off, ctx->c_half_bytes), ctx->oob_sentinel);
                acc_h = rocke_b_trunc_f32_to_f16(b, ctx->acc[w_out][P_FLUSH]);
                rocke_b_buffer_store_f16(b, ctx->d_rsrc, safe_d, ctx->c0, acc_h);
            }
        }

        /* Unconditional reset (Python lines 2711-2712). */
        for(w_out = 0; w_out < BLOCK_W; ++w_out)
        {
            ctx->acc[w_out][P_FLUSH] = ctx->zero_f32;
        }
    }

    return rocke_ir_builder_kernel(b);
}
