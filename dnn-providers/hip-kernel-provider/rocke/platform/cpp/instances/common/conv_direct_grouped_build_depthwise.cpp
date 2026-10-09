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
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.io.h"
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
    /* The weight table is sized for filters up to the depthwise limit. */
    if(ctx->p.KH > ROCKE_DCONV_DW_MAX_KH || ctx->p.KW > ROCKE_DCONV_DW_MAX_KW)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->BLOCK_W = spec->block_w;
    ctx->BLOCK_WAVES = spec->block_waves;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_depthwise_threads_per_block(spec);
    ctx->BLOCK_CH = rocke_direct_depthwise_block_ch(spec);
    ctx->c_stride_dw = ctx->p.stride > 0 ? ctx->p.stride : 1;
    ctx->Ho = (ctx->p.H + 2 * ctx->p.PAD - ctx->p.KH) / ctx->c_stride_dw + 1;
    ctx->Wo = (ctx->p.W + 2 * ctx->p.PAD - ctx->p.KW) / ctx->c_stride_dw + 1;
    ctx->n_iters = ctx->p.H + ctx->p.KH - 1;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    ctx->is_bf16 = (ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0) ? 1 : 0;
    /* params: the AOT kernarg block, in conv_abi order. */
    rocke_dconv_emit_params(
        b, &ctx->params, "fwd", rocke_b_io_ir_type(b, ctx->p.dtype ? ctx->p.dtype : "fp16"));
    ctx->A = ctx->params.A;
    ctx->Bp = ctx->params.Bp;
    ctx->D = ctx->params.D;
    ctx->A_bytes = ctx->params.A_bytes;
    ctx->B_bytes = ctx->params.B_bytes;
    ctx->D_bytes = ctx->params.D_bytes;

    /* Constants emitted in Python source order (lines 2617-2623):
     * c0, c_wave, c_W (= Wo output width), c_groups, c_half_bytes, oob_sentinel, zero_f32. */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    /* AOT: the store guard and the channel guard bound against kernargs. */
    ctx->c_W = ctx->params.p_Wo;
    ctx->c_groups = ctx->params.p_groups;
    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->zero_f32 = rocke_b_const_f32(b, 0.0);

    /* ---- thread / wave / lane decode (Python lines 2625-2627) ---- */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);

    /* Grid: bx=W-tile, by=channel-tile, bz=batch (Python lines 2630-2633). */
    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, rocke_b_const_i32(b, ctx->BLOCK_W));

    /* ch = by*BLOCK_CH + wave_id*WAVE + lane (Python lines 2635-2638). */
    {
        rocke_value_t* mul_by = rocke_b_mul(b, ctx->by, rocke_b_const_i32(b, ctx->BLOCK_CH));
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_wave);
        rocke_value_t* inner = rocke_b_add(b, mul_wave, ctx->lane);
        ctx->ch = rocke_b_add(b, mul_by, inner);
    }
    /* ch_in_range = ch < groups (Python line 2640). */
    ctx->ch_in_range = rocke_b_cmp_lt(b, ctx->ch, ctx->c_groups);

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Descriptor phase  (Python lines 2614-2646)
 *
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
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

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

    /* d_desc = D[N, Ho, Wo, total_k] with runtime extents. */
    {
        rocke_dynamic_tensor_descriptor_t* d_dyn
            = rocke_dconv_d_descriptor_dynamic(b, &ctx->params);
        if(!d_dyn)
            return;
        ctx->d_desc = &d_dyn->base;
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
            {
                rocke_value_t* safe_w = rocke_b_select(b,
                                                       ctx->ch_in_range,
                                                       rocke_b_mul(b, w_off, ctx->c_half_bytes),
                                                       ctx->oob_sentinel);
                w_h = ctx->is_bf16 ? rocke_b_buffer_load_bf16(b, ctx->b_rsrc, safe_w, ctx->c0)
                                   : rocke_b_buffer_load_f16(b, ctx->b_rsrc, safe_w, ctx->c0);
                ctx->weights_f32[r_const][s_const] = rocke_b_select(
                    b, ctx->ch_in_range, rocke_b_cast_to_f32(b, w_h), ctx->zero_f32);
            }
        }
    }
}

/* ===================================================================== *
 *  H-row streaming loop  (stride-aware)
 *
 *  Accumulator: acc[slot][w_out] — KH × BLOCK_W scalar f32 values.
 *  The flush uses stride-aware ho_row = p_flush_val / c_stride_dw and only
 *  stores when p_flush_val % c_stride_dw == 0.
 *  ch_in_range guards the stores for partial channel tiles.
 *
 *  Streaming row y holds input row hi = y - PAD. The stream starts at
 *  y = PAD (the halo rows above the image hold no data), an FMA loop covers
 *  ceil(Hi / KH) groups of KH rows with the input rows prefetched PF rows
 *  ahead, and a load-free loop flushes the output rows the rows past the
 *  image complete. The trip counts follow the runtime height (AOT), so there
 *  is no build-time-unrolled form. See the Python source for why.
 * ===================================================================== */

/* Per-kernel values the streaming phases share. */
typedef struct dw_stream
{
    int KH;
    int KW;
    int BLOCK_W;
    int stride;
    int n_cols;
    rocke_value_t* c_KH;
    rocke_value_t* c_stride_rv;
    rocke_value_t* n_iters_v;
    rocke_value_t* ch_off;
    rocke_value_t* n_off;
    rocke_value_t** col_terms; /* per distinct column; the OOB sentinel off-image */
} dw_stream_t;

/* Raw input values of streaming row y, one per distinct column: the lane's
 * channel offset plus a wave-uniform term, the OOB sentinel off the image. */
static void dw_load_row(rocke_dconv_dw_ctx_t* ctx,
                        const dw_stream_t* st,
                        rocke_value_t* y,
                        rocke_value_t** out)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* hi = rocke_b_add(b, y, rocke_b_const_i32(b, -ctx->p.PAD));
    rocke_value_t* row_ok = rocke_b_cmp_lt(b, hi, ctx->params.p_Hi);
    rocke_value_t* row_off;
    int c;

    {
        rocke_value_t* h_off = rocke_b_mul(b, hi, ctx->params.p_A_stride_hi);
        row_off = rocke_b_mul(b, rocke_b_add(b, st->n_off, h_off), ctx->c_half_bytes);
    }
    for(c = 0; c < st->n_cols; ++c)
    {
        rocke_value_t* rc = rocke_b_add(b, row_off, st->col_terms[c]);
        rocke_value_t* s_term = rocke_b_select(b, row_ok, rc, ctx->oob_sentinel);
        rocke_value_t* v_off = rocke_b_add(b, st->ch_off, s_term);
        out[c] = ctx->is_bf16 ? rocke_b_buffer_load_bf16(b, ctx->a_rsrc, v_off, ctx->c0)
                              : rocke_b_buffer_load_f16(b, ctx->a_rsrc, v_off, ctx->c0);
    }
}

/* Streaming row of step j of group grp_iv: grp_iv*KH + (PAD + j). Emits the
 * mul first, then the const, matching Python's left-to-right evaluation. */
static rocke_value_t*
    dw_row_y(rocke_dconv_dw_ctx_t* ctx, const dw_stream_t* st, rocke_value_t* grp_iv, int j)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* mul_gk = rocke_b_mul(b, grp_iv, st->c_KH);
    rocke_value_t* cj = rocke_b_const_i32(b, ctx->p.PAD + j);
    return rocke_b_add(b, mul_gk, cj);
}

/* Store the output row streaming row y_j completes; reset its slot. */
static void dw_flush(rocke_dconv_dw_ctx_t* ctx,
                     const dw_stream_t* st,
                     rocke_value_t* grp_iv,
                     int j,
                     rocke_value_t* y_j,
                     rocke_value_t** accs)
{
    rocke_ir_builder_t* b = ctx->b;
    int KH = st->KH;
    int BLOCK_W = st->BLOCK_W;
    int P_FLUSH_j = (ctx->p.PAD + j + 1) % KH; /* STATIC */
    rocke_value_t* j_valid = rocke_b_cmp_lt(b, y_j, st->n_iters_v);
    rocke_value_t* p_flush_rv = rocke_b_add(b, y_j, rocke_b_const_i32(b, -(KH - 1)));
    rocke_value_t* flush_ge;
    rocke_value_t* should_flush;
    rocke_value_t* ho_row_j;
    int w_out;

    /* y_j >= KH - 1 holds from group 1 on; in group 0 from step PAD on. */
    if(j >= ctx->p.PAD)
    {
        flush_ge = j_valid;
    }
    else
    {
        flush_ge = rocke_b_land(b, rocke_b_cmp_lt(b, ctx->c0, grp_iv), j_valid);
    }

    if(st->stride == 1)
    {
        should_flush = flush_ge;
    }
    else
    {
        rocke_value_t* flush_stride
            = rocke_b_cmp_eq(b, rocke_b_mod(b, p_flush_rv, st->c_stride_rv), ctx->c0);
        should_flush = rocke_b_land(b, flush_ge, flush_stride);
    }

    ho_row_j = rocke_b_div(b, p_flush_rv, st->c_stride_rv);

    for(w_out = 0; w_out < BLOCK_W; ++w_out)
    {
        rocke_value_t* out_q = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, w_out));
        rocke_value_t* out_q_ok
            = rocke_b_land(b, rocke_b_cmp_lt(b, out_q, ctx->c_W), ctx->ch_in_range);
        rocke_value_t* store_ok = rocke_b_land(b, out_q_ok, should_flush);
        rocke_value_t* acc_val = accs[P_FLUSH_j * BLOCK_W + w_out];
        rocke_value_t* d_off = NULL;
        rocke_value_t* d_valid = NULL;
        rocke_value_t* safe_d;
        const char* off_names[4];
        rocke_value_t* off_vals[4];

        off_names[0] = "n";
        off_vals[0] = ctx->n;
        off_names[1] = "h";
        off_vals[1] = ho_row_j;
        off_names[2] = "w";
        off_vals[2] = out_q;
        off_names[3] = "k";
        off_vals[3] = ctx->ch;
        rocke_transforms_descriptor_offset(
            b, ctx->d_desc, off_names, off_vals, 4, &d_off, &d_valid);

        safe_d = rocke_b_select(
            b, store_ok, rocke_b_mul(b, d_off, ctx->c_half_bytes), ctx->oob_sentinel);
        if(ctx->is_bf16)
            rocke_b_buffer_store_bf16(
                b, ctx->d_rsrc, safe_d, ctx->c0, rocke_b_trunc_f32_to_bf16(b, acc_val));
        else
            rocke_b_buffer_store_f16(
                b, ctx->d_rsrc, safe_d, ctx->c0, rocke_b_trunc_f32_to_f16(b, acc_val));
    }

    for(w_out = 0; w_out < BLOCK_W; ++w_out)
        accs[P_FLUSH_j * BLOCK_W + w_out] = ctx->zero_f32;
}

rocke_kernel_def_t* rocke_dconv_dw_stream_h_loop(rocke_dconv_dw_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int KW = p->KW;
    int BLOCK_W = ctx->BLOCK_W;
    int stride = ctx->c_stride_dw;
    /* Distinct input columns of a row in first-tap order: tap (w_out, s)
     * reads column w_out * stride + s; col_idx maps a column to its
     * position in that order. */
    int max_col = (BLOCK_W - 1) * stride + KW;
    int* col_num = (int*)alloca((size_t)max_col * sizeof(int));
    int* col_idx = (int*)alloca((size_t)max_col * sizeof(int));
    int n_cols = 0;
    int num_acc = KH * BLOCK_W;
    int PF;
    int num_win;
    int num_iargs;
    dw_stream_t st;
    rocke_iter_arg_t* iargs;
    rocke_for_t fma_loop;
    rocke_for_t flush_loop;
    rocke_value_t** new_accs;
    rocke_value_t** window;
    rocke_value_t** yields;
    rocke_value_t** a_row;
    rocke_value_t* c1;
    rocke_value_t* n_fma_groups_v;
    rocke_value_t* n_groups_v;
    char(*name_store)[48];
    int j, i;

    {
        int w, s;
        for(w = 0; w < max_col; ++w)
            col_idx[w] = -1;
        for(w = 0; w < BLOCK_W; ++w)
        {
            for(s = 0; s < KW; ++s)
            {
                int col = w * stride + s;
                if(col_idx[col] >= 0)
                    continue;
                col_idx[col] = n_cols;
                col_num[n_cols] = col;
                ++n_cols;
            }
        }
    }
    /* Rows in flight: enough for ~16 loads, at most a group. */
    PF = (16 + n_cols - 1) / n_cols;
    if(PF > KH)
        PF = KH;
    num_win = PF * n_cols;
    num_iargs = num_acc + num_win;

    st.KH = KH;
    st.KW = KW;
    st.BLOCK_W = BLOCK_W;
    st.stride = stride;
    st.n_cols = n_cols;
    st.col_terms = (rocke_value_t**)alloca((size_t)n_cols * sizeof(rocke_value_t*));
    /* window holds PF rows plus the one being issued. */
    window = (rocke_value_t**)alloca((size_t)(num_win + n_cols) * sizeof(rocke_value_t*));
    yields = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
    a_row = (rocke_value_t**)alloca((size_t)n_cols * sizeof(rocke_value_t*));
    new_accs = (rocke_value_t**)alloca((size_t)num_acc * sizeof(rocke_value_t*));
    iargs = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
    name_store = (char(*)[48])alloca((size_t)num_iargs * 48 * sizeof(char));

    c1 = rocke_b_const_i32(b, 1);
    st.c_KH = rocke_b_const_i32(b, KH);
    st.c_stride_rv = rocke_b_const_i32(b, stride);
    /* n_iters = Hi + KH - 1 rows, the last one completing the last output row. */
    st.n_iters_v = rocke_b_add(b, ctx->params.p_Hi, rocke_b_const_i32(b, KH - 1));
    n_fma_groups_v
        = rocke_b_div(b, rocke_b_add(b, ctx->params.p_Hi, rocke_b_const_i32(b, KH - 1)), st.c_KH);
    n_groups_v = rocke_b_div(
        b, rocke_b_add(b, ctx->params.p_Hi, rocke_b_const_i32(b, p->PAD + KH - 1)), st.c_KH);

    /* Input offset = the lane's channel + a wave-uniform term per column. */
    st.ch_off = rocke_b_mul(b, ctx->ch, ctx->c_half_bytes);
    {
        rocke_value_t* q_s = rocke_b_mul(b, ctx->q_tile_start, st.c_stride_rv);
        rocke_value_t* c_mpad = rocke_b_const_i32(b, -p->PAD);
        rocke_value_t* wi_tile = rocke_b_add(b, q_s, c_mpad);
        int c;
        for(c = 0; c < n_cols; ++c)
        {
            rocke_value_t* wi = rocke_b_add(b, wi_tile, rocke_b_const_i32(b, col_num[c]));
            rocke_value_t* ge = rocke_b_cmp_ge(b, wi, ctx->c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wi, ctx->params.p_Wi);
            rocke_value_t* col_ok = rocke_b_land(b, ge, lt);
            rocke_value_t* w_off = rocke_b_mul(b, wi, ctx->params.p_A_stride_wi);
            rocke_value_t* col_off = rocke_b_mul(b, w_off, ctx->c_half_bytes);
            st.col_terms[c] = rocke_b_select(b, col_ok, col_off, ctx->oob_sentinel);
        }
    }
    st.n_off = rocke_b_mul(b, ctx->n, ctx->params.p_A_stride_n);

    /* Iter args: accumulators, then rows PAD .. PAD+PF-1 in flight. */
    {
        int kh, w, k, c, idx = 0;
        for(kh = 0; kh < KH; ++kh)
        {
            for(w = 0; w < BLOCK_W; ++w)
            {
                snprintf(name_store[idx], 48, "dw_acc_kh%d_w%d", kh, w);
                iargs[idx].name = name_store[idx];
                iargs[idx].init = ctx->zero_f32;
                ++idx;
            }
        }
        for(k = 0; k < PF; ++k)
        {
            dw_load_row(ctx, &st, rocke_b_const_i32(b, p->PAD + k), window + k * n_cols);
            for(c = 0; c < n_cols; ++c)
            {
                snprintf(name_store[idx], 48, "dw_a_r%d_c%d", k, col_num[c]);
                iargs[idx].name = name_store[idx];
                iargs[idx].init = window[k * n_cols + c];
                ++idx;
            }
        }
    }

    fma_loop = rocke_b_scf_for_iter(b,
                                    ctx->c0,
                                    n_fma_groups_v,
                                    c1,
                                    iargs,
                                    num_iargs,
                                    "dw_grp",
                                    /*unroll=*/false,
                                    /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, fma_loop.body);

    for(i = 0; i < num_acc; ++i)
        new_accs[i] = fma_loop.iter_vars[i];
    for(i = 0; i < num_win; ++i)
        window[i] = fma_loop.iter_vars[num_acc + i];

    for(j = 0; j < KH; ++j)
    {
        rocke_value_t* y_j = dw_row_y(ctx, &st, fma_loop.iv, j);
        int w_out, s_const, r_const, c;

        /* Issue row y_j + PF, then compute row y_j (the oldest in flight). */
        {
            rocke_value_t* cpf = rocke_b_const_i32(b, PF);
            dw_load_row(ctx, &st, rocke_b_add(b, y_j, cpf), window + num_win);
        }
        for(c = 0; c < n_cols; ++c)
            a_row[c] = rocke_b_cast_to_f32(b, window[c]);
        memmove(window, window + n_cols, (size_t)num_win * sizeof(rocke_value_t*));

        /* Python order: outer w_out, inner s_const, innermost r_const. */
        for(w_out = 0; w_out < BLOCK_W; ++w_out)
        {
            for(s_const = 0; s_const < KW; ++s_const)
            {
                rocke_value_t* a_f32 = a_row[col_idx[w_out * stride + s_const]];

                for(r_const = 0; r_const < KH; ++r_const)
                {
                    /* STATIC slot of output row (y_j - r) / stride. */
                    int p_idx = ((p->PAD + j - r_const + KH) % KH) * BLOCK_W + w_out;
                    new_accs[p_idx] = rocke_b_fma(
                        b, ctx->weights_f32[r_const][s_const], a_f32, new_accs[p_idx]);
                }
            }
        }

        dw_flush(ctx, &st, fma_loop.iv, j, y_j, new_accs);
    }

    /* An empty asm on each accumulator keeps LLVM's SLP vectorizer from
     * pairing the loop-carried FMA chains into v_pk_fma_f32 (see Python). */
    {
        const rocke_type_t* f32_ty = rocke_f32();
        rocke_inline_asm_opts_t opts;
        memset(&opts, 0, sizeof opts);
        opts.sideeffect = false;
        opts.sideeffect_set = true;
        for(i = 0; i < num_acc; ++i)
        {
            rocke_op_t* asm_op
                = rocke_b_inline_asm(b, "", "=v,0", &new_accs[i], 1, &f32_ty, 1, &opts);
            new_accs[i] = asm_op ? asm_op->results[0] : NULL;
        }
    }
    for(i = 0; i < num_acc; ++i)
        yields[i] = new_accs[i];
    for(i = 0; i < num_win; ++i)
        yields[num_acc + i] = window[i];
    rocke_b_scf_yield(b, yields, num_iargs);
    rocke_b_region_leave(b);

    /* The rows past the image: no input, only output rows to complete. */
    {
        int kh, w, idx = 0;
        for(kh = 0; kh < KH; ++kh)
        {
            for(w = 0; w < BLOCK_W; ++w)
            {
                snprintf(name_store[idx], 48, "dw_tail_acc_kh%d_w%d", kh, w);
                iargs[idx].name = name_store[idx];
                iargs[idx].init = fma_loop.op->results[idx];
                ++idx;
            }
        }
    }
    flush_loop = rocke_b_scf_for_iter(b,
                                      n_fma_groups_v,
                                      n_groups_v,
                                      c1,
                                      iargs,
                                      num_acc,
                                      "dw_tail",
                                      /*unroll=*/false,
                                      /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, flush_loop.body);
    for(i = 0; i < num_acc; ++i)
        new_accs[i] = flush_loop.iter_vars[i];
    for(j = 0; j < KH; ++j)
        dw_flush(ctx, &st, flush_loop.iv, j, dw_row_y(ctx, &st, flush_loop.iv, j), new_accs);
    rocke_b_scf_yield(b, new_accs, num_acc);
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}
