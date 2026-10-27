// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_tiled.cpp -- C++ port of
 * build_direct_depthwise_tiled (library/kernels/common/conv_direct_grouped.py).
 *
 * Output-stationary depthwise convolution (cpg = kpg = 1). A block owns a
 * block_h x block_w output tile of block_ch channels, one lane per channel. A
 * runtime loop walks the filter rows: iteration r holds one weight row (the
 * next one prefetched) and the input rows of the tile's output rows. With
 * stride 1 consecutive iterations read the same rows shifted by one, so
 * block_h - 1 rows are carried in registers and one new row is loaded per
 * iteration; stride s splits the filter rows into s phases, one window each.
 *
 * Register use and the loop body are linear in the filter width, unlike the
 * row-streaming kernel (KH^2 * KW), which is what keeps large filters cheap to
 * compile.
 *
 * Byte-identical to the Python source: every rocke_b_* call below is issued in
 * the order Python evaluates the matching expression (arguments left to
 * right, innermost first).
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

#include "rocke/error_boundary.hpp"

namespace
{

/* Per-kernel values the helpers below share (the Python closures' captures). */
struct dwt_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dconv_params_t* params;
    int is_bf16;
    int KW;
    int S;
    int BW;
    int n_cols;
    rocke_value_t* c0;
    rocke_value_t* c_half;
    rocke_value_t* oob_sentinel;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* w_ch_off;
    rocke_value_t* ch_off;
    rocke_value_t* n_off;
    rocke_value_t** col_terms;
};

/* Python: load(rsrc, off). */
rocke_value_t* dwt_load(const dwt_ctx* c, rocke_value_t* rsrc, rocke_value_t* off)
{
    return c->is_bf16 ? rocke_b_buffer_load_bf16(c->b, rsrc, off, c->c0)
                      : rocke_b_buffer_load_f16(c->b, rsrc, off, c->c0);
}

/* Python: load_w_row(r) -- raw weights of filter row r. */
void dwt_load_w_row(const dwt_ctx* c, rocke_value_t* r, rocke_value_t** out)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* row_off
        = rocke_b_add(b, c->w_ch_off, rocke_b_mul(b, r, rocke_b_const_i32(b, c->KW * 2)));
    int s;
    for(s = 0; s < c->KW; ++s)
        out[s] = dwt_load(c, c->b_rsrc, rocke_b_add(b, row_off, rocke_b_const_i32(b, 2 * s)));
}

/* Python: load_row(hi) -- raw input values of row hi, one per column. */
void dwt_load_row(const dwt_ctx* c, rocke_value_t* hi, rocke_value_t** out)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* ge = rocke_b_cmp_ge(b, hi, c->c0);
    rocke_value_t* lt = rocke_b_cmp_lt(b, hi, c->params->p_Hi);
    rocke_value_t* row_ok = rocke_b_land(b, ge, lt);
    rocke_value_t* h_off = rocke_b_mul(b, hi, c->params->p_A_stride_hi);
    rocke_value_t* row_off = rocke_b_mul(b, rocke_b_add(b, c->n_off, h_off), c->c_half);
    int col;
    for(col = 0; col < c->n_cols; ++col)
    {
        rocke_value_t* rc = rocke_b_add(b, row_off, c->col_terms[col]);
        rocke_value_t* sel = rocke_b_select(b, row_ok, rc, c->oob_sentinel);
        out[col] = dwt_load(c, c->a_rsrc, rocke_b_add(b, c->ch_off, sel));
    }
}

/* Python: to_f32(vals), in place. */
void dwt_to_f32(const dwt_ctx* c, rocke_value_t** vals, int n)
{
    int i;
    for(i = 0; i < n; ++i)
        vals[i] = rocke_b_cast_to_f32(c->b, vals[i]);
}

/* Python: fma_row(accs, t, row, w). */
void dwt_fma_row(
    const dwt_ctx* c, rocke_value_t** accs, int t, rocke_value_t** row, rocke_value_t** w)
{
    int w_out, s;
    for(w_out = 0; w_out < c->BW; ++w_out)
    {
        int idx = t * c->BW + w_out;
        for(s = 0; s < c->KW; ++s)
            accs[idx] = rocke_b_fma(c->b, w[s], row[w_out * c->S + s], accs[idx]);
    }
}

} // namespace

rocke_kernel_def_t* rocke_build_direct_depthwise_tiled(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_tiled_spec_t* spec, const char* arch)
{
    const rocke_direct_conv_problem_t* p;
    char reason[ROCKE_ERR_MSG_CAP];
    rocke_dconv_params_t params;
    dwt_ctx c;
    int BW, TH, WAVE, S, KH, KW, PAD, n_cols, n_acc, ph;
    rocke_value_t *c1, *c_wave, *zero_f32;
    rocke_value_t *tid, *wave_id, *lane, *bx, *by, *bz;
    rocke_value_t *n_h_tiles, *n, *h_tile, *q0, *h0, *ch, *ch_in_range;
    rocke_value_t *d_rsrc, *hi_tile;
    rocke_value_t** accs;
    const rocke_tensor_descriptor_t* d_desc;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(!rocke_direct_depthwise_tiled_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }

    p = &spec->problem;
    BW = spec->block_w;
    TH = spec->block_h;
    WAVE = spec->wave_size;
    S = p->stride;
    KH = p->KH;
    KW = p->KW;
    PAD = p->PAD;
    n_cols = rocke_direct_depthwise_tiled_n_cols(spec);
    n_acc = TH * BW;

    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       rocke_direct_depthwise_tiled_threads_per_block(spec));

    rocke_dconv_emit_params(b, &params, "fwd", rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16"));

    memset(&c, 0, sizeof c);
    c.b = b;
    c.params = &params;
    c.is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? 1 : 0;
    c.KW = KW;
    c.S = S;
    c.BW = BW;
    c.n_cols = n_cols;

    c.c0 = rocke_b_const_i32(b, 0);
    c1 = rocke_b_const_i32(b, 1);
    c_wave = rocke_b_const_i32(b, WAVE);
    c.c_half = rocke_b_const_i32(b, 2);
    c.oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    zero_f32 = rocke_b_const_f32(b, 0.0);

    tid = rocke_b_thread_id_x(b);
    wave_id = rocke_b_div(b, tid, c_wave);
    lane = rocke_b_mod(b, tid, c_wave);

    /* Grid: bx = W tile, by = channel tile, bz = n * n_h_tiles + h tile. */
    bx = rocke_b_block_id_x(b);
    by = rocke_b_block_id_y(b);
    bz = rocke_b_block_id_z(b);
    {
        rocke_value_t* sum = rocke_b_add(b, params.p_Ho, rocke_b_const_i32(b, TH - 1));
        n_h_tiles = rocke_b_div(b, sum, rocke_b_const_i32(b, TH));
    }
    n = rocke_b_div(b, bz, n_h_tiles);
    h_tile = rocke_b_mod(b, bz, n_h_tiles);
    q0 = rocke_b_mul(b, bx, rocke_b_const_i32(b, BW));
    h0 = rocke_b_mul(b, h_tile, rocke_b_const_i32(b, TH));
    {
        rocke_value_t* mul_by
            = rocke_b_mul(b, by, rocke_b_const_i32(b, rocke_direct_depthwise_tiled_block_ch(spec)));
        rocke_value_t* mul_wave = rocke_b_mul(b, wave_id, c_wave);
        rocke_value_t* inner = rocke_b_add(b, mul_wave, lane);
        ch = rocke_b_add(b, mul_by, inner);
    }
    ch_in_range = rocke_b_cmp_lt(b, ch, params.p_groups);

    c.a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    c.b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);

    /* Weights B[C, KH, KW, 1]: the lane's filter starts at ch * KH * KW. */
    c.w_ch_off = rocke_b_mul(b, rocke_b_mul(b, ch, rocke_b_const_i32(b, KH * KW)), c.c_half);

    /* Input offsets: the lane's channel plus a wave-uniform term per column. */
    c.ch_off = rocke_b_mul(b, ch, c.c_half);
    c.col_terms = (rocke_value_t**)alloca((size_t)n_cols * sizeof(rocke_value_t*));
    {
        rocke_value_t* q_s = rocke_b_mul(b, q0, rocke_b_const_i32(b, S));
        rocke_value_t* wi_tile = rocke_b_add(b, q_s, rocke_b_const_i32(b, -PAD));
        int col;
        for(col = 0; col < n_cols; ++col)
        {
            rocke_value_t* wi = rocke_b_add(b, wi_tile, rocke_b_const_i32(b, col));
            rocke_value_t* ge = rocke_b_cmp_ge(b, wi, c.c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wi, params.p_Wi);
            rocke_value_t* col_ok = rocke_b_land(b, ge, lt);
            rocke_value_t* w_off = rocke_b_mul(b, wi, params.p_A_stride_wi);
            rocke_value_t* col_off = rocke_b_mul(b, w_off, c.c_half);
            c.col_terms[col] = rocke_b_select(b, col_ok, col_off, c.oob_sentinel);
        }
    }
    c.n_off = rocke_b_mul(b, n, params.p_A_stride_n);
    {
        rocke_value_t* h_s = rocke_b_mul(b, h0, rocke_b_const_i32(b, S));
        hi_tile = rocke_b_add(b, h_s, rocke_b_const_i32(b, -PAD));
    }

    accs = (rocke_value_t**)alloca((size_t)n_acc * sizeof(rocke_value_t*));
    {
        int i;
        for(i = 0; i < n_acc; ++i)
            accs[i] = zero_f32;
    }

    for(ph = 0; ph < S; ++ph)
    {
        /* Filter rows r = ph + S * i; output row t of iteration i reads input
         * row hi_tile + ph + S * (i + t). */
        int n_r = (KH - ph + S - 1) / S;
        int n_win = (TH - 1) * n_cols;
        int num_iargs = n_acc + n_win + KW;
        rocke_iter_arg_t* iargs
            = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
        char(*names)[48] = (char(*)[48])alloca((size_t)num_iargs * 48);
        rocke_value_t** inits = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
        rocke_value_t** rows
            = (rocke_value_t**)alloca((size_t)(n_win + n_cols) * sizeof(rocke_value_t*));
        rocke_value_t** w_cur = (rocke_value_t**)alloca((size_t)KW * sizeof(rocke_value_t*));
        rocke_value_t** w_next = (rocke_value_t**)alloca((size_t)KW * sizeof(rocke_value_t*));
        rocke_value_t** new_accs = (rocke_value_t**)alloca((size_t)n_acc * sizeof(rocke_value_t*));
        rocke_value_t** yields
            = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
        rocke_value_t* phase_base;
        rocke_for_t loop;
        char iv_name[32];
        int i, t, s, idx = 0;

        phase_base = rocke_b_add(b, hi_tile, rocke_b_const_i32(b, ph));
        for(i = 0; i < n_acc; ++i)
        {
            snprintf(names[idx], 48, "dwt_acc%d_%d", ph, i);
            inits[idx++] = accs[i];
        }
        for(t = 0; t < TH - 1; ++t)
        {
            rocke_value_t* hi = rocke_b_add(b, phase_base, rocke_b_const_i32(b, S * t));
            dwt_load_row(&c, hi, inits + idx);
            dwt_to_f32(&c, inits + idx, n_cols);
            for(i = 0; i < n_cols; ++i)
            {
                snprintf(names[idx], 48, "dwt_x%d_r%d_c%d", ph, t, i);
                ++idx;
            }
        }
        dwt_load_w_row(&c, rocke_b_const_i32(b, ph), inits + idx);
        for(s = 0; s < KW; ++s)
        {
            snprintf(names[idx], 48, "dwt_w%d_s%d", ph, s);
            ++idx;
        }
        for(i = 0; i < num_iargs; ++i)
        {
            iargs[i].name = names[i];
            iargs[i].init = inits[i];
        }

        snprintf(iv_name, sizeof iv_name, "dwt_r%d", ph);
        loop = rocke_b_scf_for_iter(b,
                                    c.c0,
                                    rocke_b_const_i32(b, n_r),
                                    c1,
                                    iargs,
                                    num_iargs,
                                    iv_name,
                                    spec->unroll_rows,
                                    /*elide_trailing_barrier=*/false);
        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* step;
            rocke_value_t* new_row_hi;
            rocke_value_t* w_next_r;

            for(i = 0; i < n_acc; ++i)
                new_accs[i] = loop.iter_vars[i];
            for(i = 0; i < n_win; ++i)
                rows[i] = loop.iter_vars[n_acc + i];
            for(s = 0; s < KW; ++s)
                w_cur[s] = loop.iter_vars[n_acc + n_win + s];
            dwt_to_f32(&c, w_cur, KW);

            /* The newest row is needed last; the next weight row is prefetched. */
            step = rocke_b_mul(b, loop.iv, rocke_b_const_i32(b, S));
            {
                rocke_value_t* base_step = rocke_b_add(b, phase_base, step);
                new_row_hi = rocke_b_add(b, base_step, rocke_b_const_i32(b, S * (TH - 1)));
            }
            dwt_load_row(&c, new_row_hi, rows + n_win);
            {
                rocke_value_t* c_ph = rocke_b_const_i32(b, ph);
                rocke_value_t* nxt = rocke_b_add(b, step, rocke_b_const_i32(b, S));
                w_next_r = rocke_b_add(b, c_ph, nxt);
            }
            dwt_load_w_row(&c, w_next_r, w_next);
            for(t = 0; t < TH - 1; ++t)
                dwt_fma_row(&c, new_accs, t, rows + t * n_cols, w_cur);
            dwt_to_f32(&c, rows + n_win, n_cols);
            dwt_fma_row(&c, new_accs, TH - 1, rows + n_win, w_cur);

            /* Keep LLVM's SLP vectorizer from pairing the FMA chains into
             * v_pk_fma_f32 (see build_direct_depthwise). */
            {
                const rocke_type_t* f32_ty = rocke_f32();
                rocke_inline_asm_opts_t opts;
                memset(&opts, 0, sizeof opts);
                opts.sideeffect = false;
                opts.sideeffect_set = true;
                for(i = 0; i < n_acc; ++i)
                {
                    rocke_op_t* asm_op
                        = rocke_b_inline_asm(b, "", "=v,0", &new_accs[i], 1, &f32_ty, 1, &opts);
                    new_accs[i] = asm_op ? asm_op->results[0] : NULL;
                }
            }
            idx = 0;
            for(i = 0; i < n_acc; ++i)
                yields[idx++] = new_accs[i];
            for(i = n_cols; i < n_win + n_cols; ++i)
                yields[idx++] = rows[i];
            for(s = 0; s < KW; ++s)
                yields[idx++] = w_next[s];
            rocke_b_scf_yield(b, yields, num_iargs);
        }
        rocke_b_region_leave(b);
        for(i = 0; i < n_acc; ++i)
            accs[i] = loop.op->results[i];
    }

    /* Epilogue: one store per output of the tile. */
    {
        rocke_dynamic_tensor_descriptor_t* d_dyn = rocke_dconv_d_descriptor_dynamic(b, &params);
        int t, w_out;
        if(!d_dyn)
            return NULL;
        d_desc = &d_dyn->base;
        for(t = 0; t < TH; ++t)
        {
            rocke_value_t* ho = rocke_b_add(b, h0, rocke_b_const_i32(b, t));
            rocke_value_t* h_ok = rocke_b_land(b, rocke_b_cmp_lt(b, ho, params.p_Ho), ch_in_range);
            for(w_out = 0; w_out < BW; ++w_out)
            {
                rocke_value_t* wo = rocke_b_add(b, q0, rocke_b_const_i32(b, w_out));
                rocke_value_t* ok = rocke_b_land(b, h_ok, rocke_b_cmp_lt(b, wo, params.p_Wo));
                rocke_value_t* d_off = NULL;
                rocke_value_t* d_valid = NULL;
                rocke_value_t* safe_d;
                rocke_value_t* acc = accs[t * BW + w_out];
                const char* off_names[4] = {"n", "h", "w", "k"};
                rocke_value_t* off_vals[4];

                off_vals[0] = n;
                off_vals[1] = ho;
                off_vals[2] = wo;
                off_vals[3] = ch;
                rocke_transforms_descriptor_offset(
                    b, d_desc, off_names, off_vals, 4, &d_off, &d_valid);
                safe_d = rocke_b_select(b, ok, rocke_b_mul(b, d_off, c.c_half), c.oob_sentinel);
                if(c.is_bf16)
                    rocke_b_buffer_store_bf16(
                        b, d_rsrc, safe_d, c.c0, rocke_b_trunc_f32_to_bf16(b, acc));
                else
                    rocke_b_buffer_store_f16(
                        b, d_rsrc, safe_d, c.c0, rocke_b_trunc_f32_to_f16(b, acc));
            }
        }
    }

    return rocke_ir_builder_kernel(b);
}

/* Initialises `b` with spec.kernel_name(), then builds. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_direct_depthwise_tiled_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_tiled_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_direct_depthwise_tiled_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_depthwise_tiled(b, spec, arch);
    });
}
