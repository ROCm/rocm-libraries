// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_wgrad.cpp -- C99 port of
 * build_direct_depthwise_wgrad (library/kernels/common/conv_direct_grouped.py):
 * the direct depthwise wgrad (cpg = kpg = 1), fp32 FMAs in registers.
 *
 *   - Lanes own chan_per_lane contiguous channels; grid x is a channel chunk,
 *     grid y a run of flattened output rows (n, ho). The block's waves split
 *     into r_split groups over the filter rows times row_waves waves over the
 *     output rows, rows_per_wave each.
 *   - A wave walks each row in wo_chunk-wide chunks: dY of the chunk, the X
 *     span of each of its filter rows, then wo_chunk * KW FMAs per filter row
 *     and channel. carry keeps the KW - stride shared X columns; prefetch
 *     issues the next chunk's loads before this chunk's FMAs.
 *   - Addresses are a per-lane channel offset plus a wave-uniform part; an
 *     out-of-image row or column jumps its uniform part past every buffer.
 *
 * Epilogue: an LDS reduce over the waves, then fp32 atomics into replica
 * block_y % ws_replicas of a [ws_replicas][C][KH * KW] scratch.
 *
 * Every helper emits in PYTHON EVALUATION ORDER, one builder call per Python
 * statement, so the SSA sequence -- and therefore the .ll bytes -- match.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV = 4 | (2 << 3). */
#define ROCKE_DW_WGRAD_CPOL_ATOMIC_CASCADE_DEV (4 | (2 << 3))

/* Fixed-capacity scratch, checked before any write. The accumulator budget
 * (128) bounds RC * KW * CPL; the X span and carried head follow from the
 * filter, wo_chunk and stride. */
#define ROCKE_DW_WGRAD_MAX_ACC 128
#define ROCKE_DW_WGRAD_MAX_RC 32
#define ROCKE_DW_WGRAD_MAX_CHUNK 64
#define ROCKE_DW_WGRAD_MAX_SPAN 256
#define ROCKE_DW_WGRAD_MAX_LOADS 1024
#define ROCKE_DW_WGRAD_MAX_ITER (ROCKE_DW_WGRAD_MAX_ACC + 2 * ROCKE_DW_WGRAD_MAX_LOADS)

typedef struct rocke_dw_wgrad_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dconv_params_t* params;
    const rocke_type_t* io_type;
    int is_bf16;
    int CPL, V, U, KW, RC, XSPAN, HC, NEW, NA;

    rocke_value_t* c0;
    rocke_value_t* c2;
    rocke_value_t* c_v;
    rocke_value_t* c_vu;
    rocke_value_t* c_pad;
    rocke_value_t* c_W;
    rocke_value_t* c_Wo;
    rocke_value_t* far;
    rocke_value_t* lane_byte;
    rocke_value_t* s_w_bytes;
    rocke_value_t* s_wo_bytes;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* col_x[ROCKE_DW_WGRAD_MAX_SPAN];
    rocke_value_t* col_dy[ROCKE_DW_WGRAD_MAX_CHUNK];
} rocke_dw_wgrad_ctx_t;

static void dw_fail(rocke_ir_builder_t* b)
{
    if(b->status == ROCKE_OK)
    {
        b->status = ROCKE_ERR_VALUE;
    }
}

/* _load(rsrc, u) */
static rocke_value_t* dw_load(rocke_dw_wgrad_ctx_t* c, rocke_value_t* rsrc, rocke_value_t* u)
{
    rocke_value_t* off = rocke_b_add(c->b, c->lane_byte, u);
    return c->is_bf16 ? rocke_b_buffer_load_vN_bf16(c->b, rsrc, off, c->c0, c->CPL / 2)
                      : rocke_b_buffer_load_vN_f16(c->b, rsrc, off, c->c0, c->CPL / 2);
}

/* _issue_dy(dy_base, j): appends V loads to out[*n]. */
static void dw_issue_dy(
    rocke_dw_wgrad_ctx_t* c, rocke_value_t* dy_base, rocke_value_t* j, rocke_value_t** out, int* n)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* wo0 = rocke_b_mul(b, j, c->c_v);
    rocke_value_t* v_max = rocke_b_sub(b, c->c_Wo, wo0);
    rocke_value_t* wo_off = rocke_b_mul(b, wo0, c->s_wo_bytes);
    rocke_value_t* base = rocke_b_add(b, dy_base, wo_off);
    int v;
    for(v = 0; v < c->V; ++v)
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, v);
        rocke_value_t* ok = rocke_b_cmp_lt(b, c_t, v_max);
        rocke_value_t* u = rocke_b_add(b, base, c->col_dy[v]);
        u = rocke_b_select(b, ok, u, c->far);
        out[(*n)++] = dw_load(c, c->a_rsrc, u);
    }
}

/* _issue_x(x_bases, j, t_lo, t_hi): appends RC * (t_hi - t_lo) loads. */
static void dw_issue_x(rocke_dw_wgrad_ctx_t* c,
                       rocke_value_t* const* x_bases,
                       rocke_value_t* j,
                       int t_lo,
                       int t_hi,
                       rocke_value_t** out,
                       int* n)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* wo_u = rocke_b_mul(b, j, c->c_vu);
    rocke_value_t* wi0 = rocke_b_sub(b, wo_u, c->c_pad);
    rocke_value_t* t_min = rocke_b_sub(b, c->c0, wi0);
    rocke_value_t* t_max = rocke_b_sub(b, c->c_W, wi0);
    rocke_value_t* x_chunk = rocke_b_mul(b, wi0, c->s_w_bytes);
    rocke_value_t* oks[ROCKE_DW_WGRAD_MAX_SPAN];
    int t, rr;
    for(t = t_lo; t < t_hi; ++t)
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, t);
        rocke_value_t* ge = rocke_b_cmp_ge(b, c_t, t_min);
        rocke_value_t* lt = rocke_b_cmp_lt(b, c_t, t_max);
        oks[t - t_lo] = rocke_b_land(b, ge, lt);
    }
    for(rr = 0; rr < c->RC; ++rr)
    {
        rocke_value_t* base = rocke_b_add(b, x_bases[rr], x_chunk);
        for(t = t_lo; t < t_hi; ++t)
        {
            rocke_value_t* u = rocke_b_add(b, base, c->col_x[t]);
            u = rocke_b_select(b, oks[t - t_lo], u, c->far);
            out[(*n)++] = dw_load(c, c->b_rsrc, u);
        }
    }
}

/* _fma(accs, dy_vecs, spans); spans[r * XSPAN + t]. */
static void dw_fma(rocke_dw_wgrad_ctx_t* c,
                   rocke_value_t** accs,
                   rocke_value_t* const* dy_vecs,
                   rocke_value_t* const* spans)
{
    rocke_ir_builder_t* b = c->b;
    const int CPL = c->CPL, V = c->V, U = c->U, KW = c->KW, XSPAN = c->XSPAN;
    rocke_value_t* dy[ROCKE_DW_WGRAD_MAX_CHUNK][8];
    rocke_value_t* xs[ROCKE_DW_WGRAD_MAX_SPAN][8];
    int v, e, r, t, s;
    for(v = 0; v < V; ++v)
    {
        for(e = 0; e < CPL; ++e)
        {
            rocke_value_t* el = rocke_b_vec_extract(b, dy_vecs[v], e);
            dy[v][e] = rocke_b_cast_to_f32(b, el);
        }
    }
    for(r = 0; r < c->RC; ++r)
    {
        for(t = 0; t < XSPAN; ++t)
        {
            for(e = 0; e < CPL; ++e)
            {
                rocke_value_t* el = rocke_b_vec_extract(b, spans[r * XSPAN + t], e);
                xs[t][e] = rocke_b_cast_to_f32(b, el);
            }
        }
        for(s = 0; s < KW; ++s)
        {
            for(e = 0; e < CPL; ++e)
            {
                rocke_value_t* a = accs[(r * KW + s) * CPL + e];
                for(v = 0; v < V; ++v)
                    a = rocke_b_fma(b, dy[v][e], xs[v * U + s][e], a);
                accs[(r * KW + s) * CPL + e] = a;
            }
        }
    }
}

/* _spans(head, new): spans[r * XSPAN + t]. */
static void dw_spans(rocke_dw_wgrad_ctx_t* c,
                     rocke_value_t* const* head,
                     rocke_value_t* const* fresh,
                     rocke_value_t** spans)
{
    int r, t;
    for(r = 0; r < c->RC; ++r)
    {
        for(t = 0; t < c->HC; ++t)
            spans[r * c->XSPAN + t] = head[r * c->HC + t];
        for(t = 0; t < c->NEW; ++t)
            spans[r * c->XSPAN + c->HC + t] = fresh[r * c->NEW + t];
    }
}

rocke_kernel_def_t* rocke_build_direct_depthwise_wgrad(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_wgrad_spec_t* spec, const char* arch)
{
    rocke_dw_wgrad_ctx_t cx;
    rocke_dw_wgrad_ctx_t* c = &cx;
    rocke_dconv_params_t params;
    const rocke_direct_conv_problem_t* p;
    int KH, KW, U, PAD, WAVE, CPL, WAVES, RPW, V, RS, RC, ROWW, THREADS, KK, BLOCK_CH;
    int XSPAN, HC, NEW, NA, TILE;
    int i, k, t, v, rr, s, e, w, k0;
    char reason[ROCKE_ERR_MSG_CAP];
    char names[ROCKE_DW_WGRAD_MAX_ITER][32];
    rocke_value_t* c1;
    rocke_value_t* c_H;
    rocke_value_t* c_Ho;
    rocke_value_t* c_N;
    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;
    rocke_value_t* by;
    rocke_value_t* ch_chunk;
    rocke_value_t* c_cpl;
    rocke_value_t* d_rsrc;
    rocke_value_t* rows_total;
    rocke_value_t* c_rc;
    rocke_value_t* r_first;
    rocke_value_t* row0;
    rocke_value_t* n_chunks;
    rocke_value_t* zero_f;
    rocke_value_t* c_u;
    rocke_value_t* accs[ROCKE_DW_WGRAD_MAX_ACC];

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_direct_depthwise_wgrad_validate(spec, reason, sizeof reason) != ROCKE_OK
       || !rocke_direct_depthwise_wgrad_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        dw_fail(b);
        return NULL;
    }

    memset(&cx, 0, sizeof cx);
    c->b = b;
    p = &spec->problem;
    KH = p->KH;
    KW = p->KW;
    U = p->stride;
    PAD = p->PAD;
    WAVE = spec->wave_size;
    CPL = spec->chan_per_lane;
    WAVES = spec->waves;
    RPW = spec->rows_per_wave;
    V = spec->wo_chunk;
    RS = spec->r_split;
    RC = rocke_direct_depthwise_wgrad_rows_per_group(spec);
    ROWW = rocke_direct_depthwise_wgrad_row_waves(spec);
    THREADS = rocke_direct_depthwise_wgrad_threads_per_block(spec);
    KK = KH * KW;
    BLOCK_CH = rocke_direct_depthwise_wgrad_block_ch(spec);
    XSPAN = (V - 1) * U + KW;
    HC = (spec->carry && KW > U) ? KW - U : 0;
    NEW = XSPAN - HC;
    NA = RC * KW * CPL;
    c->is_bf16 = p->dtype && strcmp(p->dtype, "bf16") == 0;
    c->CPL = CPL;
    c->V = V;
    c->U = U;
    c->KW = KW;
    c->RC = RC;
    c->XSPAN = XSPAN;
    c->HC = HC;
    c->NEW = NEW;
    c->NA = NA;
    if(NA > ROCKE_DW_WGRAD_MAX_ACC || RC > ROCKE_DW_WGRAD_MAX_RC || V > ROCKE_DW_WGRAD_MAX_CHUNK
       || XSPAN > ROCKE_DW_WGRAD_MAX_SPAN || RC * XSPAN + V > ROCKE_DW_WGRAD_MAX_LOADS || CPL > 8)
    {
        dw_fail(b);
        return NULL;
    }

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    c->io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    if(c->io_type == NULL)
        return NULL;
    rocke_dconv_emit_params(b, &params, "wgrad", c->io_type);
    c->params = &params;
    c->c0 = rocke_b_const_i32(b, 0);
    c1 = rocke_b_const_i32(b, 1);
    c->c2 = rocke_b_const_i32(b, 2);
    c_H = params.p_Hi;
    c->c_W = params.p_Wi;
    c_Ho = params.p_Ho;
    c->c_Wo = params.p_Wo;
    c_N = params.p_N;

    tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* c_wave = rocke_b_const_i32(b, WAVE);
        rocke_value_t* wave_raw = rocke_b_div(b, tid, c_wave);
        wave_id = rocke_b_readfirstlane(b, wave_raw);
        lane = rocke_b_mod(b, tid, c_wave);
    }
    {
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* c_block_ch;
        rocke_value_t* lane_ch;
        rocke_value_t* c_base;
        by = rocke_b_block_id_y(b);
        c_block_ch = rocke_b_const_i32(b, BLOCK_CH);
        ch_chunk = rocke_b_mul(b, bx, c_block_ch);
        c_cpl = rocke_b_const_i32(b, CPL);
        lane_ch = rocke_b_mul(b, lane, c_cpl);
        c_base = rocke_b_add(b, ch_chunk, lane_ch);
        c->a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
        c->b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
        d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);
        rows_total = rocke_b_mul(b, c_N, c_Ho);
        {
            rocke_value_t* c_rs = rocke_b_const_i32(b, RS);
            rocke_value_t* group = rocke_b_mod(b, wave_id, c_rs);
            rocke_value_t* row_wave = rocke_b_div(b, wave_id, c_rs);
            rocke_value_t* c_roww;
            rocke_value_t* blk_rows;
            rocke_value_t* wave_rows;
            rocke_value_t* c_rpw;
            c_rc = rocke_b_const_i32(b, RC);
            r_first = rocke_b_mul(b, group, c_rc);
            c_roww = rocke_b_const_i32(b, ROWW);
            blk_rows = rocke_b_mul(b, by, c_roww);
            wave_rows = rocke_b_add(b, blk_rows, row_wave);
            c_rpw = rocke_b_const_i32(b, RPW);
            row0 = rocke_b_mul(b, wave_rows, c_rpw);
        }
        {
            rocke_value_t* c_v_round = rocke_b_const_i32(b, V - 1);
            rocke_value_t* wo_round = rocke_b_add(b, c->c_Wo, c_v_round);
            c->c_v = rocke_b_const_i32(b, V);
            n_chunks = rocke_b_div(b, wo_round, c->c_v);
        }
        zero_f = rocke_b_const_f32(b, 0.0);
        c->lane_byte = rocke_b_mul(b, c_base, c->c2);
    }
    c->far = rocke_b_const_i32(b, ((int64_t)1 << 31) - ((int64_t)1 << 25));
    c->s_w_bytes = rocke_b_mul(b, params.p_B_stride_wi, c->c2);
    c->s_wo_bytes = rocke_b_mul(b, params.p_A_stride_wi, c->c2);
    for(t = 0; t < XSPAN; ++t)
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, t);
        c->col_x[t] = rocke_b_mul(b, c_t, c->s_w_bytes);
    }
    for(v = 0; v < V; ++v)
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, v);
        c->col_dy[v] = rocke_b_mul(b, c_t, c->s_wo_bytes);
    }
    c->c_vu = rocke_b_const_i32(b, V * U);
    c->c_pad = rocke_b_const_i32(b, PAD);
    c_u = rocke_b_const_i32(b, U);

    for(k = 0; k < NA; ++k)
        accs[k] = zero_f;
    for(i = 0; i < RPW; ++i)
    {
        rocke_value_t* row;
        rocke_value_t* row_ok;
        rocke_value_t* n_i;
        rocke_value_t* ho;
        rocke_value_t* dy_base;
        rocke_value_t* h0;
        rocke_value_t* x_n;
        rocke_value_t* x_bases[ROCKE_DW_WGRAD_MAX_RC];
        rocke_value_t* head0[ROCKE_DW_WGRAD_MAX_LOADS];
        rocke_value_t* pend0[ROCKE_DW_WGRAD_MAX_LOADS];
        rocke_iter_arg_t iargs[ROCKE_DW_WGRAD_MAX_ITER];
        char iv_name[32];
        int NH = 0, NP = 0, ni = 0;
        rocke_for_t loop;
        if(i)
        {
            rocke_value_t* c_i = rocke_b_const_i32(b, i);
            row = rocke_b_add(b, row0, c_i);
        }
        else
        {
            row = row0;
        }
        row_ok = rocke_b_cmp_lt(b, row, rows_total);
        n_i = rocke_b_div(b, row, c_Ho);
        ho = rocke_b_mod(b, row, c_Ho);
        {
            rocke_value_t* dy_n = rocke_b_mul(b, n_i, params.p_A_stride_n);
            rocke_value_t* dy_h = rocke_b_mul(b, ho, params.p_A_stride_hi);
            rocke_value_t* dy_row = rocke_b_add(b, dy_n, dy_h);
            rocke_value_t* dy_row_b = rocke_b_mul(b, dy_row, c->c2);
            dy_base = rocke_b_select(b, row_ok, dy_row_b, c->far);
        }
        {
            rocke_value_t* ho_u = rocke_b_mul(b, ho, c_u);
            h0 = rocke_b_sub(b, ho_u, c->c_pad);
        }
        x_n = rocke_b_mul(b, n_i, params.p_B_stride_n);
        for(rr = 0; rr < RC; ++rr)
        {
            rocke_value_t* r_rt;
            rocke_value_t* hi;
            rocke_value_t* ge;
            rocke_value_t* lt;
            rocke_value_t* h_ok;
            rocke_value_t* x_h;
            rocke_value_t* x_row;
            rocke_value_t* x_row_b;
            if(rr)
            {
                rocke_value_t* c_rr = rocke_b_const_i32(b, rr);
                r_rt = rocke_b_add(b, r_first, c_rr);
            }
            else
            {
                r_rt = r_first;
            }
            hi = rocke_b_add(b, h0, r_rt);
            ge = rocke_b_cmp_ge(b, hi, c->c0);
            lt = rocke_b_cmp_lt(b, hi, c_H);
            h_ok = rocke_b_land(b, ge, lt);
            h_ok = rocke_b_land(b, h_ok, row_ok);
            if(RC * RS != KH)
            {
                rocke_value_t* c_kh = rocke_b_const_i32(b, KH);
                rocke_value_t* r_ok = rocke_b_cmp_lt(b, r_rt, c_kh);
                h_ok = rocke_b_land(b, h_ok, r_ok);
            }
            x_h = rocke_b_mul(b, hi, params.p_B_stride_hi);
            x_row = rocke_b_add(b, x_n, x_h);
            x_row_b = rocke_b_mul(b, x_row, c->c2);
            x_bases[rr] = rocke_b_select(b, h_ok, x_row_b, c->far);
        }
        if(HC)
            dw_issue_x(c, x_bases, c->c0, 0, HC, head0, &NH);
        if(spec->prefetch)
        {
            dw_issue_dy(c, dy_base, c->c0, pend0, &NP);
            dw_issue_x(c, x_bases, c->c0, HC, XSPAN, pend0, &NP);
        }
        for(k = 0; k < NA; ++k)
        {
            snprintf(names[ni], sizeof names[0], "dwa%d_%d", i, k);
            iargs[ni].name = names[ni];
            iargs[ni].init = accs[k];
            ++ni;
        }
        for(k = 0; k < NH; ++k)
        {
            snprintf(names[ni], sizeof names[0], "dwh%d_%d", i, k);
            iargs[ni].name = names[ni];
            iargs[ni].init = head0[k];
            ++ni;
        }
        for(k = 0; k < NP; ++k)
        {
            snprintf(names[ni], sizeof names[0], "dwp%d_%d", i, k);
            iargs[ni].name = names[ni];
            iargs[ni].init = pend0[k];
            ++ni;
        }
        snprintf(iv_name, sizeof iv_name, "dwj%d", i);
        loop = rocke_b_scf_for_iter(b, c->c0, n_chunks, c1, iargs, ni, iv_name, false, true);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* cur[ROCKE_DW_WGRAD_MAX_ACC];
            rocke_value_t* head[ROCKE_DW_WGRAD_MAX_LOADS];
            rocke_value_t* pend[ROCKE_DW_WGRAD_MAX_LOADS];
            rocke_value_t* nxt[ROCKE_DW_WGRAD_MAX_LOADS];
            rocke_value_t* fresh[ROCKE_DW_WGRAD_MAX_LOADS];
            rocke_value_t* spans[ROCKE_DW_WGRAD_MAX_LOADS];
            rocke_value_t* yields[ROCKE_DW_WGRAD_MAX_ITER];
            int nn = 0, nf = 0, ny = 0;
            for(k = 0; k < NA; ++k)
                cur[k] = loop.iter_vars[k];
            for(k = 0; k < NH; ++k)
                head[k] = loop.iter_vars[NA + k];
            if(spec->prefetch)
            {
                rocke_value_t* j1;
                for(k = 0; k < NP; ++k)
                    pend[k] = loop.iter_vars[NA + NH + k];
                j1 = rocke_b_add(b, loop.iv, c1);
                dw_issue_dy(c, dy_base, j1, nxt, &nn);
                dw_issue_x(c, x_bases, j1, HC, XSPAN, nxt, &nn);
                dw_spans(c, head, pend + V, spans);
                dw_fma(c, cur, pend, spans);
            }
            else
            {
                rocke_value_t* dy_vecs[ROCKE_DW_WGRAD_MAX_CHUNK];
                int nd = 0;
                dw_issue_dy(c, dy_base, loop.iv, dy_vecs, &nd);
                dw_issue_x(c, x_bases, loop.iv, HC, XSPAN, fresh, &nf);
                dw_spans(c, head, fresh, spans);
                dw_fma(c, cur, dy_vecs, spans);
            }
            for(k = 0; k < NA; ++k)
                yields[ny++] = cur[k];
            if(HC)
            {
                for(rr = 0; rr < RC; ++rr)
                    for(t = 0; t < HC; ++t)
                        yields[ny++] = spans[rr * XSPAN + V * U + t];
            }
            for(k = 0; k < nn; ++k)
                yields[ny++] = nxt[k];
            rocke_b_scf_yield(b, yields, ny);
        }
        rocke_b_region_leave(b);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(k = 0; k < NA; ++k)
            accs[k] = loop.op->results[k];
    }

    /* ---- epilogue: LDS reduce over the waves, then coalesced replica atomics.
     * Wave w stores its taps at [w][r * KW + s][lane * CPL + e]; tap row r is
     * owned by wave group r // RC, so a tap sums that group's ROWW row waves. */
    TILE = KK * BLOCK_CH;
    {
        rocke_value_t* lds;
        rocke_value_t* c_tile;
        rocke_value_t* w_base;
        rocke_value_t* lane_off;
        rocke_value_t* c_row_stride;
        rocke_value_t* rep_c;
        rocke_value_t* c_c;
        rocke_value_t* c_kk;
        rocke_value_t* c_kw;
        rocke_value_t* c_block_ch2;
        rocke_value_t* c4;
        {
            int shape[2];
            shape[0] = 1;
            shape[1] = WAVES * TILE;
            lds = rocke_b_smem_alloc(b, rocke_f32(), shape, 2, "dw_red");
        }
        c_tile = rocke_b_const_i32(b, TILE);
        w_base = rocke_b_mul(b, wave_id, c_tile);
        lane_off = rocke_b_mul(b, lane, c_cpl);
        c_row_stride = rocke_b_const_i32(b, KW * BLOCK_CH);
        for(rr = 0; rr < RC; ++rr)
        {
            rocke_value_t* r_rt;
            rocke_value_t* r_off;
            rocke_value_t* r_base;
            rocke_value_t* row_base;
            rocke_value_t* r_ok = NULL;
            if(rr)
            {
                rocke_value_t* c_rr = rocke_b_const_i32(b, rr);
                r_rt = rocke_b_add(b, r_first, c_rr);
            }
            else
            {
                r_rt = r_first;
            }
            r_off = rocke_b_mul(b, r_rt, c_row_stride);
            r_base = rocke_b_add(b, w_base, r_off);
            row_base = rocke_b_add(b, r_base, lane_off);
            if(RC * RS != KH)
            {
                rocke_value_t* c_kh = rocke_b_const_i32(b, KH);
                r_ok = rocke_b_cmp_lt(b, r_rt, c_kh);
            }
            for(s = 0; s < KW; ++s)
            {
                for(e = 0; e < CPL; ++e)
                {
                    rocke_value_t* c_se = rocke_b_const_i32(b, s * BLOCK_CH + e);
                    rocke_value_t* idx[2];
                    rocke_value_t* comp[1];
                    rocke_value_t* val;
                    idx[0] = c->c0;
                    idx[1] = rocke_b_add(b, row_base, c_se);
                    comp[0] = accs[(rr * KW + s) * CPL + e];
                    val = rocke_b_vec_pack(b, comp, 1, rocke_f32());
                    if(r_ok == NULL)
                    {
                        rocke_b_smem_store_vN(b, lds, idx, 2, val, 1);
                    }
                    else
                    {
                        rocke_if_t gate = rocke_b_scf_if(b, r_ok);
                        rocke_b_region_enter(b, gate.then_region);
                        rocke_b_smem_store_vN(b, lds, idx, 2, val, 1);
                        rocke_b_region_leave(b);
                    }
                }
            }
        }
        rocke_b_sync_lds_only(b);
        {
            rocke_value_t* c_r = rocke_b_const_i32(b, spec->ws_replicas);
            rocke_value_t* rep = rocke_b_mod(b, by, c_r);
            c_c = params.p_groups;
            rep_c = rocke_b_mul(b, rep, c_c);
        }
        c_kk = rocke_b_const_i32(b, KK);
        c_kw = rocke_b_const_i32(b, KW);
        c_block_ch2 = rocke_b_const_i32(b, BLOCK_CH);
        c4 = rocke_b_const_i32(b, 4);
        for(k0 = 0; k0 < TILE; k0 += THREADS)
        {
            rocke_value_t* el;
            rocke_value_t* cl;
            rocke_value_t* rs;
            rocke_value_t* own = NULL;
            rocke_value_t* tap;
            rocke_value_t* tap_cl;
            rocke_value_t* tot = NULL;
            rocke_value_t* c_g;
            rocke_value_t* ok;
            rocke_value_t* slab_c;
            rocke_value_t* slab_off;
            rocke_value_t* off;
            rocke_value_t* byte;
            if(k0)
            {
                rocke_value_t* c_k0 = rocke_b_const_i32(b, k0);
                el = rocke_b_add(b, tid, c_k0);
            }
            else
            {
                el = tid;
            }
            cl = rocke_b_div(b, el, c_kk);
            rs = rocke_b_mod(b, el, c_kk);
            if(RS > 1)
            {
                rocke_value_t* r_tap = rocke_b_div(b, rs, c_kw);
                own = rocke_b_div(b, r_tap, c_rc);
            }
            tap = rocke_b_mul(b, rs, c_block_ch2);
            tap_cl = rocke_b_add(b, tap, cl);
            for(w = 0; w < ROWW; ++w)
            {
                rocke_value_t* slot;
                rocke_value_t* slot_off;
                rocke_value_t* idx[2];
                rocke_value_t* ld;
                rocke_value_t* val;
                if(own != NULL)
                {
                    rocke_value_t* c_w = rocke_b_const_i32(b, w * RS);
                    slot = rocke_b_add(b, c_w, own);
                }
                else
                {
                    slot = rocke_b_const_i32(b, w);
                }
                slot_off = rocke_b_mul(b, slot, c_tile);
                idx[0] = c->c0;
                idx[1] = rocke_b_add(b, slot_off, tap_cl);
                ld = rocke_b_smem_load_vN(b, lds, idx, 2, rocke_f32(), 1);
                val = rocke_b_vec_extract(b, ld, 0);
                tot = (tot == NULL) ? val : rocke_b_fadd(b, tot, val);
            }
            c_g = rocke_b_add(b, ch_chunk, cl);
            ok = rocke_b_cmp_lt(b, c_g, c_c);
            if(TILE % THREADS)
            {
                rocke_value_t* c_tile_n = rocke_b_const_i32(b, TILE);
                rocke_value_t* in_tile = rocke_b_cmp_lt(b, el, c_tile_n);
                ok = rocke_b_land(b, ok, in_tile);
            }
            slab_c = rocke_b_add(b, rep_c, c_g);
            slab_off = rocke_b_mul(b, slab_c, c_kk);
            off = rocke_b_add(b, slab_off, rs);
            byte = rocke_b_mul(b, off, c4);
            {
                rocke_if_t gate = rocke_b_scf_if(b, ok);
                rocke_b_region_enter(b, gate.then_region);
                rocke_b_buffer_atomic_fadd_f32(
                    b, d_rsrc, byte, c->c0, tot, ROCKE_DW_WGRAD_CPOL_ATOMIC_CASCADE_DEV);
                rocke_b_region_leave(b);
            }
        }
    }
    (void)WAVES;
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_direct_depthwise_wgrad_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_wgrad_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_direct_depthwise_wgrad_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_depthwise_wgrad(b, spec, arch);
    });
}
