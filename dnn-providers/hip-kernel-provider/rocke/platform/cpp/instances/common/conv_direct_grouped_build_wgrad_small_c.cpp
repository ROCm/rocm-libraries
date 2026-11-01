// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_wgrad_small_c.cpp -- C99 port of
 * _build_direct_conv_wgrad_small_c (library/kernels/common/conv_direct_grouped.py):
 * the small-channel direct wgrad (wave32 WMMA 16x16x32, gfx1250).
 *
 * dW is a GEMM with M = kpg and N = KH * KW * cpg (the flattened KRSC row, each
 * filter row's (s, c) run padded to the vec_x unit), reduced over the
 * N * Ho * Wo output pixels:
 *
 *   - Grid (n_k_tiles * n_n_tiles, row_blocks, 1): block x owns one
 *     block_k x block_c dW tile, block y one contiguous range of
 *     32 * sub_steps-pixel steps.
 *   - Per step and per 32 pixels every thread owns the pixel `lane` and stages
 *     8-wide groups of it: dY channels as one 16 B load each, X im2col columns
 *     gathered from NHWC X (16-bit loads, or vec_x-wide vectors of a filter
 *     row's contiguous (s, c) run). Both land as [32][16] LDS parts that
 *     ds_load_tr16_b128 reads back as the WMMA operands.
 *   - The next step's loads ride the pixel loop as iteration arguments.
 *   - With vec_x, the steps whose vector runs could cross the start or the end
 *     of X run a gathering copy of the loop.
 *
 * Epilogue: fp32 atomics into replica block_y % ws_replicas of the
 * [R][kpg][KH * KW * cpg] scratch that conv_wgrad_workspace_reduce folds.
 *
 * Every helper emits in PYTHON EVALUATION ORDER, one builder call per Python
 * statement, so the SSA sequence -- and therefore the .ll bytes -- match.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_op_for_shape */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV = 4 | (2 << 3). */
#define ROCKE_WGRAD_SC_CPOL_ATOMIC_CASCADE_DEV (4 | (2 << 3))

/* Fixed-capacity scratch. The validator caps the accumulators at 128 VGPRs
 * (16 tiles), the waves at 16 and sub_steps at 8; the pass counts follow from
 * block_k / block_c over the wave count. All checked before any write. */
#define ROCKE_WGRAD_SC_MAX_TILES 16
#define ROCKE_WGRAD_SC_MAX_PASSES 64
#define ROCKE_WGRAD_SC_MAX_PENDING 1024
#define ROCKE_WGRAD_SC_MAX_ITER (ROCKE_WGRAD_SC_MAX_TILES + ROCKE_WGRAD_SC_MAX_PENDING)

typedef struct rocke_wgrad_sc_col
{
    rocke_value_t* r;
    rocke_value_t* s;
    rocke_value_t* off;
    rocke_value_t* ok;
} rocke_wgrad_sc_col_t;

typedef struct rocke_wgrad_sc_sub
{
    rocke_value_t* r;
    rocke_value_t* off;
    rocke_value_t* s_e[8];
    rocke_value_t* ok_e[8];
} rocke_wgrad_sc_sub_t;

typedef struct rocke_wgrad_sc_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dconv_params_t* params;
    const rocke_type_t* io_type;
    int is_bf16;
    int M, PIX, VX, DY_PASSES, X_PASSES, DY_SUB, X_SUB, TK, TN, n_acc;
    const char* wmma_op_id;

    rocke_value_t* c0;
    rocke_value_t* c1;
    rocke_value_t* c2;
    rocke_value_t* c_U;
    rocke_value_t* c_PAD;
    rocke_value_t* c_PART;
    rocke_value_t* c_PIX_STEP;
    rocke_value_t* oob_sentinel;
    rocke_value_t* lane;
    rocke_value_t* n_pix;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* dy_lds;
    rocke_value_t* x_lds;
    rocke_value_t* zero_elem;
    rocke_value_t* tr_flat;
    rocke_value_t* c_tr_hi;

    rocke_value_t* dy_grp_ok[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* x_grp_ok[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* dy_runs[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* x_runs[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* dy_k[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* dy_k_ok[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_wgrad_sc_sub_t x_vec[ROCKE_WGRAD_SC_MAX_PASSES][2];
    rocke_wgrad_sc_col_t x_cols[ROCKE_WGRAD_SC_MAX_PASSES][8];
    rocke_value_t* dy_parts[ROCKE_WGRAD_SC_MAX_TILES];
    rocke_value_t* x_parts[ROCKE_WGRAD_SC_MAX_TILES];
} rocke_wgrad_sc_ctx_t;

static void sc_fail(rocke_ir_builder_t* b)
{
    if(b->status == ROCKE_OK)
    {
        b->status = ROCKE_ERR_VALUE;
    }
}

/* _buf_load_vN(b, io_dtype, rsrc, off, c0, dwords) */
static rocke_value_t*
    sc_buf_load(rocke_wgrad_sc_ctx_t* c, rocke_value_t* rsrc, rocke_value_t* off, int dwords)
{
    return c->is_bf16 ? rocke_b_buffer_load_vN_bf16(c->b, rsrc, off, c->c0, dwords)
                      : rocke_b_buffer_load_vN_f16(c->b, rsrc, off, c->c0, dwords);
}

/* _in_range(v, hi): 0 <= v < hi. */
static rocke_value_t* sc_in_range(rocke_wgrad_sc_ctx_t* c, rocke_value_t* v, rocke_value_t* hi)
{
    rocke_value_t* ge = rocke_b_cmp_ge(c->b, v, c->c0);
    rocke_value_t* lt = rocke_b_cmp_lt(c->b, v, hi);
    return rocke_b_land(c->b, ge, lt);
}

/* _lds_run(g): LDS element of group g, pixel lane: part g // 2, column
 * (g % 2) * 8. */
static rocke_value_t*
    sc_lds_run(rocke_wgrad_sc_ctx_t* c, rocke_value_t* g, rocke_value_t* c8, rocke_value_t* c16)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* half = rocke_b_div(b, g, c->c2);
    rocke_value_t* part = rocke_b_mul(b, half, c->c_PART);
    rocke_value_t* row = rocke_b_mul(b, c->lane, c16);
    rocke_value_t* base = rocke_b_add(b, part, row);
    rocke_value_t* col_half = rocke_b_mod(b, g, c->c2);
    rocke_value_t* col = rocke_b_mul(b, col_half, c8);
    return rocke_b_add(b, base, col);
}

/* _issue_px(gp, vec): loads of one 32-pixel chunk for this thread's pixel gp.
 * Appends to out[*n]. */
static void sc_issue_px(rocke_wgrad_sc_ctx_t* c,
                        rocke_value_t* gp,
                        bool vec,
                        rocke_value_t* c_Wo,
                        rocke_value_t* c_Ho,
                        rocke_value_t* c_H,
                        rocke_value_t* c_W,
                        rocke_value_t** out,
                        int* n)
{
    rocke_ir_builder_t* b = c->b;
    const rocke_dconv_params_t* params = c->params;
    rocke_value_t* p_ok = rocke_b_cmp_lt(b, gp, c->n_pix);
    rocke_value_t* wo = rocke_b_mod(b, gp, c_Wo);
    rocke_value_t* t = rocke_b_div(b, gp, c_Wo);
    rocke_value_t* ho = rocke_b_mod(b, t, c_Ho);
    rocke_value_t* n_i = rocke_b_div(b, t, c_Ho);
    rocke_value_t* dy_n = rocke_b_mul(b, n_i, params->p_A_stride_n);
    rocke_value_t* dy_h = rocke_b_mul(b, ho, params->p_A_stride_hi);
    rocke_value_t* dy_nh = rocke_b_add(b, dy_n, dy_h);
    rocke_value_t* dy_w = rocke_b_mul(b, wo, params->p_A_stride_wi);
    rocke_value_t* dy_base = rocke_b_add(b, dy_nh, dy_w);
    rocke_value_t* ho_u;
    rocke_value_t* hb;
    rocke_value_t* wo_u;
    rocke_value_t* wb;
    rocke_value_t* x_n;
    rocke_value_t* x_h;
    rocke_value_t* x_nh;
    rocke_value_t* x_w;
    rocke_value_t* x_base;
    int j, e, q;

    for(j = 0; j < c->DY_PASSES; ++j)
    {
        rocke_value_t* ok = rocke_b_land(b, p_ok, c->dy_k_ok[j]);
        rocke_value_t* elem;
        rocke_value_t* byte;
        rocke_value_t* off;
        if(c->dy_grp_ok[j] != NULL)
            ok = rocke_b_land(b, ok, c->dy_grp_ok[j]);
        elem = rocke_b_add(b, dy_base, c->dy_k[j]);
        byte = rocke_b_mul(b, elem, c->c2);
        off = rocke_b_select(b, ok, byte, c->oob_sentinel);
        out[(*n)++] = sc_buf_load(c, c->a_rsrc, off, 4);
    }
    ho_u = rocke_b_mul(b, ho, c->c_U);
    hb = rocke_b_sub(b, ho_u, c->c_PAD);
    wo_u = rocke_b_mul(b, wo, c->c_U);
    wb = rocke_b_sub(b, wo_u, c->c_PAD);
    x_n = rocke_b_mul(b, n_i, params->p_B_stride_n);
    x_h = rocke_b_mul(b, hb, params->p_B_stride_hi);
    x_nh = rocke_b_add(b, x_n, x_h);
    x_w = rocke_b_mul(b, wb, params->p_B_stride_wi);
    x_base = rocke_b_add(b, x_nh, x_w);
    for(j = 0; j < c->X_PASSES; ++j)
    {
        rocke_value_t* elems[8];
        int ne = 0;
        if(vec)
        {
            for(q = 0; q < 8 / c->VX; ++q)
            {
                const rocke_wgrad_sc_sub_t* sub = &c->x_vec[j][q];
                rocke_value_t* hi = rocke_b_add(b, hb, sub->r);
                rocke_value_t* row_ok = sc_in_range(c, hi, c_H);
                rocke_value_t* elem;
                rocke_value_t* byte;
                rocke_value_t* off;
                rocke_value_t* xv;
                row_ok = rocke_b_land(b, row_ok, p_ok);
                if(c->x_grp_ok[j] != NULL)
                    row_ok = rocke_b_land(b, row_ok, c->x_grp_ok[j]);
                elem = rocke_b_add(b, x_base, sub->off);
                byte = rocke_b_mul(b, elem, c->c2);
                off = rocke_b_select(b, row_ok, byte, c->oob_sentinel);
                xv = sc_buf_load(c, c->b_rsrc, off, c->VX / 2);
                for(e = 0; e < c->VX; ++e)
                {
                    rocke_value_t* wi = rocke_b_add(b, wb, sub->s_e[e]);
                    rocke_value_t* ok = sc_in_range(c, wi, c_W);
                    rocke_value_t* el;
                    ok = rocke_b_land(b, ok, sub->ok_e[e]);
                    el = rocke_b_vec_extract(b, xv, e);
                    elems[ne++] = rocke_b_select(b, ok, el, c->zero_elem);
                }
            }
        }
        else
        {
            for(e = 0; e < 8; ++e)
            {
                const rocke_wgrad_sc_col_t* col = &c->x_cols[j][e];
                rocke_value_t* hi = rocke_b_add(b, hb, col->r);
                rocke_value_t* wi = rocke_b_add(b, wb, col->s);
                rocke_value_t* h_ok = sc_in_range(c, hi, c_H);
                rocke_value_t* w_ok = sc_in_range(c, wi, c_W);
                rocke_value_t* ok = rocke_b_land(b, h_ok, w_ok);
                rocke_value_t* elem;
                rocke_value_t* byte;
                rocke_value_t* off;
                ok = rocke_b_land(b, ok, col->ok);
                ok = rocke_b_land(b, ok, p_ok);
                if(c->x_grp_ok[j] != NULL)
                    ok = rocke_b_land(b, ok, c->x_grp_ok[j]);
                elem = rocke_b_add(b, x_base, col->off);
                byte = rocke_b_mul(b, elem, c->c2);
                off = rocke_b_select(b, ok, byte, c->oob_sentinel);
                elems[ne++] = c->is_bf16 ? rocke_b_buffer_load_bf16(b, c->b_rsrc, off, c->c0)
                                         : rocke_b_buffer_load_f16(b, c->b_rsrc, off, c->c0);
            }
        }
        out[(*n)++] = rocke_b_vec_pack(b, elems, ne, c->io_type);
    }
}

/* _issue(step, vec): loads of one step, sub_steps 32-pixel chunks. */
static void sc_issue(rocke_wgrad_sc_ctx_t* c,
                     rocke_value_t* step,
                     bool vec,
                     rocke_value_t* c_Wo,
                     rocke_value_t* c_Ho,
                     rocke_value_t* c_H,
                     rocke_value_t* c_W,
                     rocke_value_t** out,
                     int* n)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* first = rocke_b_mul(b, step, c->c_PIX_STEP);
    rocke_value_t* base = rocke_b_add(b, first, c->lane);
    int q;
    for(q = 0; q < c->M; ++q)
    {
        rocke_value_t* gp;
        if(q == 0)
        {
            gp = base;
        }
        else
        {
            rocke_value_t* c_q = rocke_b_const_i32(b, q * c->PIX);
            gp = rocke_b_add(b, base, c_q);
        }
        sc_issue_px(c, gp, vec, c_Wo, c_Ho, c_H, c_W, out, n);
    }
}

/* _store(smem, run, sub_off, val, ok) */
static void sc_store(rocke_wgrad_sc_ctx_t* c,
                     rocke_value_t* smem,
                     rocke_value_t* run,
                     int sub_off,
                     rocke_value_t* val,
                     rocke_value_t* ok)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* idx[2];
    idx[0] = c->c0;
    if(sub_off)
    {
        rocke_value_t* c_off = rocke_b_const_i32(b, sub_off);
        idx[1] = rocke_b_add(b, run, c_off);
    }
    else
    {
        idx[1] = run;
    }
    if(ok == NULL)
    {
        rocke_b_smem_store_vN(b, smem, idx, 2, val, 8);
    }
    else
    {
        rocke_if_t gate = rocke_b_scf_if(b, ok);
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_smem_store_vN(b, smem, idx, 2, val, 8);
        rocke_b_region_leave(b);
    }
}

/* _commit(vecs) */
static void sc_commit(rocke_wgrad_sc_ctx_t* c, rocke_value_t* const* vecs)
{
    int i = 0, q, j;
    for(q = 0; q < c->M; ++q)
    {
        for(j = 0; j < c->DY_PASSES; ++j)
        {
            sc_store(c, c->dy_lds, c->dy_runs[j], q * c->DY_SUB, vecs[i], c->dy_grp_ok[j]);
            ++i;
        }
        for(j = 0; j < c->X_PASSES; ++j)
        {
            sc_store(c, c->x_lds, c->x_runs[j], q * c->X_SUB, vecs[i], c->x_grp_ok[j]);
            ++i;
        }
    }
}

/* _tr_read(smem, part, sub_off) */
static rocke_value_t*
    sc_tr_read(rocke_wgrad_sc_ctx_t* c, rocke_value_t* smem, rocke_value_t* part, int sub_off)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* part_off = rocke_b_mul(b, part, c->c_PART);
    rocke_value_t* flat;
    rocke_value_t* base;
    rocke_value_t* base_hi;
    rocke_value_t* lo;
    rocke_value_t* hi;
    rocke_value_t* idx[2];
    if(sub_off)
    {
        rocke_value_t* c_off = rocke_b_const_i32(b, sub_off);
        flat = rocke_b_add(b, c->tr_flat, c_off);
    }
    else
    {
        flat = c->tr_flat;
    }
    base = rocke_b_add(b, part_off, flat);
    idx[0] = c->c0;
    idx[1] = base;
    lo = rocke_b_ds_read_tr16_b128(b, smem, idx, 2, c->io_type);
    base_hi = rocke_b_add(b, base, c->c_tr_hi);
    idx[1] = base_hi;
    hi = rocke_b_ds_read_tr16_b128(b, smem, idx, 2, c->io_type);
    return rocke_b_vec_concat(b, lo, hi);
}

/* _pixel_loop(lo, hi, accs0, vec, tag): returns false on builder error; on
 * success accs[] holds the loop results. */
static bool sc_pixel_loop(rocke_wgrad_sc_ctx_t* c,
                          rocke_value_t* lo,
                          rocke_value_t* hi,
                          rocke_value_t** accs,
                          bool vec,
                          const char* tag,
                          rocke_value_t* c_Wo,
                          rocke_value_t* c_Ho,
                          rocke_value_t* c_H,
                          rocke_value_t* c_W)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* pending0[ROCKE_WGRAD_SC_MAX_PENDING];
    rocke_iter_arg_t iargs[ROCKE_WGRAD_SC_MAX_ITER];
    char names[ROCKE_WGRAD_SC_MAX_ITER][32];
    char iv_name[32];
    int np0 = 0, ni = 0, i;
    rocke_for_t loop;
    const int n_acc = c->n_acc;

    sc_issue(c, lo, vec, c_Wo, c_Ho, c_H, c_W, pending0, &np0);
    for(i = 0; i < n_acc; ++i)
    {
        snprintf(names[ni], sizeof names[0], "sc_acc%s%d", tag, i);
        iargs[ni].name = names[ni];
        iargs[ni].init = accs[i];
        ++ni;
    }
    for(i = 0; i < np0; ++i)
    {
        snprintf(names[ni], sizeof names[0], "sc_pend%s%d", tag, i);
        iargs[ni].name = names[ni];
        iargs[ni].init = pending0[i];
        ++ni;
    }
    snprintf(iv_name, sizeof iv_name, "sc_step%s", tag);
    loop = rocke_b_scf_for_iter(b, lo, hi, c->c1, iargs, ni, iv_name, false, true);
    if(!rocke_ir_builder_ok(b))
        return false;
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* cur[ROCKE_WGRAD_SC_MAX_TILES];
        rocke_value_t* pending[ROCKE_WGRAD_SC_MAX_PENDING];
        rocke_value_t* nxt[ROCKE_WGRAD_SC_MAX_PENDING];
        rocke_value_t* yields[ROCKE_WGRAD_SC_MAX_ITER];
        rocke_value_t* nxt_step;
        int nn = 0, ny = 0, q, tk, tn;
        for(i = 0; i < n_acc; ++i)
            cur[i] = loop.iter_vars[i];
        for(i = 0; i < np0; ++i)
            pending[i] = loop.iter_vars[n_acc + i];
        sc_commit(c, pending);
        nxt_step = rocke_b_add(b, loop.iv, c->c1);
        sc_issue(c, nxt_step, vec, c_Wo, c_Ho, c_H, c_W, nxt, &nn);
        rocke_b_sync_lds_only(b);
        for(q = 0; q < c->M; ++q)
        {
            rocke_value_t* dy_frags[ROCKE_WGRAD_SC_MAX_TILES];
            rocke_value_t* x_frags[ROCKE_WGRAD_SC_MAX_TILES];
            for(tk = 0; tk < c->TK; ++tk)
                dy_frags[tk] = sc_tr_read(c, c->dy_lds, c->dy_parts[tk], q * c->DY_SUB);
            for(tn = 0; tn < c->TN; ++tn)
                x_frags[tn] = sc_tr_read(c, c->x_lds, c->x_parts[tn], q * c->X_SUB);
            rocke_b_s_setprio(b, 1);
            for(tk = 0; tk < c->TK; ++tk)
            {
                for(tn = 0; tn < c->TN; ++tn)
                {
                    i = tk * c->TN + tn;
                    cur[i]
                        = rocke_b_mma(b, c->wmma_op_id, dy_frags[tk], x_frags[tn], cur[i], NULL, 0);
                }
            }
            rocke_b_s_setprio(b, 0);
        }
        /* Drain this step's LDS reads before the next step's commits. */
        rocke_b_sync_lds_only(b);
        for(i = 0; i < n_acc; ++i)
            yields[ny++] = cur[i];
        for(i = 0; i < nn; ++i)
            yields[ny++] = nxt[i];
        rocke_b_scf_yield(b, yields, ny);
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return false;
    for(i = 0; i < n_acc; ++i)
        accs[i] = loop.op->results[i];
    return true;
}

/* _clamp(v): v clamped into [s_begin, s_end]. */
static rocke_value_t*
    sc_clamp(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* s_begin, rocke_value_t* s_end)
{
    rocke_value_t* below = rocke_b_cmp_lt(b, v, s_begin);
    rocke_value_t* above = rocke_b_cmp_lt(b, s_end, v);
    rocke_value_t* capped = rocke_b_select(b, above, s_end, v);
    return rocke_b_select(b, below, s_begin, capped);
}

rocke_kernel_def_t* rocke_dconv_wgrad_build_small_c(rocke_ir_builder_t* b,
                                                    const rocke_direct_conv_wgrad_spec_t* spec,
                                                    const char* arch)
{
    rocke_wgrad_sc_ctx_t cx;
    rocke_wgrad_sc_ctx_t* c = &cx;
    rocke_dconv_params_t params;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    int KH, KW, C, K, U, PAD, RSC, RUN, RUNP, NG;
    int WK, WN, TK, TN, NW, THREADS, BK, BN, PIX, M, PIX_STEP, PART, C_FRAG, VX;
    int N_N_TILES, DY_GROUPS, X_GROUPS;
    int j, q, e, t, tk, tn, slot;
    rocke_value_t* c8;
    rocke_value_t* c16;
    rocke_value_t* c_K;
    rocke_value_t* c_C;
    rocke_value_t* c_RUN;
    rocke_value_t* c_RUNP;
    rocke_value_t* c_NG;
    rocke_value_t* c_H;
    rocke_value_t* c_W;
    rocke_value_t* c_Ho;
    rocke_value_t* c_Wo;
    rocke_value_t* c_N;
    rocke_value_t* s_H;
    rocke_value_t* s_W;
    rocke_value_t* tid;
    rocke_value_t* c_wave;
    rocke_value_t* wave_id;
    rocke_value_t* c4;
    rocke_value_t* q_in_lane;
    rocke_value_t* c_WK;
    rocke_value_t* wave_k_id;
    rocke_value_t* wave_n_id;
    rocke_value_t* by;
    rocke_value_t* k_origin;
    rocke_value_t* n_origin;
    rocke_value_t* d_rsrc;
    rocke_value_t* c_step_round;
    rocke_value_t* s_begin;
    rocke_value_t* s_end;
    rocke_value_t* dy_grp[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* x_grp[ROCKE_WGRAD_SC_MAX_PASSES];
    rocke_value_t* accs[ROCKE_WGRAD_SC_MAX_TILES];
    rocke_value_t* zero_acc;

    memset(&cx, 0, sizeof cx);
    c->b = b;
    c->is_bf16 = p->dtype && strcmp(p->dtype, "bf16") == 0;
    KH = p->KH;
    KW = p->KW;
    C = p->cpg;
    K = p->kpg;
    U = p->stride;
    PAD = p->PAD;
    RSC = KH * KW * C;
    RUN = KW * C;
    RUNP = rocke_direct_conv_wgrad_small_c_run(spec);
    NG = rocke_direct_conv_wgrad_small_c_n(spec);

    /* wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(family="wmma", ...) */
    {
        const char* ab = c->is_bf16 ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op
            = target
                  ? rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, 32, NULL)
                  : NULL;
        if(op == NULL || op->op_id == NULL)
        {
            sc_fail(b);
            return NULL;
        }
        c->wmma_op_id = op->op_id;
    }

    WK = spec->waves_k;
    WN = spec->waves_c;
    TK = spec->wave_tiles_k;
    TN = spec->wave_tiles_c;
    NW = WK * WN;
    THREADS = rocke_direct_conv_wgrad_threads_per_block(spec);
    BK = rocke_direct_conv_wgrad_block_k(spec);
    BN = rocke_direct_conv_wgrad_block_c(spec);
    PIX = spec->mfma_k;
    M = spec->sub_steps;
    PIX_STEP = PIX * M;
    PART = PIX * 16;
    C_FRAG = 8;
    VX = spec->vec_x;
    N_N_TILES = (NG + BN - 1) / BN;
    c->DY_SUB = WK * TK * PART;
    c->X_SUB = WN * TN * PART;
    DY_GROUPS = BK / 8;
    X_GROUPS = BN / 8;
    c->DY_PASSES = (DY_GROUPS + NW - 1) / NW;
    c->X_PASSES = (X_GROUPS + NW - 1) / NW;
    c->M = M;
    c->PIX = PIX;
    c->VX = VX;
    c->TK = TK;
    c->TN = TN;
    c->n_acc = TK * TN;
    if(c->n_acc > ROCKE_WGRAD_SC_MAX_TILES || TK > ROCKE_WGRAD_SC_MAX_TILES
       || TN > ROCKE_WGRAD_SC_MAX_TILES || c->DY_PASSES > ROCKE_WGRAD_SC_MAX_PASSES
       || c->X_PASSES > ROCKE_WGRAD_SC_MAX_PASSES
       || M * (c->DY_PASSES + c->X_PASSES) > ROCKE_WGRAD_SC_MAX_PENDING
       || (VX != 0 && VX != 4 && VX != 8))
    {
        sc_fail(b);
        return NULL;
    }

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    c->io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    if(c->io_type == NULL)
    {
        return NULL;
    }
    rocke_dconv_emit_params(b, &params, "wgrad", c->io_type);
    c->params = &params;

    c->c0 = rocke_b_const_i32(b, 0);
    c->c1 = rocke_b_const_i32(b, 1);
    c->c2 = rocke_b_const_i32(b, 2);
    c8 = rocke_b_const_i32(b, 8);
    c16 = rocke_b_const_i32(b, 16);
    c->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    c->c_U = rocke_b_const_i32(b, U);
    c->c_PAD = rocke_b_const_i32(b, PAD);
    c_K = rocke_b_const_i32(b, K);
    c_C = rocke_b_const_i32(b, C);
    c_RUN = rocke_b_const_i32(b, RUN);
    c_RUNP = rocke_b_const_i32(b, RUNP);
    c_NG = rocke_b_const_i32(b, NG);
    c->c_PART = rocke_b_const_i32(b, PART);
    c->c_PIX_STEP = rocke_b_const_i32(b, PIX_STEP);
    c_H = params.p_Hi;
    c_W = params.p_Wi;
    c_Ho = params.p_Ho;
    c_Wo = params.p_Wo;
    c_N = params.p_N;
    s_H = params.p_B_stride_hi;
    s_W = params.p_B_stride_wi;

    tid = rocke_b_thread_id_x(b);
    c_wave = rocke_b_const_i32(b, spec->wave_size);
    wave_id = rocke_b_div(b, tid, c_wave);
    if(spec->uniform_wave)
        wave_id = rocke_b_readfirstlane(b, wave_id);
    c->lane = rocke_b_mod(b, tid, c_wave);
    c4 = rocke_b_div(b, c->lane, c16);
    q_in_lane = rocke_b_mod(b, c->lane, c16);
    c_WK = rocke_b_const_i32(b, WK);
    wave_k_id = rocke_b_mod(b, wave_id, c_WK);
    wave_n_id = rocke_b_div(b, wave_id, c_WK);

    /* ---- grid decode: bx = k_tile * n_n_tiles + n_tile, by = pixel block. */
    {
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* c_n_n_tiles;
        rocke_value_t* n_tile;
        rocke_value_t* k_tile;
        rocke_value_t* c_BK;
        rocke_value_t* c_BN;
        by = rocke_b_block_id_y(b);
        c_n_n_tiles = rocke_b_const_i32(b, N_N_TILES);
        n_tile = rocke_b_mod(b, bx, c_n_n_tiles);
        k_tile = rocke_b_div(b, bx, c_n_n_tiles);
        c_BK = rocke_b_const_i32(b, BK);
        k_origin = rocke_b_mul(b, k_tile, c_BK);
        c_BN = rocke_b_const_i32(b, BN);
        n_origin = rocke_b_mul(b, n_tile, c_BN);
    }

    c->a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    c->b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);

    /* ---- this block's steps: whole PIX_STEP runs of the flattened pixels. */
    {
        rocke_value_t* n_img_pix = rocke_b_mul(b, c_N, c_Ho);
        rocke_value_t* n_pix_round;
        rocke_value_t* total_steps;
        rocke_value_t* c_pb_round;
        rocke_value_t* steps_round;
        rocke_value_t* c_pb;
        rocke_value_t* steps_per_blk;
        rocke_value_t* s_next;
        rocke_value_t* s_next_lt;
        c->n_pix = rocke_b_mul(b, n_img_pix, c_Wo);
        c_step_round = rocke_b_const_i32(b, PIX_STEP - 1);
        n_pix_round = rocke_b_add(b, c->n_pix, c_step_round);
        total_steps = rocke_b_div(b, n_pix_round, c->c_PIX_STEP);
        c_pb_round = rocke_b_const_i32(b, spec->row_blocks - 1);
        steps_round = rocke_b_add(b, total_steps, c_pb_round);
        c_pb = rocke_b_const_i32(b, spec->row_blocks);
        steps_per_blk = rocke_b_div(b, steps_round, c_pb);
        s_begin = rocke_b_mul(b, by, steps_per_blk);
        s_next = rocke_b_add(b, s_begin, steps_per_blk);
        s_next_lt = rocke_b_cmp_lt(b, s_next, total_steps);
        s_end = rocke_b_select(b, s_next_lt, s_next, total_steps);
    }

    {
        int shape[2];
        shape[0] = 1;
        shape[1] = M * c->DY_SUB;
        c->dy_lds = rocke_b_smem_alloc(b, c->io_type, shape, 2, "dy_lds");
        shape[1] = M * c->X_SUB;
        c->x_lds = rocke_b_smem_alloc(b, c->io_type, shape, 2, "x_lds");
    }

    /* ---- load groups: pass j covers group wave_id + j * NW (_groups). */
    for(j = 0; j < c->DY_PASSES; ++j)
    {
        if(j == 0)
        {
            dy_grp[j] = wave_id;
        }
        else
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j * NW);
            dy_grp[j] = rocke_b_add(b, wave_id, c_j);
        }
        if((j + 1) * NW > DY_GROUPS)
        {
            rocke_value_t* c_ng = rocke_b_const_i32(b, DY_GROUPS);
            c->dy_grp_ok[j] = rocke_b_cmp_lt(b, dy_grp[j], c_ng);
        }
    }
    for(j = 0; j < c->X_PASSES; ++j)
    {
        if(j == 0)
        {
            x_grp[j] = wave_id;
        }
        else
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j * NW);
            x_grp[j] = rocke_b_add(b, wave_id, c_j);
        }
        if((j + 1) * NW > X_GROUPS)
        {
            rocke_value_t* c_ng = rocke_b_const_i32(b, X_GROUPS);
            c->x_grp_ok[j] = rocke_b_cmp_lt(b, x_grp[j], c_ng);
        }
    }
    for(j = 0; j < c->DY_PASSES; ++j)
        c->dy_runs[j] = sc_lds_run(c, dy_grp[j], c8, c16);
    for(j = 0; j < c->X_PASSES; ++j)
        c->x_runs[j] = sc_lds_run(c, x_grp[j], c8, c16);
    for(j = 0; j < c->DY_PASSES; ++j)
    {
        rocke_value_t* g8 = rocke_b_mul(b, dy_grp[j], c8);
        c->dy_k[j] = rocke_b_add(b, k_origin, g8);
        c->dy_k_ok[j] = rocke_b_cmp_lt(b, c->dy_k[j], c_K);
    }

    /* ---- X columns: vector runs, then gathered columns. */
    if(VX)
    {
        for(j = 0; j < c->X_PASSES; ++j)
        {
            rocke_value_t* g8 = rocke_b_mul(b, x_grp[j], c8);
            for(q = 0; q < 8 / VX; ++q)
            {
                rocke_wgrad_sc_sub_t* sub = &c->x_vec[j][q];
                rocke_value_t* g8q;
                rocke_value_t* n0;
                rocke_value_t* j0;
                rocke_value_t* n0_ok;
                rocke_value_t* r_off;
                if(q == 0)
                {
                    g8q = g8;
                }
                else
                {
                    rocke_value_t* c_q = rocke_b_const_i32(b, q * VX);
                    g8q = rocke_b_add(b, g8, c_q);
                }
                n0 = rocke_b_add(b, n_origin, g8q);
                sub->r = rocke_b_div(b, n0, c_RUNP);
                j0 = rocke_b_mod(b, n0, c_RUNP);
                n0_ok = rocke_b_cmp_lt(b, n0, c_NG);
                for(e = 0; e < VX; ++e)
                {
                    rocke_value_t* je;
                    rocke_value_t* j_ok;
                    if(e == 0)
                    {
                        je = j0;
                    }
                    else
                    {
                        rocke_value_t* c_e = rocke_b_const_i32(b, e);
                        je = rocke_b_add(b, j0, c_e);
                    }
                    sub->s_e[e] = rocke_b_div(b, je, c_C);
                    j_ok = rocke_b_cmp_lt(b, je, c_RUN);
                    sub->ok_e[e] = rocke_b_land(b, j_ok, n0_ok);
                }
                r_off = rocke_b_mul(b, sub->r, s_H);
                sub->off = rocke_b_add(b, r_off, j0);
            }
        }
    }
    for(j = 0; j < c->X_PASSES; ++j)
    {
        rocke_value_t* g8 = rocke_b_mul(b, x_grp[j], c8);
        for(e = 0; e < 8; ++e)
        {
            rocke_wgrad_sc_col_t* col = &c->x_cols[j][e];
            rocke_value_t* c_e = rocke_b_const_i32(b, e);
            rocke_value_t* ge = rocke_b_add(b, g8, c_e);
            rocke_value_t* n = rocke_b_add(b, n_origin, ge);
            rocke_value_t* jn;
            rocke_value_t* cc;
            rocke_value_t* r_off;
            rocke_value_t* s_off;
            rocke_value_t* rs_off;
            col->r = rocke_b_div(b, n, c_RUNP);
            jn = rocke_b_mod(b, n, c_RUNP);
            cc = rocke_b_mod(b, jn, c_C);
            col->s = rocke_b_div(b, jn, c_C);
            r_off = rocke_b_mul(b, col->r, s_H);
            s_off = rocke_b_mul(b, col->s, s_W);
            rs_off = rocke_b_add(b, r_off, s_off);
            col->off = rocke_b_add(b, rs_off, cc);
            col->ok = rocke_b_cmp_lt(b, n, c_NG);
            if(RUNP != RUN)
            {
                rocke_value_t* j_ok = rocke_b_cmp_lt(b, jn, c_RUN);
                col->ok = rocke_b_land(b, col->ok, j_ok);
            }
        }
    }

    {
        rocke_value_t* zv = rocke_b_zero_vec(b, c->io_type, 1);
        c->zero_elem = rocke_b_vec_extract(b, zv, 0);
    }

    /* wave32 transpose-read address (see rocke_dconv_wgrad_build_image). */
    {
        rocke_value_t* tr_blk = rocke_b_div(b, q_in_lane, c8);
        rocke_value_t* tr_col = rocke_b_mul(b, tr_blk, c8);
        rocke_value_t* tr_hi = rocke_b_mul(b, c4, c16);
        rocke_value_t* tr_lo = rocke_b_mod(b, c->lane, c8);
        rocke_value_t* tr_row = rocke_b_add(b, tr_hi, tr_lo);
        rocke_value_t* tr_rowf = rocke_b_mul(b, tr_row, c16);
        c->tr_flat = rocke_b_add(b, tr_rowf, tr_col);
        c->c_tr_hi = rocke_b_const_i32(b, 8 * 16);
    }

    {
        rocke_value_t* c_TK = rocke_b_const_i32(b, TK);
        rocke_value_t* wk_parts = rocke_b_mul(b, wave_k_id, c_TK);
        rocke_value_t* c_TN;
        rocke_value_t* wn_parts;
        for(t = 0; t < TK; ++t)
        {
            rocke_value_t* c_t = rocke_b_const_i32(b, t);
            c->dy_parts[t] = rocke_b_add(b, wk_parts, c_t);
        }
        c_TN = rocke_b_const_i32(b, TN);
        wn_parts = rocke_b_mul(b, wave_n_id, c_TN);
        for(t = 0; t < TN; ++t)
        {
            rocke_value_t* c_t = rocke_b_const_i32(b, t);
            c->x_parts[t] = rocke_b_add(b, wn_parts, c_t);
        }
    }

    zero_acc = rocke_b_zero_vec_f32(b, C_FRAG);
    for(t = 0; t < c->n_acc; ++t)
        accs[t] = zero_acc;
    if(!VX)
    {
        if(!sc_pixel_loop(c, s_begin, s_end, accs, false, "", c_Wo, c_Ho, c_H, c_W))
            return NULL;
    }
    else
    {
        /* Gathering steps: image 0's rows that reach hi == 0 and the last
         * image's rows that reach hi == H - 1. */
        const int lo_rows = PAD ? PAD / U + 1 : 0;
        rocke_value_t* c_lo_rows = rocke_b_const_i32(b, lo_rows);
        rocke_value_t* lo_pix = rocke_b_mul(b, c_lo_rows, c_Wo);
        rocke_value_t* lo_pix_round = rocke_b_add(b, lo_pix, c_step_round);
        rocke_value_t* s_lo = rocke_b_div(b, lo_pix_round, c->c_PIX_STEP);
        rocke_value_t* h_pad = rocke_b_add(b, c_H, c->c_PAD);
        rocke_value_t* c_KH = rocke_b_const_i32(b, KH);
        rocke_value_t* h_reach = rocke_b_sub(b, h_pad, c_KH);
        rocke_value_t* hi_ho = rocke_b_div(b, h_reach, c->c_U);
        rocke_value_t* n_last = rocke_b_sub(b, c_N, c->c1);
        rocke_value_t* last_rows = rocke_b_mul(b, n_last, c_Ho);
        rocke_value_t* hi_rows = rocke_b_add(b, last_rows, hi_ho);
        rocke_value_t* hi_pix = rocke_b_mul(b, hi_rows, c_Wo);
        rocke_value_t* s_hi = rocke_b_div(b, hi_pix, c->c_PIX_STEP);
        rocke_value_t* mid_begin = sc_clamp(b, s_lo, s_begin, s_end);
        rocke_value_t* mid_end = sc_clamp(b, s_hi, s_begin, s_end);
        rocke_value_t* mid_empty = rocke_b_cmp_lt(b, mid_end, mid_begin);
        mid_end = rocke_b_select(b, mid_empty, mid_begin, mid_end);
        if(!sc_pixel_loop(c, s_begin, mid_begin, accs, false, "a", c_Wo, c_Ho, c_H, c_W))
            return NULL;
        if(!sc_pixel_loop(c, mid_begin, mid_end, accs, true, "b", c_Wo, c_Ho, c_H, c_W))
            return NULL;
        if(!sc_pixel_loop(c, mid_end, s_end, accs, false, "c", c_Wo, c_Ho, c_H, c_W))
            return NULL;
    }

    /* ---- epilogue: cascade atomics into replica slab (by % R) of [R][K][RSC]. */
    {
        rocke_value_t* c_R = rocke_b_const_i32(b, spec->ws_replicas);
        rocke_value_t* rep = rocke_b_mod(b, by, c_R);
        rocke_value_t* c_slab = rocke_b_const_i32(b, K * RSC);
        rocke_value_t* ws_shift = rocke_b_mul(b, rep, c_slab);
        rocke_value_t* c_wk_tile = rocke_b_const_i32(b, 16 * TK);
        rocke_value_t* k_wave = rocke_b_mul(b, wave_k_id, c_wk_tile);
        rocke_value_t* k_wave_origin = rocke_b_add(b, k_origin, k_wave);
        rocke_value_t* k_row = rocke_b_mul(b, c4, c8);
        rocke_value_t* k_base = rocke_b_add(b, k_wave_origin, k_row);
        rocke_value_t* c_wn_tile = rocke_b_const_i32(b, 16 * TN);
        rocke_value_t* n_wave = rocke_b_mul(b, wave_n_id, c_wn_tile);
        rocke_value_t* n_wave_origin = rocke_b_add(b, n_origin, n_wave);
        rocke_value_t* n_base = rocke_b_add(b, n_wave_origin, q_in_lane);
        rocke_value_t* c_RSC = rocke_b_const_i32(b, RSC);
        rocke_value_t* c4_bytes = rocke_b_const_i32(b, 4);
        for(tk = 0; tk < TK; ++tk)
        {
            for(tn = 0; tn < TN; ++tn)
            {
                rocke_value_t* n;
                rocke_value_t* n_ok;
                rocke_value_t* acc;
                if(tn == 0)
                {
                    n = n_base;
                }
                else
                {
                    rocke_value_t* c_tn = rocke_b_const_i32(b, 16 * tn);
                    n = rocke_b_add(b, n_base, c_tn);
                }
                n_ok = rocke_b_cmp_lt(b, n, c_NG);
                if(RUNP != RUN)
                {
                    /* Padded column -> dW column: row n // RUNP, offset n % RUNP. */
                    rocke_value_t* jj = rocke_b_mod(b, n, c_RUNP);
                    rocke_value_t* j_ok = rocke_b_cmp_lt(b, jj, c_RUN);
                    rocke_value_t* rr;
                    rocke_value_t* rr_off;
                    n_ok = rocke_b_land(b, n_ok, j_ok);
                    rr = rocke_b_div(b, n, c_RUNP);
                    rr_off = rocke_b_mul(b, rr, c_RUN);
                    n = rocke_b_add(b, rr_off, jj);
                }
                acc = accs[tk * TN + tn];
                for(slot = 0; slot < C_FRAG; ++slot)
                {
                    rocke_value_t* c_slot = rocke_b_const_i32(b, 16 * tk + slot);
                    rocke_value_t* k = rocke_b_add(b, k_base, c_slot);
                    rocke_value_t* k_ok = rocke_b_cmp_lt(b, k, c_K);
                    rocke_value_t* ok = rocke_b_land(b, n_ok, k_ok);
                    rocke_value_t* k_off = rocke_b_mul(b, k, c_RSC);
                    rocke_value_t* dw_off = rocke_b_add(b, k_off, n);
                    rocke_value_t* byte;
                    rocke_value_t* val;
                    dw_off = rocke_b_add(b, dw_off, ws_shift);
                    byte = rocke_b_mul(b, dw_off, c4_bytes);
                    val = rocke_b_vec_extract(b, acc, slot);
                    {
                        rocke_if_t gate = rocke_b_scf_if(b, ok);
                        rocke_b_region_enter(b, gate.then_region);
                        rocke_b_buffer_atomic_fadd_f32(
                            b, d_rsrc, byte, c->c0, val, ROCKE_WGRAD_SC_CPOL_ATOMIC_CASCADE_DEV);
                        rocke_b_region_leave(b);
                    }
                }
            }
        }
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}
