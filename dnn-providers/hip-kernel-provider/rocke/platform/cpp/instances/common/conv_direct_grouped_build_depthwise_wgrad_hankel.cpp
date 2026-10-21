// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_wgrad_hankel.cpp -- C99 port of
 * build_direct_depthwise_wgrad_hankel (library/kernels/common/conv_direct_grouped.py):
 * depthwise wgrad (cpg = kpg = 1, stride 1) on WMMA 16x16x32 through Hankel
 * operands.
 *
 * For channel c, input row hi and 32 output columns q0 .. q0+31:
 *     A[(ch, s)][q] = X[c][hi][q0 + q + s - PAD]     (a Hankel matrix)
 *     B[q][(ch, r)] = dY[c][hi - r + PAD][q0 + q]
 *     dW[c][r][s]  += (A . B)[(ch, s)][(ch, r)]
 * so one WMMA covers all KH x KW taps of CH channels for one input row and 32
 * columns. Per input row the block stages X row hi and dY row hi + PAD into
 * LDS (channel-planar; dY as a KH-row ring), one row of loads ahead.
 *
 * Epilogue: the diagonal blocks of each C tile go out as fp32 atomics into
 * replica (linear block id % ws_replicas) of a [ws_replicas][C][KH*KW] scratch.
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
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_op_for_shape */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV = 4 | (2 << 3). */
#define ROCKE_DW_HK_CPOL_ATOMIC_CASCADE_DEV (4 | (2 << 3))

/* Fixed-capacity scratch, checked before any write. */
#define ROCKE_DW_HK_MAX_ITEMS 512
#define ROCKE_DW_HK_MAX_TILES 64
#define ROCKE_DW_HK_MAX_ITER (ROCKE_DW_HK_MAX_TILES + 2 * ROCKE_DW_HK_MAX_ITEMS)

typedef struct rocke_dw_hk_item
{
    rocke_value_t* ok; /* in-image and in-range, row validity excluded */
    rocke_value_t* col_off; /* element offset within the row */
    rocke_value_t* idx; /* LDS element index */
    rocke_value_t* e_ok; /* NULL unless the item can run past the tile */
} rocke_dw_hk_item_t;

typedef struct rocke_dw_hk_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dconv_params_t* params;
    const rocke_type_t* io_type;
    int is_bf16;
    int X_ITEMS, DY_ITEMS;
    rocke_value_t* c0;
    rocke_value_t* c2;
    rocke_value_t* c8;
    rocke_value_t* oob_sentinel;
    rocke_value_t* c_H;
    rocke_value_t* c_Ho;
    rocke_value_t* c_KH;
    rocke_value_t* c_QB;
    rocke_value_t* c_XCOLS;
    rocke_value_t* c_pad;
    rocke_value_t* c_ring;
    rocke_value_t* x_n;
    rocke_value_t* dy_n;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* x_lds;
    rocke_value_t* dy_lds;
    rocke_value_t* a_ch;
    rocke_value_t* a_s;
    rocke_value_t* a_live;
    rocke_value_t* b_ch;
    rocke_value_t* b_r;
    rocke_value_t* b_live;
    rocke_value_t* k_off;
    rocke_dw_hk_item_t x_cols[ROCKE_DW_HK_MAX_ITEMS];
    rocke_dw_hk_item_t dy_cols[ROCKE_DW_HK_MAX_ITEMS];
} rocke_dw_hk_ctx_t;

static void hk_fail(rocke_ir_builder_t* b)
{
    if(b->status == ROCKE_OK)
    {
        b->status = ROCKE_ERR_VALUE;
    }
}

/* _load(rsrc, ok, elem) */
static rocke_value_t*
    hk_load(rocke_dw_hk_ctx_t* c, rocke_value_t* rsrc, rocke_value_t* ok, rocke_value_t* elem)
{
    rocke_value_t* byte = rocke_b_mul(c->b, elem, c->c2);
    rocke_value_t* off = rocke_b_select(c->b, ok, byte, c->oob_sentinel);
    return c->is_bf16 ? rocke_b_buffer_load_bf16(c->b, rsrc, off, c->c0)
                      : rocke_b_buffer_load_f16(c->b, rsrc, off, c->c0);
}

/* _stage_x(hi) / _stage_dy(ho): appends one row of loads to out[*n]. */
static void hk_stage(rocke_dw_hk_ctx_t* c, rocke_value_t* h, bool is_x, rocke_value_t** out, int* n)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* ge = rocke_b_cmp_ge(b, h, c->c0);
    rocke_value_t* lt = rocke_b_cmp_lt(b, h, is_x ? c->c_H : c->c_Ho);
    rocke_value_t* h_ok = rocke_b_land(b, ge, lt);
    rocke_value_t* h_off
        = rocke_b_mul(b, h, is_x ? c->params->p_B_stride_hi : c->params->p_A_stride_hi);
    rocke_value_t* row = rocke_b_add(b, is_x ? c->x_n : c->dy_n, h_off);
    const rocke_dw_hk_item_t* items = is_x ? c->x_cols : c->dy_cols;
    int count = is_x ? c->X_ITEMS : c->DY_ITEMS;
    int k;
    for(k = 0; k < count; ++k)
    {
        rocke_value_t* okk = rocke_b_land(b, h_ok, items[k].ok);
        rocke_value_t* elem = rocke_b_add(b, row, items[k].col_off);
        out[(*n)++] = hk_load(c, is_x ? c->b_rsrc : c->a_rsrc, okk, elem);
    }
}

/* _store(smem, idx, val, e_ok) */
static void hk_store(rocke_dw_hk_ctx_t* c,
                     rocke_value_t* smem,
                     rocke_value_t* idx,
                     rocke_value_t* val,
                     rocke_value_t* e_ok)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* comp[1];
    rocke_value_t* vec;
    rocke_value_t* ix[2];
    comp[0] = val;
    vec = rocke_b_vec_pack(b, comp, 1, c->io_type);
    ix[0] = c->c0;
    ix[1] = idx;
    if(e_ok == NULL)
    {
        rocke_b_smem_store_vN(b, smem, ix, 2, vec, 1);
    }
    else
    {
        rocke_if_t gate = rocke_b_scf_if(b, e_ok);
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_smem_store_vN(b, smem, ix, 2, vec, 1);
        rocke_b_region_leave(b);
    }
}

/* _slot_off(ho) */
static rocke_value_t* hk_slot_off(rocke_dw_hk_ctx_t* c, rocke_value_t* ho)
{
    rocke_value_t* ho_k = rocke_b_add(c->b, ho, c->c_KH);
    rocke_value_t* slot = rocke_b_mod(c->b, ho_k, c->c_KH);
    return rocke_b_mul(c->b, slot, c->c_ring);
}

/* _commit_dy(vals, ho) */
static void hk_commit_dy(rocke_dw_hk_ctx_t* c, rocke_value_t* const* vals, rocke_value_t* ho)
{
    rocke_value_t* base = hk_slot_off(c, ho);
    int k;
    for(k = 0; k < c->DY_ITEMS; ++k)
    {
        rocke_value_t* ring_idx = rocke_b_add(c->b, base, c->dy_cols[k].idx);
        hk_store(c, c->dy_lds, ring_idx, vals[k], c->dy_cols[k].e_ok);
    }
}

/* _a_frag(ch0) */
static rocke_value_t* hk_a_frag(rocke_dw_hk_ctx_t* c, rocke_value_t* ch0)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* ch = rocke_b_add(b, ch0, c->a_ch);
    rocke_value_t* row = rocke_b_mul(b, ch, c->c_XCOLS);
    rocke_value_t* base = rocke_b_add(b, row, c->k_off);
    rocke_value_t* safe;
    rocke_value_t* els[16];
    int i;
    base = rocke_b_add(b, base, c->a_s);
    safe = rocke_b_select(b, c->a_live, base, c->c0);
    for(i = 0; i < 16; ++i)
    {
        rocke_value_t* ix[2];
        rocke_value_t* ld;
        ix[0] = c->c0;
        if(i)
        {
            rocke_value_t* c_i = rocke_b_const_i32(b, i);
            ix[1] = rocke_b_add(b, safe, c_i);
        }
        else
        {
            ix[1] = safe;
        }
        ld = rocke_b_smem_load_vN(b, c->x_lds, ix, 2, c->io_type, 1);
        els[i] = rocke_b_vec_extract(b, ld, 0);
    }
    return rocke_b_vec_pack(b, els, 16, c->io_type);
}

/* _b_frag(ch0, hi) */
static rocke_value_t* hk_b_frag(rocke_dw_hk_ctx_t* c, rocke_value_t* ch0, rocke_value_t* hi)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* ch = rocke_b_add(b, ch0, c->b_ch);
    rocke_value_t* h_r = rocke_b_sub(b, hi, c->b_r);
    rocke_value_t* ho = rocke_b_add(b, h_r, c->c_pad);
    rocke_value_t* slot = hk_slot_off(c, ho);
    rocke_value_t* ch_off = rocke_b_mul(b, ch, c->c_QB);
    rocke_value_t* base = rocke_b_add(b, slot, ch_off);
    rocke_value_t* safe;
    rocke_value_t* lo;
    rocke_value_t* hi8;
    rocke_value_t* ix[2];
    base = rocke_b_add(b, base, c->k_off);
    safe = rocke_b_select(b, c->b_live, base, c->c0);
    ix[0] = c->c0;
    ix[1] = safe;
    lo = rocke_b_smem_load_vN(b, c->dy_lds, ix, 2, c->io_type, 8);
    ix[1] = rocke_b_add(b, safe, c->c8);
    hi8 = rocke_b_smem_load_vN(b, c->dy_lds, ix, 2, c->io_type, 8);
    return rocke_b_vec_concat(b, lo, hi8);
}

/* _items(n_items, n_elems): per-thread (col, ch, c_g, c_ok, e_ok). */
static void hk_items(rocke_ir_builder_t* b,
                     rocke_value_t* tid,
                     int threads,
                     int n_items,
                     int n_elems,
                     rocke_value_t* c_CB,
                     rocke_value_t* c_C,
                     rocke_value_t* c_base,
                     rocke_value_t** col,
                     rocke_value_t** ch,
                     rocke_value_t** c_g,
                     rocke_value_t** c_ok,
                     rocke_value_t** e_ok)
{
    int k;
    for(k = 0; k < n_items; ++k)
    {
        rocke_value_t* e;
        if(k)
        {
            rocke_value_t* c_k = rocke_b_const_i32(b, k * threads);
            e = rocke_b_add(b, tid, c_k);
        }
        else
        {
            e = tid;
        }
        col[k] = rocke_b_div(b, e, c_CB);
        ch[k] = rocke_b_mod(b, e, c_CB);
        c_g[k] = rocke_b_add(b, c_base, ch[k]);
        c_ok[k] = rocke_b_cmp_lt(b, c_g[k], c_C);
        e_ok[k] = NULL;
        if((k + 1) * threads > n_elems)
        {
            rocke_value_t* c_n = rocke_b_const_i32(b, n_elems);
            e_ok[k] = rocke_b_cmp_lt(b, e, c_n);
            c_ok[k] = rocke_b_land(b, c_ok[k], e_ok[k]);
        }
    }
}

rocke_kernel_def_t* rocke_build_direct_depthwise_wgrad_hankel(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_wgrad_hankel_spec_t* spec, const char* arch)
{
    rocke_dw_hk_ctx_t cx;
    rocke_dw_hk_ctx_t* c = &cx;
    rocke_dconv_params_t params;
    const rocke_direct_conv_problem_t* p;
    const char* wmma_op_id;
    char reason[ROCKE_ERR_MSG_CAP];
    char names[ROCKE_DW_HK_MAX_ITER][24];
    int KH, KW, PAD, CH, CB, TPW, THREADS, QB, XCOLS, KK, X_ELEMS, DY_ELEMS;
    int t, i, k;
    rocke_value_t* c1;
    rocke_value_t* c16;
    rocke_value_t* c_W;
    rocke_value_t* c_Wo;
    rocke_value_t* c_C;
    rocke_value_t* c_CB;
    rocke_value_t* c_KW;
    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;
    rocke_value_t* n_q;
    rocke_value_t* bx;
    rocke_value_t* rb;
    rocke_value_t* n_i;
    rocke_value_t* c_base;
    rocke_value_t* q0;
    rocke_value_t* d_rsrc;
    rocke_value_t* h_begin;
    rocke_value_t* h_end;
    rocke_value_t* khalf;
    rocke_value_t* c_mk;
    rocke_value_t* tile_ch0[ROCKE_DW_HK_MAX_TILES];
    rocke_value_t* accs[ROCKE_DW_HK_MAX_TILES];
    rocke_value_t* it_col[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* it_ch[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* it_cg[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* it_cok[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* it_eok[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* dt_col[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* dt_ch[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* dt_cg[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* dt_cok[ROCKE_DW_HK_MAX_ITEMS];
    rocke_value_t* dt_eok[ROCKE_DW_HK_MAX_ITEMS];

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(rocke_direct_depthwise_wgrad_hankel_validate(spec, reason, sizeof reason) != ROCKE_OK
       || !rocke_direct_depthwise_wgrad_hankel_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        hk_fail(b);
        return NULL;
    }
    memset(&cx, 0, sizeof cx);
    c->b = b;
    p = &spec->problem;
    KH = p->KH;
    KW = p->KW;
    PAD = p->PAD;
    CH = rocke_direct_depthwise_wgrad_hankel_channels_per_tile(spec);
    CB = rocke_direct_depthwise_wgrad_hankel_block_ch(spec);
    TPW = spec->tiles_per_wave;
    THREADS = rocke_direct_depthwise_wgrad_hankel_threads_per_block(spec);
    QB = 32;
    XCOLS = QB + KW - 1;
    KK = KH * KW;
    X_ELEMS = XCOLS * CB;
    DY_ELEMS = QB * CB;
    c->X_ITEMS = (X_ELEMS + THREADS - 1) / THREADS;
    c->DY_ITEMS = (DY_ELEMS + THREADS - 1) / THREADS;
    c->is_bf16 = p->dtype && strcmp(p->dtype, "bf16") == 0;
    if(TPW > ROCKE_DW_HK_MAX_TILES || c->X_ITEMS > ROCKE_DW_HK_MAX_ITEMS
       || c->DY_ITEMS > ROCKE_DW_HK_MAX_ITEMS)
    {
        hk_fail(b);
        return NULL;
    }
    {
        const char* ab = c->is_bf16 ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op
            = target
                  ? rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, 32, NULL)
                  : NULL;
        if(op == NULL || op->op_id == NULL)
        {
            hk_fail(b);
            return NULL;
        }
        wmma_op_id = op->op_id;
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
    c16 = rocke_b_const_i32(b, 16);
    c->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    c->c_H = params.p_Hi;
    c_W = params.p_Wi;
    c->c_Ho = params.p_Ho;
    c_Wo = params.p_Wo;
    c_C = params.p_groups;
    c_CB = rocke_b_const_i32(b, CB);
    c->c_QB = rocke_b_const_i32(b, QB);
    c->c_XCOLS = rocke_b_const_i32(b, XCOLS);
    c->c_KH = rocke_b_const_i32(b, KH);
    c_KW = rocke_b_const_i32(b, KW);
    c->c_pad = rocke_b_const_i32(b, PAD);

    tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* c_wave = rocke_b_const_i32(b, spec->wave_size);
        rocke_value_t* wave_raw = rocke_b_div(b, tid, c_wave);
        rocke_value_t* c_qb_round;
        rocke_value_t* wo_round;
        rocke_value_t* q_blk;
        rocke_value_t* ch_grp;
        wave_id = rocke_b_readfirstlane(b, wave_raw);
        lane = rocke_b_mod(b, tid, c_wave);
        c_qb_round = rocke_b_const_i32(b, QB - 1);
        wo_round = rocke_b_add(b, c_Wo, c_qb_round);
        n_q = rocke_b_div(b, wo_round, c->c_QB);
        bx = rocke_b_block_id_x(b);
        rb = rocke_b_block_id_y(b);
        n_i = rocke_b_block_id_z(b);
        q_blk = rocke_b_mod(b, bx, n_q);
        ch_grp = rocke_b_div(b, bx, n_q);
        c_base = rocke_b_mul(b, ch_grp, c_CB);
        q0 = rocke_b_mul(b, q_blk, c->c_QB);
    }
    c->a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    c->b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);
    {
        rocke_value_t* c_rb_round = rocke_b_const_i32(b, spec->row_blocks - 1);
        rocke_value_t* h_round = rocke_b_add(b, c->c_H, c_rb_round);
        rocke_value_t* c_rbs = rocke_b_const_i32(b, spec->row_blocks);
        rocke_value_t* rows_per_blk = rocke_b_div(b, h_round, c_rbs);
        rocke_value_t* h_next;
        rocke_value_t* h_next_lt;
        h_begin = rocke_b_mul(b, rb, rows_per_blk);
        h_next = rocke_b_add(b, h_begin, rows_per_blk);
        h_next_lt = rocke_b_cmp_lt(b, h_next, c->c_H);
        h_end = rocke_b_select(b, h_next_lt, h_next, c->c_H);
    }
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = CB * XCOLS;
        c->x_lds = rocke_b_smem_alloc(b, c->io_type, shape, 2, "hk_x");
        shape[1] = KH * CB * QB;
        c->dy_lds = rocke_b_smem_alloc(b, c->io_type, shape, 2, "hk_dy");
    }

    hk_items(b,
             tid,
             THREADS,
             c->X_ITEMS,
             X_ELEMS,
             c_CB,
             c_C,
             c_base,
             it_col,
             it_ch,
             it_cg,
             it_cok,
             it_eok);
    hk_items(b,
             tid,
             THREADS,
             c->DY_ITEMS,
             DY_ELEMS,
             c_CB,
             c_C,
             c_base,
             dt_col,
             dt_ch,
             dt_cg,
             dt_cok,
             dt_eok);
    {
        rocke_value_t* w_first = rocke_b_sub(b, q0, c->c_pad);
        for(k = 0; k < c->X_ITEMS; ++k)
        {
            rocke_value_t* wi = rocke_b_add(b, w_first, it_col[k]);
            rocke_value_t* ge = rocke_b_cmp_ge(b, wi, c->c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wi, c_W);
            rocke_value_t* w_ok = rocke_b_land(b, ge, lt);
            rocke_value_t* ok = rocke_b_land(b, w_ok, it_cok[k]);
            rocke_value_t* w_off = rocke_b_mul(b, wi, params.p_B_stride_wi);
            rocke_value_t* col_off = rocke_b_add(b, w_off, it_cg[k]);
            rocke_value_t* lds_ch = rocke_b_mul(b, it_ch[k], c->c_XCOLS);
            c->x_cols[k].ok = ok;
            c->x_cols[k].col_off = col_off;
            c->x_cols[k].idx = rocke_b_add(b, lds_ch, it_col[k]);
            c->x_cols[k].e_ok = it_eok[k];
        }
    }
    for(k = 0; k < c->DY_ITEMS; ++k)
    {
        rocke_value_t* wo = rocke_b_add(b, q0, dt_col[k]);
        rocke_value_t* lt = rocke_b_cmp_lt(b, wo, c_Wo);
        rocke_value_t* ok = rocke_b_land(b, lt, dt_cok[k]);
        rocke_value_t* wo_off = rocke_b_mul(b, wo, params.p_A_stride_wi);
        rocke_value_t* col_off = rocke_b_add(b, wo_off, dt_cg[k]);
        rocke_value_t* lds_ch = rocke_b_mul(b, dt_ch[k], c->c_QB);
        c->dy_cols[k].ok = ok;
        c->dy_cols[k].col_off = col_off;
        c->dy_cols[k].idx = rocke_b_add(b, lds_ch, dt_col[k]);
        c->dy_cols[k].e_ok = dt_eok[k];
    }
    c->x_n = rocke_b_mul(b, n_i, params.p_B_stride_n);
    c->dy_n = rocke_b_mul(b, n_i, params.p_A_stride_n);
    c->c_ring = rocke_b_const_i32(b, CB * QB);

    /* ---- lane roles ---- */
    {
        rocke_value_t* m = rocke_b_mod(b, lane, c16);
        rocke_value_t* c_nk;
        rocke_value_t* c_tpw;
        rocke_value_t* c_ch;
        rocke_value_t* w_tiles;
        khalf = rocke_b_div(b, lane, c16);
        c->a_ch = rocke_b_div(b, m, c_KW);
        c->a_s = rocke_b_mod(b, m, c_KW);
        c_mk = rocke_b_const_i32(b, CH * KW);
        c->a_live = rocke_b_cmp_lt(b, m, c_mk);
        c->b_ch = rocke_b_div(b, m, c->c_KH);
        c->b_r = rocke_b_mod(b, m, c->c_KH);
        c_nk = rocke_b_const_i32(b, CH * KH);
        c->b_live = rocke_b_cmp_lt(b, m, c_nk);
        c->k_off = rocke_b_mul(b, khalf, c16);
        c->c8 = rocke_b_const_i32(b, 8);
        c_tpw = rocke_b_const_i32(b, TPW);
        c_ch = rocke_b_const_i32(b, CH);
        w_tiles = rocke_b_mul(b, wave_id, c_tpw);
        for(t = 0; t < TPW; ++t)
        {
            rocke_value_t* c_t = rocke_b_const_i32(b, t);
            rocke_value_t* tile = rocke_b_add(b, w_tiles, c_t);
            tile_ch0[t] = rocke_b_mul(b, tile, c_ch);
        }
    }

    /* ---- ring prologue ---- */
    for(k = KH - 1; k > 0; --k)
    {
        rocke_value_t* c_k = rocke_b_const_i32(b, PAD - k);
        rocke_value_t* ho = rocke_b_add(b, h_begin, c_k);
        rocke_value_t* vals[ROCKE_DW_HK_MAX_ITEMS];
        int nv = 0;
        hk_stage(c, ho, false, vals, &nv);
        hk_commit_dy(c, vals, ho);
    }

    {
        rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 8);
        rocke_value_t* pend0[2 * ROCKE_DW_HK_MAX_ITEMS];
        rocke_iter_arg_t iargs[ROCKE_DW_HK_MAX_ITER];
        rocke_value_t* ho0;
        rocke_for_t loop;
        int np = 0, ni = 0;
        hk_stage(c, h_begin, true, pend0, &np);
        ho0 = rocke_b_add(b, h_begin, c->c_pad);
        hk_stage(c, ho0, false, pend0, &np);
        for(t = 0; t < TPW; ++t)
        {
            snprintf(names[ni], sizeof names[0], "hka%d", t);
            iargs[ni].name = names[ni];
            iargs[ni].init = zero_acc;
            ++ni;
        }
        for(i = 0; i < np; ++i)
        {
            snprintf(names[ni], sizeof names[0], "hkp%d", i);
            iargs[ni].name = names[ni];
            iargs[ni].init = pend0[i];
            ++ni;
        }
        loop = rocke_b_scf_for_iter(b, h_begin, h_end, c1, iargs, ni, "hk_hi", false, true);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* cur[ROCKE_DW_HK_MAX_TILES];
            rocke_value_t* pend[2 * ROCKE_DW_HK_MAX_ITEMS];
            rocke_value_t* nxt[2 * ROCKE_DW_HK_MAX_ITEMS];
            rocke_value_t* yields[ROCKE_DW_HK_MAX_ITER];
            rocke_value_t* ho_new;
            rocke_value_t* nhi;
            rocke_value_t* nho;
            int nn = 0, ny = 0;
            for(t = 0; t < TPW; ++t)
                cur[t] = loop.iter_vars[t];
            for(i = 0; i < np; ++i)
                pend[i] = loop.iter_vars[TPW + i];
            for(k = 0; k < c->X_ITEMS; ++k)
                hk_store(c, c->x_lds, c->x_cols[k].idx, pend[k], c->x_cols[k].e_ok);
            ho_new = rocke_b_add(b, loop.iv, c->c_pad);
            hk_commit_dy(c, pend + c->X_ITEMS, ho_new);
            nhi = rocke_b_add(b, loop.iv, c1);
            hk_stage(c, nhi, true, nxt, &nn);
            nho = rocke_b_add(b, nhi, c->c_pad);
            hk_stage(c, nho, false, nxt, &nn);
            rocke_b_sync_lds_only(b);
            for(t = 0; t < TPW; ++t)
            {
                rocke_value_t* af = hk_a_frag(c, tile_ch0[t]);
                rocke_value_t* bf = hk_b_frag(c, tile_ch0[t], loop.iv);
                cur[t] = rocke_b_mma(b, wmma_op_id, af, bf, cur[t], NULL, 0);
            }
            /* Drain this row's LDS reads before the next row's commits. */
            rocke_b_sync_lds_only(b);
            for(t = 0; t < TPW; ++t)
                yields[ny++] = cur[t];
            for(i = 0; i < nn; ++i)
                yields[ny++] = nxt[i];
            rocke_b_scf_yield(b, yields, ny);
        }
        rocke_b_region_leave(b);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(t = 0; t < TPW; ++t)
            accs[t] = loop.op->results[t];
    }

    /* ---- epilogue ---- */
    {
        rocke_value_t* c_rblocks = rocke_b_const_i32(b, spec->row_blocks);
        rocke_value_t* n_rb = rocke_b_mul(b, n_i, c_rblocks);
        rocke_value_t* c_cb_round;
        rocke_value_t* c_round;
        rocke_value_t* n_groups;
        rocke_value_t* grid_x;
        rocke_value_t* lin;
        rocke_value_t* c_r;
        rocke_value_t* rep;
        rocke_value_t* rep_c;
        rocke_value_t* c_kk;
        rocke_value_t* r_kw;
        rocke_value_t* k_row;
        rocke_value_t* c4;
        n_rb = rocke_b_add(b, n_rb, rb);
        c_cb_round = rocke_b_const_i32(b, CB - 1);
        c_round = rocke_b_add(b, c_C, c_cb_round);
        n_groups = rocke_b_div(b, c_round, c_CB);
        grid_x = rocke_b_mul(b, n_groups, n_q);
        lin = rocke_b_mul(b, n_rb, grid_x);
        lin = rocke_b_add(b, lin, bx);
        c_r = rocke_b_const_i32(b, spec->ws_replicas);
        rep = rocke_b_mod(b, lin, c_r);
        rep_c = rocke_b_mul(b, rep, c_C);
        c_kk = rocke_b_const_i32(b, KK);
        r_kw = rocke_b_mul(b, c->b_r, c_KW);
        k_row = rocke_b_mul(b, khalf, c->c8);
        c4 = rocke_b_const_i32(b, 4);
        for(t = 0; t < TPW; ++t)
        {
            rocke_value_t* tile_c = rocke_b_add(b, c_base, tile_ch0[t]);
            for(i = 0; i < 8; ++i)
            {
                rocke_value_t* c_i = rocke_b_const_i32(b, i);
                rocke_value_t* row = rocke_b_add(b, k_row, c_i);
                rocke_value_t* ch_a = rocke_b_div(b, row, c_KW);
                rocke_value_t* s = rocke_b_mod(b, row, c_KW);
                rocke_value_t* row_ok = rocke_b_cmp_lt(b, row, c_mk);
                rocke_value_t* ok = rocke_b_land(b, row_ok, c->b_live);
                rocke_value_t* same = rocke_b_cmp_eq(b, ch_a, c->b_ch);
                rocke_value_t* c_g;
                rocke_value_t* g_ok;
                rocke_value_t* slab;
                rocke_value_t* slab_off;
                rocke_value_t* tap;
                rocke_value_t* off;
                rocke_value_t* byte;
                rocke_value_t* val;
                ok = rocke_b_land(b, ok, same);
                c_g = rocke_b_add(b, tile_c, ch_a);
                g_ok = rocke_b_cmp_lt(b, c_g, c_C);
                ok = rocke_b_land(b, ok, g_ok);
                slab = rocke_b_add(b, rep_c, c_g);
                slab_off = rocke_b_mul(b, slab, c_kk);
                tap = rocke_b_add(b, r_kw, s);
                off = rocke_b_add(b, slab_off, tap);
                byte = rocke_b_mul(b, off, c4);
                val = rocke_b_vec_extract(b, accs[t], i);
                {
                    rocke_if_t gate = rocke_b_scf_if(b, ok);
                    rocke_b_region_enter(b, gate.then_region);
                    rocke_b_buffer_atomic_fadd_f32(
                        b, d_rsrc, byte, c->c0, val, ROCKE_DW_HK_CPOL_ATOMIC_CASCADE_DEV);
                    rocke_b_region_leave(b);
                }
            }
        }
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_direct_depthwise_wgrad_hankel_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_wgrad_hankel_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_direct_depthwise_wgrad_hankel_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_depthwise_wgrad_hankel(b, spec, arch);
    });
}
