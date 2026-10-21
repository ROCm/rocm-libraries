// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_wgrad_image.cpp -- C99 port of
 * _build_direct_conv_wgrad_image (library/kernels/common/conv_direct_grouped.py):
 * the image-per-block direct wgrad (wave32 WMMA 16x16x32, gfx1250).
 *
 * Same arithmetic as build_direct_conv_wgrad -- all KH*KW taps held as
 * accumulators, a KH-slot dY register ring, one X strip per input row read at
 * KW column shifts -- with three structural changes:
 *
 *   - Grid (groups * n_k_tiles * n_c_tiles, 1, N): one workgroup per image and
 *     k x c tile walks every wo tile (outer runtime loop) and every input row
 *     (inner runtime loop) itself, carrying the accumulators across both.
 *   - Register blocking: a wave owns wave_tiles_k x wave_tiles_c 16x16 tiles
 *     per tap.
 *   - The row loop is a runtime scf.for stepping KH rows with the body unrolled
 *     KH times, so the ring slots stay compile-time. rows_per_sync = KH stages
 *     all KH rows behind one barrier pair.
 *
 * Loads are issued one row ahead of the LDS commit that consumes them; the
 * in-flight fragments ride the row loop as iteration arguments.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_op_for_shape */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV = 4 | (2 << 3). */
#define ROCKE_WGRAD_IMG_CPOL_ATOMIC_CASCADE_DEV (4 | (2 << 3))

/* Fixed-capacity scratch. The image-reason VGPR budget (224) bounds
 * KH*KW*TK*TC, KH*TK and G*KW*TC by 28 each; the caps leave headroom and are
 * checked before any array is written. */
#define ROCKE_WGRAD_IMG_MAX_TILES 8
#define ROCKE_WGRAD_IMG_MAX_ACC 64
#define ROCKE_WGRAD_IMG_MAX_RING 64
#define ROCKE_WGRAD_IMG_MAX_ROW_LOADS 64
#define ROCKE_WGRAD_IMG_MAX_PENDING 512
#define ROCKE_WGRAD_IMG_MAX_XFRAG 64
#define ROCKE_WGRAD_IMG_MAX_ITER \
    (ROCKE_WGRAD_IMG_MAX_ACC + ROCKE_WGRAD_IMG_MAX_RING + ROCKE_WGRAD_IMG_MAX_PENDING)

typedef struct rocke_wgrad_img_ctx
{
    rocke_ir_builder_t* b;
    rocke_direct_conv_problem_t p;
    int is_bf16;
    const rocke_type_t* io_type;
    int KH, KW, TK, TC, G;
    int DY_PASSES;
    int ROWS_PER_PASS;
    int LD_VEC;
    int STRIP_PASSES_PER_WAVE;

    rocke_value_t* c0;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* c_H;
    rocke_value_t* c_Ho;
    rocke_value_t* c_Wo;
    rocke_value_t* c_TR_N;
    rocke_value_t* n_i;
    rocke_value_t* k_wave_in_group;
    rocke_value_t* c_wave_in_group;
    rocke_value_t* k_group_base;
    rocke_value_t* c_group_base;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    const rocke_tensor_descriptor_t* dy_desc;
    const rocke_tensor_descriptor_t* x_strip_desc;
    rocke_value_t* dy_lds;
    rocke_value_t* s_strip_lds;
    rocke_value_t* c_ld_sp;
    rocke_value_t* c_ld_ch;
    rocke_value_t* tr_flat;
    /* dy_parts_g[g][tk], strip_parts_g[g][tc] */
    rocke_value_t* dy_parts_g[ROCKE_DCONV_WGRAD_MAX_KH][ROCKE_WGRAD_IMG_MAX_TILES];
    rocke_value_t* strip_parts_g[ROCKE_DCONV_WGRAD_MAX_KH][ROCKE_WGRAD_IMG_MAX_TILES];
    rocke_value_t* strip_cols[ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES];
    rocke_value_t* strip_col_ok[ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES];

    /* TDM staging (tdm_stages >= 2). */
    rocke_value_t* tr_row;
    rocke_value_t* tr_col;
    rocke_value_t* wave_id;
    int n_waves;
    int WO_BLOCK, STRIP_COLS;
    int P_DY, P_X, DY_T, X_T;
    const rocke_dconv_params_t* params;
    rocke_value_t* dy_lds_addr;
    rocke_value_t* strip_lds_addr;
    rocke_value_t* k_tile_abs;
    rocke_value_t* c_tile_abs;
    rocke_value_t* tdm_k_ext;
    rocke_value_t* tdm_c_ext;
    rocke_value_t* c_DY_T;
    rocke_value_t* c_X_T;
} rocke_wgrad_img_ctx_t;

static void img_fail(rocke_ir_builder_t* b)
{
    if(b->status == ROCKE_OK)
    {
        b->status = ROCKE_ERR_VALUE;
    }
}

/* _tr_read(smem, part_off, row_shift): two ds_load_tr16_b128 8 rows apart,
 * concatenated into the 16-element WMMA operand. */
static rocke_value_t* img_tr_read(rocke_wgrad_img_ctx_t* c,
                                  rocke_value_t* smem,
                                  rocke_value_t* part_off,
                                  int row_shift)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* base;
    rocke_value_t* lo;
    rocke_value_t* hi;
    rocke_value_t* idx[2];
    {
        rocke_value_t* c_shift = rocke_b_const_i32(b, row_shift * 16);
        rocke_value_t* inner = rocke_b_add(b, c->tr_flat, c_shift);
        base = rocke_b_add(b, part_off, inner);
    }
    idx[0] = c->c0;
    idx[1] = base;
    lo = rocke_b_ds_read_tr16_b128(b, smem, idx, 2, c->io_type);
    {
        rocke_value_t* c_step = rocke_b_const_i32(b, 8 * 16);
        rocke_value_t* addr = rocke_b_add(b, base, c_step);
        rocke_value_t* idx2[2];
        idx2[0] = c->c0;
        idx2[1] = addr;
        hi = rocke_b_ds_read_tr16_b128(b, smem, idx2, 2, c->io_type);
    }
    return rocke_b_vec_concat(b, lo, hi);
}

/* _lds_run(part_off, row) = part_off + (row * TR_N + c_ld_ch). */
static rocke_value_t*
    img_lds_run(rocke_wgrad_img_ctx_t* c, rocke_value_t* part_off, rocke_value_t* row)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* mul_row = rocke_b_mul(b, row, c->c_TR_N);
    rocke_value_t* inner = rocke_b_add(b, mul_row, c->c_ld_ch);
    return rocke_b_add(b, part_off, inner);
}

/* _dy_sp(pass_idx) */
static rocke_value_t* img_dy_sp(rocke_wgrad_img_ctx_t* c, int pass_idx)
{
    rocke_value_t* c_off;
    if(pass_idx == 0)
    {
        return c->c_ld_sp;
    }
    c_off = rocke_b_const_i32(c->b, pass_idx * c->ROWS_PER_PASS);
    return rocke_b_add(c->b, c->c_ld_sp, c_off);
}

static rocke_value_t*
    img_buf_load(rocke_wgrad_img_ctx_t* c, rocke_value_t* rsrc, rocke_value_t* safe)
{
    return c->is_bf16 ? rocke_b_buffer_load_vN_bf16(c->b, rsrc, safe, c->c0, c->LD_VEC / 2)
                      : rocke_b_buffer_load_vN_f16(c->b, rsrc, safe, c->c0, c->LD_VEC / 2);
}

/* The dY half shared by _issue_row and _issue_dy_only: TK * DY_PASSES loads.
 * Appends to out[*n]. */
static bool img_issue_dy(rocke_wgrad_img_ctx_t* c,
                         rocke_value_t* ho_val,
                         rocke_value_t* ho_ok,
                         rocke_value_t* wo_start,
                         rocke_value_t** out,
                         int* n)
{
    rocke_ir_builder_t* b = c->b;
    int tk, pass_idx;
    for(tk = 0; tk < c->TK; ++tk)
    {
        rocke_value_t* k_ld;
        {
            rocke_value_t* lhs = rocke_b_add(b, c->k_group_base, c->k_wave_in_group);
            rocke_value_t* c_tk = rocke_b_const_i32(b, 16 * tk);
            rocke_value_t* rhs = rocke_b_add(b, c_tk, c->c_ld_ch);
            k_ld = rocke_b_add(b, lhs, rhs);
        }
        for(pass_idx = 0; pass_idx < c->DY_PASSES; ++pass_idx)
        {
            rocke_value_t* sp = img_dy_sp(c, pass_idx);
            rocke_value_t* wo_sp = rocke_b_add(b, wo_start, sp);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wo_sp, c->c_Wo);
            rocke_value_t* ok = rocke_b_land(b, ho_ok, lt);
            rocke_value_t* off = NULL;
            rocke_value_t* valid = NULL;
            rocke_value_t* scaled;
            rocke_value_t* safe;
            const char* names[4] = {"n", "h", "w", "k"};
            rocke_value_t* values[4];
            values[0] = c->n_i;
            values[1] = ho_val;
            values[2] = wo_sp;
            values[3] = k_ld;
            if(!rocke_transforms_descriptor_offset(b, c->dy_desc, names, values, 4, &off, &valid))
            {
                return false;
            }
            scaled = rocke_b_mul(b, off, c->c_half_bytes);
            safe = rocke_b_select(b, ok, scaled, c->oob_sentinel);
            out[(*n)++] = img_buf_load(c, c->a_rsrc, safe);
        }
    }
    return true;
}

/* _issue_row(hi_val, wo_start): DRAM reads of one input row, TK*DY_PASSES dY
 * then TC*STRIP_PASSES_PER_WAVE X. Appends to out[*n]. */
static bool img_issue_row(rocke_wgrad_img_ctx_t* c,
                          rocke_value_t* hi_val,
                          rocke_value_t* wo_start,
                          rocke_value_t** out,
                          int* n)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* hi_ok;
    rocke_value_t* ho_val;
    rocke_value_t* ho_ok;
    int tc, j;

    hi_ok = rocke_b_cmp_lt(b, hi_val, c->c_H);
    {
        rocke_value_t* c_pad = rocke_b_const_i32(b, c->p.PAD);
        ho_val = rocke_b_add(b, hi_val, c_pad);
    }
    {
        rocke_value_t* lt = rocke_b_cmp_lt(b, ho_val, c->c_Ho);
        ho_ok = rocke_b_land(b, hi_ok, lt);
    }
    if(!img_issue_dy(c, ho_val, ho_ok, wo_start, out, n))
    {
        return false;
    }
    for(tc = 0; tc < c->TC; ++tc)
    {
        rocke_value_t* c_ld;
        {
            rocke_value_t* lhs = rocke_b_add(b, c->c_group_base, c->c_wave_in_group);
            rocke_value_t* c_tc = rocke_b_const_i32(b, 16 * tc);
            rocke_value_t* rhs = rocke_b_add(b, c_tc, c->c_ld_ch);
            c_ld = rocke_b_add(b, lhs, rhs);
        }
        for(j = 0; j < c->STRIP_PASSES_PER_WAVE; ++j)
        {
            rocke_value_t* off = NULL;
            rocke_value_t* x_ok = NULL;
            rocke_value_t* ok;
            rocke_value_t* scaled;
            rocke_value_t* safe;
            const char* names[5] = {"n", "h", "wo", "s_off", "c"};
            rocke_value_t* values[5];
            values[0] = c->n_i;
            values[1] = hi_val;
            values[2] = wo_start;
            values[3] = c->strip_cols[j];
            values[4] = c_ld;
            if(!rocke_transforms_descriptor_offset(
                   b, c->x_strip_desc, names, values, 5, &off, &x_ok))
            {
                return false;
            }
            {
                rocke_value_t* inner = rocke_b_land(b, hi_ok, c->strip_col_ok[j]);
                ok = rocke_b_land(b, inner, x_ok);
            }
            scaled = rocke_b_mul(b, off, c->c_half_bytes);
            safe = rocke_b_select(b, ok, scaled, c->oob_sentinel);
            out[(*n)++] = img_buf_load(c, c->b_rsrc, safe);
        }
    }
    return true;
}

/* _commit_row(vecs, g): land one row's fragments in LDS slot g. */
static void img_commit_row(rocke_wgrad_img_ctx_t* c, rocke_value_t* const* vecs, int g)
{
    rocke_ir_builder_t* b = c->b;
    int i = 0;
    int tk, tc, pass_idx, j;
    for(tk = 0; tk < c->TK; ++tk)
    {
        for(pass_idx = 0; pass_idx < c->DY_PASSES; ++pass_idx)
        {
            rocke_value_t* idx[2];
            rocke_value_t* sp = img_dy_sp(c, pass_idx);
            idx[0] = c->c0;
            idx[1] = img_lds_run(c, c->dy_parts_g[g][tk], sp);
            rocke_b_smem_store_vN(b, c->dy_lds, idx, 2, vecs[i], c->LD_VEC);
            ++i;
        }
    }
    for(tc = 0; tc < c->TC; ++tc)
    {
        for(j = 0; j < c->STRIP_PASSES_PER_WAVE; ++j)
        {
            rocke_value_t* idx[2];
            idx[0] = c->c0;
            idx[1] = img_lds_run(c, c->strip_parts_g[g][tc], c->strip_cols[j]);
            rocke_b_smem_store_vN(b, c->s_strip_lds, idx, 2, vecs[i], c->LD_VEC);
            ++i;
        }
    }
}

/* _read_dy(g): TK dY fragments of LDS slot g. */
static void img_read_dy(rocke_wgrad_img_ctx_t* c, int g, rocke_value_t** out)
{
    int tk;
    for(tk = 0; tk < c->TK; ++tk)
    {
        out[tk] = img_tr_read(c, c->dy_lds, c->dy_parts_g[g][tk], 0);
    }
}

/* One rank-2 TDM transfer (tdm_descriptor_2d + tensor_load_to_lds, no LDS
 * padding, cachepolicy 0). The descriptor arguments are evaluated by the
 * caller in Python order. */
static void img_tdm_load(rocke_wgrad_img_ctx_t* c,
                         rocke_value_t* gptr,
                         rocke_value_t* lds_addr,
                         rocke_value_t* tensor_dim0,
                         rocke_value_t* tensor_dim1,
                         rocke_value_t* row_stride,
                         int tile_dim0,
                         int tile_dim1)
{
    rocke_value_t* d[5];
    if(!rocke_b_tdm_descriptor_2d(c->b,
                                  gptr,
                                  lds_addr,
                                  2,
                                  tensor_dim0,
                                  tensor_dim1,
                                  row_stride,
                                  tile_dim0,
                                  tile_dim1,
                                  0,
                                  -1,
                                  0,
                                  d))
    {
        img_fail(c->b);
        return;
    }
    rocke_b_tensor_load_to_lds(c->b, d[0], d[1], d[2], d[3], d[4], 0);
}

/* _tdm_row(hi_val, wo_start, g, stage, w_shift): TDM both operands of input
 * row hi_val into slot g of a stage. Wave 2g moves the dY tile and wave 2g+1
 * the X strip (mod the wave count). Rows past the image get a zero extent. */
static bool img_tdm_row(rocke_wgrad_img_ctx_t* c,
                        rocke_value_t* hi_val,
                        rocke_value_t* wo_start,
                        int g,
                        rocke_value_t* stage,
                        rocke_value_t* w_shift)
{
    rocke_ir_builder_t* b = c->b;
    const rocke_dconv_params_t* params = c->params;
    rocke_value_t* hi_ok;
    rocke_value_t* ho_val;
    rocke_value_t* ho_ok;

    hi_ok = rocke_b_cmp_lt(b, hi_val, c->c_H);
    {
        rocke_value_t* c_pad = rocke_b_const_i32(b, c->p.PAD);
        ho_val = rocke_b_add(b, hi_val, c_pad);
    }
    {
        rocke_value_t* lt = rocke_b_cmp_lt(b, ho_val, c->c_Ho);
        ho_ok = rocke_b_land(b, hi_ok, lt);
    }
    {
        rocke_value_t* c_w = rocke_b_const_i32(b, (2 * g) % c->n_waves);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c->wave_id, c_w);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* off = NULL;
            rocke_value_t* valid = NULL;
            rocke_value_t* slot;
            rocke_value_t* gptr;
            rocke_value_t* lds_addr;
            rocke_value_t* dim1;
            const char* names[4] = {"n", "h", "w", "k"};
            rocke_value_t* values[4];
            values[0] = c->n_i;
            values[1] = rocke_b_select(b, ho_ok, ho_val, c->c0);
            values[2] = wo_start;
            values[3] = c->k_tile_abs;
            if(!rocke_transforms_descriptor_offset(b, c->dy_desc, names, values, 4, &off, &valid))
                return false;
            {
                rocke_value_t* m = rocke_b_mul(b, stage, c->c_DY_T);
                rocke_value_t* c_g = rocke_b_const_i32(b, g * c->DY_T);
                slot = rocke_b_add(b, m, c_g);
            }
            {
                rocke_value_t* bytes = rocke_b_mul(b, off, c->c_half_bytes);
                gptr = rocke_b_global_ptr_add(b, params->A, bytes);
            }
            {
                rocke_value_t* bytes = rocke_b_mul(b, slot, c->c_half_bytes);
                lds_addr = rocke_b_add(b, c->dy_lds_addr, bytes);
            }
            {
                rocke_value_t* left = rocke_b_sub(b, c->c_Wo, wo_start);
                dim1 = rocke_b_select(b, ho_ok, left, c->c0);
            }
            img_tdm_load(
                c, gptr, lds_addr, c->tdm_k_ext, dim1, params->p_A_stride_wi, c->P_DY, c->WO_BLOCK);
        }
        rocke_b_region_leave(b);
    }
    {
        rocke_value_t* c_w = rocke_b_const_i32(b, (2 * g + 1) % c->n_waves);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c->wave_id, c_w);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* w0;
            rocke_value_t* off;
            rocke_value_t* slot;
            rocke_value_t* gptr;
            rocke_value_t* lds_addr;
            rocke_value_t* dim1;
            {
                rocke_value_t* c_pad = rocke_b_const_i32(b, c->p.PAD);
                rocke_value_t* d = rocke_b_sub(b, wo_start, c_pad);
                w0 = rocke_b_add(b, d, w_shift);
            }
            {
                rocke_value_t* m_n = rocke_b_mul(b, c->n_i, params->p_B_stride_n);
                rocke_value_t* h = rocke_b_select(b, hi_ok, hi_val, c->c0);
                rocke_value_t* m_h = rocke_b_mul(b, h, params->p_B_stride_hi);
                rocke_value_t* lhs = rocke_b_add(b, m_n, m_h);
                rocke_value_t* m_w = rocke_b_mul(b, w0, params->p_B_stride_wi);
                rocke_value_t* rhs = rocke_b_add(b, m_w, c->c_tile_abs);
                off = rocke_b_add(b, lhs, rhs);
            }
            {
                rocke_value_t* m = rocke_b_mul(b, stage, c->c_X_T);
                rocke_value_t* c_g = rocke_b_const_i32(b, g * c->X_T);
                rocke_value_t* lhs = rocke_b_add(b, m, c_g);
                rocke_value_t* c_px = rocke_b_const_i32(b, c->P_X);
                rocke_value_t* rhs = rocke_b_mul(b, w_shift, c_px);
                slot = rocke_b_add(b, lhs, rhs);
            }
            {
                /* A wo tile starting past the image (PAD > KW) must get a zero
                 * extent, not a negative one the TDM reads as huge unsigned. */
                rocke_value_t* left = rocke_b_sub(b, params->p_Wi, w0);
                rocke_value_t* pos = rocke_b_cmp_lt(b, c->c0, left);
                dim1 = rocke_b_select(b, rocke_b_land(b, hi_ok, pos), left, c->c0);
            }
            {
                rocke_value_t* bytes = rocke_b_mul(b, off, c->c_half_bytes);
                gptr = rocke_b_global_ptr_add(b, params->Bp, bytes);
            }
            {
                rocke_value_t* bytes = rocke_b_mul(b, slot, c->c_half_bytes);
                lds_addr = rocke_b_add(b, c->strip_lds_addr, bytes);
            }
            img_tdm_load(c,
                         gptr,
                         lds_addr,
                         c->tdm_c_ext,
                         dim1,
                         params->p_B_stride_wi,
                         c->P_X,
                         c->STRIP_COLS);
        }
        rocke_b_region_leave(b);
    }
    return true;
}

/* _tdm_tr_read(smem, base, pitch, row_shift): the transpose read of a TDM
 * slot whose rows are `pitch` elements apart. */
static rocke_value_t* img_tdm_tr_read(
    rocke_wgrad_img_ctx_t* c, rocke_value_t* smem, rocke_value_t* base, int pitch, int row_shift)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* addr;
    rocke_value_t* lo;
    rocke_value_t* hi;
    rocke_value_t* idx[2];
    {
        rocke_value_t* c_rs = rocke_b_const_i32(b, row_shift);
        rocke_value_t* row = rocke_b_add(b, c->tr_row, c_rs);
        rocke_value_t* c_p = rocke_b_const_i32(b, pitch);
        rocke_value_t* m = rocke_b_mul(b, row, c_p);
        rocke_value_t* inner = rocke_b_add(b, m, c->tr_col);
        addr = rocke_b_add(b, base, inner);
    }
    idx[0] = c->c0;
    idx[1] = addr;
    lo = rocke_b_ds_read_tr16_b128(b, smem, idx, 2, c->io_type);
    {
        rocke_value_t* c_step = rocke_b_const_i32(b, 8 * pitch);
        rocke_value_t* a2 = rocke_b_add(b, addr, c_step);
        rocke_value_t* idx2[2];
        idx2[0] = c->c0;
        idx2[1] = a2;
        hi = rocke_b_ds_read_tr16_b128(b, smem, idx2, 2, c->io_type);
    }
    return rocke_b_vec_concat(b, lo, hi);
}

/* Coalesced epilogue of the image-per-block wgrad (lds_epilogue). Python:
 * _wgrad_image_lds_epilogue. Every wave writes its accumulators into the
 * block's [block_k][KH*KW][block_c + pad] fp32 tile in LDS; after one barrier
 * the block walks the tile channel-fastest, so each atomic instruction's lanes
 * hit consecutive dW channels of one (k, r, s) row. Nested Python expressions
 * are unrolled into locals in Python's left-to-right evaluation order. */
static void img_lds_epilogue(rocke_ir_builder_t* b,
                             const rocke_direct_conv_wgrad_spec_t* spec,
                             rocke_value_t* const* final_accs,
                             const int* acc_r,
                             const int* acc_s,
                             const int* acc_tk,
                             const int* acc_tc,
                             int n_acc,
                             rocke_value_t* wave_k_id,
                             rocke_value_t* wave_c_id,
                             rocke_value_t* c4,
                             rocke_value_t* q_in_lane,
                             rocke_value_t* tid,
                             rocke_value_t* k_tile_origin,
                             rocke_value_t* c_tile_origin,
                             rocke_value_t* k_group_base,
                             rocke_value_t* c_kpg,
                             rocke_value_t* c_cpg,
                             rocke_value_t* ws_shift,
                             rocke_value_t* d_rsrc)
{
    const int KH = spec->problem.KH;
    const int KW = spec->problem.KW;
    const int TK = spec->wave_tiles_k;
    const int TC = spec->wave_tiles_c;
    const int C_FRAG = 8;
    const int BK = rocke_direct_conv_wgrad_block_k(spec);
    const int BC = rocke_direct_conv_wgrad_block_c(spec);
    const int RS = KH * KW;
    const int BCP = BC + ROCKE_WGRAD_LDS_EPI_PAD;
    const int THREADS = rocke_direct_conv_wgrad_threads_per_block(spec);
    const int n_items = BK * RS * BC;
    rocke_value_t* c0;
    rocke_value_t* tile;
    rocke_value_t* k_wave_local;
    rocke_value_t* c_wave_local;
    rocke_value_t* c_rs_cpg;
    int i, slot, j0;

    c0 = rocke_b_const_i32(b, 0);
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = BK * RS * BCP;
        tile = rocke_b_smem_alloc(b, rocke_f32(), shape, 2, "dw_tile");
    }
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, 16 * TK);
        rocke_value_t* m0 = rocke_b_mul(b, wave_k_id, c_t);
        rocke_value_t* c_cf = rocke_b_const_i32(b, C_FRAG);
        rocke_value_t* m1 = rocke_b_mul(b, c4, c_cf);
        k_wave_local = rocke_b_add(b, m0, m1);
    }
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, 16 * TC);
        rocke_value_t* m = rocke_b_mul(b, wave_c_id, c_t);
        c_wave_local = rocke_b_add(b, m, q_in_lane);
    }
    for(i = 0; i < n_acc; ++i)
    {
        rocke_value_t* c_l;
        {
            rocke_value_t* c_t = rocke_b_const_i32(b, 16 * acc_tc[i]);
            c_l = rocke_b_add(b, c_wave_local, c_t);
        }
        for(slot = 0; slot < C_FRAG; ++slot)
        {
            rocke_value_t* k_l;
            rocke_value_t* idx;
            rocke_value_t* val;
            rocke_value_t* ind[2];
            {
                rocke_value_t* c_sl = rocke_b_const_i32(b, 16 * acc_tk[i] + slot);
                k_l = rocke_b_add(b, k_wave_local, c_sl);
            }
            {
                rocke_value_t* c_row = rocke_b_const_i32(b, RS * BCP);
                rocke_value_t* m = rocke_b_mul(b, k_l, c_row);
                rocke_value_t* c_tap = rocke_b_const_i32(b, (acc_r[i] * KW + acc_s[i]) * BCP);
                rocke_value_t* sum = rocke_b_add(b, m, c_tap);
                idx = rocke_b_add(b, sum, c_l);
            }
            val = rocke_b_vec_extract(b, final_accs[i], slot);
            ind[0] = c0;
            ind[1] = idx;
            rocke_b_smem_store_vN(b, tile, ind, 2, val, 1);
        }
    }
    rocke_b_sync_lds_only(b);
    {
        rocke_value_t* c_rs = rocke_b_const_i32(b, RS);
        c_rs_cpg = rocke_b_mul(b, c_rs, c_cpg);
    }
    for(j0 = 0; j0 < n_items; j0 += THREADS)
    {
        rocke_value_t* j;
        rocke_value_t* c_l;
        rocke_value_t* row;
        rocke_value_t* k_l;
        rocke_value_t* rs;
        rocke_value_t* k_g;
        rocke_value_t* c_g;
        rocke_value_t* ok;
        rocke_value_t* lds_idx;
        rocke_value_t* val;
        rocke_value_t* dw_off;
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j0);
            j = rocke_b_add(b, tid, c_j);
        }
        {
            rocke_value_t* c_bc = rocke_b_const_i32(b, BC);
            c_l = rocke_b_mod(b, j, c_bc);
        }
        {
            rocke_value_t* c_bc = rocke_b_const_i32(b, BC);
            row = rocke_b_div(b, j, c_bc);
        }
        {
            rocke_value_t* c_rs = rocke_b_const_i32(b, RS);
            k_l = rocke_b_div(b, row, c_rs);
        }
        {
            rocke_value_t* c_rs = rocke_b_const_i32(b, RS);
            rs = rocke_b_mod(b, row, c_rs);
        }
        k_g = rocke_b_add(b, k_tile_origin, k_l);
        c_g = rocke_b_add(b, c_tile_origin, c_l);
        {
            rocke_value_t* lt_k = rocke_b_cmp_lt(b, k_g, c_kpg);
            rocke_value_t* lt_c = rocke_b_cmp_lt(b, c_g, c_cpg);
            ok = rocke_b_land(b, lt_k, lt_c);
        }
        if(n_items % THREADS)
        {
            rocke_value_t* c_n = rocke_b_const_i32(b, n_items);
            rocke_value_t* lt = rocke_b_cmp_lt(b, j, c_n);
            ok = rocke_b_land(b, ok, lt);
        }
        {
            rocke_value_t* c_row = rocke_b_const_i32(b, RS * BCP);
            rocke_value_t* m0 = rocke_b_mul(b, k_l, c_row);
            rocke_value_t* c_bcp = rocke_b_const_i32(b, BCP);
            rocke_value_t* m1 = rocke_b_mul(b, rs, c_bcp);
            rocke_value_t* sum = rocke_b_add(b, m0, m1);
            lds_idx = rocke_b_add(b, sum, c_l);
        }
        {
            rocke_value_t* ind[2];
            rocke_value_t* v;
            ind[0] = c0;
            ind[1] = lds_idx;
            v = rocke_b_smem_load_vN(b, tile, ind, 2, rocke_f32(), 1);
            val = rocke_b_vec_extract(b, v, 0);
        }
        {
            rocke_value_t* k_abs = rocke_b_add(b, k_group_base, k_g);
            rocke_value_t* m0 = rocke_b_mul(b, k_abs, c_rs_cpg);
            rocke_value_t* m1 = rocke_b_mul(b, rs, c_cpg);
            rocke_value_t* sum = rocke_b_add(b, m0, m1);
            dw_off = rocke_b_add(b, sum, c_g);
        }
        if(ws_shift != NULL)
            dw_off = rocke_b_add(b, dw_off, ws_shift);
        {
            rocke_if_t gate = rocke_b_scf_if(b, ok);
            rocke_b_region_enter(b, gate.then_region);
            {
                rocke_value_t* c_4 = rocke_b_const_i32(b, 4);
                rocke_value_t* voff = rocke_b_mul(b, dw_off, c_4);
                rocke_b_buffer_atomic_fadd_f32(
                    b, d_rsrc, voff, c0, val, ROCKE_WGRAD_IMG_CPOL_ATOMIC_CASCADE_DEV);
            }
            rocke_b_region_leave(b);
        }
    }
}

rocke_kernel_def_t* rocke_dconv_wgrad_build_image(rocke_ir_builder_t* b,
                                                  const rocke_direct_conv_wgrad_spec_t* spec,
                                                  const char* arch)
{
    rocke_wgrad_img_ctx_t cx;
    rocke_wgrad_img_ctx_t* c = &cx;
    rocke_dconv_params_t params;
    const char* wmma_op_id;
    int KH, KW, TK, TC, G, WK, WC, THREADS;
    int TR_N, WO_BLOCK, FRAG, C_FRAG;
    int STRIP_COLS, STRIP_PASSES, STRIP_GROUPS, STRIP_COLS_PAD;
    int LDS_SIZE_DY, STRIP_PER_PART, DY_SLOT, STRIP_SLOT;
    int RTC;
    int TDM, dy_elems, strip_elems;
    int block_k, block_c;
    int n_acc, n_ring, n_row_loads, n_pend;
    int g, tk, tc, j, i, r, s, k, sl;
    rocke_value_t* c_cpg;
    rocke_value_t* c_kpg;
    rocke_value_t* c_wave;
    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;
    rocke_value_t* c4;
    rocke_value_t* c16;
    rocke_value_t* q_in_lane;
    rocke_value_t* c_n_k_tiles;
    rocke_value_t* c_n_c_tiles;
    rocke_value_t* bx;
    rocke_value_t* c_tile_idx;
    rocke_value_t* gk_flat;
    rocke_value_t* k_tile_in_group;
    rocke_value_t* group;
    rocke_value_t* wave_k_id;
    rocke_value_t* wave_c_id;
    rocke_value_t* k_tile_origin;
    rocke_value_t* c_tile_origin;
    rocke_value_t* d_rsrc;
    const rocke_tensor_descriptor_t* dw_desc;
    rocke_value_t* c_rows_per_pass;
    rocke_value_t* c_STRIP_COLS;
    rocke_value_t* strip_pass_base;
    rocke_value_t* zero_acc;
    rocke_value_t* zero_frag;
    rocke_value_t* c_KH;
    rocke_value_t* c_WO_BLOCK;
    rocke_value_t* c_H_round;
    rocke_value_t* c_n_wo_tiles;
    rocke_value_t* h_start;
    rocke_value_t* h_end;
    rocke_value_t* c_TDM = NULL;
    rocke_value_t* dy_ch[ROCKE_WGRAD_IMG_MAX_TILES];
    rocke_value_t* x_ch[ROCKE_WGRAD_IMG_MAX_TILES];
    rocke_value_t* final_accs[ROCKE_WGRAD_IMG_MAX_ACC];
    int acc_r[ROCKE_WGRAD_IMG_MAX_ACC];
    int acc_s[ROCKE_WGRAD_IMG_MAX_ACC];
    int acc_tk[ROCKE_WGRAD_IMG_MAX_ACC];
    int acc_tc[ROCKE_WGRAD_IMG_MAX_ACC];
    char iter_names[ROCKE_WGRAD_IMG_MAX_ITER][24];
    rocke_iter_arg_t wo_iargs[ROCKE_WGRAD_IMG_MAX_ACC];
    rocke_for_t wo_loop;

    memset(&cx, 0, sizeof cx);
    c->b = b;
    c->p = spec->problem;
    KH = c->p.KH;
    KW = c->p.KW;
    c->KH = KH;
    c->KW = KW;
    c->is_bf16 = c->p.dtype && strcmp(c->p.dtype, "bf16") == 0;

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
            img_fail(b);
            return NULL;
        }
        wmma_op_id = op->op_id;
    }

    WK = spec->waves_k;
    WC = spec->waves_c;
    TK = spec->wave_tiles_k;
    TC = spec->wave_tiles_c;
    c->TK = TK;
    c->TC = TC;
    THREADS = rocke_direct_conv_wgrad_threads_per_block(spec);
    TR_N = 16;
    WO_BLOCK = spec->mfma_k;
    c->LD_VEC = 8;
    FRAG = 16;
    C_FRAG = 8;
    c->ROWS_PER_PASS = spec->wave_size * c->LD_VEC / TR_N;
    c->DY_PASSES = WO_BLOCK / c->ROWS_PER_PASS;
    STRIP_COLS = WO_BLOCK + KW - 1;
    STRIP_PASSES = (STRIP_COLS + c->ROWS_PER_PASS - 1) / c->ROWS_PER_PASS;
    STRIP_GROUPS = (WK < STRIP_PASSES) ? WK : STRIP_PASSES;
    c->STRIP_PASSES_PER_WAVE = (STRIP_PASSES + STRIP_GROUPS - 1) / STRIP_GROUPS;
    STRIP_COLS_PAD = c->STRIP_PASSES_PER_WAVE * STRIP_GROUPS * c->ROWS_PER_PASS;
    LDS_SIZE_DY = WO_BLOCK * TR_N;
    STRIP_PER_PART = STRIP_COLS_PAD * TR_N;
    G = spec->rows_per_sync;
    c->G = G;
    DY_SLOT = WK * TK * LDS_SIZE_DY;
    STRIP_SLOT = WC * TC * STRIP_PER_PART;
    RTC = spec->runtime_channels ? 1 : 0;
    block_k = rocke_direct_conv_wgrad_block_k(spec);
    block_c = rocke_direct_conv_wgrad_block_c(spec);
    c->WO_BLOCK = WO_BLOCK;
    c->STRIP_COLS = STRIP_COLS;
    c->n_waves = THREADS / spec->wave_size;

    /* LDS sizing. A TDM row lands as [row][block + TDM_PAD]: the TDM_PAD
     * channels past the tile edge lie outside the clipped tensor extent, so the
     * transfer zero-fills them. wo tile 0 lands PAD rows down a strip slot. The
     * dY ring prologue still stages through registers into slot 0. */
    TDM = spec->tdm_stages;
    dy_elems = G * DY_SLOT;
    strip_elems = G * STRIP_SLOT;
    if(TDM)
    {
        const int TDM_PAD = 16;
        c->P_DY = block_k + TDM_PAD;
        c->P_X = block_c + TDM_PAD;
        c->DY_T = WO_BLOCK * c->P_DY;
        c->X_T = (STRIP_COLS + c->p.PAD) * c->P_X;
        dy_elems = (DY_SLOT > TDM * G * c->DY_T) ? DY_SLOT : TDM * G * c->DY_T;
        strip_elems = TDM * G * c->X_T;
    }

    n_acc = KH * KW * TK * TC;
    n_ring = KH * TK;
    n_row_loads = TK * c->DY_PASSES + TC * c->STRIP_PASSES_PER_WAVE;
    n_pend = G * n_row_loads;
    if(KH > ROCKE_DCONV_WGRAD_MAX_KH || G < 1 || G > ROCKE_DCONV_WGRAD_MAX_KH
       || TK > ROCKE_WGRAD_IMG_MAX_TILES || TC > ROCKE_WGRAD_IMG_MAX_TILES || c->DY_PASSES < 1
       || c->STRIP_PASSES_PER_WAVE > ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES
       || n_acc > ROCKE_WGRAD_IMG_MAX_ACC || n_ring > ROCKE_WGRAD_IMG_MAX_RING
       || n_row_loads > ROCKE_WGRAD_IMG_MAX_ROW_LOADS || n_pend > ROCKE_WGRAD_IMG_MAX_PENDING
       || G * KW * TC > ROCKE_WGRAD_IMG_MAX_XFRAG || G * TK > ROCKE_WGRAD_IMG_MAX_RING)
    {
        img_fail(b);
        return NULL;
    }

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    c->io_type = rocke_b_io_ir_type(b, c->p.dtype ? c->p.dtype : "fp16");
    if(c->io_type == NULL)
    {
        return NULL;
    }
    rocke_dconv_emit_params(b, &params, "wgrad", c->io_type);
    c->params = &params;

    c->c0 = rocke_b_const_i32(b, 0);
    if(RTC)
    {
        c_cpg = rocke_b_div(b, params.p_total_c, params.p_groups);
        c_kpg = rocke_b_div(b, params.p_total_k, params.p_groups);
    }
    else
    {
        c_cpg = rocke_b_const_i32(b, c->p.cpg);
        c_kpg = rocke_b_const_i32(b, c->p.kpg);
    }
    c->c_half_bytes = rocke_b_const_i32(b, 2);
    c->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    c->c_H = params.p_Hi;
    c->c_Ho = params.p_Ho;
    c->c_Wo = params.p_Wo;
    c->c_TR_N = rocke_b_const_i32(b, TR_N);

    tid = rocke_b_thread_id_x(b);
    c_wave = rocke_b_const_i32(b, spec->wave_size);
    wave_id = rocke_b_div(b, tid, c_wave);
    c->wave_id = wave_id;
    lane = rocke_b_mod(b, tid, c_wave);
    c16 = rocke_b_const_i32(b, 16);
    c4 = rocke_b_div(b, lane, c16);
    q_in_lane = rocke_b_mod(b, lane, c16);

    /* ---- grid decode: bx = (group * n_k_tiles + k_tile) * n_c_tiles + c_tile,
     * bz = n. */
    if(RTC)
    {
        {
            rocke_value_t* c_round = rocke_b_const_i32(b, block_k - 1);
            rocke_value_t* sum = rocke_b_add(b, c_kpg, c_round);
            rocke_value_t* c_blk = rocke_b_const_i32(b, block_k);
            c_n_k_tiles = rocke_b_div(b, sum, c_blk);
        }
        {
            rocke_value_t* c_round = rocke_b_const_i32(b, block_c - 1);
            rocke_value_t* sum = rocke_b_add(b, c_cpg, c_round);
            rocke_value_t* c_blk = rocke_b_const_i32(b, block_c);
            c_n_c_tiles = rocke_b_div(b, sum, c_blk);
        }
    }
    else
    {
        c_n_k_tiles = rocke_b_const_i32(b, (c->p.kpg + block_k - 1) / block_k);
        c_n_c_tiles = rocke_b_const_i32(b, (c->p.cpg + block_c - 1) / block_c);
    }
    bx = rocke_b_block_id_x(b);
    c->n_i = rocke_b_block_id_z(b);
    c_tile_idx = rocke_b_mod(b, bx, c_n_c_tiles);
    gk_flat = rocke_b_div(b, bx, c_n_c_tiles);
    k_tile_in_group = rocke_b_mod(b, gk_flat, c_n_k_tiles);
    group = rocke_b_div(b, gk_flat, c_n_k_tiles);

    /* waves: k fastest */
    {
        rocke_value_t* c_wk = rocke_b_const_i32(b, WK);
        wave_k_id = rocke_b_mod(b, wave_id, c_wk);
    }
    {
        rocke_value_t* c_wk = rocke_b_const_i32(b, WK);
        wave_c_id = rocke_b_div(b, wave_id, c_wk);
    }
    {
        rocke_value_t* c_bk = rocke_b_const_i32(b, block_k);
        k_tile_origin = rocke_b_mul(b, k_tile_in_group, c_bk);
    }
    {
        rocke_value_t* c_bc = rocke_b_const_i32(b, block_c);
        c_tile_origin = rocke_b_mul(b, c_tile_idx, c_bc);
    }
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, 16 * TK);
        rocke_value_t* m = rocke_b_mul(b, wave_k_id, c_t);
        c->k_wave_in_group = rocke_b_add(b, k_tile_origin, m);
    }
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, 16 * TC);
        rocke_value_t* m = rocke_b_mul(b, wave_c_id, c_t);
        c->c_wave_in_group = rocke_b_add(b, c_tile_origin, m);
    }
    c->k_group_base = rocke_b_mul(b, group, c_kpg);
    c->c_group_base = rocke_b_mul(b, group, c_cpg);

    c->a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    c->b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);

    /* ---- descriptors ---- */
    {
        static const char* const dy_coords[4] = {"n", "h", "w", "k"};
        rocke_value_t* strides[4];
        rocke_dynamic_tensor_descriptor_t* dy_dyn;
        strides[0] = params.p_A_stride_n;
        strides[1] = params.p_A_stride_hi;
        strides[2] = params.p_A_stride_wi;
        strides[3] = rocke_b_const_i32(b, 1);
        dy_dyn = rocke_tensor_descriptor_naive_dynamic(b, "A", dy_coords, 4, strides);
        if(!dy_dyn)
            return NULL;
        c->dy_desc = &dy_dyn->base;
    }
    {
        static const char* const x_coords[4] = {"n", "h", "w", "c"};
        static const char* const w_upper[2] = {"wo", "s_off"};
        rocke_value_t* strides[4];
        rocke_dynamic_tensor_descriptor_t* x_dyn;
        rocke_value_t* w_strides[2] = {NULL, NULL};
        int w_strides_c[2];
        const rocke_transform_t* xforms[1];
        rocke_tensor_descriptor_t* chained;

        strides[0] = params.p_B_stride_n;
        strides[1] = params.p_B_stride_hi;
        strides[2] = params.p_B_stride_wi;
        strides[3] = rocke_b_const_i32(b, 1);
        x_dyn = rocke_tensor_descriptor_naive_dynamic(b, "B", x_coords, 4, strides);
        if(!x_dyn)
            return NULL;
        w_strides_c[0] = c->p.stride;
        w_strides_c[1] = 1;
        xforms[0] = rocke_embed_dynamic_mixed(
            b, w_upper, 2, "w", w_strides, w_strides_c, NULL, -c->p.PAD, 0, params.p_Wi);
        if(!xforms[0])
            return NULL;
        chained = rocke_tensor_descriptor_transform(b, &x_dyn->base, xforms, 1);
        if(!chained)
            return NULL;
        x_dyn->base = *chained;
        c->x_strip_desc = &x_dyn->base;
    }
    if(RTC)
    {
        static const char* const dw_coords[4] = {"k", "r", "s", "c"};
        rocke_value_t* strides[4];
        rocke_dynamic_tensor_descriptor_t* dw_dyn;
        {
            rocke_value_t* c_khkw = rocke_b_const_i32(b, KH * KW);
            strides[0] = rocke_b_mul(b, c_cpg, c_khkw);
        }
        {
            rocke_value_t* c_kw = rocke_b_const_i32(b, KW);
            strides[1] = rocke_b_mul(b, c_cpg, c_kw);
        }
        strides[2] = c_cpg;
        strides[3] = rocke_b_const_i32(b, 1);
        dw_dyn = rocke_tensor_descriptor_naive_dynamic(b, "D", dw_coords, 4, strides);
        if(!dw_dyn)
            return NULL;
        dw_desc = &dw_dyn->base;
    }
    else
    {
        static const char* const dw_coords[4] = {"k", "r", "s", "c"};
        int lengths[4];
        lengths[0] = rocke_direct_conv_problem_total_k(&c->p);
        lengths[1] = KH;
        lengths[2] = KW;
        lengths[3] = c->p.cpg;
        dw_desc = rocke_tensor_descriptor_naive(b, "D", lengths, 4, NULL, dw_coords, 4);
        if(!dw_desc)
            return NULL;
    }

    /* ---- LDS: dY tiles keyed on (wave_k, tk), X strips on (wave_c, tc). */
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = dy_elems;
        c->dy_lds = rocke_b_smem_alloc(b, c->io_type, shape, 2, "dy_lds");
    }
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = strip_elems;
        c->s_strip_lds = rocke_b_smem_alloc(b, c->io_type, shape, 2, "s_strip");
    }

    {
        rocke_value_t* c_lanes_per_sp = rocke_b_const_i32(b, TR_N / c->LD_VEC);
        c->c_ld_sp = rocke_b_div(b, lane, c_lanes_per_sp);
        {
            rocke_value_t* lane_mod = rocke_b_mod(b, lane, c_lanes_per_sp);
            rocke_value_t* c_vec = rocke_b_const_i32(b, c->LD_VEC);
            c->c_ld_ch = rocke_b_mul(b, lane_mod, c_vec);
        }
    }
    c_rows_per_pass = rocke_b_const_i32(b, c->ROWS_PER_PASS);
    c_STRIP_COLS = rocke_b_const_i32(b, STRIP_COLS);

    /* wave32 transpose-read address (see build_direct_conv_wgrad). */
    {
        rocke_value_t* c8 = rocke_b_const_i32(b, 8);
        rocke_value_t* tr_col;
        rocke_value_t* tr_row;
        {
            rocke_value_t* blk = rocke_b_div(b, q_in_lane, c8);
            tr_col = rocke_b_mul(b, blk, c8);
        }
        {
            rocke_value_t* c_frag = rocke_b_const_i32(b, FRAG);
            rocke_value_t* lhs = rocke_b_mul(b, c4, c_frag);
            rocke_value_t* rhs = rocke_b_mod(b, lane, c8);
            tr_row = rocke_b_add(b, lhs, rhs);
        }
        {
            rocke_value_t* lhs = rocke_b_mul(b, tr_row, c->c_TR_N);
            c->tr_flat = rocke_b_add(b, lhs, tr_col);
        }
        c->tr_row = tr_row;
        c->tr_col = tr_col;
    }

    /* dy_parts0 = [_dy_part(t) for t in range(TK)], then strip_parts0. */
    for(tk = 0; tk < TK; ++tk)
    {
        rocke_value_t* c_tk_n = rocke_b_const_i32(b, TK);
        rocke_value_t* m = rocke_b_mul(b, wave_k_id, c_tk_n);
        rocke_value_t* c_t = rocke_b_const_i32(b, tk);
        rocke_value_t* idx = rocke_b_add(b, m, c_t);
        rocke_value_t* c_sz = rocke_b_const_i32(b, LDS_SIZE_DY);
        c->dy_parts_g[0][tk] = rocke_b_mul(b, idx, c_sz);
    }
    for(tc = 0; tc < TC; ++tc)
    {
        rocke_value_t* c_tc_n = rocke_b_const_i32(b, TC);
        rocke_value_t* m = rocke_b_mul(b, wave_c_id, c_tc_n);
        rocke_value_t* c_t = rocke_b_const_i32(b, tc);
        rocke_value_t* idx = rocke_b_add(b, m, c_t);
        rocke_value_t* c_sz = rocke_b_const_i32(b, STRIP_PER_PART);
        c->strip_parts_g[0][tc] = rocke_b_mul(b, idx, c_sz);
    }
    /* _slot_parts(parts0, slot_elems, g): dY slots first, then strip slots. */
    for(g = 1; g < G; ++g)
    {
        rocke_value_t* c_off = rocke_b_const_i32(b, g * DY_SLOT);
        for(tk = 0; tk < TK; ++tk)
            c->dy_parts_g[g][tk] = rocke_b_add(b, c->dy_parts_g[0][tk], c_off);
    }
    for(g = 1; g < G; ++g)
    {
        rocke_value_t* c_off = rocke_b_const_i32(b, g * STRIP_SLOT);
        for(tc = 0; tc < TC; ++tc)
            c->strip_parts_g[g][tc] = rocke_b_add(b, c->strip_parts_g[0][tc], c_off);
    }

    {
        rocke_value_t* c_sg = rocke_b_const_i32(b, STRIP_GROUPS);
        strip_pass_base = rocke_b_mod(b, wave_k_id, c_sg);
    }
    for(j = 0; j < c->STRIP_PASSES_PER_WAVE; ++j)
    {
        rocke_value_t* c_j = rocke_b_const_i32(b, j * STRIP_GROUPS);
        rocke_value_t* pidx = rocke_b_add(b, strip_pass_base, c_j);
        rocke_value_t* scaled = rocke_b_mul(b, pidx, c_rows_per_pass);
        c->strip_cols[j] = rocke_b_add(b, c->c_ld_sp, scaled);
    }
    for(j = 0; j < c->STRIP_PASSES_PER_WAVE; ++j)
    {
        c->strip_col_ok[j] = rocke_b_cmp_lt(b, c->strip_cols[j], c_STRIP_COLS);
    }

    /* ---- accumulator bookkeeping: acc[(r, s, tk, tc)] flattened ---- */
    i = 0;
    for(r = 0; r < KH; ++r)
        for(s = 0; s < KW; ++s)
            for(tk = 0; tk < TK; ++tk)
                for(tc = 0; tc < TC; ++tc)
                {
                    acc_r[i] = r;
                    acc_s[i] = s;
                    acc_tk[i] = tk;
                    acc_tc[i] = tc;
                    ++i;
                }
    zero_acc = rocke_b_zero_vec_f32(b, C_FRAG);
    zero_frag = rocke_b_zero_vec(b, c->io_type, FRAG);

    c_KH = rocke_b_const_i32(b, KH);
    c_WO_BLOCK = rocke_b_const_i32(b, WO_BLOCK);
    {
        rocke_value_t* c_r = rocke_b_const_i32(b, KH - 1);
        rocke_value_t* sum = rocke_b_add(b, c->c_H, c_r);
        rocke_value_t* q = rocke_b_div(b, sum, c_KH);
        c_H_round = rocke_b_mul(b, q, c_KH);
    }
    {
        rocke_value_t* c_r = rocke_b_const_i32(b, WO_BLOCK - 1);
        rocke_value_t* sum = rocke_b_add(b, c->c_Wo, c_r);
        c_n_wo_tiles = rocke_b_div(b, sum, c_WO_BLOCK);
    }
    /* Row split: block y walks input rows [h_start, h_end) in whole KH steps. */
    if(spec->row_blocks > 1)
    {
        rocke_value_t* c_steps = rocke_b_div(b, c_H_round, c_KH);
        rocke_value_t* c_rows_blk;
        rocke_value_t* h_next;
        {
            rocke_value_t* c_rm1 = rocke_b_const_i32(b, spec->row_blocks - 1);
            rocke_value_t* sum = rocke_b_add(b, c_steps, c_rm1);
            rocke_value_t* c_rb = rocke_b_const_i32(b, spec->row_blocks);
            rocke_value_t* q = rocke_b_div(b, sum, c_rb);
            c_rows_blk = rocke_b_mul(b, q, c_KH);
        }
        {
            rocke_value_t* by = rocke_b_block_id_y(b);
            h_start = rocke_b_mul(b, by, c_rows_blk);
        }
        h_next = rocke_b_add(b, h_start, c_rows_blk);
        {
            rocke_value_t* lt = rocke_b_cmp_lt(b, h_next, c_H_round);
            h_end = rocke_b_select(b, lt, h_next, c_H_round);
        }
    }
    else
    {
        h_start = c->c0;
        h_end = c_H_round;
    }

    if(TDM)
    {
        rocke_value_t* c_BK;
        rocke_value_t* c_BC;
        rocke_value_t* k_left;
        rocke_value_t* c_left;
        {
            rocke_value_t* a = rocke_b_smem_addr_of(b, c->dy_lds);
            c->dy_lds_addr = rocke_b_trunc(b, a, rocke_i32());
        }
        {
            rocke_value_t* a = rocke_b_smem_addr_of(b, c->s_strip_lds);
            c->strip_lds_addr = rocke_b_trunc(b, a, rocke_i32());
        }
        c->k_tile_abs = rocke_b_add(b, c->k_group_base, k_tile_origin);
        c->c_tile_abs = rocke_b_add(b, c->c_group_base, c_tile_origin);
        c_BK = rocke_b_const_i32(b, block_k);
        c_BC = rocke_b_const_i32(b, block_c);
        k_left = rocke_b_sub(b, c_kpg, k_tile_origin);
        c_left = rocke_b_sub(b, c_cpg, c_tile_origin);
        /* Channel extents clipped to the tile, so the pad columns zero-fill. */
        {
            rocke_value_t* lt = rocke_b_cmp_lt(b, k_left, c_BK);
            c->tdm_k_ext = rocke_b_select(b, lt, k_left, c_BK);
        }
        {
            rocke_value_t* lt = rocke_b_cmp_lt(b, c_left, c_BC);
            c->tdm_c_ext = rocke_b_select(b, lt, c_left, c_BC);
        }
        c->c_DY_T = rocke_b_const_i32(b, G * c->DY_T);
        c->c_X_T = rocke_b_const_i32(b, G * c->X_T);
        c_TDM = rocke_b_const_i32(b, TDM);
        for(tk = 0; tk < TK; ++tk)
        {
            rocke_value_t* c_n = rocke_b_const_i32(b, TK);
            rocke_value_t* m = rocke_b_mul(b, wave_k_id, c_n);
            rocke_value_t* c_t = rocke_b_const_i32(b, tk);
            rocke_value_t* idx = rocke_b_add(b, m, c_t);
            dy_ch[tk] = rocke_b_mul(b, idx, c16);
        }
        for(tc = 0; tc < TC; ++tc)
        {
            rocke_value_t* c_n = rocke_b_const_i32(b, TC);
            rocke_value_t* m = rocke_b_mul(b, wave_c_id, c_n);
            rocke_value_t* c_t = rocke_b_const_i32(b, tc);
            rocke_value_t* idx = rocke_b_add(b, m, c_t);
            x_ch[tc] = rocke_b_mul(b, idx, c16);
        }
    }

    /* ---- outer loop: wo tiles ---- */
    for(i = 0; i < n_acc; ++i)
    {
        snprintf(iter_names[i], sizeof iter_names[i], "acc%d", i);
        wo_iargs[i].name = iter_names[i];
        wo_iargs[i].init = zero_acc;
    }
    {
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        wo_loop = rocke_b_scf_for_iter(
            b, c->c0, c_n_wo_tiles, c1, wo_iargs, n_acc, "wo_t", false, true);
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, wo_loop.body);
    {
        rocke_value_t* wo_start = rocke_b_mul(b, wo_loop.iv, c_WO_BLOCK);
        rocke_value_t* ring[ROCKE_DCONV_WGRAD_MAX_KH][ROCKE_WGRAD_IMG_MAX_TILES];
        rocke_value_t* pending0[ROCKE_WGRAD_IMG_MAX_PENDING];
        int np0 = 0;
        rocke_iter_arg_t row_iargs[ROCKE_WGRAD_IMG_MAX_ITER];
        int n_row_iter = 0;
        rocke_for_t row_loop;

        for(sl = 0; sl < KH; ++sl)
            for(tk = 0; tk < TK; ++tk)
                ring[sl][tk] = zero_frag;

        /* Ring prologue: dY rows of the KH-1 virtual input rows before row 0. */
        for(k = KH - 1; k > 0; --k)
        {
            int slot = (KH - k) % KH;
            rocke_value_t* vecs[ROCKE_WGRAD_IMG_MAX_ROW_LOADS];
            int nv = 0;
            rocke_value_t* ho_past;
            rocke_value_t* ho_ok;
            int pass_idx;
            if(spec->row_blocks > 1)
            {
                rocke_value_t* c_off = rocke_b_const_i32(b, c->p.PAD - k);
                ho_past = rocke_b_add(b, h_start, c_off);
            }
            else
            {
                ho_past = rocke_b_const_i32(b, c->p.PAD - k);
            }
            {
                rocke_value_t* ge = rocke_b_cmp_ge(b, ho_past, c->c0);
                rocke_value_t* lt = rocke_b_cmp_lt(b, ho_past, c->c_Ho);
                ho_ok = rocke_b_land(b, ge, lt);
            }
            if(!img_issue_dy(c, ho_past, ho_ok, wo_start, vecs, &nv))
                return NULL;
            for(tk = 0; tk < TK; ++tk)
            {
                for(pass_idx = 0; pass_idx < c->DY_PASSES; ++pass_idx)
                {
                    rocke_value_t* idx[2];
                    rocke_value_t* sp = img_dy_sp(c, pass_idx);
                    idx[0] = c->c0;
                    idx[1] = img_lds_run(c, c->dy_parts_g[0][tk], sp);
                    rocke_b_smem_store_vN(
                        b, c->dy_lds, idx, 2, vecs[tk * c->DY_PASSES + pass_idx], c->LD_VEC);
                }
            }
            rocke_b_sync_lds_only(b);
            img_read_dy(c, 0, ring[slot]);
            rocke_b_sync_lds_only(b);
        }

        if(TDM)
        {
            /* TDM staging: stage (step % TDM) holds the G rows of a step; each
             * step publishes its stage behind one barrier, then refills the
             * stage the previous step read, TDM - 1 steps ahead. */
            rocke_value_t* w_shift;
            int st;
            {
                rocke_value_t* eq = rocke_b_cmp_eq(b, wo_loop.iv, c->c0);
                rocke_value_t* c_pad = rocke_b_const_i32(b, c->p.PAD);
                w_shift = rocke_b_select(b, eq, c_pad, c->c0);
            }
            {
                rocke_value_t* eq = rocke_b_cmp_eq(b, wo_loop.iv, c->c0);
                rocke_if_t gate = rocke_b_scf_if(b, eq);
                rocke_b_region_enter(b, gate.then_region);
                {
                    /* Zero the PAD leading strip rows tile 0 leaves unwritten. */
                    rocke_value_t* zero_v = rocke_b_zero_vec(b, c->io_type, c->LD_VEC);
                    int row_chunks = c->p.PAD * block_c / c->LD_VEC;
                    int jj, slot;
                    for(jj = 0; jj < row_chunks; jj += THREADS)
                    {
                        rocke_value_t* c_j = rocke_b_const_i32(b, jj);
                        rocke_value_t* chunk = rocke_b_add(b, tid, c_j);
                        rocke_value_t* c_rc = rocke_b_const_i32(b, row_chunks);
                        rocke_value_t* lt = rocke_b_cmp_lt(b, chunk, c_rc);
                        rocke_if_t g2 = rocke_b_scf_if(b, lt);
                        rocke_b_region_enter(b, g2.then_region);
                        {
                            rocke_value_t* col;
                            rocke_value_t* row;
                            rocke_value_t* flat;
                            {
                                rocke_value_t* c_v = rocke_b_const_i32(b, c->LD_VEC);
                                col = rocke_b_mul(b, chunk, c_v);
                            }
                            {
                                rocke_value_t* c_bc = rocke_b_const_i32(b, block_c);
                                row = rocke_b_div(b, col, c_bc);
                            }
                            {
                                rocke_value_t* c_px = rocke_b_const_i32(b, c->P_X);
                                rocke_value_t* m = rocke_b_mul(b, row, c_px);
                                rocke_value_t* c_bc = rocke_b_const_i32(b, block_c);
                                rocke_value_t* md = rocke_b_mod(b, col, c_bc);
                                flat = rocke_b_add(b, m, md);
                            }
                            for(slot = 0; slot < TDM * G; ++slot)
                            {
                                rocke_value_t* idx[2];
                                rocke_value_t* c_s = rocke_b_const_i32(b, slot * c->X_T);
                                idx[0] = c->c0;
                                idx[1] = rocke_b_add(b, flat, c_s);
                                rocke_b_smem_store_vN(b, c->s_strip_lds, idx, 2, zero_v, c->LD_VEC);
                            }
                        }
                        rocke_b_region_leave(b);
                    }
                }
                rocke_b_region_leave(b);
            }
            for(st = 0; st < TDM - 1; ++st)
            {
                for(g = 0; g < G; ++g)
                {
                    rocke_value_t* c_o = rocke_b_const_i32(b, st * G + g);
                    rocke_value_t* hv = rocke_b_add(b, h_start, c_o);
                    rocke_value_t* c_st = rocke_b_const_i32(b, st);
                    if(!img_tdm_row(c, hv, wo_start, g, c_st, w_shift))
                        return NULL;
                }
            }

            for(i = 0; i < n_acc; ++i)
            {
                snprintf(iter_names[n_row_iter], sizeof iter_names[0], "r_acc%d", i);
                row_iargs[n_row_iter].name = iter_names[n_row_iter];
                row_iargs[n_row_iter].init = wo_loop.iter_vars[i];
                ++n_row_iter;
            }
            i = 0;
            for(sl = 0; sl < KH; ++sl)
                for(tk = 0; tk < TK; ++tk)
                {
                    snprintf(iter_names[n_row_iter], sizeof iter_names[0], "ring%d", i++);
                    row_iargs[n_row_iter].name = iter_names[n_row_iter];
                    row_iargs[n_row_iter].init = ring[sl][tk];
                    ++n_row_iter;
                }
            row_loop = rocke_b_scf_for_iter(
                b, h_start, h_end, c_KH, row_iargs, n_row_iter, "hi0", false, true);
            if(!rocke_ir_builder_ok(b))
                return NULL;
            rocke_b_region_enter(b, row_loop.body);
            {
                rocke_value_t* accs[ROCKE_WGRAD_IMG_MAX_ACC];
                rocke_value_t* yields[ROCKE_WGRAD_IMG_MAX_ITER];
                rocke_value_t* step0;
                int u0, ny = 0;

                for(i = 0; i < n_acc; ++i)
                    accs[i] = row_loop.iter_vars[i];
                for(sl = 0; sl < KH; ++sl)
                    for(tk = 0; tk < TK; ++tk)
                        ring[sl][tk] = row_loop.iter_vars[n_acc + sl * TK + tk];
                {
                    rocke_value_t* d = rocke_b_sub(b, row_loop.iv, h_start);
                    rocke_value_t* c_g = rocke_b_const_i32(b, G);
                    step0 = rocke_b_div(b, d, c_g);
                }
                for(u0 = 0; u0 < KH; u0 += G)
                {
                    rocke_value_t* step;
                    rocke_value_t* stage;
                    rocke_value_t* nxt;
                    rocke_value_t* refill;
                    rocke_value_t* dy_stage;
                    rocke_value_t* x_stage;
                    rocke_value_t* dy_rows[ROCKE_DCONV_WGRAD_MAX_KH][ROCKE_WGRAD_IMG_MAX_TILES];
                    rocke_value_t* x_rows[ROCKE_WGRAD_IMG_MAX_XFRAG]; /* [(g*KW+s)*TC+tc] */
                    {
                        rocke_value_t* c_u = rocke_b_const_i32(b, u0 / G);
                        step = rocke_b_add(b, step0, c_u);
                    }
                    stage = rocke_b_mod(b, step, c_TDM);
                    /* Each issuing wave has one transfer per step in flight. */
                    rocke_b_s_wait_tensorcnt(b, TDM - 2);
                    rocke_b_sync_lds_only(b);
                    {
                        rocke_value_t* c_n = rocke_b_const_i32(b, u0 + (TDM - 1) * G);
                        nxt = rocke_b_add(b, row_loop.iv, c_n);
                    }
                    {
                        rocke_value_t* c_r = rocke_b_const_i32(b, TDM - 1);
                        rocke_value_t* a = rocke_b_add(b, step, c_r);
                        refill = rocke_b_mod(b, a, c_TDM);
                    }
                    for(g = 0; g < G; ++g)
                    {
                        rocke_value_t* c_g = rocke_b_const_i32(b, g);
                        rocke_value_t* hv = rocke_b_add(b, nxt, c_g);
                        if(!img_tdm_row(c, hv, wo_start, g, refill, w_shift))
                            return NULL;
                    }
                    dy_stage = rocke_b_mul(b, stage, c->c_DY_T);
                    x_stage = rocke_b_mul(b, stage, c->c_X_T);
                    for(g = 0; g < G; ++g)
                    {
                        rocke_value_t* dy_slot;
                        rocke_value_t* x_slot;
                        {
                            rocke_value_t* c_o = rocke_b_const_i32(b, g * c->DY_T);
                            dy_slot = rocke_b_add(b, dy_stage, c_o);
                        }
                        {
                            rocke_value_t* c_o = rocke_b_const_i32(b, g * c->X_T);
                            x_slot = rocke_b_add(b, x_stage, c_o);
                        }
                        for(tk = 0; tk < TK; ++tk)
                        {
                            rocke_value_t* base = rocke_b_add(b, dy_slot, dy_ch[tk]);
                            dy_rows[g][tk] = img_tdm_tr_read(c, c->dy_lds, base, c->P_DY, 0);
                        }
                        for(s = 0; s < KW; ++s)
                            for(tc = 0; tc < TC; ++tc)
                            {
                                rocke_value_t* base = rocke_b_add(b, x_slot, x_ch[tc]);
                                x_rows[(g * KW + s) * TC + tc]
                                    = img_tdm_tr_read(c, c->s_strip_lds, base, c->P_X, s);
                            }
                    }
                    rocke_b_s_setprio(b, 1);
                    for(g = 0; g < G; ++g)
                    {
                        int u = u0 + g;
                        for(i = 0; i < n_acc; ++i)
                        {
                            int rr = acc_r[i];
                            rocke_value_t* dy_frag = (u - rr >= u0)
                                                         ? dy_rows[u - rr - u0][acc_tk[i]]
                                                         : ring[(u + KH - rr) % KH][acc_tk[i]];
                            rocke_value_t* xf = x_rows[(g * KW + acc_s[i]) * TC + acc_tc[i]];
                            accs[i] = rocke_b_mma(b, wmma_op_id, dy_frag, xf, accs[i], NULL, 0);
                        }
                    }
                    for(g = 0; g < G; ++g)
                        for(tk = 0; tk < TK; ++tk)
                            ring[u0 + g][tk] = dy_rows[g][tk];
                    rocke_b_s_setprio(b, 0);
                }
                for(i = 0; i < n_acc; ++i)
                    yields[ny++] = accs[i];
                for(sl = 0; sl < KH; ++sl)
                    for(tk = 0; tk < TK; ++tk)
                        yields[ny++] = ring[sl][tk];
                rocke_b_scf_yield(b, yields, ny);
            }
            rocke_b_region_leave(b);
            if(!rocke_ir_builder_ok(b))
                return NULL;
            /* The last steps prefetched rows past h_end; drain them before the
             * next wo tile's ring prologue writes LDS again. */
            rocke_b_s_wait_tensorcnt(b, 0);
            rocke_b_sync_lds_only(b);
        }
        else
        {
            for(g = 0; g < G; ++g)
            {
                rocke_value_t* hi_g;
                if(spec->row_blocks > 1)
                {
                    if(g == 0)
                    {
                        hi_g = h_start;
                    }
                    else
                    {
                        rocke_value_t* c_g = rocke_b_const_i32(b, g);
                        hi_g = rocke_b_add(b, h_start, c_g);
                    }
                }
                else
                {
                    hi_g = (g == 0) ? c->c0 : rocke_b_const_i32(b, g);
                }
                if(!img_issue_row(c, hi_g, wo_start, pending0, &np0))
                    return NULL;
            }
            if(np0 != n_pend)
            {
                img_fail(b);
                return NULL;
            }

            for(i = 0; i < n_acc; ++i)
            {
                snprintf(iter_names[n_row_iter], sizeof iter_names[0], "r_acc%d", i);
                row_iargs[n_row_iter].name = iter_names[n_row_iter];
                row_iargs[n_row_iter].init = wo_loop.iter_vars[i];
                ++n_row_iter;
            }
            i = 0;
            for(sl = 0; sl < KH; ++sl)
                for(tk = 0; tk < TK; ++tk)
                {
                    snprintf(iter_names[n_row_iter], sizeof iter_names[0], "ring%d", i++);
                    row_iargs[n_row_iter].name = iter_names[n_row_iter];
                    row_iargs[n_row_iter].init = ring[sl][tk];
                    ++n_row_iter;
                }
            for(i = 0; i < n_pend; ++i)
            {
                snprintf(iter_names[n_row_iter], sizeof iter_names[0], "pend%d", i);
                row_iargs[n_row_iter].name = iter_names[n_row_iter];
                row_iargs[n_row_iter].init = pending0[i];
                ++n_row_iter;
            }
            row_loop = rocke_b_scf_for_iter(
                b, h_start, h_end, c_KH, row_iargs, n_row_iter, "hi0", false, true);
            if(!rocke_ir_builder_ok(b))
                return NULL;
            rocke_b_region_enter(b, row_loop.body);
            {
                rocke_value_t* accs[ROCKE_WGRAD_IMG_MAX_ACC];
                rocke_value_t* pending[ROCKE_WGRAD_IMG_MAX_PENDING];
                rocke_value_t* yields[ROCKE_WGRAD_IMG_MAX_ITER];
                int u0, ny = 0;

                for(i = 0; i < n_acc; ++i)
                    accs[i] = row_loop.iter_vars[i];
                for(sl = 0; sl < KH; ++sl)
                    for(tk = 0; tk < TK; ++tk)
                        ring[sl][tk] = row_loop.iter_vars[n_acc + sl * TK + tk];
                for(i = 0; i < n_pend; ++i)
                    pending[i] = row_loop.iter_vars[n_acc + n_ring + i];

                for(u0 = 0; u0 < KH; u0 += G)
                {
                    rocke_value_t* nxt;
                    rocke_value_t* new_pending[ROCKE_WGRAD_IMG_MAX_PENDING];
                    int nnp = 0;
                    rocke_value_t* dy_rows[ROCKE_DCONV_WGRAD_MAX_KH][ROCKE_WGRAD_IMG_MAX_TILES];
                    rocke_value_t* x_rows[ROCKE_WGRAD_IMG_MAX_XFRAG]; /* [(g*KW+s)*TC+tc] */

                    rocke_b_s_setprio(b, 0);
                    for(g = 0; g < G; ++g)
                        img_commit_row(c, pending + g * n_row_loads, g);
                    {
                        rocke_value_t* c_n = rocke_b_const_i32(b, u0 + G);
                        nxt = rocke_b_add(b, row_loop.iv, c_n);
                    }
                    for(g = 0; g < G; ++g)
                    {
                        rocke_value_t* hi_g;
                        if(g == 0)
                        {
                            hi_g = nxt;
                        }
                        else
                        {
                            rocke_value_t* c_g = rocke_b_const_i32(b, g);
                            hi_g = rocke_b_add(b, nxt, c_g);
                        }
                        if(!img_issue_row(c, hi_g, wo_start, new_pending, &nnp))
                            return NULL;
                    }
                    for(i = 0; i < nnp; ++i)
                        pending[i] = new_pending[i];
                    rocke_b_sync_lds_only(b);
                    for(g = 0; g < G; ++g)
                    {
                        img_read_dy(c, g, dy_rows[g]);
                        for(s = 0; s < KW; ++s)
                            for(tc = 0; tc < TC; ++tc)
                                x_rows[(g * KW + s) * TC + tc]
                                    = img_tr_read(c, c->s_strip_lds, c->strip_parts_g[g][tc], s);
                    }
                    rocke_b_s_setprio(b, 1);
                    for(g = 0; g < G; ++g)
                    {
                        int u = u0 + g;
                        for(i = 0; i < n_acc; ++i)
                        {
                            int rr = acc_r[i];
                            rocke_value_t* dy_frag = (u - rr >= u0)
                                                         ? dy_rows[u - rr - u0][acc_tk[i]]
                                                         : ring[(u + KH - rr) % KH][acc_tk[i]];
                            rocke_value_t* xf = x_rows[(g * KW + acc_s[i]) * TC + acc_tc[i]];
                            accs[i] = rocke_b_mma(b, wmma_op_id, dy_frag, xf, accs[i], NULL, 0);
                        }
                    }
                    for(g = 0; g < G; ++g)
                        for(tk = 0; tk < TK; ++tk)
                            ring[u0 + g][tk] = dy_rows[g][tk];
                    rocke_b_s_setprio(b, 0);
                    /* Drain this step's LDS reads before the next step's commits
                     * overwrite the slots: a bare s_barrier lets a wave pass with
                     * transpose reads still in flight (measured wrong results). */
                    rocke_b_sync_lds_only(b);
                }
                for(i = 0; i < n_acc; ++i)
                    yields[ny++] = accs[i];
                for(sl = 0; sl < KH; ++sl)
                    for(tk = 0; tk < TK; ++tk)
                        yields[ny++] = ring[sl][tk];
                for(i = 0; i < n_pend; ++i)
                    yields[ny++] = pending[i];
                rocke_b_scf_yield(b, yields, ny);
            }
            rocke_b_region_leave(b);
            if(!rocke_ir_builder_ok(b))
                return NULL;
        }
        rocke_b_scf_yield(b, row_loop.op->results, n_acc);
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    for(i = 0; i < n_acc; ++i)
        final_accs[i] = wo_loop.op->results[i];

    /* ---- epilogue: cascade buffer atomics into the fp32 dW (or its replica) ---- */
    {
        rocke_value_t* k_in_group_base;
        rocke_value_t* c_in_group_base;
        rocke_value_t* ws_shift = NULL;
        if(spec->ws_replicas > 1)
        {
            /* Replica slab (group * R + rep) sits (group * (R - 1) + rep) slabs
             * past the group's own dW slab, which dw_off already addresses. */
            const int R = spec->ws_replicas;
            rocke_value_t* slab;
            rocke_value_t* rep;
            {
                rocke_value_t* c_rs = rocke_b_const_i32(b, KH * KW);
                rocke_value_t* m = rocke_b_mul(b, c_rs, c_cpg);
                slab = rocke_b_mul(b, c_kpg, m);
            }
            {
                rocke_value_t* c_rb = rocke_b_const_i32(b, spec->row_blocks);
                rocke_value_t* m = rocke_b_mul(b, c->n_i, c_rb);
                rocke_value_t* by = rocke_b_block_id_y(b);
                rocke_value_t* sum = rocke_b_add(b, m, by);
                rocke_value_t* c_r = rocke_b_const_i32(b, R);
                rep = rocke_b_mod(b, sum, c_r);
            }
            {
                rocke_value_t* c_r1 = rocke_b_const_i32(b, R - 1);
                rocke_value_t* m = rocke_b_mul(b, group, c_r1);
                rocke_value_t* sum = rocke_b_add(b, m, rep);
                ws_shift = rocke_b_mul(b, sum, slab);
            }
        }
        if(spec->lds_epilogue)
        {
            img_lds_epilogue(b,
                             spec,
                             final_accs,
                             acc_r,
                             acc_s,
                             acc_tk,
                             acc_tc,
                             n_acc,
                             wave_k_id,
                             wave_c_id,
                             c4,
                             q_in_lane,
                             tid,
                             k_tile_origin,
                             c_tile_origin,
                             c->k_group_base,
                             c_kpg,
                             c_cpg,
                             ws_shift,
                             d_rsrc);
            if(!rocke_ir_builder_ok(b))
                return NULL;
            return rocke_ir_builder_kernel(b);
        }
        {
            rocke_value_t* c_cf = rocke_b_const_i32(b, C_FRAG);
            rocke_value_t* m = rocke_b_mul(b, c4, c_cf);
            k_in_group_base = rocke_b_add(b, c->k_wave_in_group, m);
        }
        c_in_group_base = rocke_b_add(b, c->c_wave_in_group, q_in_lane);
        for(i = 0; i < n_acc; ++i)
        {
            rocke_value_t* c_in_group;
            rocke_value_t* c_ok;
            int slot;
            {
                rocke_value_t* c_t = rocke_b_const_i32(b, 16 * acc_tc[i]);
                c_in_group = rocke_b_add(b, c_in_group_base, c_t);
            }
            c_ok = rocke_b_cmp_lt(b, c_in_group, c_cpg);
            for(slot = 0; slot < C_FRAG; ++slot)
            {
                rocke_value_t* k_in_group;
                rocke_value_t* ok;
                rocke_value_t* dw_off = NULL;
                rocke_value_t* dw_valid = NULL;
                rocke_value_t* val;
                const char* names[4] = {"k", "r", "s", "c"};
                rocke_value_t* values[4];
                {
                    rocke_value_t* c_sl = rocke_b_const_i32(b, 16 * acc_tk[i] + slot);
                    k_in_group = rocke_b_add(b, k_in_group_base, c_sl);
                }
                {
                    rocke_value_t* lt = rocke_b_cmp_lt(b, k_in_group, c_kpg);
                    ok = rocke_b_land(b, lt, c_ok);
                }
                values[0] = rocke_b_add(b, c->k_group_base, k_in_group);
                values[1] = rocke_b_const_i32(b, acc_r[i]);
                values[2] = rocke_b_const_i32(b, acc_s[i]);
                values[3] = c_in_group;
                if(!rocke_transforms_descriptor_offset(
                       b, dw_desc, names, values, 4, &dw_off, &dw_valid))
                {
                    return NULL;
                }
                if(ws_shift != NULL)
                    dw_off = rocke_b_add(b, dw_off, ws_shift);
                val = rocke_b_vec_extract(b, final_accs[i], slot);
                {
                    rocke_if_t gate = rocke_b_scf_if(b, ok);
                    rocke_b_region_enter(b, gate.then_region);
                    {
                        rocke_value_t* c_4 = rocke_b_const_i32(b, 4);
                        rocke_value_t* voff = rocke_b_mul(b, dw_off, c_4);
                        rocke_b_buffer_atomic_fadd_f32(
                            b, d_rsrc, voff, c->c0, val, ROCKE_WGRAD_IMG_CPOL_ATOMIC_CASCADE_DEV);
                    }
                    rocke_b_region_leave(b);
                }
            }
        }
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}
