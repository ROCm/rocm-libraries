// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_generic.cpp -- C99 port of build_direct_conv, the
 * parametric direct grouped convolution kernel for any cpg that is a multiple
 * of 4 (library/kernels/common/conv_direct_grouped.py). The MFMA dgrad
 * pipeline runs this kernel as its compute pass, on the transposed problem.
 *
 *   Python (conv_direct_grouped.py)         C99 (this file)
 *   build_direct_conv(spec, arch)       ->  rocke_build_direct_conv()
 *     issue_dram_load(y, g_tile)        ->  dconv_issue_dram_load()
 *     store_to_lds(loads, lds)          ->  dconv_store_to_lds()
 *   _krsc_descriptor_dynamic(...)       ->  rocke_dconv_krsc_descriptor_dynamic()
 *
 * The Python body is one long function with two closures over a wide set of
 * locals; here those locals live in a TU-private context struct and the
 * closures are static helpers over it. The persistent-grid path re-binds n,
 * g, h_tile_start and the q-tile starts inside its cell loop, which the
 * closures read at call time -- the ctx fields are updated at the same point
 * (and xcd_tiles re-binds bx / by / bz before any of them is read).
 *
 * Every DirectConvSpec form is mirrored: plain-weight runtime loops, the
 * runtime K-atom loop, waves_q / waves_k (with the cross-wave reduction),
 * the persistent grid, runtime channels, fold_k32, and the fused-weight dgrad
 * and row-stream knobs (preload_weights, dgrad_fused_weights,
 * dgrad_weights_lds, waves_per_eu, row_guard, prefetch_rows, lds_only_sync,
 * waves_m, xcd_tiles, lds_pad, stage_out, row_exit).
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first): C argument evaluation order is unspecified, so every
 * expression with more than one IR-emitting operand is sequenced through
 * locals. That order is what keeps the SSA numbering -- and therefore the .ll
 * bytes -- identical to the Python engine.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.grid.h" /* chiplet_transform_chunked_runtime */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* 256 = 8 XCDs x 32 blocks per XCD: the persistent grid's block count. */
#define DCONV_GEN_NUM_XCD 8
#define DCONV_GEN_BLOCKS_PER_XCD 32
/* Bound on the DRAM-load passes per row (ceil(NUM_CHUNKS / THREADS)). */
#define DCONV_GEN_MAX_PASSES 256

typedef struct dconv_gen_chunk
{
    rocke_value_t* chunk_idx;
    rocke_value_t* ch_block;
    rocke_value_t* W_lds;
    rocke_value_t* in_bounds;
    rocke_value_t* group_in_wg;
} dconv_gen_chunk_t;

typedef struct dconv_gen_ctx
{
    rocke_ir_builder_t* b;
    rocke_direct_conv_problem_t p;
    bool is_bf16;
    const rocke_type_t* io_type;
    int LOAD_VEC;
    int PASSES;
    int LDS_PAD;

    rocke_dconv_params_t params;
    rocke_value_t* c0;
    rocke_value_t* c_BG;
    rocke_value_t* c_cpg;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* by;
    rocke_value_t* a_rsrc;
    rocke_tensor_descriptor_t* a_desc;

    /* Re-bound per cell by the persistent grid; read at call time. */
    rocke_value_t* n;
    rocke_value_t* q_tile_start_lds;

    dconv_gen_chunk_t chunk_meta[DCONV_GEN_MAX_PASSES];
} dconv_gen_ctx_t;

static rocke_value_t*
    dconv_gen_buf_load(dconv_gen_ctx_t* ctx, rocke_value_t* rsrc, rocke_value_t* voff, int dwords)
{
    if(ctx->is_bf16)
        return rocke_b_buffer_load_vN_bf16(ctx->b, rsrc, voff, ctx->c0, dwords);
    return rocke_b_buffer_load_vN_f16(ctx->b, rsrc, voff, ctx->c0, dwords);
}

static void dconv_gen_buf_store(
    dconv_gen_ctx_t* ctx, rocke_value_t* rsrc, rocke_value_t* voff, rocke_value_t* val, int dwords)
{
    if(ctx->is_bf16)
        rocke_b_buffer_store_vN_bf16(ctx->b, rsrc, voff, ctx->c0, val, dwords);
    else
        rocke_b_buffer_store_vN_f16(ctx->b, rsrc, voff, ctx->c0, val, dwords);
}

static rocke_value_t* dconv_gen_mfma(
    dconv_gen_ctx_t* ctx, bool fold_k32, rocke_value_t* a, rocke_value_t* bb, rocke_value_t* acc)
{
    rocke_ir_builder_t* b = ctx->b;
    if(fold_k32)
    {
        return ctx->is_bf16 ? rocke_b_mfma_f32_16x16x32_bf16(b, a, bb, acc)
                            : rocke_b_mfma_f32_16x16x32_f16(b, a, bb, acc);
    }
    return ctx->is_bf16 ? rocke_b_mfma_f32_16x16x16_bf16(b, a, bb, acc)
                        : rocke_b_mfma_f32_16x16x16_f16(b, a, bb, acc);
}

/* _krsc_descriptor_dynamic(b, name, coords, inner, p): [rows, KH, KW, inner]
 * with a runtime innermost extent, strides [KH*KW*inner, KW*inner, inner, 1]. */
rocke_tensor_descriptor_t* rocke_dconv_krsc_descriptor_dynamic(rocke_ir_builder_t* b,
                                                               const char* name,
                                                               const char* const* coords,
                                                               rocke_value_t* inner,
                                                               int KH,
                                                               int KW)
{
    rocke_value_t* strides[4];
    rocke_dynamic_tensor_descriptor_t* dyn;
    {
        rocke_value_t* c_khkw = rocke_b_const_i32(b, KH * KW);
        strides[0] = rocke_b_mul(b, inner, c_khkw);
    }
    {
        rocke_value_t* c_kw = rocke_b_const_i32(b, KW);
        strides[1] = rocke_b_mul(b, inner, c_kw);
    }
    strides[2] = inner;
    strides[3] = rocke_b_const_i32(b, 1);
    dyn = rocke_tensor_descriptor_naive_dynamic(b, name, coords, 4, strides);
    return dyn ? &dyn->base : NULL;
}

/* issue_dram_load(y_iter_val, g_tile_val): one LOAD_VEC-wide DRAM read per
 * loader pass; out-of-range chunks read zeros. Writes PASSES (vec, lds_idx). */
static bool dconv_issue_dram_load(dconv_gen_ctx_t* ctx,
                                  rocke_value_t* y_iter_val,
                                  rocke_value_t* g_tile_val,
                                  rocke_value_t** out_vec,
                                  rocke_value_t** out_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* g_tile = g_tile_val ? g_tile_val : ctx->by;
    int i;
    for(i = 0; i < ctx->PASSES; ++i)
    {
        const dconv_gen_chunk_t* cm = &ctx->chunk_meta[i];
        rocke_value_t* abs_group;
        rocke_value_t* c_val;
        rocke_value_t* a_off = NULL;
        rocke_value_t* addr_valid = NULL;
        rocke_value_t* valid;
        rocke_value_t* safe_off;
        rocke_value_t* a_vec;
        const char* names[5];
        rocke_value_t* values[5];
        {
            rocke_value_t* mul_g = rocke_b_mul(b, g_tile, ctx->c_BG);
            abs_group = rocke_b_add(b, mul_g, cm->group_in_wg);
        }
        {
            rocke_value_t* mul_ag = rocke_b_mul(b, abs_group, ctx->c_cpg);
            rocke_value_t* c_lv = rocke_b_const_i32(b, ctx->LOAD_VEC);
            rocke_value_t* mul_cb = rocke_b_mul(b, cm->ch_block, c_lv);
            c_val = rocke_b_add(b, mul_ag, mul_cb);
        }
        names[0] = "n";
        values[0] = ctx->n;
        names[1] = "y_iter";
        values[1] = y_iter_val;
        names[2] = "q_pos";
        values[2] = ctx->q_tile_start_lds;
        names[3] = "W_lds_pos";
        values[3] = cm->W_lds;
        names[4] = "c";
        values[4] = c_val;
        if(!rocke_transforms_descriptor_offset(
               b, ctx->a_desc, names, values, 5, &a_off, &addr_valid))
            return false;
        valid = rocke_b_land(b, addr_valid, cm->in_bounds);
        {
            rocke_value_t* off_bytes = rocke_b_mul(b, a_off, ctx->c_half_bytes);
            safe_off = rocke_b_select(b, valid, off_bytes, ctx->oob_sentinel);
        }
        a_vec = dconv_gen_buf_load(ctx, ctx->a_rsrc, safe_off, ctx->LOAD_VEC / 2);
        {
            rocke_value_t* zero = rocke_b_zero_vec(b, ctx->io_type, ctx->LOAD_VEC);
            a_vec = rocke_b_select(b, valid, a_vec, zero);
        }
        out_vec[i] = a_vec;
        out_idx[i] = rocke_b_mul(b, cm->chunk_idx, rocke_b_const_i32(b, ctx->LOAD_VEC));
        if(ctx->LDS_PAD)
        {
            /* Column W_lds starts LDS_PAD * W_lds elements later. */
            rocke_value_t* pad = rocke_b_mul(b, cm->W_lds, rocke_b_const_i32(b, ctx->LDS_PAD));
            out_idx[i] = rocke_b_add(b, out_idx[i], pad);
        }
    }
    return true;
}

/* store_to_lds(loads, lds) */
static void dconv_store_to_lds(dconv_gen_ctx_t* ctx,
                               rocke_value_t* const* vecs,
                               rocke_value_t* const* idx,
                               rocke_value_t* lds)
{
    int i;
    for(i = 0; i < ctx->PASSES; ++i)
    {
        rocke_value_t* indices[2];
        indices[0] = ctx->c0;
        indices[1] = idx[i];
        rocke_b_smem_store_vN(ctx->b, lds, indices, 2, vecs[i], ctx->LOAD_VEC);
    }
}

/* W_lds column of q-subtile qt at filter column s:
 * (q_in_lane + qt_w_base) * stride + s_val [+ wave_id_q * BLOCK_Q_WAVE]. */
static rocke_value_t* dconv_gen_w_lds_idx(rocke_ir_builder_t* b,
                                          rocke_value_t* q_in_lane,
                                          int qt_w_base,
                                          int stride,
                                          rocke_value_t* s_val)
{
    rocke_value_t* c_base = rocke_b_const_i32(b, qt_w_base);
    rocke_value_t* q = rocke_b_add(b, q_in_lane, c_base);
    rocke_value_t* c_stride = rocke_b_const_i32(b, stride);
    rocke_value_t* qs = rocke_b_mul(b, q, c_stride);
    return rocke_b_add(b, qs, s_val);
}

/* lds_idx = (W_lds_idx * c_BG_cpg + wave_group_idx * c_cpg) + ch_off */
static rocke_value_t* dconv_gen_lds_idx(rocke_ir_builder_t* b,
                                        rocke_value_t* w_lds_idx,
                                        rocke_value_t* c_BG_cpg,
                                        rocke_value_t* wave_group_idx,
                                        rocke_value_t* c_cpg,
                                        rocke_value_t* ch_off)
{
    rocke_value_t* m1 = rocke_b_mul(b, w_lds_idx, c_BG_cpg);
    rocke_value_t* m2 = rocke_b_mul(b, wave_group_idx, c_cpg);
    rocke_value_t* s = rocke_b_add(b, m1, m2);
    return rocke_b_add(b, s, ch_off);
}

/* k_tail_bound(): channels of a group, the bound the K-tail masks compare
 * against -- the runtime cpg, or a fresh const_i32(cpg) per use. */
static rocke_value_t* k_tail_bound(rocke_ir_builder_t* b, bool rtc, rocke_value_t* c_cpg, int cpg)
{
    return rtc ? c_cpg : rocke_b_const_i32(b, cpg);
}

static int dconv_gen_pymod(int a, int m)
{
    int r = a % m;
    return r < 0 ? r + m : r;
}

rocke_kernel_def_t* rocke_build_direct_conv(rocke_ir_builder_t* b,
                                            const rocke_direct_conv_spec_t* spec,
                                            const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    dconv_gen_ctx_t* ctx;
    rocke_direct_conv_problem_t p;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    /* spec.validate(), then the is_valid_spec gate -- both raise ValueError. */
    if(rocke_direct_conv_validate(spec, reason, sizeof reason) != ROCKE_OK)
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
    if(!rocke_direct_conv_is_valid_spec(spec, arch, reason, sizeof reason))
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid DirectConvSpec for %s: %s", arch, reason);
    /* Python raises here; stop before the geometry below divides by an
     * unvalidated wave count. */
    if(!rocke_ir_builder_ok(b))
        return NULL;

    ctx = (dconv_gen_ctx_t*)rocke_arena_calloc(&b->arena, sizeof(*ctx));
    if(ctx == NULL)
        return NULL;
    ctx->b = b;
    p = spec->problem;
    ctx->p = p;
    ctx->is_bf16 = p.dtype != NULL && strcmp(p.dtype, "bf16") == 0;
    ctx->io_type = rocke_b_io_ir_type(b, p.dtype);
    if(ctx->io_type == NULL)
        return NULL;

    const int BLOCK_Q = spec->block_q;
    const int BLOCK_GROUPS = spec->block_groups;
    const int WAVES_Q = spec->waves_q;
    const int WAVES_K = spec->waves_k;
    const int WAVE = spec->wave_size;
    const int THREADS = rocke_direct_conv_threads_per_block(spec);
    const bool FOLD_K32 = spec->fold_k32;
    const int K_ATOM_SIZE = FOLD_K32 ? 32 : 16;
    const int LOAD_VEC = FOLD_K32 ? 8 : 4;
    const bool RTC = spec->runtime_channels;
    int N_K_ATOMS;
    if(RTC)
        N_K_ATOMS = spec->k_atoms;
    else if(FOLD_K32)
        N_K_ATOMS = p.cpg / K_ATOM_SIZE;
    else
        N_K_ATOMS = (p.cpg + K_ATOM_SIZE - 1) / K_ATOM_SIZE;
    const int N_K_LOCAL = N_K_ATOMS / WAVES_K;
    const int N_M_TILES = RTC ? spec->m_tiles : (p.kpg + 15) / 16;
    const int WAVES_M = spec->waves_m;
    const int M_LOCAL = N_M_TILES / WAVES_M; /* M-tiles owned by one wave */
    const int CPG_LDS = RTC ? N_K_ATOMS * K_ATOM_SIZE : p.cpg;
    const int N_VECS = CPG_LDS / LOAD_VEC;
    const int BLOCK_Q_WAVE = BLOCK_Q / WAVES_Q;
    const int q_subtiles = BLOCK_Q_WAVE / 16;
    const int LDS_W = (BLOCK_Q - 1) * p.stride + p.KW;
    const int NUM_CHUNKS = LDS_W * BLOCK_GROUPS * N_VECS;
    const int PASSES = (NUM_CHUNKS + THREADS - 1) / THREADS;
    /* lds_pad: LDS_PAD extra elements after every staged input column; room
     * for every column the passes can touch. */
    const int LDS_PAD = spec->lds_pad;
    int lds_total = PASSES * THREADS * LOAD_VEC;
    if(LDS_PAD)
    {
        const int col_vecs = BLOCK_GROUPS * N_VECS;
        lds_total += ((PASSES * THREADS + col_vecs - 1) / col_vecs + 1) * LDS_PAD;
    }
    const int BLOCK_H = spec->block_h;
    const bool PERSISTENT = spec->persistent_grid;
    const int KH = p.KH;
    const int KW = p.KW;
    /* A partial last K-atom reads past the group's channels; with runtime
     * channels any cpg may leave one, against the runtime bound -- except
     * under fold_k32, whose cpg is a multiple of the atom (validated). */
    const bool mask_k_tail = RTC ? !FOLD_K32 : (p.cpg % K_ATOM_SIZE) != 0;
    /* The preloads_weights property. */
    const bool PRELOADS = WAVES_K > 1 || spec->preload_weights || spec->dgrad_fused_weights;

    if(PASSES > DCONV_GEN_MAX_PASSES)
    {
        /* chunk_meta[] holds DCONV_GEN_MAX_PASSES entries. */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "build_direct_conv: %d LDS loader passes per row exceed %d",
                        PASSES,
                        DCONV_GEN_MAX_PASSES);
        return NULL;
    }
    ctx->LOAD_VEC = LOAD_VEC;
    ctx->PASSES = PASSES;
    ctx->LDS_PAD = LDS_PAD;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->waves_per_eu > 0)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", (int64_t)spec->waves_per_eu);
    rocke_dconv_emit_params(b, &ctx->params, "fwd", ctx->io_type);
    rocke_dconv_params_t* params = &ctx->params;
    rocke_value_t* p_Hi = params->p_Hi;
    rocke_value_t* p_Wo = params->p_Wo;

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_wave = rocke_b_const_i32(b, WAVE);
    rocke_value_t* c_BG = rocke_b_const_i32(b, BLOCK_GROUPS);
    rocke_value_t* c_BQ = rocke_b_const_i32(b, BLOCK_Q);
    rocke_value_t* c_cpg;
    rocke_value_t* c_kpg;
    if(RTC)
    {
        c_cpg = rocke_b_div(b, params->p_total_c, params->p_groups);
        c_kpg = rocke_b_div(b, params->p_total_k, params->p_groups);
    }
    else
    {
        c_cpg = rocke_b_const_i32(b, p.cpg);
        c_kpg = rocke_b_const_i32(b, p.kpg);
    }
    rocke_value_t* c_W = p_Wo;
    rocke_value_t* c_KW = rocke_b_const_i32(b, KW);
    rocke_value_t* c_N_K_LOCAL = rocke_b_const_i32(b, N_K_LOCAL);
    rocke_value_t* c_BG_cpg;
    if(RTC)
        c_BG_cpg = rocke_b_mul(b, c_cpg, rocke_b_const_i32(b, BLOCK_GROUPS));
    else
        c_BG_cpg = rocke_b_const_i32(b, BLOCK_GROUPS * p.cpg + LDS_PAD);
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, 2);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);

    (void)rocke_b_zero_vec(b, ctx->io_type, 4); /* fp16x4_zero (unused) */
    rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 4);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    const int waves_per_group = WAVES_Q * WAVES_K;
    rocke_value_t* wave_id_full = rocke_b_div(b, tid, c_wave);
    rocke_value_t* wave_id_in_group
        = rocke_b_mod(b, wave_id_full, rocke_b_const_i32(b, waves_per_group));
    rocke_value_t* wave_group_idx
        = rocke_b_div(b, wave_id_full, rocke_b_const_i32(b, waves_per_group));
    rocke_value_t* wave_id_q = rocke_b_div(b, wave_id_in_group, rocke_b_const_i32(b, WAVES_K));
    rocke_value_t* wave_id_k = rocke_b_mod(b, wave_id_in_group, rocke_b_const_i32(b, WAVES_K));
    rocke_value_t* wm_base16 = NULL;
    if(WAVES_M > 1)
    {
        /* waves_q == waves_k == 1 (validated): the in-group wave index is the
         * M-wave, which owns output channels [wave_id_m * M_LOCAL * 16,
         * +16 * M_LOCAL) of its group. */
        wave_group_idx = rocke_b_div(b, wave_id_full, rocke_b_const_i32(b, WAVES_M));
        rocke_value_t* wave_id_m = rocke_b_mod(b, wave_id_full, rocke_b_const_i32(b, WAVES_M));
        wave_id_q = c0;
        wave_id_k = c0;
        wm_base16 = rocke_b_mul(b, wave_id_m, rocke_b_const_i32(b, M_LOCAL * 16));
    }
    rocke_value_t* k_atom_base = rocke_b_mul(b, wave_id_k, rocke_b_const_i32(b, N_K_LOCAL));

    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);
    rocke_value_t* c4 = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    rocke_value_t* q_in_lane = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));

    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* bz = rocke_b_block_id_z(b);

    rocke_value_t* n = NULL;
    rocke_value_t* h_tile_start = NULL;
    rocke_value_t* n_h_tiles_v = NULL;
    rocke_value_t* q_tiles_p_v = NULL;
    rocke_value_t* nhq_v = NULL;
    rocke_value_t* nhqg_v = NULL;
    rocke_value_t* n_cells_v = NULL;
    rocke_value_t* wg_in_xcd = NULL;
    rocke_value_t* xcd_start = NULL;
    if(PERSISTENT)
    {
        rocke_value_t* c_block_h_p = rocke_b_const_i32(b, BLOCK_H);
        {
            rocke_value_t* c_r = rocke_b_const_i32(b, BLOCK_H - 1);
            rocke_value_t* s = rocke_b_add(b, p_Hi, c_r);
            n_h_tiles_v = rocke_b_div(b, s, c_block_h_p);
        }
        {
            rocke_value_t* c_r = rocke_b_const_i32(b, BLOCK_Q - 1);
            rocke_value_t* s = rocke_b_add(b, p_Wo, c_r);
            q_tiles_p_v = rocke_b_div(b, s, c_BQ);
        }
        rocke_value_t* g_tiles_p_v = rocke_b_div(b, params->p_groups, c_BG);
        nhq_v = rocke_b_mul(b, n_h_tiles_v, q_tiles_p_v);
        nhqg_v = rocke_b_mul(b, nhq_v, g_tiles_p_v);
        n_cells_v = rocke_b_mul(b, params->p_N, nhqg_v);
        rocke_value_t* cells_per_xcd_v;
        {
            rocke_value_t* c_r = rocke_b_const_i32(b, DCONV_GEN_NUM_XCD - 1);
            rocke_value_t* s = rocke_b_add(b, n_cells_v, c_r);
            rocke_value_t* c_x = rocke_b_const_i32(b, DCONV_GEN_NUM_XCD);
            cells_per_xcd_v = rocke_b_div(b, s, c_x);
        }
        rocke_value_t* xcd_id = rocke_b_mod(b, bx, rocke_b_const_i32(b, DCONV_GEN_NUM_XCD));
        wg_in_xcd = rocke_b_div(b, bx, rocke_b_const_i32(b, DCONV_GEN_NUM_XCD));
        xcd_start = rocke_b_mul(b, xcd_id, cells_per_xcd_v);

        n = rocke_b_const_i32(b, 0);
        h_tile_start = rocke_b_const_i32(b, 0);
        (void)rocke_b_const_i32(b, 0); /* q_tile_start_cell */
        (void)rocke_b_const_i32(b, 0); /* g_tile_cell */
    }
    else
    {
        rocke_value_t* c_block_h = rocke_b_const_i32(b, BLOCK_H);
        rocke_value_t* c_n_h_tiles;
        {
            rocke_value_t* c_r = rocke_b_const_i32(b, BLOCK_H - 1);
            rocke_value_t* s = rocke_b_add(b, p_Hi, c_r);
            c_n_h_tiles = rocke_b_div(b, s, c_block_h);
        }
        if(spec->xcd_tiles)
        {
            /* Launch order is q tile fastest, then group tile, then image row
             * band (n, h_tile). Remap it so each run of q_tiles * group_tiles
             * workgroups -- one row band -- lands on one XCD. Grid and chunk
             * come from the kernargs; only the XCD count is baked. */
            rocke_value_t* xc_gx;
            {
                rocke_value_t* c_r = rocke_b_const_i32(b, BLOCK_Q - 1);
                rocke_value_t* s = rocke_b_add(b, p_Wo, c_r);
                xc_gx = rocke_b_div(b, s, c_BQ);
            }
            rocke_value_t* xc_gy = rocke_b_div(b, params->p_groups, c_BG);
            rocke_value_t* xc_gxy = rocke_b_mul(b, xc_gx, xc_gy);
            rocke_value_t* xc_lin;
            {
                rocke_value_t* m1 = rocke_b_mul(b, by, xc_gx);
                rocke_value_t* m2 = rocke_b_mul(b, bz, xc_gxy);
                rocke_value_t* a1 = rocke_b_add(b, m1, m2);
                xc_lin = rocke_b_add(b, bx, a1);
            }
            rocke_value_t* num_wgs;
            {
                rocke_value_t* m1 = rocke_b_mul(b, params->p_N, c_n_h_tiles);
                num_wgs = rocke_b_mul(b, xc_gxy, m1);
            }
            /* _XCD_TILES_ARCHES[arch]: is_valid_spec admits gfx950 only. */
            xc_lin = rocke_chiplet_transform_chunked_runtime(
                b, xc_lin, num_wgs, DCONV_GEN_NUM_XCD, xc_gxy);
            if(xc_lin == NULL)
                return NULL;
            bx = rocke_b_mod(b, xc_lin, xc_gx);
            {
                rocke_value_t* d = rocke_b_div(b, xc_lin, xc_gx);
                by = rocke_b_mod(b, d, xc_gy);
            }
            bz = rocke_b_div(b, xc_lin, xc_gxy);
        }
        n = rocke_b_div(b, bz, c_n_h_tiles);
        rocke_value_t* h_tile_idx = rocke_b_mod(b, bz, c_n_h_tiles);
        h_tile_start = rocke_b_mul(b, h_tile_idx, c_block_h);
    }

    rocke_value_t* g;
    if(PERSISTENT)
        g = rocke_b_const_i32(b, 0);
    else
    {
        rocke_value_t* m = rocke_b_mul(b, by, c_BG);
        g = rocke_b_add(b, m, wave_group_idx);
    }
    rocke_value_t* block_q_start = rocke_b_mul(b, bx, c_BQ);
    rocke_value_t* q_tile_start;
    {
        rocke_value_t* c_bqw = rocke_b_const_i32(b, BLOCK_Q_WAVE);
        rocke_value_t* m = rocke_b_mul(b, wave_id_q, c_bqw);
        q_tile_start = rocke_b_add(b, block_q_start, m);
    }
    rocke_value_t* q_tile_start_lds = block_q_start;

    /* _alloc_row_buffers(). With dgrad_weights_lds they are allocated after
     * the weight transpose reads, so their live ranges start after lds_w dies
     * and the pool packer overlays them. */
    rocke_value_t* A_smem = NULL;
    rocke_value_t* B_smem = NULL;
    int row_shape[2];
    row_shape[0] = 1;
    row_shape[1] = lds_total;
#define DCONV_GEN_ALLOC_ROW_BUFFERS()                                                             \
    do                                                                                            \
    {                                                                                             \
        A_smem = rocke_b_smem_alloc(b, ctx->io_type, row_shape, 2, "lds_a");                      \
        B_smem = spec->double_buffer ? rocke_b_smem_alloc(b, ctx->io_type, row_shape, 2, "lds_b") \
                                     : A_smem;                                                    \
    } while(0)
    if(!spec->dgrad_weights_lds)
        DCONV_GEN_ALLOC_ROW_BUFFERS();

    rocke_value_t* red_lds = NULL;
    rocke_value_t* lds_row_idx = NULL;
    if(WAVES_K > 1)
    {
        int shape[2];
        shape[0] = WAVES_Q * WAVES_K;
        shape[1] = WAVE * 4;
        red_lds = rocke_b_smem_alloc_f32(b, shape, 2, "red_lds");
        rocke_value_t* m = rocke_b_mul(b, wave_id_q, rocke_b_const_i32(b, WAVES_K));
        lds_row_idx = rocke_b_add(b, m, wave_id_k);
    }

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, params->A, params->A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, params->Bp, params->B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, params->D, params->D_bytes);

    {
        rocke_dynamic_tensor_descriptor_t* a_dyn
            = rocke_dconv_a_descriptor_dynamic(b, params, p.PAD, p.stride, "q_pos", "W_lds_pos");
        if(a_dyn == NULL)
            return NULL;
        ctx->a_desc = &a_dyn->base;
    }
    const int c_stride_gen = p.stride;

    static const char* const b_coords[4] = {"k_out", "r", "s", "c"};
    const bool uses_plain_weights = WAVES_K == 1 && !spec->runtime_k_loop;
    rocke_tensor_descriptor_t* b_desc;
    if(RTC && uses_plain_weights)
    {
        b_desc = rocke_dconv_krsc_descriptor_dynamic(b, "B", b_coords, c_cpg, KH, KW);
    }
    else
    {
        int lengths[4];
        lengths[0] = rocke_direct_conv_problem_total_k(&p);
        lengths[1] = KH;
        lengths[2] = KW;
        lengths[3] = p.cpg;
        b_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, b_coords, 4);
    }
    if(b_desc == NULL)
        return NULL;

    rocke_tensor_descriptor_t* d_desc;
    {
        rocke_dynamic_tensor_descriptor_t* d_dyn = rocke_dconv_d_descriptor_dynamic(b, params);
        if(d_dyn == NULL)
            return NULL;
        d_desc = &d_dyn->base;
    }

    /* chunk_desc: chunk_idx -> (W_lds, group_in_wg, ch_block). With runtime
     * channels the decode divides by runtime extents instead. */
    rocke_tensor_descriptor_t* chunk_desc = NULL;
    rocke_value_t* rtc_n_vecs = NULL;
    rocke_value_t* rtc_col_vecs = NULL;
    rocke_value_t* rtc_num_chunks = NULL;
    if(RTC)
    {
        rtc_n_vecs = rocke_b_div(b, c_cpg, rocke_b_const_i32(b, LOAD_VEC));
        rtc_col_vecs = rocke_b_mul(b, c_BG, rtc_n_vecs);
        rocke_value_t* c_ldsw = rocke_b_const_i32(b, LDS_W);
        rtc_num_chunks = rocke_b_mul(b, c_ldsw, rtc_col_vecs);
    }
    else
    {
        static const char* const chunk_coords[3] = {"W_lds", "group_in_wg", "ch_block"};
        int dims[3];
        dims[0] = LDS_W;
        dims[1] = BLOCK_GROUPS;
        dims[2] = N_VECS;
        rocke_tensor_descriptor_t* naive
            = rocke_tensor_descriptor_naive(b, "chunk_unmerge", dims, 3, NULL, chunk_coords, 3);
        const rocke_transform_t* xforms[1];
        xforms[0] = rocke_unmerge_magic(b, "chunk_idx", chunk_coords, 3, dims);
        chunk_desc = rocke_tensor_descriptor_transform(b, naive, xforms, 1);
        if(chunk_desc == NULL)
            return NULL;
    }
    for(int pass_idx = 0; pass_idx < PASSES; ++pass_idx)
    {
        dconv_gen_chunk_t* cm = &ctx->chunk_meta[pass_idx];
        rocke_value_t* chunk_idx = rocke_b_add(b, tid, rocke_b_const_i32(b, pass_idx * THREADS));
        cm->chunk_idx = chunk_idx;
        if(RTC)
        {
            rocke_value_t* rem = rocke_b_mod(b, chunk_idx, rtc_col_vecs);
            cm->W_lds = rocke_b_div(b, chunk_idx, rtc_col_vecs);
            cm->group_in_wg = rocke_b_div(b, rem, rtc_n_vecs);
            cm->ch_block = rocke_b_mod(b, rem, rtc_n_vecs);
            cm->in_bounds = rocke_b_cmp_lt(b, chunk_idx, rtc_num_chunks);
            continue;
        }
        const char* in_names[1] = {"chunk_idx"};
        rocke_value_t* in_values[1];
        const char* out_names[8];
        rocke_value_t* out_values[8];
        in_values[0] = chunk_idx;
        int n_out = rocke_tensor_descriptor_unmerge_lower(
            b, chunk_desc, in_names, in_values, 1, out_names, out_values, 8);
        if(n_out < 0)
            return NULL;
        for(int i = 0; i < n_out; ++i)
        {
            if(strcmp(out_names[i], "ch_block") == 0)
                cm->ch_block = out_values[i];
            else if(strcmp(out_names[i], "W_lds") == 0)
                cm->W_lds = out_values[i];
            else if(strcmp(out_names[i], "group_in_wg") == 0)
                cm->group_in_wg = out_values[i];
        }
        cm->in_bounds = rocke_b_cmp_lt(b, chunk_idx, rocke_b_const_i32(b, NUM_CHUNKS));
    }

    ctx->c0 = c0;
    ctx->c_BG = c_BG;
    ctx->c_cpg = c_cpg;
    ctx->c_half_bytes = c_half_bytes;
    ctx->oob_sentinel = oob_sentinel;
    ctx->by = by;
    ctx->a_rsrc = a_rsrc;
    ctx->n = n;
    ctx->q_tile_start_lds = q_tile_start_lds;

    /* ---- Persistent cell loop ---- */
    rocke_for_t cell_loop;
    rocke_value_t* pg_gt_v = NULL;
    rocke_value_t* pg_in_bounds = NULL;
    rocke_value_t* cell_idx = NULL;
    memset(&cell_loop, 0, sizeof(cell_loop));
    if(PERSISTENT)
    {
        rocke_iter_arg_t carry[1];
        carry[0].name = "pg_cell_idx_carry";
        carry[0].init = xcd_start;
        rocke_value_t* c_rounds = rocke_b_const_i32(b, spec->persistent_rounds);
        cell_loop = rocke_b_scf_for_iter(b, c0, c_rounds, c1, carry, 1, "pg_round", false, false);
        rocke_b_region_enter(b, cell_loop.body);
        rocke_value_t* pg_round_iv = cell_loop.iv;
        {
            rocke_value_t* c_bpx = rocke_b_const_i32(b, DCONV_GEN_BLOCKS_PER_XCD);
            rocke_value_t* m = rocke_b_mul(b, pg_round_iv, c_bpx);
            rocke_value_t* inner = rocke_b_add(b, wg_in_xcd, m);
            cell_idx = rocke_b_add(b, xcd_start, inner);
        }
        pg_in_bounds = rocke_b_cmp_lt(b, cell_idx, n_cells_v);
        rocke_value_t* pg_n_v = rocke_b_div(b, cell_idx, nhqg_v);
        rocke_value_t* pg_rem_n = rocke_b_mod(b, cell_idx, nhqg_v);
        pg_gt_v = rocke_b_div(b, pg_rem_n, nhq_v);
        rocke_value_t* pg_rem_gt = rocke_b_mod(b, pg_rem_n, nhq_v);
        rocke_value_t* pg_ht_v = rocke_b_div(b, pg_rem_gt, q_tiles_p_v);
        rocke_value_t* pg_qt_v = rocke_b_mod(b, pg_rem_gt, q_tiles_p_v);

        n = rocke_b_select(b, pg_in_bounds, pg_n_v, c0);
        {
            rocke_value_t* c_bh = rocke_b_const_i32(b, BLOCK_H);
            rocke_value_t* m = rocke_b_mul(b, pg_ht_v, c_bh);
            h_tile_start = rocke_b_select(b, pg_in_bounds, m, c0);
        }
        {
            rocke_value_t* m = rocke_b_mul(b, pg_gt_v, c_BG);
            rocke_value_t* a = rocke_b_add(b, m, wave_group_idx);
            g = rocke_b_select(b, pg_in_bounds, a, c0);
        }
        {
            rocke_value_t* m1 = rocke_b_mul(b, pg_qt_v, c_BQ);
            rocke_value_t* c_bqw = rocke_b_const_i32(b, BLOCK_Q_WAVE);
            rocke_value_t* m2 = rocke_b_mul(b, wave_id_q, c_bqw);
            rocke_value_t* a = rocke_b_add(b, m1, m2);
            q_tile_start = rocke_b_select(b, pg_in_bounds, a, c0);
        }
        {
            rocke_value_t* m = rocke_b_mul(b, pg_qt_v, c_BQ);
            q_tile_start_lds = rocke_b_select(b, pg_in_bounds, m, c0);
        }
        ctx->n = n;
        ctx->q_tile_start_lds = q_tile_start_lds;
    }

    /* Row loads in flight: slot y holds the PASSES (vec, lds_idx) of input
     * row y of the tile (slot 0 is the prologue row). */
    const int n_iters = BLOCK_H + KH - 1;
    rocke_value_t** row_vec = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, (size_t)(n_iters + 1) * PASSES * sizeof(rocke_value_t*));
    rocke_value_t** row_idx = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, (size_t)(n_iters + 1) * PASSES * sizeof(rocke_value_t*));
    if(row_vec == NULL || row_idx == NULL)
        return NULL;
#define DCONV_GEN_ROW_VEC(y) (row_vec + (size_t)(y) * PASSES)
#define DCONV_GEN_ROW_IDX(y) (row_idx + (size_t)(y) * PASSES)

    rocke_value_t* load_g_tile = PERSISTENT ? pg_gt_v : NULL;
    rocke_value_t* prologue_y = h_tile_start; /* BLOCK_H > 0 (validated) */
    /* dgrad_weights_lds: the staged weight slice (lds_w) and its geometry. */
    const int WL_GROUP = p.cpg * KH * KW * p.kpg; /* elements per group */
    rocke_value_t* wl_lds = NULL;
    if(spec->dgrad_weights_lds)
    {
        /* Stage the raw weight slice of this workgroup's BLOCK_GROUPS groups:
         * W[(g_tile*BG .. +BG)*cpg, KH, KW, kpg] is one contiguous run of
         * BG * cpg*KH*KW*kpg elements, copied 1:1 with 16-byte loads. */
        const int WL_VECS = BLOCK_GROUPS * WL_GROUP / 8;
        const int WL_PASSES = (WL_VECS + THREADS - 1) / THREADS;
        int wl_shape[2];
        wl_shape[0] = 1;
        wl_shape[1] = WL_PASSES * THREADS * 8;
        wl_lds = rocke_b_smem_alloc(b, ctx->io_type, wl_shape, 2, "lds_w");
        rocke_value_t* wl_base = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK_GROUPS * WL_GROUP));
        rocke_value_t** wl_vec = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)WL_PASSES * sizeof(rocke_value_t*));
        rocke_value_t** wl_ix = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)WL_PASSES * sizeof(rocke_value_t*));
        if(wl_vec == NULL || wl_ix == NULL)
            return NULL;
        for(int pi = 0; pi < WL_PASSES; ++pi)
        {
            rocke_value_t* wl_v = rocke_b_add(b, tid, rocke_b_const_i32(b, pi * THREADS));
            rocke_value_t* wl_off;
            {
                rocke_value_t* m8 = rocke_b_mul(b, wl_v, rocke_b_const_i32(b, 8));
                rocke_value_t* a1 = rocke_b_add(b, wl_base, m8);
                wl_off = rocke_b_mul(b, a1, c_half_bytes);
            }
            if((pi + 1) * THREADS > WL_VECS)
            {
                rocke_value_t* lt = rocke_b_cmp_lt(b, wl_v, rocke_b_const_i32(b, WL_VECS));
                wl_off = rocke_b_select(b, lt, wl_off, oob_sentinel);
            }
            wl_vec[pi] = dconv_gen_buf_load(ctx, b_rsrc, wl_off, 4);
            wl_ix[pi] = rocke_b_mul(b, wl_v, rocke_b_const_i32(b, 8));
        }
        /* The first input row is fetched now but published to LDS only after
         * the transpose reads: lds_w is dead by then, so the smem pool packer
         * overlays it with the input row buffers. */
        if(!dconv_issue_dram_load(
               ctx, prologue_y, load_g_tile, DCONV_GEN_ROW_VEC(0), DCONV_GEN_ROW_IDX(0)))
            return NULL;
        for(int pi = 0; pi < WL_PASSES; ++pi)
        {
            rocke_value_t* ix[2];
            ix[0] = c0;
            ix[1] = wl_ix[pi];
            rocke_b_smem_store_vN(b, wl_lds, ix, 2, wl_vec[pi], 8);
        }
        rocke_b_sync(b);
    }
    else
    {
        if(!dconv_issue_dram_load(
               ctx, prologue_y, load_g_tile, DCONV_GEN_ROW_VEC(0), DCONV_GEN_ROW_IDX(0)))
            return NULL;
        dconv_store_to_lds(ctx, DCONV_GEN_ROW_VEC(0), DCONV_GEN_ROW_IDX(0), A_smem);
        rocke_b_sync(b);
    }

    /* acc_tiles[qt][m][slot] */
    const int n_acc = q_subtiles * N_M_TILES * KH;
    rocke_value_t** acc
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_acc * sizeof(rocke_value_t*));
    if(acc == NULL)
        return NULL;
    for(int i = 0; i < n_acc; ++i)
        acc[i] = zero_acc;
#define DCONV_GEN_ACC(qt, m, slot) acc[((qt) * N_M_TILES + (m)) * KH + (slot)]

    /* ---- Weight preloading (WAVES_K > 1) ---- */
    rocke_value_t** preloaded = NULL;
#define DCONV_GEN_PW(r, s, la, m) preloaded[(((r) * KW + (s)) * N_K_LOCAL + (la)) * N_M_TILES + (m)]
    if(WAVES_K > 1)
    {
        const size_t n_pw = (size_t)KH * KW * N_K_LOCAL * N_M_TILES;
        preloaded = (rocke_value_t**)rocke_arena_calloc(&b->arena, n_pw * sizeof(rocke_value_t*));
        if(preloaded == NULL)
            return NULL;
        rocke_value_t* lane_id_pw = rocke_b_mod(b, tid, rocke_b_const_i32(b, WAVE));
        rocke_value_t* c_block_sz = rocke_b_const_i32(b, WAVE * LOAD_VEC);
        const int pw_blocks_per_group = KH * KW * N_K_ATOMS * N_M_TILES;
        rocke_value_t* pw_g_abs = PERSISTENT ? rocke_b_const_i32(b, 0) : g;
        rocke_value_t* pw_group_base
            = rocke_b_mul(b, pw_g_abs, rocke_b_const_i32(b, pw_blocks_per_group));
        for(int r = 0; r < KH; ++r)
        {
            for(int s = 0; s < KW; ++s)
            {
                for(int la = 0; la < N_K_LOCAL; ++la)
                {
                    rocke_value_t* atom_global
                        = rocke_b_add(b, k_atom_base, rocke_b_const_i32(b, la));
                    rocke_value_t* rs_base_pw;
                    {
                        rocke_value_t* c_nmt = rocke_b_const_i32(b, N_M_TILES);
                        rocke_value_t* m1 = rocke_b_mul(b, atom_global, c_nmt);
                        rocke_value_t* c_rs = rocke_b_const_i32(
                            b, (r * KW * N_K_ATOMS + s * N_K_ATOMS) * N_M_TILES);
                        rocke_value_t* a1 = rocke_b_add(b, m1, c_rs);
                        rs_base_pw = rocke_b_add(b, pw_group_base, a1);
                    }
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        if(m * 16 >= p.kpg)
                        {
                            DCONV_GEN_PW(r, s, la, m) = NULL;
                            continue;
                        }
                        rocke_value_t* block_idx
                            = rocke_b_add(b, rs_base_pw, rocke_b_const_i32(b, m));
                        rocke_value_t* elem_off;
                        {
                            rocke_value_t* m1 = rocke_b_mul(b, block_idx, c_block_sz);
                            rocke_value_t* c_lv = rocke_b_const_i32(b, LOAD_VEC);
                            rocke_value_t* m2 = rocke_b_mul(b, lane_id_pw, c_lv);
                            elem_off = rocke_b_add(b, m1, m2);
                        }
                        rocke_value_t* off_bytes = rocke_b_mul(b, elem_off, c_half_bytes);
                        DCONV_GEN_PW(r, s, la, m)
                            = dconv_gen_buf_load(ctx, b_rsrc, off_bytes, LOAD_VEC / 2);
                    }
                }
            }
        }
    }
    else if(PRELOADS)
    {
        /* ---- Weight preloading (WAVES_K == 1) ----
         * Every (r, s, atom, m) fragment is loaded once here and stays in
         * registers for the whole row loop. dgrad_fused_weights: B is the
         * original dgrad weight W[groups*cpg, KH, KW, kpg] (transposed-problem
         * terms), gathered with flipped taps and k<->c transposed addressing;
         * dgrad_weights_lds builds each fragment with transpose reads of the
         * staged slice instead. */
        const size_t n_pw = (size_t)KH * KW * N_K_LOCAL * N_M_TILES;
        preloaded = (rocke_value_t**)rocke_arena_calloc(&b->arena, n_pw * sizeof(rocke_value_t*));
        if(preloaded == NULL)
            return NULL;
        const int c4_step_pl = K_ATOM_SIZE / 4;
        const int fw_k_stride = KH * KW * p.kpg; /* elements between W rows k, k+1 */
        rocke_value_t* wl_lane_row = NULL;
        rocke_value_t* wl_lane_col = NULL;
        rocke_value_t* wl_grp_off = NULL;
        rocke_value_t* c_wl_row = NULL;
        rocke_value_t* c_fw_k_stride = NULL;
        if(spec->dgrad_weights_lds)
        {
            /* Transpose-read lane address pieces: the ch_off term already
             * carries c4 * KS; add the in-chunk row (lane // 4) % 4 and the
             * 4-column group (lane % 4) * 4. */
            {
                rocke_value_t* d = rocke_b_div(b, lane, rocke_b_const_i32(b, 4));
                wl_lane_row = rocke_b_mod(b, d, rocke_b_const_i32(b, 4));
            }
            {
                rocke_value_t* md = rocke_b_mod(b, lane, rocke_b_const_i32(b, 4));
                wl_lane_col = rocke_b_mul(b, md, rocke_b_const_i32(b, 4));
            }
            wl_grp_off = rocke_b_mul(b, wave_group_idx, rocke_b_const_i32(b, WL_GROUP));
            c_wl_row = rocke_b_const_i32(b, KH * KW * p.kpg);
        }
        else if(spec->dgrad_fused_weights)
            c_fw_k_stride = rocke_b_const_i32(b, fw_k_stride);
        for(int r = 0; r < KH; ++r)
        {
            for(int s = 0; s < KW; ++s)
            {
                for(int la = 0; la < N_K_LOCAL; ++la)
                {
                    rocke_value_t* ch_off_pl;
                    {
                        rocke_value_t* m1
                            = rocke_b_mul(b, k_atom_base, rocke_b_const_i32(b, K_ATOM_SIZE));
                        rocke_value_t* c_la = rocke_b_const_i32(b, la * K_ATOM_SIZE);
                        rocke_value_t* m2 = rocke_b_mul(b, c4, rocke_b_const_i32(b, c4_step_pl));
                        rocke_value_t* a1 = rocke_b_add(b, c_la, m2);
                        ch_off_pl = rocke_b_add(b, m1, a1);
                    }
                    /* Flipped tap (KH-1-r, KW-1-s) column offset of M-tile m. */
                    const int flip_base = ((KH - 1 - r) * KW + (KW - 1 - s)) * p.kpg;
                    for(int m = 0; m < M_LOCAL; ++m)
                    {
                        rocke_value_t* w_frag;
                        if(m * 16 >= p.kpg)
                        {
                            DCONV_GEN_PW(r, s, la, m) = NULL;
                            continue;
                        }
                        if(spec->dgrad_weights_lds)
                        {
                            /* Lane (c4, q) needs W[k0 + c4*KS + j, flip, c=m*16+q];
                             * ds_read_b64_tr_b16 hands lane 16h+4a+b element b of
                             * lane 16h+4j+a's 8-byte read, so lane 16h+4j'+a'
                             * reads row k0+h*KS+j', columns m*16+4a'..+3 of the
                             * staged W slice. */
                            rocke_value_t* wl_row = rocke_b_add(b, ch_off_pl, wl_lane_row);
                            rocke_value_t* wl_idx;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, wl_row, c_wl_row);
                                rocke_value_t* a1 = rocke_b_add(b, wl_grp_off, m1);
                                rocke_value_t* c_f = rocke_b_const_i32(b, flip_base + m * 16);
                                rocke_value_t* a2 = rocke_b_add(b, c_f, wl_lane_col);
                                wl_idx = rocke_b_add(b, a1, a2);
                            }
                            if(wm_base16 != NULL)
                                wl_idx = rocke_b_add(b, wl_idx, wm_base16);
                            {
                                rocke_value_t* ix[2];
                                ix[0] = c0;
                                ix[1] = wl_idx;
                                w_frag = rocke_b_ds_read_tr16_b64(b, wl_lds, ix, 2, ctx->io_type);
                            }
                            for(int rd = 1; rd < LOAD_VEC / 4; ++rd)
                            {
                                rocke_value_t* ix[2];
                                rocke_value_t* c_rd
                                    = rocke_b_const_i32(b, 4 * rd * KH * KW * p.kpg);
                                ix[0] = c0;
                                ix[1] = rocke_b_add(b, wl_idx, c_rd);
                                rocke_value_t* hi
                                    = rocke_b_ds_read_tr16_b64(b, wl_lds, ix, 2, ctx->io_type);
                                w_frag = rocke_b_vec_concat(b, w_frag, hi);
                            }
                            if(mask_k_tail)
                            {
                                rocke_value_t* oob = rocke_b_cmp_ge(
                                    b, ch_off_pl, k_tail_bound(b, RTC, c_cpg, p.cpg));
                                rocke_value_t* zero = rocke_b_zero_vec(b, ctx->io_type, LOAD_VEC);
                                w_frag = rocke_b_select(b, oob, zero, w_frag);
                            }
                        }
                        else if(spec->dgrad_fused_weights)
                        {
                            /* W[(g*cpg + ch_off) row, KH-1-r, KW-1-s, m*16 + q] */
                            rocke_value_t* fw_row;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, g, c_cpg);
                                fw_row = rocke_b_add(b, m1, ch_off_pl);
                            }
                            rocke_value_t* fw_off;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, fw_row, c_fw_k_stride);
                                rocke_value_t* c_f = rocke_b_const_i32(b, flip_base + m * 16);
                                rocke_value_t* a1 = rocke_b_add(b, c_f, q_in_lane);
                                fw_off = rocke_b_add(b, m1, a1);
                            }
                            if(wm_base16 != NULL)
                                fw_off = rocke_b_add(b, fw_off, wm_base16);
                            rocke_value_t* elems[8];
                            for(int e = 0; e < LOAD_VEC; ++e)
                            {
                                rocke_value_t* e_off;
                                {
                                    rocke_value_t* c_e = rocke_b_const_i32(b, e * fw_k_stride);
                                    rocke_value_t* a1 = rocke_b_add(b, fw_off, c_e);
                                    e_off = rocke_b_mul(b, a1, c_half_bytes);
                                }
                                if(mask_k_tail)
                                {
                                    rocke_value_t* a1
                                        = rocke_b_add(b, ch_off_pl, rocke_b_const_i32(b, e));
                                    rocke_value_t* e_ok
                                        = rocke_b_cmp_lt(b, a1, k_tail_bound(b, RTC, c_cpg, p.cpg));
                                    e_off = rocke_b_select(b, e_ok, e_off, oob_sentinel);
                                }
                                elems[e] = ctx->is_bf16
                                               ? rocke_b_buffer_load_bf16(b, b_rsrc, e_off, c0)
                                               : rocke_b_buffer_load_f16(b, b_rsrc, e_off, c0);
                            }
                            w_frag = rocke_b_vec_pack(b, elems, LOAD_VEC, ctx->io_type);
                        }
                        else
                        {
                            rocke_value_t* k_out_pl;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, g, c_kpg);
                                rocke_value_t* c_m = rocke_b_const_i32(b, m * 16);
                                rocke_value_t* a1 = rocke_b_add(b, c_m, q_in_lane);
                                k_out_pl = rocke_b_add(b, m1, a1);
                            }
                            if(wm_base16 != NULL)
                                k_out_pl = rocke_b_add(b, k_out_pl, wm_base16);
                            rocke_value_t* w_off = NULL;
                            rocke_value_t* w_valid = NULL;
                            const char* names[4] = {"k_out", "r", "s", "c"};
                            rocke_value_t* values[4];
                            values[0] = k_out_pl;
                            values[1] = rocke_b_const_i32(b, r);
                            values[2] = rocke_b_const_i32(b, s);
                            values[3] = ch_off_pl;
                            if(!rocke_transforms_descriptor_offset(
                                   b, b_desc, names, values, 4, &w_off, &w_valid))
                                return NULL;
                            rocke_value_t* off_bytes = rocke_b_mul(b, w_off, c_half_bytes);
                            w_frag = dconv_gen_buf_load(ctx, b_rsrc, off_bytes, LOAD_VEC / 2);
                            if(mask_k_tail)
                            {
                                rocke_value_t* oob = rocke_b_cmp_ge(
                                    b, ch_off_pl, k_tail_bound(b, RTC, c_cpg, p.cpg));
                                rocke_value_t* zero = rocke_b_zero_vec(b, ctx->io_type, LOAD_VEC);
                                w_frag = rocke_b_select(b, oob, zero, w_frag);
                            }
                        }
                        DCONV_GEN_PW(r, s, la, m) = w_frag;
                    }
                }
            }
        }
        if(spec->dgrad_weights_lds)
        {
            /* Every wave's transpose reads complete before any wave overwrites
             * the (pool-aliased) bytes with the first input row. */
            rocke_b_sync(b);
            DCONV_GEN_ALLOC_ROW_BUFFERS();
            dconv_store_to_lds(ctx, DCONV_GEN_ROW_VEC(0), DCONV_GEN_ROW_IDX(0), A_smem);
            rocke_b_sync(b);
        }
    }

    const int c4_step = K_ATOM_SIZE / 4;
    const int PF = spec->prefetch_rows > 1 ? spec->prefetch_rows : 1;

    /* stage_out: two LDS output-row tiles [BLOCK_Q, SO_CH + 8]. A finished row
     * is written in the MFMA layout before the row barrier, then read back as
     * 16-byte vectors over the workgroup's contiguous channel span and stored
     * after it; the global offsets come from the output stride kernargs. */
    rocke_value_t* so_bufs[2] = {NULL, NULL};
    rocke_value_t* c_so_stride = NULL;
    int so_n = 0;
    rocke_value_t** so_ok = NULL;
    rocke_value_t** so_gel = NULL;
    rocke_value_t** so_lidx = NULL;
    if(spec->stage_out)
    {
        const int SO_CH = BLOCK_GROUPS * p.kpg; /* output channels per workgroup */
        const int SO_STRIDE = (int)(BLOCK_GROUPS * p.kpg + 8); /* direct_conv_stage_out_stride */
        const int SO_VPC = SO_CH / 8;
        const int SO_VECS = BLOCK_Q * SO_VPC;
        for(int i = 0; i < 2; ++i)
        {
            int so_shape[2];
            so_shape[0] = 1;
            so_shape[1] = BLOCK_Q * SO_STRIDE;
            so_bufs[i] = rocke_b_smem_alloc(
                b, ctx->io_type, so_shape, 2, i == 0 ? "lds_out0" : "lds_out1");
        }
        c_so_stride = rocke_b_const_i32(b, SO_STRIDE);
        rocke_value_t* so_n_base = rocke_b_mul(b, n, params->p_D_stride_n);
        so_n = (SO_VECS + THREADS - 1) / THREADS;
        so_ok
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)so_n * sizeof(rocke_value_t*));
        so_gel
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)so_n * sizeof(rocke_value_t*));
        so_lidx
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)so_n * sizeof(rocke_value_t*));
        if(so_ok == NULL || so_gel == NULL || so_lidx == NULL)
            return NULL;
        for(int pi = 0; pi < so_n; ++pi)
        {
            rocke_value_t* t = rocke_b_add(b, tid, rocke_b_const_i32(b, pi * THREADS));
            rocke_value_t* q = rocke_b_div(b, t, rocke_b_const_i32(b, SO_VPC));
            rocke_value_t* v = rocke_b_mod(b, t, rocke_b_const_i32(b, SO_VPC));
            rocke_value_t* qg = rocke_b_add(b, block_q_start, q);
            rocke_value_t* ok = rocke_b_cmp_lt(b, qg, c_W);
            if((pi + 1) * THREADS > SO_VECS)
            {
                rocke_value_t* lt = rocke_b_cmp_lt(b, t, rocke_b_const_i32(b, SO_VECS));
                ok = rocke_b_land(b, ok, lt);
            }
            {
                rocke_value_t* m1 = rocke_b_mul(b, qg, params->p_D_stride_wo);
                rocke_value_t* a1 = rocke_b_add(b, so_n_base, m1);
                rocke_value_t* m2 = rocke_b_mul(b, by, rocke_b_const_i32(b, SO_CH));
                rocke_value_t* m3 = rocke_b_mul(b, v, rocke_b_const_i32(b, 8));
                rocke_value_t* a2 = rocke_b_add(b, m2, m3);
                so_gel[pi] = rocke_b_add(b, a1, a2);
            }
            {
                rocke_value_t* m1 = rocke_b_mul(b, q, c_so_stride);
                rocke_value_t* m2 = rocke_b_mul(b, v, rocke_b_const_i32(b, 8));
                so_lidx[pi] = rocke_b_add(b, m1, m2);
            }
            so_ok[pi] = ok;
        }
    }

    /* prefetch_rows > 1: rows 1 .. PF-1 are in flight before the loop; row
     * y + PF is issued at row y and committed to LDS at the end of row
     * y + PF - 1. */
    for(int y_l = 1; y_l < (PF < n_iters ? PF : n_iters); ++y_l)
    {
        rocke_value_t* row_y = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, y_l));
        if(!dconv_issue_dram_load(
               ctx, row_y, load_g_tile, DCONV_GEN_ROW_VEC(y_l), DCONV_GEN_ROW_IDX(y_l)))
            return NULL;
    }

    for(int y_local = 0; y_local < n_iters; ++y_local)
    {
        (void)rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, y_local)); /* y */
        const bool even = (y_local % 2 == 0) || !spec->double_buffer;
        rocke_value_t* cur = even ? A_smem : B_smem;
        rocke_value_t* nxt = even ? B_smem : A_smem;

        bool have_next = false;
        if(PF > 1)
        {
            if(y_local + PF < n_iters)
            {
                rocke_value_t* row_y
                    = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, y_local + PF));
                if(!dconv_issue_dram_load(ctx,
                                          row_y,
                                          load_g_tile,
                                          DCONV_GEN_ROW_VEC(y_local + PF),
                                          DCONV_GEN_ROW_IDX(y_local + PF)))
                    return NULL;
            }
            have_next = y_local + 1 < n_iters;
        }
        else if(y_local + 1 < n_iters)
        {
            rocke_value_t* next_y = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, y_local + 1));
            if(!dconv_issue_dram_load(ctx,
                                      next_y,
                                      load_g_tile,
                                      DCONV_GEN_ROW_VEC(y_local + 1),
                                      DCONV_GEN_ROW_IDX(y_local + 1)))
                return NULL;
            have_next = true;
        }

        /* row_guard: input row h_tile_start + y_local - PAD past the image adds
         * nothing, so the row's MFMAs run in a block-uniform one-trip loop
         * carrying the accumulators they touch. Rows y_local <= PAD are never
         * past the image (a tile starts inside it). */
        rocke_for_t guard_loop;
        bool guarded = false;
        int n_guard = 0;
        int* guard_keys = NULL; /* flat acc indices */
        if(spec->row_guard && y_local > p.PAD)
        {
            guard_keys = (int*)rocke_arena_alloc(&b->arena,
                                                 (size_t)q_subtiles * M_LOCAL * KH * sizeof(int));
            if(guard_keys == NULL)
                return NULL;
            for(int qt = 0; qt < q_subtiles; ++qt)
                for(int m = 0; m < M_LOCAL; ++m)
                {
                    if(m * 16 >= p.kpg)
                        continue;
                    for(int slot = 0; slot < KH; ++slot)
                        guard_keys[n_guard++] = (qt * N_M_TILES + m) * KH + slot;
                }
            rocke_value_t* in_row
                = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, y_local - p.PAD));
            rocke_value_t* lt = rocke_b_cmp_lt(b, in_row, p_Hi);
            rocke_value_t* trip = rocke_b_select(b, lt, c1, c0);
            rocke_iter_arg_t* g_args = (rocke_iter_arg_t*)rocke_arena_alloc(
                &b->arena, (size_t)(n_guard > 0 ? n_guard : 1) * sizeof(rocke_iter_arg_t));
            if(g_args == NULL)
                return NULL;
            for(int i = 0; i < n_guard; ++i)
            {
                g_args[i].name = rocke_arena_printf(&b->arena, "rg_y%d_%d", y_local, i);
                g_args[i].init = acc[guard_keys[i]];
            }
            const char* g_iv = rocke_arena_printf(&b->arena, "rg_y%d", y_local);
            guard_loop = rocke_b_scf_for_iter(b, c0, trip, c1, g_args, n_guard, g_iv, false, true);
            rocke_b_region_enter(b, guard_loop.body);
            for(int i = 0; i < n_guard; ++i)
                acc[guard_keys[i]] = guard_loop.iter_vars[i];
            guarded = true;
        }

        for(int qt = 0; qt < q_subtiles; ++qt)
        {
            const int qt_w_base = qt * 16;
            for(int r = 0; r < KH; ++r)
            {
                const int p_idx = dconv_gen_pymod(y_local - r, KH);
                rocke_value_t* r_i = rocke_b_const_i32(b, r);

                if(PRELOADS)
                {
                    for(int s = 0; s < KW; ++s)
                    {
                        rocke_value_t* s_val = rocke_b_const_i32(b, s);
                        for(int la = 0; la < N_K_LOCAL; ++la)
                        {
                            rocke_value_t* ch_off;
                            {
                                rocke_value_t* c_kas = rocke_b_const_i32(b, K_ATOM_SIZE);
                                rocke_value_t* m1 = rocke_b_mul(b, k_atom_base, c_kas);
                                rocke_value_t* c_la = rocke_b_const_i32(b, la * K_ATOM_SIZE);
                                rocke_value_t* a1 = rocke_b_add(b, m1, c_la);
                                rocke_value_t* c_st = rocke_b_const_i32(b, c4_step);
                                rocke_value_t* m2 = rocke_b_mul(b, c4, c_st);
                                ch_off = rocke_b_add(b, a1, m2);
                            }
                            rocke_value_t* w_lds
                                = dconv_gen_w_lds_idx(b, q_in_lane, qt_w_base, c_stride_gen, s_val);
                            if(WAVES_Q > 1)
                            {
                                rocke_value_t* c_bqw = rocke_b_const_i32(b, BLOCK_Q_WAVE);
                                rocke_value_t* m = rocke_b_mul(b, wave_id_q, c_bqw);
                                w_lds = rocke_b_add(b, m, w_lds);
                            }
                            rocke_value_t* lds_idx = dconv_gen_lds_idx(
                                b, w_lds, c_BG_cpg, wave_group_idx, c_cpg, ch_off);
                            rocke_value_t* ld_ix[2];
                            ld_ix[0] = c0;
                            ld_ix[1] = lds_idx;
                            rocke_value_t* x_frag
                                = rocke_b_smem_load_vN(b, cur, ld_ix, 2, ctx->io_type, LOAD_VEC);
                            if(mask_k_tail)
                            {
                                rocke_value_t* oob
                                    = rocke_b_cmp_ge(b, ch_off, k_tail_bound(b, RTC, c_cpg, p.cpg));
                                rocke_value_t* zero = rocke_b_zero_vec(b, ctx->io_type, LOAD_VEC);
                                x_frag = rocke_b_select(b, oob, zero, x_frag);
                            }
                            for(int m = 0; m < M_LOCAL; ++m)
                            {
                                if(m * 16 >= p.kpg)
                                    continue;
                                DCONV_GEN_ACC(qt, m, p_idx)
                                    = dconv_gen_mfma(ctx,
                                                     FOLD_K32,
                                                     DCONV_GEN_PW(r, s, la, m),
                                                     x_frag,
                                                     DCONV_GEN_ACC(qt, m, p_idx));
                            }
                        }
                    }
                    continue;
                }

                if(spec->runtime_k_loop)
                {
                    rocke_iter_arg_t* iter_args = (rocke_iter_arg_t*)rocke_arena_alloc(
                        &b->arena, (size_t)N_M_TILES * sizeof(rocke_iter_arg_t));
                    rocke_value_t** new_accs = (rocke_value_t**)rocke_arena_alloc(
                        &b->arena, (size_t)N_M_TILES * sizeof(rocke_value_t*));
                    if(iter_args == NULL || new_accs == NULL)
                        return NULL;
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        iter_args[m].name = rocke_arena_printf(
                            &b->arena, "rk_acc_m%d_y%d_qt%d_r%d", m, y_local, qt, r);
                        iter_args[m].init = DCONV_GEN_ACC(qt, m, p_idx);
                    }
                    const char* iv_name
                        = rocke_arena_printf(&b->arena, "rk_iv_y%d_qt%d_r%d", y_local, qt, r);
                    rocke_for_t k_loop = rocke_b_scf_for_iter(
                        b, c0, c_N_K_LOCAL, c1, iter_args, N_M_TILES, iv_name, false, false);
                    rocke_b_region_enter(b, k_loop.body);
                    {
                        rocke_value_t* k_atom_global = rocke_b_add(b, k_atom_base, k_loop.iv);
                        rocke_value_t* ch_off_rk;
                        {
                            rocke_value_t* c_kas = rocke_b_const_i32(b, K_ATOM_SIZE);
                            rocke_value_t* m1 = rocke_b_mul(b, k_atom_global, c_kas);
                            rocke_value_t* c_st = rocke_b_const_i32(b, c4_step);
                            rocke_value_t* m2 = rocke_b_mul(b, c4, c_st);
                            ch_off_rk = rocke_b_add(b, m1, m2);
                        }
                        for(int m = 0; m < N_M_TILES; ++m)
                            new_accs[m] = k_loop.iter_vars[m];
                        for(int s = 0; s < KW; ++s)
                        {
                            rocke_value_t* s_val_rk = rocke_b_const_i32(b, s);
                            rocke_value_t* w_lds = dconv_gen_w_lds_idx(
                                b, q_in_lane, qt_w_base, c_stride_gen, s_val_rk);
                            if(WAVES_Q > 1)
                            {
                                rocke_value_t* c_bqw = rocke_b_const_i32(b, BLOCK_Q_WAVE);
                                rocke_value_t* m = rocke_b_mul(b, wave_id_q, c_bqw);
                                w_lds = rocke_b_add(b, m, w_lds);
                            }
                            rocke_value_t* lds_idx = dconv_gen_lds_idx(
                                b, w_lds, c_BG_cpg, wave_group_idx, c_cpg, ch_off_rk);
                            rocke_value_t* ld_ix[2];
                            ld_ix[0] = c0;
                            ld_ix[1] = lds_idx;
                            rocke_value_t* x_frag
                                = rocke_b_smem_load_vN(b, cur, ld_ix, 2, ctx->io_type, LOAD_VEC);
                            if(mask_k_tail)
                            {
                                rocke_value_t* oob = rocke_b_cmp_ge(
                                    b, ch_off_rk, k_tail_bound(b, RTC, c_cpg, p.cpg));
                                rocke_value_t* zero = rocke_b_zero_vec(b, ctx->io_type, LOAD_VEC);
                                x_frag = rocke_b_select(b, oob, zero, x_frag);
                            }
                            for(int m = 0; m < N_M_TILES; ++m)
                            {
                                if(m * 16 >= p.kpg)
                                    continue;
                                const int rs_const_base
                                    = (r * KW * N_K_ATOMS + s * N_K_ATOMS) * N_M_TILES;
                                const int rk_blocks_per_group = KH * KW * N_K_ATOMS * N_M_TILES;
                                rocke_value_t* rk_g_abs = PERSISTENT ? rocke_b_const_i32(b, 0) : g;
                                rocke_value_t* rk_group_base = rocke_b_mul(
                                    b, rk_g_abs, rocke_b_const_i32(b, rk_blocks_per_group));
                                rocke_value_t* block_idx_rk;
                                {
                                    rocke_value_t* c_rs = rocke_b_const_i32(b, rs_const_base + m);
                                    rocke_value_t* c_nmt = rocke_b_const_i32(b, N_M_TILES);
                                    rocke_value_t* m1 = rocke_b_mul(b, k_atom_global, c_nmt);
                                    rocke_value_t* a1 = rocke_b_add(b, c_rs, m1);
                                    block_idx_rk = rocke_b_add(b, rk_group_base, a1);
                                }
                                rocke_value_t* elem_off_rk;
                                {
                                    rocke_value_t* c_bs = rocke_b_const_i32(b, WAVE * LOAD_VEC);
                                    rocke_value_t* m1 = rocke_b_mul(b, block_idx_rk, c_bs);
                                    rocke_value_t* c_w = rocke_b_const_i32(b, WAVE);
                                    rocke_value_t* md = rocke_b_mod(b, tid, c_w);
                                    rocke_value_t* c_lv = rocke_b_const_i32(b, LOAD_VEC);
                                    rocke_value_t* m2 = rocke_b_mul(b, md, c_lv);
                                    elem_off_rk = rocke_b_add(b, m1, m2);
                                }
                                rocke_value_t* off_bytes
                                    = rocke_b_mul(b, elem_off_rk, c_half_bytes);
                                rocke_value_t* w_frag
                                    = dconv_gen_buf_load(ctx, b_rsrc, off_bytes, LOAD_VEC / 2);
                                new_accs[m]
                                    = dconv_gen_mfma(ctx, FOLD_K32, w_frag, x_frag, new_accs[m]);
                            }
                        }
                        rocke_b_scf_yield(b, new_accs, N_M_TILES);
                    }
                    rocke_b_region_leave(b);
                    for(int m = 0; m < N_M_TILES; ++m)
                        DCONV_GEN_ACC(qt, m, p_idx) = k_loop.op->results[m];
                    continue;
                }

                /* ---- plain-weight runtime loops (s, then K-atom) ---- */
                rocke_iter_arg_t* s_args = (rocke_iter_arg_t*)rocke_arena_alloc(
                    &b->arena, (size_t)N_M_TILES * sizeof(rocke_iter_arg_t));
                rocke_iter_arg_t* a_args = (rocke_iter_arg_t*)rocke_arena_alloc(
                    &b->arena, (size_t)N_M_TILES * sizeof(rocke_iter_arg_t));
                rocke_value_t** new_accs = (rocke_value_t**)rocke_arena_alloc(
                    &b->arena, (size_t)N_M_TILES * sizeof(rocke_value_t*));
                if(s_args == NULL || a_args == NULL || new_accs == NULL)
                    return NULL;
                for(int m = 0; m < N_M_TILES; ++m)
                {
                    s_args[m].name = rocke_arena_printf(
                        &b->arena, "siv_acc_m%d_y%d_qt%d_r%d", m, y_local, qt, r);
                    s_args[m].init = DCONV_GEN_ACC(qt, m, p_idx);
                }
                const char* s_iv_name
                    = rocke_arena_printf(&b->arena, "s_iv_y%d_qt%d_r%d", y_local, qt, r);
                rocke_for_t s_loop = rocke_b_scf_for_iter(
                    b, c0, c_KW, c1, s_args, N_M_TILES, s_iv_name, false, false);
                rocke_b_region_enter(b, s_loop.body);
                {
                    rocke_value_t* s_iv = s_loop.iv;
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        a_args[m].name = rocke_arena_printf(
                            &b->arena, "katom_acc_m%d_y%d_qt%d_r%d", m, y_local, qt, r);
                        a_args[m].init = s_loop.iter_vars[m];
                    }
                    const char* a_iv_name
                        = rocke_arena_printf(&b->arena, "atom_iv_y%d_qt%d_r%d", y_local, qt, r);
                    rocke_for_t atom_loop = rocke_b_scf_for_iter(
                        b, c0, c_N_K_LOCAL, c1, a_args, N_M_TILES, a_iv_name, false, false);
                    rocke_b_region_enter(b, atom_loop.body);
                    {
                        rocke_value_t* atom_iv_global = rocke_b_add(b, k_atom_base, atom_loop.iv);
                        rocke_value_t* ch_off;
                        {
                            rocke_value_t* c_kas = rocke_b_const_i32(b, K_ATOM_SIZE);
                            rocke_value_t* m1 = rocke_b_mul(b, atom_iv_global, c_kas);
                            rocke_value_t* c_st = rocke_b_const_i32(b, c4_step);
                            rocke_value_t* m2 = rocke_b_mul(b, c4, c_st);
                            ch_off = rocke_b_add(b, m1, m2);
                        }
                        rocke_value_t* w_lds
                            = dconv_gen_w_lds_idx(b, q_in_lane, qt_w_base, c_stride_gen, s_iv);
                        if(WAVES_Q > 1)
                        {
                            rocke_value_t* c_bqw = rocke_b_const_i32(b, BLOCK_Q_WAVE);
                            rocke_value_t* base = rocke_b_mul(b, wave_id_q, c_bqw);
                            w_lds = rocke_b_add(b, base, w_lds);
                        }
                        rocke_value_t* lds_idx
                            = dconv_gen_lds_idx(b, w_lds, c_BG_cpg, wave_group_idx, c_cpg, ch_off);
                        rocke_value_t* ld_ix[2];
                        ld_ix[0] = c0;
                        ld_ix[1] = lds_idx;
                        rocke_value_t* x_frag
                            = rocke_b_smem_load_vN(b, cur, ld_ix, 2, ctx->io_type, LOAD_VEC);
                        rocke_value_t* oob = NULL;
                        rocke_value_t* zero_std = NULL;
                        if(mask_k_tail)
                        {
                            oob = rocke_b_cmp_ge(b, ch_off, k_tail_bound(b, RTC, c_cpg, p.cpg));
                            zero_std = rocke_b_zero_vec(b, ctx->io_type, LOAD_VEC);
                            x_frag = rocke_b_select(b, oob, zero_std, x_frag);
                        }
                        for(int m = 0; m < N_M_TILES; ++m)
                        {
                            rocke_value_t* k_out_m;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, g, c_kpg);
                                rocke_value_t* c_m = rocke_b_const_i32(b, m * 16);
                                rocke_value_t* a1 = rocke_b_add(b, c_m, q_in_lane);
                                k_out_m = rocke_b_add(b, m1, a1);
                            }
                            rocke_value_t* w_off = NULL;
                            rocke_value_t* w_valid = NULL;
                            const char* names[4] = {"k_out", "r", "s", "c"};
                            rocke_value_t* values[4];
                            values[0] = k_out_m;
                            values[1] = r_i;
                            values[2] = s_iv;
                            values[3] = ch_off;
                            if(!rocke_transforms_descriptor_offset(
                                   b, b_desc, names, values, 4, &w_off, &w_valid))
                                return NULL;
                            rocke_value_t* off_bytes = rocke_b_mul(b, w_off, c_half_bytes);
                            rocke_value_t* w_frag
                                = dconv_gen_buf_load(ctx, b_rsrc, off_bytes, LOAD_VEC / 2);
                            if(mask_k_tail)
                                w_frag = rocke_b_select(b, oob, zero_std, w_frag);
                            new_accs[m] = dconv_gen_mfma(
                                ctx, FOLD_K32, w_frag, x_frag, atom_loop.iter_vars[m]);
                        }
                        rocke_b_scf_yield(b, new_accs, N_M_TILES);
                    }
                    rocke_b_region_leave(b);
                    rocke_b_scf_yield(b, atom_loop.op->results, N_M_TILES);
                }
                rocke_b_region_leave(b);
                for(int m = 0; m < N_M_TILES; ++m)
                    DCONV_GEN_ACC(qt, m, p_idx) = s_loop.op->results[m];
            }
        }

        if(guarded)
        {
            rocke_value_t** yv = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, (size_t)(n_guard > 0 ? n_guard : 1) * sizeof(rocke_value_t*));
            if(yv == NULL)
                return NULL;
            for(int i = 0; i < n_guard; ++i)
                yv[i] = acc[guard_keys[i]];
            rocke_b_scf_yield(b, yv, n_guard);
            rocke_b_region_leave(b);
            for(int i = 0; i < n_guard; ++i)
                acc[guard_keys[i]] = guard_loop.op->results[i];
        }

        if(have_next)
        {
            if(!spec->double_buffer)
                rocke_b_sync(b);
            dconv_store_to_lds(
                ctx, DCONV_GEN_ROW_VEC(y_local + 1), DCONV_GEN_ROW_IDX(y_local + 1), nxt);
        }
        /* Tile-local index of the output row that completes at this y_local. */
        const int so_row = y_local - (KH - 1);
        if(spec->stage_out && so_row >= 0 && so_row < BLOCK_H)
        {
            /* Publish the finished row to LDS before the row barrier (rows of
             * a tile past the image are published too; the flush guard masks
             * their global stores). */
            rocke_value_t* so_buf = so_bufs[y_local % 2];
            const int so_slot = so_row % KH;
            for(int qt = 0; qt < q_subtiles; ++qt)
            {
                rocke_value_t* col = rocke_b_add(b, q_in_lane, rocke_b_const_i32(b, qt * 16));
                for(int m = 0; m < M_LOCAL; ++m)
                {
                    rocke_value_t* ch;
                    {
                        rocke_value_t* m1 = rocke_b_mul(b, wave_group_idx, c_kpg);
                        rocke_value_t* c_m = rocke_b_const_i32(b, m * 16);
                        rocke_value_t* m2 = rocke_b_mul(b, c4, rocke_b_const_i32(b, 4));
                        rocke_value_t* a1 = rocke_b_add(b, c_m, m2);
                        ch = rocke_b_add(b, m1, a1);
                    }
                    if(wm_base16 != NULL)
                        ch = rocke_b_add(b, ch, wm_base16);
                    rocke_value_t* ix[2];
                    ix[0] = c0;
                    {
                        rocke_value_t* m1 = rocke_b_mul(b, col, c_so_stride);
                        ix[1] = rocke_b_add(b, m1, ch);
                    }
                    rocke_value_t* a = DCONV_GEN_ACC(qt, m, so_slot);
                    rocke_value_t* h = ctx->is_bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, a)
                                                    : rocke_b_vec_trunc_f32_to_f16(b, a);
                    rocke_b_smem_store_vN(b, so_buf, ix, 2, h, 4);
                }
            }
        }
        if(spec->lds_only_sync)
            rocke_b_sync_lds_only(b);
        else
            rocke_b_sync(b);

        const int p_flush_local = y_local - (KH - 1);
        const int P_FLUSH = dconv_gen_pymod(p_flush_local, KH);
        if(p_flush_local >= 0)
        {
            rocke_value_t* p_flush_global
                = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, p_flush_local));
            rocke_value_t* in_h_range;
            {
                rocke_value_t* ge = rocke_b_cmp_ge(b, p_flush_global, c0);
                rocke_value_t* lt = rocke_b_cmp_lt(b, p_flush_global, p_Hi);
                in_h_range = rocke_b_land(b, ge, lt);
            }
            if(PERSISTENT)
                in_h_range = rocke_b_land(b, in_h_range, pg_in_bounds);
            if(c_stride_gen > 1)
            {
                rocke_value_t* md
                    = rocke_b_mod(b, p_flush_global, rocke_b_const_i32(b, c_stride_gen));
                rocke_value_t* stride_ok = rocke_b_cmp_eq(b, md, c0);
                in_h_range = rocke_b_land(b, in_h_range, stride_ok);
            }
            rocke_value_t* ho_row_val
                = rocke_b_div(b, p_flush_global, rocke_b_const_i32(b, c_stride_gen));

            if(spec->stage_out)
            {
                rocke_value_t* so_buf = so_bufs[y_local % 2];
                rocke_value_t* row_off = rocke_b_mul(b, ho_row_val, params->p_D_stride_ho);
                for(int pi = 0; pi < so_n; ++pi)
                {
                    rocke_value_t* ok = rocke_b_land(b, so_ok[pi], in_h_range);
                    rocke_value_t* ix[2];
                    ix[0] = c0;
                    ix[1] = so_lidx[pi];
                    rocke_value_t* vec = rocke_b_smem_load_vN(b, so_buf, ix, 2, ctx->io_type, 8);
                    rocke_value_t* off;
                    {
                        rocke_value_t* a1 = rocke_b_add(b, so_gel[pi], row_off);
                        rocke_value_t* m1 = rocke_b_mul(b, a1, c_half_bytes);
                        off = rocke_b_select(b, ok, m1, oob_sentinel);
                    }
                    dconv_gen_buf_store(ctx, d_rsrc, off, vec, 4);
                }
            }
            else
            {
                for(int qt = 0; qt < q_subtiles; ++qt)
                {
                    const int qt_w_base = qt * 16;
                    rocke_value_t* out_q;
                    {
                        rocke_value_t* c_b = rocke_b_const_i32(b, qt_w_base);
                        rocke_value_t* a1 = rocke_b_add(b, q_tile_start, c_b);
                        out_q = rocke_b_add(b, a1, q_in_lane);
                    }
                    rocke_value_t* out_q_ok = rocke_b_cmp_lt(b, out_q, c_W);

                    for(int m = 0; m < M_LOCAL; ++m)
                    {
                        if(m * 16 >= p.kpg)
                            continue;
                        rocke_value_t* acc_to_flush = DCONV_GEN_ACC(qt, m, P_FLUSH);
                        rocke_value_t* k_val;
                        {
                            rocke_value_t* m1 = rocke_b_mul(b, g, c_kpg);
                            rocke_value_t* c_m = rocke_b_const_i32(b, m * 16);
                            rocke_value_t* c_4 = rocke_b_const_i32(b, 4);
                            rocke_value_t* m2 = rocke_b_mul(b, c4, c_4);
                            rocke_value_t* a1 = rocke_b_add(b, c_m, m2);
                            k_val = rocke_b_add(b, m1, a1);
                        }
                        if(wm_base16 != NULL)
                            k_val = rocke_b_add(b, k_val, wm_base16);
                        rocke_value_t* store_ok;
                        if(RTC)
                        {
                            if(m == N_M_TILES - 1)
                            {
                                rocke_value_t* c_4 = rocke_b_const_i32(b, 4);
                                rocke_value_t* rows = rocke_b_mul(b, c4, c_4);
                                rocke_value_t* c_m = rocke_b_const_i32(b, m * 16);
                                rocke_value_t* left = rocke_b_sub(b, c_kpg, c_m);
                                rocke_value_t* c4_ok = rocke_b_cmp_lt(b, rows, left);
                                store_ok = rocke_b_land(b, out_q_ok, c4_ok);
                            }
                            else
                                store_ok = out_q_ok;
                        }
                        else
                        {
                            const int rows_in_tile = p.kpg - m * 16;
                            if(rows_in_tile < 16)
                            {
                                rocke_value_t* c_4 = rocke_b_const_i32(b, 4);
                                rocke_value_t* rows = rocke_b_mul(b, c4, c_4);
                                rocke_value_t* c_rows = rocke_b_const_i32(b, rows_in_tile);
                                rocke_value_t* c4_ok = rocke_b_cmp_lt(b, rows, c_rows);
                                store_ok = rocke_b_land(b, out_q_ok, c4_ok);
                            }
                            else
                                store_ok = out_q_ok;
                        }
                        store_ok = rocke_b_land(b, store_ok, in_h_range);

                        rocke_value_t* d_base = NULL;
                        rocke_value_t* d_valid = NULL;
                        {
                            const char* names[4] = {"n", "h", "w", "k"};
                            rocke_value_t* values[4];
                            values[0] = n;
                            values[1] = ho_row_val;
                            values[2] = out_q;
                            values[3] = k_val;
                            if(!rocke_transforms_descriptor_offset(
                                   b, d_desc, names, values, 4, &d_base, &d_valid))
                                return NULL;
                        }

                        if(WAVES_K > 1)
                        {
                            rocke_value_t* lane_col = rocke_b_mul(b, lane, rocke_b_const_i32(b, 4));
                            {
                                rocke_value_t* ix[2];
                                ix[0] = lds_row_idx;
                                ix[1] = lane_col;
                                rocke_b_smem_store_vN_f32(b, red_lds, ix, 2, acc_to_flush, 4);
                            }
                            rocke_b_sync(b);
                            rocke_value_t* is_k0 = rocke_b_cmp_eq(b, wave_id_k, c0);
                            rocke_if_t if_k0 = rocke_b_scf_if(b, is_k0);
                            rocke_b_region_enter(b, if_k0.then_region);
                            {
                                rocke_value_t* row_base
                                    = rocke_b_mul(b, wave_id_q, rocke_b_const_i32(b, WAVES_K));
                                rocke_value_t** rows_f32 = (rocke_value_t**)rocke_arena_alloc(
                                    &b->arena, (size_t)WAVES_K * sizeof(rocke_value_t*));
                                if(rows_f32 == NULL)
                                    return NULL;
                                for(int wk = 0; wk < WAVES_K; ++wk)
                                {
                                    rocke_value_t* ix[2];
                                    ix[0] = rocke_b_add(b, row_base, rocke_b_const_i32(b, wk));
                                    ix[1] = lane_col;
                                    rows_f32[wk] = rocke_b_smem_load_vN_f32(b, red_lds, ix, 2, 4);
                                }
                                rocke_value_t* sum_slots[4];
                                for(int slot = 0; slot < 4; ++slot)
                                {
                                    rocke_value_t* sv = rocke_b_vec_extract(b, rows_f32[0], slot);
                                    for(int wk = 1; wk < WAVES_K; ++wk)
                                    {
                                        rocke_value_t* e
                                            = rocke_b_vec_extract(b, rows_f32[wk], slot);
                                        sv = rocke_b_fadd(b, sv, e);
                                    }
                                    sum_slots[slot] = sv;
                                }
                                rocke_value_t* partial
                                    = rocke_b_vec_pack(b, sum_slots, 4, rocke_f32());
                                rocke_value_t* off_bytes = rocke_b_mul(b, d_base, c_half_bytes);
                                rocke_value_t* safe_d
                                    = rocke_b_select(b, store_ok, off_bytes, oob_sentinel);
                                rocke_value_t* acc_h
                                    = ctx->is_bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, partial)
                                                   : rocke_b_vec_trunc_f32_to_f16(b, partial);
                                dconv_gen_buf_store(ctx, d_rsrc, safe_d, acc_h, 2);
                            }
                            rocke_b_region_leave(b);
                            rocke_b_sync(b);
                        }
                        else
                        {
                            rocke_value_t* off_bytes = rocke_b_mul(b, d_base, c_half_bytes);
                            rocke_value_t* safe_d
                                = rocke_b_select(b, store_ok, off_bytes, oob_sentinel);
                            rocke_value_t* acc_h
                                = ctx->is_bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, acc_to_flush)
                                               : rocke_b_vec_trunc_f32_to_f16(b, acc_to_flush);
                            dconv_gen_buf_store(ctx, d_rsrc, safe_d, acc_h, 2);
                        }
                    }
                }
            }
        }

        for(int qt = 0; qt < q_subtiles; ++qt)
            for(int m = 0; m < M_LOCAL; ++m)
                DCONV_GEN_ACC(qt, m, P_FLUSH) = zero_acc;

        if(spec->row_exit && KH - 1 <= y_local && y_local < n_iters - 1)
        {
            /* The next flush is past the image: every remaining row of the
             * tile is dead, so the workgroup leaves. The condition is
             * block-uniform, so no wave is left waiting at a barrier. */
            rocke_value_t* next_flush
                = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, y_local + 2 - KH));
            rocke_value_t* past = rocke_b_cmp_ge(b, next_flush, p_Hi);
            rocke_if_t if_exit = rocke_b_scf_if(b, past);
            rocke_b_region_enter(b, if_exit.then_region);
            rocke_b_ret(b);
            rocke_b_region_leave(b);
        }
    }

    if(PERSISTENT)
    {
        rocke_value_t* yv[1];
        yv[0] = cell_idx;
        rocke_b_scf_yield(b, yv, 1);
        rocke_b_region_leave(b);
    }
#undef DCONV_GEN_ACC
#undef DCONV_GEN_PW
#undef DCONV_GEN_ROW_VEC
#undef DCONV_GEN_ROW_IDX
#undef DCONV_GEN_ALLOC_ROW_BUFFERS

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}
