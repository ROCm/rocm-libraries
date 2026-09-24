// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_wgrad.cpp -- C99 port of build_direct_conv_wgrad
 * (rocke/instances/common/conv_direct_grouped.py).
 *
 * Computes dW[k, r, s, c] = sum_{n,ho,wo} dY[n,ho,wo,k] * X[n,hi,wi,c] with
 * hi = ho*stride + r - PAD and wi = wo*stride + s - PAD, at stride 1.
 *
 * ALGORITHM (delta register ring + S-row strip). The outer loop walks INPUT
 * rows hi. Per row:
 *   - one dY row lands in the KH-slot register ring (reused KH times),
 *   - one X strip of STRIP_COLS columns lands in LDS and serves all KW s-taps
 *     as a row shift, so the block owns one wo_tile and there is no wo loop,
 *   - KH*KW MFMAs accumulate into KH*KW <4 x float> accumulators.
 * Both LDS tiles are stored spatial-major exactly as NHWC delivers them and are
 * read back through ds_read_b64_tr_b16, which hands the MFMA its per-lane
 * operand for free. The epilogue atomically adds the accumulators into an fp32
 * dW, which the caller must zero before launch.
 *
 * The Python body is one function with closures over a wide prologue; here the
 * closures are static helpers over rocke_dconv_wgrad_ctx_t and the phases are
 * the four functions declared in the internal header. Every helper emits in
 * PYTHON EVALUATION ORDER (arguments left-to-right, innermost call first) so
 * the SSA sequence -- and therefore the .ll bytes -- match exactly.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  Closures (Python nested defs) -- readers
 * ===================================================================== */

/* _tr_read(smem, part_off, row_shift): N_TR_READS transpose reads concatenated
 * into one <VEC_CH x half> MFMA operand fragment. */
static rocke_value_t* rocke_dconv_wgrad__tr_read(rocke_dconv_wgrad_ctx_t* ctx,
                                                 rocke_value_t* smem,
                                                 rocke_value_t* part_off,
                                                 int row_shift)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* base;
    rocke_value_t* frag;
    rocke_value_t* indices[2];
    int rd;

    /* base = part_off + (tr_flat + row_shift*TR_N) */
    {
        rocke_value_t* c_shift = rocke_b_const_i32(b, row_shift * ctx->TR_N);
        rocke_value_t* inner = rocke_b_add(b, ctx->tr_flat, c_shift);
        base = rocke_b_add(b, part_off, inner);
    }

    indices[0] = ctx->c0;
    indices[1] = base;
    frag = rocke_b_ds_read_tr16_b64(b, smem, indices, 2, rocke_f16());

    for(rd = 1; rd < ctx->N_TR_READS; ++rd)
    {
        rocke_value_t* c_step = rocke_b_const_i32(b, 4 * rd * ctx->TR_N);
        rocke_value_t* addr = rocke_b_add(b, base, c_step);
        rocke_value_t* nxt;
        rocke_value_t* idx2[2];
        idx2[0] = ctx->c0;
        idx2[1] = addr;
        nxt = rocke_b_ds_read_tr16_b64(b, smem, idx2, 2, rocke_f16());
        frag = rocke_b_vec_concat(b, frag, nxt);
    }
    return frag;
}

/* _read_dy(): dY fragment -- transpose read of dy_lds[sp][k_ch]. */
static rocke_value_t* rocke_dconv_wgrad__read_dy(rocke_dconv_wgrad_ctx_t* ctx)
{
    return rocke_dconv_wgrad__tr_read(ctx, ctx->dy_lds, ctx->dy_wave_off_f16, 0);
}

/* _read_strip(s): X fragment for filter column s -- the strip shifted s rows. */
static rocke_value_t* rocke_dconv_wgrad__read_strip(rocke_dconv_wgrad_ctx_t* ctx, int s_const)
{
    return rocke_dconv_wgrad__tr_read(ctx, ctx->s_strip_lds, ctx->s_strip_off_f16, s_const);
}

/* ===================================================================== *
 *  Closures -- loaders (ISSUE: DRAM -> VGPR, COMMIT: VGPR -> LDS)
 * ===================================================================== */

/* _lds_run(part_off, row): flat f16 index of this lane's VEC_CH-wide run. */
static rocke_value_t* rocke_dconv_wgrad__lds_run(rocke_dconv_wgrad_ctx_t* ctx,
                                                 rocke_value_t* part_off,
                                                 rocke_value_t* row)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* mul_row = rocke_b_mul(b, row, ctx->c_TR_N);
    rocke_value_t* inner = rocke_b_add(b, mul_row, ctx->c_ld_ch);
    return rocke_b_add(b, part_off, inner);
}

/* _issue_delta(ho_val, ho_ok): start the DRAM read of one dY row.
 *
 * No OOB masking of the loaded value: an out-of-range lane gets oob_sentinel as
 * its buffer offset and a buffer load past num_records returns zero. */
static rocke_value_t* rocke_dconv_wgrad__issue_delta(rocke_dconv_wgrad_ctx_t* ctx,
                                                     rocke_value_t* ho_val,
                                                     rocke_value_t* ho_ok)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* wo_sp;
    rocke_value_t* both_ok;
    rocke_value_t* k_ld;
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_value_t* safe_off;
    const char* names[4];
    rocke_value_t* values[4];

    wo_sp = rocke_b_add(b, ctx->wo_tile_start, ctx->c_ld_sp);
    both_ok = rocke_b_land(b, ho_ok, rocke_b_cmp_lt(b, wo_sp, ctx->c_Wo));
    k_ld = rocke_b_add(b, ctx->k_wave_base, ctx->c_ld_ch);

    names[0] = "n";
    values[0] = ctx->n_i;
    names[1] = "h";
    values[1] = ho_val;
    names[2] = "w";
    values[2] = wo_sp;
    names[3] = "k";
    values[3] = k_ld;
    if(!rocke_transforms_descriptor_offset(b, ctx->dy_desc, names, values, 4, &off, &valid))
    {
        return NULL;
    }

    safe_off
        = rocke_b_select(b, both_ok, rocke_b_mul(b, off, ctx->c_half_bytes), ctx->oob_sentinel);
    return rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_off, ctx->c0, ctx->VEC_CH / 2);
}

/* _commit_delta(vec): land a dY fragment in dy_lds[sp][k_ch]. */
static void rocke_dconv_wgrad__commit_delta(rocke_dconv_wgrad_ctx_t* ctx, rocke_value_t* vec)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* indices[2];
    indices[0] = ctx->c0;
    indices[1] = rocke_dconv_wgrad__lds_run(ctx, ctx->dy_wave_off_f16, ctx->c_ld_sp);
    rocke_b_smem_store_vN_f16(b, ctx->dy_lds, indices, 2, vec, ctx->VEC_CH);
}

/* _issue_s_strip(hi_val, hi_ok): start the DRAM reads of one X strip.
 * Writes STRIP_PASSES_PER_WAVE in-flight fragments into out[]. */
static bool rocke_dconv_wgrad__issue_s_strip(rocke_dconv_wgrad_ctx_t* ctx,
                                             rocke_value_t* hi_val,
                                             rocke_value_t* hi_ok,
                                             rocke_value_t** out)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c_ld_base;
    int pass_idx;

    /* c_ld_base = (group*cpg + c_tile_origin) + (wave_c_origin + c_ld_ch) */
    {
        rocke_value_t* mul_g = rocke_b_mul(b, ctx->group, ctx->c_cpg);
        rocke_value_t* lhs = rocke_b_add(b, mul_g, ctx->c_tile_origin);
        rocke_value_t* rhs = rocke_b_add(b, ctx->wave_c_origin, ctx->c_ld_ch);
        c_ld_base = rocke_b_add(b, lhs, rhs);
    }

    for(pass_idx = 0; pass_idx < ctx->STRIP_PASSES_PER_WAVE; ++pass_idx)
    {
        rocke_value_t* off = NULL;
        rocke_value_t* x_ok = NULL;
        rocke_value_t* both_ok;
        rocke_value_t* safe;
        const char* names[5];
        rocke_value_t* values[5];

        names[0] = "n";
        values[0] = ctx->n_i;
        names[1] = "h";
        values[1] = hi_val;
        names[2] = "wo";
        values[2] = ctx->wo_tile_start;
        names[3] = "s_off";
        values[3] = ctx->strip_cols[pass_idx];
        names[4] = "c";
        values[4] = c_ld_base;
        if(!rocke_transforms_descriptor_offset(b, ctx->x_strip_desc, names, values, 5, &off, &x_ok))
        {
            return false;
        }

        both_ok = rocke_b_land(b, rocke_b_land(b, hi_ok, ctx->strip_col_ok[pass_idx]), x_ok);
        safe
            = rocke_b_select(b, both_ok, rocke_b_mul(b, off, ctx->c_half_bytes), ctx->oob_sentinel);
        out[pass_idx] = rocke_b_buffer_load_vN_f16(b, ctx->b_rsrc, safe, ctx->c0, ctx->VEC_CH / 2);
    }
    return true;
}

/* _commit_s_strip(vecs): land the X strip fragments in s_strip_lds[col][c_ch].
 *
 * Unconditional in every pass: the tail pass's dead lanes carry zeros and land
 * them in the partition's pad rows, which nothing reads. */
static void rocke_dconv_wgrad__commit_s_strip(rocke_dconv_wgrad_ctx_t* ctx,
                                              rocke_value_t* const* vecs)
{
    rocke_ir_builder_t* b = ctx->b;
    int pass_idx;
    for(pass_idx = 0; pass_idx < ctx->STRIP_PASSES_PER_WAVE; ++pass_idx)
    {
        rocke_value_t* indices[2];
        indices[0] = ctx->c0;
        indices[1]
            = rocke_dconv_wgrad__lds_run(ctx, ctx->s_strip_off_f16, ctx->strip_cols[pass_idx]);
        rocke_b_smem_store_vN_f16(b, ctx->s_strip_lds, indices, 2, vecs[pass_idx], ctx->VEC_CH);
    }
}

/* _row_coords(hi_in_blk) -> (hi, hi_ok, ho, ho_ok) for one input row. */
static void rocke_dconv_wgrad__row_coords(rocke_dconv_wgrad_ctx_t* ctx,
                                          int hi_in_blk,
                                          rocke_value_t** out_hi,
                                          rocke_value_t** out_hi_ok,
                                          rocke_value_t** out_ho,
                                          rocke_value_t** out_ho_ok)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* hi_val;
    rocke_value_t* hi_ok;
    rocke_value_t* ho_val;

    hi_val = rocke_b_add(b, ctx->hi_block_start, rocke_b_const_i32(b, hi_in_blk));
    hi_ok = rocke_b_cmp_lt(b, hi_val, ctx->c_H);
    /* dY row this input row feeds through r = 0. */
    ho_val = rocke_b_add(b, hi_val, rocke_b_const_i32(b, ctx->p.PAD));

    *out_hi = hi_val;
    *out_hi_ok = hi_ok;
    *out_ho = ho_val;
    *out_ho_ok = rocke_b_land(b, hi_ok, rocke_b_cmp_lt(b, ho_val, ctx->c_Ho));
}

/* ===================================================================== *
 *  Prologue
 * ===================================================================== */
bool rocke_dconv_wgrad_prologue(rocke_dconv_wgrad_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_wgrad_spec_t* spec = ctx->spec;
    char reason[ROCKE_ERR_MSG_CAP];
    int r, s, j;

    /* spec.validate(); ok, why = is_valid_wgrad_spec(spec, arch=arch) */
    if(rocke_direct_conv_wgrad_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    if(!rocke_direct_conv_wgrad_is_valid_spec(spec, ctx->arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->p = spec->problem;
    ctx->Ho = rocke_direct_conv_problem_Ho(&ctx->p);
    ctx->Wo = rocke_direct_conv_problem_Wo(&ctx->p);
    ctx->KH = ctx->p.KH;
    ctx->KW = ctx->p.KW;
    if(ctx->KH > ROCKE_DCONV_WGRAD_MAX_KH || ctx->KW > ROCKE_DCONV_WGRAD_MAX_KW)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->WAVE_K = spec->wave_tile_k;
    ctx->WAVE_C = spec->wave_tile_c;
    ctx->WAVES_K = spec->waves_k;
    ctx->WAVES_C = spec->waves_c;
    ctx->WAVES_Q = spec->waves_q;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_conv_wgrad_threads_per_block(spec);
    ctx->HPB = spec->ho_per_block;

    ctx->WO_BLOCK = spec->mfma_k;
    ctx->VEC_CH = spec->mfma_k / 4;
    ctx->n_wo_tiles = rocke_direct_conv_wgrad_n_wo_tiles(spec);
    ctx->STRIP_COLS = ctx->WO_BLOCK + ctx->KW - 1;

    ctx->TR_N = ctx->WAVE_K;
    ctx->TR_K_L = ctx->WO_BLOCK / 4;
    ctx->N_TR_READS = ctx->VEC_CH / 4;
    ctx->LDS_SIZE_DY = ctx->WO_BLOCK * ctx->TR_N;

    ctx->STRIP_PASSES = (ctx->STRIP_COLS + ctx->WO_BLOCK - 1) / ctx->WO_BLOCK;
    ctx->STRIP_GROUPS = (ctx->WAVES_K < ctx->STRIP_PASSES) ? ctx->WAVES_K : ctx->STRIP_PASSES;
    ctx->STRIP_PASSES_PER_WAVE = (ctx->STRIP_PASSES + ctx->STRIP_GROUPS - 1) / ctx->STRIP_GROUPS;
    ctx->STRIP_COLS_PAD = ctx->STRIP_PASSES_PER_WAVE * ctx->STRIP_GROUPS * ctx->WO_BLOCK;
    ctx->STRIP_PER_Q = ctx->STRIP_COLS_PAD * ctx->TR_N;
    if(ctx->STRIP_PASSES_PER_WAVE > ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->n_k_tiles = (ctx->p.kpg + rocke_direct_conv_wgrad_block_k(spec) - 1)
                     / rocke_direct_conv_wgrad_block_k(spec);
    ctx->n_c_tiles = (ctx->p.cpg + rocke_direct_conv_wgrad_block_c(spec) - 1)
                     / rocke_direct_conv_wgrad_block_c(spec);
    ctx->n_q_blocks = rocke_direct_conv_wgrad_n_q_blocks(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    {
        const rocke_type_t* f16ptr = rocke_ptr_type(b, rocke_f16(), "global");
        const rocke_type_t* f32ptr = rocke_ptr_type(b, rocke_f32(), "global");
        rocke_param_opts_t ro;
        rocke_param_opts_t dw;
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

        /* D: read-modify-write through global_atomic_add -> neither readonly
         * nor writeonly; align 4 (fp32). */
        dw = (rocke_param_opts_t){0};
        dw.noalias = true;
        dw.noalias_set = true;
        dw.align = 4;
        dw.align_set = true;
        ctx->D = rocke_b_param(b, "D", f32ptr, &dw);

        none = (rocke_param_opts_t){0};
        ctx->A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
        ctx->B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
        /* dW is reached by plain global_atomic_add, not a buffer resource, so
         * the size is unused -- but the parameter still has to be declared to
         * match the launch signature every conv kernel shares. */
        (void)rocke_b_param(b, "D_bytes", rocke_i32(), &none);
    }

    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_cpg = rocke_b_const_i32(b, ctx->p.cpg);
    ctx->c_kpg = rocke_b_const_i32(b, ctx->p.kpg);
    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->c_H = rocke_b_const_i32(b, ctx->p.H);
    ctx->c_Ho = rocke_b_const_i32(b, ctx->Ho);
    ctx->c_Wo = rocke_b_const_i32(b, ctx->Wo);

    ctx->zero_acc = rocke_b_zero_vec_f32(b, 4);

    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);
    ctx->c4 = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 16));
    ctx->q_in_lane = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 16));

    /* ---- Grid decode ----
     * bx = (group * n_k_tiles + k_tile) * n_c_tiles + c_tile
     * by = hi_block
     * bz = n * n_q_blocks + q_block */
    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->bz = rocke_b_block_id_z(b);

    ctx->c_n_k_tiles = rocke_b_const_i32(b, ctx->n_k_tiles);
    ctx->c_n_c_tiles = rocke_b_const_i32(b, ctx->n_c_tiles);
    ctx->c_n_q_blocks = rocke_b_const_i32(b, ctx->n_q_blocks);

    ctx->c_tile_idx = rocke_b_mod(b, ctx->bx, ctx->c_n_c_tiles);
    ctx->gk_flat = rocke_b_div(b, ctx->bx, ctx->c_n_c_tiles);
    ctx->k_tile_in_group = rocke_b_mod(b, ctx->gk_flat, ctx->c_n_k_tiles);
    ctx->group = rocke_b_div(b, ctx->gk_flat, ctx->c_n_k_tiles);

    ctx->n_i = rocke_b_div(b, ctx->bz, ctx->c_n_q_blocks);
    ctx->q_block = rocke_b_mod(b, ctx->bz, ctx->c_n_q_blocks);

    /* hi_block = by */
    ctx->hi_block_start = rocke_b_mul(b, ctx->by, rocke_b_const_i32(b, ctx->HPB));

    /* Wave decomposition: KCQ layout (q is fastest-varying). */
    ctx->wave_q_id = rocke_b_mod(b, ctx->wave_id, rocke_b_const_i32(b, ctx->WAVES_Q));
    ctx->wave_kc_id = rocke_b_div(b, ctx->wave_id, rocke_b_const_i32(b, ctx->WAVES_Q));
    ctx->wave_k_id = rocke_b_mod(b, ctx->wave_kc_id, rocke_b_const_i32(b, ctx->WAVES_K));
    ctx->wave_c_id = rocke_b_div(b, ctx->wave_kc_id, rocke_b_const_i32(b, ctx->WAVES_K));

    ctx->wave_k_origin = rocke_b_mul(b, ctx->wave_k_id, rocke_b_const_i32(b, ctx->WAVE_K));
    ctx->wave_c_origin = rocke_b_mul(b, ctx->wave_c_id, rocke_b_const_i32(b, ctx->WAVE_C));

    ctx->wo_tile = rocke_b_add(
        b, rocke_b_mul(b, ctx->q_block, rocke_b_const_i32(b, ctx->WAVES_Q)), ctx->wave_q_id);
    ctx->wo_tile_start = rocke_b_mul(b, ctx->wo_tile, rocke_b_const_i32(b, ctx->WO_BLOCK));
    ctx->wo_tile_valid = rocke_b_cmp_lt(b, ctx->wo_tile, rocke_b_const_i32(b, ctx->n_wo_tiles));

    ctx->k_tile_origin = rocke_b_mul(
        b, ctx->k_tile_in_group, rocke_b_const_i32(b, rocke_direct_conv_wgrad_block_k(spec)));
    ctx->c_tile_origin = rocke_b_mul(
        b, ctx->c_tile_idx, rocke_b_const_i32(b, rocke_direct_conv_wgrad_block_c(spec)));

    {
        rocke_value_t* mul_g = rocke_b_mul(b, ctx->group, ctx->c_kpg);
        rocke_value_t* inner = rocke_b_add(b, mul_g, ctx->k_tile_origin);
        ctx->k_wave_base = rocke_b_add(b, inner, ctx->wave_k_origin);
    }

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);

    /* ---- Descriptors (pure data; no IR emitted here) ---- */
    {
        static const char* const dy_coords[4] = {"n", "h", "w", "k"};
        int lengths[4];
        lengths[0] = ctx->p.N;
        lengths[1] = ctx->Ho;
        lengths[2] = ctx->Wo;
        lengths[3] = rocke_direct_conv_problem_total_k(&ctx->p);
        ctx->dy_desc = rocke_tensor_descriptor_naive(b, "A", lengths, 4, NULL, dy_coords, 4);
    }
    {
        static const char* const x_coords[4] = {"n", "h", "w", "c"};
        static const char* const w_upper[2] = {"wo", "s_off"};
        int lengths[4];
        int strides[2];
        rocke_tensor_descriptor_t* x_naive;
        const rocke_transform_t* xforms[1];

        lengths[0] = ctx->p.N;
        lengths[1] = ctx->p.H;
        lengths[2] = ctx->p.W;
        lengths[3] = rocke_direct_conv_problem_total_c(&ctx->p);
        x_naive = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, x_coords, 4);

        strides[0] = ctx->p.stride;
        strides[1] = 1;
        xforms[0] = rocke_embed_bounded(b, w_upper, 2, "w", strides, -ctx->p.PAD, 0, ctx->p.W);
        ctx->x_strip_desc = rocke_tensor_descriptor_transform(b, x_naive, xforms, 1);
    }
    {
        static const char* const dw_coords[4] = {"k", "r", "s", "c"};
        int lengths[4];
        lengths[0] = rocke_direct_conv_problem_total_k(&ctx->p);
        lengths[1] = ctx->p.KH;
        lengths[2] = ctx->p.KW;
        lengths[3] = ctx->p.cpg;
        ctx->dw_desc = rocke_tensor_descriptor_naive(b, "D", lengths, 4, NULL, dw_coords, 4);
    }

    /* ---- LDS allocation ----
     * Each tile is keyed on exactly the wave axes its contents depend on and
     * every wave writes precisely the bytes it later reads, which is what lets
     * the row loop run on ONE barrier per iteration. */
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = ctx->WAVES_K * ctx->WAVES_Q * ctx->LDS_SIZE_DY;
        ctx->dy_lds = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "dy_lds");
    }
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = ctx->WAVES_C * ctx->WAVES_Q * ctx->STRIP_PER_Q;
        ctx->s_strip_lds = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "s_strip");
    }

    /* ---- Per-thread loader decomposition ---- */
    ctx->c_lanes_per_sp = rocke_b_const_i32(b, ctx->TR_N / ctx->VEC_CH);
    ctx->c_ld_sp = rocke_b_div(b, ctx->lane, ctx->c_lanes_per_sp);
    {
        /* Python: b.mul(b.mod(lane, c_lanes_per_sp), b.const_i32(VEC_CH)) --
         * the mod is emitted before the const. C++ leaves sibling argument
         * evaluation unsequenced, so hoist to pin the order. */
        rocke_value_t* lane_mod = rocke_b_mod(b, ctx->lane, ctx->c_lanes_per_sp);
        rocke_value_t* c_vec = rocke_b_const_i32(b, ctx->VEC_CH);
        ctx->c_ld_ch = rocke_b_mul(b, lane_mod, c_vec);
    }

    ctx->dy_part_idx = rocke_b_add(
        b, rocke_b_mul(b, ctx->wave_k_id, rocke_b_const_i32(b, ctx->WAVES_Q)), ctx->wave_q_id);
    ctx->dy_wave_off_f16 = rocke_b_mul(b, ctx->dy_part_idx, rocke_b_const_i32(b, ctx->LDS_SIZE_DY));
    ctx->s_strip_part_idx = rocke_b_add(
        b, rocke_b_mul(b, ctx->wave_c_id, rocke_b_const_i32(b, ctx->WAVES_Q)), ctx->wave_q_id);
    ctx->s_strip_off_f16
        = rocke_b_mul(b, ctx->s_strip_part_idx, rocke_b_const_i32(b, ctx->STRIP_PER_Q));
    ctx->c_TR_N = rocke_b_const_i32(b, ctx->TR_N);
    ctx->c_WO_BLOCK = rocke_b_const_i32(b, ctx->WO_BLOCK);
    ctx->c_STRIP_COLS = rocke_b_const_i32(b, ctx->STRIP_COLS);

    /* Transpose-read lane address, shared by both operands (same tile width):
     *   tr_row  = (lane/16)*TR_K_L + (lane/4) % 4
     *   tr_flat = tr_row*TR_N + (lane % 4)*4 */
    {
        rocke_value_t* lhs = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, ctx->TR_K_L));
        rocke_value_t* div4 = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 4));
        rocke_value_t* rhs = rocke_b_mod(b, div4, rocke_b_const_i32(b, 4));
        ctx->tr_row = rocke_b_add(b, lhs, rhs);
    }
    {
        rocke_value_t* lhs = rocke_b_mul(b, ctx->tr_row, ctx->c_TR_N);
        rocke_value_t* mod4 = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 4));
        rocke_value_t* rhs = rocke_b_mul(b, mod4, rocke_b_const_i32(b, 4));
        ctx->tr_flat = rocke_b_add(b, lhs, rhs);
    }

    /* The strip columns this wave loads: pass base + j*STRIP_GROUPS of the
     * partition it shares with the other k-waves. This is the one place a wave
     * reads LDS another wave wrote; safe because sync_lds_only is a real
     * barrier, not just a waitcnt. */
    ctx->strip_pass_base = rocke_b_mod(b, ctx->wave_k_id, rocke_b_const_i32(b, ctx->STRIP_GROUPS));
    for(j = 0; j < ctx->STRIP_PASSES_PER_WAVE; ++j)
    {
        rocke_value_t* c_j = rocke_b_const_i32(b, j * ctx->STRIP_GROUPS);
        rocke_value_t* pass_idx = rocke_b_add(b, ctx->strip_pass_base, c_j);
        rocke_value_t* scaled = rocke_b_mul(b, pass_idx, ctx->c_WO_BLOCK);
        ctx->strip_cols[j] = rocke_b_add(b, ctx->c_ld_sp, scaled);
    }
    for(j = 0; j < ctx->STRIP_PASSES_PER_WAVE; ++j)
    {
        ctx->strip_col_ok[j] = rocke_b_cmp_lt(b, ctx->strip_cols[j], ctx->c_STRIP_COLS);
    }

    /* ---- Accumulators and delta register ring ----
     * Python builds acc as [[zero_acc]*KW]*KH and the ring as
     * [b.zero_vec_f16(VEC_CH)]*KH: ONE zero_vec_f16 op shared by every slot. */
    for(r = 0; r < ctx->KH; ++r)
    {
        for(s = 0; s < ctx->KW; ++s)
        {
            ctx->acc[r][s] = ctx->zero_acc;
        }
    }
    {
        rocke_value_t* ring_zero = rocke_b_zero_vec_f16(b, ctx->VEC_CH);
        for(r = 0; r < ctx->KH; ++r)
        {
            ctx->delta_ring[r] = ring_zero;
        }
    }

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Ring prologue: pre-load the KH-1 past dY rows.
 *
 *  For hi_block B the ring needs the delta rows of the virtual input rows
 *  hi_block_start - 1 .. hi_block_start - (KH-1); each maps to output row
 *  ho = hi_virtual + PAD (the r = 0 pairing). A negative ho reads zero through
 *  the OOB sentinel, which is exactly the boundary condition.
 * ===================================================================== */
void rocke_dconv_wgrad_ring_prologue(rocke_dconv_wgrad_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int k;

    for(k = ctx->KH - 1; k > 0; --k)
    {
        int slot_pre = (ctx->KH - k) % ctx->KH;
        rocke_value_t* ho_past;
        rocke_value_t* ho_past_ok;
        rocke_value_t* vec;

        ho_past = rocke_b_add(b, ctx->hi_block_start, rocke_b_const_i32(b, ctx->p.PAD - k));
        {
            /* Python: b.land(b.cmp_ge(...), b.cmp_lt(...)) -- ge first. */
            rocke_value_t* ge = rocke_b_cmp_ge(b, ho_past, ctx->c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, ho_past, ctx->c_Ho);
            ho_past_ok = rocke_b_land(b, ge, lt);
        }
        vec = rocke_dconv_wgrad__issue_delta(ctx, ho_past, ho_past_ok);
        if(vec == NULL)
        {
            return;
        }
        rocke_dconv_wgrad__commit_delta(ctx, vec);
        rocke_b_sync_lds_only(b);
        ctx->delta_ring[slot_pre] = rocke_dconv_wgrad__read_dy(ctx);
        rocke_b_s_barrier_bare(b);
    }
}

/* ===================================================================== *
 *  The unrolled row loop (HPB input rows).
 *
 *  Software-pipelined by one row: iteration i commits the fragments issued at
 *  i - 1 and issues row i + 1's, so every s_waitcnt vmcnt for a DRAM read is
 *  separated from its load by a whole compute phase.
 * ===================================================================== */
void rocke_dconv_wgrad_row_loop(rocke_dconv_wgrad_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int KH = ctx->KH;
    int KW = ctx->KW;
    int hi_in_blk;

    {
        rocke_value_t* hi0;
        rocke_value_t* hi0_ok;
        rocke_value_t* ho0;
        rocke_value_t* ho0_ok;
        rocke_dconv_wgrad__row_coords(ctx, 0, &hi0, &hi0_ok, &ho0, &ho0_ok);
        ctx->pending_dy = rocke_dconv_wgrad__issue_delta(ctx, ho0, ho0_ok);
        if(ctx->pending_dy == NULL)
        {
            return;
        }
        if(!rocke_dconv_wgrad__issue_s_strip(ctx, hi0, hi0_ok, ctx->pending_x))
        {
            return;
        }
    }

    for(hi_in_blk = 0; hi_in_blk < ctx->HPB; ++hi_in_blk)
    {
        int slot_fill = hi_in_blk % KH;
        rocke_value_t* x_vecs[ROCKE_DCONV_WGRAD_MAX_KW];
        int r, s;

        /* 1. Land the fragments issued last iteration. */
        rocke_b_s_setprio(b, 0);
        rocke_dconv_wgrad__commit_delta(ctx, ctx->pending_dy);
        rocke_dconv_wgrad__commit_s_strip(ctx, ctx->pending_x);

        /* 2. Issue the next row's DRAM reads before waiting on this one's LDS
         *    writes, so their latency runs under the compute phase below. */
        if(hi_in_blk + 1 < ctx->HPB)
        {
            rocke_value_t* nxt_hi;
            rocke_value_t* nxt_hi_ok;
            rocke_value_t* nxt_ho;
            rocke_value_t* nxt_ho_ok;
            rocke_dconv_wgrad__row_coords(
                ctx, hi_in_blk + 1, &nxt_hi, &nxt_hi_ok, &nxt_ho, &nxt_ho_ok);
            ctx->pending_dy = rocke_dconv_wgrad__issue_delta(ctx, nxt_ho, nxt_ho_ok);
            if(ctx->pending_dy == NULL)
            {
                return;
            }
            if(!rocke_dconv_wgrad__issue_s_strip(ctx, nxt_hi, nxt_hi_ok, ctx->pending_x))
            {
                return;
            }
        }

        /* 3. Wait for the LDS writes (lgkmcnt=0) of both dY and X. */
        rocke_b_sync_lds_only(b);

        /* 4. Update the delta ring -> VGPR and read the KW S fragments once
         *    each. Hoisted out of the r loop: the same KW fragments feed all KH
         *    rows, so re-reading them per r would triple the LDS read traffic. */
        ctx->delta_ring[slot_fill] = rocke_dconv_wgrad__read_dy(ctx);
        for(s = 0; s < KW; ++s)
        {
            x_vecs[s] = rocke_dconv_wgrad__read_strip(ctx, s);
        }

        /* 5. Compute phase: s_setprio(1) + KH*KW MFMAs. */
        rocke_b_s_setprio(b, 1);
        for(r = 0; r < KH; ++r)
        {
            int ring_slot = (hi_in_blk + KH - r) % KH;
            rocke_value_t* dy_vec = ctx->delta_ring[ring_slot];
            for(s = 0; s < KW; ++s)
            {
                if(ctx->spec->mfma_k == 32)
                {
                    ctx->acc[r][s]
                        = rocke_b_mfma_f32_16x16x32_f16(b, dy_vec, x_vecs[s], ctx->acc[r][s]);
                }
                else
                {
                    ctx->acc[r][s]
                        = rocke_b_mfma_f32_16x16x16_f16(b, dy_vec, x_vecs[s], ctx->acc[r][s]);
                }
            }
        }

        rocke_b_s_setprio(b, 0);
        rocke_b_s_barrier_bare(b);
    }
}

/* ===================================================================== *
 *  Epilogue: atomic-add into dW.
 *
 *  dW is [total_k, KH, KW, cpg]: the k axis is global but the c axis is
 *  per-group, so the channel index here is the IN-GROUP one. Feeding the global
 *  channel (which is what X is addressed by) would walk off the end of the
 *  filter's c extent for every group > 0.
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv_wgrad_epilogue(rocke_dconv_wgrad_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c_in_group_lane;
    rocke_value_t* c_valid_guard;
    rocke_value_t* k_in_group_base;
    int r, s, slot;

    c_in_group_lane
        = rocke_b_add(b, rocke_b_add(b, ctx->c_tile_origin, ctx->wave_c_origin), ctx->q_in_lane);
    c_valid_guard = rocke_b_cmp_lt(b, c_in_group_lane, ctx->c_cpg);
    k_in_group_base = rocke_b_sub(b, ctx->k_wave_base, rocke_b_mul(b, ctx->group, ctx->c_kpg));

    for(r = 0; r < ctx->KH; ++r)
    {
        for(s = 0; s < ctx->KW; ++s)
        {
            for(slot = 0; slot < 4; ++slot)
            {
                rocke_value_t* k_abs;
                rocke_value_t* k_in_group;
                rocke_value_t* k_valid;
                rocke_value_t* both_valid;
                rocke_value_t* acc_val;
                rocke_value_t* dw_off = NULL;
                rocke_value_t* dw_valid = NULL;
                const char* names[4];
                rocke_value_t* values[4];

                {
                    rocke_value_t* mul_c4 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));
                    rocke_value_t* row = rocke_b_add(b, mul_c4, rocke_b_const_i32(b, slot));
                    k_abs = rocke_b_add(b, ctx->k_wave_base, row);
                }
                {
                    rocke_value_t* mul_c4 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));
                    rocke_value_t* row = rocke_b_add(b, mul_c4, rocke_b_const_i32(b, slot));
                    k_in_group = rocke_b_add(b, k_in_group_base, row);
                }
                k_valid = rocke_b_cmp_lt(b, k_in_group, ctx->c_kpg);
                both_valid
                    = rocke_b_land(b, rocke_b_land(b, k_valid, c_valid_guard), ctx->wo_tile_valid);
                acc_val = rocke_b_vec_extract(b, ctx->acc[r][s], slot);

                /* Python evaluates the const_i32(r) / const_i32(s) keyword
                 * arguments before the .offset() call body runs. */
                names[0] = "k";
                values[0] = k_abs;
                names[1] = "r";
                values[1] = rocke_b_const_i32(b, r);
                names[2] = "s";
                values[2] = rocke_b_const_i32(b, s);
                names[3] = "c";
                values[3] = c_in_group_lane;
                if(!rocke_transforms_descriptor_offset(
                       b, ctx->dw_desc, names, values, 4, &dw_off, &dw_valid))
                {
                    return NULL;
                }

                {
                    rocke_if_t gate = rocke_b_scf_if(b, both_valid);
                    rocke_b_region_enter(b, gate.then_region);
                    (void)rocke_b_global_atomic_add(b, ctx->D, dw_off, acc_val, NULL);
                    rocke_b_region_leave(b);
                }
            }
        }
    }

    return rocke_ir_builder_kernel(b);
}
