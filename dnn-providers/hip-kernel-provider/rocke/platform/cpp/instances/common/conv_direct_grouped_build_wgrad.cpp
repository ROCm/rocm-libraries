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
#include <string.h> /* strcmp for the p.dtype test */

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_op_for_shape */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV: gfx12+ atomic cache-policy immediate,
 * th = TH_ATOMIC_CASCADE (4), scope = SCOPE_DEV (2) in bits [4:3]. */
#define ROCKE_WGRAD_CPOL_ATOMIC_CASCADE_DEV (4 | (2 << 3))

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

    /* ds_read_tr16_b64 (wave64, 4 rows apart) or ds_load_tr16_b128 (wave32,
     * 8 rows apart). */
    indices[0] = ctx->c0;
    indices[1] = base;
    frag = ctx->W32 ? rocke_b_ds_read_tr16_b128(b, smem, indices, 2, ctx->io_type)
                    : rocke_b_ds_read_tr16_b64(b, smem, indices, 2, ctx->io_type);

    for(rd = 1; rd < ctx->N_TR_READS; ++rd)
    {
        int tr_step = ctx->W32 ? 8 : 4;
        rocke_value_t* c_step = rocke_b_const_i32(b, tr_step * rd * ctx->TR_N);
        rocke_value_t* addr = rocke_b_add(b, base, c_step);
        rocke_value_t* nxt;
        rocke_value_t* idx2[2];
        idx2[0] = ctx->c0;
        idx2[1] = addr;
        nxt = ctx->W32 ? rocke_b_ds_read_tr16_b128(b, smem, idx2, 2, ctx->io_type)
                       : rocke_b_ds_read_tr16_b64(b, smem, idx2, 2, ctx->io_type);
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

/* _dy_pass_sp(pass_idx): this lane's spatial row in load pass pass_idx of a
 * dY tile. */
static rocke_value_t* rocke_dconv_wgrad__dy_pass_sp(rocke_dconv_wgrad_ctx_t* ctx, int pass_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c_off;
    if(pass_idx == 0)
    {
        return ctx->c_ld_sp;
    }
    c_off = rocke_b_const_i32(b, pass_idx * ctx->ROWS_PER_PASS);
    return rocke_b_add(b, ctx->c_ld_sp, c_off);
}

/* _issue_delta(ho_val, ho_ok): start the DRAM reads of one dY row; writes the
 * DY_PASSES in-flight fragments (1 on wave64, 2 on wave32) into out[].
 *
 * No OOB masking of the loaded value: an out-of-range lane gets oob_sentinel as
 * its buffer offset and a buffer load past num_records returns zero. The
 * CHANNEL tail is unmasked too -- k and c are the MFMA M/N axes here, so a
 * ragged kpg/cpg only pollutes accumulator rows/columns the epilogue already
 * drops. See the long note in the Python loader block for the argument. */
static bool rocke_dconv_wgrad__issue_delta(rocke_dconv_wgrad_ctx_t* ctx,
                                           rocke_value_t* ho_val,
                                           rocke_value_t* ho_ok,
                                           rocke_value_t** out)
{
    rocke_ir_builder_t* b = ctx->b;
    int pass_idx;

    for(pass_idx = 0; pass_idx < ctx->DY_PASSES; ++pass_idx)
    {
        rocke_value_t* sp;
        rocke_value_t* wo_sp;
        rocke_value_t* both_ok;
        rocke_value_t* k_ld;
        rocke_value_t* off = NULL;
        rocke_value_t* valid = NULL;
        rocke_value_t* safe_off;
        const char* names[4];
        rocke_value_t* values[4];

        sp = rocke_dconv_wgrad__dy_pass_sp(ctx, pass_idx);
        wo_sp = rocke_b_add(b, ctx->wo_tile_start, sp);
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
            return false;
        }

        safe_off
            = rocke_b_select(b, both_ok, rocke_b_mul(b, off, ctx->c_half_bytes), ctx->oob_sentinel);
        out[pass_idx]
            = ctx->is_bf16
                  ? rocke_b_buffer_load_vN_bf16(b, ctx->a_rsrc, safe_off, ctx->c0, ctx->LD_VEC / 2)
                  : rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_off, ctx->c0, ctx->LD_VEC / 2);
    }
    return true;
}

/* _commit_delta(vecs): land a dY tile in dy_lds[sp][k_ch] -- one ds_write per
 * lane per pass. */
static void rocke_dconv_wgrad__commit_delta(rocke_dconv_wgrad_ctx_t* ctx,
                                            rocke_value_t* const* vecs)
{
    rocke_ir_builder_t* b = ctx->b;
    int pass_idx;
    for(pass_idx = 0; pass_idx < ctx->DY_PASSES; ++pass_idx)
    {
        rocke_value_t* indices[2];
        rocke_value_t* sp = rocke_dconv_wgrad__dy_pass_sp(ctx, pass_idx);
        indices[0] = ctx->c0;
        indices[1] = rocke_dconv_wgrad__lds_run(ctx, ctx->dy_wave_off_f16, sp);
        rocke_b_smem_store_vN(b, ctx->dy_lds, indices, 2, vecs[pass_idx], ctx->LD_VEC);
    }
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
        out[pass_idx]
            = ctx->is_bf16
                  ? rocke_b_buffer_load_vN_bf16(b, ctx->b_rsrc, safe, ctx->c0, ctx->LD_VEC / 2)
                  : rocke_b_buffer_load_vN_f16(b, ctx->b_rsrc, safe, ctx->c0, ctx->LD_VEC / 2);
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
        rocke_b_smem_store_vN(b, ctx->s_strip_lds, indices, 2, vecs[pass_idx], ctx->LD_VEC);
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
    ctx->wmma_op_id = NULL;
    if(spec->wave_size == 32)
    {
        /* _wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(family="wmma",
         * ab, ab, "fp32", 16, 16, 32) -- no IR emitted. */
        const char* ab = (ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0) ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(ctx->arch);
        const rocke_mmaop_t* op
            = target
                  ? rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, 32, NULL)
                  : NULL;
        if(op == NULL || op->op_id == NULL)
        {
            if(b->status == ROCKE_OK)
            {
                b->status = ROCKE_ERR_VALUE;
            }
            return false;
        }
        ctx->wmma_op_id = op->op_id;
    }
    ctx->KH = ctx->p.KH;
    ctx->KW = ctx->p.KW;
    /* Unreachable: both validators above already gate KH/KW on the same caps,
     * so Python and C++ agree on which specs are buildable. Kept as a hard stop
     * in front of the fixed-size ctx arrays. */
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
    ctx->STRIP_COLS = ctx->WO_BLOCK + ctx->KW - 1;

    /* wave64 uses VEC_CH three ways: load width, MMA operand length and the
     * per-lane run of a transpose read, and 64 lanes x VEC_CH cover exactly
     * WO_BLOCK x 16 per load pass. wave32 (WMMA 16x16x32) splits them: a lane
     * loads 8 channels (16 B), so a pass covers 16 spatial rows and a dY tile
     * takes two; the operand is 16 elements, two ds_load_tr16_b128. */
    ctx->W32 = (ctx->WAVE == 32) ? 1 : 0;
    ctx->LD_VEC = ctx->W32 ? 8 : ctx->VEC_CH;
    ctx->FRAG = ctx->W32 ? 16 : ctx->VEC_CH;
    ctx->ROWS_PER_PASS = ctx->WAVE * ctx->LD_VEC / ctx->WAVE_K;
    ctx->DY_PASSES = ctx->WO_BLOCK / ctx->ROWS_PER_PASS;
    ctx->C_FRAG = ctx->W32 ? 8 : 4;
    if(ctx->DY_PASSES < 1 || ctx->DY_PASSES > ROCKE_DCONV_WGRAD_MAX_DY_PASSES)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->TR_N = ctx->WAVE_K;
    ctx->TR_K_L = ctx->WO_BLOCK / 4;
    /* ds_read_b64_tr_b16 (4/lane) or ds_load_tr16_b128 (8/lane) per fragment. */
    ctx->N_TR_READS = ctx->W32 ? ctx->FRAG / 8 : ctx->VEC_CH / 4;
    ctx->LDS_SIZE_DY = ctx->WO_BLOCK * ctx->TR_N;

    ctx->STRIP_PASSES = (ctx->STRIP_COLS + ctx->ROWS_PER_PASS - 1) / ctx->ROWS_PER_PASS;
    ctx->STRIP_GROUPS = (ctx->WAVES_K < ctx->STRIP_PASSES) ? ctx->WAVES_K : ctx->STRIP_PASSES;
    ctx->STRIP_PASSES_PER_WAVE = (ctx->STRIP_PASSES + ctx->STRIP_GROUPS - 1) / ctx->STRIP_GROUPS;
    ctx->STRIP_COLS_PAD = ctx->STRIP_PASSES_PER_WAVE * ctx->STRIP_GROUPS * ctx->ROWS_PER_PASS;
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

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    /* io_type = _io_type(p.dtype): dY and X are f16 or bf16. dW is always
     * fp32 -- the split-K reduction lands through fp32 global atomics. */
    ctx->io_type = rocke_b_io_ir_type(b, ctx->p.dtype ? ctx->p.dtype : "fp16");
    if(ctx->io_type == NULL)
    {
        return false; /* builder sticky error already set by rocke_b_io_ir_type */
    }
    ctx->is_bf16 = ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0;

    /* AOT kernarg block, conv_abi wgrad order. dW is reached by plain
     * global_atomic_add, not a buffer resource, so D_bytes is unused -- but the
     * parameter still has to be declared to match the launch signature. */
    rocke_dconv_emit_params(b, &ctx->params, "wgrad", ctx->io_type);
    ctx->A = ctx->params.A;
    ctx->Bp = ctx->params.Bp;
    ctx->D = ctx->params.D;
    ctx->A_bytes = ctx->params.A_bytes;
    ctx->B_bytes = ctx->params.B_bytes;

    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    if(spec->runtime_channels)
    {
        /* Per-group channel counts from the kernargs, computed once:
         * total_c = groups * cpg, total_k = groups * kpg. */
        ctx->c_cpg = rocke_b_div(b, ctx->params.p_total_c, ctx->params.p_groups);
        ctx->c_kpg = rocke_b_div(b, ctx->params.p_total_k, ctx->params.p_groups);
    }
    else
    {
        ctx->c_cpg = rocke_b_const_i32(b, ctx->p.cpg);
        ctx->c_kpg = rocke_b_const_i32(b, ctx->p.kpg);
    }
    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->c_H = ctx->params.p_Hi;
    ctx->c_Ho = ctx->params.p_Ho;
    ctx->c_Wo = ctx->params.p_Wo;

    /* MFMA 16x16: 4 f32 per lane; WMMA 16x16: 8. */
    ctx->zero_acc = rocke_b_zero_vec_f32(b, ctx->C_FRAG);

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

    if(spec->runtime_channels)
    {
        /* n_k_tiles = ceil(kpg / block_k), n_c_tiles = ceil(cpg / block_c) from
         * the runtime channel counts. Sequenced through locals to pin Python's
         * left-to-right SSA order. */
        int block_k = rocke_direct_conv_wgrad_block_k(spec);
        int block_c = rocke_direct_conv_wgrad_block_c(spec);
        {
            rocke_value_t* c_round = rocke_b_const_i32(b, block_k - 1);
            rocke_value_t* sum = rocke_b_add(b, ctx->c_kpg, c_round);
            rocke_value_t* c_blk = rocke_b_const_i32(b, block_k);
            ctx->c_n_k_tiles = rocke_b_div(b, sum, c_blk);
        }
        {
            rocke_value_t* c_round = rocke_b_const_i32(b, block_c - 1);
            rocke_value_t* sum = rocke_b_add(b, ctx->c_cpg, c_round);
            rocke_value_t* c_blk = rocke_b_const_i32(b, block_c);
            ctx->c_n_c_tiles = rocke_b_div(b, sum, c_blk);
        }
    }
    else
    {
        ctx->c_n_k_tiles = rocke_b_const_i32(b, ctx->n_k_tiles);
        ctx->c_n_c_tiles = rocke_b_const_i32(b, ctx->n_c_tiles);
    }
    /* AOT: the wo-tile count follows the runtime output width.
     *   n_wo_tiles = ceil(Wo / WO_BLOCK);  n_q_blocks = ceil(n_wo_tiles / WAVES_Q)
     * Sequenced through locals to pin Python's left-to-right SSA order. */
    {
        rocke_value_t* c_round = rocke_b_const_i32(b, ctx->WO_BLOCK - 1);
        rocke_value_t* sum = rocke_b_add(b, ctx->c_Wo, c_round);
        rocke_value_t* c_blk = rocke_b_const_i32(b, ctx->WO_BLOCK);
        ctx->c_n_wo_tiles = rocke_b_div(b, sum, c_blk);
    }
    if(ctx->WAVES_Q == 1)
    {
        ctx->c_n_q_blocks = ctx->c_n_wo_tiles;
    }
    else
    {
        rocke_value_t* c_round = rocke_b_const_i32(b, ctx->WAVES_Q - 1);
        rocke_value_t* sum = rocke_b_add(b, ctx->c_n_wo_tiles, c_round);
        rocke_value_t* c_wq = rocke_b_const_i32(b, ctx->WAVES_Q);
        ctx->c_n_q_blocks = rocke_b_div(b, sum, c_wq);
    }

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
    ctx->wo_tile_valid = rocke_b_cmp_lt(b, ctx->wo_tile, ctx->c_n_wo_tiles);

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
    /* wave32 reduces through buffer atomics carrying the gfx1250 cascade cache
     * policy, so dW needs a resource there; wave64 keeps plain global atomics. */
    ctx->d_rsrc = ctx->W32 ? rocke_b_buffer_rsrc(b, ctx->D, ctx->params.D_bytes) : NULL;

    /* ---- Descriptors ----
     * AOT: dY and X carry runtime base strides, and the X column embed is
     * bounded by the runtime input width. Each dynamic descriptor's unit
     * stride is a const_i32(1) Python emits as it builds the descriptor, so
     * the constants are created here, in the same order. */
    {
        static const char* const dy_coords[4] = {"n", "h", "w", "k"};
        rocke_value_t* strides[4];
        rocke_dynamic_tensor_descriptor_t* dy_dyn;
        strides[0] = ctx->params.p_A_stride_n; /* p_dY_stride_n  */
        strides[1] = ctx->params.p_A_stride_hi; /* p_dY_stride_ho */
        strides[2] = ctx->params.p_A_stride_wi; /* p_dY_stride_wo */
        strides[3] = rocke_b_const_i32(b, 1);
        dy_dyn = rocke_tensor_descriptor_naive_dynamic(b, "A", dy_coords, 4, strides);
        if(!dy_dyn)
            return false;
        ctx->dy_desc = &dy_dyn->base;
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

        strides[0] = ctx->params.p_B_stride_n; /* p_X_stride_n  */
        strides[1] = ctx->params.p_B_stride_hi; /* p_X_stride_hi */
        strides[2] = ctx->params.p_B_stride_wi; /* p_X_stride_wi */
        strides[3] = rocke_b_const_i32(b, 1);
        x_dyn = rocke_tensor_descriptor_naive_dynamic(b, "B", x_coords, 4, strides);
        if(!x_dyn)
            return false;

        /* embed(("wo","s_off") -> "w", strides=(stride,1), offset=-PAD, lo=0, hi=Wi) */
        w_strides_c[0] = ctx->p.stride;
        w_strides_c[1] = 1;
        xforms[0] = rocke_embed_dynamic_mixed(
            b, w_upper, 2, "w", w_strides, w_strides_c, NULL, -ctx->p.PAD, 0, ctx->params.p_Wi);
        if(!xforms[0])
            return false;
        chained = rocke_tensor_descriptor_transform(b, &x_dyn->base, xforms, 1);
        if(!chained)
            return false;
        x_dyn->base = *chained;
        ctx->x_strip_desc = &x_dyn->base;
    }
    /* dW[total_k, KH, KW, cpg]: every stride is a product of build-time
     * extents, so it stays a static descriptor. total_k is only a length,
     * which offset() never reads -- the emitted IR does not depend on groups.
     * With runtime channels the strides scale with the runtime cpg instead:
     * [KH*KW*cpg, KW*cpg, cpg, 1]. */
    if(spec->runtime_channels)
    {
        static const char* const dw_coords[4] = {"k", "r", "s", "c"};
        rocke_value_t* strides[4];
        rocke_dynamic_tensor_descriptor_t* dw_dyn;
        {
            rocke_value_t* c_khkw = rocke_b_const_i32(b, ctx->KH * ctx->KW);
            strides[0] = rocke_b_mul(b, ctx->c_cpg, c_khkw);
        }
        {
            rocke_value_t* c_kw = rocke_b_const_i32(b, ctx->KW);
            strides[1] = rocke_b_mul(b, ctx->c_cpg, c_kw);
        }
        strides[2] = ctx->c_cpg;
        strides[3] = rocke_b_const_i32(b, 1);
        dw_dyn = rocke_tensor_descriptor_naive_dynamic(b, "D", dw_coords, 4, strides);
        if(!dw_dyn)
            return false;
        ctx->dw_desc = &dw_dyn->base;
    }
    else
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
        ctx->dy_lds = rocke_b_smem_alloc(b, ctx->io_type, shape, 2, "dy_lds");
    }
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = ctx->WAVES_C * ctx->WAVES_Q * ctx->STRIP_PER_Q;
        ctx->s_strip_lds = rocke_b_smem_alloc(b, ctx->io_type, shape, 2, "s_strip");
    }

    /* ---- Per-thread loader decomposition ---- */
    ctx->c_lanes_per_sp = rocke_b_const_i32(b, ctx->TR_N / ctx->LD_VEC);
    ctx->c_ld_sp = rocke_b_div(b, ctx->lane, ctx->c_lanes_per_sp);
    {
        /* Python: b.mul(b.mod(lane, c_lanes_per_sp), b.const_i32(VEC_CH)) --
         * the mod is emitted before the const. C++ leaves sibling argument
         * evaluation unsequenced, so hoist to pin the order. */
        rocke_value_t* lane_mod = rocke_b_mod(b, ctx->lane, ctx->c_lanes_per_sp);
        rocke_value_t* c_vec = rocke_b_const_i32(b, ctx->LD_VEC);
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
    /* Rows one load pass covers: WO_BLOCK on wave64, 16 on wave32. */
    ctx->c_rows_per_pass = ctx->W32 ? rocke_b_const_i32(b, ctx->ROWS_PER_PASS) : ctx->c_WO_BLOCK;

    if(ctx->W32)
    {
        /* ds_load_tr16_b128 transposes an 8x8 block within each group of 8
         * lanes: the group addresses the 8-channel block holding column
         * lane % 16, and lane l supplies spatial row (l / 16) * 16 + l % 8.
         * Two reads 8 rows apart give lane l spatial rows (l / 16) * 16 +
         * 0..15 of channel l % 16 -- the WMMA A/B fragment.
         *   tr_col  = ((lane % 16) / 8) * 8
         *   tr_row  = (lane/16)*FRAG + lane % 8
         *   tr_flat = tr_row*TR_N + tr_col */
        rocke_value_t* c8 = rocke_b_const_i32(b, 8);
        rocke_value_t* tr_col;
        {
            rocke_value_t* c16 = rocke_b_const_i32(b, 16);
            rocke_value_t* lane_mod16 = rocke_b_mod(b, ctx->lane, c16);
            rocke_value_t* blk = rocke_b_div(b, lane_mod16, c8);
            tr_col = rocke_b_mul(b, blk, c8);
        }
        {
            rocke_value_t* c_frag = rocke_b_const_i32(b, ctx->FRAG);
            rocke_value_t* lhs = rocke_b_mul(b, ctx->c4, c_frag);
            rocke_value_t* rhs = rocke_b_mod(b, ctx->lane, c8);
            ctx->tr_row = rocke_b_add(b, lhs, rhs);
        }
        {
            rocke_value_t* lhs = rocke_b_mul(b, ctx->tr_row, ctx->c_TR_N);
            ctx->tr_flat = rocke_b_add(b, lhs, tr_col);
        }
    }
    else
    {
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
        rocke_value_t* scaled = rocke_b_mul(b, pass_idx, ctx->c_rows_per_pass);
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
        rocke_value_t* ring_zero = rocke_b_zero_vec(b, ctx->io_type, ctx->FRAG);
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
        rocke_value_t* vecs[ROCKE_DCONV_WGRAD_MAX_DY_PASSES];

        ho_past = rocke_b_add(b, ctx->hi_block_start, rocke_b_const_i32(b, ctx->p.PAD - k));
        {
            /* Python: b.land(b.cmp_ge(...), b.cmp_lt(...)) -- ge first. */
            rocke_value_t* ge = rocke_b_cmp_ge(b, ho_past, ctx->c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, ho_past, ctx->c_Ho);
            ho_past_ok = rocke_b_land(b, ge, lt);
        }
        if(!rocke_dconv_wgrad__issue_delta(ctx, ho_past, ho_past_ok, vecs))
        {
            return;
        }
        rocke_dconv_wgrad__commit_delta(ctx, vecs);
        rocke_b_sync_lds_only(b);
        ctx->delta_ring[slot_pre] = rocke_dconv_wgrad__read_dy(ctx);
        /* Full sync_lds_only, NOT the bare barrier the row loop ends on: this
         * fragment is not consumed until row-loop iteration 0, so nothing
         * forces the ds_read to drain before a waves_c sibling's next
         * _commit_delta overwrites the shared dY partition. Mirrors Python. */
        rocke_b_sync_lds_only(b);
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
        if(!rocke_dconv_wgrad__issue_delta(ctx, ho0, ho0_ok, ctx->pending_dy))
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
            if(!rocke_dconv_wgrad__issue_delta(ctx, nxt_ho, nxt_ho_ok, ctx->pending_dy))
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
                /* _mfma(io_dtype, "16x16x{32,16}", dy_vec, x_vecs[s], acc), or
                 * b.mma(wmma 16x16x32) on wave32 */
                if(ctx->W32)
                {
                    ctx->acc[r][s] = rocke_b_mma(
                        b, ctx->wmma_op_id, dy_vec, x_vecs[s], ctx->acc[r][s], NULL, 0);
                }
                else if(ctx->spec->mfma_k == 32)
                {
                    ctx->acc[r][s]
                        = ctx->is_bf16
                              ? rocke_b_mfma_f32_16x16x32_bf16(b, dy_vec, x_vecs[s], ctx->acc[r][s])
                              : rocke_b_mfma_f32_16x16x32_f16(b, dy_vec, x_vecs[s], ctx->acc[r][s]);
                }
                else
                {
                    ctx->acc[r][s]
                        = ctx->is_bf16
                              ? rocke_b_mfma_f32_16x16x16_bf16(b, dy_vec, x_vecs[s], ctx->acc[r][s])
                              : rocke_b_mfma_f32_16x16x16_f16(b, dy_vec, x_vecs[s], ctx->acc[r][s]);
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
            for(slot = 0; slot < ctx->C_FRAG; ++slot)
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
                    rocke_value_t* mul_c4
                        = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, ctx->C_FRAG));
                    rocke_value_t* row = rocke_b_add(b, mul_c4, rocke_b_const_i32(b, slot));
                    k_abs = rocke_b_add(b, ctx->k_wave_base, row);
                }
                {
                    rocke_value_t* mul_c4
                        = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, ctx->C_FRAG));
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
                    if(ctx->W32)
                    {
                        /* Guarded rather than sent to the out-of-range
                         * sentinel: a buffer atomic at the sentinel faults on
                         * gfx1250, where a load reads 0. Cache policy
                         * CPOL_ATOMIC_CASCADE_DEV = 4 | (2 << 3). */
                        rocke_value_t* c4b = rocke_b_const_i32(b, 4);
                        rocke_value_t* voff = rocke_b_mul(b, dw_off, c4b);
                        rocke_b_buffer_atomic_fadd_f32(b,
                                                       ctx->d_rsrc,
                                                       voff,
                                                       ctx->c0,
                                                       acc_val,
                                                       ROCKE_WGRAD_CPOL_ATOMIC_CASCADE_DEV);
                    }
                    else
                    {
                        (void)rocke_b_global_atomic_add(b, ctx->D, dw_off, acc_val, NULL);
                    }
                    rocke_b_region_leave(b);
                }
            }
        }
    }

    return rocke_ir_builder_kernel(b);
}
