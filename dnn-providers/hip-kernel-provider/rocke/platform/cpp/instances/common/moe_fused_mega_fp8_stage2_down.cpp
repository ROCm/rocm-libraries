// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_fp8_moe_fused_mega_fp8_stage2_down.c -- C99 port of the
 * STAGE 2 (down GEMM + weighted atomic reduce) module emitters of
 * rocke/instances/common/moe_fused_mega_fp8.py:
 *
 *   _emit_fp8_down_group_gemm  (lines 1258-1395)
 *       -> rocke_moe_fp8_emit_fp8_down_group_gemm
 *   _emit_down_atomic_reduce   (lines 1398-1473)
 *       -> rocke_moe_fp8_emit_down_atomic_reduce
 *
 * Both bind to the shared build context (rocke_moe_fp8_build_ctx_t) via the private
 * internal header. The down group GEMM is the LDS-A Hidden . global W_down
 * group-accumulator GEMM with per-128-group dequant by hidden_dyn_scale *
 * down_scale and a W_down register double-buffer prefetch pipeline. The atomic
 * reduce hoists the per-row SortedTokenIds / SortedWeights loads out of the
 * column loop and drains once, then performs token-validity-masked weighted
 * global_atomic_add into Y.
 *
 * Peers (gate/up loads, the cadence/MFMA dispatchers) are reached through the
 * internal-header prototypes; this file does not redefine them. The two MfmaAtom
 * methods used here (zero_acc, lane_to_output) are reproduced inline as tiny
 * static helpers -- byte-for-byte with rocke/helpers/atoms.py -- because the
 * atoms.h port does not expose them.
 *
 * Byte-faithful builder-call order against rocke/ir.h + the sibling helper headers.
 */
#include "rocke/instance_moe_fused_mega_fp8_internal.h"

#include <stdio.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ====================================================================== *
 *  Inline MfmaAtom method reproductions (atoms.py).
 * ====================================================================== */

/* MfmaAtom.zero_acc: fresh <c_per_lane x float> accumulator (all zeros). */
static rocke_value_t* rocke_moe_fp8_atom_zero_acc(rocke_ir_builder_t* b,
                                                  const rocke_mfma_atom_t* atom)
{
    return rocke_b_zero_vec_f32(b, atom->c_per_lane);
}

/* MfmaAtom.lane_to_output(b, lane, i): per-lane (row_in_atom, col_in_atom) of
 * accumulator slot i. Reproduces atoms.py:536-591 (16x16 / 32x32 / 4x4). On an
 * unsupported atom shape sets the sticky error and leaves the out-params NULL
 * (the Python NotImplementedError path). */
static void rocke_moe_fp8_atom_lane_to_output(rocke_ir_builder_t* b,
                                              const rocke_mfma_atom_t* atom,
                                              rocke_value_t* lane,
                                              int i,
                                              rocke_value_t** out_row,
                                              rocke_value_t** out_col)
{
    *out_row = NULL;
    *out_col = NULL;
    if(atom->m == 16 && atom->n == 16)
    {
        rocke_value_t* c_atom_n = rocke_b_const_i32(b, atom->n);
        rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_atom_n);
        rocke_value_t* m_blk = rocke_b_div(b, lane, c_atom_n);
        /* Python: row = b.add(b.mul(m_blk, c_per_lane), b.const_i32(i)) -- the
         * mul is emitted FIRST, then the const_i32(i). Force C arg-eval order. */
        rocke_value_t* mul_v = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, atom->c_per_lane));
        rocke_value_t* row = rocke_b_add(b, mul_v, rocke_b_const_i32(b, i));
        *out_row = row;
        *out_col = n_in_atom;
        return;
    }
    if(atom->m == 32 && atom->n == 32)
    {
        rocke_value_t* c_atom_n = rocke_b_const_i32(b, atom->n);
        rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_atom_n);
        rocke_value_t* m_blk = rocke_b_div(b, lane, c_atom_n);
        int rb = i / 4;
        int ri = i % 4;
        /* Python: row = b.add(b.add(b.const_i32(rb*8), b.mul(m_blk, b.const_i32(4))),
         *                     b.const_i32(ri)) -- const_i32(rb*8) first, then the
         * mul, then the inner add, then const_i32(ri), then the outer add. */
        rocke_value_t* c_rb = rocke_b_const_i32(b, rb * 8);
        rocke_value_t* mul_v = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 4));
        rocke_value_t* inner = rocke_b_add(b, c_rb, mul_v);
        rocke_value_t* row = rocke_b_add(b, inner, rocke_b_const_i32(b, ri));
        *out_row = row;
        *out_col = n_in_atom;
        return;
    }
    if(atom->m == 4 && atom->n == 4)
    {
        rocke_value_t* c4 = rocke_b_const_i32(b, 4);
        rocke_value_t* lane_in_batch = rocke_b_mod(b, lane, c4);
        *out_row = rocke_b_const_i32(b, i);
        *out_col = lane_in_batch;
        return;
    }
    /* Faithful mirror of atoms.py:AtomLayout.lane_to_output, which raises
     * NotImplementedError("no lane_to_output dispatch for atom {m}x{n}") for any
     * MFMA shape other than the three dispatched above (16x16, 32x32, 4x4). This
     * is byte-faithful reject behaviour (not unported C work): only atom shapes
     * Python also rejects reach here, and the message text matches verbatim. */
    rocke_i_set_err(
        b, ROCKE_ERR_NOTIMPL, "no lane_to_output dispatch for atom %dx%d", atom->m, atom->n);
}

/* ====================================================================== *
 *  _emit_fp8_down_group_gemm  (Python lines 1258-1395)
 *
 *  Down fp8 GEMM for one warp-atom output cell -> per-lane f32 vector.
 *  Group-accumulator pattern: the outer loop walks 128-wide groups along the
 *  inter contraction; per group, 4 fp8_16x16x32 atoms accumulate into a fresh
 *  group_acc, then the group is folded into the outer accumulator scaled by
 *  hidden_dyn_scale * down_b_scale via a single vector_fma.
 * ====================================================================== */

rocke_value_t* rocke_moe_fp8_emit_fp8_down_group_gemm(rocke_moe_fp8_build_ctx_t* ctx,
                                                      const rocke_tensor_view_t* a_view,
                                                      rocke_value_t* WDown,
                                                      rocke_value_t* WDownScale,
                                                      rocke_value_t* n_tile_base,
                                                      const rocke_tensor_view_t* scale_view,
                                                      int inter_slice,
                                                      rocke_value_t* inter_full,
                                                      rocke_value_t* inter_blk_base,
                                                      rocke_value_t* stride_down_scale,
                                                      rocke_value_t* m_row_base,
                                                      const char* tag,
                                                      const char* cadence)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;

    rocke_value_t* c_group_k = rocke_b_const_i32(b, ROCKE_MOE_FP8_GROUP_K);
    rocke_value_t* c_atom_k = rocke_b_const_i32(b, atom->k);
    int atoms_per_group = ROCKE_MOE_FP8_GROUP_K / atom->k; /* 4 */

    rocke_value_t* n_col = rocke_b_add(b, n_tile_base, lane_decode->n_in_atom);
    rocke_value_t* h_out_blk = rocke_b_div(b, n_col, c_group_k);
    /* CORRECTNESS FIX: the down-GEMM A operand row follows the per-mi atom
     * m-base (m_row_base = down_warp_m_off + mi*atom.m), not a hardcoded 0. */
    rocke_value_t* m_row = rocke_b_add(b, m_row_base, lane_decode->m_in_atom);

    /* Global inter column base for this TG's slice (W_down contraction origin). */
    rocke_value_t* inter_col_base = rocke_b_mul(b, inter_blk_base, c_group_k);

    rocke_value_t* zero = rocke_moe_fp8_atom_zero_acc(b, atom);
    rocke_value_t* outer_zero = rocke_moe_fp8_atom_zero_acc(b, atom);

    /* num_groups = inter_slice // GROUP_K (local inter slice / 128). */
    rocke_value_t* num_groups = rocke_b_const_i32(b, inter_slice / ROCKE_MOE_FP8_GROUP_K);

    char outer_name[48];
    char iv_name[32];
    snprintf(outer_name, sizeof(outer_name), "down_outer_%s", tag ? tag : "");
    snprintf(iv_name, sizeof(iv_name), "dg_%s", tag ? tag : "");

    rocke_iter_arg_t iter_args[1];
    iter_args[0].name = outer_name;
    iter_args[0].init = outer_zero;

    rocke_value_t* lo_dg = rocke_b_const_i32(b, 0);
    rocke_value_t* st_dg = rocke_b_const_i32(b, 1);
    rocke_for_t outer
        = rocke_b_scf_for_iter(b, lo_dg, num_groups, st_dg, iter_args, 1, iv_name, false, true);
    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* kg = outer.iv;
        rocke_value_t* down_outer = outer.iter_vars[0];

        rocke_moe_fp8_emit_loop_cadence_hint(ctx, cadence);

        /* A-side dynamic scale: HiddenScale_smem[m_row, local-inter-block kg]. */
        rocke_value_t* scale_idx[2] = {m_row, kg};
        rocke_value_t* a_scale_v = rocke_b_vec_extract(
            b, rocke_b_smem_load_vN(b, scale_view->base, scale_idx, 2, rocke_f32(), 1), 0);
        /* B-side W_down scale: per (GLOBAL inter-block, H_out-block). */
        rocke_value_t* global_blk = rocke_b_add(b, inter_blk_base, kg);
        rocke_value_t* down_scale_off
            = rocke_b_add(b, rocke_b_mul(b, global_blk, stride_down_scale), h_out_blk);
        rocke_value_t* down_scale_v = rocke_b_global_load_f32(b, WDownScale, down_scale_off, 0);
        rocke_value_t* ab_scale = rocke_b_fmul(b, a_scale_v, down_scale_v);

        rocke_value_t* local_k_group = rocke_b_mul(b, kg, c_group_k);
        rocke_value_t* global_k_group = rocke_b_add(b, inter_col_base, local_k_group);

        /* L5: software-pipeline the W_down (B) tile. The inner k-group loop has a
         * compile-time-constant trip count (atoms_per_group == 4), so it is
         * unrolled in C and the B=W_down fragments are register double-buffered:
         * the next atom's global B load is issued before the current b_frag is
         * consumed in the MFMA, keeping a load in flight under the MFMA. A stays a
         * direct LDS read: cheap, already resident. */

        /* Prefetch the first atom's W_down fragment, then pipeline. */
        rocke_value_t* b_cur = rocke_moe_fp8_load_b_fp8(
            ctx,
            WDown,
            n_tile_base,
            rocke_b_add(b, global_k_group, rocke_b_mul(b, rocke_b_const_i32(b, 0), c_atom_k)),
            inter_full);
        rocke_value_t* group_acc = zero;
        for(int kk = 0; kk < atoms_per_group; ++kk)
        {
            rocke_value_t* a_frag = rocke_moe_fp8_load_a_fp8_lds(
                ctx,
                a_view,
                m_row_base,
                rocke_b_add(b, local_k_group, rocke_b_mul(b, rocke_b_const_i32(b, kk), c_atom_k)));
            /* Issue the NEXT atom's W_down load (in flight during the MFMA). */
            rocke_value_t* b_next = NULL;
            if(kk + 1 < atoms_per_group)
            {
                b_next = rocke_moe_fp8_load_b_fp8(
                    ctx,
                    WDown,
                    n_tile_base,
                    rocke_b_add(
                        b, global_k_group, rocke_b_mul(b, rocke_b_const_i32(b, kk + 1), c_atom_k)),
                    inter_full);
            }
            rocke_value_t* d_new = rocke_moe_fp8_emit_mfma_down(ctx, a_frag, b_cur, group_acc);
            group_acc = d_new;
            if(kk + 1 < atoms_per_group)
            {
                b_cur = b_next;
            }
        }

        /* D5 sgb: place the next-group W_down VMEM under this group's MFMA(s). */
        rocke_moe_fp8_emit_sgb_down_group(ctx, atoms_per_group, cadence);

        rocke_value_t* scale_vec = rocke_b_vector_splat(b, ab_scale, atom->c_per_lane);
        rocke_value_t* down_outer_new = rocke_b_vector_fma(b, group_acc, scale_vec, down_outer);
        rocke_value_t* yielded[1] = {down_outer_new};
        rocke_b_scf_yield(b, yielded, 1);
    }
    rocke_b_region_leave(b);

    return (outer.op != NULL) ? outer.op->results[0] : NULL;
}

/* ====================================================================== *
 *  Packed MXFP4 down GEMM, independently scaled per K=32 MFMA.
 * ====================================================================== */
rocke_value_t* rocke_moe_fp8_emit_mxfp4_down_group_gemm(rocke_moe_fp8_build_ctx_t* ctx,
                                                        const rocke_tensor_view_t* a_view,
                                                        rocke_value_t* WDown,
                                                        rocke_value_t* WDownScale,
                                                        rocke_value_t* n_tile_base,
                                                        const rocke_tensor_view_t* scale_view,
                                                        int inter_slice,
                                                        rocke_value_t* inter_full,
                                                        rocke_value_t* inter_blk_base,
                                                        rocke_value_t* stride_down_scale,
                                                        rocke_value_t* m_row_base,
                                                        const char* tag)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;
    rocke_value_t* c_group;
    rocke_value_t* n_col;
    rocke_value_t* m_row;
    rocke_value_t* global_inter_base;
    char iter_name[64];
    char iv_name[64];
    rocke_iter_arg_t iter_args[1];
    rocke_for_t outer;

    if(atom->k != 32)
        return NULL;

    c_group = rocke_b_const_i32(b, 32);
    n_col = rocke_b_add(b, n_tile_base, lane_decode->n_in_atom);
    m_row = rocke_b_add(b, m_row_base, lane_decode->m_in_atom);
    global_inter_base = rocke_b_mul(b, inter_blk_base, rocke_b_const_i32(b, ROCKE_MOE_FP8_GROUP_K));
    snprintf(iter_name, sizeof(iter_name), "mxdown_outer_%s", tag ? tag : "");
    snprintf(iv_name, sizeof(iv_name), "mxdg_%s", tag ? tag : "");
    {
        rocke_value_t* lo = rocke_b_const_i32(b, 0);
        rocke_value_t* hi = rocke_b_const_i32(b, inter_slice / 32);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        iter_args[0].name = iter_name;
        iter_args[0].init = rocke_moe_fp8_atom_zero_acc(b, atom);
        outer = rocke_b_scf_for_iter(b, lo, hi, step, iter_args, 1, iv_name, false, true);
    }

    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* kg = outer.iv;
        rocke_value_t* down_outer = outer.iter_vars[0];
        rocke_value_t* hidden_group = rocke_b_div(b, kg, rocke_b_const_i32(b, 4));
        rocke_value_t* scale_idx[2] = {m_row, hidden_group};
        rocke_value_t* hidden_scale_vec
            = rocke_b_smem_load_vN(b, scale_view->base, scale_idx, 2, rocke_f32(), 1);
        rocke_value_t* hidden_scale = rocke_b_vec_extract(b, hidden_scale_vec, 0);
        rocke_value_t* expert_group = rocke_b_mul(b, inter_blk_base, rocke_b_const_i32(b, 4));
        rocke_value_t* global_group = rocke_b_add(b, expert_group, kg);
        rocke_value_t* weight_scale_row = rocke_b_mul(b, n_col, stride_down_scale);
        rocke_value_t* weight_scale_offset = rocke_b_add(b, weight_scale_row, global_group);
        rocke_value_t* weight_encoded
            = rocke_b_global_load_i8(b, WDownScale, weight_scale_offset, 0);
        rocke_value_t* weight_scale = rocke_moe_fp8_decode_e8m0_scale(ctx, weight_encoded);
        rocke_value_t* local_k = rocke_b_mul(b, kg, c_group);
        rocke_value_t* global_k = rocke_b_add(b, global_inter_base, local_k);
        rocke_value_t* a_fragment = rocke_moe_fp8_load_a_fp8_lds(ctx, a_view, m_row_base, local_k);
        rocke_value_t* b_fragment
            = rocke_moe_fp8_load_b_mxfp4_as_fp8(ctx, WDown, n_tile_base, global_k, inter_full);
        rocke_value_t* group_zero = rocke_moe_fp8_atom_zero_acc(b, atom);
        rocke_value_t* group
            = rocke_b_mma(b, atom->name, a_fragment, b_fragment, group_zero, NULL, 0);
        rocke_value_t* scale_product = rocke_b_fmul(b, hidden_scale, weight_scale);
        rocke_value_t* scale = rocke_b_vector_splat(b, scale_product, atom->c_per_lane);
        rocke_value_t* result = rocke_b_vector_fma(b, group, scale, down_outer);
        rocke_value_t* yielded[1] = {result};
        rocke_b_scf_yield(b, yielded, 1);
    }
    rocke_b_region_leave(b);
    return outer.op != NULL ? outer.op->results[0] : NULL;
}

typedef struct native_down_group
{
    rocke_value_t* hidden_scale;
    rocke_value_t* a_fragment;
    rocke_value_t* b_fragment;
    rocke_value_t* weight_scale;
} native_down_group_t;

static native_down_group_t native_down_load_group(rocke_moe_fp8_build_ctx_t* ctx,
                                                  const rocke_tensor_view_t* a_view,
                                                  rocke_value_t* WDown,
                                                  rocke_value_t* WDownScale,
                                                  rocke_value_t* n_tile_base,
                                                  const rocke_tensor_view_t* scale_view,
                                                  rocke_value_t* inter_full,
                                                  rocke_value_t* global_inter_base,
                                                  rocke_value_t* stride_down_scale,
                                                  rocke_value_t* m_row_base,
                                                  rocke_value_t* m_row,
                                                  rocke_value_t* n_col,
                                                  rocke_value_t* c_group,
                                                  rocke_value_t* kg)
{
    rocke_ir_builder_t* b = ctx->b;
    native_down_group_t g;
    rocke_value_t* scale_idx[2] = {m_row, kg};
    rocke_value_t* scale_vec
        = rocke_b_smem_load_vN(b, scale_view->base, scale_idx, 2, rocke_f32(), 1);
    g.hidden_scale = rocke_b_vec_extract(b, scale_vec, 0);
    rocke_value_t* local_k = rocke_b_mul(b, kg, c_group);
    rocke_value_t* global_k = rocke_b_add(b, global_inter_base, local_k);
    g.a_fragment = rocke_moe_fp8_load_a_fp8_lds_native(ctx, a_view, m_row_base, local_k);
    g.b_fragment = rocke_moe_fp8_load_b_mxfp4_native(
        ctx, WDown, n_tile_base, global_k, inter_full, ctx->spec->mxfp4_preshuffled, true);
    g.weight_scale = rocke_moe_fp8_load_mxfp4_scale_word(ctx,
                                                         WDownScale,
                                                         n_col,
                                                         global_k,
                                                         stride_down_scale,
                                                         ctx->lane_decode.k_blk,
                                                         ctx->spec->mxfp4_preshuffled);
    return g;
}

static rocke_value_t* native_down_accumulate(rocke_moe_fp8_build_ctx_t* ctx,
                                             native_down_group_t g,
                                             rocke_value_t* down_outer)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* zero = rocke_moe_fp8_atom_zero_acc(b, ctx->atom);
    rocke_value_t* group = rocke_b_mfma_scale_f32_16x16x128_fp8_fp4(
        b, g.a_fragment, g.b_fragment, zero, ctx->MxScaleA, g.weight_scale);
    rocke_value_t* scale = rocke_b_vector_splat(b, g.hidden_scale, ctx->atom->c_per_lane);
    return rocke_b_vector_fma(b, group, scale, down_outer);
}

rocke_value_t*
    rocke_moe_fp8_emit_mxfp4_native_down_group_gemm(rocke_moe_fp8_build_ctx_t* ctx,
                                                    const rocke_tensor_view_t* a_view,
                                                    rocke_value_t* WDown,
                                                    rocke_value_t* WDownScale,
                                                    rocke_value_t* n_tile_base,
                                                    const rocke_tensor_view_t* scale_view,
                                                    int inter_slice,
                                                    rocke_value_t* inter_full,
                                                    rocke_value_t* inter_blk_base,
                                                    rocke_value_t* stride_down_scale,
                                                    rocke_value_t* m_row_base,
                                                    const char* tag)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c_group = rocke_b_const_i32(b, 128);
    rocke_value_t* n_col = rocke_b_add(b, n_tile_base, ctx->lane_decode.n_in_atom);
    rocke_value_t* m_row = rocke_b_add(b, m_row_base, ctx->lane_decode.m_in_atom);
    rocke_value_t* global_inter_base
        = rocke_b_mul(b, inter_blk_base, rocke_b_const_i32(b, ROCKE_MOE_FP8_GROUP_K));
    if(ctx->spec->pipeline_native_down)
    {
        int groups = inter_slice / 128;
        native_down_group_t current = native_down_load_group(ctx,
                                                             a_view,
                                                             WDown,
                                                             WDownScale,
                                                             n_tile_base,
                                                             scale_view,
                                                             inter_full,
                                                             global_inter_base,
                                                             stride_down_scale,
                                                             m_row_base,
                                                             m_row,
                                                             n_col,
                                                             c_group,
                                                             rocke_b_const_i32(b, 0));
        rocke_value_t* outer = rocke_moe_fp8_atom_zero_acc(b, ctx->atom);
        int i;
        for(i = 0; i < groups; ++i)
        {
            native_down_group_t next;
            bool has_next = i + 1 < groups;
            if(has_next)
                next = native_down_load_group(ctx,
                                              a_view,
                                              WDown,
                                              WDownScale,
                                              n_tile_base,
                                              scale_view,
                                              inter_full,
                                              global_inter_base,
                                              stride_down_scale,
                                              m_row_base,
                                              m_row,
                                              n_col,
                                              c_group,
                                              rocke_b_const_i32(b, i + 1));
            outer = native_down_accumulate(ctx, current, outer);
            if(has_next)
                current = next;
        }
        return outer;
    }

    char iter_name[64];
    char iv_name[64];
    snprintf(iter_name, sizeof(iter_name), "mxndown_outer_%s", tag ? tag : "");
    snprintf(iv_name, sizeof(iv_name), "mxndg_%s", tag ? tag : "");
    rocke_value_t* lo = rocke_b_const_i32(b, 0);
    rocke_value_t* hi = rocke_b_const_i32(b, inter_slice / 128);
    rocke_value_t* step = rocke_b_const_i32(b, 1);
    rocke_iter_arg_t arg;
    arg.name = iter_name;
    arg.init = rocke_moe_fp8_atom_zero_acc(b, ctx->atom);
    rocke_for_t loop = rocke_b_scf_for_iter(b, lo, hi, step, &arg, 1, iv_name, false, true);
    rocke_b_region_enter(b, loop.body);
    {
        native_down_group_t g = native_down_load_group(ctx,
                                                       a_view,
                                                       WDown,
                                                       WDownScale,
                                                       n_tile_base,
                                                       scale_view,
                                                       inter_full,
                                                       global_inter_base,
                                                       stride_down_scale,
                                                       m_row_base,
                                                       m_row,
                                                       n_col,
                                                       c_group,
                                                       loop.iv);
        rocke_value_t* result = native_down_accumulate(ctx, g, loop.iter_vars[0]);
        rocke_value_t* yielded[1] = {result};
        rocke_b_scf_yield(b, yielded, 1);
    }
    rocke_b_region_leave(b);
    return loop.op ? loop.op->results[0] : NULL;
}

/* ====================================================================== *
 *  _emit_down_atomic_reduce  (Python lines 1398-1473)
 *
 *  Weighted, token-validity-masked atomic reduce of the down result into Y:
 *  Y[token, h_out] += weight * down_dq (f32 atomic add). Padded rows (sorted
 *  token id < 0 or >= tokens) are skipped.
 *
 *  L1: the sorted-token id and routing weight are per-ROW (bucket == row); they
 *  do not depend on the output column. Hoist the token/weight load + validity
 *  check to ONE per (mi, i) row, drain once for the whole batch, then run the
 *  per-ni atomics with the operands already resident.
 * ====================================================================== */

#define ROCKE_MOE_FP8_DOWN_MAX_C_PER_LANE 32

int rocke_moe_fp8_prefetch_down_routing_rows(rocke_moe_fp8_build_ctx_t* ctx,
                                             rocke_value_t* warp_m_off,
                                             rocke_value_t* lane,
                                             int mfmas_m,
                                             rocke_value_t* block_m_off,
                                             rocke_value_t* SortedTokenIds,
                                             rocke_value_t* SortedWeights,
                                             rocke_moe_fp8_down_row_t* out_rows)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    int count = 0;
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        for(int i = 0; i < atom->c_per_lane; ++i)
        {
            rocke_value_t* row_in = NULL;
            rocke_value_t* col_in = NULL;
            rocke_moe_fp8_atom_lane_to_output(b, atom, lane, i, &row_in, &col_in);
            rocke_value_t* row = rocke_b_add(
                b,
                block_m_off,
                rocke_b_add(
                    b, warp_m_off, rocke_b_add(b, rocke_b_const_i32(b, mi * atom->m), row_in)));
            out_rows[count].i = i;
            out_rows[count].col_in = col_in;
            out_rows[count].token = rocke_b_global_load_i32(b, SortedTokenIds, row, 0);
            out_rows[count].w = rocke_b_global_load_f32(b, SortedWeights, row, 0);
            ++count;
        }
    }
    return count;
}

void rocke_moe_fp8_emit_down_atomic_reduce(rocke_moe_fp8_build_ctx_t* ctx,
                                           rocke_value_t* const* down_list,
                                           rocke_value_t* warp_m_off,
                                           rocke_value_t* warp_n_off,
                                           rocke_value_t* lane,
                                           int mfmas_m,
                                           int mfmas_n,
                                           rocke_value_t* block_m_off,
                                           rocke_value_t* ho_off,
                                           rocke_value_t* H_out,
                                           rocke_value_t* SortedTokenIds,
                                           rocke_value_t* SortedWeights,
                                           rocke_value_t* Y,
                                           rocke_value_t* tokens,
                                           const rocke_moe_fp8_down_row_t* prefetched_rows)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);

    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        /* Issue every row's token + weight load up front into distinct registers,
         * then drain once for the whole batch, then run the per-row validity-
         * masked atomics with the operands already resident. */
        rocke_moe_fp8_down_row_t rows[ROCKE_MOE_FP8_DOWN_MAX_C_PER_LANE];
        int num_rows = 0;
        if(prefetched_rows != NULL)
        {
            for(int i = 0; i < atom->c_per_lane; ++i)
                rows[num_rows++] = prefetched_rows[mi * atom->c_per_lane + i];
        }
        else
        {
            for(int i = 0; i < atom->c_per_lane; ++i)
            {
                rocke_value_t* row_in = NULL;
                rocke_value_t* col_in = NULL;
                rocke_moe_fp8_atom_lane_to_output(b, atom, lane, i, &row_in, &col_in);
                rocke_value_t* row = rocke_b_add(
                    b,
                    block_m_off,
                    rocke_b_add(
                        b, warp_m_off, rocke_b_add(b, rocke_b_const_i32(b, mi * atom->m), row_in)));
                rows[num_rows].i = i;
                rows[num_rows].col_in = col_in;
                rows[num_rows].token = rocke_b_global_load_i32(b, SortedTokenIds, row, 0);
                rows[num_rows].w = rocke_b_global_load_f32(b, SortedWeights, row, 0);
                num_rows++;
            }
        }
        /* One rolling drain covers all c_per_lane (token,weight) loads instead of
         * one vmcnt(0) per row. */
        rocke_b_s_waitcnt(b, 0, -1, -1);
        for(int r = 0; r < num_rows; ++r)
        {
            int i = rows[r].i;
            rocke_value_t* col_in = rows[r].col_in;
            rocke_value_t* token = rows[r].token;
            rocke_value_t* w = rows[r].w;
            /* Python: b.land(b.cmp_ge(token, c0), b.cmp_lt(token, tokens)) -- the
             * cmp_ge is emitted FIRST. Force C arg-eval order. */
            rocke_value_t* ge = rocke_b_cmp_ge(b, token, c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, token, tokens);
            rocke_value_t* valid = rocke_b_land(b, ge, lt);
            rocke_if_t if_op = rocke_b_scf_if(b, valid);
            rocke_b_region_enter(b, if_op.then_region);
            {
                rocke_value_t* token_h = rocke_b_mul(b, token, H_out);
                for(int ni = 0; ni < mfmas_n; ++ni)
                {
                    int flat = mi * mfmas_n + ni;
                    rocke_value_t* acc = down_list[flat];
                    rocke_value_t* col = rocke_b_add(
                        b,
                        ho_off,
                        rocke_b_add(b,
                                    warp_n_off,
                                    rocke_b_add(b, rocke_b_const_i32(b, ni * atom->n), col_in)));
                    rocke_value_t* v = rocke_b_vec_extract(b, acc, i);
                    rocke_value_t* contrib = rocke_b_fmul(b, w, v);
                    rocke_value_t* y_off = rocke_b_add(b, token_h, col);
                    rocke_b_global_atomic_add(b, Y, y_off, contrib, NULL);
                }
            }
            rocke_b_region_leave(b);
        }
    }
}
