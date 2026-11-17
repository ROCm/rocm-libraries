// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_fp8_moe_fused_mega_fp8_stage1_gateup.c
 *
 * STAGE 1 GATE+UP GEMM + DYN-QUANT EMITTERS of the chunked C99 port of
 * rocke/instances/common/moe_fused_mega_fp8.py.
 *
 * Implements (Python -> C99 phase function):
 *   _emit_fp8_gateup_group_gemm  (lines 504-630)
 *       -> rocke_moe_fp8_emit_fp8_gateup_group_gemm
 *   _emit_fp8_gateup_fused_kloop (lines 633-980; legacy / DTLA / cluster paths)
 *       -> rocke_moe_fp8_emit_fp8_gateup_fused_kloop
 *   _silu_mul_f32                (lines 1481-1492)
 *       -> rocke_moe_fp8_silu_mul_f32
 *   _store_hidden_f32_pass       (lines 1495-1537)
 *       -> rocke_moe_fp8_store_hidden_f32_pass
 *   f32_view_store / f32_view_load (lines 1540-1546)
 *       -> rocke_moe_fp8_f32_view_store / rocke_moe_fp8_f32_view_load
 *
 * Binds only to the shared ctx + peer prototypes in
 * rocke/instance_moe_fused_mega_fp8_internal.h. Peer functions (_load_a_fp8,
 * _load_b_fp8, _emit_mfma, the cadence hints, the DTLA stage/read pairs, the
 * X-DTLA pair) are declared there and resolved at link time by their own part
 * files.
 *
 * The atom @property/method calls that have no standalone C entry point are
 * reproduced inline byte-for-byte from atoms.py, identically to the rest of the
 * port:
 *   atom.zero_acc(b)         -> rocke_b_zero_vec_f32(b, atom->c_per_lane)
 *   atom.lane_to_output(...) -> the 16x16 / 32x32 / 4x4 arith from atoms.py
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h> /* snprintf */

#include "rocke/helper_helpers.asm.h" /* rocke_mfma_f8f6f4_agpr_cluster */
#include "rocke/instance_moe_fused_mega_fp8_internal.h"

/* ===================================================================== *
 * atom method reproductions (no standalone C symbol; inline per port)
 * ===================================================================== */

/* atom.zero_acc(b) -> b.zero_vec_f32(self.c_per_lane). */
static rocke_value_t* moe_fp8_atom_zero_acc(rocke_ir_builder_t* b, const rocke_mfma_atom_t* atom)
{
    return rocke_b_zero_vec_f32(b, atom->c_per_lane);
}

/* atom.lane_to_output(b, lane, i) -> (row_in_atom, col_in_atom). Writes both via
 * out params. Faithful reproduction of atoms.py:536-591 (16x16 / 32x32 / 4x4).
 * The unsupported-shape branch leaves the outputs NULL (the Python
 * NotImplementedError path; the gate/up atom is always 16x16 here). */
static void moe_fp8_atom_lane_to_output(rocke_ir_builder_t* b,
                                        const rocke_mfma_atom_t* atom,
                                        rocke_value_t* lane,
                                        int i,
                                        rocke_value_t** out_row,
                                        rocke_value_t** out_col)
{
    if(atom->m == 16 && atom->n == 16)
    {
        rocke_value_t* c_atom_n = rocke_b_const_i32(b, atom->n);
        rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_atom_n);
        rocke_value_t* m_blk = rocke_b_div(b, lane, c_atom_n);
        /* Python: row = b.add(b.mul(m_blk, c_per_lane), b.const_i32(i)) -- mul
         * emitted FIRST. Force C arg-eval order with an ordered temporary. */
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
         *                     b.const_i32(ri)) -- const(rb*8), mul, inner add,
         * const(ri), outer add. Force the order. */
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
    *out_row = NULL;
    *out_col = NULL;
}

/* ===================================================================== *
 * STAGE 1b: SiLU*up + dyn-quant helpers (Python lines 1481-1546)
 * ===================================================================== */

/* f32_view_store: b.smem_store_vN(view.base, [row, col], val, 1). */
void rocke_moe_fp8_f32_view_store(rocke_moe_fp8_build_ctx_t* ctx,
                                  const rocke_tensor_view_t* view,
                                  rocke_value_t* row,
                                  rocke_value_t* col,
                                  rocke_value_t* val)
{
    rocke_value_t* idx[2];
    idx[0] = row;
    idx[1] = col;
    rocke_b_smem_store_vN(ctx->b, view->base, idx, 2, val, 1);
}

/* f32_view_load: b.smem_load_vN(view.base, row, col, dtype=F32, n=1)[0]. */
rocke_value_t* rocke_moe_fp8_f32_view_load(rocke_moe_fp8_build_ctx_t* ctx,
                                           const rocke_tensor_view_t* view,
                                           rocke_value_t* row,
                                           rocke_value_t* col)
{
    rocke_value_t* idx[2];
    rocke_value_t* v;
    idx[0] = row;
    idx[1] = col;
    v = rocke_b_smem_load_vN(ctx->b, view->base, idx, 2, rocke_f32(), 1);
    return rocke_b_vec_extract(ctx->b, v, 0);
}

/* _silu_mul_f32: f32 SwiGLU chain silu(g)*u (sigmoid via exp2 + rcp_fast). */
rocke_value_t* rocke_moe_fp8_silu_mul_f32(rocke_moe_fp8_build_ctx_t* ctx,
                                          rocke_value_t* g,
                                          rocke_value_t* u,
                                          rocke_value_t* one_f32,
                                          rocke_value_t* c_neg_log2e)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* sig = rocke_b_rcp_fast(
        b, rocke_b_fadd(b, one_f32, rocke_b_exp2(b, rocke_b_fmul(b, c_neg_log2e, g))));
    rocke_value_t* silu = rocke_b_fmul(b, g, sig);
    return rocke_b_fmul(b, silu, u);
}

/* _store_hidden_f32_pass: Pass A -- silu(gate)*up -> f32 LDS scratch + in-register
 * per-lane amax. Returns the per-lane partial amax (floored). */
rocke_value_t* rocke_moe_fp8_store_hidden_f32_pass(rocke_moe_fp8_build_ctx_t* ctx,
                                                   rocke_value_t* const* gate_list,
                                                   rocke_value_t* const* up_list,
                                                   const rocke_tensor_view_t* f32_view,
                                                   rocke_value_t* warp_m_off,
                                                   rocke_value_t* warp_n_off,
                                                   rocke_value_t* lane,
                                                   int mfmas_m,
                                                   int mfmas_n,
                                                   rocke_value_t* one_f32,
                                                   rocke_value_t* c_neg_log2e,
                                                   rocke_value_t* c_floor)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    rocke_value_t* amax_partial = c_floor;
    int mi, ni, i;

    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            int flat = mi * mfmas_n + ni;
            rocke_value_t* g_vec = gate_list[flat];
            rocke_value_t* u_vec = up_list[flat];
            for(i = 0; i < atom->c_per_lane; ++i)
            {
                rocke_value_t* row_in = NULL;
                rocke_value_t* col_in = NULL;
                rocke_value_t* row;
                rocke_value_t* col;
                rocke_value_t* g;
                rocke_value_t* u;
                rocke_value_t* h;
                moe_fp8_atom_lane_to_output(b, atom, lane, i, &row_in, &col_in);
                row = rocke_b_add(
                    b, warp_m_off, rocke_b_add(b, rocke_b_const_i32(b, mi * atom->m), row_in));
                col = rocke_b_add(
                    b, warp_n_off, rocke_b_add(b, rocke_b_const_i32(b, ni * atom->n), col_in));
                g = rocke_b_vec_extract(b, g_vec, i);
                u = rocke_b_vec_extract(b, u_vec, i);
                h = rocke_moe_fp8_silu_mul_f32(ctx, g, u, one_f32, c_neg_log2e);
                rocke_moe_fp8_f32_view_store(ctx, f32_view, row, col, h);
                amax_partial = rocke_b_fmax(b, amax_partial, rocke_b_fabs(b, h));
            }
        }
    }
    return amax_partial;
}

/* ===================================================================== *
 * STAGE 1a: legacy per-(mi,ni) gate+up group GEMM (Python lines 504-630)
 * ===================================================================== */

void rocke_moe_fp8_emit_fp8_gateup_group_gemm(rocke_moe_fp8_build_ctx_t* ctx,
                                              rocke_value_t* A,
                                              rocke_value_t* WGate,
                                              rocke_value_t* WUp,
                                              rocke_value_t* AScale,
                                              rocke_value_t* WGateScale,
                                              rocke_value_t* WUpScale,
                                              rocke_value_t* m_tile_base,
                                              rocke_value_t* n_tile_base,
                                              rocke_value_t* K,
                                              rocke_value_t* stride_a_scale,
                                              rocke_value_t* stride_gate_scale,
                                              rocke_value_t* stride_up_scale,
                                              const char* tag,
                                              rocke_value_t** out_gate,
                                              rocke_value_t** out_up)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;

    rocke_value_t* c_group_k = rocke_b_const_i32(b, ROCKE_MOE_FP8_GROUP_K);
    rocke_value_t* c_atom_k = rocke_b_const_i32(b, atom->k);
    int atoms_per_group = ROCKE_MOE_FP8_GROUP_K / atom->k; /* 4 */

    rocke_value_t* m_row = rocke_b_add(b, m_tile_base, lane_decode->m_in_atom);
    rocke_value_t* n_col = rocke_b_add(b, n_tile_base, lane_decode->n_in_atom);

    rocke_value_t* a_row_scale_base = rocke_b_mul(b, m_row, stride_a_scale);
    rocke_value_t* n_blk = rocke_b_div(b, n_col, c_group_k);

    rocke_value_t* zero = moe_fp8_atom_zero_acc(b, atom);
    rocke_value_t* gate_zero = moe_fp8_atom_zero_acc(b, atom);
    rocke_value_t* up_zero = moe_fp8_atom_zero_acc(b, atom);

    rocke_value_t* num_groups = rocke_b_div(b, K, c_group_k);

    char nm_gate[64];
    char nm_up[64];
    char nm_kg[64];
    rocke_iter_arg_t outer_args[2];
    rocke_for_t outer;

    /* iter_arg names: "gate_outer_{tag}", "up_outer_{tag}"; iv "kg_{tag}". */
    snprintf(nm_gate, sizeof(nm_gate), "gate_outer_%s", tag ? tag : "");
    snprintf(nm_up, sizeof(nm_up), "up_outer_%s", tag ? tag : "");
    snprintf(nm_kg, sizeof(nm_kg), "kg_%s", tag ? tag : "");
    outer_args[0].name = nm_gate;
    outer_args[0].init = gate_zero;
    outer_args[1].name = nm_up;
    outer_args[1].init = up_zero;

    {
        rocke_value_t* lo_kg = rocke_b_const_i32(b, 0);
        rocke_value_t* st_kg = rocke_b_const_i32(b, 1);
        outer = rocke_b_scf_for_iter(b,
                                     lo_kg,
                                     num_groups,
                                     st_kg,
                                     outer_args,
                                     2,
                                     nm_kg,
                                     /*unroll=*/false,
                                     /*elide_trailing_barrier=*/true);
    }

    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* kg = outer.iv;
        rocke_value_t* gate_outer = outer.iter_vars[0];
        rocke_value_t* up_outer = outer.iter_vars[1];

        rocke_value_t* a_scale_off = rocke_b_add(b, a_row_scale_base, kg);
        rocke_value_t* a_scale_v = rocke_b_global_load_f32(b, AScale, a_scale_off, 0);
        rocke_value_t* gate_scale_off
            = rocke_b_add(b, rocke_b_mul(b, kg, stride_gate_scale), n_blk);
        rocke_value_t* up_scale_off = rocke_b_add(b, rocke_b_mul(b, kg, stride_up_scale), n_blk);
        rocke_value_t* gate_scale_v = rocke_b_global_load_f32(b, WGateScale, gate_scale_off, 0);
        rocke_value_t* up_scale_v = rocke_b_global_load_f32(b, WUpScale, up_scale_off, 0);
        rocke_value_t* gate_ab = rocke_b_fmul(b, a_scale_v, gate_scale_v);
        rocke_value_t* up_ab = rocke_b_fmul(b, a_scale_v, up_scale_v);

        rocke_value_t* k_group_base = rocke_b_mul(b, kg, c_group_k);

        char nm_g[64];
        char nm_u[64];
        char nm_kk[64];
        rocke_iter_arg_t inner_args[2];
        rocke_for_t ginner;

        snprintf(nm_g, sizeof(nm_g), "g_acc_%s", tag ? tag : "");
        snprintf(nm_u, sizeof(nm_u), "u_acc_%s", tag ? tag : "");
        snprintf(nm_kk, sizeof(nm_kk), "kk_%s", tag ? tag : "");
        inner_args[0].name = nm_g;
        inner_args[0].init = zero;
        inner_args[1].name = nm_u;
        inner_args[1].init = zero;

        {
            rocke_value_t* lo_kk = rocke_b_const_i32(b, 0);
            rocke_value_t* hi_kk = rocke_b_const_i32(b, atoms_per_group);
            rocke_value_t* st_kk = rocke_b_const_i32(b, 1);
            ginner = rocke_b_scf_for_iter(b,
                                          lo_kk,
                                          hi_kk,
                                          st_kk,
                                          inner_args,
                                          2,
                                          nm_kk,
                                          /*unroll=*/false,
                                          /*elide_trailing_barrier=*/true);
        }

        rocke_b_region_enter(b, ginner.body);
        {
            rocke_value_t* kk = ginner.iv;
            rocke_value_t* g_acc = ginner.iter_vars[0];
            rocke_value_t* u_acc = ginner.iter_vars[1];

            rocke_value_t* k_tile_base = rocke_b_add(b, k_group_base, rocke_b_mul(b, kk, c_atom_k));
            rocke_value_t* a_frag = rocke_moe_fp8_load_a_fp8(ctx, A, m_tile_base, k_tile_base, K);
            rocke_value_t* gb_frag
                = rocke_moe_fp8_load_b_fp8(ctx, WGate, n_tile_base, k_tile_base, K);
            rocke_value_t* ub_frag
                = rocke_moe_fp8_load_b_fp8(ctx, WUp, n_tile_base, k_tile_base, K);
            rocke_value_t* g_new = rocke_moe_fp8_emit_mfma(ctx, a_frag, gb_frag, g_acc);
            rocke_value_t* u_new = rocke_moe_fp8_emit_mfma(ctx, a_frag, ub_frag, u_acc);
            rocke_value_t* yld[2];
            yld[0] = g_new;
            yld[1] = u_new;
            rocke_b_scf_yield(b, yld, 2);
        }
        rocke_b_region_leave(b);

        {
            rocke_value_t* group_gate = (ginner.op != NULL) ? ginner.op->results[0] : NULL;
            rocke_value_t* group_up = (ginner.op != NULL) ? ginner.op->results[1] : NULL;
            rocke_value_t* gate_scale_vec = rocke_b_vector_splat(b, gate_ab, atom->c_per_lane);
            rocke_value_t* up_scale_vec = rocke_b_vector_splat(b, up_ab, atom->c_per_lane);
            rocke_value_t* gate_outer_new
                = rocke_b_vector_fma(b, group_gate, gate_scale_vec, gate_outer);
            rocke_value_t* up_outer_new = rocke_b_vector_fma(b, group_up, up_scale_vec, up_outer);
            rocke_value_t* yld[2];
            yld[0] = gate_outer_new;
            yld[1] = up_outer_new;
            rocke_b_scf_yield(b, yld, 2);
        }
    }
    rocke_b_region_leave(b);

    if(out_gate != NULL)
    {
        *out_gate = (outer.op != NULL) ? outer.op->results[0] : NULL;
    }
    if(out_up != NULL)
    {
        *out_up = (outer.op != NULL) ? outer.op->results[1] : NULL;
    }
}

/* ===================================================================== *
 * STAGE 1a: gate+up fp8 GEMM fused across ALL ni (Python lines 633-980)
 * ===================================================================== */

/* _a_at(kk): _load_a_fp8 at k_group_base + kk*atom.k. */
static rocke_value_t* fused_a_at(rocke_moe_fp8_build_ctx_t* ctx,
                                 rocke_value_t* A,
                                 rocke_value_t* m_tile_base,
                                 rocke_value_t* k_group_base,
                                 rocke_value_t* c_atom_k,
                                 rocke_value_t* K,
                                 int kk)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* kbase
        = rocke_b_add(b, k_group_base, rocke_b_mul(b, rocke_b_const_i32(b, kk), c_atom_k));
    return rocke_moe_fp8_load_a_fp8(ctx, A, m_tile_base, kbase, K);
}

/* _gb_at / _ub_at(ni, kk): _load_b_fp8 of WGate/WUp at n_tile_bases[ni],
 * k_group_base + kk*atom.k. */
static rocke_value_t* fused_b_at(rocke_moe_fp8_build_ctx_t* ctx,
                                 rocke_value_t* B,
                                 rocke_value_t* const* n_tile_bases,
                                 rocke_value_t* k_group_base,
                                 rocke_value_t* c_atom_k,
                                 rocke_value_t* K,
                                 int ni,
                                 int kk)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* kbase
        = rocke_b_add(b, k_group_base, rocke_b_mul(b, rocke_b_const_i32(b, kk), c_atom_k));
    return rocke_moe_fp8_load_b_fp8(ctx, B, n_tile_bases[ni], kbase, K);
}

void rocke_moe_fp8_emit_fp8_gateup_fused_kloop(rocke_moe_fp8_build_ctx_t* ctx,
                                               rocke_value_t* A,
                                               rocke_value_t* WGate,
                                               rocke_value_t* WUp,
                                               rocke_value_t* AScale,
                                               rocke_value_t* WGateScale,
                                               rocke_value_t* WUpScale,
                                               rocke_value_t* m_tile_base,
                                               rocke_value_t* const* n_tile_bases,
                                               int nni,
                                               rocke_value_t* K,
                                               rocke_value_t* stride_a_scale,
                                               rocke_value_t* stride_gate_scale,
                                               rocke_value_t* stride_up_scale,
                                               const char* tag,
                                               const rocke_moe_fp8_dtla_bundle_t* dtla,
                                               const char* cadence,
                                               rocke_value_t** out_gate,
                                               rocke_value_t** out_up)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;
    rocke_fused_mega_fp8_levers_t* lv = &ctx->levers;

    rocke_value_t* c_group_k = rocke_b_const_i32(b, ROCKE_MOE_FP8_GROUP_K);
    rocke_value_t* c_atom_k = rocke_b_const_i32(b, atom->k);
    int atoms_per_group = ROCKE_MOE_FP8_GROUP_K / atom->k; /* 4 (K=32) or 1 (K=128) */

    rocke_value_t* m_row = rocke_b_add(b, m_tile_base, lane_decode->m_in_atom);
    rocke_value_t* a_row_scale_base = rocke_b_mul(b, m_row, stride_a_scale);

    /* Per-ni n_col / n_blk for the weight-scale index math. Python emits these
     * as TWO separate list comprehensions:
     *   n_cols = [b.add(nb, n_in_atom) for nb in n_tile_bases]
     *   n_blks = [b.div(nc, c_group_k) for nc in n_cols]
     * i.e. ALL the adds first, THEN all the divs. Keep that op-emit order. */
    rocke_value_t* n_cols[ROCKE_MOE_FP8_MAX_NNI];
    rocke_value_t* n_blks[ROCKE_MOE_FP8_MAX_NNI];
    int ni, kk;
    for(ni = 0; ni < nni; ++ni)
    {
        n_cols[ni] = rocke_b_add(b, n_tile_bases[ni], lane_decode->n_in_atom);
    }
    for(ni = 0; ni < nni; ++ni)
    {
        n_blks[ni] = rocke_b_div(b, n_cols[ni], c_group_k);
    }

    /* Outer iter-args: gate[0..nni), up[0..nni). */
    rocke_value_t* zero = moe_fp8_atom_zero_acc(b, atom);
    rocke_iter_arg_t iter_args[ROCKE_MOE_FP8_MAX_ACCS];
    char names[ROCKE_MOE_FP8_MAX_ACCS][64];
    int na = 0;
    for(ni = 0; ni < nni; ++ni)
    {
        snprintf(names[na], sizeof(names[na]), "g_out_%s_%d", tag ? tag : "", ni);
        iter_args[na].name = names[na];
        iter_args[na].init = moe_fp8_atom_zero_acc(b, atom);
        ++na;
    }
    for(ni = 0; ni < nni; ++ni)
    {
        snprintf(names[na], sizeof(names[na]), "u_out_%s_%d", tag ? tag : "", ni);
        iter_args[na].name = names[na];
        iter_args[na].init = moe_fp8_atom_zero_acc(b, atom);
        ++na;
    }

    rocke_value_t* num_groups = rocke_b_div(b, K, c_group_k);

    char nm_kg[64];
    rocke_for_t outer;
    snprintf(nm_kg, sizeof(nm_kg), "kg_%s", tag ? tag : "");
    {
        rocke_value_t* lo_kg = rocke_b_const_i32(b, 0);
        rocke_value_t* st_kg = rocke_b_const_i32(b, 1);
        outer = rocke_b_scf_for_iter(b,
                                     lo_kg,
                                     num_groups,
                                     st_kg,
                                     iter_args,
                                     na,
                                     nm_kg,
                                     /*unroll=*/false,
                                     /*elide_trailing_barrier=*/true);
    }

    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* kg = outer.iv;
        rocke_value_t* gate_outer[ROCKE_MOE_FP8_MAX_NNI];
        rocke_value_t* up_outer[ROCKE_MOE_FP8_MAX_NNI];

        rocke_value_t* a_scale_v;
        rocke_value_t* kg_gate;
        rocke_value_t* kg_up;
        rocke_value_t* gate_ab[ROCKE_MOE_FP8_MAX_NNI];
        rocke_value_t* up_ab[ROCKE_MOE_FP8_MAX_NNI];
        rocke_value_t* k_group_base;
        rocke_value_t* g_acc[ROCKE_MOE_FP8_MAX_NNI];
        rocke_value_t* u_acc[ROCKE_MOE_FP8_MAX_NNI];

        rocke_moe_fp8_emit_loop_cadence_hint(ctx, cadence);

        for(ni = 0; ni < nni; ++ni)
        {
            gate_outer[ni] = (outer.iter_vars != NULL) ? outer.iter_vars[ni] : NULL;
        }
        for(ni = 0; ni < nni; ++ni)
        {
            up_outer[ni] = (outer.iter_vars != NULL) ? outer.iter_vars[nni + ni] : NULL;
        }

        /* Per-group scales (hoisted alongside the operand prefetch). */
        {
            rocke_value_t* a_scale_off = rocke_b_add(b, a_row_scale_base, kg);
            a_scale_v = rocke_b_global_load_f32(b, AScale, a_scale_off, 0);
        }
        kg_gate = rocke_b_mul(b, kg, stride_gate_scale);
        kg_up = rocke_b_mul(b, kg, stride_up_scale);
        for(ni = 0; ni < nni; ++ni)
        {
            rocke_value_t* gsc
                = rocke_b_global_load_f32(b, WGateScale, rocke_b_add(b, kg_gate, n_blks[ni]), 0);
            rocke_value_t* usc
                = rocke_b_global_load_f32(b, WUpScale, rocke_b_add(b, kg_up, n_blks[ni]), 0);
            gate_ab[ni] = rocke_b_fmul(b, a_scale_v, gsc);
            up_ab[ni] = rocke_b_fmul(b, a_scale_v, usc);
        }

        k_group_base = rocke_b_mul(b, kg, c_group_k);

        /* Fresh per-group accumulators, one gate + one up per ni. */
        for(ni = 0; ni < nni; ++ni)
        {
            g_acc[ni] = zero;
            u_acc[ni] = zero;
        }

        if(dtla != NULL && dtla->present && atoms_per_group == 1)
        {
            /* ---- DTLA path (GOAL 1): direct-to-LDS gate+up B operands ----- */
            rocke_value_t* kbase0 = k_group_base;
            rocke_value_t* a_cur = NULL;
            int chunks_per_frag;
            int dmas_per_stage;

            if(lv->use_x_dtla && dtla->has_x_slot)
            {
                rocke_moe_fp8_xdtla_stage_a_fp8(ctx,
                                                A,
                                                m_tile_base,
                                                kbase0,
                                                K,
                                                dtla->view,
                                                dtla->x_slot,
                                                dtla->base,
                                                dtla->lane,
                                                dtla->wave_size);
            }
            else
            {
                a_cur = rocke_moe_fp8_load_a_fp8(ctx, A, m_tile_base, kbase0, K);
            }

            /* Prime: stage ni=0 into ping slot 0 (gate slot 0, up slot 1). */
            rocke_moe_fp8_dtla_stage_b_fp8(ctx,
                                           WGate,
                                           n_tile_bases[0],
                                           kbase0,
                                           K,
                                           dtla->view,
                                           2 * 0,
                                           dtla->base,
                                           dtla->lane,
                                           dtla->wave_size);
            rocke_moe_fp8_dtla_stage_b_fp8(ctx,
                                           WUp,
                                           n_tile_bases[0],
                                           kbase0,
                                           K,
                                           dtla->view,
                                           2 * 0 + 1,
                                           dtla->base,
                                           dtla->lane,
                                           dtla->wave_size);

            chunks_per_frag
                = (atom->b_per_lane + ROCKE_MOE_FP8_DTLA_CHUNK - 1) / ROCKE_MOE_FP8_DTLA_CHUNK;
            dmas_per_stage = 2 * chunks_per_frag;

            if(lv->use_mfma_cluster)
            {
                /* ---- NUCLEAR cluster path: 2*nni MFMAs as ONE asm block --- */
                rocke_value_t* gb_all[ROCKE_MOE_FP8_MAX_NNI];
                rocke_value_t* ub_all[ROCKE_MOE_FP8_MAX_NNI];
                rocke_value_t* accs[ROCKE_MOE_FP8_MAX_ACCS];
                rocke_value_t* srcs_a[ROCKE_MOE_FP8_MAX_ACCS];
                rocke_value_t* srcs_b[ROCKE_MOE_FP8_MAX_ACCS];
                rocke_value_t* outs[ROCKE_MOE_FP8_MAX_ACCS];
                int ncl = 0;

                if(a_cur == NULL)
                {
                    a_cur = rocke_moe_fp8_xdtla_read_a_fp8(ctx,
                                                           dtla->view,
                                                           dtla->x_slot,
                                                           dtla->lane,
                                                           dtla->warp_row_base,
                                                           dtla->wave_size);
                }
                /* Issue every remaining stage up front; stage in pairs and drain
                 * to the still-outstanding count to keep the prefetch in flight. */
                for(ni = 0; ni < nni; ++ni)
                {
                    if(ni + 1 < nni)
                    {
                        int pair = (ni + 1) % 2;
                        rocke_moe_fp8_dtla_stage_b_fp8(ctx,
                                                       WGate,
                                                       n_tile_bases[ni + 1],
                                                       kbase0,
                                                       K,
                                                       dtla->view,
                                                       2 * pair,
                                                       dtla->base,
                                                       dtla->lane,
                                                       dtla->wave_size);
                        rocke_moe_fp8_dtla_stage_b_fp8(ctx,
                                                       WUp,
                                                       n_tile_bases[ni + 1],
                                                       kbase0,
                                                       K,
                                                       dtla->view,
                                                       2 * pair + 1,
                                                       dtla->base,
                                                       dtla->lane,
                                                       dtla->wave_size);
                        rocke_b_s_waitcnt(b, dmas_per_stage, -1, -1);
                    }
                    else
                    {
                        rocke_b_s_waitcnt(b, 0, -1, -1);
                    }
                    {
                        int pair = ni % 2;
                        gb_all[ni] = rocke_moe_fp8_dtla_read_b_fp8(ctx,
                                                                   dtla->view,
                                                                   2 * pair,
                                                                   dtla->lane,
                                                                   dtla->warp_row_base,
                                                                   dtla->wave_size);
                        ub_all[ni] = rocke_moe_fp8_dtla_read_b_fp8(ctx,
                                                                   dtla->view,
                                                                   2 * pair + 1,
                                                                   dtla->lane,
                                                                   dtla->warp_row_base,
                                                                   dtla->wave_size);
                    }
                }
                for(ni = 0; ni < nni; ++ni)
                {
                    accs[ncl] = g_acc[ni];
                    srcs_a[ncl] = a_cur;
                    srcs_b[ncl] = gb_all[ni];
                    ++ncl;
                    accs[ncl] = u_acc[ni];
                    srcs_a[ncl] = a_cur;
                    srcs_b[ncl] = ub_all[ni];
                    ++ncl;
                }
                rocke_mfma_f8f6f4_agpr_cluster(b,
                                               accs,
                                               srcs_a,
                                               srcs_b,
                                               ncl,
                                               /*tail_nop=*/lv->asm_mfma_hazard_nop,
                                               /*inter_nop=*/0,
                                               /*convergent=*/true,
                                               outs);
                for(ni = 0; ni < nni; ++ni)
                {
                    g_acc[ni] = outs[2 * ni];
                    u_acc[ni] = outs[2 * ni + 1];
                }
            }
            else
            {
                for(ni = 0; ni < nni; ++ni)
                {
                    int pair = ni % 2;
                    rocke_value_t* gb;
                    rocke_value_t* ub;
                    if(ni + 1 < nni)
                    {
                        int npair = (ni + 1) % 2;
                        rocke_moe_fp8_dtla_stage_b_fp8(ctx,
                                                       WGate,
                                                       n_tile_bases[ni + 1],
                                                       kbase0,
                                                       K,
                                                       dtla->view,
                                                       2 * npair,
                                                       dtla->base,
                                                       dtla->lane,
                                                       dtla->wave_size);
                        rocke_moe_fp8_dtla_stage_b_fp8(ctx,
                                                       WUp,
                                                       n_tile_bases[ni + 1],
                                                       kbase0,
                                                       K,
                                                       dtla->view,
                                                       2 * npair + 1,
                                                       dtla->base,
                                                       dtla->lane,
                                                       dtla->wave_size);
                        rocke_b_s_waitcnt(b, dmas_per_stage, -1, -1);
                    }
                    else
                    {
                        rocke_b_s_waitcnt(b, 0, -1, -1);
                    }
                    if(a_cur == NULL)
                    {
                        a_cur = rocke_moe_fp8_xdtla_read_a_fp8(ctx,
                                                               dtla->view,
                                                               dtla->x_slot,
                                                               dtla->lane,
                                                               dtla->warp_row_base,
                                                               dtla->wave_size);
                    }
                    gb = rocke_moe_fp8_dtla_read_b_fp8(ctx,
                                                       dtla->view,
                                                       2 * pair,
                                                       dtla->lane,
                                                       dtla->warp_row_base,
                                                       dtla->wave_size);
                    ub = rocke_moe_fp8_dtla_read_b_fp8(ctx,
                                                       dtla->view,
                                                       2 * pair + 1,
                                                       dtla->lane,
                                                       dtla->warp_row_base,
                                                       dtla->wave_size);
                    g_acc[ni] = rocke_moe_fp8_emit_mfma(ctx, a_cur, gb, g_acc[ni]);
                    u_acc[ni] = rocke_moe_fp8_emit_mfma(ctx, a_cur, ub, u_acc[ni]);
                    rocke_moe_fp8_emit_sgb_gateup_dtla(ctx, /*n_mfma=*/2, cadence);
                }
            }
        }
        else
        {
            /* ---- legacy global->VGPR->MFMA path -------------------------- */
            rocke_value_t* a_cur;
            rocke_value_t* gb_cur[ROCKE_MOE_FP8_MAX_NNI];
            rocke_value_t* ub_cur[ROCKE_MOE_FP8_MAX_NNI];

            a_cur = fused_a_at(ctx, A, m_tile_base, k_group_base, c_atom_k, K, 0);
            for(ni = 0; ni < nni; ++ni)
            {
                gb_cur[ni] = fused_b_at(ctx, WGate, n_tile_bases, k_group_base, c_atom_k, K, ni, 0);
            }
            for(ni = 0; ni < nni; ++ni)
            {
                ub_cur[ni] = fused_b_at(ctx, WUp, n_tile_bases, k_group_base, c_atom_k, K, ni, 0);
            }

            for(kk = 0; kk < atoms_per_group; ++kk)
            {
                bool last = (kk + 1 >= atoms_per_group);
                rocke_value_t* a_next = NULL;
                if(!last)
                {
                    a_next = fused_a_at(ctx, A, m_tile_base, k_group_base, c_atom_k, K, kk + 1);
                }
                for(ni = 0; ni < nni; ++ni)
                {
                    rocke_value_t* gb_next_ni = NULL;
                    rocke_value_t* ub_next_ni = NULL;
                    if(!last)
                    {
                        gb_next_ni = fused_b_at(
                            ctx, WGate, n_tile_bases, k_group_base, c_atom_k, K, ni, kk + 1);
                        ub_next_ni = fused_b_at(
                            ctx, WUp, n_tile_bases, k_group_base, c_atom_k, K, ni, kk + 1);
                    }
                    g_acc[ni] = rocke_moe_fp8_emit_mfma(ctx, a_cur, gb_cur[ni], g_acc[ni]);
                    u_acc[ni] = rocke_moe_fp8_emit_mfma(ctx, a_cur, ub_cur[ni], u_acc[ni]);
                    if(!last)
                    {
                        gb_cur[ni] = gb_next_ni;
                        ub_cur[ni] = ub_next_ni;
                    }
                }
                if(!last)
                {
                    a_cur = a_next;
                }
            }
        }

        /* Fold each group accumulator (post-MFMA) by a_scale * b_scale. */
        {
            rocke_value_t* yld[ROCKE_MOE_FP8_MAX_ACCS];
            int ny = 0;
            rocke_value_t* new_gate[ROCKE_MOE_FP8_MAX_NNI];
            rocke_value_t* new_up[ROCKE_MOE_FP8_MAX_NNI];
            for(ni = 0; ni < nni; ++ni)
            {
                rocke_value_t* gvec = rocke_b_vector_splat(b, gate_ab[ni], atom->c_per_lane);
                rocke_value_t* uvec = rocke_b_vector_splat(b, up_ab[ni], atom->c_per_lane);
                new_gate[ni] = rocke_b_vector_fma(b, g_acc[ni], gvec, gate_outer[ni]);
                new_up[ni] = rocke_b_vector_fma(b, u_acc[ni], uvec, up_outer[ni]);
            }
            for(ni = 0; ni < nni; ++ni)
            {
                yld[ny++] = new_gate[ni];
            }
            for(ni = 0; ni < nni; ++ni)
            {
                yld[ny++] = new_up[ni];
            }
            rocke_b_scf_yield(b, yld, ny);
        }
    }
    rocke_b_region_leave(b);

    /* res = outer.results; return list(res[:nni]), list(res[nni:]). */
    for(ni = 0; ni < nni; ++ni)
    {
        if(out_gate != NULL)
        {
            out_gate[ni] = (outer.op != NULL) ? outer.op->results[ni] : NULL;
        }
        if(out_up != NULL)
        {
            out_up[ni] = (outer.op != NULL) ? outer.op->results[nni + ni] : NULL;
        }
    }
}
