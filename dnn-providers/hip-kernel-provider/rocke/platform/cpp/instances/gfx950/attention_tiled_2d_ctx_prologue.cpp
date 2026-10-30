// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_ctx_prologue.c --
 * per-lane row and accumulator helpers of the chunked C99 port of
 * rocke/instances/gfx950/attention_tiled_2d.py (arch gfx950, WIDE-K atoms).
 * The build prologue itself lives in the public entry glue.
 *
 * Scope (this part-file):
 *   - the pure per-lane row/bit closures _in_warp_row / _state_row / _bit2 /
 *       _select_lane_rg (lines 1270-1290).
 *   - the accumulator-index helpers _acc_idx / _acc_get / _acc_final_get
 *       (lines 1386 / 2505 / 3605).
 *
 * The Python closures capture the enclosing-function locals; here they read the
 * shared rocke_gfx950_attn2d_build_ctx_t. The per-warp lane-decomposition scalars
 * (lane_rg / lane_col / lane_col_mod4 / lane_col_div4 / lane_rg_is*) are ctx
 * fields filled by the (peer) emit_loop_bounds_and_inits phase; the row-map
 * closures reuse them when present and otherwise recompute from ctx->lane (the
 * recomputed IR is DCE-identical to Python's shared SSA values).
 *
 * Every IR node is arena-owned (ctx->b->arena).
 */

#include <stdbool.h>

#include "rocke/instance_gfx950_attention_tiled_2d_internal.h"

/* ===================================================================== *
 *  Per-lane row map + bit helpers (lines 1270-1290).
 *
 *  Reuse the cached per-warp lane-decomposition SSA values (ctx->lane_rg_v /
 *  ctx->lane_col_*_v), filled by the peer emit_loop_bounds_and_inits phase.
 *  Fall back to a fresh op only if a caller runs before that phase (the emitted
 *  IR is DCE-identical to Python's shared SSA values).
 * ===================================================================== */

static rocke_value_t* rocke_g950a2d_lane_rg(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return rocke_b_div(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 16));
}

rocke_value_t* rocke_gfx950_attn2d_in_warp_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r)
{
    /* atom_idx = r // 4 ; in_atom = r % 4
     * row = lane_rg*4 + (atom_idx*16 + in_atom) */
    int atom_idx = r / 4;
    int in_atom = r % 4;
    /* Python evaluates b.mul(lane_rg, const(4)) BEFORE the trailing const
     * (left-to-right arg order). Bind the mul to a temp so C's unspecified
     * arg-eval order does not allocate the trailing const ahead of the mul. */
    rocke_value_t* rg4
        = rocke_b_mul(ctx->b, rocke_g950a2d_lane_rg(ctx), rocke_b_const_i32(ctx->b, 4));
    return rocke_b_add(ctx->b, rg4, rocke_b_const_i32(ctx->b, atom_idx * 16 + in_atom));
}

rocke_value_t* rocke_gfx950_attn2d_state_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r)
{
    if(ctx->USE_MFMA_32X32 && !ctx->TRANSPOSED_QK_32X32)
        return rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row(ctx->b, ctx->lane, r);
    return rocke_gfx950_attn2d_in_warp_row(ctx, r);
}

rocke_value_t*
    rocke_gfx950_attn2d_bit2(rocke_gfx950_attn2d_build_ctx_t* ctx, rocke_value_t* v, int bit)
{
    /* Python: b.land(b.lshr(v, const(bit)), b.const_i32(1)) creates the lshr
     * (and its bit const) BEFORE the trailing const(1) (left-to-right). Bind the
     * lshr to a temp so C's right-to-left arg eval does not allocate const(1)
     * ahead of the lshr and shift the SSA ids. */
    rocke_value_t* sh = rocke_b_lshr(ctx->b, v, rocke_b_const_i32(ctx->b, bit));
    return rocke_b_land(ctx->b, sh, rocke_b_const_i32(ctx->b, 1));
}

rocke_value_t* rocke_gfx950_attn2d_select_lane_rg(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* v0,
                                                  rocke_value_t* v1,
                                                  rocke_value_t* v2,
                                                  rocke_value_t* v3)
{
    /* select(lane_rg_is0, v0, select(lane_rg_is1, v1, select(lane_rg_is2, v2, v3))).
     * Reuse the cached lane_rg_is* SSA values when present, else recompute. */
    rocke_value_t* is0 = ctx->lane_rg_is0_v;
    rocke_value_t* is1 = ctx->lane_rg_is1_v;
    rocke_value_t* is2 = ctx->lane_rg_is2_v;
    if(is0 == NULL || is1 == NULL || is2 == NULL)
    {
        rocke_value_t* lane_rg = rocke_g950a2d_lane_rg(ctx);
        is0 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 0));
        is1 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 1));
        is2 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 2));
    }
    return rocke_b_select(
        ctx->b, is0, v0, rocke_b_select(ctx->b, is1, v1, rocke_b_select(ctx->b, is2, v2, v3)));
}

/* ===================================================================== *
 *  Accumulator index helpers (lines 1386, 2505, 3605).
 * ===================================================================== */

int rocke_gfx950_attn2d_acc_idx(const rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    return n * ctx->ACC_M_ATOMS + atom;
}

rocke_value_t* rocke_gfx950_attn2d_acc_get(rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    /* Reads the unpacked loop-body acc carry (set per _emit_kv_body). */
    return ctx->acc_cur[rocke_gfx950_attn2d_acc_idx(ctx, n, atom)];
}

rocke_value_t*
    rocke_gfx950_attn2d_acc_final_get(rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    /* Reads the final loop-result acc (set by the epilogue read-back). */
    return ctx->acc_final[rocke_gfx950_attn2d_acc_idx(ctx, n, atom)];
}
