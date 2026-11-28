// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * The per-lane row and bit helpers and the accumulator-index helpers of the
 * gfx950 tiled 2D attention C++ port (library/kernels/gfx950/attention_tiled_2d.py):
 *   - the pure per-lane row/bit closures _in_warp_row / _state_row / _bit2 /
 *       _select_lane_rg (lines 1270-1290).
 *   - the accumulator-index helpers _acc_idx / _acc_get / _acc_final_get
 *       (lines 1386 / 2505 / 3605).
 *
 * The build prologue itself lives in the driver TU
 * (attention_tiled_2d_public_entry_glue.cpp, rocke_gfx950_attn2d_build_ctx_init).
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

#include "rocke/instance_attention_tiled_common_internal.h"
#include "rocke/instance_gfx950_attention_tiled_2d_internal.h"

/* ===================================================================== *
 *  Per-lane row map, register-P and accumulator-index helpers.
 *
 *  Thin wrappers over the shared TiledLaneMap helpers in
 *  instances/common/attention_tiled_common.cpp (one copy for gfx942 and
 *  gfx950); each fills the lane map from this ctx's cached SSA values.
 * ===================================================================== */

rocke_value_t* rocke_gfx950_attn2d_in_warp_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r)
{
    rocke_attn_tiled_lane_map_t lm = rocke_gfx950_attn2d_lane_map(ctx);
    return rocke_attn_tiled_in_warp_row(&lm, r);
}

rocke_value_t* rocke_gfx950_attn2d_state_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r)
{
    rocke_attn_tiled_lane_map_t lm = rocke_gfx950_attn2d_lane_map(ctx);
    return rocke_attn_tiled_state_row(&lm, r);
}

rocke_value_t*
    rocke_gfx950_attn2d_bit2(rocke_gfx950_attn2d_build_ctx_t* ctx, rocke_value_t* v, int bit)
{
    return rocke_attn_tiled_bit2(ctx->b, v, bit);
}

rocke_value_t* rocke_gfx950_attn2d_select_lane_rg(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* v0,
                                                  rocke_value_t* v1,
                                                  rocke_value_t* v2,
                                                  rocke_value_t* v3)
{
    rocke_attn_tiled_lane_map_t lm = rocke_gfx950_attn2d_lane_map(ctx);
    return rocke_attn_tiled_select_lane_rg(&lm, v0, v1, v2, v3);
}

/* ===================================================================== *
 *  Accumulator index helpers (lines 1386, 2505, 3605).
 * ===================================================================== */

int rocke_gfx950_attn2d_acc_idx(const rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    return rocke_attn_tiled_acc_idx(ctx->ACC_M_ATOMS, n, atom);
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
