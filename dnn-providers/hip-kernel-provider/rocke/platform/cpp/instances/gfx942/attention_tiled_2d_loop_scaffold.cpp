// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_loop_scaffold.c -- the KV-loop scaffolding
 * bucket of the C99 port of rocke.instances.gfx942.attention_tiled_2d.
 *
 * Ports the three pieces of the Python emitter that previously had no C
 * counterpart, which is why ``rocke_build_unified_attention_2d_tiled_scalar`` was
 * calling ``rocke_gfx942_attn2d_emit_kv_body`` with a NULL ``ctx->kv_tile_iv``
 * (the very first ``b.mul(kv_tile_iv, ...)`` then aborted the build with
 * "binop: NULL operand"):
 *
 *   1. rocke_gfx942_attn2d_emit_loop_bounds_and_inits
 *        max_seq_prefix_len -> tile_start / tile_end (Python 1982-2005) and the
 *        online-softmax m/l/acc carry inits + named iter_args (2113-2161).
 *   2. rocke_gfx942_attn2d_emit_preloop_prefetch
 *        _issue_k(tile_start, 0) (3591), the ("cur_buf", 0) carry append (3606-
 *        3607) and kv_step (3689).
 *   3. rocke_gfx942_attn2d_drive_kv_loop (defined in the kv_body_pv TU; declared
 *        in the internal header) builds the scf.for and runs the body inside it.
 *
 * Field names mirror the Python locals 1:1.
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"

#define B (ctx->b)

/* one_f / neg_inf / rcp_ln2 SSA constants (Python 1892-1895). These are
 * created ONCE in the prologue constants block and cached on ctx; reuse those
 * exact SSA values so we do not allocate duplicate consts (which would shift
 * every downstream %N). Fall back to a fresh op only if the prologue has not
 * run yet. */
static rocke_value_t* sc_neg_inf(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->neg_inf_v != NULL)
        return ctx->neg_inf_v;
    return rocke_b_const_f32(B, -INFINITY);
}
static rocke_value_t* sc_one_f(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->one_f_v != NULL)
        return ctx->one_f_v;
    return rocke_b_const_f32(B, 1.0);
}
static rocke_value_t* sc_rcp_ln2(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->rcp_ln2_v != NULL)
        return ctx->rcp_ln2_v;
    return rocke_b_const_f32(B, 1.4426950408889634);
}

/* _in_warp_row(r) (Python 2031-2037): the in-warp row for softmax state slot r
 * on the narrow (block_m_per_warp=16/32) layout. */
static rocke_value_t*
    sc_in_warp_row(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* lane_rg, int r)
{
    int atom_idx = r / 4;
    int in_atom = r % 4;
    /* Python evaluates b.mul(lane_rg, const(4)) BEFORE the trailing const
     * (left-to-right). Bind the mul to a temp so C arg-eval order does not
     * allocate the trailing const ahead of the mul and shift the %value. */
    rocke_value_t* rg4 = rocke_b_mul(B, lane_rg, rocke_b_const_i32(B, 4));
    return rocke_b_add(B, rg4, rocke_b_const_i32(B, atom_idx * 16 + in_atom));
}

/* _state_row(r) (Python 2039-2043). The 32x32 / transposed forms route through
 * _mfma_32x32_c_row; the narrow gfx942 build (the only buildable path here) uses
 * _in_warp_row. */
static rocke_value_t*
    sc_state_row(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* lane_rg, int r)
{
    /* USE_MFMA_32X32 narrow-only build: _in_warp_row. */
    return sc_in_warp_row(ctx, lane_rg, r);
}

/* ===================================================================== *
 *  tile_start / tile_end + online-softmax iter-arg inits (1982-2161).
 * ===================================================================== */
void rocke_gfx942_attn2d_emit_loop_bounds_and_inits(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx == NULL)
        return;

    const int T = ctx->T;
    const int BLOCK_M = ctx->BLOCK_M;
    const int NQK = ctx->NQK;
    const int SLIDING_WINDOW = ctx->SLIDING_WINDOW;
    const int NUM_QH = ctx->NUM_QH;
    const int SOFTMAX_STATE_SLOTS = ctx->SOFTMAX_STATE_SLOTS;

    /* ---- max_seq_prefix_len (1982-1984) ---- */
    int bm1_div_nqk = (BLOCK_M - 1) / NQK;
    /* Python emits the inner add before the outer const; sequence via a temp. */
    rocke_value_t* msp_inner = rocke_b_add(B, ctx->context_len, ctx->qb_start_pos);
    rocke_value_t* msp_raw = rocke_b_add(B, msp_inner, rocke_b_const_i32(B, bm1_div_nqk + 1));
    rocke_value_t* max_seq_prefix_len
        = rocke_b_select(B, rocke_b_cmp_lt(B, msp_raw, ctx->seq_len), msp_raw, ctx->seq_len);
    /* Cache for the KV-loop body (Python keeps one local; the body must reuse
     * this exact SSA value instead of recomputing it). */
    ctx->max_seq_prefix_len_v = max_seq_prefix_len;

    /* num_tiles = (max_seq_prefix_len + (T-1)) // T (1985). Inner add (and its
     * const) emitted before the outer divisor const, matching Python. */
    rocke_value_t* nt_inner = rocke_b_add(B, max_seq_prefix_len, rocke_b_const_i32(B, T - 1));
    rocke_value_t* num_tiles = rocke_b_div(B, nt_inner, rocke_b_const_i32(B, T));

    /* tile_start / tile_end (1987-2005). */
    if(SLIDING_WINDOW > 0)
    {
        /* Reuse the cached sw_const (line 1910), do NOT create a fresh const
         * here -- Python references the same SSA value. */
        rocke_value_t* sw_const = ctx->sw_const_v;
        rocke_value_t* qpos_hi_raw
            = rocke_b_add(B, ctx->qb_start_pos, rocke_b_const_i32(B, bm1_div_nqk));
        rocke_value_t* cur_q_minus1 = rocke_b_sub(B, ctx->cur_batch_q_len, rocke_b_const_i32(B, 1));
        rocke_value_t* qpos_hi = rocke_b_select(
            B, rocke_b_cmp_lt(B, qpos_hi_raw, cur_q_minus1), qpos_hi_raw, cur_q_minus1);
        /* Python evaluates the inner sub-chain (add then sub) BEFORE the
         * b.const_i32(1) literal (left-to-right arg order). C arg evaluation
         * is unspecified (typically right-to-left), which would allocate the
         * const first and shift the inner add/sub. Bind the sub-chain to a
         * temp first to force Python's value-creation order. */
        rocke_value_t* fak_inner
            = rocke_b_sub(B, rocke_b_add(B, ctx->context_len, ctx->qb_start_pos), sw_const);
        rocke_value_t* first_allowed_key = rocke_b_add(B, fak_inner, rocke_b_const_i32(B, 1));
        rocke_value_t* last_allowed_key = rocke_b_add(B, ctx->context_len, qpos_hi);
        rocke_value_t* tile_start_raw = rocke_b_div(B, first_allowed_key, rocke_b_const_i32(B, T));
        /* Python evaluates the cmp_lt (its const + the compare) BEFORE the
         * select's second const operand (left-to-right arg order). Bind the
         * predicate first so C does not allocate the second const ahead of the
         * compare under right-to-left arg evaluation. */
        rocke_value_t* ts_lt = rocke_b_cmp_lt(B, tile_start_raw, rocke_b_const_i32(B, 0));
        ctx->tile_start = rocke_b_select(B, ts_lt, rocke_b_const_i32(B, 0), tile_start_raw);
        /* Same left-to-right ordering: build the div BEFORE the +1 const, and
         * the cmp_lt BEFORE the select operands. */
        rocke_value_t* tile_end_div = rocke_b_div(B, last_allowed_key, rocke_b_const_i32(B, T));
        rocke_value_t* tile_end_raw = rocke_b_add(B, tile_end_div, rocke_b_const_i32(B, 1));
        rocke_value_t* te_lt = rocke_b_cmp_lt(B, tile_end_raw, num_tiles);
        ctx->tile_end = rocke_b_select(B, te_lt, tile_end_raw, num_tiles);
    }
    else
    {
        ctx->tile_start = rocke_b_const_i32(B, 0);
        ctx->tile_end = num_tiles;
    }

    /* ---- per-lane row map invariants (2014-2022) ----
     * Emit the full lane decomposition here, in Python source order, and cache
     * on ctx so every downstream consumer reuses the SAME SSA value (Python has
     * one local each; the C port previously recomputed them, which duplicated
     * the ops and broke byte-identity). */
    rocke_value_t* lane_rg = rocke_b_div(B, ctx->lane, rocke_b_const_i32(B, 16));
    ctx->lane_rg_v = lane_rg;
    ctx->lane_col_v = rocke_b_mod(B, ctx->lane, rocke_b_const_i32(B, 16));
    ctx->lane_half32_v = rocke_b_div(B, ctx->lane, rocke_b_const_i32(B, 32));
    ctx->lane_col32_v = rocke_b_mod(B, ctx->lane, rocke_b_const_i32(B, 32));
    ctx->lane_col_div4_v = rocke_b_div(B, ctx->lane_col_v, rocke_b_const_i32(B, 4));
    ctx->lane_col_mod4_v = rocke_b_mod(B, ctx->lane_col_v, rocke_b_const_i32(B, 4));
    ctx->lane_rg_is0_v = rocke_b_cmp_eq(B, lane_rg, rocke_b_const_i32(B, 0));
    ctx->lane_rg_is1_v = rocke_b_cmp_eq(B, lane_rg, rocke_b_const_i32(B, 1));
    ctx->lane_rg_is2_v = rocke_b_cmp_eq(B, lane_rg, rocke_b_const_i32(B, 2));

    /* ---- m_inits / l_inits (2113-2136) ---- */
    rocke_value_t* neg_inf = sc_neg_inf(ctx);
    rocke_value_t* one_f = sc_one_f(ctx);
    rocke_value_t* rcp_ln2 = sc_rcp_ln2(ctx);

    rocke_value_t* m_inits[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* l_inits[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];

    if(ctx->USE_SINKS)
    {
        /* Narrow gfx942 path (the transposed-scalar 32x32 branch is gfx950). */
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            rocke_value_t* row = rocke_b_add(B, ctx->wave_row_base, sc_state_row(ctx, lane_rg, r));
            /* qh = kv_head_idx*NQK + row%NQK (Python emits the mul before the
             * mod; sequence via temps to defeat C arg-eval order). */
            rocke_value_t* qh_mul = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, NQK));
            rocke_value_t* qh_mod = rocke_b_mod(B, row, rocke_b_const_i32(B, NQK));
            rocke_value_t* qh = rocke_b_add(B, qh_mul, qh_mod);
            rocke_value_t* qh_in = rocke_b_cmp_lt(B, qh, rocke_b_const_i32(B, NUM_QH));
            rocke_value_t* sink_h = rocke_b_global_load(B, ctx->sinks, qh, ctx->dtype, 2);
            rocke_value_t* sink_f = rocke_b_fmul(B, rocke_b_cast_to_f32(B, sink_h), rcp_ln2);
            m_inits[r] = rocke_b_select(B, qh_in, sink_f, neg_inf);
        }
    }
    else
    {
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            m_inits[r] = neg_inf;
    }
    for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        l_inits[r] = one_f;

    /* ---- acc inits (2138-2145) ---- *
     * Narrow path: one vec_f32(4) per (N-tile, M-atom). */
    const int PV32_N_TILES = ctx->HD / 32;
    const int ACC_N_TILES = ctx->USE_MFMA_32X32 ? PV32_N_TILES : ctx->PV_N_TILES;
    const int ACC_M_ATOMS = ctx->USE_MFMA_32X32 ? 1 : ctx->M_ATOMS_PER_WARP;
    rocke_value_t* acc_zero = rocke_b_zero_vec_f32(B, ctx->USE_MFMA_32X32 ? 16 : 4);

    ctx->ACC_N_TILES = ACC_N_TILES;
    ctx->ACC_M_ATOMS = ACC_M_ATOMS;

    /* ---- iter_args list build with names (2150-2161) ---- */
    int n_args = 0;
    for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
    {
        snprintf(ctx->iter_args_name_buf[n_args], sizeof(ctx->iter_args_name_buf[0]), "m%d", r);
        ctx->iter_args_names[n_args] = ctx->iter_args_name_buf[n_args];
        ctx->iter_args[n_args] = m_inits[r];
        ++n_args;
        snprintf(ctx->iter_args_name_buf[n_args], sizeof(ctx->iter_args_name_buf[0]), "l%d", r);
        ctx->iter_args_names[n_args] = ctx->iter_args_name_buf[n_args];
        ctx->iter_args[n_args] = l_inits[r];
        ++n_args;
    }
    ctx->ml_count = n_args; /* == 2 * SOFTMAX_STATE_SLOTS */

    for(int n = 0; n < ACC_N_TILES; ++n)
    {
        for(int atom = 0; atom < ACC_M_ATOMS; ++atom)
        {
            if(ACC_M_ATOMS > 1)
                snprintf(ctx->iter_args_name_buf[n_args],
                         sizeof(ctx->iter_args_name_buf[0]),
                         "acc%da%d",
                         n,
                         atom);
            else
                snprintf(ctx->iter_args_name_buf[n_args],
                         sizeof(ctx->iter_args_name_buf[0]),
                         "acc%d",
                         n);
            ctx->iter_args_names[n_args] = ctx->iter_args_name_buf[n_args];
            ctx->iter_args[n_args] = acc_zero;
            ++n_args;
        }
    }
    ctx->iter_args_count = n_args;
}

/* ===================================================================== *
 *  tile-0 K prefetch + ("cur_buf", 0) carry + kv_step (3591, 3606-3689).
 * ===================================================================== */
void rocke_gfx942_attn2d_emit_preloop_prefetch(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx == NULL)
        return;

    /* Prefetch tile_start's K into buffer 0 BEFORE the loop (3591). */
    rocke_gfx942_attn2d_issue_k(ctx, ctx->tile_start, rocke_b_const_i32(B, 0));

    /* cur_buf carry append (3606-3607). */
    int idx = ctx->iter_args_count;
    snprintf(ctx->iter_args_name_buf[idx], sizeof(ctx->iter_args_name_buf[0]), "cur_buf");
    ctx->iter_args_names[idx] = ctx->iter_args_name_buf[idx];
    ctx->iter_args[idx] = rocke_b_const_i32(B, 0);
    ctx->iter_args_count = idx + 1;

    /* kv_step (3689) is emitted by the caller AFTER emit_licm_hoist to keep the
     * const in Python emission order; see rocke_gfx942_attn2d_emit_kv_step. */
}

/* kv_step const (Python 3689); emitted after the LICM hoist. */
void rocke_gfx942_attn2d_emit_kv_step(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx == NULL)
        return;
    ctx->kv_step = rocke_b_const_i32(B, ctx->GROUPED_KV2 ? 2 : 1);
}

#undef B
