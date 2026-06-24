// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_q_load_gather.c --
 * Q STAGING + GATHER bucket of the chunked C99 port of
 * ck_dsl/instances/gfx950/attention_tiled_2d.py (arch gfx950, WIDE-K atoms).
 *
 * Implements five phase functions of the gfx950 wide-atom tiled-2D unified
 * attention builder. Each reads/writes ONLY ctx (and the builder it carries) and
 * emits IR builder calls in byte-identical Python order; peer closures are
 * reached through the shared internal header.
 *
 *   ckc_gfx950_attn2d_emit_q_load
 *       Cooperative Q[BLOCK_M, HD] global -> LDS vec8 stage with padding-row /
 *       out-of-range-head zero-fill (Python lines 1151-1215).
 *   ckc_gfx950_attn2d_emit_loop_bounds_and_inits
 *       max_seq_prefix_len -> tile_start / tile_end (Python 1220-1244), the
 *       per-warp lane decomposition SSA ctx->lane_*_v (1253-1261) and the
 *       online-softmax m/l/acc carry inits + named iter_args (1352-1400).
 *   ckc_gfx950_attn2d_emit_q_gather
 *       Per-lane Q -> VGPR MFMA-A-operand gather: the legacy 16x16 path
 *       (q_regs, n=8) and the 32x32x16 path (q32_regs), Python 2250-2363.
 *   ckc_gfx950_attn2d_emit_preloop_prefetch
 *       _issue_k(tile_start, 0) (2384) + ("cur_buf", 0) carry append (2400).
 *   ckc_gfx950_attn2d_emit_kv_step
 *       kv_step const (2482), emitted after the LICM hoist.
 *
 * gfx950 divergences from the gfx942 sibling captured here:
 *   - the legacy Q gather reads <8 x dtype> (n=8) and steps the K col by
 *     k*32 + lane_rg*8 (gfx942 uses n=4 and k*16 + lane_rg*4);
 *   - the 32x32 gather col is k*16 + lane_half*8 (Q32_FRAG is always 8, there is
 *     no use_mfma_32x32x8 variant), and there is NO Q_DIRECT_GLOBAL nor any
 *     FP8_NATIVE_QK path (FP8_NATIVE_QK is hardcoded False on gfx950);
 *   - the sink m_init has the transposed-scalar-state branch (row = wave_row_base
 *     + lane_col32) that gfx942 lacks.
 *
 * Every IR node is arena-owned.
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ckc/instance_gfx950_attention_tiled_2d_internal.h"
#include "ckc/error.hpp"
#include "ckc/ir.h"
#include "ckc/helper_ck_dsl.helpers.transforms.h"

#define B (ctx->b)

/* Latch the first Python ValueError/NotImplementedError onto the sticky-error
 * IRBuilder, mirroring the peer buckets' error model. */
/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing call sites' trailing
 * return valid -- it is simply never reached. */
[[noreturn]] static void ckc_q_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* msg)
{
    (void)b;
    ckc::raise_status(st, msg ? msg : "");
}

/* neg_inf / one_f / rcp_ln2 SSA constants (Python 1132-1135). Created ONCE in
 * the prologue constants block and cached on ctx; reuse those exact SSA values
 * so we do not allocate duplicate consts (which would shift every downstream
 * %N). Fall back to a fresh op only if the prologue has not run yet. */
static ckc_value_t* sc_neg_inf(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->neg_inf_v != NULL)
        return ctx->neg_inf_v;
    return ckc_b_const_f32(B, -INFINITY);
}
static ckc_value_t* sc_one_f(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->one_f_v != NULL)
        return ctx->one_f_v;
    return ckc_b_const_f32(B, 1.0);
}
static ckc_value_t* sc_rcp_ln2(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->rcp_ln2_v != NULL)
        return ctx->rcp_ln2_v;
    return ckc_b_const_f32(B, 1.4426950408889634);
}

/* ============================================================ *
 *  Q -> LDS cooperative stage (Python lines 1151-1215).
 * ============================================================ */
void ckc_gfx950_attn2d_emit_q_load(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_type_t* dtype;
    int Q_VECS_PER_ROW, Q_VECS_PER_THREAD;
    ckc_value_t* z8;
    int li;

    if(ctx == NULL)
        return;
    b     = ctx->b;
    dtype = ctx->dtype;

    /* z8 = b.zero_vec(dtype, 8) (Python line 1151). Built immediately before the
     * cooperative loop; emit it here to reproduce the exact IR order (it is not a
     * shared ctx field -- its sole consumer is the staged-store fallback below). */
    z8 = ckc_b_zero_vec(b, dtype, 8);

    /* ---------------- Q -> LDS (cooperative vec8 chunks) ----------------
     * Q_VECS_PER_ROW = HD // 8 ; Q_VECS_PER_THREAD = (BLOCK_M * Q_VECS_PER_ROW)
     * // THREADS. Both are host-only ints. */
    Q_VECS_PER_ROW    = ctx->HD / 8;
    Q_VECS_PER_THREAD = (ctx->BLOCK_M * Q_VECS_PER_ROW) / ctx->THREADS;

    /* q_desc = TensorDescriptor.naive("Q", lengths=[1<<30, NUM_QH, HD],
     *          coord_names=("token","head","dim")). Stored on ctx for the
     *          downstream Q gather + the epilogue output store. */
    {
        int lengths[3];
        static const char* const coord_names[3] = {"token", "head", "dim"};
        lengths[0]                              = 1 << 30;
        lengths[1]                              = ctx->NUM_QH;
        lengths[2]                              = ctx->HD;
        ctx->q_desc = ckc_tensor_descriptor_naive(b, "Q", lengths, 3, NULL, coord_names, 3);
    }

    for(li = 0; li < Q_VECS_PER_THREAD; ++li)
    {
        ckc_value_t* q_vid;
        ckc_value_t* Q_row;
        ckc_value_t* Q_col;
        ckc_value_t* q_pos_t;
        ckc_value_t* qh_t;
        ckc_value_t* qmask_t;
        ckc_value_t* q_pos_safe;
        ckc_value_t* qh_safe;
        ckc_value_t* q_off_base = NULL;
        ckc_value_t* v8;
        ckc_value_t* store_val;
        ckc_value_t* q_store_idx[3];
        int num_store_idx;

        /* q_vid = li*THREADS + tid */
        {
            ckc_value_t* q_li  = ckc_b_const_i32(b, (int64_t)li);
            ckc_value_t* q_thr = ckc_b_const_i32(b, (int64_t)ctx->THREADS);
            q_vid              = ckc_b_add(b, ckc_b_mul(b, q_li, q_thr), ctx->tid);
        }
        /* Q_row = q_vid // Q_VECS_PER_ROW */
        Q_row = ckc_b_div(b, q_vid, ckc_b_const_i32(b, (int64_t)Q_VECS_PER_ROW));
        /* Q_col = (q_vid % Q_VECS_PER_ROW) * 8. Python emits the inner mod (and
         * its const) before the outer const; sequence via a temp. */
        {
            ckc_value_t* q_mod = ckc_b_mod(b, q_vid, ckc_b_const_i32(b, (int64_t)Q_VECS_PER_ROW));
            Q_col              = ckc_b_mul(b, q_mod, ckc_b_const_i32(b, 8));
        }
        /* q_pos_t = qb_start_pos + Q_row // NQK */
        q_pos_t = ckc_b_add(
            b, ctx->qb_start_pos, ckc_b_div(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->NQK)));
        /* qh_t = kv_head_idx * NQK + Q_row % NQK (Python emits the mul before the
         * mod; sequence via temps to defeat C arg-eval order). */
        {
            ckc_value_t* qh_mul =
                ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, (int64_t)ctx->NQK));
            ckc_value_t* qh_mod = ckc_b_mod(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->NQK));
            qh_t                = ckc_b_add(b, qh_mul, qh_mod);
        }
        /* qmask_t = (q_pos_t < cur_batch_q_len) && (qh_t < NUM_QH) */
        {
            ckc_value_t* lt_pos = ckc_b_cmp_lt(b, q_pos_t, ctx->cur_batch_q_len);
            ckc_value_t* lt_qh  = ckc_b_cmp_lt(b, qh_t, ckc_b_const_i32(b, (int64_t)ctx->NUM_QH));
            qmask_t             = ckc_b_land(b, lt_pos, lt_qh);
        }
        /* q_pos_safe / qh_safe = select(qmask_t, ..., 0) */
        q_pos_safe = ckc_b_select(b, qmask_t, q_pos_t, ckc_b_const_i32(b, 0));
        qh_safe    = ckc_b_select(b, qmask_t, qh_t, ckc_b_const_i32(b, 0));

        /* q_off_base, _ = q_desc.offset(b, token=cu_q_start+q_pos_safe,
         *                               head=qh_safe, dim=0) */
        {
            static const char* const in_names[3] = {"token", "head", "dim"};
            ckc_value_t* in_values[3];
            in_values[0] = ckc_b_add(b, ctx->cu_q_start, q_pos_safe);
            in_values[1] = qh_safe;
            in_values[2] = ckc_b_const_i32(b, 0);
            ckc_transforms_descriptor_offset(
                b, ctx->q_desc, in_names, in_values, 3, &q_off_base, NULL);
        }

        /* v8 = b.global_load_vN(query, q_off_base + Q_col, dtype, 8, align=16) */
        v8 = ckc_b_global_load_vN(b, ctx->query, ckc_b_add(b, q_off_base, Q_col), dtype, 8, 16);

        /* Q store index: K_lds-aliased (buf,row_in_buf,col) or plain (row,col). */
        if(ctx->Q_ALIAS_K)
        {
            ckc_value_t* q_buf;
            ckc_value_t* q_row_in_buf;
            if(ctx->Q_USES_DUAL_SLOT)
            {
                q_buf        = ckc_b_div(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->T));
                q_row_in_buf = ckc_b_mod(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->T));
            }
            else
            {
                q_buf        = ckc_b_const_i32(b, 0);
                q_row_in_buf = Q_row;
            }
            q_store_idx[0] = q_buf;
            q_store_idx[1] = q_row_in_buf;
            q_store_idx[2] = Q_col;
            num_store_idx  = 3;
        }
        else
        {
            q_store_idx[0] = Q_row;
            q_store_idx[1] = Q_col;
            num_store_idx  = 2;
        }

        /* b.smem_store_vN(Q_lds, idx, vector_select(splat(qmask_t,8), v8, z8), 8) */
        store_val = ckc_b_vector_select(b, ckc_b_vector_splat(b, qmask_t, 8), v8, z8);
        ckc_b_smem_store_vN(b, ctx->Q_lds, q_store_idx, num_store_idx, store_val, 8);
    }
    ckc_b_sync(b);
}

/* ===================================================================== *
 *  KV tile-loop bounds + lane decomposition + online-softmax iter-arg inits
 *  (Python lines 1220-1244, 1253-1261, 1352-1400).
 * ===================================================================== */
void ckc_gfx950_attn2d_emit_loop_bounds_and_inits(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx == NULL)
        return;

    const int T                   = ctx->T;
    const int BLOCK_M             = ctx->BLOCK_M;
    const int NQK                 = ctx->NQK;
    const int SLIDING_WINDOW      = ctx->SLIDING_WINDOW;
    const int NUM_QH              = ctx->NUM_QH;
    const int SOFTMAX_STATE_SLOTS = ctx->SOFTMAX_STATE_SLOTS;

    /* ---- max_seq_prefix_len (1221-1223) ---- */
    int bm1_div_nqk = (BLOCK_M - 1) / NQK;
    /* Python emits the inner add before the outer const; sequence via a temp. */
    ckc_value_t* msp_inner = ckc_b_add(B, ctx->context_len, ctx->qb_start_pos);
    ckc_value_t* msp_raw   = ckc_b_add(B, msp_inner, ckc_b_const_i32(B, bm1_div_nqk + 1));
    ckc_value_t* max_seq_prefix_len =
        ckc_b_select(B, ckc_b_cmp_lt(B, msp_raw, ctx->seq_len), msp_raw, ctx->seq_len);
    /* Cache for the KV-loop body (Python keeps one local; the body must reuse
     * this exact SSA value instead of recomputing it). */
    ctx->max_seq_prefix_len_v = max_seq_prefix_len;

    /* num_tiles = (max_seq_prefix_len + (T-1)) // T (1224). Inner add (and its
     * const) emitted before the outer divisor const, matching Python. */
    ckc_value_t* nt_inner  = ckc_b_add(B, max_seq_prefix_len, ckc_b_const_i32(B, T - 1));
    ckc_value_t* num_tiles = ckc_b_div(B, nt_inner, ckc_b_const_i32(B, T));

    /* tile_start / tile_end (1226-1244). */
    if(SLIDING_WINDOW > 0)
    {
        /* Reuse the cached sw_const (line 1244 in the prologue), do NOT create a
         * fresh const here -- Python references the same SSA value. */
        ckc_value_t* sw_const    = ctx->sw_const_v;
        ckc_value_t* qpos_hi_raw = ckc_b_add(B, ctx->qb_start_pos, ckc_b_const_i32(B, bm1_div_nqk));
        ckc_value_t* cur_q_minus1 = ckc_b_sub(B, ctx->cur_batch_q_len, ckc_b_const_i32(B, 1));
        ckc_value_t* qpos_hi =
            ckc_b_select(B, ckc_b_cmp_lt(B, qpos_hi_raw, cur_q_minus1), qpos_hi_raw, cur_q_minus1);
        /* Python evaluates the inner sub-chain (add then sub) BEFORE the
         * b.const_i32(1) literal; bind to a temp first to force Python's
         * value-creation order under C's unspecified arg-eval order. */
        ckc_value_t* fak_inner =
            ckc_b_sub(B, ckc_b_add(B, ctx->context_len, ctx->qb_start_pos), sw_const);
        ckc_value_t* first_allowed_key = ckc_b_add(B, fak_inner, ckc_b_const_i32(B, 1));
        ckc_value_t* last_allowed_key  = ckc_b_add(B, ctx->context_len, qpos_hi);
        ckc_value_t* tile_start_raw    = ckc_b_div(B, first_allowed_key, ckc_b_const_i32(B, T));
        /* Bind the cmp_lt predicate first so C does not allocate the select's
         * second const ahead of the compare under right-to-left arg eval. */
        ckc_value_t* ts_lt = ckc_b_cmp_lt(B, tile_start_raw, ckc_b_const_i32(B, 0));
        ctx->tile_start    = ckc_b_select(B, ts_lt, ckc_b_const_i32(B, 0), tile_start_raw);
        /* Build the div BEFORE the +1 const, and the cmp_lt BEFORE the select. */
        ckc_value_t* tile_end_div = ckc_b_div(B, last_allowed_key, ckc_b_const_i32(B, T));
        ckc_value_t* tile_end_raw = ckc_b_add(B, tile_end_div, ckc_b_const_i32(B, 1));
        ckc_value_t* te_lt        = ckc_b_cmp_lt(B, tile_end_raw, num_tiles);
        ctx->tile_end             = ckc_b_select(B, te_lt, tile_end_raw, num_tiles);
    }
    else
    {
        ctx->tile_start = ckc_b_const_i32(B, 0);
        ctx->tile_end   = num_tiles;
    }

    /* ---- per-warp lane decomposition (1253-1261) ----
     * Emit the full lane decomposition here, in Python source order, and cache
     * on ctx so every downstream consumer reuses the SAME SSA value. */
    ckc_value_t* lane_rg = ckc_b_div(B, ctx->lane, ckc_b_const_i32(B, 16));
    ctx->lane_rg_v       = lane_rg;
    ctx->lane_col_v      = ckc_b_mod(B, ctx->lane, ckc_b_const_i32(B, 16));
    ctx->lane_half32_v   = ckc_b_div(B, ctx->lane, ckc_b_const_i32(B, 32));
    ctx->lane_col32_v    = ckc_b_mod(B, ctx->lane, ckc_b_const_i32(B, 32));
    ctx->lane_col_div4_v = ckc_b_div(B, ctx->lane_col_v, ckc_b_const_i32(B, 4));
    ctx->lane_col_mod4_v = ckc_b_mod(B, ctx->lane_col_v, ckc_b_const_i32(B, 4));
    ctx->lane_rg_is0_v   = ckc_b_cmp_eq(B, lane_rg, ckc_b_const_i32(B, 0));
    ctx->lane_rg_is1_v   = ckc_b_cmp_eq(B, lane_rg, ckc_b_const_i32(B, 1));
    ctx->lane_rg_is2_v   = ckc_b_cmp_eq(B, lane_rg, ckc_b_const_i32(B, 2));

    /* ---- m_inits / l_inits (1352-1375) ---- */
    ckc_value_t* neg_inf = sc_neg_inf(ctx);
    ckc_value_t* one_f   = sc_one_f(ctx);
    ckc_value_t* rcp_ln2 = sc_rcp_ln2(ctx);

    ckc_value_t* m_inits[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    ckc_value_t* l_inits[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];

    if(ctx->USE_SINKS)
    {
        if(ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32 && ctx->TRANSPOSED_SCALAR_STATE)
        {
            /* Transposed-scalar-state branch (1354-1362): a single state slot
             * whose row is wave_row_base + lane_col32. */
            ckc_value_t* row    = ckc_b_add(B, ctx->wave_row_base, ctx->lane_col32_v);
            ckc_value_t* qh_mul = ckc_b_mul(B, ctx->kv_head_idx, ckc_b_const_i32(B, NQK));
            ckc_value_t* qh_mod = ckc_b_mod(B, row, ckc_b_const_i32(B, NQK));
            ckc_value_t* qh     = ckc_b_add(B, qh_mul, qh_mod);
            ckc_value_t* qh_in  = ckc_b_cmp_lt(B, qh, ckc_b_const_i32(B, NUM_QH));
            ckc_value_t* sink_h = ckc_b_global_load(B, ctx->sinks, qh, ctx->dtype, 2);
            ckc_value_t* sink_f = ckc_b_fmul(B, ckc_b_cast_to_f32(B, sink_h), rcp_ln2);
            m_inits[0]          = ckc_b_select(B, qh_in, sink_f, neg_inf);
        }
        else
        {
            for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            {
                /* row = wave_row_base + _state_row(r) (peer closure). */
                ckc_value_t* row =
                    ckc_b_add(B, ctx->wave_row_base, ckc_gfx950_attn2d_state_row(ctx, r));
                ckc_value_t* qh_mul = ckc_b_mul(B, ctx->kv_head_idx, ckc_b_const_i32(B, NQK));
                ckc_value_t* qh_mod = ckc_b_mod(B, row, ckc_b_const_i32(B, NQK));
                ckc_value_t* qh     = ckc_b_add(B, qh_mul, qh_mod);
                ckc_value_t* qh_in  = ckc_b_cmp_lt(B, qh, ckc_b_const_i32(B, NUM_QH));
                ckc_value_t* sink_h = ckc_b_global_load(B, ctx->sinks, qh, ctx->dtype, 2);
                ckc_value_t* sink_f = ckc_b_fmul(B, ckc_b_cast_to_f32(B, sink_h), rcp_ln2);
                m_inits[r]          = ckc_b_select(B, qh_in, sink_f, neg_inf);
            }
        }
    }
    else
    {
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            m_inits[r] = neg_inf;
    }
    for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        l_inits[r] = one_f;

    /* ---- acc inits (1377-1384) ----
     * 32x32 path: one vec_f32(16) per 32-column N-tile, one M-atom per warp.
     * Narrow path: one vec_f32(4) per (N-tile, M-atom). */
    const int PV32_N_TILES = ctx->HD / 32;
    const int ACC_N_TILES  = ctx->USE_MFMA_32X32 ? PV32_N_TILES : ctx->PV_N_TILES;
    const int ACC_M_ATOMS  = ctx->USE_MFMA_32X32 ? 1 : ctx->M_ATOMS_PER_WARP;
    ckc_value_t* acc_zero  = ckc_b_zero_vec_f32(B, ctx->USE_MFMA_32X32 ? 16 : 4);

    ctx->ACC_N_TILES = ACC_N_TILES;
    ctx->ACC_M_ATOMS = ACC_M_ATOMS;

    /* ---- iter_args list build with names (1389-1400) ---- */
    int n_args = 0;
    for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
    {
        snprintf(ctx->iter_args_name_buf[n_args], sizeof(ctx->iter_args_name_buf[0]), "m%d", r);
        ctx->iter_args_names[n_args] = ctx->iter_args_name_buf[n_args];
        ctx->iter_args[n_args]       = m_inits[r];
        ++n_args;
        snprintf(ctx->iter_args_name_buf[n_args], sizeof(ctx->iter_args_name_buf[0]), "l%d", r);
        ctx->iter_args_names[n_args] = ctx->iter_args_name_buf[n_args];
        ctx->iter_args[n_args]       = l_inits[r];
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
            ctx->iter_args[n_args]       = acc_zero;
            ++n_args;
        }
    }
    ctx->iter_args_count = n_args;
}

/* ============================================================ *
 *  Per-lane Q -> VGPR gather (Python lines 2250-2363, excluding the trailing
 *  _issue_k(tile_start, 0) prefetch which the preloop bucket owns).
 * ============================================================ */
void ckc_gfx950_attn2d_emit_q_gather(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_type_t* dtype;
    /* lane_rg / lane_col: cached by emit_loop_bounds_and_inits (1253-1254). Fall
     * back to a fresh op only if that phase has not run yet. */
    ckc_value_t* lane_rg;
    ckc_value_t* lane_col;
    int atom, k;

    if(ctx == NULL)
        return;
    b     = ctx->b;
    dtype = ctx->dtype;

    lane_rg =
        (ctx->lane_rg_v != NULL) ? ctx->lane_rg_v : ckc_b_div(b, ctx->lane, ckc_b_const_i32(b, 16));
    lane_col = (ctx->lane_col_v != NULL) ? ctx->lane_col_v
                                         : ckc_b_mod(b, ctx->lane, ckc_b_const_i32(b, 16));

    ctx->q_regs_count   = 0;
    ctx->q32_regs_count = 0;

    /* Q_reg = [[None]*QK_K_ITERS for _ in range(M_ATOMS_PER_WARP)]
     * Flattened into ctx->q_regs in (atom, k) row-major order:
     *   ctx->q_regs[atom * QK_K_ITERS + k]. */
    ctx->q_regs_count = ctx->M_ATOMS_PER_WARP * ctx->QK_K_ITERS;

    /* if not (USE_MFMA_32X32 and SKIP_LEGACY_QREG): (line 2258) */
    if(!(ctx->USE_MFMA_32X32 && ctx->SKIP_LEGACY_QREG))
    {
        for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
        {
            ckc_value_t* q_row_atom;
            ckc_value_t* q_buf_atom        = NULL;
            ckc_value_t* q_row_in_buf_atom = NULL;

            /* q_row_atom = wave_row_base + (atom*16 + lane_col) */
            q_row_atom =
                ckc_b_add(b,
                          ctx->wave_row_base,
                          ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(atom * 16)), lane_col));

            if(ctx->Q_ALIAS_K)
            {
                if(ctx->Q_USES_DUAL_SLOT)
                {
                    q_buf_atom = ckc_b_div(b, q_row_atom, ckc_b_const_i32(b, (int64_t)ctx->T));
                    q_row_in_buf_atom =
                        ckc_b_mod(b, q_row_atom, ckc_b_const_i32(b, (int64_t)ctx->T));
                }
                else
                {
                    q_buf_atom        = ckc_b_const_i32(b, 0);
                    q_row_in_buf_atom = q_row_atom;
                }
            }

            for(k = 0; k < ctx->QK_K_ITERS; ++k)
            {
                ckc_value_t* q_col_off;
                ckc_value_t* idx[3];
                int num_idx;
                ckc_value_t* qreg;

                /* q_col_off = k*32 + lane_rg*8. Python evaluates the left
                 * const(k*32) BEFORE the mul (left-to-right); bind it to a temp
                 * so C's arg-eval order does not allocate the mul's const ahead
                 * of it and shift the mul's %value. */
                ckc_value_t* q_col_base = ckc_b_const_i32(b, (int64_t)(k * 32));
                q_col_off = ckc_b_add(b, q_col_base, ckc_b_mul(b, lane_rg, ckc_b_const_i32(b, 8)));

                if(ctx->Q_ALIAS_K)
                {
                    idx[0]  = q_buf_atom;
                    idx[1]  = q_row_in_buf_atom;
                    idx[2]  = q_col_off;
                    num_idx = 3;
                }
                else
                {
                    idx[0]  = q_row_atom;
                    idx[1]  = q_col_off;
                    num_idx = 2;
                }

                /* Q_reg[atom][k] = b.smem_load_vN(Q_lds, *idx, dtype=dtype, n=8) */
                qreg = ckc_b_smem_load_vN(b, ctx->Q_lds, idx, num_idx, dtype, 8);
                ctx->q_regs[atom * ctx->QK_K_ITERS + k] = qreg;
            }
        }

        if(ctx->Q_ALIAS_K)
        {
            /* Drain the per-lane Q-gather LDS reads before the K[0] async write
             * to the same K_lds[0] slot (2289-2290). */
            ckc_b_s_waitcnt(b, -1, 0, -1);
            ckc_b_sync(b);
        }
    }

    /* ---- Per-lane Q gather for CK Tile/Triton 32x32x16 QK geometry ----
     * (2314-2363). */
    if(ctx->USE_MFMA_32X32)
    {
        ckc_value_t* lane_half;
        ckc_value_t* lane_col32;
        ckc_value_t* q32_row;
        ckc_value_t* q32_buf        = NULL;
        ckc_value_t* q32_row_in_buf = NULL;

        if(ctx->FP8_MFMA_PV)
        {
            /* raise NotImplementedError(...) (2322-2326) */
            ckc_q_set_err(b,
                          CKC_ERR_NOTIMPL,
                          "32x32x16 PV needs bf16 V in LDS; disable "
                          "use_fp8_mfma_pv (it is broken / slower anyway) for "
                          "the fp8 combo");
            return;
        }

        /* Q32_reg = [None] * QK_K_ITERS -> ctx->q32_regs (count QK_K_ITERS). */
        ctx->q32_regs_count = ctx->QK_K_ITERS;

        /* lane_half / lane_col32 are RE-derived locally here in Python
         * (2328-2329) with the same div/mod constants the cached fields use.
         * Python's reassignment shadows the earlier locals, so every later use
         * (kv_body QK / transposed softmax / PV / epilogue) reads THESE freshly
         * created SSA values -- update the cached ctx fields to match. */
        lane_half  = ckc_b_div(b, ctx->lane, ckc_b_const_i32(b, 32));
        lane_col32 = ckc_b_mod(b, ctx->lane, ckc_b_const_i32(b, 32));
        /* Python reassigns ``lane_col32`` (1256 -> 2329), shadowing the preloop
         * value, so all later uses read THIS one. ``lane_half32`` (1255) is a
         * DIFFERENT name that is NOT reassigned, so leave lane_half32_v alone.
         * ``lane_half`` (2328) is a separate local read by the non-transposed
         * 32x32 QK; cache it for the kv_body. */
        ctx->lane_col32_v   = lane_col32;
        ctx->lane_half_qg_v = lane_half;
        q32_row             = ckc_b_add(b, ctx->wave_row_base, lane_col32);

        if(ctx->Q_ALIAS_K)
        {
            if(ctx->Q_USES_DUAL_SLOT)
            {
                q32_buf        = ckc_b_div(b, q32_row, ckc_b_const_i32(b, (int64_t)ctx->T));
                q32_row_in_buf = ckc_b_mod(b, q32_row, ckc_b_const_i32(b, (int64_t)ctx->T));
            }
            else
            {
                q32_buf        = ckc_b_const_i32(b, 0);
                q32_row_in_buf = q32_row;
            }
        }

        for(k = 0; k < ctx->QK_K_ITERS; ++k)
        {
            ckc_value_t* q32_col;
            ckc_value_t* q32;
            ckc_value_t* idx[3];
            int num_idx;

            /* q32_col = k*16 + lane_half*8. Python evaluates the left const
             * BEFORE the mul; bind it to a temp first. */
            ckc_value_t* q32_col_base = ckc_b_const_i32(b, (int64_t)(k * 16));
            q32_col = ckc_b_add(b, q32_col_base, ckc_b_mul(b, lane_half, ckc_b_const_i32(b, 8)));

            if(ctx->Q_ALIAS_K)
            {
                idx[0]  = q32_buf;
                idx[1]  = q32_row_in_buf;
                idx[2]  = q32_col;
                num_idx = 3;
            }
            else
            {
                idx[0]  = q32_row;
                idx[1]  = q32_col;
                num_idx = 2;
            }
            /* q32 = b.smem_load_vN(Q_lds, *idx, dtype=dtype, n=8). FP8_NATIVE_QK
             * is hardcoded False on gfx950, so no Q-quantize branch follows. */
            q32              = ckc_b_smem_load_vN(b, ctx->Q_lds, idx, num_idx, dtype, 8);
            ctx->q32_regs[k] = q32;
        }

        if(ctx->Q_ALIAS_K)
        {
            /* Drain the 32x32 Q reads before the K[0] prefetch overwrites the
             * aliased LDS slabs (2362-2363). */
            ckc_b_s_waitcnt(b, -1, 0, -1);
            ckc_b_sync(b);
        }
    }

    /* The trailing _issue_k(tile_start, b.const_i32(0)) prefetch (line 2384)
     * belongs to the preloop bucket and is emitted there, not here. */
}

/* ===================================================================== *
 *  tile-0 K prefetch + ("cur_buf", 0) carry append (Python 2384, 2399-2400).
 * ===================================================================== */
void ckc_gfx950_attn2d_emit_preloop_prefetch(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx == NULL)
        return;

    /* Prefetch tile_start's K into buffer 0 BEFORE the loop (2384). */
    ckc_gfx950_attn2d_issue_k(ctx, ctx->tile_start, ckc_b_const_i32(B, 0));

    /* cur_buf_init = const_i32(0); iter_args.append(("cur_buf", cur_buf_init))
     * (2399-2400). */
    int idx = ctx->iter_args_count;
    snprintf(ctx->iter_args_name_buf[idx], sizeof(ctx->iter_args_name_buf[0]), "cur_buf");
    ctx->iter_args_names[idx] = ctx->iter_args_name_buf[idx];
    ctx->iter_args[idx]       = ckc_b_const_i32(B, 0);
    ctx->iter_args_count      = idx + 1;

    /* kv_step (2482) is emitted by the caller AFTER emit_licm_hoist to keep the
     * const in Python emission order; see ckc_gfx950_attn2d_emit_kv_step. */
}

/* kv_step const (Python 2482); emitted after the LICM hoist. */
void ckc_gfx950_attn2d_emit_kv_step(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx == NULL)
        return;
    ctx->kv_step = ckc_b_const_i32(B, ctx->GROUPED_KV2 ? 2 : 1);
}

#undef B
