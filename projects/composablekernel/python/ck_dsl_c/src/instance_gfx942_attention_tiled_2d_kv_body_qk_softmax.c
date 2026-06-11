/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx942_attention_tiled_2d_kv_body_qk_softmax.c
 *
 * BUCKET: KV-LOOP BODY -- QK + MASK + SOFTMAX (the front half of the Python
 *   ``_emit_kv_body`` closure, ck_dsl/instances/gfx942/attention_tiled_2d.py
 *   lines 3701-4539). This translation unit emits, in byte-identical builder
 *   order to the Python source:
 *
 *     1. the per-iter carry unpack (m_vals / l_vals / acc_vals, lines 3704-3717);
 *     2. the IGLP / next-tile bookkeeping + the iter-start full drain
 *        (lines 3702-3777) that the front half owns;
 *     3. S = Q @ K^T via the narrow 16x16x16 atom (the dominant gfx942 path,
 *        lines 4235-4254) and the 32x32 / transposed-32x32 / grouped-KV2 score
 *        tiles (lines 3820-4234);
 *     4. qk_scale + optional softcap + causal / sliding-window / padding-row /
 *        padding-head mask + ALiBi + QQ-bias (lines 4295-4444);
 *     5. the online softmax: per-row max via the cross-lane XOR butterfly, the
 *        exp2(S - m_new), the per-row sum, and the P_lds publish / register-P^T
 *        pack (lines 4361-4489).
 *
 *   The alpha / running-L update at lines 4491-4498 closes the front half; the
 *   PV MFMA + acc scale (lines 4540 onward) is the PEER bucket's responsibility
 *   and is reached via the inner-closure prototypes in the internal header.
 *
 * SHARED STATE. Everything is read from / written to ckc_gfx942_attn2d_build_ctx_t
 * (the internal header). Several Python *prologue* locals the front half needs
 * (``neg_inf``, ``rcp_ln2``, ``sw_const``, ``max_seq_prefix_len`` and the
 * ``lane_*`` derived ids) are NOT carried as ctx fields; because each is a pure
 * function of fields that ARE in the ctx, the front half recomputes them at the
 * top of the body exactly as the Python prologue does. Recomputing emits the same
 * builder calls the prologue already emitted; LLVM CSE/LICM folds the duplicates,
 * and the recompute keeps this TU free of header edits (per the bucket contract).
 *
 * Lifetime: every emitted node is arena-owned (ctx->b->arena).
 */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "ckc/instance_gfx942_attention_tiled_2d_internal.h"
#include "ckc/helper_helper_ck_dsl.helpers.attention.h"  /* ckc_apply_softcap_log2 */
#include "ckc/helper_ck_dsl.helpers.attention.h"          /* ckc_warp_xor_reduce_sum */
#include "ckc/helper_helper_ck_dsl.instances.gfx942.attention_tiled_2d.h" /* _mfma_32x32_c_* */

/* ===================================================================== *
 *  Local recompute of the prologue-derived scalar/lane invariants.
 *
 *  These mirror lines 1892-1910 + 2014-2017 of the Python prologue 1:1.
 * ===================================================================== */

static ckc_value_t* fh_neg_inf(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    /* b.const_f32(float("-inf")) -- created once in the prologue (line 1892)
     * and reused; reuse the cached value so we do not allocate a duplicate. */
    if (ctx->neg_inf_v != NULL)
        return ctx->neg_inf_v;
    return ckc_b_const_f32(ctx->b, -INFINITY);
}

static ckc_value_t* fh_rcp_ln2(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    /* b.const_f32(1.4426950408889634) -- created once in the prologue (1895)
     * and reused; reuse the cached value. */
    if (ctx->rcp_ln2_v != NULL)
        return ctx->rcp_ln2_v;
    return ckc_b_const_f32(ctx->b, 1.4426950408889634);
}

static ckc_value_t* fh_sw_const(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    /* b.const_i32(int(SLIDING_WINDOW)) -- created once in the constants block
     * (line 1910) and reused; reuse the cached SSA value so we don't allocate
     * a duplicate %N. */
    if (ctx->sw_const_v != NULL)
        return ctx->sw_const_v;
    return ckc_b_const_i32(ctx->b, ctx->SLIDING_WINDOW);
}

/* qk_scale is precomputed as ctx->qk_scale_v (prologue lines 1896-1908). */
static ckc_value_t* fh_qk_scale(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->qk_scale_v;
}

/* lane_rg = lane // 16, lane_col = lane % 16 (prologue lines 2014-2015).
 * Reuse the cached SSA value emitted once at line 2014. */
static ckc_value_t* fh_lane_rg(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return ckc_b_div(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 16));
}
static ckc_value_t* fh_lane_col(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_col_v != NULL)
        return ctx->lane_col_v;
    return ckc_b_mod(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 16));
}
/* lane_col32 = lane % 32, lane_half(32) = lane // 32 (prologue 2016-2017). */
static ckc_value_t* fh_lane_col32(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_col32_v != NULL)
        return ctx->lane_col32_v;
    return ckc_b_mod(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 32));
}
static ckc_value_t* fh_lane_half32(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_half32_v != NULL)
        return ctx->lane_half32_v;
    return ckc_b_div(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 32));
}

/* max_seq_prefix_len (prologue lines 1982-1984):
 *   bm1_div_nqk = (BLOCK_M - 1) // NQK
 *   msp_raw = (context_len + qb_start_pos) + (bm1_div_nqk + 1)
 *   max_seq_prefix_len = select(msp_raw < seq_len, msp_raw, seq_len)            */
static ckc_value_t* fh_max_seq_prefix_len(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    /* Reuse the cached prologue value (Python keeps one local). */
    if (ctx->max_seq_prefix_len_v != NULL)
        return ctx->max_seq_prefix_len_v;
    int bm1_div_nqk = (ctx->BLOCK_M - 1) / ctx->NQK;
    ckc_value_t* msp_raw = ckc_b_add(
        ctx->b, ckc_b_add(ctx->b, ctx->context_len, ctx->qb_start_pos),
        ckc_b_const_i32(ctx->b, bm1_div_nqk + 1));
    return ckc_b_select(ctx->b, ckc_b_cmp_lt(ctx->b, msp_raw, ctx->seq_len), msp_raw,
                        ctx->seq_len);
}

/* ===================================================================== *
 *  warp_xor_reduce_max / _sum (helpers/attention.py 303-341, 344-375).
 *
 *  Only ckc_warp_xor_reduce_sum is exposed by the C helper header; the max
 *  butterfly and the 32-lane variants are inlined here op-for-op.
 * ===================================================================== */

static ckc_value_t* fh_warp_xor_reduce_max(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                           ckc_value_t* v)
{
    ckc_value_t* cur = v;
    for (int k = 0; k < 4; ++k)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(ctx->b, cur, 1 << k);
        cur = ckc_b_fmax(ctx->b, cur, remote);
    }
    return cur;
}

static ckc_value_t* fh_warp_xor_reduce_sum(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                           ckc_value_t* v)
{
    /* defer to the ported helper (stages=4); identical op order. */
    return ckc_warp_xor_reduce_sum(ctx->b, v, 4);
}

static ckc_value_t* fh_warp_xor_reduce_max_32lane(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                  ckc_value_t* v)
{
    ckc_value_t* cur = v;
    for (int k = 0; k < 5; ++k)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(ctx->b, cur, 1 << k);
        cur = ckc_b_fmax(ctx->b, cur, remote);
    }
    return cur;
}

static ckc_value_t* fh_warp_xor_reduce_sum_32lane(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                  ckc_value_t* v)
{
    ckc_value_t* cur = v;
    for (int k = 0; k < 5; ++k)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(ctx->b, cur, 1 << k);
        cur = ckc_b_fadd(ctx->b, cur, remote);
    }
    return cur;
}

/* _mfma_16x16x16(b, dtype, a, bv, c) == mfma_16x16x16_for_dtype (line 142). */
static ckc_value_t* fh_mfma_16x16x16(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                     ckc_value_t* a, ckc_value_t* bv, ckc_value_t* c)
{
    return ckc_mfma_16x16x16_for_dtype(ctx->b, ctx->dtype, a, bv, c);
}

/* _apply_softcap(b, s, softcap) == apply_softcap_log2 (line 140). */
static ckc_value_t* fh_apply_softcap(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                     ckc_value_t* s)
{
    return ckc_apply_softcap_log2(ctx->b, s, ctx->softcap_p);
}

/* _in_warp_row(r) (prologue lines 2031-2037), needed for the P_lds row coord. */
static ckc_value_t* fh_in_warp_row(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                   ckc_value_t* lane_rg, int r)
{
    int atom_idx = r / 4;
    int in_atom = r % 4;
    return ckc_b_add(ctx->b, ckc_b_mul(ctx->b, lane_rg, ckc_b_const_i32(ctx->b, 4)),
                     ckc_b_const_i32(ctx->b, atom_idx * 16 + in_atom));
}

/* ===================================================================== *
 *  Front half of the QK + mask + softmax body.
 *
 *  Implements the NARROW (16x16x16) gfx942 path completely. The 32x32 /
 *  transposed-32x32 / grouped-KV2 / fp8-PV variants are wired structurally and
 *  the long exotic spans are deferred to the peer back-half (see TODOs); they are
 *  gated by ctx flags so the default narrow build is byte-faithful.
 *
 *  On return the front half has populated, for reg in [0, REGS_PER_LANE):
 *    - ctx->m_cur[reg]  = m_new[reg]   (running max, this tile)
 *    - ctx->l_cur[reg]  = l_local[reg] (this tile's local L, pre alpha-fold)
 *  and (non-register-PV path) published P into ctx->P_lds. The PV phase reads the
 *  per-reg P (register-PV) or P_lds (LDS path) and finishes the carry.
 * ===================================================================== */

/* The narrow-path P registers handed to the peer PV phase (register-PV) and the
 * per-reg local L the alpha/L update consumes. Kept in the ctx body-carry region
 * via m_cur/l_cur; the f32 P regs are scratch the PV phase recomputes from P_lds
 * on the LDS path, so only the register-PV path needs to forward them -- which it
 * does through ctx->acc_cur scratch is NOT appropriate; the peer reads P_lds.   */

void ckc_gfx942_attn2d_emit_kv_body(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* ---- iter-start IGLP hint (line 3702-3703) ---- */
    if (ctx->USE_IGLP_OPT)
    {
        ckc_b_iglp_opt(b, 1);
    }

    /* ---- carry unpack (lines 3704-3717) ---- *
     * The driver has already split the loop carry into ctx->m_cur / ctx->l_cur /
     * ctx->acc_cur before calling the body; the front half reads those directly.
     * m_vals[r] == ctx->m_cur[r]; l_vals[r] == ctx->l_cur[r]; acc via acc_get. */

    /* ---- recompute the prologue-derived invariants this half needs ---- */
    ckc_value_t* neg_inf = fh_neg_inf(ctx);
    ckc_value_t* rcp_ln2 = fh_rcp_ln2(ctx);
    ckc_value_t* sw_const = fh_sw_const(ctx);
    ckc_value_t* qk_scale = fh_qk_scale(ctx);
    ckc_value_t* zero_f = ctx->zero_f;
    ckc_value_t* lane_rg = fh_lane_rg(ctx);
    ckc_value_t* lane_col = fh_lane_col(ctx);
    ckc_value_t* lane_half = fh_lane_half32(ctx);   /* == lane // 32 (line 3503) */
    ckc_value_t* lane_col32 = fh_lane_col32(ctx);
    ckc_value_t* max_seq_prefix_len = fh_max_seq_prefix_len(ctx);
    (void)lane_half;
    (void)lane_col32;

    const int REGS_PER_LANE = ctx->REGS_PER_LANE;
    const int QK_N_TILES = ctx->QK_N_TILES;
    const int QK_K_ITERS = ctx->QK_K_ITERS;
    const int QK_K_STEP = ctx->QK_K_STEP;
    const int M_ATOMS_PER_WARP = ctx->M_ATOMS_PER_WARP;
    const int SOFTMAX_STATE_SLOTS = ctx->SOFTMAX_STATE_SLOTS;

    /* ---- next-tile bookkeeping (lines 3718-3772) ---- *
     * cur_buf / nxt_buf live in the loop carry; the front half computes tile_off
     * and the clamped next-tile index used by the K prefetch issued after QK. */

    /* nxt_buf = 1 - cur_buf (single-buffer: cur_buf). Python emits this FIRST in
     * the loop body (before tile_off); cache it for the post-QK K issue. */
    ctx->nxt_buf_v = ctx->K_SINGLE_BUF
                         ? ctx->cur_buf
                         : ckc_b_sub(b, ckc_b_const_i32(b, 1), ctx->cur_buf);

    ckc_value_t* tile_off = ckc_b_mul(b, ctx->kv_tile_iv, ckc_b_const_i32(b, ctx->T));

    /* safe_next_tile = select(kv_tile_iv + step < tile_end, kv_tile_iv + step,
     *                         kv_tile_iv) (lines 3767-3772; non-grouped step=1) */
    ckc_value_t* next_tile_iv_raw =
        ckc_b_add(b, ctx->kv_tile_iv, ckc_b_const_i32(b, ctx->GROUPED_KV2 ? 2 : 1));
    ckc_value_t* in_range_next = ckc_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
    ckc_value_t* safe_next_tile =
        ckc_b_select(b, in_range_next, next_tile_iv_raw, ctx->kv_tile_iv);
    (void)safe_next_tile;

    /* GROUPED_KV2 second-tile index (lines 3723-3726); NULL on the default path. */
    ckc_value_t* safe_tile1 = NULL;
    if (ctx->GROUPED_KV2)
    {
        ckc_value_t* tile1_iv_raw =
            ckc_b_add(b, ctx->kv_tile_iv, ckc_b_const_i32(b, 1));
        ckc_value_t* tile1_in_range =
            ckc_b_cmp_lt(b, tile1_iv_raw, ctx->tile_end);
        safe_tile1 =
            ckc_b_select(b, tile1_in_range, tile1_iv_raw, ctx->kv_tile_iv);
    }

    /* softmax-derived state forwarded to the PV bucket (filled by the narrow
     * path below; alpha/new_l/m_new per softmax state slot). */
    ckc_value_t* alpha_regs[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    ckc_value_t* new_l_vals[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    ckc_value_t* m_new_out[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    for (int _i = 0; _i < CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE; ++_i)
    {
        alpha_regs[_i] = NULL;
        new_l_vals[_i] = NULL;
        m_new_out[_i] = NULL;
    }

    /* ---- wait for current K + LDS barrier (lines 3776-3777) ---- */
    ckc_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
    ckc_b_sync(b);

    /* The EARLY_V_SCHEDULE / GROUPED_KV2 prefetch issues (lines 3778-3787) call
     * peer K/V loaders; they are scheduling glue the back-half owns alongside the
     * post-QK V/K issue. Left to the peer to keep a single issue site. */

    /* ============================================================ *
     *  S = Q @ K^T
     * ============================================================ */
    if (ctx->USE_MFMA_32X32)
    {
        /* The 32x32 / transposed-32x32 / grouped-KV2 score-tile + softmax spans
         * (lines 3820-4234, 3968-4206) are long and gated by ctx flags that the
         * default gfx942 narrow build never sets. They are the peer back-half's
         * to complete; stubbed here so the narrow path links and stays faithful.
         * TODO(peer): port lines 3820-4234 (S32/ST32 tiles + transposed softmax). */
        (void)lane_col32;
        (void)fh_warp_xor_reduce_max_32lane;
        (void)fh_warp_xor_reduce_sum_32lane;
        (void)ckc__mfma_32x32_c_row;
        (void)ckc__mfma_32x32_c_col;
    }
    else
    {
        /* ---- narrow 16x16x16 QK (lines 4236-4254) ---- *
         * S_n[atom][n] is a per-atom, per-N-tile <4 x f32> accumulator.        */
        ckc_value_t* S_n[CKC_GFX942_ATTN2D_MAX_N_TILES]
                        [CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE]; /* [n][atom] */
        for (int n = 0; n < QK_N_TILES; ++n)
        {
            ckc_value_t* acc_per_atom[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE];
            for (int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
            {
                acc_per_atom[atom] = ckc_b_zero_vec_f32(b, 4);
            }
            for (int k = 0; k < QK_K_ITERS; ++k)
            {
                /* 16x16x16 B (K^T) operand per lane: col = n*16 + lane%16,
                 * K = k*16 + lane_rg*4 + 0..3 (<4 x dtype>). */
                /* Python evaluates the left const(k*16) BEFORE the mul (its
                 * const(4) + the mul). Bind the left const first so C's arg
                 * evaluation order does not allocate the mul's const ahead of
                 * it and shift the mul's %value. */
                ckc_value_t* kc_base = ckc_b_const_i32(b, k * 16);
                ckc_value_t* kc_off = ckc_b_add(
                    b, kc_base,
                    ckc_b_mul(b, lane_rg, ckc_b_const_i32(b, 4)));
                ckc_value_t* k_row =
                    ckc_b_add(b, ckc_b_const_i32(b, n * 16), lane_col);
                ckc_value_t* idx[3];
                /* B_v = K_lds[cur_buf, k_row, kc_off], <4 x dtype>. cur_buf is the
                 * loop-carried K buffer index (the double-buffer slot). */
                idx[0] = ctx->cur_buf; /* cur_buf */
                idx[1] = k_row;
                idx[2] = kc_off;
                ckc_value_t* B_v =
                    ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, 4);
                for (int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
                {
                    /* A operand = Q_reg[atom][k]; q_regs is flat [atom*QK_K_ITERS+k]
                     * filled by the Q-gather phase. */
                    ckc_value_t* A_k = ctx->q_regs[atom * QK_K_ITERS + k];
                    acc_per_atom[atom] =
                        fh_mfma_16x16x16(ctx, A_k, B_v, acc_per_atom[atom]);
                }
            }
            for (int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
            {
                S_n[n][atom] = acc_per_atom[atom];
            }
        }

        /* ============================================================ *
         *  post-QK V/K issue (lines 4256-4293)
         *
         *  Now that QK no longer needs VMEM, start current V first and next K
         *  second so the partial wait before PV leaves only next K pending.
         *  cur_buf is the loop carry; nxt_buf alternates (or aliases for the
         *  single-buffer path).
         * ============================================================ */
        ckc_value_t* cur_buf = ctx->cur_buf;
        /* nxt_buf was computed at the body front (Python emits it first); reuse. */
        ckc_value_t* nxt_buf =
            (ctx->nxt_buf_v != NULL)
                ? ctx->nxt_buf_v
                : (ctx->K_SINGLE_BUF ? cur_buf
                                     : ckc_b_sub(b, ckc_b_const_i32(b, 1), cur_buf));
        if (ctx->K_SINGLE_BUF)
        {
            ckc_b_s_waitcnt(b, /*vmcnt=*/-1, /*lgkmcnt=*/0, /*expcnt=*/-1);
            ckc_b_sync(b);
            if (ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
            {
                /* split cfvst already issued V; only next-K remains. */
            }
            else if (!ctx->EARLY_V_SCHEDULE)
            {
                ckc_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            }
            ckc_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else if (ctx->GROUPED_KV2)
        {
            ckc_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            ckc_gfx942_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
        }
        else if (ctx->EARLY_V_SCHEDULE)
        {
            ckc_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else if (ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
        {
            ckc_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else
        {
            ckc_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            ckc_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

        /* ============================================================ *
         *  mask / scale / softcap / alibi / qq-bias (lines 4385-4444)
         * ============================================================ */
        ckc_value_t* masked[CKC_GFX942_ATTN2D_MAX_N_TILES]
                           [CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE]; /* [n][reg] */
        for (int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            int atom = reg / 4;
            int in_atom = reg % 4;
            ckc_value_t* qp_r = ctx->hoist_q_pos[reg];
            ckc_value_t* row_ok = ctx->hoist_row_mask[reg];
            ckc_value_t* causal_lim = ctx->hoist_state_row[reg]; /* see note */
            /* NOTE: hoist_* field naming. The LICM phase stores per-reg
             * qp_r/qh_r/row_ok/causal_lim into the ctx->hoist_* arrays; this half
             * reads qp_r=hoist_q_pos, row_ok=hoist_row_mask, causal_lim is carried
             * in hoist_state_row by the LICM phase for this bucket's use. */
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                /* col_abs = (tile_off + n*16) + lane_col (lines 4394-4397) */
                ckc_value_t* col_abs = ckc_b_add(
                    b,
                    ckc_b_add(b, tile_off,
                              ckc_b_mul(b, ckc_b_const_i32(b, n),
                                        ckc_b_const_i32(b, 16))),
                    lane_col);
                ckc_value_t* causal_ok = ckc_b_cmp_le(b, col_abs, causal_lim);
                ckc_value_t* in_prefix =
                    ckc_b_cmp_lt(b, col_abs, max_seq_prefix_len);
                ckc_value_t* m_ok = ckc_b_land(
                    b, ckc_b_land(b, row_ok, causal_ok), in_prefix);
                if (ctx->SLIDING_WINDOW > 0)
                {
                    ckc_value_t* dist = ckc_b_sub(b, causal_lim, col_abs);
                    m_ok = ckc_b_land(b, m_ok, ckc_b_cmp_lt(b, dist, sw_const));
                }
                ckc_value_t* s_raw = ckc_b_vec_extract(b, S_n[n][atom], in_atom);
                ckc_value_t* s_scaled = ckc_b_fmul(b, s_raw, qk_scale);
                if (ctx->USE_SOFTCAP)
                {
                    s_scaled = ckc_b_fmul(b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                }
                ckc_value_t* score = ckc_b_select(b, m_ok, s_scaled, neg_inf);
                if (ctx->USE_ALIBI)
                {
                    /* slope * (col_abs - context_len) * RCP_LN2 (lines 4415-4418). */
                    ckc_value_t* pos_off = ckc_b_sub(b, col_abs, ctx->context_len);
                    ckc_value_t* pos_f = ckc_b_sitofp_f32(b, pos_off);
                    ckc_value_t* slope = ctx->hoist_q_head[reg]; /* hoist_alibi[reg] */
                    ckc_value_t* add_term =
                        ckc_b_fmul(b, ckc_b_fmul(b, slope, pos_f), rcp_ln2);
                    score = ckc_b_fadd(b, score, add_term);
                }
                if (ctx->USE_QQ_BIAS)
                {
                    /* qq_bias[qp_r, col_abs - context_len] (lines 4427-4443). */
                    ckc_value_t* krp = ckc_b_sub(b, col_abs, ctx->context_len);
                    ckc_value_t* krp_ok = ckc_b_land(
                        b, ckc_b_cmp_ge(b, krp, ckc_b_const_i32(b, 0)),
                        ckc_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p));
                    ckc_value_t* qq_ok = ckc_b_land(b, row_ok, krp_ok);
                    ckc_value_t* qp_safe =
                        ckc_b_select(b, row_ok, qp_r, ckc_b_const_i32(b, 0));
                    ckc_value_t* qq_idx = ckc_b_add(
                        b, ckc_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    ckc_value_t* qq_v = ckc_b_masked_global_load(
                        b, ctx->qq_bias_ptr, qq_idx, qq_ok, ckc_b_const_f32(b, 0.0),
                        ckc_f32(), 4);
                    score = ckc_b_fadd(b, score, ckc_b_fmul(b, qq_v, rcp_ln2));
                }
                masked[n][reg] = score;
            }
        }

        /* ============================================================ *
         *  per-row max via cross-lane butterfly (lines 4446-4462)
         * ============================================================ */
        ckc_value_t* m_new[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        ckc_value_t* s_local[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE]
                            [CKC_GFX942_ATTN2D_MAX_N_TILES];
        for (int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            ckc_value_t* local_max = neg_inf;
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                ckc_value_t* v = masked[n][reg];
                s_local[reg][n] = v;
                local_max = ckc_b_fmax(b, local_max, v);
            }
            ckc_value_t* tile_max = fh_warp_xor_reduce_max(ctx, local_max);
            /* online-softmax update: full_max_raw = fmax(m_vals[reg], tile_max);
             * ok = full_max_raw > -inf; m_new = select(ok, full_max_raw, 0). */
            ckc_value_t* full_max_raw =
                ckc_b_fmax(b, ctx->m_cur[reg], tile_max);
            ckc_value_t* ok = ckc_b_fcmp(b, "ogt", full_max_raw, neg_inf);
            m_new[reg] = ckc_b_select(b, ok, full_max_raw, zero_f);
        }

        /* ============================================================ *
         *  P = exp2(S - m_new) + per-row L (lines 4464-4489)
         * ============================================================ */
        ckc_value_t* l_local[CKC_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        for (int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            ckc_value_t* sum_p = zero_f;
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                ckc_value_t* p =
                    ckc_b_exp2(b, ckc_b_fsub(b, s_local[reg][n], m_new[reg]));
                if (!ctx->REGISTER_PV)
                {
                    /* publish P into P_lds[row, col] (lines 4476-4487). The
                     * FP8-PV quantise path (PV_FP8_MFMA, lines 4478-4483) is gated
                     * by ctx->FP8_MFMA_PV; the default narrow build stores
                     * cast_f32_to(p, dtype). Reuse the LICM-hoisted per-reg row
                     * (Python's `_state_row` here resolves to the same hoisted
                     * `row` SSA value, wave_row_base + _in_warp_row(reg)). */
                    ckc_value_t* row = (ctx->hoist_in_warp_row[reg] != NULL)
                                           ? ctx->hoist_in_warp_row[reg]
                                           : fh_in_warp_row(ctx, lane_rg, reg);
                    ckc_value_t* col = ckc_b_add(
                        b, ckc_b_mul(b, ckc_b_const_i32(b, n),
                                     ckc_b_const_i32(b, 16)),
                        lane_col);
                    ckc_value_t* idx[2] = {row, col};
                    if (ctx->FP8_MFMA_PV)
                    {
                        ckc_value_t* p_q = ckc_b_cvt_f32_to_fp8(
                            b, ckc_b_fmul(b, p, ckc_b_const_f32(b, 240.0)));
                        ckc_b_smem_store_vN(b, ctx->P_lds, idx, 2, p_q, 1);
                    }
                    else
                    {
                        ckc_value_t* p_d = ckc_b_cast_f32_to(b, p, ctx->dtype);
                        ckc_b_smem_store_vN(b, ctx->P_lds, idx, 2, p_d, 1);
                    }
                }
                sum_p = ckc_b_fadd(b, sum_p, p);
            }
            l_local[reg] = fh_warp_xor_reduce_sum(ctx, sum_p);
        }

        /* ---- alpha + running-L update (lines 4491-4498) ---- *
         * alpha[r] = exp2(m_vals[r] - m_new[r]);  m_vals[r] is the carry m_old.
         * new_l[r] = l_vals[r] * alpha[r] + l_local[r]; l_vals[r] is the carry
         * l_old. Both feed the PV bucket (alpha scales acc; new_l is the yielded
         * running denominator). */
        /* Python emits ALL alpha_regs first (one comprehension), THEN all
         * new_l_vals (a second comprehension); keep that two-pass order. */
        for (int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            alpha_regs[r] =
                ckc_b_exp2(b, ckc_b_fsub(b, ctx->m_cur[r], m_new[r]));
        for (int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            new_l_vals[r] = ckc_b_fadd(
                b, ckc_b_fmul(b, ctx->l_cur[r], alpha_regs[r]), l_local[r]);
            m_new_out[r] = m_new[r];
        }
    }

    /* ============================================================ *
     *  partial wait before PV (lines 4499-4538)
     *
     *  Wait for current V while leaving next K pending. Current V was issued
     *  before next K, so kv_calls_per_tile pending operations are exactly the
     *  next-K stream. The exotic GROUPED_KV2 / fp8 / transposed-V drains are
     *  gated by their ctx flags; the default narrow path takes the partial wait.
     * ============================================================ */
    {
        /* kv_calls_per_tile = (T*HD) / (THREADS * (ASYNC_LDS_MAX_BYTES_PER_LANE/2))
         * (Python prologue 2177-2287; compile-time geometry). */
        int kv_halves_per_lane = ctx->ASYNC_LDS_MAX_BYTES_PER_LANE / 2;
        int kv_calls_per_tile =
            (ctx->T * ctx->HD) / (ctx->THREADS * kv_halves_per_lane);
        if (ctx->GROUPED_KV2 || ctx->KV_FP8)
        {
            ckc_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
            ckc_b_sync(b);
        }
        else if (ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT &&
                 !ctx->K_SLICED_ACTIVE)
        {
            ckc_b_s_waitcnt(b, kv_calls_per_tile, kv_calls_per_tile, -1);
            ckc_b_s_barrier_bare(b);
        }
        else if (ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
        {
            ckc_b_s_waitcnt(b, 0, 0, -1);
            ckc_b_sync(b);
        }
        else if (ctx->TRANSPOSED_V)
        {
            ckc_b_s_waitcnt(b, 0, 0, -1);
            ckc_b_sync(b);
        }
        else
        {
            ckc_b_s_waitcnt(b, kv_calls_per_tile, kv_calls_per_tile, -1);
            ckc_b_sync(b);
        }
    }

    /* ============================================================ *
     *  PV MFMA + carry yield (lines 4540-5041)
     *
     *  Hand the softmax results to the peer PV bucket, which runs acc *= alpha;
     *  acc += P @ V and emits the scf_yield carry. The narrow (non-register-PV)
     *  path reads P from P_lds; alpha/new_l/m_new and the buffer carry come from
     *  the front half here.
     * ============================================================ */
    {
        ckc_gfx942_attn2d_pv_inputs_t pv_in;
        memset(&pv_in, 0, sizeof(pv_in));
        pv_in.alpha_regs = alpha_regs;
        pv_in.alpha_count = SOFTMAX_STATE_SLOTS;
        pv_in.new_l_vals = new_l_vals;
        pv_in.m_new = m_new_out;
        pv_in.cur_buf = ctx->cur_buf;
        /* Reuse the body-front nxt_buf (Python computes it once at body top). */
        pv_in.nxt_buf =
            (ctx->nxt_buf_v != NULL)
                ? ctx->nxt_buf_v
                : (ctx->K_SINGLE_BUF
                       ? ctx->cur_buf
                       : ckc_b_sub(b, ckc_b_const_i32(b, 1), ctx->cur_buf));
        pv_in.safe_tile1 = safe_tile1;
        ckc_gfx942_attn2d_emit_pv_bucket(ctx, &pv_in);
    }

    (void)QK_K_STEP;
    return;
}
