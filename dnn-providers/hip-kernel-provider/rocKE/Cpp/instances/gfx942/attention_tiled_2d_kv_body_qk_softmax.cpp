// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_kv_body_qk_softmax.c
 *
 * BUCKET: KV-LOOP BODY -- QK + MASK + SOFTMAX (the front half of the Python
 *   ``_emit_kv_body`` closure, rocke/instances/gfx942/attention_tiled_2d.py
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
 * SHARED STATE. Everything is read from / written to rocke_gfx942_attn2d_build_ctx_t
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

#include "rocke/helper_helper_rocke.helpers.attention.h" /* rocke_apply_softcap_log2 */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h" /* _mfma_32x32_c_* */
#include "rocke/helper_rocke.helpers.attention.h" /* rocke_warp_xor_reduce_sum */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* rocke_mfma_attn_mfma_32x32x16_for_dtype */
#include "rocke/helper_rocke.helpers.schedule.h" /* ROCKE_SCHED_DS_READ / ROCKE_SCHED_MFMA, T8 */
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"

/* ===================================================================== *
 *  Local recompute of the prologue-derived scalar/lane invariants.
 *
 *  These mirror lines 1892-1910 + 2014-2017 of the Python prologue 1:1.
 * ===================================================================== */

static rocke_value_t* fh_neg_inf(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    /* b.const_f32(float("-inf")) -- created once in the prologue (line 1892)
     * and reused; reuse the cached value so we do not allocate a duplicate. */
    if(ctx->neg_inf_v != NULL)
        return ctx->neg_inf_v;
    return rocke_b_const_f32(ctx->b, -INFINITY);
}

static rocke_value_t* fh_rcp_ln2(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    /* b.const_f32(1.4426950408889634) -- created once in the prologue (1895)
     * and reused; reuse the cached value. */
    if(ctx->rcp_ln2_v != NULL)
        return ctx->rcp_ln2_v;
    return rocke_b_const_f32(ctx->b, 1.4426950408889634);
}

static rocke_value_t* fh_sw_const(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    /* b.const_i32(int(SLIDING_WINDOW)) -- created once in the constants block
     * (line 1910) and reused; reuse the cached SSA value so we don't allocate
     * a duplicate %N. */
    if(ctx->sw_const_v != NULL)
        return ctx->sw_const_v;
    return rocke_b_const_i32(ctx->b, ctx->SLIDING_WINDOW);
}

/* qk_scale is precomputed as ctx->qk_scale_v (prologue lines 1896-1908). */
static rocke_value_t* fh_qk_scale(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->qk_scale_v;
}

/* lane_rg = lane // 16, lane_col = lane % 16 (prologue lines 2014-2015).
 * Reuse the cached SSA value emitted once at line 2014. */
static rocke_value_t* fh_lane_rg(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return rocke_b_div(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 16));
}
static rocke_value_t* fh_lane_col(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_col_v != NULL)
        return ctx->lane_col_v;
    return rocke_b_mod(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 16));
}
/* lane_col32 = lane % 32, lane_half(32) = lane // 32 (prologue 2016-2017). */
static rocke_value_t* fh_lane_col32(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_col32_v != NULL)
        return ctx->lane_col32_v;
    return rocke_b_mod(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 32));
}
static rocke_value_t* fh_lane_half32(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_half32_v != NULL)
        return ctx->lane_half32_v;
    return rocke_b_div(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 32));
}

/* max_seq_prefix_len (prologue lines 1982-1984):
 *   bm1_div_nqk = (BLOCK_M - 1) // NQK
 *   msp_raw = (context_len + qb_start_pos) + (bm1_div_nqk + 1)
 *   max_seq_prefix_len = select(msp_raw < seq_len, msp_raw, seq_len)            */
static rocke_value_t* fh_max_seq_prefix_len(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    /* Reuse the cached prologue value (Python keeps one local). */
    if(ctx->max_seq_prefix_len_v != NULL)
        return ctx->max_seq_prefix_len_v;
    int bm1_div_nqk = (ctx->BLOCK_M - 1) / ctx->NQK;
    rocke_value_t* msp_raw = rocke_b_add(ctx->b,
                                         rocke_b_add(ctx->b, ctx->context_len, ctx->qb_start_pos),
                                         rocke_b_const_i32(ctx->b, bm1_div_nqk + 1));
    return rocke_b_select(
        ctx->b, rocke_b_cmp_lt(ctx->b, msp_raw, ctx->seq_len), msp_raw, ctx->seq_len);
}

/* ===================================================================== *
 *  warp_xor_reduce_max / _sum (helpers/attention.py 303-341, 344-375).
 *
 *  Only rocke_warp_xor_reduce_sum is exposed by the C helper header; the max
 *  butterfly and the 32-lane variants are inlined here op-for-op.
 * ===================================================================== */

static rocke_value_t* fh_warp_xor_reduce_max(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* v)
{
    rocke_value_t* cur = v;
    for(int k = 0; k < 4; ++k)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(ctx->b, cur, 1 << k);
        cur = rocke_b_fmax(ctx->b, cur, remote);
    }
    return cur;
}

static rocke_value_t* fh_warp_xor_reduce_sum(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* v)
{
    /* defer to the ported helper (stages=4); identical op order. */
    return rocke_warp_xor_reduce_sum(ctx->b, v, 4);
}

static rocke_value_t* fh_warp_xor_reduce_max_32lane(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                    rocke_value_t* v)
{
    rocke_value_t* cur = v;
    for(int k = 0; k < 5; ++k)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(ctx->b, cur, 1 << k);
        cur = rocke_b_fmax(ctx->b, cur, remote);
    }
    return cur;
}

static rocke_value_t* fh_warp_xor_reduce_sum_32lane(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                    rocke_value_t* v)
{
    rocke_value_t* cur = v;
    for(int k = 0; k < 5; ++k)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(ctx->b, cur, 1 << k);
        cur = rocke_b_fadd(ctx->b, cur, remote);
    }
    return cur;
}

/* _mfma_16x16x16(b, dtype, a, bv, c) == mfma_16x16x16_for_dtype (line 142). */
static rocke_value_t* fh_mfma_16x16x16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                       rocke_value_t* a,
                                       rocke_value_t* bv,
                                       rocke_value_t* c)
{
    return rocke_mfma_16x16x16_for_dtype(ctx->b, ctx->dtype, a, bv, c);
}

/* _mfma_32x32x16(b, dtype, a, bv, c): transposed-32x32 QK/PV atom dispatch. */
static rocke_value_t* fh_mfma_32x32x16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                       rocke_value_t* a,
                                       rocke_value_t* bv,
                                       rocke_value_t* c)
{
    return rocke_mfma_attn_mfma_32x32x16_for_dtype(ctx->b, ctx->dtype, a, bv, c);
}

/* _mfma_32x32x8(b, dtype, a, bv, c): the gfx942-legal wide-K 32x32x8 QK atom.
 * Mirrors helpers/attention.py:mfma_32x32x8_for_dtype -- BOTH fp16 and bf16 are
 * CDNA3-legal here (f16 -> mfma_f32_32x32x8_f16, bf16 -> the `.1k` intrinsic
 * mfma_f32_32x32x8_bf16). The wider K=16 bf16 atom is gfx950-only, so the gfx942
 * wide-bf16 flash path uses THIS K=8 atom. */
static rocke_value_t* fh_mfma_32x32x8(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                      rocke_value_t* a,
                                      rocke_value_t* bv,
                                      rocke_value_t* c)
{
    if(ctx->dtype != NULL && ctx->dtype->name != NULL && strcmp(ctx->dtype->name, "bf16") == 0)
        return rocke_b_mfma_f32_32x32x8_bf16(ctx->b, a, bv, c);
    return rocke_b_mfma_f32_32x32x8_f16(ctx->b, a, bv, c);
}

/* T8 (DSL-novel): pin one MFMA k-step as a sched_group_barrier-ordered block. */
static void fh_sched_group_pin_mfma_step(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                         int ds_reads,
                                         int mfma_count,
                                         int sgid)
{
    rocke_b_sched_group_barrier(ctx->b, ROCKE_SCHED_DS_READ, ds_reads, sgid);
    rocke_b_sched_group_barrier(ctx->b, ROCKE_SCHED_MFMA, mfma_count, sgid);
}

/* _read_k8_mfma_operand(buf, k_row, k_off, frag) for the non-fp8 path (Python
 * 3415-3416): b.smem_load_vN(K_lds, buf, k_row, k_off, dtype, n=frag). frag is
 * 8 for the K=16 (32x32x16) atom, 4 for the K=8 (32x32x8) atom (gfx942). */
static rocke_value_t* fh_read_k8_mfma_operand_frag(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                   rocke_value_t* buf,
                                                   rocke_value_t* k_row,
                                                   rocke_value_t* k_off,
                                                   int frag)
{
    rocke_value_t* idx[3] = {buf, k_row, k_off};
    return rocke_b_smem_load_vN(ctx->b, ctx->K_lds, idx, 3, ctx->dtype, frag);
}

/* _apply_softcap(b, s, softcap) == apply_softcap_log2 (line 140). */
static rocke_value_t* fh_apply_softcap(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* s)
{
    return rocke_apply_softcap_log2(ctx->b, s, ctx->softcap_p);
}

/* _in_warp_row(r) (prologue lines 2031-2037), needed for the P_lds row coord. */
static rocke_value_t*
    fh_in_warp_row(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* lane_rg, int r)
{
    int atom_idx = r / 4;
    int in_atom = r % 4;
    return rocke_b_add(ctx->b,
                       rocke_b_mul(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 4)),
                       rocke_b_const_i32(ctx->b, atom_idx * 16 + in_atom));
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

/* REGISTER_PV scratch: P kept in registers, flattened [reg][n] (stride
 * QK_N_TILES) to hand from the softmax sub-block to the narrow PV bucket within
 * one emit_kv_body call (Python p_regs_f32[reg][n]). It is filled in the softmax
 * emit and read in the PV emit of the SAME build, so it must outlive the softmax
 * sub-block -- hence file scope rather than a stack local.
 *
 * Re-entrancy: the slots hold builder-bound rocke_value_t* that dangle once a
 * build's arena is freed. Each build's fill loop overwrites only the slots its
 * own (REGS_PER_LANE x QK_N_TILES) geometry touches, so a later build with a
 * smaller extent could otherwise hand a previous build's freed pointer to the PV
 * bucket. rocke_gfx942_attn2d_reset_softmax_scratch() zeroes the buffer at the
 * build entry so every build starts from clean NULL. */
static rocke_value_t*
    p_regs_f32_buf[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE * ROCKE_GFX942_ATTN2D_MAX_N_TILES];

/* Re-entrancy reset: clear the REGISTER_PV scratch before a new build. */
void rocke_gfx942_attn2d_reset_softmax_scratch(void)
{
    memset(p_regs_f32_buf, 0, sizeof(p_regs_f32_buf));
}

void rocke_gfx942_attn2d_emit_kv_body(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* ---- iter-start IGLP hint (line 3702-3703) ---- */
    if(ctx->USE_IGLP_OPT)
    {
        rocke_b_iglp_opt(b, 1);
    }

    /* ---- carry unpack (lines 3704-3717) ---- *
     * The driver has already split the loop carry into ctx->m_cur / ctx->l_cur /
     * ctx->acc_cur before calling the body; the front half reads those directly.
     * m_vals[r] == ctx->m_cur[r]; l_vals[r] == ctx->l_cur[r]; acc via acc_get. */

    /* ---- recompute the prologue-derived invariants this half needs ---- */
    rocke_value_t* neg_inf = fh_neg_inf(ctx);
    rocke_value_t* rcp_ln2 = fh_rcp_ln2(ctx);
    rocke_value_t* sw_const = fh_sw_const(ctx);
    rocke_value_t* qk_scale = fh_qk_scale(ctx);
    rocke_value_t* zero_f = ctx->zero_f;
    rocke_value_t* lane_rg = fh_lane_rg(ctx);
    rocke_value_t* lane_col = fh_lane_col(ctx);
    rocke_value_t* lane_half = fh_lane_half32(ctx); /* == lane // 32 (line 3503) */
    rocke_value_t* lane_col32 = fh_lane_col32(ctx);
    rocke_value_t* max_seq_prefix_len = fh_max_seq_prefix_len(ctx);
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
    ctx->nxt_buf_v
        = ctx->K_SINGLE_BUF ? ctx->cur_buf : rocke_b_sub(b, rocke_b_const_i32(b, 1), ctx->cur_buf);

    rocke_value_t* tile_off = rocke_b_mul(b, ctx->kv_tile_iv, rocke_b_const_i32(b, ctx->T));

    /* GROUPED_KV2 second-tile index + tile_off_g1 (Python 3723-3729). Python
     * emits this block BEFORE the st_*_iter mask hoist and BEFORE next_tile_iv,
     * so keep that order for SSA value numbering. NULL on the default path. */
    rocke_value_t* safe_tile1 = NULL;
    rocke_value_t* tile_off_g1 = NULL;
    if(ctx->GROUPED_KV2)
    {
        rocke_value_t* tile1_iv_raw = rocke_b_add(b, ctx->kv_tile_iv, rocke_b_const_i32(b, 1));
        rocke_value_t* tile1_in_range = rocke_b_cmp_lt(b, tile1_iv_raw, ctx->tile_end);
        safe_tile1 = rocke_b_select(b, tile1_in_range, tile1_iv_raw, ctx->kv_tile_iv);
        tile_off_g1 = rocke_b_add(b, tile_off, rocke_b_const_i32(b, ctx->T));
    }

    /* Transposed-32x32 mask-once iter-scoped invariant hoist (Python 2523-2549).
     * Fires only on the TRANSPOSED_QK_32X32 + TRANSPOSED_MASK_ONCE path; emitted
     * here (before next_tile_iv_raw) to match Python's value order. The transposed
     * softmax below reads st_*_iter through these locals. */
    rocke_value_t* st_qp_iter = NULL;
    rocke_value_t* st_row_ok_iter = NULL;
    rocke_value_t* st_causal_lim_iter = NULL;
    rocke_value_t* st_alibi_slope_iter = NULL;
    if(ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32 && ctx->TRANSPOSED_MASK_ONCE)
    {
        rocke_value_t* st_q_row_iter
            = rocke_b_add(b, ctx->wave_row_base, rocke__mfma_32x32_c_col(b, ctx->lane, 0));
        st_qp_iter = rocke_b_add(
            b, ctx->qb_start_pos, rocke_b_div(b, st_q_row_iter, rocke_b_const_i32(b, ctx->NQK)));
        /* st_qh_iter = kv_head_idx*NQK + st_q_row_iter%NQK. Python evaluates the
         * left mul BEFORE the right mod; bind sub-exprs in source order so C's
         * unspecified arg-eval order does not swap the value numbers. */
        rocke_value_t* st_qh_mul = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, ctx->NQK));
        rocke_value_t* st_qh_mod = rocke_b_mod(b, st_q_row_iter, rocke_b_const_i32(b, ctx->NQK));
        rocke_value_t* st_qh_iter = rocke_b_add(b, st_qh_mul, st_qh_mod);
        /* row_ok = land(qp<cur_batch_q_len, qh<NUM_QH); qp cmp emitted first. */
        rocke_value_t* st_ok_qp = rocke_b_cmp_lt(b, st_qp_iter, ctx->cur_batch_q_len);
        rocke_value_t* st_ok_qh = rocke_b_cmp_lt(b, st_qh_iter, rocke_b_const_i32(b, ctx->NUM_QH));
        st_row_ok_iter = rocke_b_land(b, st_ok_qp, st_ok_qh);
        st_causal_lim_iter = rocke_b_add(b, ctx->context_len, st_qp_iter);
        if(ctx->USE_ALIBI)
        {
            rocke_value_t* st_qh_iter_ok
                = rocke_b_cmp_lt(b, st_qh_iter, rocke_b_const_i32(b, ctx->NUM_QH));
            st_alibi_slope_iter = rocke_b_masked_global_load(b,
                                                             ctx->alibi_slopes_ptr,
                                                             st_qh_iter,
                                                             st_qh_iter_ok,
                                                             rocke_b_const_f32(b, 0.0),
                                                             rocke_f32(),
                                                             4);
        }
    }

    /* safe_next_tile = select(kv_tile_iv + step < tile_end, kv_tile_iv + step,
     *                         kv_tile_iv) (lines 3767-3772; non-grouped step=1) */
    rocke_value_t* next_tile_iv_raw
        = rocke_b_add(b, ctx->kv_tile_iv, rocke_b_const_i32(b, ctx->GROUPED_KV2 ? 2 : 1));
    rocke_value_t* in_range_next = rocke_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
    rocke_value_t* safe_next_tile
        = rocke_b_select(b, in_range_next, next_tile_iv_raw, ctx->kv_tile_iv);
    (void)safe_next_tile;

    /* softmax-derived state forwarded to the PV bucket (filled by the narrow
     * path below; alpha/new_l/m_new per softmax state slot). */
    rocke_value_t* alpha_regs[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* new_l_vals[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* m_new_out[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    for(int _i = 0; _i < ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE; ++_i)
    {
        alpha_regs[_i] = NULL;
        new_l_vals[_i] = NULL;
        m_new_out[_i] = NULL;
    }

    /* PT32 register groups for the transposed register-P PV (Python PT32_groups,
     * 2858-2873). Flat [p_tile * 16 + reg] per group; group 0 = current tile,
     * group 1 = GROUPED_KV2 second tile. Filled by the transposed QK/softmax
     * branch and threaded into the PV bucket. The PV side indexes
     * p_regs[p_tile * REGS_PER_LANE + reg] (REGS_PER_LANE == 16 here). */
    rocke_value_t* pt32_g0[ROCKE_GFX942_ATTN2D_MAX_N_TILES * 16];
    rocke_value_t* pt32_g1[ROCKE_GFX942_ATTN2D_MAX_N_TILES * 16];
    for(int _i = 0; _i < ROCKE_GFX942_ATTN2D_MAX_N_TILES * 16; ++_i)
    {
        pt32_g0[_i] = NULL;
        pt32_g1[_i] = NULL;
    }
    bool transposed_path = (ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32);

    /* ---- wait for current K + LDS barrier (lines 3776-3777) ---- */
    rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
    rocke_b_sync(b);

    /* EARLY_V_SCHEDULE / GROUPED_KV2 iter-start prefetch (Python 2566-2575).
     * EARLY_V: issue current V before QK. GROUPED_KV2: while QK0 reads cur_buf,
     * prefetch QK1's K tile into nxt_buf. */
    if(ctx->EARLY_V_SCHEDULE)
    {
        rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, ctx->cur_buf);
    }
    if(ctx->GROUPED_KV2)
    {
        rocke_gfx942_attn2d_issue_k(ctx, safe_tile1, ctx->nxt_buf_v);
    }

    /* ============================================================ *
     *  S = Q @ K^T
     * ============================================================ */
    if(ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32)
    {
        /* ============================================================ *
         *  Transposed-score QK: S^T = K @ Q^T (Python 2611-2666)
         *
         *  A = K tile rows (KV tokens), B = Q rows (queries). C tile:
         *  row = KV position in tile, col = query row in warp's 32-row Q tile.
         *  Consumes Q32_reg (ctx->q32_regs), NOT Q_lds (Q_ALIAS_K race).
         * ============================================================ */
        /* Python reassigned lane_col32 in the Q32 gather (gfx950 2329); the
         * transposed QK reads that value for k_row_t, not the prologue one. */
        rocke_value_t* lane_col32_qk
            = (ctx->lane_col32_q32_v != NULL) ? ctx->lane_col32_q32_v : lane_col32;
        /* TQK fragment / K-step / atom select (Python 3851-3852 + 3937-3940):
         * x8 atom => frag=4, half-stride=4, K-step=8; x16 => 8/8/16. */
        int TQK_FRAG = ctx->USE_MFMA_32X32X8 ? 4 : 8;
        int TQK_HALF_STRIDE = ctx->USE_MFMA_32X32X8 ? 4 : 8;
        rocke_value_t* ST32_n[ROCKE_GFX942_ATTN2D_MAX_N_TILES];
        rocke_value_t* ST32_n_g1[ROCKE_GFX942_ATTN2D_MAX_N_TILES];
        for(int n = 0; n < QK_N_TILES; ++n)
        {
            rocke_value_t* acc32 = rocke_b_zero_vec_f32(b, 16);
            rocke_value_t* k_row_t = rocke_b_add(b, rocke_b_const_i32(b, n * 32), lane_col32_qk);
            for(int k = 0; k < QK_K_ITERS; ++k)
            {
                /* Python emits const(k*QK_K_STEP) before the mul (left-to-right);
                 * bind it first so C arg-eval order keeps the value numbering. */
                rocke_value_t* k_off_base = rocke_b_const_i32(b, k * QK_K_STEP);
                rocke_value_t* k_off_t
                    = rocke_b_add(b,
                                  k_off_base,
                                  rocke_b_mul(b, lane_half, rocke_b_const_i32(b, TQK_HALF_STRIDE)));
                rocke_value_t* A_k_t
                    = fh_read_k8_mfma_operand_frag(ctx, ctx->cur_buf, k_row_t, k_off_t, TQK_FRAG);
                rocke_value_t* B_q_t = ctx->q32_regs[k];
                if(ctx->USE_MFMA_32X32X8)
                    acc32 = fh_mfma_32x32x8(ctx, A_k_t, B_q_t, acc32);
                else
                    acc32 = fh_mfma_32x32x16(ctx, A_k_t, B_q_t, acc32);
                if(ctx->USE_QK_PV_SCHED_GROUP_BARRIER)
                {
                    int ds_reads = TQK_FRAG / 4;
                    if(ds_reads < 1)
                        ds_reads = 1;
                    fh_sched_group_pin_mfma_step(ctx, ds_reads, 1, 0);
                }
            }
            ST32_n[n] = acc32;
        }
        if(ctx->GROUPED_KV2)
        {
            /* Second score tile for grouped online-softmax (Python 2651-2666). */
            rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
            rocke_b_sync(b);
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                rocke_value_t* acc32 = rocke_b_zero_vec_f32(b, 16);
                rocke_value_t* k_row_t
                    = rocke_b_add(b, rocke_b_const_i32(b, n * 32), lane_col32_qk);
                for(int k = 0; k < QK_K_ITERS; ++k)
                {
                    /* const(k*QK_K_STEP) bound before the mul (left-to-right). */
                    rocke_value_t* k_off_base = rocke_b_const_i32(b, k * QK_K_STEP);
                    rocke_value_t* k_off_t = rocke_b_add(
                        b,
                        k_off_base,
                        rocke_b_mul(b, lane_half, rocke_b_const_i32(b, TQK_HALF_STRIDE)));
                    rocke_value_t* A_k_t = fh_read_k8_mfma_operand_frag(
                        ctx, ctx->nxt_buf_v, k_row_t, k_off_t, TQK_FRAG);
                    rocke_value_t* B_q_t = ctx->q32_regs[k];
                    if(ctx->USE_MFMA_32X32X8)
                        acc32 = fh_mfma_32x32x8(ctx, A_k_t, B_q_t, acc32);
                    else
                        acc32 = fh_mfma_32x32x16(ctx, A_k_t, B_q_t, acc32);
                }
                ST32_n_g1[n] = acc32;
            }
        }

        /* ============================================================ *
         *  Transposed softmax (Python 2667-2899).
         *
         *  Config family: TRANSPOSED_MASK_LIMIT=False, INVARIANT_HOIST=False,
         *  MASK_ONCE=True. Per-element mask via st_*_iter (the else branch of the
         *  Python mask switch). skip_mask is always False here.
         * ============================================================ */
        rocke_value_t* st_local_max = neg_inf;
        /* st_scores[group][n][reg] (group 0/1). */
        rocke_value_t* st_scores0[ROCKE_GFX942_ATTN2D_MAX_N_TILES][16];
        rocke_value_t* st_scores1[ROCKE_GFX942_ATTN2D_MAX_N_TILES][16];
        int n_groups = ctx->GROUPED_KV2 ? 2 : 1;
        for(int group_idx = 0; group_idx < n_groups; ++group_idx)
        {
            rocke_value_t* const* st_regs = (group_idx == 0) ? ST32_n : ST32_n_g1;
            rocke_value_t* group_tile_off = (group_idx == 0) ? tile_off : tile_off_g1;
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                for(int reg = 0; reg < 16; ++reg)
                {
                    /* else branch (non-mask-limit): per-element mask.
                     * Python emits const(n*32) before _mfma_32x32_c_row (left-to-
                     * right); bind it first to keep the value numbering. */
                    rocke_value_t* k_local_base = rocke_b_const_i32(b, n * 32);
                    rocke_value_t* k_local
                        = rocke_b_add(b, k_local_base, rocke__mfma_32x32_c_row(b, ctx->lane, reg));
                    /* row_ok / qp_r / causal_lim. The MASK_ONCE path hoists these
                     * once (st_*_iter); otherwise (Python 4086-4105) recompute the
                     * per-lane query row inline, in source order: q_row_t, qp_r,
                     * qh_r, row_ok BEFORE col_abs, then causal_lim AFTER col_abs. */
                    rocke_value_t* row_ok = st_row_ok_iter;
                    rocke_value_t* causal_lim = st_causal_lim_iter;
                    rocke_value_t* qp_r = st_qp_iter;
                    bool mask_fallback = (st_row_ok_iter == NULL);
                    if(mask_fallback)
                    {
                        rocke_value_t* q_row_t = rocke_b_add(
                            b, ctx->wave_row_base, rocke__mfma_32x32_c_col(b, ctx->lane, 0));
                        qp_r = rocke_b_add(b,
                                           ctx->qb_start_pos,
                                           rocke_b_div(b, q_row_t, rocke_b_const_i32(b, ctx->NQK)));
                        /* qh_r = kv_head*NQK + q_row_t%NQK (mul before mod). */
                        rocke_value_t* qh_mul
                            = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, ctx->NQK));
                        rocke_value_t* qh_mod
                            = rocke_b_mod(b, q_row_t, rocke_b_const_i32(b, ctx->NQK));
                        rocke_value_t* qh_r = rocke_b_add(b, qh_mul, qh_mod);
                        /* row_ok = land(qp<q_len, qh<NUM_QH) (qp cmp first). */
                        rocke_value_t* ok_pos = rocke_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
                        rocke_value_t* ok_qh
                            = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, ctx->NUM_QH));
                        row_ok = rocke_b_land(b, ok_pos, ok_qh);
                    }
                    rocke_value_t* col_abs = rocke_b_add(b, group_tile_off, k_local);
                    if(mask_fallback)
                        causal_lim = rocke_b_add(b, ctx->context_len, qp_r);
                    rocke_value_t* causal_ok = rocke_b_cmp_le(b, col_abs, causal_lim);
                    rocke_value_t* in_prefix = rocke_b_cmp_lt(b, col_abs, max_seq_prefix_len);
                    rocke_value_t* m_ok
                        = rocke_b_land(b, rocke_b_land(b, row_ok, causal_ok), in_prefix);
                    if(ctx->SLIDING_WINDOW > 0)
                    {
                        rocke_value_t* dist = rocke_b_sub(b, causal_lim, col_abs);
                        m_ok = rocke_b_land(b, m_ok, rocke_b_cmp_lt(b, dist, sw_const));
                    }
                    rocke_value_t* s_raw = rocke_b_vec_extract(b, st_regs[n], reg);
                    rocke_value_t* s_scaled = rocke_b_fmul(b, s_raw, qk_scale);
                    if(ctx->USE_SOFTCAP)
                    {
                        s_scaled = rocke_b_fmul(b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                    }
                    rocke_value_t* score = rocke_b_select(b, m_ok, s_scaled, neg_inf);
                    if(ctx->USE_ALIBI)
                    {
                        /* slope * (col_abs - context_len) * RCP_LN2 (2825-2829). */
                        rocke_value_t* pos_off = rocke_b_sub(b, col_abs, ctx->context_len);
                        rocke_value_t* pos_f = rocke_b_sitofp_f32(b, pos_off);
                        rocke_value_t* add_term
                            = rocke_b_fmul(b, rocke_b_fmul(b, st_alibi_slope_iter, pos_f), rcp_ln2);
                        score = rocke_b_fadd(b, score, add_term);
                    }
                    if(ctx->USE_QQ_BIAS)
                    {
                        /* qq_bias[qp_r, col_abs - context_len] (2831-2851). */
                        rocke_value_t* krp = rocke_b_sub(b, col_abs, ctx->context_len);
                        /* Python order: cmp_ge created before cmp_lt; bind to
                         * temps so SSA ids match (C land() args eval R-to-L). */
                        rocke_value_t* krp_ge = rocke_b_cmp_ge(b, krp, rocke_b_const_i32(b, 0));
                        rocke_value_t* krp_lt = rocke_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                        rocke_value_t* krp_ok = rocke_b_land(b, krp_ge, krp_lt);
                        rocke_value_t* qq_ok = rocke_b_land(b, row_ok, krp_ok);
                        rocke_value_t* qp_safe
                            = rocke_b_select(b, row_ok, qp_r, rocke_b_const_i32(b, 0));
                        rocke_value_t* qq_idx
                            = rocke_b_add(b, rocke_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                        rocke_value_t* qq_v = rocke_b_masked_global_load(b,
                                                                         ctx->qq_bias_ptr,
                                                                         qq_idx,
                                                                         qq_ok,
                                                                         rocke_b_const_f32(b, 0.0),
                                                                         rocke_f32(),
                                                                         4);
                        score = rocke_b_fadd(b, score, rocke_b_fmul(b, qq_v, rcp_ln2));
                    }
                    if(group_idx == 0)
                        st_scores0[n][reg] = score;
                    else
                        st_scores1[n][reg] = score;
                    st_local_max = rocke_b_fmax(b, st_local_max, score);
                }
            }
        }
        /* cross-half max exchange + online max (Python 2853-2858). */
        rocke_value_t* st_remote_max = rocke_b_warp_shuffle_xor(b, st_local_max, 32);
        rocke_value_t* st_tile_max = rocke_b_fmax(b, st_local_max, st_remote_max);
        rocke_value_t* st_m_raw = rocke_b_fmax(b, ctx->m_cur[0], st_tile_max);
        rocke_value_t* st_ok = rocke_b_fcmp(b, "ogt", st_m_raw, neg_inf);
        rocke_value_t* st_m_new = rocke_b_select(b, st_ok, st_m_raw, zero_f);

        /* PT32_groups build + l_local (Python 2858-2873). */
        rocke_value_t* st_l_local = zero_f;
        for(int group_idx = 0; group_idx < n_groups; ++group_idx)
        {
            rocke_value_t** pt32 = (group_idx == 0) ? pt32_g0 : pt32_g1;
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                for(int reg = 0; reg < 16; ++reg)
                {
                    rocke_value_t* score
                        = (group_idx == 0) ? st_scores0[n][reg] : st_scores1[n][reg];
                    rocke_value_t* p_t = rocke_b_exp2(b, rocke_b_fsub(b, score, st_m_new));
                    pt32[n * 16 + reg] = p_t;
                    st_l_local = rocke_b_fadd(b, st_l_local, p_t);
                }
            }
        }
        /* l_sum = l_local + lane^32 (Python 2876-2877). */
        rocke_value_t* st_l_remote = rocke_b_warp_shuffle_xor(b, st_l_local, 32);
        rocke_value_t* st_l_sum = rocke_b_fadd(b, st_l_local, st_l_remote);

        /* m_new / l_local broadcast across SOFTMAX_STATE_SLOTS (Python 2885-2886);
         * stored to ctx scratch so the downstream alpha/L block reads them. */
        rocke_value_t* m_new[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        rocke_value_t* l_local[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            m_new[r] = st_m_new;
            l_local[r] = st_l_sum;
        }

        /* ============================================================ *
         *  post-QK V/K issue (Python 2962-2974). For the transposed path this
         *  happens AFTER the softmax above (matches Python ordering).
         * ============================================================ */
        rocke_value_t* cur_buf = ctx->cur_buf;
        rocke_value_t* nxt_buf = ctx->nxt_buf_v;
        if(ctx->GROUPED_KV2)
        {
            rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
        }
        else if(ctx->EARLY_V_SCHEDULE)
        {
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else
        {
            rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

        /* ============================================================ *
         *  alpha + running-L update (Python 3171-3181). Shared form; the
         *  transposed path threads broadcast m_new/l_local. Two-pass order.
         * ============================================================ */
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            alpha_regs[r] = rocke_b_exp2(b, rocke_b_fsub(b, ctx->m_cur[r], m_new[r]));
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            new_l_vals[r]
                = rocke_b_fadd(b, rocke_b_fmul(b, ctx->l_cur[r], alpha_regs[r]), l_local[r]);
            m_new_out[r] = m_new[r];
        }
    }
    else if(ctx->USE_MFMA_32X32)
    {
        /* ============================================================ *
         *  Non-transposed 32x32 QK (S = Q @ K^T) + softmax (Python
         *  4208-4234 QK, 4256-4293 V/K issue, 4315-4383 mask/softmax).
         *
         *  A = Q (ctx->q32_regs[k]), B = K tile rows. C tile per warp is
         *  vec_f32(16): one 32x32 score tile per N-tile. For the 32x32x8
         *  atom B is <4 x half>, col = n*32 + lane%32, K = k*8 +
         *  (lane//32)*4 + [0..3].
         * ============================================================ */
        /* Python reassigns lane_col32 in the Q32 gather (gfx950 2329); the
         * non-transposed QK reads that value for k_row32, not the prologue
         * one. lane_half is fh_lane_half32 (== lane//32). */
        rocke_value_t* lane_col32_qk
            = (ctx->lane_col32_q32_v != NULL) ? ctx->lane_col32_q32_v : lane_col32;
        rocke_value_t* lane_half_qk
            = (ctx->lane_half32_q32_v != NULL) ? ctx->lane_half32_q32_v : lane_half;
        int B32_FRAG = ctx->USE_MFMA_32X32X8 ? 4 : 8;
        int B32_HALF_STRIDE = ctx->USE_MFMA_32X32X8 ? 4 : 8;
        rocke_value_t* S32_n[ROCKE_GFX942_ATTN2D_MAX_N_TILES];
        for(int n = 0; n < QK_N_TILES; ++n)
        {
            rocke_value_t* acc32 = rocke_b_zero_vec_f32(b, 16);
            rocke_value_t* k_row32 = rocke_b_add(b, rocke_b_const_i32(b, n * 32), lane_col32_qk);
            for(int k = 0; k < QK_K_ITERS; ++k)
            {
                /* Python emits const(k*QK_K_STEP) before the mul (left-to-
                 * right); bind it first so C arg-eval order keeps the value
                 * numbering. */
                rocke_value_t* kc_base = rocke_b_const_i32(b, k * QK_K_STEP);
                rocke_value_t* kc_off32 = rocke_b_add(
                    b,
                    kc_base,
                    rocke_b_mul(b, lane_half_qk, rocke_b_const_i32(b, B32_HALF_STRIDE)));
                rocke_value_t* idx[3] = {ctx->cur_buf, k_row32, kc_off32};
                rocke_value_t* B32_v
                    = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, B32_FRAG);
                if(ctx->USE_MFMA_32X32X8)
                    acc32 = fh_mfma_32x32x8(ctx, ctx->q32_regs[k], B32_v, acc32);
                else
                    acc32 = fh_mfma_32x32x16(ctx, ctx->q32_regs[k], B32_v, acc32);
            }
            S32_n[n] = acc32;
        }

        /* ---- post-QK V/K issue (Python 4256-4293), same schedule as the
         * narrow 16x16 path below. ---- */
        rocke_value_t* cur_buf = ctx->cur_buf;
        rocke_value_t* nxt_buf
            = (ctx->nxt_buf_v != NULL)
                  ? ctx->nxt_buf_v
                  : (ctx->K_SINGLE_BUF ? cur_buf
                                       : rocke_b_sub(b, rocke_b_const_i32(b, 1), cur_buf));
        if(ctx->K_SINGLE_BUF)
        {
            rocke_b_s_waitcnt(b, /*vmcnt=*/-1, /*lgkmcnt=*/0, /*expcnt=*/-1);
            rocke_b_sync(b);
            if(ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
            {
                /* split cfvst already issued V; only next-K remains. */
            }
            else if(!ctx->EARLY_V_SCHEDULE)
            {
                rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            }
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else if(ctx->GROUPED_KV2)
        {
            rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
        }
        else if(ctx->EARLY_V_SCHEDULE)
        {
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else if(ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
        {
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else
        {
            rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

        /* ---- mask / scale / softcap / alibi / qq-bias (Python 4315-4359) ---- */
        rocke_value_t* masked[ROCKE_GFX942_ATTN2D_MAX_N_TILES]
                             [ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE]; /* [n][reg] */
        for(int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            rocke_value_t* qp_r = ctx->hoist_q_pos[reg];
            rocke_value_t* row_ok = ctx->hoist_row_mask[reg];
            rocke_value_t* causal_lim = ctx->hoist_state_row[reg];
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                /* col_abs = tile_off + _mfma_32x32_c_col(lane, n). */
                rocke_value_t* col_abs
                    = rocke_b_add(b, tile_off, rocke__mfma_32x32_c_col(b, ctx->lane, n));
                rocke_value_t* causal_ok = rocke_b_cmp_le(b, col_abs, causal_lim);
                rocke_value_t* in_prefix = rocke_b_cmp_lt(b, col_abs, max_seq_prefix_len);
                rocke_value_t* m_ok
                    = rocke_b_land(b, rocke_b_land(b, row_ok, causal_ok), in_prefix);
                if(ctx->SLIDING_WINDOW > 0)
                {
                    rocke_value_t* dist = rocke_b_sub(b, causal_lim, col_abs);
                    m_ok = rocke_b_land(b, m_ok, rocke_b_cmp_lt(b, dist, sw_const));
                }
                rocke_value_t* s_raw = rocke_b_vec_extract(b, S32_n[n], reg);
                rocke_value_t* s_scaled = rocke_b_fmul(b, s_raw, qk_scale);
                if(ctx->USE_SOFTCAP)
                {
                    s_scaled = rocke_b_fmul(b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                }
                rocke_value_t* score = rocke_b_select(b, m_ok, s_scaled, neg_inf);
                if(ctx->USE_ALIBI)
                {
                    rocke_value_t* pos_off = rocke_b_sub(b, col_abs, ctx->context_len);
                    rocke_value_t* pos_f = rocke_b_sitofp_f32(b, pos_off);
                    rocke_value_t* slope = ctx->hoist_q_head[reg];
                    rocke_value_t* add_term
                        = rocke_b_fmul(b, rocke_b_fmul(b, slope, pos_f), rcp_ln2);
                    score = rocke_b_fadd(b, score, add_term);
                }
                if(ctx->USE_QQ_BIAS)
                {
                    rocke_value_t* krp = rocke_b_sub(b, col_abs, ctx->context_len);
                    /* Python order: cmp_ge created before cmp_lt; bind to temps. */
                    rocke_value_t* krp_ge = rocke_b_cmp_ge(b, krp, rocke_b_const_i32(b, 0));
                    rocke_value_t* krp_lt = rocke_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                    rocke_value_t* krp_ok = rocke_b_land(b, krp_ge, krp_lt);
                    rocke_value_t* qq_ok = rocke_b_land(b, row_ok, krp_ok);
                    rocke_value_t* qp_safe
                        = rocke_b_select(b, row_ok, qp_r, rocke_b_const_i32(b, 0));
                    rocke_value_t* qq_idx
                        = rocke_b_add(b, rocke_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    rocke_value_t* qq_v = rocke_b_masked_global_load(b,
                                                                     ctx->qq_bias_ptr,
                                                                     qq_idx,
                                                                     qq_ok,
                                                                     rocke_b_const_f32(b, 0.0),
                                                                     rocke_f32(),
                                                                     4);
                    score = rocke_b_fadd(b, score, rocke_b_fmul(b, qq_v, rcp_ln2));
                }
                masked[n][reg] = score;
            }
        }

        /* ---- per-row max via 32-lane butterfly (Python 4361-4372) ---- */
        rocke_value_t* m_new[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        rocke_value_t* s_local[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE]
                              [ROCKE_GFX942_ATTN2D_MAX_N_TILES];
        for(int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            rocke_value_t* local_max = neg_inf;
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                rocke_value_t* v = masked[n][reg];
                s_local[reg][n] = v;
                local_max = rocke_b_fmax(b, local_max, v);
            }
            rocke_value_t* tile_max = fh_warp_xor_reduce_max_32lane(ctx, local_max);
            rocke_value_t* full_max_raw = rocke_b_fmax(b, ctx->m_cur[reg], tile_max);
            rocke_value_t* ok = rocke_b_fcmp(b, "ogt", full_max_raw, neg_inf);
            m_new[reg] = rocke_b_select(b, ok, full_max_raw, zero_f);
        }

        /* ---- P = exp2(S - m_new), store to P_lds, per-row L (Python
         * 4374-4383). Always writes P_lds (this path is not register-P). ---- */
        rocke_value_t* l_local[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        for(int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            rocke_value_t* row = ctx->hoist_in_warp_row[reg];
            rocke_value_t* sum_p = zero_f;
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                rocke_value_t* p = rocke_b_exp2(b, rocke_b_fsub(b, s_local[reg][n], m_new[reg]));
                rocke_value_t* col = rocke__mfma_32x32_c_col(b, ctx->lane, n);
                rocke_value_t* idx[2] = {row, col};
                rocke_value_t* p_d = rocke_b_cast_f32_to(b, p, ctx->dtype);
                rocke_b_smem_store_vN(b, ctx->P_lds, idx, 2, p_d, 1);
                sum_p = rocke_b_fadd(b, sum_p, p);
            }
            l_local[reg] = fh_warp_xor_reduce_sum_32lane(ctx, sum_p);
        }

        /* ---- alpha + running-L update (Python 4491-4498), two-pass order. ---- */
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            alpha_regs[r] = rocke_b_exp2(b, rocke_b_fsub(b, ctx->m_cur[r], m_new[r]));
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            new_l_vals[r]
                = rocke_b_fadd(b, rocke_b_fmul(b, ctx->l_cur[r], alpha_regs[r]), l_local[r]);
            m_new_out[r] = m_new[r];
        }
    }
    else
    {
        /* ---- narrow 16x16x16 QK (lines 4236-4254) ---- *
         * S_n[atom][n] is a per-atom, per-N-tile <4 x f32> accumulator.        */
        rocke_value_t* S_n[ROCKE_GFX942_ATTN2D_MAX_N_TILES]
                          [ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE]; /* [n][atom] */
        for(int n = 0; n < QK_N_TILES; ++n)
        {
            rocke_value_t* acc_per_atom[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
            for(int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
            {
                acc_per_atom[atom] = rocke_b_zero_vec_f32(b, 4);
            }
            for(int k = 0; k < QK_K_ITERS; ++k)
            {
                /* 16x16x16 B (K^T) operand per lane: col = n*16 + lane%16,
                 * K = k*16 + lane_rg*4 + 0..3 (<4 x dtype>). */
                /* Python evaluates the left const(k*16) BEFORE the mul (its
                 * const(4) + the mul). Bind the left const first so C's arg
                 * evaluation order does not allocate the mul's const ahead of
                 * it and shift the mul's %value. */
                rocke_value_t* kc_base = rocke_b_const_i32(b, k * 16);
                rocke_value_t* kc_off
                    = rocke_b_add(b, kc_base, rocke_b_mul(b, lane_rg, rocke_b_const_i32(b, 4)));
                rocke_value_t* k_row = rocke_b_add(b, rocke_b_const_i32(b, n * 16), lane_col);
                rocke_value_t* idx[3];
                /* B_v = K_lds[cur_buf, k_row, kc_off], <4 x dtype>. cur_buf is the
                 * loop-carried K buffer index (the double-buffer slot). */
                idx[0] = ctx->cur_buf; /* cur_buf */
                idx[1] = k_row;
                idx[2] = kc_off;
                rocke_value_t* B_v = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, 4);
                for(int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
                {
                    /* A operand = Q_reg[atom][k]; q_regs is flat [atom*QK_K_ITERS+k]
                     * filled by the Q-gather phase. */
                    rocke_value_t* A_k = ctx->q_regs[atom * QK_K_ITERS + k];
                    acc_per_atom[atom] = fh_mfma_16x16x16(ctx, A_k, B_v, acc_per_atom[atom]);
                }
            }
            for(int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
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
        rocke_value_t* cur_buf = ctx->cur_buf;
        /* nxt_buf was computed at the body front (Python emits it first); reuse. */
        rocke_value_t* nxt_buf
            = (ctx->nxt_buf_v != NULL)
                  ? ctx->nxt_buf_v
                  : (ctx->K_SINGLE_BUF ? cur_buf
                                       : rocke_b_sub(b, rocke_b_const_i32(b, 1), cur_buf));
        if(ctx->K_SINGLE_BUF)
        {
            rocke_b_s_waitcnt(b, /*vmcnt=*/-1, /*lgkmcnt=*/0, /*expcnt=*/-1);
            rocke_b_sync(b);
            if(ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
            {
                /* split cfvst already issued V; only next-K remains. */
            }
            else if(!ctx->EARLY_V_SCHEDULE)
            {
                rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            }
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else if(ctx->GROUPED_KV2)
        {
            rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
        }
        else if(ctx->EARLY_V_SCHEDULE)
        {
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else if(ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
        {
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else
        {
            rocke_gfx942_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            rocke_gfx942_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

        /* ============================================================ *
         *  mask / scale / softcap / alibi / qq-bias (lines 4385-4444)
         * ============================================================ */
        rocke_value_t* masked[ROCKE_GFX942_ATTN2D_MAX_N_TILES]
                             [ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE]; /* [n][reg] */
        for(int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            int atom = reg / 4;
            int in_atom = reg % 4;
            rocke_value_t* qp_r = ctx->hoist_q_pos[reg];
            rocke_value_t* row_ok = ctx->hoist_row_mask[reg];
            rocke_value_t* causal_lim = ctx->hoist_state_row[reg]; /* see note */
            /* NOTE: hoist_* field naming. The LICM phase stores per-reg
             * qp_r/qh_r/row_ok/causal_lim into the ctx->hoist_* arrays; this half
             * reads qp_r=hoist_q_pos, row_ok=hoist_row_mask, causal_lim is carried
             * in hoist_state_row by the LICM phase for this bucket's use. */
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                /* col_abs = (tile_off + n*16) + lane_col (lines 4394-4397) */
                rocke_value_t* col_abs;
                {
                    rocke_value_t* cn = rocke_b_const_i32(b, n);
                    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
                    col_abs = rocke_b_add(
                        b, rocke_b_add(b, tile_off, rocke_b_mul(b, cn, c16)), lane_col);
                }
                rocke_value_t* causal_ok = rocke_b_cmp_le(b, col_abs, causal_lim);
                rocke_value_t* in_prefix = rocke_b_cmp_lt(b, col_abs, max_seq_prefix_len);
                rocke_value_t* m_ok
                    = rocke_b_land(b, rocke_b_land(b, row_ok, causal_ok), in_prefix);
                if(ctx->SLIDING_WINDOW > 0)
                {
                    rocke_value_t* dist = rocke_b_sub(b, causal_lim, col_abs);
                    m_ok = rocke_b_land(b, m_ok, rocke_b_cmp_lt(b, dist, sw_const));
                }
                rocke_value_t* s_raw = rocke_b_vec_extract(b, S_n[n][atom], in_atom);
                rocke_value_t* s_scaled = rocke_b_fmul(b, s_raw, qk_scale);
                if(ctx->USE_SOFTCAP)
                {
                    s_scaled = rocke_b_fmul(b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                }
                rocke_value_t* score = rocke_b_select(b, m_ok, s_scaled, neg_inf);
                if(ctx->USE_ALIBI)
                {
                    /* slope * (col_abs - context_len) * RCP_LN2 (lines 4415-4418). */
                    rocke_value_t* pos_off = rocke_b_sub(b, col_abs, ctx->context_len);
                    rocke_value_t* pos_f = rocke_b_sitofp_f32(b, pos_off);
                    rocke_value_t* slope = ctx->hoist_q_head[reg]; /* hoist_alibi[reg] */
                    rocke_value_t* add_term
                        = rocke_b_fmul(b, rocke_b_fmul(b, slope, pos_f), rcp_ln2);
                    score = rocke_b_fadd(b, score, add_term);
                }
                if(ctx->USE_QQ_BIAS)
                {
                    /* qq_bias[qp_r, col_abs - context_len] (lines 4427-4443). */
                    rocke_value_t* krp = rocke_b_sub(b, col_abs, ctx->context_len);
                    /* Python order: cmp_ge created before cmp_lt; bind to
                     * temps so SSA ids match (C land() args eval R-to-L). */
                    rocke_value_t* krp_ge = rocke_b_cmp_ge(b, krp, rocke_b_const_i32(b, 0));
                    rocke_value_t* krp_lt = rocke_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                    rocke_value_t* krp_ok = rocke_b_land(b, krp_ge, krp_lt);
                    rocke_value_t* qq_ok = rocke_b_land(b, row_ok, krp_ok);
                    rocke_value_t* qp_safe
                        = rocke_b_select(b, row_ok, qp_r, rocke_b_const_i32(b, 0));
                    rocke_value_t* qq_idx
                        = rocke_b_add(b, rocke_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    rocke_value_t* qq_v = rocke_b_masked_global_load(b,
                                                                     ctx->qq_bias_ptr,
                                                                     qq_idx,
                                                                     qq_ok,
                                                                     rocke_b_const_f32(b, 0.0),
                                                                     rocke_f32(),
                                                                     4);
                    score = rocke_b_fadd(b, score, rocke_b_fmul(b, qq_v, rcp_ln2));
                }
                masked[n][reg] = score;
            }
        }

        /* ============================================================ *
         *  per-row max via cross-lane butterfly (lines 4446-4462)
         * ============================================================ */
        rocke_value_t* m_new[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        rocke_value_t* s_local[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE]
                              [ROCKE_GFX942_ATTN2D_MAX_N_TILES];
        for(int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            rocke_value_t* local_max = neg_inf;
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                rocke_value_t* v = masked[n][reg];
                s_local[reg][n] = v;
                local_max = rocke_b_fmax(b, local_max, v);
            }
            rocke_value_t* tile_max = fh_warp_xor_reduce_max(ctx, local_max);
            /* online-softmax update: full_max_raw = fmax(m_vals[reg], tile_max);
             * ok = full_max_raw > -inf; m_new = select(ok, full_max_raw, 0). */
            rocke_value_t* full_max_raw = rocke_b_fmax(b, ctx->m_cur[reg], tile_max);
            rocke_value_t* ok = rocke_b_fcmp(b, "ogt", full_max_raw, neg_inf);
            m_new[reg] = rocke_b_select(b, ok, full_max_raw, zero_f);
        }

        /* ============================================================ *
         *  P = exp2(S - m_new) + per-row L (lines 4464-4489)
         * ============================================================ */
        rocke_value_t* l_local[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
        for(int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            rocke_value_t* sum_p = zero_f;
            for(int n = 0; n < QK_N_TILES; ++n)
            {
                rocke_value_t* p = rocke_b_exp2(b, rocke_b_fsub(b, s_local[reg][n], m_new[reg]));
                /* Python stores p_regs_f32[reg][n] = p ALWAYS (line 4474), then
                 * only writes P_lds when not REGISTER_PV. The register-P PV bucket
                 * consumes p_regs_f32 flattened [reg*QK_N_TILES + n]. */
                if(ctx->REGISTER_PV)
                    p_regs_f32_buf[reg * QK_N_TILES + n] = p;
                if(!ctx->REGISTER_PV)
                {
                    /* publish P into P_lds[row, col] (lines 4476-4487). The
                     * FP8-PV quantise path (PV_FP8_MFMA, lines 4478-4483) is gated
                     * by ctx->FP8_MFMA_PV; the default narrow build stores
                     * cast_f32_to(p, dtype). Reuse the LICM-hoisted per-reg row
                     * (Python's `_state_row` here resolves to the same hoisted
                     * `row` SSA value, wave_row_base + _in_warp_row(reg)). */
                    rocke_value_t* row = (ctx->hoist_in_warp_row[reg] != NULL)
                                             ? ctx->hoist_in_warp_row[reg]
                                             : fh_in_warp_row(ctx, lane_rg, reg);
                    rocke_value_t* col;
                    {
                        rocke_value_t* cn = rocke_b_const_i32(b, n);
                        rocke_value_t* c16 = rocke_b_const_i32(b, 16);
                        col = rocke_b_add(b, rocke_b_mul(b, cn, c16), lane_col);
                    }
                    rocke_value_t* idx[2] = {row, col};
                    if(ctx->FP8_MFMA_PV)
                    {
                        rocke_value_t* p_q = rocke_b_cvt_f32_to_fp8(
                            b, rocke_b_fmul(b, p, rocke_b_const_f32(b, 240.0)));
                        rocke_b_smem_store_vN(b, ctx->P_lds, idx, 2, p_q, 1);
                    }
                    else
                    {
                        rocke_value_t* p_d = rocke_b_cast_f32_to(b, p, ctx->dtype);
                        rocke_b_smem_store_vN(b, ctx->P_lds, idx, 2, p_d, 1);
                    }
                }
                sum_p = rocke_b_fadd(b, sum_p, p);
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
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            alpha_regs[r] = rocke_b_exp2(b, rocke_b_fsub(b, ctx->m_cur[r], m_new[r]));
        for(int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            new_l_vals[r]
                = rocke_b_fadd(b, rocke_b_fmul(b, ctx->l_cur[r], alpha_regs[r]), l_local[r]);
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
        /* kv_calls_per_tile = (T*HD) / (THREADS * KV_DMA_HALVES_PER_LANE) (Python
         * prologue 2177-2181 gfx942 / 1413-1415 gfx950; compile-time geometry). */
        int kv_halves_per_lane = ctx->KV_DMA_HALVES_PER_LANE;
        int kv_calls_per_tile = (ctx->T * ctx->HD) / (ctx->THREADS * kv_halves_per_lane);
        if(ctx->GROUPED_KV2 || ctx->KV_FP8)
        {
            rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
            rocke_b_sync(b);
        }
        else if(ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT && !ctx->K_SLICED_ACTIVE)
        {
            rocke_b_s_waitcnt(b, kv_calls_per_tile, kv_calls_per_tile, -1);
            rocke_b_s_barrier_bare(b);
        }
        else if(ctx->TRANSPOSED_V_STORE && ctx->CFV_STORE_SPLIT)
        {
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_sync(b);
        }
        else if(ctx->TRANSPOSED_V)
        {
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_sync(b);
        }
        else
        {
            rocke_b_s_waitcnt(b, kv_calls_per_tile, kv_calls_per_tile, -1);
            rocke_b_sync(b);
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
        rocke_gfx942_attn2d_pv_inputs_t pv_in;
        memset(&pv_in, 0, sizeof(pv_in));
        pv_in.alpha_regs = alpha_regs;
        pv_in.alpha_count = SOFTMAX_STATE_SLOTS;
        pv_in.new_l_vals = new_l_vals;
        pv_in.m_new = m_new_out;
        if(ctx->REGISTER_PV)
        {
            pv_in.p_regs_f32 = p_regs_f32_buf;
            pv_in.p_regs_f32_stride = QK_N_TILES;
        }
        if(transposed_path)
        {
            /* Register-P^T groups built by the transposed softmax above. The PV
             * bucket consumes pt32_g0 (and pt32_g1 under GROUPED_KV2) directly. */
            pv_in.pt32_g0 = pt32_g0;
            pv_in.pt32_g1 = pt32_g1;
            pv_in.pt32_count = QK_N_TILES * 16;
        }
        pv_in.cur_buf = ctx->cur_buf;
        /* Reuse the body-front nxt_buf (Python computes it once at body top). */
        pv_in.nxt_buf
            = (ctx->nxt_buf_v != NULL)
                  ? ctx->nxt_buf_v
                  : (ctx->K_SINGLE_BUF ? ctx->cur_buf
                                       : rocke_b_sub(b, rocke_b_const_i32(b, 1), ctx->cur_buf));
        pv_in.safe_tile1 = safe_tile1;
        rocke_gfx942_attn2d_emit_pv_bucket(ctx, &pv_in);
    }

    (void)QK_K_STEP;
    return;
}
