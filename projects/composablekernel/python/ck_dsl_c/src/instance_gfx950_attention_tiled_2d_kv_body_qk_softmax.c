/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_kv_body_qk_softmax.c
 *
 * BUCKET: KV-LOOP BODY -- QK + MASK + SOFTMAX (the FRONT half of the gfx950
 *   ``_emit_kv_body`` closure, ck_dsl/instances/gfx950/attention_tiled_2d.py) plus
 *   the two prologue helpers that were declared in the internal header but whose
 *   bodies belong to this front-half scope:
 *
 *     - ckc_gfx950_attn2d_emit_kv_body        QK MFMA + mask/scale/softcap/alibi/
 *                                             qq-bias + online softmax, then hands
 *                                             the softmax state to the PEER
 *                                             ckc_gfx950_attn2d_emit_pv_bucket.
 *     - ckc_gfx950_attn2d_emit_licm_hoist     per-reg row/pos/head/mask invariants
 *                                             hoisted out of the KV-loop.
 *     - ckc_gfx950_attn2d_permute_p_c_to_a16  P C-dist -> A16 register permute.
 *     - ckc_gfx950_attn2d_pack_p_a16          pack permuted P into the 16-K MFMA A.
 *     - ckc_gfx950_attn2d_pack_p_a32          pack two permuted P groups into the
 *                                             32-K MFMA A operand.
 *
 *   The gfx950 closure set is the SUBSET documented in the internal header: it has
 *   NONE of gfx942's transposed-V-store / K-sliced-ring / IGLP / single-buffer
 *   families, so the corresponding gfx942 ctx fields and code branches are absent
 *   here by design. The kept paths are the narrow 16x16x16 QK + the wide/transposed
 *   32x32x16 QK, both of which the gfx950 ctx fully supports.
 *
 * SHARED STATE. Everything is read from / written to ckc_gfx950_attn2d_build_ctx_t
 * (the internal header). The lane-decomposition + scalar invariants the front half
 * needs are cached on the ctx by the loop-bounds / q-gather phases (ctx->lane_*_v,
 * ctx->neg_inf_v, ctx->rcp_ln2_v, ctx->qk_scale_v, ctx->sw_const_v,
 * ctx->max_seq_prefix_len_v); where a cache is NULL the helper recomputes the value
 * exactly as the prologue did (LLVM CSE folds the duplicate). This TU edits no
 * header.
 *
 * Lifetime: every emitted node is arena-owned (ctx->b->arena).
 */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "ckc/instance_gfx950_attention_tiled_2d_internal.h"
#include "ckc/instance_gfx950_attention_tiled_2d.h"        /* mfma_32x32_c_row/_col */
#include "ckc/helper_helper_ck_dsl.helpers.attention.h"    /* softcap_log2 / mfma_16x16x16 */
#include "ckc/helper_ck_dsl.helpers.attention.h"           /* warp_xor_reduce_sum   */
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h"      /* 32x32x16_for_dtype    */

/* ===================================================================== *
 *  Local recompute of the prologue-derived scalar / lane invariants.
 *  Each mirrors the cached ctx field and recomputes only when the cache is
 *  NULL (DCE/CSE-identical to the Python prologue's shared SSA value).
 * ===================================================================== */

static ckc_value_t* fh_neg_inf(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->neg_inf_v != NULL)
        return ctx->neg_inf_v;
    return ckc_b_const_f32(ctx->b, -INFINITY);
}

static ckc_value_t* fh_rcp_ln2(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->rcp_ln2_v != NULL)
        return ctx->rcp_ln2_v;
    return ckc_b_const_f32(ctx->b, 1.4426950408889634);
}

static ckc_value_t* fh_sw_const(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->sw_const_v != NULL)
        return ctx->sw_const_v;
    return ckc_b_const_i32(ctx->b, ctx->SLIDING_WINDOW);
}

static ckc_value_t* fh_qk_scale(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    return ctx->qk_scale_v;
}

static ckc_value_t* fh_lane_rg(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return ckc_b_div(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 16));
}
static ckc_value_t* fh_lane_col(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_col_v != NULL)
        return ctx->lane_col_v;
    return ckc_b_mod(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 16));
}
static ckc_value_t* fh_lane_col32(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_col32_v != NULL)
        return ctx->lane_col32_v;
    return ckc_b_mod(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 32));
}
static ckc_value_t* fh_lane_half32(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->lane_half32_v != NULL)
        return ctx->lane_half32_v;
    return ckc_b_div(ctx->b, ctx->lane, ckc_b_const_i32(ctx->b, 32));
}

static ckc_value_t* fh_max_seq_prefix_len(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    if (ctx->max_seq_prefix_len_v != NULL)
        return ctx->max_seq_prefix_len_v;
    int bm1_div_nqk = (ctx->BLOCK_M - 1) / ctx->NQK;
    ckc_value_t* msp_raw = ckc_b_add(
        ctx->b, ckc_b_add(ctx->b, ctx->context_len, ctx->qb_start_pos),
        ckc_b_const_i32(ctx->b, bm1_div_nqk + 1));
    return ckc_b_select(ctx->b, ckc_b_cmp_lt(ctx->b, msp_raw, ctx->seq_len),
                        msp_raw, ctx->seq_len);
}

static ckc_value_t* fh_warp_xor_reduce_max(ckc_gfx950_attn2d_build_ctx_t* ctx,
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

static ckc_value_t* fh_warp_xor_reduce_sum(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                           ckc_value_t* v)
{
    return ckc_warp_xor_reduce_sum(ctx->b, v, 4);
}

static ckc_value_t* fh_mfma_16x16x16(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                     ckc_value_t* a, ckc_value_t* bv, ckc_value_t* c)
{
    return ckc_mfma_16x16x16_for_dtype(ctx->b, ctx->dtype, a, bv, c);
}

/* mfma_16x16x32_for_dtype (helpers/attention.py lines 593-601): dispatch
 * mfma_f32_16x16x32_<dtype> for fp16 / bf16 (the gfx950 wide-K QK atom). */
static ckc_value_t* fh_mfma_16x16x32(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                     ckc_value_t* a, ckc_value_t* bv, ckc_value_t* c)
{
    if (ctx->dtype != NULL && ctx->dtype->name != NULL &&
        strcmp(ctx->dtype->name, "f16") == 0)
        return ckc_b_mfma_f32_16x16x32_f16(ctx->b, a, bv, c);
    return ckc_b_mfma_f32_16x16x32_bf16(ctx->b, a, bv, c);
}

static ckc_value_t* fh_mfma_32x32x16(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                     ckc_value_t* a, ckc_value_t* bv, ckc_value_t* c)
{
    return ckc_mfma_attn_mfma_32x32x16_for_dtype(ctx->b, ctx->dtype, a, bv, c);
}

static ckc_value_t* fh_read_k8_mfma_operand(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                            ckc_value_t* buf,
                                            ckc_value_t* k_row,
                                            ckc_value_t* k_off)
{
    ckc_value_t* idx[3] = {buf, k_row, k_off};
    return ckc_b_smem_load_vN(ctx->b, ctx->K_lds, idx, 3, ctx->dtype, 8);
}

static ckc_value_t* fh_apply_softcap(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                     ckc_value_t* s)
{
    return ckc_apply_softcap_log2(ctx->b, s, ctx->softcap_p);
}

static ckc_value_t* fh_in_warp_row(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                   ckc_value_t* lane_rg, int r)
{
    int atom_idx = r / 4;
    int in_atom = r % 4;
    return ckc_b_add(ctx->b, ckc_b_mul(ctx->b, lane_rg, ckc_b_const_i32(ctx->b, 4)),
                     ckc_b_const_i32(ctx->b, atom_idx * 16 + in_atom));
}

/* 32-lane intra-half XOR reductions for the gfx950 mfma_32x32x16 output tiles
 * (helpers/attention.py warp_xor_reduce_{max,sum}_32lane): 5 stages, masks
 * 1,2,4,8,16. */
static ckc_value_t* fh_warp_xor_reduce_max_32lane(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                                  ckc_value_t* v)
{
    ckc_value_t* cur = v;
    for (int k = 0; k < 5; ++k)
        cur = ckc_b_fmax(ctx->b, cur, ckc_b_warp_shuffle_xor(ctx->b, cur, 1 << k));
    return cur;
}
static ckc_value_t* fh_warp_xor_reduce_sum_32lane(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                                  ckc_value_t* v)
{
    ckc_value_t* cur = v;
    for (int k = 0; k < 5; ++k)
        cur = ckc_b_fadd(ctx->b, cur, ckc_b_warp_shuffle_xor(ctx->b, cur, 1 << k));
    return cur;
}

/* 32x32 C-distribution row/col, gfx950 spelling. */
static ckc_value_t* fh_c32_row(ckc_gfx950_attn2d_build_ctx_t* ctx, int reg)
{
    return ckc_gfx950_attention_tiled_2d_mfma_32x32_c_row(ctx->b, ctx->lane, reg);
}
static ckc_value_t* fh_c32_col(ckc_gfx950_attn2d_build_ctx_t* ctx, int n_tile32)
{
    return ckc_gfx950_attention_tiled_2d_mfma_32x32_c_col(ctx->b, ctx->lane, n_tile32);
}

/* ===================================================================== *
 *  LICM hoist of per-reg invariants (gfx942 prologue lines 3609-3651 analogue).
 *
 *  Fills ctx->hoist_* before the KV-loop with the per-reg query position / head /
 *  row-validity / causal-limit invariants the QK/softmax bucket reads. Field-name
 *  mapping consumed by the front half:
 *      hoist_q_pos   <- qp_r        hoist_row_mask <- row_ok
 *      hoist_state_row <- causal_lim   hoist_q_head <- qh_r (or ALiBi slope)
 *  hoist_in_warp_row caches the per-reg row.
 * ===================================================================== */
void ckc_gfx950_attn2d_emit_licm_hoist(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    int reg;
    for (reg = 0; reg < ctx->REGS_PER_LANE; ++reg)
    {
        ckc_value_t* row;
        if (ctx->USE_MFMA_32X32)
            row = ckc_b_add(b, ctx->wave_row_base, fh_c32_row(ctx, reg));
        else
            row = ckc_b_add(b, ctx->wave_row_base,
                            ckc_gfx950_attn2d_in_warp_row(ctx, reg));

        ckc_value_t* qp_r = ckc_b_add(
            b, ctx->qb_start_pos,
            ckc_b_div(b, row, ckc_b_const_i32(b, ctx->NQK)));
        ckc_value_t* qh_mul =
            ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, ctx->NQK));
        ckc_value_t* qh_mod = ckc_b_mod(b, row, ckc_b_const_i32(b, ctx->NQK));
        ckc_value_t* qh_r = ckc_b_add(b, qh_mul, qh_mod);
        ckc_value_t* row_ok_pos = ckc_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
        ckc_value_t* row_ok_qh =
            ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, ctx->NUM_QH));
        ckc_value_t* row_ok = ckc_b_land(b, row_ok_pos, row_ok_qh);
        ckc_value_t* causal_lim = ckc_b_add(b, ctx->context_len, qp_r);

        ctx->hoist_in_warp_row[reg] = row;
        ctx->hoist_q_pos[reg]       = qp_r;
        ctx->hoist_q_head[reg]      = qh_r;
        ctx->hoist_row_mask[reg]    = row_ok;
        ctx->hoist_state_row[reg]   = causal_lim;
    }

    if (ctx->USE_ALIBI)
    {
        const ckc_type_t* f32 = ckc_f32();
        for (reg = 0; reg < ctx->REGS_PER_LANE; ++reg)
        {
            ckc_value_t* qh_r = ctx->hoist_q_head[reg];
            ckc_value_t* qh_ok =
                ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, ctx->NUM_QH));
            ckc_value_t* slope = ckc_b_masked_global_load(
                b, ctx->alibi_slopes_ptr, qh_r, qh_ok, ckc_b_const_f32(b, 0.0),
                f32, 4);
            ctx->hoist_q_head[reg] = slope;
        }
    }

    /* Transposed-32x32 invariant hoist (Python 2446-2480): compute the per-lane
     * st_q_row/st_qp/st_qh/st_row_ok/st_causal_lim once before the KV loop. */
    ctx->st_qp_hoist = NULL;
    ctx->st_qh_hoist = NULL;
    ctx->st_row_ok_hoist = NULL;
    ctx->st_causal_lim_hoist = NULL;
    ctx->st_alibi_slope_hoist = NULL;
    if (ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32 && ctx->TRANSPOSED_INVARIANT_HOIST)
    {
        ckc_value_t* st_q_row =
            ckc_b_add(b, ctx->wave_row_base, fh_c32_col(ctx, 0));
        ckc_value_t* st_qp = ckc_b_add(
            b, ctx->qb_start_pos,
            ckc_b_div(b, st_q_row, ckc_b_const_i32(b, ctx->NQK)));
        ckc_value_t* st_qh_mul =
            ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, ctx->NQK));
        ckc_value_t* st_qh_mod = ckc_b_mod(b, st_q_row, ckc_b_const_i32(b, ctx->NQK));
        ckc_value_t* st_qh = ckc_b_add(b, st_qh_mul, st_qh_mod);
        ckc_value_t* st_ok_qp = ckc_b_cmp_lt(b, st_qp, ctx->cur_batch_q_len);
        ckc_value_t* st_ok_qh =
            ckc_b_cmp_lt(b, st_qh, ckc_b_const_i32(b, ctx->NUM_QH));
        ctx->st_qp_hoist = st_qp;
        ctx->st_qh_hoist = st_qh;
        ctx->st_row_ok_hoist = ckc_b_land(b, st_ok_qp, st_ok_qh);
        ctx->st_causal_lim_hoist = ckc_b_add(b, ctx->context_len, st_qp);
        if (ctx->USE_ALIBI)
        {
            ckc_value_t* st_qh_ok =
                ckc_b_cmp_lt(b, st_qh, ckc_b_const_i32(b, ctx->NUM_QH));
            ctx->st_alibi_slope_hoist = ckc_b_masked_global_load(
                b, ctx->alibi_slopes_ptr, st_qh, st_qh_ok,
                ckc_b_const_f32(b, 0.0), ckc_f32(), 4);
        }
    }
}

/* ===================================================================== *
 *  P permute / pack helpers (gfx950 lines 1292-1384).
 *
 *  _permute_p_c_to_a16: one 16-col P tile, from 16x16 MFMA-C regs (the QK C
 *  distribution) to the PV-A register layout. The transform is (A,B,C,R) ->
 *  (B,A,R,C): two bit-level transposes swap C with R inside a lane quad, then
 *  lane-field swaps exchange A and B. The per-lane scalars (lane_col_mod4 /
 *  lane_rg / lane_col_div4) are read from the cached ctx fields (recomputed if
 *  NULL), DCE-identical to Python's shared SSA values.
 * ===================================================================== */
void ckc_gfx950_attn2d_permute_p_c_to_a16(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                          ckc_value_t* const* p_regs_f32, int n,
                                          ckc_value_t** out, int* out_n)
{
    ckc_ir_builder_t* b = ctx->b;
    ckc_value_t* lane_col_mod4 =
        (ctx->lane_col_mod4_v != NULL)
            ? ctx->lane_col_mod4_v
            : ckc_b_mod(b, fh_lane_col(ctx), ckc_b_const_i32(b, 4));
    ckc_value_t* lane_col_div4 =
        (ctx->lane_col_div4_v != NULL)
            ? ctx->lane_col_div4_v
            : ckc_b_div(b, fh_lane_col(ctx), ckc_b_const_i32(b, 4));
    ckc_value_t* lane_rg = fh_lane_rg(ctx);

    ckc_value_t* vals[4];
    int i;
    for (i = 0; i < 4; ++i)
        vals[i] = (i < n) ? p_regs_f32[i] : NULL;

    /* Two bit-level transposes: swap C (lane_col_mod4) with R (reg index). */
    for (int bit = 0; bit < 2; ++bit)
    {
        ckc_value_t* lane_bit = ckc_gfx950_attn2d_bit2(ctx, lane_col_mod4, bit);
        int reg_bit = 1 << bit;
        ckc_value_t* old[4];
        for (i = 0; i < 4; ++i)
            old[i] = vals[i];
        for (int reg = 0; reg < 4; ++reg)
        {
            ckc_value_t* partner =
                ckc_b_warp_shuffle_xor(b, old[reg ^ reg_bit], reg_bit);
            ckc_value_t* same_bit =
                ckc_b_cmp_eq(b, lane_bit, ckc_b_const_i32(b, (reg >> bit) & 1));
            vals[reg] = ckc_b_select(b, same_bit, old[reg], partner);
        }
    }

    /* Lane-field swaps: exchange A (lane_rg) and B (lane_col_div4). */
    {
        const int bits[2]      = {0, 1};
        const int lane_xors[2] = {20, 40};
        for (int j = 0; j < 2; ++j)
        {
            int bit            = bits[j];
            int lane_xor       = lane_xors[j];
            ckc_value_t* a_bit = ckc_gfx950_attn2d_bit2(ctx, lane_rg, bit);
            ckc_value_t* b_bit = ckc_gfx950_attn2d_bit2(ctx, lane_col_div4, bit);
            ckc_value_t* swap  = ckc_b_cmp_ne(b, a_bit, b_bit);
            for (i = 0; i < 4; ++i)
                vals[i] = ckc_b_select(b, swap,
                                       ckc_b_warp_shuffle_xor(b, vals[i], lane_xor),
                                       vals[i]);
        }
    }

    for (i = 0; i < 4; ++i)
        out[i] = vals[i];
    if (out_n != NULL)
        *out_n = 4;
}

ckc_value_t* ckc_gfx950_attn2d_pack_p_a16(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                          ckc_value_t* const* p_regs_f32, int n)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    ckc_value_t* permuted[4];
    ckc_value_t* casted[4];
    int out_n = 0;
    ckc_gfx950_attn2d_permute_p_c_to_a16(ctx, p_regs_f32, n, permuted, &out_n);
    for (int i = 0; i < 4; ++i)
        casted[i] = ckc_b_cast_f32_to(b, permuted[i], dtype);
    return ckc_b_vec_pack(b, casted, 4, dtype);
}

ckc_value_t* ckc_gfx950_attn2d_pack_p_a32(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                          ckc_value_t* const* p_regs0_f32,
                                          ckc_value_t* const* p_regs1_f32, int n)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    ckc_value_t* vals0[4];
    ckc_value_t* vals1[4];
    ckc_value_t* lohi[8];
    int dummy = 0;
    ckc_gfx950_attn2d_permute_p_c_to_a16(ctx, p_regs0_f32, n, vals0, &dummy);
    ckc_gfx950_attn2d_permute_p_c_to_a16(ctx, p_regs1_f32, n, vals1, &dummy);
    for (int j = 0; j < 4; ++j)
    {
        ckc_value_t* v0 = vals0[j];
        ckc_value_t* v1 = vals1[j];
        ckc_value_t* v0_x16 = ckc_b_warp_shuffle_xor(b, v0, 16);
        ckc_value_t* v0_x32 = ckc_b_warp_shuffle_xor(b, v0, 32);
        ckc_value_t* v0_x48 = ckc_b_warp_shuffle_xor(b, v0, 48);
        ckc_value_t* v1_x16 = ckc_b_warp_shuffle_xor(b, v1, 16);
        ckc_value_t* v1_x32 = ckc_b_warp_shuffle_xor(b, v1, 32);
        ckc_value_t* v1_x48 = ckc_b_warp_shuffle_xor(b, v1, 48);
        lohi[j]     = ckc_gfx950_attn2d_select_lane_rg(ctx, v0, v0_x48, v1_x32, v1_x16);
        lohi[4 + j] = ckc_gfx950_attn2d_select_lane_rg(ctx, v0_x16, v0_x32, v1_x48, v1);
    }
    {
        ckc_value_t* casted[8];
        for (int i = 0; i < 8; ++i)
            casted[i] = ckc_b_cast_f32_to(b, lohi[i], dtype);
        return ckc_b_vec_pack(b, casted, 8, dtype);
    }
}

/* REGISTER_PV scratch: P kept in registers, flattened [reg][n] (stride
 * QK_N_TILES) to hand from the softmax sub-block to the PV bucket within one
 * emit_kv_body call (Python p_regs_f32[reg][n]). It is filled in the softmax
 * emit and read in the PV emit of the SAME build, so it must outlive the softmax
 * sub-block -- hence file scope rather than a stack local.
 *
 * Re-entrancy: the slots hold builder-bound ckc_value_t* that dangle once a
 * build's arena is freed. Each build's fill loop overwrites only the slots its
 * own (REGS_PER_LANE x QK_N_TILES) geometry touches, so a later build with a
 * smaller extent could otherwise hand a previous build's freed pointer to the PV
 * bucket. ckc_gfx950_attn2d_reset_softmax_scratch() zeroes the buffer at the
 * build entry so every build starts from clean NULL. */
static ckc_value_t* p_regs_f32_buf[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE
                                   * CKC_GFX950_ATTN2D_MAX_N_TILES];

/* Re-entrancy reset: clear the REGISTER_PV scratch before a new build. */
void ckc_gfx950_attn2d_reset_softmax_scratch(void)
{
    memset(p_regs_f32_buf, 0, sizeof(p_regs_f32_buf));
}

/* ===================================================================== *
 *  Front half of the QK + mask + softmax body.
 *
 *  Narrow 16x16x16 and transposed-32x32 QK/softmax paths; both supported by the
 *  gfx950 ctx. On return the softmax-derived state (alpha / new_l / m_new and,
 *  on the transposed path, the PT32 register-P groups) is handed to the peer
 *  ckc_gfx950_attn2d_emit_pv_bucket, which runs acc *= alpha; acc += P @ V and
 *  emits the scf_yield carry.
 * ===================================================================== */
void ckc_gfx950_attn2d_emit_kv_body(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* ---- recompute the prologue-derived invariants this half needs ---- */
    ckc_value_t* neg_inf = fh_neg_inf(ctx);
    ckc_value_t* rcp_ln2 = fh_rcp_ln2(ctx);
    ckc_value_t* sw_const = fh_sw_const(ctx);
    ckc_value_t* qk_scale = fh_qk_scale(ctx);
    ckc_value_t* zero_f = ctx->zero_f;
    ckc_value_t* lane_rg = fh_lane_rg(ctx);
    ckc_value_t* lane_col = fh_lane_col(ctx);
    ckc_value_t* lane_half = fh_lane_half32(ctx);
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

    /* nxt_buf = 1 - cur_buf (gfx950 is always double-buffered). */
    ctx->nxt_buf_v = ckc_b_sub(b, ckc_b_const_i32(b, 1), ctx->cur_buf);

    ckc_value_t* tile_off = ckc_b_mul(b, ctx->kv_tile_iv, ckc_b_const_i32(b, ctx->T));

    /* GROUPED_KV2 second-tile index + tile_off_g1. NULL on the default path. */
    ckc_value_t* safe_tile1 = NULL;
    ckc_value_t* tile_off_g1 = NULL;
    if (ctx->GROUPED_KV2)
    {
        ckc_value_t* tile1_iv_raw =
            ckc_b_add(b, ctx->kv_tile_iv, ckc_b_const_i32(b, 1));
        ckc_value_t* tile1_in_range = ckc_b_cmp_lt(b, tile1_iv_raw, ctx->tile_end);
        safe_tile1 = ckc_b_select(b, tile1_in_range, tile1_iv_raw, ctx->kv_tile_iv);
        tile_off_g1 = ckc_b_add(b, tile_off, ckc_b_const_i32(b, ctx->T));
    }

    /* Transposed-32x32 mask-once iter-scoped invariant hoist. */
    ckc_value_t* st_qp_iter = NULL;
    ckc_value_t* st_row_ok_iter = NULL;
    ckc_value_t* st_causal_lim_iter = NULL;
    ckc_value_t* st_alibi_slope_iter = NULL;
    if (ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32 && ctx->TRANSPOSED_MASK_ONCE)
    {
        ckc_value_t* st_q_row_iter =
            ckc_b_add(b, ctx->wave_row_base, fh_c32_col(ctx, 0));
        st_qp_iter = ckc_b_add(
            b, ctx->qb_start_pos,
            ckc_b_div(b, st_q_row_iter, ckc_b_const_i32(b, ctx->NQK)));
        ckc_value_t* st_qh_mul =
            ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, ctx->NQK));
        ckc_value_t* st_qh_mod =
            ckc_b_mod(b, st_q_row_iter, ckc_b_const_i32(b, ctx->NQK));
        ckc_value_t* st_qh_iter = ckc_b_add(b, st_qh_mul, st_qh_mod);
        ckc_value_t* st_ok_qp = ckc_b_cmp_lt(b, st_qp_iter, ctx->cur_batch_q_len);
        ckc_value_t* st_ok_qh =
            ckc_b_cmp_lt(b, st_qh_iter, ckc_b_const_i32(b, ctx->NUM_QH));
        st_row_ok_iter = ckc_b_land(b, st_ok_qp, st_ok_qh);
        st_causal_lim_iter = ckc_b_add(b, ctx->context_len, st_qp_iter);
        if (ctx->USE_ALIBI)
        {
            ckc_value_t* st_qh_iter_ok =
                ckc_b_cmp_lt(b, st_qh_iter, ckc_b_const_i32(b, ctx->NUM_QH));
            st_alibi_slope_iter = ckc_b_masked_global_load(
                b, ctx->alibi_slopes_ptr, st_qh_iter, st_qh_iter_ok,
                ckc_b_const_f32(b, 0.0), ckc_f32(), 4);
        }
    }

    /* safe_next_tile = select(next < tile_end, next, kv_tile_iv). */
    ckc_value_t* next_tile_iv_raw =
        ckc_b_add(b, ctx->kv_tile_iv, ckc_b_const_i32(b, ctx->GROUPED_KV2 ? 2 : 1));
    ckc_value_t* in_range_next = ckc_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
    ckc_value_t* safe_next_tile =
        ckc_b_select(b, in_range_next, next_tile_iv_raw, ctx->kv_tile_iv);

    /* softmax-derived state forwarded to the PV bucket. */
    ckc_value_t* alpha_regs[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    ckc_value_t* new_l_vals[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    ckc_value_t* m_new_out[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    for (int _i = 0; _i < CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE; ++_i)
    {
        alpha_regs[_i] = NULL;
        new_l_vals[_i] = NULL;
        m_new_out[_i] = NULL;
    }

    /* PT32 register groups for the transposed register-P PV. */
    ckc_value_t* pt32_g0[CKC_GFX950_ATTN2D_MAX_N_TILES * 16];
    ckc_value_t* pt32_g1[CKC_GFX950_ATTN2D_MAX_N_TILES * 16];
    for (int _i = 0; _i < CKC_GFX950_ATTN2D_MAX_N_TILES * 16; ++_i)
    {
        pt32_g0[_i] = NULL;
        pt32_g1[_i] = NULL;
    }
    bool transposed_path = (ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32);

    /* ---- wait for current K + LDS barrier ---- */
    ckc_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
    ckc_b_sync(b);

    /* Early-V schedule: V_lds is single-buffered and the iter-start full drain
     * guarantees the previous PV's V reads retired, so the current V copy can be
     * issued before QK to overlap with QK + softmax (Python 2566-2570). */
    if (ctx->EARLY_V_SCHEDULE)
        ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, ctx->cur_buf);
    /* GROUPED_KV2: prefetch the second K tile into nxt_buf while QK0 reads
     * cur_buf (Python 2571-2575). */
    if (ctx->GROUPED_KV2)
        ckc_gfx950_attn2d_issue_k(ctx, safe_tile1, ctx->nxt_buf_v);

    /* ============================================================ *
     *  S = Q @ K^T
     * ============================================================ */
    if (ctx->USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32)
    {
        /* Transposed-score QK: S^T = K @ Q^T. */
        ckc_value_t* lane_col32_qk = lane_col32;
        ckc_value_t* ST32_n[CKC_GFX950_ATTN2D_MAX_N_TILES];
        ckc_value_t* ST32_n_g1[CKC_GFX950_ATTN2D_MAX_N_TILES];
        for (int n = 0; n < QK_N_TILES; ++n)
        {
            ckc_value_t* acc32 = ckc_b_zero_vec_f32(b, 16);
            ckc_value_t* k_row_t =
                ckc_b_add(b, ckc_b_const_i32(b, n * 32), lane_col32_qk);
            for (int k = 0; k < QK_K_ITERS; ++k)
            {
                ckc_value_t* k_off_base = ckc_b_const_i32(b, k * 16);
                ckc_value_t* k_off_t = ckc_b_add(
                    b, k_off_base, ckc_b_mul(b, lane_half, ckc_b_const_i32(b, 8)));
                ckc_value_t* A_k_t =
                    fh_read_k8_mfma_operand(ctx, ctx->cur_buf, k_row_t, k_off_t);
                ckc_value_t* B_q_t = ctx->q32_regs[k];
                acc32 = fh_mfma_32x32x16(ctx, A_k_t, B_q_t, acc32);
            }
            ST32_n[n] = acc32;
        }
        if (ctx->GROUPED_KV2)
        {
            ckc_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
            ckc_b_sync(b);
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                ckc_value_t* acc32 = ckc_b_zero_vec_f32(b, 16);
                ckc_value_t* k_row_t =
                    ckc_b_add(b, ckc_b_const_i32(b, n * 32), lane_col32);
                for (int k = 0; k < QK_K_ITERS; ++k)
                {
                    /* Python: b.add(const(k*16), b.mul(lane_half, 8)) creates the
                     * const(k*16) BEFORE the mul (left-to-right); bind it so C's
                     * arg eval order matches the SSA ids. */
                    ckc_value_t* k_off_base = ckc_b_const_i32(b, k * 16);
                    ckc_value_t* k_off_t = ckc_b_add(
                        b, k_off_base, ckc_b_mul(b, lane_half, ckc_b_const_i32(b, 8)));
                    ckc_value_t* A_k_t = fh_read_k8_mfma_operand(
                        ctx, ctx->nxt_buf_v, k_row_t, k_off_t);
                    ckc_value_t* B_q_t = ctx->q32_regs[k];
                    acc32 = fh_mfma_32x32x16(ctx, A_k_t, B_q_t, acc32);
                }
                ST32_n_g1[n] = acc32;
            }
        }

        /* Transposed softmax (per-element mask via st_*_iter). */
        ckc_value_t* st_local_max = neg_inf;
        ckc_value_t* st_scores0[CKC_GFX950_ATTN2D_MAX_N_TILES][16];
        ckc_value_t* st_scores1[CKC_GFX950_ATTN2D_MAX_N_TILES][16];
        int n_groups = ctx->GROUPED_KV2 ? 2 : 1;
        for (int group_idx = 0; group_idx < n_groups; ++group_idx)
        {
            ckc_value_t* const* st_regs = (group_idx == 0) ? ST32_n : ST32_n_g1;
            ckc_value_t* group_tile_off =
                (group_idx == 0) ? tile_off : tile_off_g1;
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                for (int reg = 0; reg < 16; ++reg)
                {
                    ckc_value_t* k_local_base = ckc_b_const_i32(b, n * 32);
                    ckc_value_t* k_local =
                        ckc_b_add(b, k_local_base, fh_c32_row(ctx, reg));
                    ckc_value_t* qp_r;
                    ckc_value_t* row_ok;
                    ckc_value_t* causal_lim;
                    ckc_value_t* col_abs;
                    ckc_value_t* causal_ok;
                    ckc_value_t* in_prefix;
                    ckc_value_t* m_ok;
                    if (ctx->TRANSPOSED_INVARIANT_HOIST)
                    {
                        /* invariant-hoist: row_ok / causal_lim hoisted pre-loop
                         * (Python 2776-2779). */
                        qp_r = ctx->st_qp_hoist;
                        row_ok = ctx->st_row_ok_hoist;
                        causal_lim = ctx->st_causal_lim_hoist;
                        col_abs = ckc_b_add(b, group_tile_off, k_local);
                    }
                    else if (ctx->TRANSPOSED_MASK_ONCE)
                    {
                        /* mask-once: row_ok / causal_lim hoisted per-iter. */
                        qp_r = st_qp_iter;
                        row_ok = st_row_ok_iter;
                        causal_lim = st_causal_lim_iter;
                        col_abs = ckc_b_add(b, group_tile_off, k_local);
                    }
                    else
                    {
                        /* Plain transposed (Python 2784-2806): recompute the
                         * per-element row position / mask inline. */
                        ckc_value_t* q_row_t =
                            ckc_b_add(b, ctx->wave_row_base, fh_c32_col(ctx, 0));
                        ckc_value_t* qh_r;
                        ckc_value_t* qh_mul;
                        ckc_value_t* qh_mod;
                        ckc_value_t* row_ok_pos;
                        ckc_value_t* row_ok_qh;
                        qp_r = ckc_b_add(
                            b, ctx->qb_start_pos,
                            ckc_b_div(b, q_row_t, ckc_b_const_i32(b, ctx->NQK)));
                        /* mul before mod; first cmp on qp_r (Python arg order). */
                        qh_mul = ckc_b_mul(b, ctx->kv_head_idx,
                                           ckc_b_const_i32(b, ctx->NQK));
                        qh_mod = ckc_b_mod(b, q_row_t, ckc_b_const_i32(b, ctx->NQK));
                        qh_r = ckc_b_add(b, qh_mul, qh_mod);
                        row_ok_pos = ckc_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
                        row_ok_qh =
                            ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, ctx->NUM_QH));
                        row_ok = ckc_b_land(b, row_ok_pos, row_ok_qh);
                        col_abs = ckc_b_add(b, group_tile_off, k_local);
                        causal_lim = ckc_b_add(b, ctx->context_len, qp_r);
                    }
                    causal_ok = ckc_b_cmp_le(b, col_abs, causal_lim);
                    in_prefix = ckc_b_cmp_lt(b, col_abs, max_seq_prefix_len);
                    m_ok = ckc_b_land(
                        b, ckc_b_land(b, row_ok, causal_ok), in_prefix);
                    if (ctx->SLIDING_WINDOW > 0)
                    {
                        ckc_value_t* dist = ckc_b_sub(b, causal_lim, col_abs);
                        m_ok = ckc_b_land(b, m_ok, ckc_b_cmp_lt(b, dist, sw_const));
                    }
                    ckc_value_t* s_raw = ckc_b_vec_extract(b, st_regs[n], reg);
                    ckc_value_t* s_scaled = ckc_b_fmul(b, s_raw, qk_scale);
                    if (ctx->USE_SOFTCAP)
                    {
                        s_scaled = ckc_b_fmul(
                            b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                    }
                    ckc_value_t* score = ckc_b_select(b, m_ok, s_scaled, neg_inf);
                    if (ctx->USE_ALIBI)
                    {
                        ckc_value_t* pos_off =
                            ckc_b_sub(b, col_abs, ctx->context_len);
                        ckc_value_t* pos_f = ckc_b_sitofp_f32(b, pos_off);
                        ckc_value_t* add_term = ckc_b_fmul(
                            b, ckc_b_fmul(b, st_alibi_slope_iter, pos_f), rcp_ln2);
                        score = ckc_b_fadd(b, score, add_term);
                    }
                    if (ctx->USE_QQ_BIAS)
                    {
                        ckc_value_t* krp = ckc_b_sub(b, col_abs, ctx->context_len);
                        /* Python: b.land(b.cmp_ge(...), b.cmp_lt(...)) emits the
                         * cmp_ge BEFORE the cmp_lt (left-to-right). Bind each in
                         * Python order so C's right-to-left arg eval does not swap
                         * the two compare SSA ids. */
                        ckc_value_t* krp_ge = ckc_b_cmp_ge(b, krp, ckc_b_const_i32(b, 0));
                        ckc_value_t* krp_lt = ckc_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                        ckc_value_t* krp_ok = ckc_b_land(b, krp_ge, krp_lt);
                        ckc_value_t* qq_ok = ckc_b_land(b, row_ok, krp_ok);
                        ckc_value_t* qp_safe = ckc_b_select(
                            b, row_ok, qp_r, ckc_b_const_i32(b, 0));
                        ckc_value_t* qq_idx = ckc_b_add(
                            b, ckc_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                        ckc_value_t* qq_v = ckc_b_masked_global_load(
                            b, ctx->qq_bias_ptr, qq_idx, qq_ok,
                            ckc_b_const_f32(b, 0.0), ckc_f32(), 4);
                        score = ckc_b_fadd(b, score, ckc_b_fmul(b, qq_v, rcp_ln2));
                    }
                    if (group_idx == 0)
                        st_scores0[n][reg] = score;
                    else
                        st_scores1[n][reg] = score;
                    st_local_max = ckc_b_fmax(b, st_local_max, score);
                }
            }
        }
        ckc_value_t* st_remote_max = ckc_b_warp_shuffle_xor(b, st_local_max, 32);
        ckc_value_t* st_tile_max = ckc_b_fmax(b, st_local_max, st_remote_max);
        ckc_value_t* st_m_raw = ckc_b_fmax(b, ctx->m_cur[0], st_tile_max);
        ckc_value_t* st_ok = ckc_b_fcmp(b, "ogt", st_m_raw, neg_inf);
        ckc_value_t* st_m_new = ckc_b_select(b, st_ok, st_m_raw, zero_f);

        ckc_value_t* st_l_local = zero_f;
        for (int group_idx = 0; group_idx < n_groups; ++group_idx)
        {
            ckc_value_t** pt32 = (group_idx == 0) ? pt32_g0 : pt32_g1;
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                for (int reg = 0; reg < 16; ++reg)
                {
                    ckc_value_t* score = (group_idx == 0) ? st_scores0[n][reg]
                                                          : st_scores1[n][reg];
                    ckc_value_t* p_t =
                        ckc_b_exp2(b, ckc_b_fsub(b, score, st_m_new));
                    pt32[n * 16 + reg] = p_t;
                    st_l_local = ckc_b_fadd(b, st_l_local, p_t);
                }
            }
        }
        ckc_value_t* st_l_remote = ckc_b_warp_shuffle_xor(b, st_l_local, 32);
        ckc_value_t* st_l_sum = ckc_b_fadd(b, st_l_local, st_l_remote);

        ckc_value_t* m_new[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
        ckc_value_t* l_local[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
        for (int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        {
            m_new[r] = st_m_new;
            l_local[r] = st_l_sum;
        }

        /* post-QK V/K issue. */
        ckc_value_t* cur_buf = ctx->cur_buf;
        ckc_value_t* nxt_buf = ctx->nxt_buf_v;
        if (ctx->GROUPED_KV2)
        {
            ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
        }
        else if (ctx->EARLY_V_SCHEDULE)
        {
            ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else
        {
            ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

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
    else if (ctx->USE_MFMA_32X32)
    {
        /* ---- non-transposed 32x32 QK + mask/softmax (Python 2902-2914,
         * 2997-3065) ---- one 32x32x16 C tile per warp per N-tile. */
        ckc_value_t* lane_half_qg =
            (ctx->lane_half_qg_v != NULL) ? ctx->lane_half_qg_v : lane_half;
        ckc_value_t* S32_n[CKC_GFX950_ATTN2D_MAX_N_TILES];
        ckc_value_t* masked32[CKC_GFX950_ATTN2D_MAX_N_TILES][16]; /* [n][reg] */

        /* QK: S32_n[n] = sum_k mfma_32x32x16(Q32_reg[k], K_lds[k_row32, kc_off32]). */
        for (int n = 0; n < QK_N_TILES; ++n)
        {
            ckc_value_t* acc32 = ckc_b_zero_vec_f32(b, 16);
            ckc_value_t* k_row32 =
                ckc_b_add(b, ckc_b_const_i32(b, n * 32), lane_col32);
            for (int k = 0; k < QK_K_ITERS; ++k)
            {
                ckc_value_t* kc_base = ckc_b_const_i32(b, k * 16);
                ckc_value_t* kc_off32 = ckc_b_add(
                    b, kc_base, ckc_b_mul(b, lane_half_qg, ckc_b_const_i32(b, 8)));
                ckc_value_t* idx[3];
                ckc_value_t* B32_v;
                idx[0] = ctx->cur_buf;
                idx[1] = k_row32;
                idx[2] = kc_off32;
                B32_v = ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, 8);
                acc32 = fh_mfma_32x32x16(ctx, ctx->q32_regs[k], B32_v, acc32);
            }
            S32_n[n] = acc32;
        }

        /* post-QK V/K issue (Python 2966-2975). */
        {
            ckc_value_t* cur_buf = ctx->cur_buf;
            ckc_value_t* nxt_buf = ctx->nxt_buf_v;
            if (ctx->GROUPED_KV2)
            {
                ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
                ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
            }
            else if (ctx->EARLY_V_SCHEDULE)
            {
                ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
            }
            else
            {
                ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
                ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
            }
        }

        /* mask / scale / softcap / alibi / qq-bias (Python 2998-3041). */
        for (int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            ckc_value_t* qp_r = ctx->hoist_q_pos[reg];
            ckc_value_t* row_ok = ctx->hoist_row_mask[reg];
            ckc_value_t* causal_lim = ctx->hoist_state_row[reg];
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                ckc_value_t* col_abs = ckc_b_add(b, tile_off, fh_c32_col(ctx, n));
                ckc_value_t* causal_ok = ckc_b_cmp_le(b, col_abs, causal_lim);
                ckc_value_t* in_prefix =
                    ckc_b_cmp_lt(b, col_abs, max_seq_prefix_len);
                ckc_value_t* m_ok = ckc_b_land(
                    b, ckc_b_land(b, row_ok, causal_ok), in_prefix);
                ckc_value_t* s_raw;
                ckc_value_t* s_scaled;
                ckc_value_t* score;
                if (ctx->SLIDING_WINDOW > 0)
                {
                    ckc_value_t* dist = ckc_b_sub(b, causal_lim, col_abs);
                    m_ok = ckc_b_land(b, m_ok, ckc_b_cmp_lt(b, dist, sw_const));
                }
                s_raw = ckc_b_vec_extract(b, S32_n[n], reg);
                s_scaled = ckc_b_fmul(b, s_raw, qk_scale);
                if (ctx->USE_SOFTCAP)
                    s_scaled =
                        ckc_b_fmul(b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                score = ckc_b_select(b, m_ok, s_scaled, neg_inf);
                if (ctx->USE_ALIBI)
                {
                    ckc_value_t* pos_off = ckc_b_sub(b, col_abs, ctx->context_len);
                    ckc_value_t* pos_f = ckc_b_sitofp_f32(b, pos_off);
                    ckc_value_t* slope = ctx->hoist_q_head[reg];
                    ckc_value_t* add_term =
                        ckc_b_fmul(b, ckc_b_fmul(b, slope, pos_f), rcp_ln2);
                    score = ckc_b_fadd(b, score, add_term);
                }
                if (ctx->USE_QQ_BIAS)
                {
                    ckc_value_t* krp = ckc_b_sub(b, col_abs, ctx->context_len);
                    /* Python emits cmp_ge before cmp_lt (left-to-right); bind in
                     * order so C's arg eval does not swap the compare SSA ids. */
                    ckc_value_t* krp_ge = ckc_b_cmp_ge(b, krp, ckc_b_const_i32(b, 0));
                    ckc_value_t* krp_lt = ckc_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                    ckc_value_t* krp_ok = ckc_b_land(b, krp_ge, krp_lt);
                    ckc_value_t* qq_ok = ckc_b_land(b, row_ok, krp_ok);
                    ckc_value_t* qp_safe =
                        ckc_b_select(b, row_ok, qp_r, ckc_b_const_i32(b, 0));
                    ckc_value_t* qq_idx = ckc_b_add(
                        b, ckc_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    ckc_value_t* qq_v = ckc_b_masked_global_load(
                        b, ctx->qq_bias_ptr, qq_idx, qq_ok,
                        ckc_b_const_f32(b, 0.0), ckc_f32(), 4);
                    score = ckc_b_fadd(b, score, ckc_b_fmul(b, qq_v, rcp_ln2));
                }
                masked32[n][reg] = score;
            }
        }

        /* per-row max via 32-lane butterfly + online update (Python 3043-3054). */
        {
            ckc_value_t* s_local[16][CKC_GFX950_ATTN2D_MAX_N_TILES];
            for (int reg = 0; reg < REGS_PER_LANE; ++reg)
            {
                ckc_value_t* local_max = neg_inf;
                ckc_value_t* tile_max;
                ckc_value_t* full_max_raw;
                ckc_value_t* ok;
                for (int n = 0; n < QK_N_TILES; ++n)
                {
                    ckc_value_t* v = masked32[n][reg];
                    s_local[reg][n] = v;
                    local_max = ckc_b_fmax(b, local_max, v);
                }
                tile_max = fh_warp_xor_reduce_max_32lane(ctx, local_max);
                full_max_raw = ckc_b_fmax(b, ctx->m_cur[reg], tile_max);
                ok = ckc_b_fcmp(b, "ogt", full_max_raw, neg_inf);
                m_new_out[reg] = ckc_b_select(b, ok, full_max_raw, zero_f);
            }

            /* P = exp2(S - m_new); publish to P_lds; per-row L (Python 3056-3065). */
            for (int reg = 0; reg < REGS_PER_LANE; ++reg)
            {
                ckc_value_t* row = ctx->hoist_in_warp_row[reg];
                ckc_value_t* sum_p = zero_f;
                for (int n = 0; n < QK_N_TILES; ++n)
                {
                    ckc_value_t* p = ckc_b_exp2(
                        b, ckc_b_fsub(b, s_local[reg][n], m_new_out[reg]));
                    ckc_value_t* col = fh_c32_col(ctx, n);
                    ckc_value_t* sidx[2];
                    ckc_value_t* p_d = ckc_b_cast_f32_to(b, p, ctx->dtype);
                    sidx[0] = row;
                    sidx[1] = col;
                    ckc_b_smem_store_vN(b, ctx->P_lds, sidx, 2, p_d, 1);
                    sum_p = ckc_b_fadd(b, sum_p, p);
                }
                new_l_vals[reg] = fh_warp_xor_reduce_sum_32lane(ctx, sum_p);
            }
        }

        /* alpha + running-L update (Python shared block 3066-3172). */
        for (int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            alpha_regs[r] =
                ckc_b_exp2(b, ckc_b_fsub(b, ctx->m_cur[r], m_new_out[r]));
        for (int r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
            new_l_vals[r] = ckc_b_fadd(
                b, ckc_b_fmul(b, ctx->l_cur[r], alpha_regs[r]), new_l_vals[r]);
    }
    else
    {
        /* ---- wide 16x16x32 QK (Python lines 2916-2961) ----
         * Storage [atom][n] mirrors Python's S_n[atom][n]. */
        ckc_value_t* S_n[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE]
                        [CKC_GFX950_ATTN2D_MAX_N_TILES]; /* [atom][n] */
        for (int n = 0; n < QK_N_TILES; ++n)
        {
            ckc_value_t* acc_per_atom[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
            for (int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
                acc_per_atom[atom] = ckc_b_zero_vec_f32(b, 4);
            for (int k = 0; k < QK_K_ITERS; ++k)
            {
                /* Match Python arg-eval order (left-to-right): const(k*32)
                 * is created BEFORE the mul(lane_rg, 8) sub-expression. C
                 * arg-eval order is unspecified, so sequence explicitly. */
                ckc_value_t* kc_base = ckc_b_const_i32(b, k * 32);
                ckc_value_t* kc_off =
                    ckc_b_add(b, kc_base, ckc_b_mul(b, lane_rg, ckc_b_const_i32(b, 8)));
                ckc_value_t* kr_base = ckc_b_const_i32(b, n * 16);
                ckc_value_t* k_row = ckc_b_add(b, kr_base, lane_col);
                ckc_value_t* idx[3];
                idx[0] = ctx->cur_buf;
                idx[1] = k_row;
                idx[2] = kc_off;
                ckc_value_t* B_v;
                if (ctx->FP8_MFMA_QK)
                {
                    ckc_value_t* B_fp8 = ckc_b_smem_load_vN(
                        b, ctx->K_lds, idx, 3, ckc_fp8e4m3(), 8);
                    B_v = ckc_dequant_fp8x8_to_dtype(
                        b, B_fp8, ctx->k_scale_p, ctx->dtype);
                }
                else
                {
                    B_v = ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, 8);
                }
                for (int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
                {
                    ckc_value_t* A_k = ctx->q_regs[atom * QK_K_ITERS + k];
                    acc_per_atom[atom] =
                        fh_mfma_16x16x32(ctx, A_k, B_v, acc_per_atom[atom]);
                }
            }
            for (int atom = 0; atom < M_ATOMS_PER_WARP; ++atom)
                S_n[atom][n] = acc_per_atom[atom];
        }

        /* post-QK V/K issue. */
        ckc_value_t* cur_buf = ctx->cur_buf;
        ckc_value_t* nxt_buf = ctx->nxt_buf_v;
        if (ctx->GROUPED_KV2)
        {
            ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, cur_buf);
        }
        else if (ctx->EARLY_V_SCHEDULE)
        {
            ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }
        else
        {
            ckc_gfx950_attn2d_issue_v(ctx, ctx->kv_tile_iv, cur_buf);
            ckc_gfx950_attn2d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

        /* mask / scale / softcap / alibi / qq-bias. */
        ckc_value_t* masked[CKC_GFX950_ATTN2D_MAX_N_TILES]
                           [CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE]; /* [n][reg] */
        for (int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            int atom = reg / 4;
            int in_atom = reg % 4;
            ckc_value_t* qp_r = ctx->hoist_q_pos[reg];
            ckc_value_t* row_ok = ctx->hoist_row_mask[reg];
            ckc_value_t* causal_lim = ctx->hoist_state_row[reg];
            for (int n = 0; n < QK_N_TILES; ++n)
            {
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
                ckc_value_t* s_raw = ckc_b_vec_extract(b, S_n[atom][n], in_atom);
                ckc_value_t* s_scaled = ckc_b_fmul(b, s_raw, qk_scale);
                if (ctx->USE_SOFTCAP)
                {
                    s_scaled =
                        ckc_b_fmul(b, fh_apply_softcap(ctx, s_scaled), rcp_ln2);
                }
                ckc_value_t* score = ckc_b_select(b, m_ok, s_scaled, neg_inf);
                if (ctx->USE_ALIBI)
                {
                    ckc_value_t* pos_off = ckc_b_sub(b, col_abs, ctx->context_len);
                    ckc_value_t* pos_f = ckc_b_sitofp_f32(b, pos_off);
                    ckc_value_t* slope = ctx->hoist_q_head[reg];
                    ckc_value_t* add_term =
                        ckc_b_fmul(b, ckc_b_fmul(b, slope, pos_f), rcp_ln2);
                    score = ckc_b_fadd(b, score, add_term);
                }
                if (ctx->USE_QQ_BIAS)
                {
                    ckc_value_t* krp = ckc_b_sub(b, col_abs, ctx->context_len);
                    /* Python emits cmp_ge before cmp_lt (left-to-right); bind in
                     * order so C's arg eval does not swap the compare SSA ids. */
                    ckc_value_t* krp_ge = ckc_b_cmp_ge(b, krp, ckc_b_const_i32(b, 0));
                    ckc_value_t* krp_lt = ckc_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                    ckc_value_t* krp_ok = ckc_b_land(b, krp_ge, krp_lt);
                    ckc_value_t* qq_ok = ckc_b_land(b, row_ok, krp_ok);
                    ckc_value_t* qp_safe =
                        ckc_b_select(b, row_ok, qp_r, ckc_b_const_i32(b, 0));
                    ckc_value_t* qq_idx = ckc_b_add(
                        b, ckc_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    ckc_value_t* qq_v = ckc_b_masked_global_load(
                        b, ctx->qq_bias_ptr, qq_idx, qq_ok,
                        ckc_b_const_f32(b, 0.0), ckc_f32(), 4);
                    score = ckc_b_fadd(b, score, ckc_b_fmul(b, qq_v, rcp_ln2));
                }
                masked[n][reg] = score;
            }
        }

        /* per-row max via cross-lane butterfly + online update. */
        ckc_value_t* m_new[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
        ckc_value_t* s_local[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE]
                            [CKC_GFX950_ATTN2D_MAX_N_TILES];
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
            ckc_value_t* full_max_raw = ckc_b_fmax(b, ctx->m_cur[reg], tile_max);
            ckc_value_t* ok = ckc_b_fcmp(b, "ogt", full_max_raw, neg_inf);
            m_new[reg] = ckc_b_select(b, ok, full_max_raw, zero_f);
        }

        /* P = exp2(S - m_new) + per-row L. */
        ckc_value_t* l_local[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
        for (int reg = 0; reg < REGS_PER_LANE; ++reg)
        {
            ckc_value_t* sum_p = zero_f;
            for (int n = 0; n < QK_N_TILES; ++n)
            {
                ckc_value_t* p =
                    ckc_b_exp2(b, ckc_b_fsub(b, s_local[reg][n], m_new[reg]));
                if (ctx->REGISTER_PV)
                    p_regs_f32_buf[reg * QK_N_TILES + n] = p;
                if (!ctx->REGISTER_PV)
                {
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

        /* alpha + running-L update (two-pass order). */
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
     *  partial wait before PV is emitted by emit_pv_bucket (Python lines
     *  3181-3199), which is the first thing the shared PV section does.
     *  Emitting it here too would duplicate the waitcnt/barrier.
     *  PV MFMA + carry yield -- hand off to the peer PV bucket.
     * ============================================================ */
    {
        ckc_gfx950_attn2d_pv_inputs_t pv_in;
        memset(&pv_in, 0, sizeof(pv_in));
        pv_in.alpha_regs = alpha_regs;
        pv_in.alpha_count = SOFTMAX_STATE_SLOTS;
        pv_in.new_l_vals = new_l_vals;
        pv_in.m_new = m_new_out;
        if (ctx->REGISTER_PV)
        {
            pv_in.p_regs_f32 = p_regs_f32_buf;
            pv_in.p_regs_f32_stride = QK_N_TILES;
        }
        if (transposed_path)
        {
            pv_in.pt32_g0 = pt32_g0;
            pv_in.pt32_g1 = pt32_g1;
            pv_in.pt32_count = QK_N_TILES * 16;
        }
        pv_in.cur_buf = ctx->cur_buf;
        pv_in.nxt_buf = ctx->nxt_buf_v;
        pv_in.safe_tile1 = safe_tile1;
        ckc_gfx950_attn2d_emit_pv_bucket(ctx, &pv_in);
    }

    (void)QK_K_STEP;
    return;
}
