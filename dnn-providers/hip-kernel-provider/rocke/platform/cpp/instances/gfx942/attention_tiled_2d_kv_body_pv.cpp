// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_gfx942_attention_tiled_2d_kv_body_pv.c
 *
 * CHUNKED PORT -- KV-LOOP BODY: PV bucket of
 * rocke/instances/gfx942/attention_tiled_2d.py::_emit_kv_body, the back half
 * (Python lines 4540-5052):
 *
 *   - ``acc *= alpha`` online rescale + ``acc += P @ V``
 *   - the 32x32 transposed PV path (O^T = V^T @ P^T) via the gfx942-legal
 *     16x16x16 / 32x32x8 MFMA atoms and the register-P^T consume
 *     (_apply_transposed_pv_regs)
 *   - the gfx942 non-transpose V B-operand built from strided LDS reads that
 *     reproduce the MFMA distribution (_strided_v_b_operand)
 *   - the narrow 16x16x16 / 16x16x32 PV consumer (per-atom acc scale + add)
 *   - the scf_yield carry assembly and the scf_for_iter loop driver wiring
 *     (_phases full/boundary split is a single scf_for_iter on gfx942).
 *
 * Binds ONLY to rocke/instance_gfx942_attention_tiled_2d_internal.h (the shared
 * ctx + peer phase prototypes), rocke/instance_gfx942_attention_tiled_2d.h, the
 * helper_*.h family, and rocke/ir.h. Builder calls are byte-identical to the
 * Python span. Peer closures (V loaders, acc helpers, the rest of _emit_kv_body)
 * are reached through the internal header; symbols this TU does not own are left
 * to the linker (peer-unresolved is expected for -fsyntax-only).
 *
 * Lifetime: every IR node is arena-owned (rocke_ir_builder_t.arena).
 */

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "rocke/helper_helper_rocke.helpers.attention.h" /* rocke_mfma_16x16x16_for_dtype */
#include "rocke/helper_rocke.helpers.schedule.h" /* ROCKE_SCHED_DS_READ / ROCKE_SCHED_MFMA, T8 */
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"
#include "rocke/ir.h"

/* ============================================================ *
 *  local builder aliases (mirror the Python ``b.<op>`` surface)
 * ============================================================ */
#define B (ctx->b)
#define DT (ctx->dtype)

/* Lane decode recomputed from ctx->lane, byte-identical to the Python prologue
 * (lines 2014-2017). The closures capture these from the enclosing scope; in C
 * we recompute the same SSA div/mod each time they are needed inside a phase
 * function (the lowerer/CSE folds the repeats, matching Python LICM). */
static rocke_value_t* lane_rg_of(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return rocke_b_div(B, ctx->lane, rocke_b_const_i32(B, 16));
}
static rocke_value_t* lane_col_of(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_col_v != NULL)
        return ctx->lane_col_v;
    return rocke_b_mod(B, ctx->lane, rocke_b_const_i32(B, 16));
}
static rocke_value_t* lane_half32_of(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_half32_v != NULL)
        return ctx->lane_half32_v;
    return rocke_b_div(B, ctx->lane, rocke_b_const_i32(B, 32));
}
static rocke_value_t* lane_col32_of(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_col32_v != NULL)
        return ctx->lane_col32_v;
    return rocke_b_mod(B, ctx->lane, rocke_b_const_i32(B, 32));
}

static bool dtype_is(const rocke_type_t* dt, const char* nm)
{
    return dt != NULL && dt->name != NULL && nm != NULL && (strcmp(dt->name, nm) == 0);
}

/* ---- MFMA dtype dispatch (helpers/attention.py mfma_*_for_dtype) ---- *
 * Only 16x16x16 has a ported C helper; the wide-K atoms dispatch inline over
 * the ISA-named wrappers in ir.h, faithfully reproducing the Python raise on an
 * unsupported dtype. */

static rocke_value_t* mfma_16x16x16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                    rocke_value_t* a,
                                    rocke_value_t* bv,
                                    rocke_value_t* c)
{
    return rocke_mfma_16x16x16_for_dtype(B, DT, a, bv, c);
}

static rocke_value_t* mfma_16x16x32(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                    rocke_value_t* a,
                                    rocke_value_t* bv,
                                    rocke_value_t* c)
{
    if(dtype_is(DT, "f16"))
    {
        return rocke_b_mfma_f32_16x16x32_f16(B, a, bv, c);
    }
    if(dtype_is(DT, "bf16"))
    {
        return rocke_b_mfma_f32_16x16x32_bf16(B, a, bv, c);
    }
    /* Python: raise ValueError("unsupported MFMA 16x16x32 dtype ...") */
    return NULL;
}

static rocke_value_t* mfma_32x32x8(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                   rocke_value_t* a,
                                   rocke_value_t* bv,
                                   rocke_value_t* c)
{
    if(dtype_is(DT, "f16"))
    {
        return rocke_b_mfma_f32_32x32x8_f16(B, a, bv, c);
    }
    if(dtype_is(DT, "bf16"))
    {
        /* gfx942-legal wide-K bf16 K=8 atom (the `.1k` intrinsic). Mirrors
         * helpers/attention.py:mfma_32x32x8_for_dtype. */
        return rocke_b_mfma_f32_32x32x8_bf16(B, a, bv, c);
    }
    /* Python: raise ValueError("...32x32x8 dtype ... (fp16/bf16 only)") */
    return NULL;
}

static rocke_value_t* mfma_32x32x16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                    rocke_value_t* a,
                                    rocke_value_t* bv,
                                    rocke_value_t* c)
{
    if(dtype_is(DT, "f16"))
    {
        return rocke_b_mfma_f32_32x32x16_f16(B, a, bv, c);
    }
    if(dtype_is(DT, "bf16"))
    {
        return rocke_b_mfma_f32_32x32x16_bf16(B, a, bv, c);
    }
    return NULL;
}

/* pv32_v_load_paired (helpers/attention.py:760) is a gfx950-only
 * ds_read_tr16_b64 transpose-read promotion reached only on the
 * TRANSPOSED_HALF_LOCAL_PV branch (unreachable on gfx942, where the 32x32 path
 * is the K=8 register-P^T form). Faithful port of the Python helper so the
 * structurally-kept branch links and is byte-faithful where it is reached;
 * defined static here as it has a single call site (the dead K=16 branch). */
static rocke_value_t* rocke_pv32_v_load_paired(rocke_ir_builder_t* b,
                                               rocke_value_t* V_lds,
                                               rocke_value_t* v_buf,
                                               int n,
                                               int k,
                                               rocke_value_t* lane_half32,
                                               rocke_value_t* lane_col32,
                                               const rocke_type_t* dtype)
{
    /* All sub-expressions are bound in Python source (left-to-right) order so
     * C's unspecified arg-eval order does not shuffle the SSA value numbering. */
    /* col_group16 = (lane_col32 / 16) * 16. Python emits the div (with its const
     * 16) BEFORE the mul's const 16; bind the div first so C's arg-eval order
     * does not allocate the mul's const ahead of the div and shift its %value. */
    rocke_value_t* col_div16 = rocke_b_div(b, lane_col32, rocke_b_const_i32(b, 16));
    rocke_value_t* col_group16 = rocke_b_mul(b, col_div16, rocke_b_const_i32(b, 16));
    /* tr_col32 = col_group16 + (lane_col32 % 4) * 4. Bind the mod before the
     * mul's const so C arg-eval keeps Python's value order. */
    rocke_value_t* tr_col_mod = rocke_b_mod(b, lane_col32, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_col_rhs = rocke_b_mul(b, tr_col_mod, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_col32 = rocke_b_add(b, col_group16, tr_col_rhs);
    /* tr_row_base32 = (k*16 + lane_half32*4) + ((lane_col32 / 4) % 4). Python
     * evaluates the inner add (const(k*16), mul(lane_half32,4)) BEFORE the mod. */
    rocke_value_t* tr_row_k = rocke_b_const_i32(b, k * 16);
    rocke_value_t* tr_row_inner
        = rocke_b_add(b, tr_row_k, rocke_b_mul(b, lane_half32, rocke_b_const_i32(b, 4)));
    rocke_value_t* tr_row_div4 = rocke_b_div(b, lane_col32, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_row_mod = rocke_b_mod(b, tr_row_div4, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_row_base32 = rocke_b_add(b, tr_row_inner, tr_row_mod);

    /* Python computes the col = add(const(n*32), tr_col32) INLINE inside EACH
     * ds_read call (helpers/attention.py 811-813, 818-820), so it is emitted
     * TWICE -- once per read. Mirror that (do NOT hoist) for value numbering. */
    rocke_value_t* col0 = rocke_b_add(b, rocke_b_const_i32(b, n * 32), tr_col32);
    rocke_value_t* idx0[3] = {v_buf, tr_row_base32, col0};
    rocke_value_t* A_r0 = rocke_b_ds_read_tr16_b64(b, V_lds, idx0, 3, dtype);
    rocke_value_t* row1 = rocke_b_add(b, tr_row_base32, rocke_b_const_i32(b, 8));
    rocke_value_t* col1 = rocke_b_add(b, rocke_b_const_i32(b, n * 32), tr_col32);
    rocke_value_t* idx1[3] = {v_buf, row1, col1};
    rocke_value_t* A_r1 = rocke_b_ds_read_tr16_b64(b, V_lds, idx1, 3, dtype);
    return rocke_b_vec_concat(b, A_r0, A_r1);
}

/* ============================================================ *
 *  _strided_v_b_operand (Python lines 4900-4916)
 *
 *  <4 x dtype> PV B operand for K-iter ``k_iter`` via strided LDS reads,
 *  reproducing the per-lane (row, col) mapping a ds_read_tr16_b64 would deliver
 *  for a 16x16x16 atom over V_lds at (row base = k*16, col base = n*16). Each of
 *  the 4 elements is a distinct V row; _v_load1 applies the swizzle slot mapping.
 * ============================================================ */
/* v_n_col / v_k_chunk_base / v_buf are computed ONCE per N-tile by the caller
 * (Python hoists them out of the PV k-loop, lines 4897-4919); passed in here so
 * we don't re-emit the same SSA values on every k iteration. */
rocke_value_t* rocke_gfx942_attn2d_strided_v_b_operand(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                       int k_iter,
                                                       rocke_value_t* v_n_col,
                                                       rocke_value_t* v_k_chunk_base,
                                                       rocke_value_t* v_buf)
{
    rocke_value_t* bv = rocke_b_zero_vec(B, DT, 4);
    for(int j = 0; j < 4; ++j)
    {
        rocke_value_t* v_row
            = rocke_b_add(B, rocke_b_const_i32(B, k_iter * 16 + j), v_k_chunk_base);
        rocke_value_t* loaded = rocke_gfx942_attn2d_v_load1(ctx, v_buf, v_row, v_n_col);
        rocke_value_t* elem = rocke_b_vec_extract(B, loaded, 0);
        bv = rocke_b_vec_insert(B, bv, elem, j);
    }
    return bv;
}

/* ============================================================ *
 *  _apply_transposed_pv_regs (Python lines 4573-4664/4743)
 *
 *  Register-P^T PV: O^T = V^T @ P^T. A operand = V^T[M=d, K=kv-token]; B operand
 *  = P^T[K=kv-token, N=query], consumed DIRECTLY from registers (no P_lds), with
 *  cross-half rows fetched via a single lane^32 exchange. gfx942 takes the
 *  USE_MFMA_32X32X8 (K=8) branch; the K=16 branches are kept structurally for
 *  parity with the gfx950 file.
 *
 *  ``p_regs`` is the flat [p_tile][reg] PT32 register set passed as a
 *  contiguous (p_count == n_tiles * REGS_PER_LANE) array; index as
 *  p_regs[p_tile * REGS_PER_LANE + reg].
 * ============================================================ */
rocke_value_t* rocke_gfx942_attn2d_apply_transposed_pv_regs(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                            rocke_value_t* acc32,
                                                            int n,
                                                            rocke_value_t* const* p_regs,
                                                            int p_count)
{
    const int RPL = ctx->REGS_PER_LANE;
    const int T = ctx->T;

    rocke_value_t* lane_half32 = lane_half32_of(ctx);
    /* Python's transposed PV reads the Q32-gather-reassigned lane_col32 (gfx950
     * 2329 / 3235), not the prologue one. Use the published value if present. */
    rocke_value_t* lane_col32
        = (ctx->lane_col32_q32_v != NULL) ? ctx->lane_col32_q32_v : lane_col32_of(ctx);
    /* Reuse the v_buf / use_hi emitted ONCE by the PV bucket before the scaling
     * loop (Python emits them once outside _apply_transposed_pv_regs, gfx950
     * 3231-3232). Fall back to fresh values if not pre-seeded. */
    rocke_value_t* v_buf = (ctx->pv_v_buf_v != NULL) ? ctx->pv_v_buf_v : rocke_b_const_i32(B, 0);
    rocke_value_t* use_hi = (ctx->pv_use_hi_v != NULL)
                                ? ctx->pv_use_hi_v
                                : rocke_b_cmp_eq(B, lane_half32, rocke_b_const_i32(B, 1));

    /* v_dim32 = n*32 + lane_col32 (line 4574). Python emits const(n*32) first. */
    rocke_value_t* v_dim32_base = rocke_b_const_i32(B, n * 32);
    rocke_value_t* v_dim32 = rocke_b_add(B, v_dim32_base, lane_col32);

    (void)p_count;

    if(ctx->USE_MFMA_32X32X8)
    {
        /* gfx942-legal K=8 transposed PV (lines 4575-4664). */
        for(int k = 0; k < T / 8; ++k)
        {
            rocke_value_t* A_v_t;
            if(ctx->TRANSPOSED_V)
            {
                /* CK technique #1: V stored K-contiguous in V_lds[HD, T+pad];
                 * the 4 tokens are contiguous along the inner axis -> one wide
                 * read (lines 4596-4628). The naive token base is
                 * k*8 + lane_half32*4. */
                rocke_value_t* tok_base
                    = rocke_b_add(B,
                                  rocke_b_const_i32(B, k * 8),
                                  rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 4)));
                if(ctx->CFV_SCALAR_READ)
                {
                    /* ISOLATION: 4 scalar n=1 reads, same values/order. */
                    rocke_value_t* av[4];
                    for(int kk = 0; kk < 4; ++kk)
                    {
                        rocke_value_t* t = rocke_b_add(B, tok_base, rocke_b_const_i32(B, kk));
                        rocke_value_t* v1 = rocke_gfx942_attn2d_v_t_load(ctx, v_dim32, t, 1);
                        av[kk] = rocke_b_vec_extract(B, v1, 0);
                    }
                    A_v_t = rocke_b_vec_pack(B, av, 4, DT);
                }
                else
                {
                    A_v_t = rocke_gfx942_attn2d_v_t_load(ctx, v_dim32, tok_base, 4);
                }
            }
            else
            {
                rocke_value_t* a_v_elems[4];
                for(int kk = 0; kk < 4; ++kk)
                {
                    /* v_row = (k*8 + kk) + lane_half32*4 (lines 4632-4635).
                     * Python emits const(k*8+kk) BEFORE the mul (left-to-right);
                     * bind it first so C's right-to-left arg eval keeps the SSA
                     * numbering (the const folds to an immediate but still
                     * advances Python's value counter). */
                    rocke_value_t* v_row_base = rocke_b_const_i32(B, k * 8 + kk);
                    rocke_value_t* v_row = rocke_b_add(
                        B, v_row_base, rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 4)));
                    rocke_value_t* idx[3] = {v_buf, v_row, v_dim32};
                    rocke_value_t* v1 = rocke_b_smem_load_vN(B, ctx->V_lds, idx, 3, DT, 1);
                    a_v_elems[kk] = rocke_b_vec_extract(B, v1, 0);
                }
                A_v_t = rocke_b_vec_pack(B, a_v_elems, 4, DT);
            }

            rocke_value_t* b_p_elems[4];
            for(int kk = 0; kk < 4; ++kk)
            {
                /* Low half wants K-row k0; high half wants k1 (lines 4644-4661). */
                int k0 = k * 8 + kk;
                int k1 = k * 8 + 4 + kk;
                int p_tile0 = k0 / 32;
                int p_tile1 = k1 / 32;
                int row0 = k0 % 32;
                int row1 = k1 % 32;
                int owner_half0 = (row0 % 8) / 4;
                int owner_half1 = (row1 % 8) / 4;
                int reg0 = (row0 / 8) * 4 + (row0 % 4);
                int reg1 = (row1 / 8) * 4 + (row1 % 4);
                rocke_value_t* p0 = p_regs[p_tile0 * RPL + reg0];
                rocke_value_t* p1 = p_regs[p_tile1 * RPL + reg1];
                if(owner_half0 == 1)
                {
                    p0 = rocke_b_warp_shuffle_xor(B, p0, 32);
                }
                if(owner_half1 == 0)
                {
                    p1 = rocke_b_warp_shuffle_xor(B, p1, 32);
                }
                rocke_value_t* p_val = rocke_b_select(B, use_hi, p1, p0);
                b_p_elems[kk] = rocke_b_cast_f32_to(B, p_val, DT);
            }
            rocke_value_t* B_p_t = rocke_b_vec_pack(B, b_p_elems, 4, DT);
            acc32 = mfma_32x32x8(ctx, A_v_t, B_p_t, acc32);
            if(ctx->USE_QK_PV_SCHED_GROUP_BARRIER)
            {
                rocke_b_sched_group_barrier(B, ROCKE_SCHED_DS_READ, 4, 1);
                rocke_b_sched_group_barrier(B, ROCKE_SCHED_MFMA, 1, 1);
            }
        }
        return acc32;
    }

    /* K=16 transposed branches -- gfx950-only, kept structurally (lines
     * 4665-4743). Unreachable on gfx942 (USE_MFMA_32X32X8 is the live path). */
    for(int k = 0; k < T / 16; ++k)
    {
        if(ctx->TRANSPOSED_HALF_LOCAL_PV)
        {
            rocke_value_t* A_v_t
                = rocke_pv32_v_load_paired(B, ctx->V_lds, v_buf, n, k, lane_half32, lane_col32, DT);
            rocke_value_t* b_p_elems[8];
            for(int kk = 0; kk < 8; ++kk)
            {
                int local_in_group = kk % 4;
                int band = kk / 4;
                int p_tile = (k * 16 + band * 8 + local_in_group) / 32;
                int row_static = (k * 16 + band * 8 + local_in_group) % 32;
                int preg = (row_static / 8) * 4 + (row_static % 4);
                b_p_elems[kk] = rocke_b_cast_f32_to(B, p_regs[p_tile * RPL + preg], DT);
            }
            rocke_value_t* B_p_t = rocke_b_vec_pack(B, b_p_elems, 8, DT);
            acc32 = mfma_32x32x16(ctx, A_v_t, B_p_t, acc32);
        }
        else
        {
            rocke_value_t* a_v_elems[8];
            for(int kk = 0; kk < 8; ++kk)
            {
                int k_static = k * 16 + kk;
                /* Python emits const(k_static) before the mul (left-to-right);
                 * bind it first so C arg-eval order keeps the value numbering. */
                rocke_value_t* v_row_base = rocke_b_const_i32(B, k_static);
                rocke_value_t* v_row = rocke_b_add(
                    B, v_row_base, rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 8)));
                rocke_value_t* idx[3] = {v_buf, v_row, v_dim32};
                rocke_value_t* v1 = rocke_b_smem_load_vN(B, ctx->V_lds, idx, 3, DT, 1);
                a_v_elems[kk] = rocke_b_vec_extract(B, v1, 0);
            }
            rocke_value_t* b_p_elems[8];
            for(int kk = 0; kk < 8; ++kk)
            {
                int k_static = k * 16 + kk;
                int k0 = k_static;
                int k1 = k_static + 8;
                int p_tile0 = k0 / 32;
                int p_tile1 = k1 / 32;
                int row0 = k0 % 32;
                int row1 = k1 % 32;
                int owner_half0 = (row0 % 8) / 4;
                int owner_half1 = (row1 % 8) / 4;
                int reg0 = (row0 / 8) * 4 + (row0 % 4);
                int reg1 = (row1 / 8) * 4 + (row1 % 4);
                rocke_value_t* p0 = p_regs[p_tile0 * RPL + reg0];
                rocke_value_t* p1 = p_regs[p_tile1 * RPL + reg1];
                if(owner_half0 == 1)
                {
                    p0 = rocke_b_warp_shuffle_xor(B, p0, 32);
                }
                if(owner_half1 == 0)
                {
                    p1 = rocke_b_warp_shuffle_xor(B, p1, 32);
                }
                rocke_value_t* p_val = rocke_b_select(B, use_hi, p1, p0);
                b_p_elems[kk] = rocke_b_cast_f32_to(B, p_val, DT);
            }
            rocke_value_t* A_v_t = rocke_b_vec_pack(B, a_v_elems, 8, DT);
            rocke_value_t* B_p_t = rocke_b_vec_pack(B, b_p_elems, 8, DT);
            acc32 = mfma_32x32x16(ctx, A_v_t, B_p_t, acc32);
        }
    }
    return acc32;
}

/* ============================================================ *
 *  PV bucket of _emit_kv_body (Python lines 4540-5041)
 *
 *  Reads the already-rescaled softmax state from ctx (m_cur/l_cur/acc_cur set by
 *  the QK/softmax bucket peer) plus the PT32 register set + alpha_regs +
 *  p_regs_f32 it threads in via ctx scratch, computes acc *= alpha + acc += P@V,
 *  assembles the scf_yield carry into ctx->out_carry, and emits scf_yield.
 *
 *  This worker is invoked from the _emit_kv_body driver (peer-owned); the
 *  softmax-derived per-reg lists (alpha_regs, new_l_vals, m_new, p_regs_f32, the
 *  PT32 register groups) are passed in by the caller through the parameter block
 *  so this TU owns ONLY the PV math + yield/loop wiring. The long QK/softmax
 *  prelude is the peer's bucket; here we stub the inbound lists from ctx scratch
 *  where the shared header does not (yet) carry them, so the file links in
 *  isolation.
 * ============================================================ */

/* rocke_gfx942_attn2d_pv_inputs_t is declared in the internal header so the peer
 * QK/softmax bucket can fill it and hand it to rocke_gfx942_attn2d_emit_pv_bucket. */

/* Pack helpers (Python _pack_p_a16 / _pack_p_a32 via the peer phase functions).
 * The register-PV A operand for a 16x16x16 / 16x16x32 atom. */
static rocke_value_t* pack_p_a16(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* const* regs4)
{
    return rocke_gfx942_attn2d_pack_p_a16(ctx, regs4, 4);
}
static rocke_value_t* pack_p_a32(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* const* regs0_4,
                                 rocke_value_t* const* regs1_4)
{
    return rocke_gfx942_attn2d_pack_p_a32(ctx, regs0_4, regs1_4, 4);
}

void rocke_gfx942_attn2d_emit_pv_bucket(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                        const rocke_gfx942_attn2d_pv_inputs_t* in)
{
    const int ACC_N_TILES = ctx->ACC_N_TILES;
    const int ACC_M_ATOMS = ctx->ACC_M_ATOMS;
    const int REGS_PER_LANE = ctx->REGS_PER_LANE;
    const int M_ATOMS = ctx->M_ATOMS_PER_WARP;
    const int PV_N_TILES = ctx->PV_N_TILES;
    const int PV_K_ITERS = ctx->PV_K_ITERS;
    const rocke_type_t* F32 = rocke_f32();

    /* new_acc[ACC_N_TILES * ACC_M_ATOMS] */
    rocke_value_t* new_acc[ROCKE_GFX942_ATTN2D_MAX_ACCS];
    for(int i = 0; i < ACC_N_TILES * ACC_M_ATOMS; ++i)
    {
        new_acc[i] = NULL;
    }

    if(ctx->USE_MFMA_32X32)
    {
        if(ctx->TRANSPOSED_QK_32X32)
        {
            /* Python emits v_buf = const(0) + use_hi = cmp_eq(lane_half32, 1)
             * ONCE here, BEFORE the acc-scaling loop (gfx950 3231-3232);
             * _apply_transposed_pv_regs reuses them. Cache so the per-n apply
             * calls below reuse the same SSA values. */
            ctx->pv_v_buf_v = rocke_b_const_i32(B, 0);
            ctx->pv_use_hi_v = rocke_b_cmp_eq(B, lane_half32_of(ctx), rocke_b_const_i32(B, 1));
            for(int n = 0; n < ACC_N_TILES; ++n)
            {
                rocke_value_t* old_acc = rocke_gfx942_attn2d_acc_get(ctx, n, 0);
                rocke_value_t* alpha_t = in->alpha_regs[0];
                rocke_value_t* scaled[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
                for(int reg = 0; reg < REGS_PER_LANE; ++reg)
                {
                    rocke_value_t* e = rocke_b_vec_extract(B, old_acc, reg);
                    rocke_value_t* a = ctx->TRANSPOSED_SCALAR_STATE ? alpha_t : in->alpha_regs[reg];
                    scaled[reg] = rocke_b_fmul(B, e, a);
                }
                rocke_value_t* acc32 = rocke_b_vec_pack(B, scaled, REGS_PER_LANE, F32);
                acc32 = rocke_gfx942_attn2d_apply_transposed_pv_regs(
                    ctx, acc32, n, in->pt32_g0, in->pt32_count);
                new_acc[n] = acc32;
            }
            if(ctx->GROUPED_KV2)
            {
                /* V_lds single-buffered: finish tile0 reads, then reissue
                 * tile1's V and accumulate (lines 4760-4770). */
                rocke_b_s_waitcnt(B, -1, 0, -1);
                rocke_b_sync(B);
                rocke_gfx942_attn2d_issue_v(ctx, in->safe_tile1, in->nxt_buf);
                rocke_b_s_waitcnt(B, 0, 0, -1);
                rocke_b_sync(B);
                for(int n = 0; n < ACC_N_TILES; ++n)
                {
                    new_acc[n] = rocke_gfx942_attn2d_apply_transposed_pv_regs(
                        ctx, new_acc[n], n, in->pt32_g1, in->pt32_count);
                }
            }
        }
        else if(ctx->USE_MFMA_32X32X8)
        {
            /* gfx942 32x32x8 PV (O = P @ V): P_lds bridge + naive strided V
             * (lines 4771-4815). Python's lane_col32 is the Q-gather-reassigned
             * value (line 3504); lane_half32 is the prologue one (line 2016). */
            rocke_value_t* v_buf = rocke_b_const_i32(B, 0);
            rocke_value_t* lane_col32
                = (ctx->lane_col32_q32_v != NULL) ? ctx->lane_col32_q32_v : lane_col32_of(ctx);
            rocke_value_t* lane_half32 = lane_half32_of(ctx);
            rocke_value_t* p_row32 = rocke_b_add(B, ctx->wave_row_base, lane_col32);
            for(int n = 0; n < ACC_N_TILES; ++n)
            {
                rocke_value_t* old_acc = rocke_gfx942_attn2d_acc_get(ctx, n, 0);
                rocke_value_t* scaled[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
                for(int reg = 0; reg < REGS_PER_LANE; ++reg)
                {
                    rocke_value_t* e = rocke_b_vec_extract(B, old_acc, reg);
                    scaled[reg] = rocke_b_fmul(B, e, in->alpha_regs[reg]);
                }
                rocke_value_t* acc32 = rocke_b_vec_pack(B, scaled, REGS_PER_LANE, F32);
                rocke_value_t* v_col32 = rocke_b_add(B, rocke_b_const_i32(B, n * 32), lane_col32);
                for(int k = 0; k < PV_K_ITERS; ++k)
                {
                    /* Python emits const(k*8) BEFORE the mul (left-to-right);
                     * bind const first so C's right-to-left arg eval does not
                     * allocate the mul ahead of it and shift the %value. */
                    rocke_value_t* p_off_base = rocke_b_const_i32(B, k * 8);
                    rocke_value_t* p_off32 = rocke_b_add(
                        B, p_off_base, rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 4)));
                    rocke_value_t* pidx[2] = {p_row32, p_off32};
                    rocke_value_t* A_p32 = rocke_b_smem_load_vN(B, ctx->P_lds, pidx, 2, DT, 4);
                    rocke_value_t* k_base = rocke_b_const_i32(B, k * 8);
                    rocke_value_t* k_row_base = rocke_b_add(
                        B, k_base, rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 4)));
                    rocke_value_t* B_v32 = rocke_b_zero_vec(B, DT, 4);
                    for(int j = 0; j < 4; ++j)
                    {
                        rocke_value_t* v_row = rocke_b_add(B, k_row_base, rocke_b_const_i32(B, j));
                        rocke_value_t* loaded
                            = rocke_gfx942_attn2d_v_load1(ctx, v_buf, v_row, v_col32);
                        rocke_value_t* elem = rocke_b_vec_extract(B, loaded, 0);
                        B_v32 = rocke_b_vec_insert(B, B_v32, elem, j);
                    }
                    acc32 = mfma_32x32x8(ctx, A_p32, B_v32, acc32);
                }
                new_acc[n] = acc32;
            }
        }
        else
        {
            /* Transitional gfx950-only 32x32x16 PV (lines 4816-4874). Kept for
             * structural parity; unreachable on gfx942. */
            rocke_value_t* v_buf = rocke_b_const_i32(B, 0);
            rocke_value_t* lane_col32 = lane_col32_of(ctx);
            rocke_value_t* lane_half32 = lane_half32_of(ctx);
            for(int n = 0; n < ACC_N_TILES; ++n)
            {
                rocke_value_t* old_acc = rocke_gfx942_attn2d_acc_get(ctx, n, 0);
                rocke_value_t* scaled[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
                for(int reg = 0; reg < REGS_PER_LANE; ++reg)
                {
                    rocke_value_t* e = rocke_b_vec_extract(B, old_acc, reg);
                    scaled[reg] = rocke_b_fmul(B, e, in->alpha_regs[reg]);
                }
                rocke_value_t* acc32 = rocke_b_vec_pack(B, scaled, REGS_PER_LANE, F32);
                for(int k = 0; k < ctx->T / 16; ++k)
                {
                    rocke_value_t* p_off32
                        = rocke_b_add(B,
                                      rocke_b_const_i32(B, k * 16),
                                      rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 8)));
                    rocke_value_t* p_row32 = rocke_b_add(B, ctx->wave_row_base, lane_col32);
                    rocke_value_t* pidx[2] = {p_row32, p_off32};
                    rocke_value_t* A_p32 = rocke_b_smem_load_vN(B, ctx->P_lds, pidx, 2, DT, 8);
                    rocke_value_t* col_group16
                        = rocke_b_mul(B,
                                      rocke_b_div(B, lane_col32, rocke_b_const_i32(B, 16)),
                                      rocke_b_const_i32(B, 16));
                    rocke_value_t* tr_col32 = rocke_b_add(
                        B,
                        col_group16,
                        rocke_b_mul(B,
                                    rocke_b_mod(B, lane_col32, rocke_b_const_i32(B, 4)),
                                    rocke_b_const_i32(B, 4)));
                    rocke_value_t* tr_row_base32 = rocke_b_add(
                        B,
                        rocke_b_add(B,
                                    rocke_b_const_i32(B, k * 16),
                                    rocke_b_mul(B, lane_half32, rocke_b_const_i32(B, 8))),
                        rocke_b_mod(B,
                                    rocke_b_div(B, lane_col32, rocke_b_const_i32(B, 4)),
                                    rocke_b_const_i32(B, 4)));
                    rocke_value_t* col0 = rocke_b_add(B, rocke_b_const_i32(B, n * 32), tr_col32);
                    rocke_value_t* r0idx[3] = {v_buf, tr_row_base32, col0};
                    rocke_value_t* B32_r0 = rocke_b_ds_read_tr16_b64(B, ctx->V_lds, r0idx, 3, DT);
                    rocke_value_t* r1row = rocke_b_add(B, tr_row_base32, rocke_b_const_i32(B, 4));
                    rocke_value_t* r1idx[3] = {v_buf, r1row, col0};
                    rocke_value_t* B32_r1 = rocke_b_ds_read_tr16_b64(B, ctx->V_lds, r1idx, 3, DT);
                    rocke_value_t* B_v32 = rocke_b_vec_concat(B, B32_r0, B32_r1);
                    acc32 = mfma_32x32x16(ctx, A_p32, B_v32, acc32);
                }
                new_acc[n] = acc32;
            }
        }
    }

    /* Narrow 16x16x16 PV consumer (lines 4875-5031). Loop bound is 0 when the
     * 32x32 path already filled new_acc. */
    {
        rocke_value_t* lane_col = lane_col_of(ctx);
        const int narrow_n = ctx->USE_MFMA_32X32 ? 0 : PV_N_TILES;
        for(int n = 0; n < narrow_n; ++n)
        {
            rocke_value_t* acc_per_atom[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
            for(int atom = 0; atom < M_ATOMS; ++atom)
            {
                rocke_value_t* scaled[4];
                for(int in_atom = 0; in_atom < 4; ++in_atom)
                {
                    int reg = atom * 4 + in_atom;
                    rocke_value_t* e = rocke_b_vec_extract(
                        B, rocke_gfx942_attn2d_acc_get(ctx, n, atom), in_atom);
                    scaled[in_atom] = rocke_b_fmul(B, e, in->alpha_regs[reg]);
                }
                acc_per_atom[atom] = rocke_b_vec_pack(B, scaled, 4, F32);
            }

            /* v_n_col = n*16 + lane_col ; v_k_chunk_base = lane_rg*4 ; v_buf = 0
             * computed ONCE per N-tile (Python 4897-4919, hoisted out of the
             * k-loop) and threaded into _strided_v_b_operand. */
            rocke_value_t* vn_n = rocke_b_const_i32(B, n);
            rocke_value_t* vn_16 = rocke_b_const_i32(B, 16);
            rocke_value_t* v_n_col = rocke_b_add(B, rocke_b_mul(B, vn_n, vn_16), lane_col);
            rocke_value_t* v_k_chunk_base
                = rocke_b_mul(B, lane_rg_of(ctx), rocke_b_const_i32(B, 4));
            rocke_value_t* v_buf = rocke_b_const_i32(B, 0);

            /* K=16 narrow atom (gfx942): build <4 x dtype> B operand from 4
             * strided LDS loads via _strided_v_b_operand (lines 5009-5029). */
            for(int k = 0; k < PV_K_ITERS; ++k)
            {
                /* p_off = k*16 + lane_rg*4. Python emits a FRESH lane_rg*4 here
                 * (not the hoisted v_k_chunk_base), so re-emit the mul. Python
                 * evaluates the left const(k*16) BEFORE the mul (left-to-right);
                 * bind it to a temp so C arg-eval order does not allocate the
                 * mul's const ahead of it and shift the mul's %value. */
                rocke_value_t* p_off_base = rocke_b_const_i32(B, k * 16);
                rocke_value_t* p_off = rocke_b_add(
                    B, p_off_base, rocke_b_mul(B, lane_rg_of(ctx), rocke_b_const_i32(B, 4)));
                rocke_value_t* B_v = rocke_gfx942_attn2d_strided_v_b_operand(
                    ctx, k, v_n_col, v_k_chunk_base, v_buf);
                for(int atom = 0; atom < M_ATOMS; ++atom)
                {
                    rocke_value_t* A_p;
                    if(ctx->REGISTER_PV)
                    {
                        rocke_value_t* regs4[4];
                        for(int r = 0; r < 4; ++r)
                        {
                            regs4[r] = in->p_regs_f32[(atom * 4 + r) * in->p_regs_f32_stride + k];
                        }
                        A_p = pack_p_a16(ctx, regs4);
                    }
                    else
                    {
                        rocke_value_t* p_row = rocke_b_add(
                            B,
                            ctx->wave_row_base,
                            rocke_b_add(B, rocke_b_const_i32(B, atom * 16), lane_col));
                        rocke_value_t* pidx[2] = {p_row, p_off};
                        A_p = rocke_b_smem_load_vN(B, ctx->P_lds, pidx, 2, DT, 4);
                    }
                    acc_per_atom[atom] = mfma_16x16x16(ctx, A_p, B_v, acc_per_atom[atom]);
                }
            }
            for(int atom = 0; atom < M_ATOMS; ++atom)
            {
                new_acc[n * M_ATOMS + atom] = acc_per_atom[atom];
            }
        }
        (void)pack_p_a32; /* used only on the dead gfx950 wide-K path */
        (void)mfma_16x16x32;
    }

    /* ---- assemble the scf_yield carry (lines 5033-5041) ---- */
    {
        const int S = ctx->SOFTMAX_STATE_SLOTS;
        rocke_value_t* yields[ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS];
        int y = 0;
        for(int r = 0; r < S; ++r)
        {
            yields[y++] = in->m_new[r];
            yields[y++] = in->new_l_vals[r];
        }
        for(int n = 0; n < ACC_N_TILES; ++n)
        {
            for(int atom = 0; atom < ACC_M_ATOMS; ++atom)
            {
                yields[y++] = new_acc[n * ACC_M_ATOMS + atom];
            }
        }
        /* GROUPED_KV2 keeps V single-buffered: yield cur_buf, else nxt_buf. */
        yields[y++] = ctx->GROUPED_KV2 ? in->cur_buf : in->nxt_buf;

        /* mirror into ctx->out_carry for the driver, then emit. */
        for(int i = 0; i < y; ++i)
        {
            ctx->out_carry[i] = yields[i];
        }
        ctx->out_carry_count = y;
        rocke_b_scf_yield(B, (rocke_value_t* const*)yields, y);
    }
}

/* ============================================================ *
 *  Loop driver wiring (Python lines 5043-5052)
 *
 *  ``_phases`` is a single (tile_start, tile_end, skip_mask=False) scf_for_iter
 *  on every gfx942 path (the no-SW transposed full/boundary split is a gfx950
 *  shape). The softmax/acc/buffer carry threads through scf_for_iter results so
 *  a second phase (if any) resumes at the same K/V slot + online-softmax state.
 *
 *  This emits the loop op, enters its body region, sets the live-carry ctx
 *  fields, calls the peer _emit_kv_body bucket (which runs QK/softmax then the
 *  PV bucket above), and leaves the region. The kvloop handle (its results) is
 *  consumed by the epilogue peer via ctx scratch.
 * ============================================================ */
rocke_for_t rocke_gfx942_attn2d_drive_kv_loop(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    /* Build the (name, init) iter-arg list from ctx->iter_args. The carry slot
     * names ("m0","l0",...,"acc0"/"acc0a0","cur_buf") are emitted verbatim as the
     * loop's phi-node names, so they must come from ctx->iter_args_names to match
     * the Python iter_args tuple names byte-for-byte. */
    rocke_iter_arg_t iargs[ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS];
    static const char* const kIvName = "kv_tile";
    for(int i = 0; i < ctx->iter_args_count; ++i)
    {
        iargs[i].name = ctx->iter_args_names[i];
        iargs[i].init = ctx->iter_args[i];
    }

    rocke_for_t kvloop = rocke_b_scf_for_iter(B,
                                              ctx->tile_start,
                                              ctx->tile_end,
                                              ctx->kv_step,
                                              iargs,
                                              ctx->iter_args_count,
                                              kIvName,
                                              /*unroll=*/false,
                                              /*elide_trailing_barrier=*/true);

    /* Enter the body region; expose iv + carry, run the body bucket. */
    rocke_b_region_enter(B, kvloop.body);
    ctx->kv_tile_iv = kvloop.iv;
    ctx->skip_mask = false;

    /* Unpack the loop-carried values into the live-carry ctx fields the inner
     * QK/softmax/PV closures read (mirrors the Python ``with kvloop as
     * (kv_tile_iv, carry)`` unpack at line 5051). */
    {
        const int S = ctx->SOFTMAX_STATE_SLOTS;
        const int ml = ctx->ml_count;
        rocke_value_t** iv = kvloop.iter_vars;
        for(int r = 0; r < S; ++r)
        {
            ctx->m_cur[r] = iv[2 * r];
            ctx->l_cur[r] = iv[2 * r + 1];
        }
        int num_accs = ctx->iter_args_count - ml - 1; /* trailing buf carry */
        for(int i = 0; i < num_accs; ++i)
        {
            ctx->acc_cur[i] = iv[ml + i];
        }
        /* trailing cur_buf carry (carry[ml + num_accs]). */
        ctx->cur_buf = iv[ml + num_accs];
    }

    rocke_gfx942_attn2d_emit_kv_body(ctx);
    rocke_b_region_leave(B);
    return kvloop;
}

#undef B
#undef DT
