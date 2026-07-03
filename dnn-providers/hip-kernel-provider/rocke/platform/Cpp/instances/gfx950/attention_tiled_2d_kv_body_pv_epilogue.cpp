// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_kv_body_pv_epilogue.c
 *   -- C99 port of the back half of the gfx950 unified-attention-2d-tiled build.
 *
 * SCOPE (this TU): the KV-loop body BACK HALF + the loop driver + the epilogue,
 * faithfully tracking rocke/instances/gfx950/attention_tiled_2d.py:
 *
 *   - rocke_gfx950_attn2d_emit_pv_bucket        Python lines 3174-3572 (PV bucket)
 *       acc *= alpha; acc += P @ V via the wide 32x32x16 atom + pv_tr_reader
 *       ds_read_b64_tr_b16/b8 transpose reads, the register-P^T path, the narrow
 *       16x16x{16,32} per-atom PV, the P->A pack (_pack_p_a16/_pack_p_a32), and
 *       the scf_yield carry.
 *   - rocke_gfx950_attn2d_apply_transposed_pv_regs   Python lines 3234-3314
 *   - rocke_gfx950_attn2d_drive_kv_loop         Python line 3581-3583 (scf_for_iter)
 *   - rocke_gfx950_attn2d_emit_epilogue         Python lines 3585-3817 (epilogue)
 *
 * The P permute/pack helpers (_permute_p_c_to_a16 / _pack_p_a16 / _pack_p_a32,
 * Python 1292-1384) live in this scope and are implemented here.
 *
 * BYTE-IDENTICAL CALL ORDER. Every emitter issues the exact rocke_b_* builder
 * calls in the exact order and with the exact operands the Python body uses.
 *
 * PEER CALLS. The front half of _emit_kv_body (QK + mask + softmax) is a peer
 * (rocke_gfx950_attn2d_emit_kv_body, declared in the internal header); the loop
 * driver below calls it. The P permute/pack helpers, the acc index helpers, the
 * _issue_v loader and the module-static 32x32 C-row/col helpers are peers.
 *
 * Bindings: rocke/instance_gfx950_attention_tiled_2d_internal.h (the shared ctx +
 * peer prototypes), rocke/ir.h (the builder), the layouts + mfma_attention helper
 * headers. This TU edits no header.
 */

#include "rocke/helper_rocke.helpers.layouts.h" /* pv_tr_reader row/col  */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* 32x32x16_for_dtype    */
#include "rocke/instance_gfx950_attention_tiled_2d.h" /* mfma_32x32_c_row/_col */
#include "rocke/instance_gfx950_attention_tiled_2d_internal.h"

#include <assert.h>

/* ============================================================ *
 *  Local dtype-dispatch wrappers mirroring the Python module aliases
 *  (_mfma_16x16x32 / _mfma_16x16x16 = mfma_{16x16x32,16x16x16}_for_dtype,
 *  _mfma_32x32x16 = mfma_32x32x16_for_dtype). The narrow dispatchers select
 *  the f16 / bf16 ISA atom on dtype, exactly as the Python helper does.
 * ============================================================ */
static rocke_value_t* rocke__attn2d_mfma_16x16x16(rocke_ir_builder_t* b,
                                                  const rocke_type_t* dtype,
                                                  rocke_value_t* a,
                                                  rocke_value_t* bv,
                                                  rocke_value_t* c)
{
    if(dtype != NULL && dtype->kind == ROCKE_TYPE_SCALAR)
    {
        if(dtype->scalar == ROCKE_SCALAR_F16)
            return rocke_b_mfma_f32_16x16x16_f16(b, a, bv, c);
        if(dtype->scalar == ROCKE_SCALAR_BF16)
            return rocke_b_mfma_f32_16x16x16_bf16(b, a, bv, c);
    }
    return NULL;
}

static rocke_value_t* rocke__attn2d_mfma_16x16x32(rocke_ir_builder_t* b,
                                                  const rocke_type_t* dtype,
                                                  rocke_value_t* a,
                                                  rocke_value_t* bv,
                                                  rocke_value_t* c)
{
    if(dtype != NULL && dtype->kind == ROCKE_TYPE_SCALAR)
    {
        if(dtype->scalar == ROCKE_SCALAR_F16)
            return rocke_b_mfma_f32_16x16x32_f16(b, a, bv, c);
        if(dtype->scalar == ROCKE_SCALAR_BF16)
            return rocke_b_mfma_f32_16x16x32_bf16(b, a, bv, c);
    }
    return NULL;
}

#define MFMA_N_CONST 16 /* Python module constant MFMA_N = 16 */

/* ============================================================ *
 *  rocke_pv32_v_load_paired   (helpers/attention.py:794-857)
 *
 *  Promoted 32x32x16 PV V-load reached on the TRANSPOSED_HALF_LOCAL_PV branch:
 *  two paired ds_read_tr16_b64 transpose reads + vec_concat -> <8 x dtype> per
 *  lane. Faithful port of the Python helper (gfx942 twin has an identical port
 *  at instances/gfx942/attention_tiled_2d_kv_body_pv.cpp).
 *
 *  Every sub-expression is bound in Python source (left-to-right) order so C's
 *  unspecified arg-eval order does not shuffle the SSA value numbering.
 * ============================================================ */
static rocke_value_t* rocke_pv32_v_load_paired(rocke_ir_builder_t* b,
                                               rocke_value_t* V_lds,
                                               rocke_value_t* v_buf,
                                               int n,
                                               int k,
                                               rocke_value_t* lane_half32,
                                               rocke_value_t* lane_col32,
                                               const rocke_type_t* dtype)
{
    /* col_group16 = (lane_col32 / 16) * 16. Python emits the div (with its const
     * 16) BEFORE the mul's const 16; bind the div first. */
    rocke_value_t* col_div16 = rocke_b_div(b, lane_col32, rocke_b_const_i32(b, 16));
    rocke_value_t* col_group16 = rocke_b_mul(b, col_div16, rocke_b_const_i32(b, 16));
    /* tr_col32 = col_group16 + (lane_col32 % 4) * 4. Bind the mod first. */
    rocke_value_t* tr_col_mod = rocke_b_mod(b, lane_col32, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_col_rhs = rocke_b_mul(b, tr_col_mod, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_col32 = rocke_b_add(b, col_group16, tr_col_rhs);
    /* tr_row_base32 = (k*16 + lane_half32*4) + ((lane_col32 / 4) % 4). Python
     * evaluates the inner add (const(k*16), mul(lane_half32,4)) BEFORE the mod. */
    rocke_value_t* tr_row_k = rocke_b_const_i32(b, (int64_t)(k * 16));
    rocke_value_t* tr_row_inner
        = rocke_b_add(b, tr_row_k, rocke_b_mul(b, lane_half32, rocke_b_const_i32(b, 4)));
    rocke_value_t* tr_row_div4 = rocke_b_div(b, lane_col32, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_row_mod = rocke_b_mod(b, tr_row_div4, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_row_base32 = rocke_b_add(b, tr_row_inner, tr_row_mod);

    /* Python computes col = add(const(n*32), tr_col32) INLINE inside EACH
     * ds_read call (helpers/attention.py 844-846, 851-853), so it is emitted
     * TWICE -- once per read. Mirror that (do NOT hoist) for value numbering. */
    rocke_value_t* col0 = rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n * 32)), tr_col32);
    rocke_value_t* idx0[3] = {v_buf, tr_row_base32, col0};
    rocke_value_t* A_r0 = rocke_b_ds_read_tr16_b64(b, V_lds, idx0, 3, dtype);
    rocke_value_t* row1 = rocke_b_add(b, tr_row_base32, rocke_b_const_i32(b, 8));
    rocke_value_t* col1 = rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n * 32)), tr_col32);
    rocke_value_t* idx1[3] = {v_buf, row1, col1};
    rocke_value_t* A_r1 = rocke_b_ds_read_tr16_b64(b, V_lds, idx1, 3, dtype);
    return rocke_b_vec_concat(b, A_r0, A_r1);
}

/* ============================================================ *
 *  rocke_gfx950_attn2d_apply_transposed_pv_regs   (Python lines 3234-3314)
 *
 *  Transposed PV via registers: O^T = V^T @ P^T. For each K=16 sub-tile,
 *  assemble the V A-operand from 8 scalar V_lds loads and the P^T B-operand
 *  from the PT32 registers (with cross-half warp_shuffle_xor as needed), then
 *  one 32x32x16 MFMA into acc32.
 *
 *  ``p_regs`` is the flat [p_tile * RPL + reg] array (RPL = REGS_PER_LANE for
 *  the 32x32 path == 16); ``p_count`` is its length. The TRANSPOSED_HALF_LOCAL_PV
 *  branch (Python 3361-3391) uses pv32_v_load_paired (ported above) for the
 *  half-local K orientation; both branches are byte-faithful to Python.
 * ============================================================ */
rocke_value_t* rocke_gfx950_attn2d_apply_transposed_pv_regs(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                            rocke_value_t* acc32,
                                                            int n,
                                                            rocke_value_t* const* p_regs,
                                                            int p_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    const int RPL = 16; /* PT32 stores 16 regs per (p_tile) */
    rocke_value_t* v_buf;
    rocke_value_t* use_hi;
    rocke_value_t* v_dim32;
    int k;

    (void)p_count;
    /* v_buf / use_hi are hoisted by the caller (Python 3231-3232) and emitted
     * once before the per-N loop; reuse the cached SSA values. */
    v_buf = (ctx->pv_v_buf_v != NULL) ? ctx->pv_v_buf_v : rocke_b_const_i32(b, 0);
    use_hi = (ctx->pv_use_hi_v != NULL)
                 ? ctx->pv_use_hi_v
                 : rocke_b_cmp_eq(b, ctx->lane_half32_v, rocke_b_const_i32(b, 1));

    v_dim32 = rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n * 32)), ctx->lane_col32_v);
    for(k = 0; k < ctx->T / 16; ++k)
    {
        rocke_value_t* a_v_elems[8];
        rocke_value_t* b_p_elems[8];
        rocke_value_t* A_v_t;
        rocke_value_t* B_p_t;
        int kk;

        if(ctx->TRANSPOSED_HALF_LOCAL_PV)
        {
            /* Half-local PV (Python 3361-3391). Each 32-lane half consumes only
             * the P rows already local to that half via the paired transpose
             * V-read so V and P share the same permuted K order. */
            A_v_t = rocke_pv32_v_load_paired(
                b, ctx->V_lds, v_buf, n, k, ctx->lane_half32_v, ctx->lane_col32_v, dtype);
            for(kk = 0; kk < 8; ++kk)
            {
                int local_in_group = kk % 4;
                int band = kk / 4;
                int p_tile = (k * 16 + band * 8 + local_in_group) / 32;
                int row_static = (k * 16 + band * 8 + local_in_group) % 32;
                int preg = (row_static / 8) * 4 + (row_static % 4);
                b_p_elems[kk] = rocke_b_cast_f32_to(b, p_regs[p_tile * RPL + preg], dtype);
            }
            B_p_t = rocke_b_vec_pack(b, b_p_elems, 8, dtype);
            acc32 = rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, A_v_t, B_p_t, acc32);
            continue;
        }

        /* Issue the 8 V scalar loads first (Python 3271-3281). */
        for(kk = 0; kk < 8; ++kk)
        {
            int k_static = k * 16 + kk;
            /* const(k_static) created before the mul (Python arg order). */
            rocke_value_t* v_row_base = rocke_b_const_i32(b, (int64_t)k_static);
            rocke_value_t* v_row = rocke_b_add(
                b, v_row_base, rocke_b_mul(b, ctx->lane_half32_v, rocke_b_const_i32(b, 8)));
            rocke_value_t* idx[3];
            rocke_value_t* v1;
            idx[0] = v_buf;
            idx[1] = v_row;
            idx[2] = v_dim32;
            v1 = rocke_b_smem_load_vN(b, ctx->V_lds, idx, 3, dtype, 1);
            a_v_elems[kk] = rocke_b_vec_extract(b, v1, 0);
        }
        /* Assemble the P^T operand (Python 3286-3310). */
        for(kk = 0; kk < 8; ++kk)
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
            rocke_value_t* p_val;
            if(owner_half0 == 1)
                p0 = rocke_b_warp_shuffle_xor(b, p0, 32);
            if(owner_half1 == 0)
                p1 = rocke_b_warp_shuffle_xor(b, p1, 32);
            p_val = rocke_b_select(b, use_hi, p1, p0);
            b_p_elems[kk] = rocke_b_cast_f32_to(b, p_val, dtype);
        }
        A_v_t = rocke_b_vec_pack(b, a_v_elems, 8, dtype);
        B_p_t = rocke_b_vec_pack(b, b_p_elems, 8, dtype);
        acc32 = rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, A_v_t, B_p_t, acc32);
    }
    return acc32;
}

/* ============================================================ *
 *  rocke_gfx950_attn2d_emit_pv_bucket   (Python lines 3174-3572)
 *
 *  The PV back half of _emit_kv_body. Consumes the softmax-derived state from
 *  ``in`` (alpha_regs, new_l_vals, m_new, PT32 register groups, register-P
 *  groups, GROUPED_KV2 re-issue inputs), computes acc *= alpha; acc += P @ V,
 *  and emits the scf_yield carry.
 * ============================================================ */
void rocke_gfx950_attn2d_emit_pv_bucket(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                        const rocke_gfx950_attn2d_pv_inputs_t* in)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    const rocke_type_t* F32 = rocke_f32();
    const rocke_type_t* FP8E4M3 = rocke_fp8e4m3();
    const int RPL = 16; /* PT32 regs-per-tile */
    int kv_calls_per_tile = (ctx->T * ctx->HD) / (ctx->THREADS * 8);
    rocke_value_t* new_acc[ROCKE_GFX950_ATTN2D_MAX_ACCS];
    rocke_value_t* yields[ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS];
    rocke_value_t* pv_fp8_scale = NULL;
    int n, r, atom, k, i, yc;

    for(i = 0; i < ROCKE_GFX950_ATTN2D_MAX_ACCS; ++i)
        new_acc[i] = NULL;

    /* ---- pre-PV wait/sync (Python 3181-3199) ---- */
    if(ctx->GROUPED_KV2)
    {
        rocke_b_s_waitcnt(b, 0, 0, -1);
        rocke_b_sync(b);
    }
    else if(ctx->KV_FP8)
    {
        rocke_b_s_waitcnt(b, 0, 0, -1);
        rocke_b_sync(b);
    }
    else if(ctx->K_SINGLE_BUFFER)
    {
        /* single K slot. Only V[i] is in flight here (next-K was deferred).
         * Fully drain it for PV, then s_barrier. After the barrier all QK[i]
         * K_lds reads are retired, so re-issue the next-K prefetch into the
         * single slot (overlaps PV[i]; the next iter-start full drain makes it
         * visible before QK[i+1]). No WAR race. */
        rocke_b_s_waitcnt(b, 0, 0, -1);
        rocke_b_sync(b);
        rocke_gfx950_attn2d_issue_k(ctx, in->safe_next_tile, ctx->nxt_buf_v);
    }
    else
    {
        rocke_b_s_waitcnt(b, kv_calls_per_tile, kv_calls_per_tile, -1);
        rocke_b_sync(b);
    }

    /* The fp8 PV scale (Python line 1149: v_scale_p / 240.0) is hoisted in the
     * prologue (ctx->pv_fp8_scale_v) so its SSA id matches Python; reuse it. */
    if(ctx->FP8_MFMA_PV)
        pv_fp8_scale = ctx->pv_fp8_scale_v;

    /* ---- acc *= alpha, acc += P @ V ---- (Python 3219-3412) ---- */
    if(ctx->USE_MFMA_32X32)
    {
        if(ctx->TRANSPOSED_QK_32X32)
        {
            /* Transposed PV: O^T = V^T @ P^T via register P^T (Python 3221-3341).
             * v_buf and use_hi are emitted ONCE here (Python 3231-3232), before
             * the per-N acc-scale + apply loop, and reused by every apply call. */
            ctx->pv_v_buf_v = rocke_b_const_i32(b, 0);
            ctx->pv_use_hi_v = rocke_b_cmp_eq(b, ctx->lane_half32_v, rocke_b_const_i32(b, 1));
            for(n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                rocke_value_t* scaled[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
                rocke_value_t* old_acc = rocke_gfx950_attn2d_acc_get(ctx, n, 0);
                rocke_value_t* alpha_t = in->alpha_regs[0];
                rocke_value_t* acc32;
                rocke_value_t* const* PT32_n = in->pt32_g0 + (size_t)n * RPL;
                for(r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    rocke_value_t* e = rocke_b_vec_extract(b, old_acc, r);
                    rocke_value_t* a = ctx->TRANSPOSED_SCALAR_STATE ? alpha_t : in->alpha_regs[r];
                    scaled[r] = rocke_b_fmul(b, e, a);
                }
                acc32 = rocke_b_vec_pack(b, scaled, ctx->REGS_PER_LANE, F32);
                /* PT32 is addressed absolutely as [p_tile][reg] inside the
                 * helper (p_tile = k//32 spans all QK_N_TILES); Python passes
                 * the FULL PT32_n 2D array (NOT an n-slice). Pass the whole
                 * pt32_g0 base; the n arg only drives v_dim32 / _acc_get. */
                (void)PT32_n;
                acc32 = rocke_gfx950_attn2d_apply_transposed_pv_regs(
                    ctx, acc32, n, in->pt32_g0, in->pt32_count);
                new_acc[n] = acc32;
            }
            if(ctx->GROUPED_KV2)
            {
                rocke_b_s_waitcnt(b, -1, 0, -1);
                rocke_b_sync(b);
                rocke_gfx950_attn2d_issue_v(ctx, in->safe_tile1, in->nxt_buf);
                rocke_b_s_waitcnt(b, 0, 0, -1);
                rocke_b_sync(b);
                for(n = 0; n < ctx->ACC_N_TILES; ++n)
                {
                    new_acc[n] = rocke_gfx950_attn2d_apply_transposed_pv_regs(
                        ctx, new_acc[n], n, in->pt32_g1, in->pt32_count);
                }
            }
        }
        else
        {
            /* Transitional 32x32 PV from the P_lds bridge (Python 3342-3412). */
            rocke_value_t* v_buf = rocke_b_const_i32(b, 0);
            for(n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                rocke_value_t* scaled[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
                rocke_value_t* old_acc = rocke_gfx950_attn2d_acc_get(ctx, n, 0);
                rocke_value_t* acc32;
                for(r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    rocke_value_t* e = rocke_b_vec_extract(b, old_acc, r);
                    scaled[r] = rocke_b_fmul(b, e, in->alpha_regs[r]);
                }
                acc32 = rocke_b_vec_pack(b, scaled, ctx->REGS_PER_LANE, F32);
                for(k = 0; k < ctx->T / 16; ++k)
                {
                    /* const(k*16) created before the mul (Python arg order). */
                    rocke_value_t* p_off32_base = rocke_b_const_i32(b, (int64_t)(k * 16));
                    rocke_value_t* p_off32
                        = rocke_b_add(b,
                                      p_off32_base,
                                      rocke_b_mul(b, ctx->lane_half32_v, rocke_b_const_i32(b, 8)));
                    rocke_value_t* p_row32 = rocke_b_add(b, ctx->wave_row_base, ctx->lane_col32_v);
                    rocke_value_t* pidx[2];
                    rocke_value_t* A_p32;
                    rocke_value_t* col_group16;
                    rocke_value_t* tr_col32;
                    rocke_value_t* tr_row_base32;
                    rocke_value_t* B32_r0;
                    rocke_value_t* B32_r1;
                    rocke_value_t* B_v32;
                    rocke_value_t* ridx0[3];
                    rocke_value_t* ridx1[3];

                    pidx[0] = p_row32;
                    pidx[1] = p_off32;
                    A_p32 = rocke_b_smem_load_vN(b, ctx->P_lds, pidx, 2, dtype, 8);

                    /* Sequence sub-expressions so C arg-eval order matches
                     * Python's left-to-right value creation. */
                    {
                        rocke_value_t* cg_div
                            = rocke_b_div(b, ctx->lane_col32_v, rocke_b_const_i32(b, 16));
                        col_group16 = rocke_b_mul(b, cg_div, rocke_b_const_i32(b, 16));
                    }
                    {
                        rocke_value_t* tc_mod
                            = rocke_b_mod(b, ctx->lane_col32_v, rocke_b_const_i32(b, 4));
                        rocke_value_t* tc_mul = rocke_b_mul(b, tc_mod, rocke_b_const_i32(b, 4));
                        tr_col32 = rocke_b_add(b, col_group16, tc_mul);
                    }
                    {
                        rocke_value_t* trb_base = rocke_b_const_i32(b, (int64_t)(k * 16));
                        rocke_value_t* trb_inner = rocke_b_add(
                            b,
                            trb_base,
                            rocke_b_mul(b, ctx->lane_half32_v, rocke_b_const_i32(b, 8)));
                        rocke_value_t* trb_div
                            = rocke_b_div(b, ctx->lane_col32_v, rocke_b_const_i32(b, 4));
                        rocke_value_t* trb_mod = rocke_b_mod(b, trb_div, rocke_b_const_i32(b, 4));
                        tr_row_base32 = rocke_b_add(b, trb_inner, trb_mod);
                    }
                    ridx0[0] = v_buf;
                    ridx0[1] = tr_row_base32;
                    {
                        rocke_value_t* r0_base = rocke_b_const_i32(b, (int64_t)(n * 32));
                        ridx0[2] = rocke_b_add(b, r0_base, tr_col32);
                    }
                    B32_r0 = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, ridx0, 3, dtype);
                    ridx1[0] = v_buf;
                    ridx1[1] = rocke_b_add(b, tr_row_base32, rocke_b_const_i32(b, 4));
                    {
                        rocke_value_t* r1_base = rocke_b_const_i32(b, (int64_t)(n * 32));
                        ridx1[2] = rocke_b_add(b, r1_base, tr_col32);
                    }
                    B32_r1 = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, ridx1, 3, dtype);
                    B_v32 = rocke_b_vec_concat(b, B32_r0, B32_r1);
                    acc32 = rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, A_p32, B_v32, acc32);
                }
                new_acc[n] = acc32;
            }
        }
    }

    /* ---- narrow 16x16 per-atom PV (Python 3413-3562) ---- */
    {
        int n_lim = ctx->USE_MFMA_32X32 ? 0 : ctx->PV_N_TILES;
        for(n = 0; n < n_lim; ++n)
        {
            rocke_value_t* acc_per_atom[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE / 4];
            rocke_value_t* n_col_base;
            rocke_value_t* v_buf;

            for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
            {
                rocke_value_t* scaled_comps[4];
                int in_atom;
                for(in_atom = 0; in_atom < 4; ++in_atom)
                {
                    int reg = atom * 4 + in_atom;
                    rocke_value_t* e = rocke_b_vec_extract(
                        b, rocke_gfx950_attn2d_acc_get(ctx, n, atom), in_atom);
                    scaled_comps[in_atom] = rocke_b_fmul(b, e, in->alpha_regs[reg]);
                }
                acc_per_atom[atom] = rocke_b_vec_pack(b, scaled_comps, 4, F32);
            }

            {
                rocke_value_t* ncb_n = rocke_b_const_i32(b, (int64_t)n);
                rocke_value_t* ncb_16 = rocke_b_const_i32(b, 16);
                n_col_base = rocke_b_add(b, rocke_b_mul(b, ncb_n, ncb_16), ctx->pv_tr_reader->col);
            }
            v_buf = rocke_b_const_i32(b, 0);

            for(k = 0; k < ctx->PV_K_ITERS; ++k)
            {
                if(ctx->PV_K_STEP == 32)
                {
                    /* const(k*32) created before the mul (Python arg order). */
                    rocke_value_t* p_off_base = rocke_b_const_i32(b, (int64_t)(k * 32));
                    rocke_value_t* p_off = rocke_b_add(
                        b, p_off_base, rocke_b_mul(b, ctx->lane_rg_v, rocke_b_const_i32(b, 8)));
                    rocke_value_t* row_r0
                        = rocke_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 32, 0);
                    rocke_value_t* row_r1
                        = rocke_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 32, 1);
                    if(ctx->FP8_MFMA_PV)
                    {
                        /* native-fp8 PV stripe path (Python 3439-3507) */
                        rocke_value_t* stripe_const = rocke_b_const_i32(b, (int64_t)n);
                        /* Python: b.add(b.mul(lane_rg, 8), b.div(lane_col, 2))
                         * creates the mul BEFORE the div (left-to-right). Bind in
                         * order so C's right-to-left arg eval matches the SSA ids. */
                        rocke_value_t* krpl_mul
                            = rocke_b_mul(b, ctx->lane_rg_v, rocke_b_const_i32(b, 8));
                        rocke_value_t* krpl_div
                            = rocke_b_div(b, ctx->lane_col_v, rocke_b_const_i32(b, 2));
                        rocke_value_t* k_row_per_lane = rocke_b_add(b, krpl_mul, krpl_div);
                        rocke_value_t* k_row_for_iter = rocke_b_add(
                            b, rocke_b_const_i32(b, (int64_t)(k * 32)), k_row_per_lane);
                        rocke_value_t* lo_idx[4];
                        rocke_value_t* hi_idx[4];
                        rocke_value_t* B_v8_lo;
                        rocke_value_t* B_v8_hi;
                        rocke_value_t* lo_mask;
                        rocke_value_t* B_v8;
                        lo_idx[0] = v_buf;
                        lo_idx[1] = stripe_const;
                        lo_idx[2] = k_row_for_iter;
                        lo_idx[3] = rocke_b_const_i32(b, 0);
                        B_v8_lo = rocke_b_ds_read_tr_b8(b, ctx->V_lds, lo_idx, 4, FP8E4M3);
                        hi_idx[0] = v_buf;
                        hi_idx[1] = stripe_const;
                        hi_idx[2] = k_row_for_iter;
                        hi_idx[3] = rocke_b_const_i32(b, 8);
                        B_v8_hi = rocke_b_ds_read_tr_b8(b, ctx->V_lds, hi_idx, 4, FP8E4M3);
                        lo_mask = rocke_b_cmp_lt(b, ctx->lane_col_v, rocke_b_const_i32(b, 8));
                        B_v8 = rocke_b_vector_select(
                            b, rocke_b_vector_splat(b, lo_mask, 8), B_v8_lo, B_v8_hi);
                        for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                        {
                            rocke_value_t* p_row = rocke_b_add(
                                b,
                                ctx->wave_row_base,
                                rocke_b_add(b,
                                            rocke_b_const_i32(b, (int64_t)(atom * 16)),
                                            ctx->lane_col_v));
                            rocke_value_t* pidx[2];
                            rocke_value_t* A_p8;
                            rocke_value_t* raw;
                            rocke_value_t* comps[4];
                            int ii;
                            pidx[0] = p_row;
                            pidx[1] = p_off;
                            A_p8 = rocke_b_smem_load_vN(b, ctx->P_lds, pidx, 2, FP8E4M3, 8);
                            raw = rocke_b_mfma_f32_16x16x32_fp8(
                                b, A_p8, B_v8, rocke_b_zero_vec_f32(b, 4));
                            for(ii = 0; ii < 4; ++ii)
                            {
                                rocke_value_t* old = rocke_b_vec_extract(b, acc_per_atom[atom], ii);
                                rocke_value_t* add = rocke_b_fmul(
                                    b, rocke_b_vec_extract(b, raw, ii), pv_fp8_scale);
                                comps[ii] = rocke_b_fadd(b, old, add);
                            }
                            acc_per_atom[atom] = rocke_b_vec_pack(b, comps, 4, F32);
                        }
                    }
                    else
                    {
                        rocke_value_t* r0idx[3];
                        rocke_value_t* r1idx[3];
                        rocke_value_t* B_r0;
                        rocke_value_t* B_r1;
                        rocke_value_t* B_v;
                        r0idx[0] = v_buf;
                        r0idx[1] = row_r0;
                        r0idx[2] = n_col_base;
                        B_r0 = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, r0idx, 3, dtype);
                        r1idx[0] = v_buf;
                        r1idx[1] = row_r1;
                        r1idx[2] = n_col_base;
                        B_r1 = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, r1idx, 3, dtype);
                        B_v = rocke_b_vec_concat(b, B_r0, B_r1);
                        for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                        {
                            rocke_value_t* A_p;
                            if(ctx->REGISTER_PV)
                            {
                                rocke_value_t* g0[4];
                                rocke_value_t* g1[4];
                                int rr;
                                for(rr = 0; rr < 4; ++rr)
                                {
                                    int base = (atom * 4 + rr) * in->p_regs_f32_stride;
                                    g0[rr] = in->p_regs_f32[base + 2 * k];
                                    g1[rr] = in->p_regs_f32[base + 2 * k + 1];
                                }
                                A_p = rocke_gfx950_attn2d_pack_p_a32(ctx, g0, g1, 4);
                            }
                            else
                            {
                                rocke_value_t* p_row = rocke_b_add(
                                    b,
                                    ctx->wave_row_base,
                                    rocke_b_add(b,
                                                rocke_b_const_i32(b, (int64_t)(atom * 16)),
                                                ctx->lane_col_v));
                                rocke_value_t* pidx[2];
                                pidx[0] = p_row;
                                pidx[1] = p_off;
                                A_p = rocke_b_smem_load_vN(b, ctx->P_lds, pidx, 2, dtype, 8);
                            }
                            acc_per_atom[atom] = rocke__attn2d_mfma_16x16x32(
                                b, dtype, A_p, B_v, acc_per_atom[atom]);
                        }
                    }
                }
                else
                {
                    rocke_value_t* p_off;
                    rocke_value_t* row_lane;
                    rocke_value_t* ridx[3];
                    rocke_value_t* B_v;
                    /* PV_FP8_MFMA with K=16 is unsupported (Python 3541-3542). */
                    assert(!ctx->FP8_MFMA_PV);
                    /* Python: b.add(b.const_i32(k*16), b.mul(lane_rg, const(4)))
                     * evaluates the const(k*16) arg BEFORE the mul (left-to-right).
                     * C arg-eval order is unspecified (typically right-to-left), so
                     * bind both operands to temps in Python's order. */
                    {
                        rocke_value_t* p_off_c = rocke_b_const_i32(b, (int64_t)(k * 16));
                        rocke_value_t* p_off_rg
                            = rocke_b_mul(b, ctx->lane_rg_v, rocke_b_const_i32(b, 4));
                        p_off = rocke_b_add(b, p_off_c, p_off_rg);
                    }
                    row_lane
                        = rocke_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 16, 0);
                    ridx[0] = v_buf;
                    ridx[1] = row_lane;
                    ridx[2] = n_col_base;
                    B_v = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, ridx, 3, dtype);
                    for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                    {
                        rocke_value_t* A_p;
                        if(ctx->REGISTER_PV)
                        {
                            rocke_value_t* g[4];
                            int rr;
                            for(rr = 0; rr < 4; ++rr)
                            {
                                int base = (atom * 4 + rr) * in->p_regs_f32_stride;
                                g[rr] = in->p_regs_f32[base + k];
                            }
                            A_p = rocke_gfx950_attn2d_pack_p_a16(ctx, g, 4);
                        }
                        else
                        {
                            rocke_value_t* p_row = rocke_b_add(
                                b,
                                ctx->wave_row_base,
                                rocke_b_add(b,
                                            rocke_b_const_i32(b, (int64_t)(atom * 16)),
                                            ctx->lane_col_v));
                            rocke_value_t* pidx[2];
                            pidx[0] = p_row;
                            pidx[1] = p_off;
                            A_p = rocke_b_smem_load_vN(b, ctx->P_lds, pidx, 2, dtype, 4);
                        }
                        acc_per_atom[atom]
                            = rocke__attn2d_mfma_16x16x16(b, dtype, A_p, B_v, acc_per_atom[atom]);
                    }
                }
            }
            for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                new_acc[n * ctx->M_ATOMS_PER_WARP + atom] = acc_per_atom[atom];
        }
    }

    /* ---- assemble the scf_yield carry (Python 3564-3572) ---- */
    yc = 0;
    for(r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
    {
        yields[yc++] = in->m_new[r];
        yields[yc++] = in->new_l_vals[r];
    }
    for(n = 0; n < ctx->ACC_N_TILES; ++n)
        for(atom = 0; atom < ctx->ACC_M_ATOMS; ++atom)
            yields[yc++] = new_acc[n * ctx->ACC_M_ATOMS + atom];
    yields[yc++] = ctx->GROUPED_KV2 ? in->cur_buf : ctx->nxt_buf_v;

    /* Record the rewritten carry for callers that thread phases (the driver's
     * single-phase path reads this back via the loop results). */
    for(i = 0; i < yc; ++i)
        ctx->out_carry[i] = yields[i];
    ctx->out_carry_count = yc;

    rocke_b_scf_yield(b, yields, yc);
}

/* ============================================================ *
 *  rocke_gfx950_attn2d_drive_kv_loop   (Python lines 3581-3583)
 *
 *  scf_for_iter over [tile_start, tile_end) step kv_step with the named
 *  iter_args carry. Enter the body, unpack the carry into ctx->m_cur/l_cur/
 *  acc_cur + ctx->cur_buf and set ctx->kv_tile_iv, run the (peer) full body
 *  emitter, then leave. Returns the loop handle; the epilogue reads the
 *  rewritten carry from kvloop.op->results.
 * ============================================================ */
rocke_for_t rocke_gfx950_attn2d_drive_kv_loop(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_iter_arg_t iter_args[ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS];
    rocke_for_t kvloop;
    int i;
    int ml = ctx->ml_count;
    int num_accs = ctx->ACC_N_TILES * ctx->ACC_M_ATOMS;

    for(i = 0; i < ctx->iter_args_count; ++i)
    {
        iter_args[i].name = ctx->iter_args_names[i];
        iter_args[i].init = ctx->iter_args[i];
    }

    kvloop = rocke_b_scf_for_iter(b,
                                  ctx->tile_start,
                                  ctx->tile_end,
                                  ctx->kv_step,
                                  iter_args,
                                  ctx->iter_args_count,
                                  "kv_tile",
                                  false,
                                  true);

    rocke_b_region_enter(b, kvloop.body);
    ctx->kv_tile_iv = kvloop.iv;
    /* Unpack the carry the same way the Python body does (lines 2495-2508). */
    for(i = 0; i < ctx->SOFTMAX_STATE_SLOTS; ++i)
    {
        ctx->m_cur[i] = kvloop.iter_vars[2 * i];
        ctx->l_cur[i] = kvloop.iter_vars[2 * i + 1];
    }
    for(i = 0; i < num_accs; ++i)
        ctx->acc_cur[i] = kvloop.iter_vars[ml + i];
    ctx->cur_buf = kvloop.iter_vars[ml + num_accs];
    ctx->skip_mask = false;

    rocke_gfx950_attn2d_emit_kv_body(ctx);

    rocke_b_region_leave(b);
    return kvloop;
}

/* ============================================================ *
 *  rocke_gfx950_attn2d_emit_epilogue   (Python lines 3585-3817)
 * ============================================================ */
rocke_kernel_def_t* rocke_gfx950_attn2d_emit_epilogue(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    const char* coord_names[3] = {"token", "head", "dim"};
    int ml_count_final;
    int n, r, atom, i;

    /* This entry expects rocke_gfx950_attn2d_drive_kv_loop to have stashed the
     * loop results into ctx->iter_args (reused as the final-carry slot) and the
     * per-slot finals into ctx->l_final / acc_final. The driver/glue populate
     * ctx->l_final / ctx->acc_final from kvloop.op->results before calling. */

    /* Drain async + close outstanding copies (Python 3592-3593). */
    rocke_b_s_waitcnt(b, 0, 0, -1);
    rocke_b_sync(b);

    ml_count_final = 2 * ctx->SOFTMAX_STATE_SLOTS;

    /* Read the rewritten loop carry into l_final / acc_final, mirroring
     * Python lines 3595-3603:
     *   final     = kvloop.results
     *   l_final[r]= final[2*r + 1]
     *   acc_final[n*ACC_M_ATOMS + atom]
     *             = final[ml_count_final + n*ACC_M_ATOMS + atom]
     * The driver stashed kvloop.op->results into ctx->out_carry. */
    for(r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
        ctx->l_final[r] = ctx->out_carry[2 * r + 1];
    for(n = 0; n < ctx->ACC_N_TILES; ++n)
        for(atom = 0; atom < ctx->ACC_M_ATOMS; ++atom)
            ctx->acc_final[n * ctx->ACC_M_ATOMS + atom]
                = ctx->out_carry[ml_count_final + n * ctx->ACC_M_ATOMS + atom];

    /* Per-row reciprocal of L and the nonzero predicate (Python 3609-3610).
     * Python emits ALL rcps first (list comp), THEN all fcmps -- two loops. */
    for(r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
        ctx->rcp_l[r] = rocke_b_rcp(b, ctx->l_final[r]);
    for(r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
        ctx->l_nonzero[r] = rocke_b_fcmp(b, "ogt", ctx->l_final[r], ctx->zero_f);

    if(ctx->USE_MFMA_32X32)
    {
        if(ctx->TRANSPOSED_QK_32X32)
        {
            /* Per-lane direct scalar global stores (Python 3612-3656). */
            rocke_value_t* q_row_t = rocke_b_add(b, ctx->wave_row_base, ctx->lane_col32_v);
            rocke_value_t* op_pos_t = rocke_b_add(
                b, ctx->qb_start_pos, rocke_b_div(b, q_row_t, rocke_b_const_i32(b, ctx->NQK)));
            /* Sequence sub-expressions so C arg-eval order matches Python's
             * left-to-right value creation (mul before mod; op_pos cmp before
             * op_qh cmp). */
            rocke_value_t* op_qh_mul
                = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, ctx->NQK));
            rocke_value_t* op_qh_mod = rocke_b_mod(b, q_row_t, rocke_b_const_i32(b, ctx->NQK));
            rocke_value_t* op_qh_t = rocke_b_add(b, op_qh_mul, op_qh_mod);
            rocke_value_t* op_mask_pos = rocke_b_cmp_lt(b, op_pos_t, ctx->cur_batch_q_len);
            rocke_value_t* op_mask_qh
                = rocke_b_cmp_lt(b, op_qh_t, rocke_b_const_i32(b, ctx->NUM_QH));
            rocke_value_t* op_mask_t = rocke_b_land(b, op_mask_pos, op_mask_qh);
            rocke_value_t* out_base_t = NULL;
            rocke_value_t* inv_l_t;
            rocke_value_t* l_nonzero_t;
            const char* in_names[3];
            rocke_value_t* in_values[3];

            in_names[0] = coord_names[0];
            in_names[1] = coord_names[1];
            in_names[2] = coord_names[2];
            in_values[0] = rocke_b_add(b, ctx->cu_q_start, op_pos_t);
            in_values[1] = op_qh_t;
            in_values[2] = rocke_b_const_i32(b, 0);
            if(!rocke_transforms_descriptor_offset(
                   b, ctx->q_desc, in_names, in_values, 3, &out_base_t, NULL))
                return NULL;

            inv_l_t = rocke_b_rcp(b, ctx->l_final[0]);
            l_nonzero_t = rocke_b_fcmp(b, "ogt", ctx->l_final[0], ctx->zero_f);
            for(n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                rocke_value_t* acc32 = rocke_gfx950_attn2d_acc_final_get(ctx, n, 0);
                for(r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    /* const(n*32) created before _mfma_32x32_c_row (Py arg order). */
                    rocke_value_t* out_col_base = rocke_b_const_i32(b, (int64_t)(n * 32));
                    rocke_value_t* out_col_t = rocke_b_add(
                        b,
                        out_col_base,
                        rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row(b, ctx->lane, r));
                    rocke_value_t* v = rocke_b_vec_extract(b, acc32, r);
                    rocke_value_t* normalized = rocke_b_fmul(b, v, inv_l_t);
                    rocke_value_t* final_h = rocke_b_cast_f32_to(
                        b, rocke_b_select(b, l_nonzero_t, normalized, ctx->zero_f), dtype);
                    rocke_if_t iff = rocke_b_scf_if(b, op_mask_t);
                    rocke_b_region_enter(b, iff.then_region);
                    rocke_b_global_store(
                        b, ctx->output, rocke_b_add(b, out_base_t, out_col_t), final_h, 2);
                    rocke_b_region_leave(b);
                }
            }
            return ctx->b->kernel;
        }

        /* Coalesced Acc_lds-staged 32x32 epilogue (Python 3658-3715). */
        {
            const int OUT_VEC32 = 8;
            int OUT_PER_THREAD_HALVES32 = (ctx->BLOCK_M * 32) / ctx->THREADS;
            int OUT_CHUNKS_PER_THREAD32;
            int OUT_THREADS_PER_ROW32;
            rocke_value_t* OUT_ROW_BASE32;
            rocke_value_t* OUT_col_base32;
            rocke_value_t* op_pos32_base;
            rocke_value_t* op_qh32_base;
            rocke_value_t* op_mask32_base;
            rocke_value_t* out_base32_base = NULL;
            const char* in_names[3];
            rocke_value_t* in_values[3];
            int chunk;

            assert(OUT_PER_THREAD_HALVES32 % OUT_VEC32 == 0);
            OUT_CHUNKS_PER_THREAD32 = OUT_PER_THREAD_HALVES32 / OUT_VEC32;
            OUT_THREADS_PER_ROW32 = 32 / (OUT_CHUNKS_PER_THREAD32 * OUT_VEC32);
            /* Sequence sub-expressions to match Python value-creation order. */
            OUT_ROW_BASE32 = rocke_b_div(b, ctx->tid, rocke_b_const_i32(b, OUT_THREADS_PER_ROW32));
            {
                rocke_value_t* ocb_mod
                    = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, OUT_THREADS_PER_ROW32));
                OUT_col_base32 = rocke_b_mul(
                    b, ocb_mod, rocke_b_const_i32(b, OUT_CHUNKS_PER_THREAD32 * OUT_VEC32));
            }
            op_pos32_base
                = rocke_b_add(b,
                              ctx->qb_start_pos,
                              rocke_b_div(b, OUT_ROW_BASE32, rocke_b_const_i32(b, ctx->NQK)));
            {
                rocke_value_t* qh_mul
                    = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, ctx->NQK));
                rocke_value_t* qh_mod
                    = rocke_b_mod(b, OUT_ROW_BASE32, rocke_b_const_i32(b, ctx->NQK));
                op_qh32_base = rocke_b_add(b, qh_mul, qh_mod);
            }
            {
                rocke_value_t* mask_pos = rocke_b_cmp_lt(b, op_pos32_base, ctx->cur_batch_q_len);
                rocke_value_t* mask_qh
                    = rocke_b_cmp_lt(b, op_qh32_base, rocke_b_const_i32(b, ctx->NUM_QH));
                op_mask32_base = rocke_b_land(b, mask_pos, mask_qh);
            }
            in_names[0] = coord_names[0];
            in_names[1] = coord_names[1];
            in_names[2] = coord_names[2];
            in_values[0] = rocke_b_add(b, ctx->cu_q_start, op_pos32_base);
            in_values[1] = op_qh32_base;
            in_values[2] = rocke_b_const_i32(b, 0);
            if(!rocke_transforms_descriptor_offset(
                   b, ctx->q_desc, in_names, in_values, 3, &out_base32_base, NULL))
                return NULL;

            for(n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                rocke_value_t* acc32 = rocke_gfx950_attn2d_acc_final_get(ctx, n, 0);
                for(r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    rocke_value_t* row = rocke_b_add(
                        b,
                        ctx->wave_row_base,
                        rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row(b, ctx->lane, r));
                    rocke_value_t* col_in_stripe = ctx->lane_col32_v;
                    rocke_value_t* v = rocke_b_vec_extract(b, acc32, r);
                    rocke_value_t* normalized = rocke_b_fmul(b, v, ctx->rcp_l[r]);
                    rocke_value_t* final_h = rocke_b_cast_f32_to(
                        b, rocke_b_select(b, ctx->l_nonzero[r], normalized, ctx->zero_f), dtype);
                    rocke_value_t* sidx[2];
                    sidx[0] = row;
                    sidx[1] = col_in_stripe;
                    rocke_b_smem_store_vN(b, ctx->Acc_lds, sidx, 2, final_h, 1);
                }
                rocke_b_sync(b);
                for(chunk = 0; chunk < OUT_CHUNKS_PER_THREAD32; ++chunk)
                {
                    rocke_value_t* col_in_stripe = rocke_b_add(
                        b, OUT_col_base32, rocke_b_const_i32(b, (int64_t)(chunk * OUT_VEC32)));
                    rocke_value_t* lidx[2];
                    rocke_value_t* v8h;
                    rocke_value_t* out_col;
                    rocke_if_t iff;
                    lidx[0] = OUT_ROW_BASE32;
                    lidx[1] = col_in_stripe;
                    v8h = rocke_b_smem_load_vN(b, ctx->Acc_lds, lidx, 2, dtype, OUT_VEC32);
                    out_col
                        = rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n * 32)), col_in_stripe);
                    iff = rocke_b_scf_if(b, op_mask32_base);
                    rocke_b_region_enter(b, iff.then_region);
                    rocke_b_global_store_vN(b,
                                            ctx->output,
                                            rocke_b_add(b, out_base32_base, out_col),
                                            v8h,
                                            OUT_VEC32,
                                            16);
                    rocke_b_region_leave(b);
                }
                if(n + 1 < ctx->ACC_N_TILES)
                    rocke_b_sync(b);
            }
            return ctx->b->kernel;
        }
    }

    /* ---------------- striped epilogue (Python 3717-3817) ---------------- */
    {
        const int MFMA_N = MFMA_N_CONST;
        int N_TILES_PER_STRIPE = ctx->OUT_STRIPE_COLS / MFMA_N;
        const int OUT_VEC = 8;
        int OUT_PER_THREAD_HALVES = (ctx->BLOCK_M * ctx->OUT_STRIPE_COLS) / ctx->THREADS;
        int OUT_CHUNKS_PER_THREAD;
        int OUT_THREADS_PER_ROW;
        rocke_value_t* OUT_ROW_BASE;
        rocke_value_t* OUT_col_base_in_stripe;
        rocke_value_t* op_pos;
        rocke_value_t* op_qh;
        rocke_value_t* op_mask;
        rocke_value_t* out_base = NULL;
        const char* in_names[3];
        rocke_value_t* in_values[3];
        int stripe;

        assert(ctx->PV_N_TILES % N_TILES_PER_STRIPE == 0);
        assert(OUT_PER_THREAD_HALVES % OUT_VEC == 0 && OUT_PER_THREAD_HALVES > 0);
        OUT_CHUNKS_PER_THREAD = OUT_PER_THREAD_HALVES / OUT_VEC;
        OUT_THREADS_PER_ROW = ctx->OUT_STRIPE_COLS / (OUT_CHUNKS_PER_THREAD * OUT_VEC);
        assert(ctx->THREADS / OUT_THREADS_PER_ROW == ctx->BLOCK_M);

        /* Match Python value-creation order (left-to-right arg eval): C's
         * function-arg eval order is unspecified, so sequence sub-expressions
         * into temporaries to keep the SSA stream byte-identical. */
        OUT_ROW_BASE = rocke_b_div(b, ctx->tid, rocke_b_const_i32(b, OUT_THREADS_PER_ROW));
        {
            rocke_value_t* ocb_mod
                = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, OUT_THREADS_PER_ROW));
            OUT_col_base_in_stripe
                = rocke_b_mul(b, ocb_mod, rocke_b_const_i32(b, OUT_CHUNKS_PER_THREAD * OUT_VEC));
        }

        op_pos = rocke_b_add(
            b, ctx->qb_start_pos, rocke_b_div(b, OUT_ROW_BASE, rocke_b_const_i32(b, ctx->NQK)));
        {
            rocke_value_t* qh_mul
                = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, ctx->NQK));
            rocke_value_t* qh_mod = rocke_b_mod(b, OUT_ROW_BASE, rocke_b_const_i32(b, ctx->NQK));
            op_qh = rocke_b_add(b, qh_mul, qh_mod);
        }
        {
            rocke_value_t* mask_pos = rocke_b_cmp_lt(b, op_pos, ctx->cur_batch_q_len);
            rocke_value_t* mask_qh = rocke_b_cmp_lt(b, op_qh, rocke_b_const_i32(b, ctx->NUM_QH));
            op_mask = rocke_b_land(b, mask_pos, mask_qh);
        }
        in_names[0] = coord_names[0];
        in_names[1] = coord_names[1];
        in_names[2] = coord_names[2];
        in_values[0] = rocke_b_add(b, ctx->cu_q_start, op_pos);
        in_values[1] = op_qh;
        in_values[2] = rocke_b_const_i32(b, 0);
        if(!rocke_transforms_descriptor_offset(
               b, ctx->q_desc, in_names, in_values, 3, &out_base, NULL))
            return NULL;

        for(stripe = 0; stripe < ctx->OUT_STRIPES; ++stripe)
        {
            int n_start = stripe * N_TILES_PER_STRIPE;
            int n_local;
            int chunk;
            for(n_local = 0; n_local < N_TILES_PER_STRIPE; ++n_local)
            {
                n = n_start + n_local;
                for(r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    rocke_value_t* row;
                    rocke_value_t* col_in_stripe;
                    rocke_value_t* v;
                    rocke_value_t* normalized;
                    rocke_value_t* final_h;
                    rocke_value_t* sidx[2];
                    atom = r / 4;
                    row = rocke_b_add(
                        b, ctx->wave_row_base, rocke_gfx950_attn2d_in_warp_row(ctx, r));
                    col_in_stripe = rocke_b_add(
                        b, rocke_b_const_i32(b, (int64_t)(n_local * MFMA_N)), ctx->lane_col_v);
                    v = rocke_b_vec_extract(
                        b, rocke_gfx950_attn2d_acc_final_get(ctx, n, atom), r % 4);
                    normalized = rocke_b_fmul(b, v, ctx->rcp_l[r]);
                    final_h = rocke_b_cast_f32_to(
                        b, rocke_b_select(b, ctx->l_nonzero[r], normalized, ctx->zero_f), dtype);
                    sidx[0] = row;
                    sidx[1] = col_in_stripe;
                    rocke_b_smem_store_vN(b, ctx->Acc_lds, sidx, 2, final_h, 1);
                }
            }
            rocke_b_sync(b);
            for(chunk = 0; chunk < OUT_CHUNKS_PER_THREAD; ++chunk)
            {
                rocke_value_t* col_in_stripe = rocke_b_add(
                    b, OUT_col_base_in_stripe, rocke_b_const_i32(b, (int64_t)(chunk * OUT_VEC)));
                rocke_value_t* lidx[2];
                rocke_value_t* v8h;
                rocke_value_t* out_col;
                rocke_if_t iff;
                lidx[0] = OUT_ROW_BASE;
                lidx[1] = col_in_stripe;
                v8h = rocke_b_smem_load_vN(b, ctx->Acc_lds, lidx, 2, dtype, OUT_VEC);
                out_col
                    = rocke_b_add(b,
                                  rocke_b_const_i32(b, (int64_t)(stripe * ctx->OUT_STRIPE_COLS)),
                                  col_in_stripe);
                iff = rocke_b_scf_if(b, op_mask);
                rocke_b_region_enter(b, iff.then_region);
                rocke_b_global_store_vN(
                    b, ctx->output, rocke_b_add(b, out_base, out_col), v8h, OUT_VEC, 16);
                rocke_b_region_leave(b);
            }
            if(stripe + 1 < ctx->OUT_STRIPES)
                rocke_b_sync(b);
        }
    }

    (void)i;
    return ctx->b->kernel;
}
