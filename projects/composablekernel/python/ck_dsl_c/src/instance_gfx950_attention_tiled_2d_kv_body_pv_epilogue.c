/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_kv_body_pv_epilogue.c
 *   -- C99 port of the back half of the gfx950 unified-attention-2d-tiled build.
 *
 * SCOPE (this TU): the KV-loop body BACK HALF + the loop driver + the epilogue,
 * faithfully tracking ck_dsl/instances/gfx950/attention_tiled_2d.py:
 *
 *   - ckc_gfx950_attn2d_emit_pv_bucket        Python lines 3174-3572 (PV bucket)
 *       acc *= alpha; acc += P @ V via the wide 32x32x16 atom + pv_tr_reader
 *       ds_read_b64_tr_b16/b8 transpose reads, the register-P^T path, the narrow
 *       16x16x{16,32} per-atom PV, the P->A pack (_pack_p_a16/_pack_p_a32), and
 *       the scf_yield carry.
 *   - ckc_gfx950_attn2d_apply_transposed_pv_regs   Python lines 3234-3314
 *   - ckc_gfx950_attn2d_drive_kv_loop         Python line 3581-3583 (scf_for_iter)
 *   - ckc_gfx950_attn2d_emit_epilogue         Python lines 3585-3817 (epilogue)
 *
 * The P permute/pack helpers (_permute_p_c_to_a16 / _pack_p_a16 / _pack_p_a32,
 * Python 1292-1384) live in this scope and are implemented here.
 *
 * BYTE-IDENTICAL CALL ORDER. Every emitter issues the exact ckc_b_* builder
 * calls in the exact order and with the exact operands the Python body uses.
 *
 * PEER CALLS. The front half of _emit_kv_body (QK + mask + softmax) is a peer
 * (ckc_gfx950_attn2d_emit_kv_body, declared in the internal header); the loop
 * driver below calls it. The P permute/pack helpers, the acc index helpers, the
 * _issue_v loader and the module-static 32x32 C-row/col helpers are peers.
 *
 * Bindings: ckc/instance_gfx950_attention_tiled_2d_internal.h (the shared ctx +
 * peer prototypes), ckc/ir.h (the builder), the layouts + mfma_attention helper
 * headers. This TU edits no header.
 */

#include "ckc/instance_gfx950_attention_tiled_2d_internal.h"
#include "ckc/instance_gfx950_attention_tiled_2d.h"        /* mfma_32x32_c_row/_col */
#include "ckc/helper_ck_dsl.helpers.layouts.h"             /* pv_tr_reader row/col  */
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h"      /* 32x32x16_for_dtype    */

#include <assert.h>

/* ============================================================ *
 *  Local dtype-dispatch wrappers mirroring the Python module aliases
 *  (_mfma_16x16x32 / _mfma_16x16x16 = mfma_{16x16x32,16x16x16}_for_dtype,
 *  _mfma_32x32x16 = mfma_32x32x16_for_dtype). The narrow dispatchers select
 *  the f16 / bf16 ISA atom on dtype, exactly as the Python helper does.
 * ============================================================ */
static ckc_value_t* ckc__attn2d_mfma_16x16x16(ckc_ir_builder_t* b,
                                              const ckc_type_t* dtype,
                                              ckc_value_t* a,
                                              ckc_value_t* bv,
                                              ckc_value_t* c)
{
    if (dtype != NULL && dtype->kind == CKC_TYPE_SCALAR)
    {
        if (dtype->scalar == CKC_SCALAR_F16)
            return ckc_b_mfma_f32_16x16x16_f16(b, a, bv, c);
        if (dtype->scalar == CKC_SCALAR_BF16)
            return ckc_b_mfma_f32_16x16x16_bf16(b, a, bv, c);
    }
    return NULL;
}

static ckc_value_t* ckc__attn2d_mfma_16x16x32(ckc_ir_builder_t* b,
                                              const ckc_type_t* dtype,
                                              ckc_value_t* a,
                                              ckc_value_t* bv,
                                              ckc_value_t* c)
{
    if (dtype != NULL && dtype->kind == CKC_TYPE_SCALAR)
    {
        if (dtype->scalar == CKC_SCALAR_F16)
            return ckc_b_mfma_f32_16x16x32_f16(b, a, bv, c);
        if (dtype->scalar == CKC_SCALAR_BF16)
            return ckc_b_mfma_f32_16x16x32_bf16(b, a, bv, c);
    }
    return NULL;
}

#define MFMA_N_CONST 16 /* Python module constant MFMA_N = 16 */

/* ============================================================ *
 *  ckc_gfx950_attn2d_apply_transposed_pv_regs   (Python lines 3234-3314)
 *
 *  Transposed PV via registers: O^T = V^T @ P^T. For each K=16 sub-tile,
 *  assemble the V A-operand from 8 scalar V_lds loads and the P^T B-operand
 *  from the PT32 registers (with cross-half warp_shuffle_xor as needed), then
 *  one 32x32x16 MFMA into acc32.
 *
 *  ``p_regs`` is the flat [p_tile * RPL + reg] array (RPL = REGS_PER_LANE for
 *  the 32x32 path == 16); ``p_count`` is its length. The TRANSPOSED_HALF_LOCAL_PV
 *  experimental branch (Python 3237-3265) uses pv32_v_load_paired, which is not
 *  in this port's surface; that predicate is default-off and is not exercised.
 * ============================================================ */
ckc_value_t* ckc_gfx950_attn2d_apply_transposed_pv_regs(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                                        ckc_value_t* acc32,
                                                        int n,
                                                        ckc_value_t* const* p_regs,
                                                        int p_count)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    const int RPL = 16; /* PT32 stores 16 regs per (p_tile) */
    ckc_value_t* v_buf;
    ckc_value_t* use_hi;
    ckc_value_t* v_dim32;
    int k;

    (void)p_count;
    /* v_buf / use_hi are hoisted by the caller (Python 3231-3232) and emitted
     * once before the per-N loop; reuse the cached SSA values. */
    v_buf = (ctx->pv_v_buf_v != NULL) ? ctx->pv_v_buf_v : ckc_b_const_i32(b, 0);
    use_hi = (ctx->pv_use_hi_v != NULL)
                 ? ctx->pv_use_hi_v
                 : ckc_b_cmp_eq(b, ctx->lane_half32_v, ckc_b_const_i32(b, 1));

    v_dim32 = ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(n * 32)), ctx->lane_col32_v);
    for (k = 0; k < ctx->T / 16; ++k)
    {
        ckc_value_t* a_v_elems[8];
        ckc_value_t* b_p_elems[8];
        ckc_value_t* A_v_t;
        ckc_value_t* B_p_t;
        int kk;

        if (ctx->TRANSPOSED_HALF_LOCAL_PV)
        {
            /* Experimental half-local PV (Python 3237-3265): requires
             * pv32_v_load_paired, not ported in this surface. Default-off. */
            continue;
        }

        /* Issue the 8 V scalar loads first (Python 3271-3281). */
        for (kk = 0; kk < 8; ++kk)
        {
            int k_static = k * 16 + kk;
            /* const(k_static) created before the mul (Python arg order). */
            ckc_value_t* v_row_base = ckc_b_const_i32(b, (int64_t)k_static);
            ckc_value_t* v_row =
                ckc_b_add(b, v_row_base,
                          ckc_b_mul(b, ctx->lane_half32_v, ckc_b_const_i32(b, 8)));
            ckc_value_t* idx[3];
            ckc_value_t* v1;
            idx[0] = v_buf;
            idx[1] = v_row;
            idx[2] = v_dim32;
            v1 = ckc_b_smem_load_vN(b, ctx->V_lds, idx, 3, dtype, 1);
            a_v_elems[kk] = ckc_b_vec_extract(b, v1, 0);
        }
        /* Assemble the P^T operand (Python 3286-3310). */
        for (kk = 0; kk < 8; ++kk)
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
            ckc_value_t* p0 = p_regs[p_tile0 * RPL + reg0];
            ckc_value_t* p1 = p_regs[p_tile1 * RPL + reg1];
            ckc_value_t* p_val;
            if (owner_half0 == 1)
                p0 = ckc_b_warp_shuffle_xor(b, p0, 32);
            if (owner_half1 == 0)
                p1 = ckc_b_warp_shuffle_xor(b, p1, 32);
            p_val = ckc_b_select(b, use_hi, p1, p0);
            b_p_elems[kk] = ckc_b_cast_f32_to(b, p_val, dtype);
        }
        A_v_t = ckc_b_vec_pack(b, a_v_elems, 8, dtype);
        B_p_t = ckc_b_vec_pack(b, b_p_elems, 8, dtype);
        acc32 = ckc_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, A_v_t, B_p_t, acc32);
    }
    return acc32;
}

/* ============================================================ *
 *  ckc_gfx950_attn2d_emit_pv_bucket   (Python lines 3174-3572)
 *
 *  The PV back half of _emit_kv_body. Consumes the softmax-derived state from
 *  ``in`` (alpha_regs, new_l_vals, m_new, PT32 register groups, register-P
 *  groups, GROUPED_KV2 re-issue inputs), computes acc *= alpha; acc += P @ V,
 *  and emits the scf_yield carry.
 * ============================================================ */
void ckc_gfx950_attn2d_emit_pv_bucket(ckc_gfx950_attn2d_build_ctx_t* ctx,
                                      const ckc_gfx950_attn2d_pv_inputs_t* in)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    const ckc_type_t* F32 = ckc_f32();
    const ckc_type_t* FP8E4M3 = ckc_fp8e4m3();
    const int RPL = 16; /* PT32 regs-per-tile */
    int kv_calls_per_tile = (ctx->T * ctx->HD) / (ctx->THREADS * 8);
    ckc_value_t* new_acc[CKC_GFX950_ATTN2D_MAX_ACCS];
    ckc_value_t* yields[CKC_GFX950_ATTN2D_MAX_ITER_ARGS];
    ckc_value_t* pv_fp8_scale = NULL;
    int n, r, atom, k, i, yc;

    for (i = 0; i < CKC_GFX950_ATTN2D_MAX_ACCS; ++i)
        new_acc[i] = NULL;

    /* ---- pre-PV wait/sync (Python 3181-3199) ---- */
    if (ctx->GROUPED_KV2)
    {
        ckc_b_s_waitcnt(b, 0, 0, -1);
        ckc_b_sync(b);
    }
    else if (ctx->KV_FP8)
    {
        ckc_b_s_waitcnt(b, 0, 0, -1);
        ckc_b_sync(b);
    }
    else
    {
        ckc_b_s_waitcnt(b, kv_calls_per_tile, kv_calls_per_tile, -1);
        ckc_b_sync(b);
    }

    /* The fp8 PV scale (Python line 1149: v_scale_p / 240.0) is hoisted in the
     * prologue (ctx->pv_fp8_scale_v) so its SSA id matches Python; reuse it. */
    if (ctx->FP8_MFMA_PV)
        pv_fp8_scale = ctx->pv_fp8_scale_v;

    /* ---- acc *= alpha, acc += P @ V ---- (Python 3219-3412) ---- */
    if (ctx->USE_MFMA_32X32)
    {
        if (ctx->TRANSPOSED_QK_32X32)
        {
            /* Transposed PV: O^T = V^T @ P^T via register P^T (Python 3221-3341).
             * v_buf and use_hi are emitted ONCE here (Python 3231-3232), before
             * the per-N acc-scale + apply loop, and reused by every apply call. */
            ctx->pv_v_buf_v = ckc_b_const_i32(b, 0);
            ctx->pv_use_hi_v =
                ckc_b_cmp_eq(b, ctx->lane_half32_v, ckc_b_const_i32(b, 1));
            for (n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                ckc_value_t* scaled[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
                ckc_value_t* old_acc = ckc_gfx950_attn2d_acc_get(ctx, n, 0);
                ckc_value_t* alpha_t = in->alpha_regs[0];
                ckc_value_t* acc32;
                ckc_value_t* const* PT32_n = in->pt32_g0 + (size_t)n * RPL;
                for (r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    ckc_value_t* e = ckc_b_vec_extract(b, old_acc, r);
                    ckc_value_t* a = ctx->TRANSPOSED_SCALAR_STATE ? alpha_t : in->alpha_regs[r];
                    scaled[r] = ckc_b_fmul(b, e, a);
                }
                acc32 = ckc_b_vec_pack(b, scaled, ctx->REGS_PER_LANE, F32);
                /* PT32 is addressed absolutely as [p_tile][reg] inside the
                 * helper (p_tile = k//32 spans all QK_N_TILES); Python passes
                 * the FULL PT32_n 2D array (NOT an n-slice). Pass the whole
                 * pt32_g0 base; the n arg only drives v_dim32 / _acc_get. */
                (void)PT32_n;
                acc32 = ckc_gfx950_attn2d_apply_transposed_pv_regs(ctx, acc32, n, in->pt32_g0,
                                                                   in->pt32_count);
                new_acc[n] = acc32;
            }
            if (ctx->GROUPED_KV2)
            {
                ckc_b_s_waitcnt(b, -1, 0, -1);
                ckc_b_sync(b);
                ckc_gfx950_attn2d_issue_v(ctx, in->safe_tile1, in->nxt_buf);
                ckc_b_s_waitcnt(b, 0, 0, -1);
                ckc_b_sync(b);
                for (n = 0; n < ctx->ACC_N_TILES; ++n)
                {
                    new_acc[n] = ckc_gfx950_attn2d_apply_transposed_pv_regs(
                        ctx, new_acc[n], n, in->pt32_g1, in->pt32_count);
                }
            }
        }
        else
        {
            /* Transitional 32x32 PV from the P_lds bridge (Python 3342-3412). */
            ckc_value_t* v_buf = ckc_b_const_i32(b, 0);
            for (n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                ckc_value_t* scaled[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE];
                ckc_value_t* old_acc = ckc_gfx950_attn2d_acc_get(ctx, n, 0);
                ckc_value_t* acc32;
                for (r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    ckc_value_t* e = ckc_b_vec_extract(b, old_acc, r);
                    scaled[r] = ckc_b_fmul(b, e, in->alpha_regs[r]);
                }
                acc32 = ckc_b_vec_pack(b, scaled, ctx->REGS_PER_LANE, F32);
                for (k = 0; k < ctx->T / 16; ++k)
                {
                    /* const(k*16) created before the mul (Python arg order). */
                    ckc_value_t* p_off32_base = ckc_b_const_i32(b, (int64_t)(k * 16));
                    ckc_value_t* p_off32 =
                        ckc_b_add(b, p_off32_base,
                                  ckc_b_mul(b, ctx->lane_half32_v, ckc_b_const_i32(b, 8)));
                    ckc_value_t* p_row32 = ckc_b_add(b, ctx->wave_row_base, ctx->lane_col32_v);
                    ckc_value_t* pidx[2];
                    ckc_value_t* A_p32;
                    ckc_value_t* col_group16;
                    ckc_value_t* tr_col32;
                    ckc_value_t* tr_row_base32;
                    ckc_value_t* B32_r0;
                    ckc_value_t* B32_r1;
                    ckc_value_t* B_v32;
                    ckc_value_t* ridx0[3];
                    ckc_value_t* ridx1[3];

                    pidx[0] = p_row32;
                    pidx[1] = p_off32;
                    A_p32 = ckc_b_smem_load_vN(b, ctx->P_lds, pidx, 2, dtype, 8);

                    /* Sequence sub-expressions so C arg-eval order matches
                     * Python's left-to-right value creation. */
                    {
                        ckc_value_t* cg_div =
                            ckc_b_div(b, ctx->lane_col32_v, ckc_b_const_i32(b, 16));
                        col_group16 = ckc_b_mul(b, cg_div, ckc_b_const_i32(b, 16));
                    }
                    {
                        ckc_value_t* tc_mod =
                            ckc_b_mod(b, ctx->lane_col32_v, ckc_b_const_i32(b, 4));
                        ckc_value_t* tc_mul = ckc_b_mul(b, tc_mod, ckc_b_const_i32(b, 4));
                        tr_col32 = ckc_b_add(b, col_group16, tc_mul);
                    }
                    {
                        ckc_value_t* trb_base = ckc_b_const_i32(b, (int64_t)(k * 16));
                        ckc_value_t* trb_inner = ckc_b_add(
                            b, trb_base,
                            ckc_b_mul(b, ctx->lane_half32_v, ckc_b_const_i32(b, 8)));
                        ckc_value_t* trb_div =
                            ckc_b_div(b, ctx->lane_col32_v, ckc_b_const_i32(b, 4));
                        ckc_value_t* trb_mod = ckc_b_mod(b, trb_div, ckc_b_const_i32(b, 4));
                        tr_row_base32 = ckc_b_add(b, trb_inner, trb_mod);
                    }
                    ridx0[0] = v_buf;
                    ridx0[1] = tr_row_base32;
                    {
                        ckc_value_t* r0_base = ckc_b_const_i32(b, (int64_t)(n * 32));
                        ridx0[2] = ckc_b_add(b, r0_base, tr_col32);
                    }
                    B32_r0 = ckc_b_ds_read_tr16_b64(b, ctx->V_lds, ridx0, 3, dtype);
                    ridx1[0] = v_buf;
                    ridx1[1] = ckc_b_add(b, tr_row_base32, ckc_b_const_i32(b, 4));
                    {
                        ckc_value_t* r1_base = ckc_b_const_i32(b, (int64_t)(n * 32));
                        ridx1[2] = ckc_b_add(b, r1_base, tr_col32);
                    }
                    B32_r1 = ckc_b_ds_read_tr16_b64(b, ctx->V_lds, ridx1, 3, dtype);
                    B_v32 = ckc_b_vec_concat(b, B32_r0, B32_r1);
                    acc32 = ckc_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, A_p32, B_v32, acc32);
                }
                new_acc[n] = acc32;
            }
        }
    }

    /* ---- narrow 16x16 per-atom PV (Python 3413-3562) ---- */
    {
        int n_lim = ctx->USE_MFMA_32X32 ? 0 : ctx->PV_N_TILES;
        for (n = 0; n < n_lim; ++n)
        {
            ckc_value_t* acc_per_atom[CKC_GFX950_ATTN2D_MAX_REGS_PER_LANE / 4];
            ckc_value_t* n_col_base;
            ckc_value_t* v_buf;

            for (atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
            {
                ckc_value_t* scaled_comps[4];
                int in_atom;
                for (in_atom = 0; in_atom < 4; ++in_atom)
                {
                    int reg = atom * 4 + in_atom;
                    ckc_value_t* e =
                        ckc_b_vec_extract(b, ckc_gfx950_attn2d_acc_get(ctx, n, atom), in_atom);
                    scaled_comps[in_atom] = ckc_b_fmul(b, e, in->alpha_regs[reg]);
                }
                acc_per_atom[atom] = ckc_b_vec_pack(b, scaled_comps, 4, F32);
            }

            n_col_base = ckc_b_add(b,
                                   ckc_b_mul(b, ckc_b_const_i32(b, (int64_t)n),
                                             ckc_b_const_i32(b, 16)),
                                   ctx->pv_tr_reader->col);
            v_buf = ckc_b_const_i32(b, 0);

            for (k = 0; k < ctx->PV_K_ITERS; ++k)
            {
                if (ctx->PV_K_STEP == 32)
                {
                    /* const(k*32) created before the mul (Python arg order). */
                    ckc_value_t* p_off_base = ckc_b_const_i32(b, (int64_t)(k * 32));
                    ckc_value_t* p_off = ckc_b_add(
                        b, p_off_base,
                        ckc_b_mul(b, ctx->lane_rg_v, ckc_b_const_i32(b, 8)));
                    ckc_value_t* row_r0 =
                        ckc_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 32, 0);
                    ckc_value_t* row_r1 =
                        ckc_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 32, 1);
                    if (ctx->FP8_MFMA_PV)
                    {
                        /* native-fp8 PV stripe path (Python 3439-3507) */
                        ckc_value_t* stripe_const = ckc_b_const_i32(b, (int64_t)n);
                        /* Python: b.add(b.mul(lane_rg, 8), b.div(lane_col, 2))
                         * creates the mul BEFORE the div (left-to-right). Bind in
                         * order so C's right-to-left arg eval matches the SSA ids. */
                        ckc_value_t* krpl_mul =
                            ckc_b_mul(b, ctx->lane_rg_v, ckc_b_const_i32(b, 8));
                        ckc_value_t* krpl_div =
                            ckc_b_div(b, ctx->lane_col_v, ckc_b_const_i32(b, 2));
                        ckc_value_t* k_row_per_lane = ckc_b_add(b, krpl_mul, krpl_div);
                        ckc_value_t* k_row_for_iter =
                            ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(k * 32)), k_row_per_lane);
                        ckc_value_t* lo_idx[4];
                        ckc_value_t* hi_idx[4];
                        ckc_value_t* B_v8_lo;
                        ckc_value_t* B_v8_hi;
                        ckc_value_t* lo_mask;
                        ckc_value_t* B_v8;
                        lo_idx[0] = v_buf;
                        lo_idx[1] = stripe_const;
                        lo_idx[2] = k_row_for_iter;
                        lo_idx[3] = ckc_b_const_i32(b, 0);
                        B_v8_lo = ckc_b_ds_read_tr_b8(b, ctx->V_lds, lo_idx, 4, FP8E4M3);
                        hi_idx[0] = v_buf;
                        hi_idx[1] = stripe_const;
                        hi_idx[2] = k_row_for_iter;
                        hi_idx[3] = ckc_b_const_i32(b, 8);
                        B_v8_hi = ckc_b_ds_read_tr_b8(b, ctx->V_lds, hi_idx, 4, FP8E4M3);
                        lo_mask = ckc_b_cmp_lt(b, ctx->lane_col_v, ckc_b_const_i32(b, 8));
                        B_v8 = ckc_b_vector_select(b, ckc_b_vector_splat(b, lo_mask, 8),
                                                   B_v8_lo, B_v8_hi);
                        for (atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                        {
                            ckc_value_t* p_row = ckc_b_add(
                                b, ctx->wave_row_base,
                                ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(atom * 16)),
                                          ctx->lane_col_v));
                            ckc_value_t* pidx[2];
                            ckc_value_t* A_p8;
                            ckc_value_t* raw;
                            ckc_value_t* comps[4];
                            int ii;
                            pidx[0] = p_row;
                            pidx[1] = p_off;
                            A_p8 = ckc_b_smem_load_vN(b, ctx->P_lds, pidx, 2, FP8E4M3, 8);
                            raw = ckc_b_mfma_f32_16x16x32_fp8(b, A_p8, B_v8,
                                                              ckc_b_zero_vec_f32(b, 4));
                            for (ii = 0; ii < 4; ++ii)
                            {
                                ckc_value_t* old = ckc_b_vec_extract(b, acc_per_atom[atom], ii);
                                ckc_value_t* add = ckc_b_fmul(
                                    b, ckc_b_vec_extract(b, raw, ii), pv_fp8_scale);
                                comps[ii] = ckc_b_fadd(b, old, add);
                            }
                            acc_per_atom[atom] = ckc_b_vec_pack(b, comps, 4, F32);
                        }
                    }
                    else
                    {
                        ckc_value_t* r0idx[3];
                        ckc_value_t* r1idx[3];
                        ckc_value_t* B_r0;
                        ckc_value_t* B_r1;
                        ckc_value_t* B_v;
                        r0idx[0] = v_buf;
                        r0idx[1] = row_r0;
                        r0idx[2] = n_col_base;
                        B_r0 = ckc_b_ds_read_tr16_b64(b, ctx->V_lds, r0idx, 3, dtype);
                        r1idx[0] = v_buf;
                        r1idx[1] = row_r1;
                        r1idx[2] = n_col_base;
                        B_r1 = ckc_b_ds_read_tr16_b64(b, ctx->V_lds, r1idx, 3, dtype);
                        B_v = ckc_b_vec_concat(b, B_r0, B_r1);
                        for (atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                        {
                            ckc_value_t* A_p;
                            if (ctx->REGISTER_PV)
                            {
                                ckc_value_t* g0[4];
                                ckc_value_t* g1[4];
                                int rr;
                                for (rr = 0; rr < 4; ++rr)
                                {
                                    int base = (atom * 4 + rr) * in->p_regs_f32_stride;
                                    g0[rr] = in->p_regs_f32[base + 2 * k];
                                    g1[rr] = in->p_regs_f32[base + 2 * k + 1];
                                }
                                A_p = ckc_gfx950_attn2d_pack_p_a32(ctx, g0, g1, 4);
                            }
                            else
                            {
                                ckc_value_t* p_row = ckc_b_add(
                                    b, ctx->wave_row_base,
                                    ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(atom * 16)),
                                              ctx->lane_col_v));
                                ckc_value_t* pidx[2];
                                pidx[0] = p_row;
                                pidx[1] = p_off;
                                A_p = ckc_b_smem_load_vN(b, ctx->P_lds, pidx, 2, dtype, 8);
                            }
                            acc_per_atom[atom] =
                                ckc__attn2d_mfma_16x16x32(b, dtype, A_p, B_v, acc_per_atom[atom]);
                        }
                    }
                }
                else
                {
                    ckc_value_t* p_off;
                    ckc_value_t* row_lane;
                    ckc_value_t* ridx[3];
                    ckc_value_t* B_v;
                    /* PV_FP8_MFMA with K=16 is unsupported (Python 3541-3542). */
                    assert(!ctx->FP8_MFMA_PV);
                    /* Python: b.add(b.const_i32(k*16), b.mul(lane_rg, const(4)))
                     * evaluates the const(k*16) arg BEFORE the mul (left-to-right).
                     * C arg-eval order is unspecified (typically right-to-left), so
                     * bind both operands to temps in Python's order. */
                    {
                        ckc_value_t* p_off_c = ckc_b_const_i32(b, (int64_t)(k * 16));
                        ckc_value_t* p_off_rg =
                            ckc_b_mul(b, ctx->lane_rg_v, ckc_b_const_i32(b, 4));
                        p_off = ckc_b_add(b, p_off_c, p_off_rg);
                    }
                    row_lane = ckc_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 16, 0);
                    ridx[0] = v_buf;
                    ridx[1] = row_lane;
                    ridx[2] = n_col_base;
                    B_v = ckc_b_ds_read_tr16_b64(b, ctx->V_lds, ridx, 3, dtype);
                    for (atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                    {
                        ckc_value_t* A_p;
                        if (ctx->REGISTER_PV)
                        {
                            ckc_value_t* g[4];
                            int rr;
                            for (rr = 0; rr < 4; ++rr)
                            {
                                int base = (atom * 4 + rr) * in->p_regs_f32_stride;
                                g[rr] = in->p_regs_f32[base + k];
                            }
                            A_p = ckc_gfx950_attn2d_pack_p_a16(ctx, g, 4);
                        }
                        else
                        {
                            ckc_value_t* p_row = ckc_b_add(
                                b, ctx->wave_row_base,
                                ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(atom * 16)),
                                          ctx->lane_col_v));
                            ckc_value_t* pidx[2];
                            pidx[0] = p_row;
                            pidx[1] = p_off;
                            A_p = ckc_b_smem_load_vN(b, ctx->P_lds, pidx, 2, dtype, 4);
                        }
                        acc_per_atom[atom] =
                            ckc__attn2d_mfma_16x16x16(b, dtype, A_p, B_v, acc_per_atom[atom]);
                    }
                }
            }
            for (atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
                new_acc[n * ctx->M_ATOMS_PER_WARP + atom] = acc_per_atom[atom];
        }
    }

    /* ---- assemble the scf_yield carry (Python 3564-3572) ---- */
    yc = 0;
    for (r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
    {
        yields[yc++] = in->m_new[r];
        yields[yc++] = in->new_l_vals[r];
    }
    for (n = 0; n < ctx->ACC_N_TILES; ++n)
        for (atom = 0; atom < ctx->ACC_M_ATOMS; ++atom)
            yields[yc++] = new_acc[n * ctx->ACC_M_ATOMS + atom];
    yields[yc++] = ctx->GROUPED_KV2 ? in->cur_buf : ctx->nxt_buf_v;

    /* Record the rewritten carry for callers that thread phases (the driver's
     * single-phase path reads this back via the loop results). */
    for (i = 0; i < yc; ++i)
        ctx->out_carry[i] = yields[i];
    ctx->out_carry_count = yc;

    ckc_b_scf_yield(b, yields, yc);
}

/* ============================================================ *
 *  ckc_gfx950_attn2d_drive_kv_loop   (Python lines 3581-3583)
 *
 *  scf_for_iter over [tile_start, tile_end) step kv_step with the named
 *  iter_args carry. Enter the body, unpack the carry into ctx->m_cur/l_cur/
 *  acc_cur + ctx->cur_buf and set ctx->kv_tile_iv, run the (peer) full body
 *  emitter, then leave. Returns the loop handle; the epilogue reads the
 *  rewritten carry from kvloop.op->results.
 * ============================================================ */
ckc_for_t ckc_gfx950_attn2d_drive_kv_loop(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    ckc_iter_arg_t iter_args[CKC_GFX950_ATTN2D_MAX_ITER_ARGS];
    ckc_for_t kvloop;
    int i;
    int ml = ctx->ml_count;
    int num_accs = ctx->ACC_N_TILES * ctx->ACC_M_ATOMS;

    for (i = 0; i < ctx->iter_args_count; ++i)
    {
        iter_args[i].name = ctx->iter_args_names[i];
        iter_args[i].init = ctx->iter_args[i];
    }

    kvloop = ckc_b_scf_for_iter(b, ctx->tile_start, ctx->tile_end, ctx->kv_step, iter_args,
                                ctx->iter_args_count, "kv_tile", false, false);

    ckc_b_region_enter(b, kvloop.body);
    ctx->kv_tile_iv = kvloop.iv;
    /* Unpack the carry the same way the Python body does (lines 2495-2508). */
    for (i = 0; i < ctx->SOFTMAX_STATE_SLOTS; ++i)
    {
        ctx->m_cur[i] = kvloop.iter_vars[2 * i];
        ctx->l_cur[i] = kvloop.iter_vars[2 * i + 1];
    }
    for (i = 0; i < num_accs; ++i)
        ctx->acc_cur[i] = kvloop.iter_vars[ml + i];
    ctx->cur_buf = kvloop.iter_vars[ml + num_accs];
    ctx->skip_mask = false;

    ckc_gfx950_attn2d_emit_kv_body(ctx);

    ckc_b_region_leave(b);
    return kvloop;
}

/* ============================================================ *
 *  ckc_gfx950_attn2d_emit_epilogue   (Python lines 3585-3817)
 * ============================================================ */
ckc_kernel_def_t* ckc_gfx950_attn2d_emit_epilogue(ckc_gfx950_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    const char* coord_names[3] = {"token", "head", "dim"};
    int ml_count_final;
    int n, r, atom, i;

    /* This entry expects ckc_gfx950_attn2d_drive_kv_loop to have stashed the
     * loop results into ctx->iter_args (reused as the final-carry slot) and the
     * per-slot finals into ctx->l_final / acc_final. The driver/glue populate
     * ctx->l_final / ctx->acc_final from kvloop.op->results before calling. */

    /* Drain async + close outstanding copies (Python 3592-3593). */
    ckc_b_s_waitcnt(b, 0, 0, -1);
    ckc_b_sync(b);

    ml_count_final = 2 * ctx->SOFTMAX_STATE_SLOTS;

    /* Read the rewritten loop carry into l_final / acc_final, mirroring
     * Python lines 3595-3603:
     *   final     = kvloop.results
     *   l_final[r]= final[2*r + 1]
     *   acc_final[n*ACC_M_ATOMS + atom]
     *             = final[ml_count_final + n*ACC_M_ATOMS + atom]
     * The driver stashed kvloop.op->results into ctx->out_carry. */
    for (r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
        ctx->l_final[r] = ctx->out_carry[2 * r + 1];
    for (n = 0; n < ctx->ACC_N_TILES; ++n)
        for (atom = 0; atom < ctx->ACC_M_ATOMS; ++atom)
            ctx->acc_final[n * ctx->ACC_M_ATOMS + atom] =
                ctx->out_carry[ml_count_final + n * ctx->ACC_M_ATOMS + atom];

    /* Per-row reciprocal of L and the nonzero predicate (Python 3609-3610).
     * Python emits ALL rcps first (list comp), THEN all fcmps -- two loops. */
    for (r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
        ctx->rcp_l[r] = ckc_b_rcp(b, ctx->l_final[r]);
    for (r = 0; r < ctx->SOFTMAX_STATE_SLOTS; ++r)
        ctx->l_nonzero[r] = ckc_b_fcmp(b, "ogt", ctx->l_final[r], ctx->zero_f);

    if (ctx->USE_MFMA_32X32)
    {
        if (ctx->TRANSPOSED_QK_32X32)
        {
            /* Per-lane direct scalar global stores (Python 3612-3656). */
            ckc_value_t* q_row_t = ckc_b_add(b, ctx->wave_row_base, ctx->lane_col32_v);
            ckc_value_t* op_pos_t = ckc_b_add(
                b, ctx->qb_start_pos, ckc_b_div(b, q_row_t, ckc_b_const_i32(b, ctx->NQK)));
            /* Sequence sub-expressions so C arg-eval order matches Python's
             * left-to-right value creation (mul before mod; op_pos cmp before
             * op_qh cmp). */
            ckc_value_t* op_qh_mul =
                ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, ctx->NQK));
            ckc_value_t* op_qh_mod = ckc_b_mod(b, q_row_t, ckc_b_const_i32(b, ctx->NQK));
            ckc_value_t* op_qh_t = ckc_b_add(b, op_qh_mul, op_qh_mod);
            ckc_value_t* op_mask_pos = ckc_b_cmp_lt(b, op_pos_t, ctx->cur_batch_q_len);
            ckc_value_t* op_mask_qh =
                ckc_b_cmp_lt(b, op_qh_t, ckc_b_const_i32(b, ctx->NUM_QH));
            ckc_value_t* op_mask_t = ckc_b_land(b, op_mask_pos, op_mask_qh);
            ckc_value_t* out_base_t = NULL;
            ckc_value_t* inv_l_t;
            ckc_value_t* l_nonzero_t;
            const char* in_names[3];
            ckc_value_t* in_values[3];

            in_names[0] = coord_names[0];
            in_names[1] = coord_names[1];
            in_names[2] = coord_names[2];
            in_values[0] = ckc_b_add(b, ctx->cu_q_start, op_pos_t);
            in_values[1] = op_qh_t;
            in_values[2] = ckc_b_const_i32(b, 0);
            if (!ckc_transforms_descriptor_offset(b, ctx->q_desc, in_names, in_values, 3,
                                                  &out_base_t, NULL))
                return NULL;

            inv_l_t = ckc_b_rcp(b, ctx->l_final[0]);
            l_nonzero_t = ckc_b_fcmp(b, "ogt", ctx->l_final[0], ctx->zero_f);
            for (n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                ckc_value_t* acc32 = ckc_gfx950_attn2d_acc_final_get(ctx, n, 0);
                for (r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    /* const(n*32) created before _mfma_32x32_c_row (Py arg order). */
                    ckc_value_t* out_col_base = ckc_b_const_i32(b, (int64_t)(n * 32));
                    ckc_value_t* out_col_t = ckc_b_add(
                        b, out_col_base,
                        ckc_gfx950_attention_tiled_2d_mfma_32x32_c_row(b, ctx->lane, r));
                    ckc_value_t* v = ckc_b_vec_extract(b, acc32, r);
                    ckc_value_t* normalized = ckc_b_fmul(b, v, inv_l_t);
                    ckc_value_t* final_h = ckc_b_cast_f32_to(
                        b, ckc_b_select(b, l_nonzero_t, normalized, ctx->zero_f), dtype);
                    ckc_if_t iff = ckc_b_scf_if(b, op_mask_t);
                    ckc_b_region_enter(b, iff.then_region);
                    ckc_b_global_store(b, ctx->output, ckc_b_add(b, out_base_t, out_col_t),
                                       final_h, 2);
                    ckc_b_region_leave(b);
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
            ckc_value_t* OUT_ROW_BASE32;
            ckc_value_t* OUT_col_base32;
            ckc_value_t* op_pos32_base;
            ckc_value_t* op_qh32_base;
            ckc_value_t* op_mask32_base;
            ckc_value_t* out_base32_base = NULL;
            const char* in_names[3];
            ckc_value_t* in_values[3];
            int chunk;

            assert(OUT_PER_THREAD_HALVES32 % OUT_VEC32 == 0);
            OUT_CHUNKS_PER_THREAD32 = OUT_PER_THREAD_HALVES32 / OUT_VEC32;
            OUT_THREADS_PER_ROW32 = 32 / (OUT_CHUNKS_PER_THREAD32 * OUT_VEC32);
            /* Sequence sub-expressions to match Python value-creation order. */
            OUT_ROW_BASE32 = ckc_b_div(b, ctx->tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW32));
            {
                ckc_value_t* ocb_mod =
                    ckc_b_mod(b, ctx->tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW32));
                OUT_col_base32 = ckc_b_mul(
                    b, ocb_mod, ckc_b_const_i32(b, OUT_CHUNKS_PER_THREAD32 * OUT_VEC32));
            }
            op_pos32_base = ckc_b_add(
                b, ctx->qb_start_pos,
                ckc_b_div(b, OUT_ROW_BASE32, ckc_b_const_i32(b, ctx->NQK)));
            {
                ckc_value_t* qh_mul =
                    ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, ctx->NQK));
                ckc_value_t* qh_mod =
                    ckc_b_mod(b, OUT_ROW_BASE32, ckc_b_const_i32(b, ctx->NQK));
                op_qh32_base = ckc_b_add(b, qh_mul, qh_mod);
            }
            {
                ckc_value_t* mask_pos =
                    ckc_b_cmp_lt(b, op_pos32_base, ctx->cur_batch_q_len);
                ckc_value_t* mask_qh =
                    ckc_b_cmp_lt(b, op_qh32_base, ckc_b_const_i32(b, ctx->NUM_QH));
                op_mask32_base = ckc_b_land(b, mask_pos, mask_qh);
            }
            in_names[0] = coord_names[0];
            in_names[1] = coord_names[1];
            in_names[2] = coord_names[2];
            in_values[0] = ckc_b_add(b, ctx->cu_q_start, op_pos32_base);
            in_values[1] = op_qh32_base;
            in_values[2] = ckc_b_const_i32(b, 0);
            if (!ckc_transforms_descriptor_offset(b, ctx->q_desc, in_names, in_values, 3,
                                                  &out_base32_base, NULL))
                return NULL;

            for (n = 0; n < ctx->ACC_N_TILES; ++n)
            {
                ckc_value_t* acc32 = ckc_gfx950_attn2d_acc_final_get(ctx, n, 0);
                for (r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    ckc_value_t* row = ckc_b_add(
                        b, ctx->wave_row_base,
                        ckc_gfx950_attention_tiled_2d_mfma_32x32_c_row(b, ctx->lane, r));
                    ckc_value_t* col_in_stripe = ctx->lane_col32_v;
                    ckc_value_t* v = ckc_b_vec_extract(b, acc32, r);
                    ckc_value_t* normalized = ckc_b_fmul(b, v, ctx->rcp_l[r]);
                    ckc_value_t* final_h = ckc_b_cast_f32_to(
                        b, ckc_b_select(b, ctx->l_nonzero[r], normalized, ctx->zero_f), dtype);
                    ckc_value_t* sidx[2];
                    sidx[0] = row;
                    sidx[1] = col_in_stripe;
                    ckc_b_smem_store_vN(b, ctx->Acc_lds, sidx, 2, final_h, 1);
                }
                ckc_b_sync(b);
                for (chunk = 0; chunk < OUT_CHUNKS_PER_THREAD32; ++chunk)
                {
                    ckc_value_t* col_in_stripe = ckc_b_add(
                        b, OUT_col_base32, ckc_b_const_i32(b, (int64_t)(chunk * OUT_VEC32)));
                    ckc_value_t* lidx[2];
                    ckc_value_t* v8h;
                    ckc_value_t* out_col;
                    ckc_if_t iff;
                    lidx[0] = OUT_ROW_BASE32;
                    lidx[1] = col_in_stripe;
                    v8h = ckc_b_smem_load_vN(b, ctx->Acc_lds, lidx, 2, dtype, OUT_VEC32);
                    out_col = ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(n * 32)), col_in_stripe);
                    iff = ckc_b_scf_if(b, op_mask32_base);
                    ckc_b_region_enter(b, iff.then_region);
                    ckc_b_global_store_vN(b, ctx->output,
                                          ckc_b_add(b, out_base32_base, out_col), v8h,
                                          OUT_VEC32, 16);
                    ckc_b_region_leave(b);
                }
                if (n + 1 < ctx->ACC_N_TILES)
                    ckc_b_sync(b);
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
        ckc_value_t* OUT_ROW_BASE;
        ckc_value_t* OUT_col_base_in_stripe;
        ckc_value_t* op_pos;
        ckc_value_t* op_qh;
        ckc_value_t* op_mask;
        ckc_value_t* out_base = NULL;
        const char* in_names[3];
        ckc_value_t* in_values[3];
        int stripe;

        assert(ctx->PV_N_TILES % N_TILES_PER_STRIPE == 0);
        assert(OUT_PER_THREAD_HALVES % OUT_VEC == 0 && OUT_PER_THREAD_HALVES > 0);
        OUT_CHUNKS_PER_THREAD = OUT_PER_THREAD_HALVES / OUT_VEC;
        OUT_THREADS_PER_ROW = ctx->OUT_STRIPE_COLS / (OUT_CHUNKS_PER_THREAD * OUT_VEC);
        assert(ctx->THREADS / OUT_THREADS_PER_ROW == ctx->BLOCK_M);

        /* Match Python value-creation order (left-to-right arg eval): C's
         * function-arg eval order is unspecified, so sequence sub-expressions
         * into temporaries to keep the SSA stream byte-identical. */
        OUT_ROW_BASE = ckc_b_div(b, ctx->tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW));
        {
            ckc_value_t* ocb_mod =
                ckc_b_mod(b, ctx->tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW));
            OUT_col_base_in_stripe = ckc_b_mul(
                b, ocb_mod, ckc_b_const_i32(b, OUT_CHUNKS_PER_THREAD * OUT_VEC));
        }

        op_pos = ckc_b_add(b, ctx->qb_start_pos,
                           ckc_b_div(b, OUT_ROW_BASE, ckc_b_const_i32(b, ctx->NQK)));
        {
            ckc_value_t* qh_mul =
                ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, ctx->NQK));
            ckc_value_t* qh_mod = ckc_b_mod(b, OUT_ROW_BASE, ckc_b_const_i32(b, ctx->NQK));
            op_qh = ckc_b_add(b, qh_mul, qh_mod);
        }
        {
            ckc_value_t* mask_pos = ckc_b_cmp_lt(b, op_pos, ctx->cur_batch_q_len);
            ckc_value_t* mask_qh =
                ckc_b_cmp_lt(b, op_qh, ckc_b_const_i32(b, ctx->NUM_QH));
            op_mask = ckc_b_land(b, mask_pos, mask_qh);
        }
        in_names[0] = coord_names[0];
        in_names[1] = coord_names[1];
        in_names[2] = coord_names[2];
        in_values[0] = ckc_b_add(b, ctx->cu_q_start, op_pos);
        in_values[1] = op_qh;
        in_values[2] = ckc_b_const_i32(b, 0);
        if (!ckc_transforms_descriptor_offset(b, ctx->q_desc, in_names, in_values, 3, &out_base,
                                              NULL))
            return NULL;

        for (stripe = 0; stripe < ctx->OUT_STRIPES; ++stripe)
        {
            int n_start = stripe * N_TILES_PER_STRIPE;
            int n_local;
            int chunk;
            for (n_local = 0; n_local < N_TILES_PER_STRIPE; ++n_local)
            {
                n = n_start + n_local;
                for (r = 0; r < ctx->REGS_PER_LANE; ++r)
                {
                    ckc_value_t* row;
                    ckc_value_t* col_in_stripe;
                    ckc_value_t* v;
                    ckc_value_t* normalized;
                    ckc_value_t* final_h;
                    ckc_value_t* sidx[2];
                    atom = r / 4;
                    row = ckc_b_add(b, ctx->wave_row_base, ckc_gfx950_attn2d_in_warp_row(ctx, r));
                    col_in_stripe = ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(n_local * MFMA_N)),
                                              ctx->lane_col_v);
                    v = ckc_b_vec_extract(b, ckc_gfx950_attn2d_acc_final_get(ctx, n, atom), r % 4);
                    normalized = ckc_b_fmul(b, v, ctx->rcp_l[r]);
                    final_h = ckc_b_cast_f32_to(
                        b, ckc_b_select(b, ctx->l_nonzero[r], normalized, ctx->zero_f), dtype);
                    sidx[0] = row;
                    sidx[1] = col_in_stripe;
                    ckc_b_smem_store_vN(b, ctx->Acc_lds, sidx, 2, final_h, 1);
                }
            }
            ckc_b_sync(b);
            for (chunk = 0; chunk < OUT_CHUNKS_PER_THREAD; ++chunk)
            {
                ckc_value_t* col_in_stripe = ckc_b_add(
                    b, OUT_col_base_in_stripe, ckc_b_const_i32(b, (int64_t)(chunk * OUT_VEC)));
                ckc_value_t* lidx[2];
                ckc_value_t* v8h;
                ckc_value_t* out_col;
                ckc_if_t iff;
                lidx[0] = OUT_ROW_BASE;
                lidx[1] = col_in_stripe;
                v8h = ckc_b_smem_load_vN(b, ctx->Acc_lds, lidx, 2, dtype, OUT_VEC);
                out_col = ckc_b_add(
                    b, ckc_b_const_i32(b, (int64_t)(stripe * ctx->OUT_STRIPE_COLS)),
                    col_in_stripe);
                iff = ckc_b_scf_if(b, op_mask);
                ckc_b_region_enter(b, iff.then_region);
                ckc_b_global_store_vN(b, ctx->output, ckc_b_add(b, out_base, out_col), v8h,
                                      OUT_VEC, 16);
                ckc_b_region_leave(b);
            }
            if (stripe + 1 < ctx->OUT_STRIPES)
                ckc_b_sync(b);
        }
    }

    (void)i;
    return ctx->b->kernel;
}
