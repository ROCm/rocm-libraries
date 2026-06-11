/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx942_attention_tiled_2d_epilogue.c -- EPILOGUE bucket of the C99
 * port of build_unified_attention_2d_tiled
 * (ck_dsl/instances/gfx942/attention_tiled_2d.py, lines 5054-5287).
 *
 * Implements ckc_gfx942_attn2d_emit_epilogue: the post-loop async-copy drain,
 * the read of the loop results into l_final/acc_final/rcp_l/l_nonzero, the
 * acc /= L normalization, and the three store variants:
 *   1. the transposed-x8 direct per-lane scalar global stores,
 *   2. the 32x32 Acc_lds 32-col stripe store, and
 *   3. the narrow OUT_STRIPES stripe-cooperative vec8 Acc_lds->global store.
 *
 * Faithful, byte-identical builder-call port. Reads/writes only ctx (+ the
 * builder it carries) and calls peer phase helpers via the internal header.
 */
#include "ckc/instance_gfx942_attention_tiled_2d_internal.h"
#include "ckc/helper_helper_ck_dsl.instances.gfx942.attention_tiled_2d.h" /* ckc__mfma_32x32_c_row */
#include "ckc/helper_ck_dsl.helpers.transforms.h"                         /* descriptor offset     */

/* q_desc.offset(b, token=tok, head=hd, dim=dimv) -> offset (the `_` valid is
 * discarded by the Python epilogue). Mirrors TensorDescriptor.offset(**upper).
 */
static ckc_value_t* epilogue_q_desc_offset(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                           ckc_value_t* tok,
                                           ckc_value_t* hd,
                                           ckc_value_t* dimv)
{
    ckc_ir_builder_t* b = ctx->b;
    const char* names[3] = {"token", "head", "dim"};
    ckc_value_t* vals[3];
    ckc_value_t* offset = NULL;
    ckc_value_t* valid  = NULL;
    vals[0]             = tok;
    vals[1]             = hd;
    vals[2]             = dimv;
    (void)ckc_transforms_descriptor_offset(b, ctx->q_desc, names, vals, 3, &offset, &valid);
    (void)valid;
    return offset;
}

ckc_kernel_def_t* ckc_gfx942_attn2d_emit_epilogue(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* Convenience aliases for the captured locals (1:1 with the Python names). */
    const ckc_type_t* dtype       = ctx->dtype;
    ckc_value_t* output           = ctx->output;
    ckc_value_t* lane             = ctx->lane;
    ckc_value_t* wave_row_base    = ctx->wave_row_base;
    ckc_value_t* kv_head_idx      = ctx->kv_head_idx;
    ckc_value_t* qb_start_pos     = ctx->qb_start_pos;
    ckc_value_t* cur_batch_q_len  = ctx->cur_batch_q_len;
    ckc_value_t* cu_q_start       = ctx->cu_q_start;
    ckc_value_t* zero_f           = ctx->zero_f;
    ckc_value_t* Acc_lds          = ctx->Acc_lds;

    const int SOFTMAX_STATE_SLOTS = ctx->SOFTMAX_STATE_SLOTS;
    const int ACC_N_TILES         = ctx->ACC_N_TILES;
    const int ACC_M_ATOMS         = ctx->ACC_M_ATOMS;
    const int REGS_PER_LANE       = ctx->REGS_PER_LANE;
    const int NQK                 = ctx->NQK;
    const int NUM_QH              = ctx->NUM_QH;
    const int BLOCK_M             = ctx->BLOCK_M;
    const int THREADS             = ctx->THREADS;
    const int MFMA_N              = 16; /* module-level constant MFMA_N (attention_tiled_2d.py:81) */
    const int OUT_STRIPE_COLS     = ctx->OUT_STRIPE_COLS;
    const int OUT_STRIPES         = ctx->OUT_STRIPES;
    const int PV_N_TILES          = ctx->PV_N_TILES;

    int r, n, atom, reg, chunk, stripe, n_local;

    /* The loop issues a uniform "next K" async load every iteration, including
     * the final iteration where that load is intentionally never consumed. The
     * partial wait before PV leaves that final prefetch in flight. CK Tile
     * kernels always close outstanding async-copy groups before the CTA exits;
     * do the same here so no raw global->LDS operation can outlive the kernel
     * and corrupt later launches in the same process. */
    ckc_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
    ckc_b_sync(b);

    /* final = kvloop.results (the rewritten online-softmax carry). */
    ckc_value_t* const* final = ctx->out_carry;
    const int ml_count_final  = 2 * SOFTMAX_STATE_SLOTS;

    /* l_final[r] = final[2*r + 1] */
    for (r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        ctx->l_final[r] = final[2 * r + 1];

    /* acc_final indexed by (n * ACC_M_ATOMS + atom). */
    for (n = 0; n < ACC_N_TILES; ++n)
        for (atom = 0; atom < ACC_M_ATOMS; ++atom)
            ctx->acc_final[n * ACC_M_ATOMS + atom] =
                final[ml_count_final + n * ACC_M_ATOMS + atom];

    /* Per-row reciprocal of L (computed once, reused across stripes). */
    for (r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        ctx->rcp_l[r] = ckc_b_rcp(b, ctx->l_final[r]);
    for (r = 0; r < SOFTMAX_STATE_SLOTS; ++r)
        ctx->l_nonzero[r] = ckc_b_fcmp(b, "ogt", ctx->l_final[r], zero_f);

    /* lane_col = lane % 16 ; lane_col32 = lane % 32 (recomputed; not in ctx). */
    ckc_value_t* lane_col   = ckc_b_mod(b, lane, ckc_b_const_i32(b, 16));
    ckc_value_t* lane_col32 = ckc_b_mod(b, lane, ckc_b_const_i32(b, 32));

    if (ctx->USE_MFMA_32X32)
    {
        if (ctx->TRANSPOSED_QK_32X32)
        {
            /* Per-lane direct scalar global stores. Each lane already owns 16 d
             * positions at one fixed q_row, so the 32 lanes in a half-wave
             * naturally produce 32 adjacent-token, 16-element stores per cycle. */
            ckc_value_t* q_row_t = ckc_b_add(b, wave_row_base, lane_col32);
            ckc_value_t* op_pos_t =
                ckc_b_add(b, qb_start_pos, ckc_b_div(b, q_row_t, ckc_b_const_i32(b, NQK)));
            ckc_value_t* op_qh_t = ckc_b_add(
                b,
                ckc_b_mul(b, kv_head_idx, ckc_b_const_i32(b, NQK)),
                ckc_b_mod(b, q_row_t, ckc_b_const_i32(b, NQK)));
            ckc_value_t* op_mask_t = ckc_b_land(
                b,
                ckc_b_cmp_lt(b, op_pos_t, cur_batch_q_len),
                ckc_b_cmp_lt(b, op_qh_t, ckc_b_const_i32(b, NUM_QH)));
            ckc_value_t* out_base_t = epilogue_q_desc_offset(
                ctx, ckc_b_add(b, cu_q_start, op_pos_t), op_qh_t, ckc_b_const_i32(b, 0));

            ckc_value_t* inv_l_t      = ckc_b_rcp(b, ctx->l_final[0]);
            ckc_value_t* l_nonzero_t  = ckc_b_fcmp(b, "ogt", ctx->l_final[0], zero_f);

            for (n = 0; n < ACC_N_TILES; ++n)
            {
                ckc_value_t* acc32 = ckc_gfx942_attn2d_acc_final_get(ctx, n, 0);
                for (reg = 0; reg < REGS_PER_LANE; ++reg)
                {
                    ckc_value_t* out_col_t = ckc_b_add(
                        b, ckc_b_const_i32(b, n * 32), ckc__mfma_32x32_c_row(b, lane, reg));
                    ckc_value_t* v          = ckc_b_vec_extract(b, acc32, reg);
                    ckc_value_t* normalized = ckc_b_fmul(b, v, inv_l_t);
                    ckc_value_t* final_h    = ckc_b_cast_f32_to(
                        b, ckc_b_select(b, l_nonzero_t, normalized, zero_f), dtype);
                    ckc_if_t iff = ckc_b_scf_if(b, op_mask_t);
                    ckc_b_region_enter(b, iff.then_region);
                    {
                        ckc_b_global_store(
                            b, output, ckc_b_add(b, out_base_t, out_col_t), final_h, 2);
                    }
                    ckc_b_region_leave(b);
                }
            }
            return ckc_ir_builder_kernel(b);
        }

        /* Coalesced correctness-first epilogue for the M32N32K16 accumulator
         * layout. Stage one 32-column stripe into Acc_lds, then reuse the vec8
         * cooperative global-store pattern. */
        const int OUT_VEC32                = 8;
        const int OUT_PER_THREAD_HALVES32  = (BLOCK_M * 32) / THREADS;
        /* assert OUT_PER_THREAD_HALVES32 % OUT_VEC32 == 0 */
        const int OUT_CHUNKS_PER_THREAD32  = OUT_PER_THREAD_HALVES32 / OUT_VEC32;
        const int OUT_THREADS_PER_ROW32    = 32 / (OUT_CHUNKS_PER_THREAD32 * OUT_VEC32);
        ckc_value_t* tid                   = ctx->tid;
        ckc_value_t* OUT_ROW_BASE32 =
            ckc_b_div(b, tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW32));
        ckc_value_t* OUT_col_base32 = ckc_b_mul(
            b,
            ckc_b_mod(b, tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW32)),
            ckc_b_const_i32(b, OUT_CHUNKS_PER_THREAD32 * OUT_VEC32));
        ckc_value_t* op_pos32_base = ckc_b_add(
            b, qb_start_pos, ckc_b_div(b, OUT_ROW_BASE32, ckc_b_const_i32(b, NQK)));
        ckc_value_t* op_qh32_base = ckc_b_add(
            b,
            ckc_b_mul(b, kv_head_idx, ckc_b_const_i32(b, NQK)),
            ckc_b_mod(b, OUT_ROW_BASE32, ckc_b_const_i32(b, NQK)));
        ckc_value_t* op_mask32_base = ckc_b_land(
            b,
            ckc_b_cmp_lt(b, op_pos32_base, cur_batch_q_len),
            ckc_b_cmp_lt(b, op_qh32_base, ckc_b_const_i32(b, NUM_QH)));
        ckc_value_t* out_base32_base = epilogue_q_desc_offset(
            ctx, ckc_b_add(b, cu_q_start, op_pos32_base), op_qh32_base, ckc_b_const_i32(b, 0));

        for (n = 0; n < ACC_N_TILES; ++n)
        {
            ckc_value_t* acc32 = ckc_gfx942_attn2d_acc_final_get(ctx, n, 0);
            for (reg = 0; reg < REGS_PER_LANE; ++reg)
            {
                ckc_value_t* row =
                    ckc_b_add(b, wave_row_base, ckc__mfma_32x32_c_row(b, lane, reg));
                ckc_value_t* col_in_stripe = lane_col32;
                ckc_value_t* v             = ckc_b_vec_extract(b, acc32, reg);
                ckc_value_t* normalized    = ckc_b_fmul(b, v, ctx->rcp_l[reg]);
                ckc_value_t* final_h       = ckc_b_cast_f32_to(
                    b, ckc_b_select(b, ctx->l_nonzero[reg], normalized, zero_f), dtype);
                ckc_value_t* idx[2];
                idx[0] = row;
                idx[1] = col_in_stripe;
                ckc_b_smem_store_vN(b, Acc_lds, idx, 2, final_h, 1);
            }
            ckc_b_sync(b);
            for (chunk = 0; chunk < OUT_CHUNKS_PER_THREAD32; ++chunk)
            {
                ckc_value_t* col_in_stripe =
                    ckc_b_add(b, OUT_col_base32, ckc_b_const_i32(b, chunk * OUT_VEC32));
                ckc_value_t* lidx[2];
                lidx[0]          = OUT_ROW_BASE32;
                lidx[1]          = col_in_stripe;
                ckc_value_t* v8h = ckc_b_smem_load_vN(b, Acc_lds, lidx, 2, dtype, OUT_VEC32);
                ckc_value_t* out_col =
                    ckc_b_add(b, ckc_b_const_i32(b, n * 32), col_in_stripe);
                ckc_if_t iff = ckc_b_scf_if(b, op_mask32_base);
                ckc_b_region_enter(b, iff.then_region);
                {
                    ckc_b_global_store_vN(
                        b, output, ckc_b_add(b, out_base32_base, out_col), v8h, OUT_VEC32, 16);
                }
                ckc_b_region_leave(b);
            }
            if (n + 1 < ACC_N_TILES)
                ckc_b_sync(b);
        }
        return ckc_ir_builder_kernel(b);
    }

    /* ---------------- striped epilogue ----------------
     * Loop in OUT_STRIPES stripes, each covering OUT_STRIPE_COLS = 32
     * consecutive output columns (= 2 PV N-tiles). */
    const int N_TILES_PER_STRIPE = OUT_STRIPE_COLS / MFMA_N;
    /* assert PV_N_TILES % N_TILES_PER_STRIPE == 0 */
    (void)PV_N_TILES;

    const int OUT_VEC                = 8;
    const int OUT_PER_THREAD_HALVES  = (BLOCK_M * OUT_STRIPE_COLS) / THREADS;
    /* assert OUT_PER_THREAD_HALVES % OUT_VEC == 0 and > 0 */
    const int OUT_CHUNKS_PER_THREAD  = OUT_PER_THREAD_HALVES / OUT_VEC;
    const int OUT_THREADS_PER_ROW    = OUT_STRIPE_COLS / (OUT_CHUNKS_PER_THREAD * OUT_VEC);
    /* OUT_ROWS_PER_ITER = THREADS / OUT_THREADS_PER_ROW (asserted == BLOCK_M) */

    ckc_value_t* tid = ctx->tid;
    ckc_value_t* OUT_ROW_BASE =
        ckc_b_div(b, tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW));
    ckc_value_t* OUT_col_base_in_stripe = ckc_b_mul(
        b,
        ckc_b_mod(b, tid, ckc_b_const_i32(b, OUT_THREADS_PER_ROW)),
        ckc_b_const_i32(b, OUT_CHUNKS_PER_THREAD * OUT_VEC));

    /* Compute (op_pos, op_qh, op_mask, out_base) once per CTA -- these depend
     * only on OUT_row, which is loop-invariant across stripes. */
    ckc_value_t* op_pos = ckc_b_add(
        b, qb_start_pos, ckc_b_div(b, OUT_ROW_BASE, ckc_b_const_i32(b, NQK)));
    ckc_value_t* op_qh = ckc_b_add(
        b,
        ckc_b_mul(b, kv_head_idx, ckc_b_const_i32(b, NQK)),
        ckc_b_mod(b, OUT_ROW_BASE, ckc_b_const_i32(b, NQK)));
    ckc_value_t* op_mask = ckc_b_land(
        b,
        ckc_b_cmp_lt(b, op_pos, cur_batch_q_len),
        ckc_b_cmp_lt(b, op_qh, ckc_b_const_i32(b, NUM_QH)));
    ckc_value_t* out_base = epilogue_q_desc_offset(
        ctx, ckc_b_add(b, cu_q_start, op_pos), op_qh, ckc_b_const_i32(b, 0));

    for (stripe = 0; stripe < OUT_STRIPES; ++stripe)
    {
        int n_start = stripe * N_TILES_PER_STRIPE;
        /* ---- stage 1: write this stripe's N-tiles into Acc_lds ---- */
        for (n_local = 0; n_local < N_TILES_PER_STRIPE; ++n_local)
        {
            n = n_start + n_local;
            for (reg = 0; reg < REGS_PER_LANE; ++reg)
            {
                int atom_i    = reg / 4;
                int in_atom   = reg % 4;
                ckc_value_t* row =
                    ckc_b_add(b, wave_row_base, ckc_gfx942_attn2d_in_warp_row(ctx, reg));
                /* Column within the stripe = n_local*16 + lane_col */
                ckc_value_t* col_in_stripe =
                    ckc_b_add(b, ckc_b_const_i32(b, n_local * MFMA_N), lane_col);
                ckc_value_t* v = ckc_b_vec_extract(
                    b, ckc_gfx942_attn2d_acc_final_get(ctx, n, atom_i), in_atom);
                ckc_value_t* normalized = ckc_b_fmul(b, v, ctx->rcp_l[reg]);
                ckc_value_t* final_h    = ckc_b_cast_f32_to(
                    b, ckc_b_select(b, ctx->l_nonzero[reg], normalized, zero_f), dtype);
                ckc_value_t* idx[2];
                idx[0] = row;
                idx[1] = col_in_stripe;
                ckc_b_smem_store_vN(b, Acc_lds, idx, 2, final_h, 1);
            }
        }
        ckc_b_sync(b);
        /* ---- stage 2: cooperative vec8 store(s) from Acc_lds to global ---- */
        for (chunk = 0; chunk < OUT_CHUNKS_PER_THREAD; ++chunk)
        {
            ckc_value_t* col_in_stripe =
                ckc_b_add(b, OUT_col_base_in_stripe, ckc_b_const_i32(b, chunk * OUT_VEC));
            ckc_value_t* lidx[2];
            lidx[0]          = OUT_ROW_BASE;
            lidx[1]          = col_in_stripe;
            ckc_value_t* v8h = ckc_b_smem_load_vN(b, Acc_lds, lidx, 2, dtype, OUT_VEC);
            ckc_value_t* out_col = ckc_b_add(
                b, ckc_b_const_i32(b, stripe * OUT_STRIPE_COLS), col_in_stripe);
            ckc_if_t iff = ckc_b_scf_if(b, op_mask);
            ckc_b_region_enter(b, iff.then_region);
            {
                ckc_b_global_store_vN(
                    b, output, ckc_b_add(b, out_base, out_col), v8h, OUT_VEC, 16);
            }
            ckc_b_region_leave(b);
        }
        /* ---- stage 3: sync so the next stripe can overwrite Acc_lds ---- */
        if (stripe + 1 < OUT_STRIPES)
            ckc_b_sync(b);
    }

    return ckc_ir_builder_kernel(b);
}
