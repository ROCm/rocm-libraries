/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx942_attention_tiled_2d_gfx942_attention_tiled_2d_q_load_gather.c --
 * Q STAGING bucket of the chunked C99 port of
 * ck_dsl/instances/gfx942/attention_tiled_2d.py (arch gfx942).
 *
 * Implements two phase functions of the gfx942 narrow-atom tiled-2D unified
 * attention builder:
 *
 *   ckc_gfx942_attn2d_emit_q_load   -- Q[BLOCK_M, HD] global -> LDS cooperative
 *                                      vec8 stage with padding / out-of-range
 *                                      head zero-fill (Python lines 1913-1980).
 *   ckc_gfx942_attn2d_emit_q_gather -- per-lane Q -> VGPR MFMA-A-operand gather
 *                                      for both the legacy 16x16 path and the
 *                                      32x32x16 / direct-global path; fills
 *                                      ctx->q_regs and ctx->q32_regs
 *                                      (Python lines 3426-3592, through the
 *                                      _issue_k(tile_start, 0) prefetch which is
 *                                      OWNED BY A PEER bucket -- not emitted here).
 *
 * Both functions read/write ONLY ctx (and the builder it carries) and emit the
 * IR builder calls in byte-identical Python order. Peer closures are reached
 * through the shared internal header. Every IR node is arena-owned.
 *
 * NOTE on the per-lane derived ids (lane_rg / lane_col / lane_half / lane_col32).
 *   The Python prologue materializes ``lane_rg = b.div(lane,16)`` /
 *   ``lane_col = b.mod(lane,16)`` at line 2014-2015 (the row-map bucket, a PEER
 *   file's responsibility), and the 32x32 gather RE-derives its own
 *   ``lane_half = b.div(lane,32)`` / ``lane_col32 = b.mod(lane,32)`` locally at
 *   line 3503-3504. The internal ctx exposes ``ctx->lane`` and
 *   ``ctx->wave_row_base`` but NOT the 16-group derived ids; they are not phase
 *   contract fields. To stay self-contained without touching the header, this
 *   bucket re-derives the 16-group ids from ctx->lane with the identical
 *   div/mod constants. The 32x32 ids already match Python verbatim (Python
 *   re-derives them too).
 */

#include <stdio.h>
#include <string.h>

#include "ckc/instance_gfx942_attention_tiled_2d_internal.h"
#include "ckc/ir.h"
#include "ckc/helper_ck_dsl.helpers.transforms.h"

/* Latch the first Python ValueError/NotImplementedError onto the sticky-error
 * IRBuilder, mirroring the peer buckets' error model. */
static void ckc_q_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* msg)
{
    if(b == NULL || b->status != CKC_OK)
    {
        return;
    }
    snprintf(b->err, (size_t)CKC_ERR_MSG_CAP, "%s", msg);
    b->status = st;
}

/* ============================================================ *
 *  Q -> LDS cooperative stage (Python lines 1913-1980).
 * ============================================================ */
void ckc_gfx942_attn2d_emit_q_load(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_type_t* dtype;
    int Q_VECS_PER_ROW, Q_VECS_PER_THREAD;
    ckc_value_t* z8;
    int li;

    if(ctx == NULL)
    {
        return;
    }
    b     = ctx->b;
    dtype = ctx->dtype;

    /* ---------------- Q -> LDS (cooperative vec8 chunks) ---------------- */
    /* Q_VECS_PER_ROW = HD // 8 ; Q_VECS_PER_THREAD = (BLOCK_M * Q_VECS_PER_ROW)
     * // THREADS. */
    Q_VECS_PER_ROW    = ctx->HD / 8;
    Q_VECS_PER_THREAD = (ctx->BLOCK_M * Q_VECS_PER_ROW) / ctx->THREADS;

    /* z8 = b.zero_vec(dtype, 8) is built in the constants block (line 1911);
     * recompute here for the staged-store fallback path. */
    z8 = ckc_b_zero_vec(b, dtype, 8);

    /* q_desc = TensorDescriptor.naive("Q", lengths=[1<<30, NUM_QH, HD],
     *          coord_names=("token","head","dim")). Stored on ctx for the
     *          downstream Q gather (32x32 direct-global) + the epilogue store. */
    {
        int lengths[3];
        static const char* const coord_names[3] = {"token", "head", "dim"};
        lengths[0] = 1 << 30;
        lengths[1] = ctx->NUM_QH;
        lengths[2] = ctx->HD;
        ctx->q_desc =
            ckc_tensor_descriptor_naive(b, "Q", lengths, 3, NULL, coord_names, 3);
    }

    if(ctx->Q_DIRECT_GLOBAL)
    {
        /* Direct-from-global Q path stages no Q into LDS. */
        return;
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
        q_vid = ckc_b_add(b,
                          ckc_b_mul(b, ckc_b_const_i32(b, (int64_t)li),
                                    ckc_b_const_i32(b, (int64_t)ctx->THREADS)),
                          ctx->tid);
        /* Q_row = q_vid // Q_VECS_PER_ROW */
        Q_row = ckc_b_div(b, q_vid, ckc_b_const_i32(b, (int64_t)Q_VECS_PER_ROW));
        /* Q_col = (q_vid % Q_VECS_PER_ROW) * 8. Python emits the inner mod (and
         * its const) before the outer const; sequence via a temp. */
        {
            ckc_value_t* q_mod =
                ckc_b_mod(b, q_vid, ckc_b_const_i32(b, (int64_t)Q_VECS_PER_ROW));
            Q_col = ckc_b_mul(b, q_mod, ckc_b_const_i32(b, 8));
        }
        /* q_pos_t = qb_start_pos + Q_row // NQK */
        q_pos_t = ckc_b_add(b, ctx->qb_start_pos,
                            ckc_b_div(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->NQK)));
        /* qh_t = kv_head_idx * NQK + Q_row % NQK
         * (force left-to-right arg emission: Python emits the mul before the
         * mod; C arg-eval order is unspecified, so sequence them via temps). */
        {
            ckc_value_t* qh_mul = ckc_b_mul(b, ctx->kv_head_idx,
                                            ckc_b_const_i32(b, (int64_t)ctx->NQK));
            ckc_value_t* qh_mod =
                ckc_b_mod(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->NQK));
            qh_t = ckc_b_add(b, qh_mul, qh_mod);
        }
        /* qmask_t = (q_pos_t < cur_batch_q_len) && (qh_t < NUM_QH)
         * (Python emits the q_pos cmp before the qh cmp). */
        {
            ckc_value_t* lt_pos = ckc_b_cmp_lt(b, q_pos_t, ctx->cur_batch_q_len);
            ckc_value_t* lt_qh =
                ckc_b_cmp_lt(b, qh_t, ckc_b_const_i32(b, (int64_t)ctx->NUM_QH));
            qmask_t = ckc_b_land(b, lt_pos, lt_qh);
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
            ckc_transforms_descriptor_offset(b, ctx->q_desc, in_names, in_values,
                                             3, &q_off_base, NULL);
        }

        /* v8 = b.global_load_vN(query, q_off_base + Q_col, dtype, 8, align=16) */
        v8 = ckc_b_global_load_vN(b, ctx->query, ckc_b_add(b, q_off_base, Q_col),
                                  dtype, 8, 16);

        /* Q store index: K_lds-aliased (buf,row_in_buf,col) or plain (row,col). */
        if(ctx->Q_ALIAS_K)
        {
            ckc_value_t* q_buf;
            ckc_value_t* q_row_in_buf;
            if(ctx->Q_USES_DUAL_SLOT)
            {
                q_buf = ckc_b_div(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->T));
                q_row_in_buf =
                    ckc_b_mod(b, Q_row, ckc_b_const_i32(b, (int64_t)ctx->T));
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
        store_val =
            ckc_b_vector_select(b, ckc_b_vector_splat(b, qmask_t, 8), v8, z8);
        ckc_b_smem_store_vN(b, ctx->Q_lds, q_store_idx, num_store_idx, store_val, 8);
    }
    ckc_b_sync(b);
}

/* ============================================================ *
 *  Per-lane Q -> VGPR gather (Python lines 3426-3592, excluding the trailing
 *  _issue_k(tile_start, 0) prefetch which a peer bucket owns).
 * ============================================================ */
void ckc_gfx942_attn2d_emit_q_gather(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_type_t* dtype;
    /* lane_rg / lane_col: the 16-group ids (Python line 2014-2015). Not ctx
     * fields; re-derived here from ctx->lane with identical div/mod constants. */
    ckc_value_t* lane_rg;
    ckc_value_t* lane_col;
    int atom, k;

    if(ctx == NULL)
    {
        return;
    }
    b     = ctx->b;
    dtype = ctx->dtype;

    /* Reuse the cached lane decomposition (emitted once at line 2014). */
    lane_rg  = (ctx->lane_rg_v != NULL)
                   ? ctx->lane_rg_v
                   : ckc_b_div(b, ctx->lane, ckc_b_const_i32(b, 16));
    lane_col = (ctx->lane_col_v != NULL)
                   ? ctx->lane_col_v
                   : ckc_b_mod(b, ctx->lane, ckc_b_const_i32(b, 16));

    ctx->q_regs_count   = 0;
    ctx->q32_regs_count = 0;

    /* Q_reg = [[None]*QK_K_ITERS for _ in range(M_ATOMS_PER_WARP)]
     *
     * Flattened into ctx->q_regs in (atom, k) row-major order:
     *   ctx->q_regs[atom * QK_K_ITERS + k]. */
    ctx->q_regs_count = ctx->M_ATOMS_PER_WARP * ctx->QK_K_ITERS;

    /* Legacy Q-gather A-operand width. gfx950 reads <8 x dtype> with
     * q_col_off = k*32 + lane_rg*8 (16x16x32-style A; Python gfx950 2270-2274);
     * gfx942 reads <4 x dtype> with k*16 + lane_rg*4 (Python gfx942 3449-3456). */
    const bool q_gather_wide =
        (ctx->target != NULL && ctx->target->memory.has_ds_read_tr);
    const int q_kstep = q_gather_wide ? 32 : 16;
    const int q_kwide = q_gather_wide ? 8 : 4;
    const int q_n     = q_gather_wide ? 8 : 4;

    /* if not (Q_DIRECT_GLOBAL or (USE_MFMA_32X32 and SKIP_LEGACY_QREG)): */
    if(!(ctx->Q_DIRECT_GLOBAL ||
         (ctx->USE_MFMA_32X32 && ctx->SKIP_LEGACY_QREG)))
    {
        for(atom = 0; atom < ctx->M_ATOMS_PER_WARP; ++atom)
        {
            ckc_value_t* q_row_atom;
            ckc_value_t* q_buf_atom        = NULL;
            ckc_value_t* q_row_in_buf_atom = NULL;

            /* q_row_atom = wave_row_base + (atom*16 + lane_col) */
            q_row_atom = ckc_b_add(
                b, ctx->wave_row_base,
                ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(atom * 16)), lane_col));

            if(ctx->Q_ALIAS_K)
            {
                if(ctx->Q_USES_DUAL_SLOT)
                {
                    q_buf_atom = ckc_b_div(b, q_row_atom,
                                           ckc_b_const_i32(b, (int64_t)ctx->T));
                    q_row_in_buf_atom = ckc_b_mod(
                        b, q_row_atom, ckc_b_const_i32(b, (int64_t)ctx->T));
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

                /* q_col_off = k*q_kstep + lane_rg*q_kwide. Python evaluates the
                 * left const(k*q_kstep) BEFORE the mul (left-to-right); bind it
                 * to a temp so C's arg-eval order does not allocate the mul's
                 * const ahead of it and shift the mul's %value. */
                ckc_value_t* q_col_base = ckc_b_const_i32(b, (int64_t)(k * q_kstep));
                q_col_off =
                    ckc_b_add(b, q_col_base,
                              ckc_b_mul(b, lane_rg, ckc_b_const_i32(b, q_kwide)));

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

                /* Q_reg[atom][k] = b.smem_load_vN(Q_lds, *idx, dtype, n=q_n) */
                qreg = ckc_b_smem_load_vN(b, ctx->Q_lds, idx, num_idx, dtype, q_n);
                ctx->q_regs[atom * ctx->QK_K_ITERS + k] = qreg;
            }
        }

        if(ctx->Q_ALIAS_K)
        {
            /* Drain the per-lane Q-gather LDS reads before the K[0] async write
             * to the same K_lds[0] slot. */
            ckc_b_s_waitcnt(b, -1, 0, -1);
            ckc_b_sync(b);
        }
    }

    /* ---- Per-lane Q gather for CK Tile/Triton 32x32x16 QK geometry ---- */
    if(ctx->USE_MFMA_32X32)
    {
        ckc_value_t* lane_half;
        ckc_value_t* lane_col32;
        ckc_value_t* q32_row;
        ckc_value_t* q32_buf         = NULL;
        ckc_value_t* q32_row_in_buf  = NULL;
        int Q32_FRAG;
        int Q32_HALF_STRIDE;

        if(ctx->FP8_MFMA_PV)
        {
            /* raise NotImplementedError(...) */
            ckc_q_set_err(b, CKC_ERR_NOTIMPL,
                          "32x32x16 PV needs bf16 V in LDS; disable "
                          "use_fp8_mfma_pv (it is broken / slower anyway) for "
                          "the fp8 combo");
            return;
        }

        /* Q32_reg = [None] * QK_K_ITERS -> ctx->q32_regs (count QK_K_ITERS). */
        ctx->q32_regs_count = ctx->QK_K_ITERS;

        lane_half  = ckc_b_div(b, ctx->lane, ckc_b_const_i32(b, 32));
        lane_col32 = ckc_b_mod(b, ctx->lane, ckc_b_const_i32(b, 32));
        /* Python reassigns lane_half / lane_col32 here (gfx950 3503-3504); the
         * non-transposed QK reads these fresh values. Publish so the QK/softmax
         * TU reuses the same SSA. */
        ctx->lane_half32_q32_v = lane_half;
        ctx->lane_col32_q32_v = lane_col32;
        q32_row    = ckc_b_add(b, ctx->wave_row_base, lane_col32);

        if(ctx->Q_ALIAS_K)
        {
            if(ctx->Q_USES_DUAL_SLOT)
            {
                q32_buf = ckc_b_div(b, q32_row,
                                    ckc_b_const_i32(b, (int64_t)ctx->T));
                q32_row_in_buf =
                    ckc_b_mod(b, q32_row, ckc_b_const_i32(b, (int64_t)ctx->T));
            }
            else
            {
                q32_buf        = ckc_b_const_i32(b, 0);
                q32_row_in_buf = q32_row;
            }
        }

        Q32_FRAG        = ctx->USE_MFMA_32X32X8 ? 4 : 8;
        Q32_HALF_STRIDE = ctx->USE_MFMA_32X32X8 ? 4 : 8;

        for(k = 0; k < ctx->QK_K_ITERS; ++k)
        {
            ckc_value_t* q32_col;
            ckc_value_t* q32 = NULL;

            /* q32_col = k*QK_K_STEP + lane_half*Q32_HALF_STRIDE. Python evaluates
             * the left const(k*QK_K_STEP) BEFORE the mul (left-to-right arg eval),
             * which consumes a counter value even when it folds to a literal 0 in
             * the output. Bind it to a temp first so C's arg-eval order matches
             * and does not shift the mul's %value. */
            ckc_value_t* q32_col_base =
                ckc_b_const_i32(b, (int64_t)(k * ctx->QK_K_STEP));
            q32_col = ckc_b_add(
                b, q32_col_base,
                ckc_b_mul(b, lane_half,
                          ckc_b_const_i32(b, (int64_t)Q32_HALF_STRIDE)));

            if(ctx->Q_DIRECT_GLOBAL)
            {
                ckc_value_t* q32_pos;
                ckc_value_t* q32_h;
                ckc_value_t* q32_mask;
                ckc_value_t* q32_pos_safe;
                ckc_value_t* q32_h_safe;
                ckc_value_t* q32_off = NULL;
                ckc_value_t* q32_raw;

                /* q32_pos = qb_start_pos + q32_row // NQK */
                q32_pos = ckc_b_add(
                    b, ctx->qb_start_pos,
                    ckc_b_div(b, q32_row, ckc_b_const_i32(b, (int64_t)ctx->NQK)));
                /* q32_h = kv_head_idx*NQK + q32_row % NQK */
                q32_h = ckc_b_add(
                    b,
                    ckc_b_mul(b, ctx->kv_head_idx,
                              ckc_b_const_i32(b, (int64_t)ctx->NQK)),
                    ckc_b_mod(b, q32_row, ckc_b_const_i32(b, (int64_t)ctx->NQK)));
                /* q32_mask = (q32_pos<cur_batch_q_len) && (q32_h<NUM_QH) */
                q32_mask = ckc_b_land(
                    b, ckc_b_cmp_lt(b, q32_pos, ctx->cur_batch_q_len),
                    ckc_b_cmp_lt(b, q32_h,
                                 ckc_b_const_i32(b, (int64_t)ctx->NUM_QH)));
                q32_pos_safe =
                    ckc_b_select(b, q32_mask, q32_pos, ckc_b_const_i32(b, 0));
                q32_h_safe =
                    ckc_b_select(b, q32_mask, q32_h, ckc_b_const_i32(b, 0));

                /* q32_off, _ = q_desc.offset(b, token=cu_q_start+q32_pos_safe,
                 *                            head=q32_h_safe, dim=q32_col) */
                {
                    static const char* const in_names[3] = {"token", "head",
                                                             "dim"};
                    ckc_value_t* in_values[3];
                    in_values[0] = ckc_b_add(b, ctx->cu_q_start, q32_pos_safe);
                    in_values[1] = q32_h_safe;
                    in_values[2] = q32_col;
                    ckc_transforms_descriptor_offset(b, ctx->q_desc, in_names,
                                                     in_values, 3, &q32_off,
                                                     NULL);
                }

                /* q32_raw = b.global_load_vN(query, q32_off, dtype, Q32_FRAG,
                 *                            align=8) */
                q32_raw = ckc_b_global_load_vN(b, ctx->query, q32_off, dtype,
                                               Q32_FRAG, 8);
                /* q32 = vector_select(splat(q32_mask, Q32_FRAG), q32_raw,
                 *                     zero_vec(dtype, Q32_FRAG)) */
                q32 = ckc_b_vector_select(
                    b, ckc_b_vector_splat(b, q32_mask, Q32_FRAG), q32_raw,
                    ckc_b_zero_vec(b, dtype, Q32_FRAG));
            }
            else
            {
                ckc_value_t* idx[3];
                int num_idx;
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
                q32 = ckc_b_smem_load_vN(b, ctx->Q_lds, idx, num_idx, dtype,
                                         Q32_FRAG);
            }

            if(ctx->FP8_NATIVE_QK)
            {
                /* q32 = vec_pack([cvt_f32_to_fp8(cast_to_f32(vec_extract(q32,i)))
                 *                 for i in range(8)], FP8E4M3) */
                ckc_value_t* parts[8];
                int i;
                for(i = 0; i < 8; ++i)
                {
                    parts[i] = ckc_b_cvt_f32_to_fp8(
                        b, ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, q32, i)));
                }
                q32 = ckc_b_vec_pack(b, parts, 8, ckc_fp8e4m3());
            }

            ctx->q32_regs[k] = q32;
        }

        if(ctx->Q_ALIAS_K)
        {
            /* Drain the 32x32 Q reads before the K[0] prefetch overwrites the
             * aliased LDS slabs. */
            ckc_b_s_waitcnt(b, -1, 0, -1);
            ckc_b_sync(b);
        }
    }

    /* The trailing _issue_k(tile_start, b.const_i32(0)) prefetch (line 3591)
     * belongs to the KV-load bucket and is emitted by the driver after this
     * gather, not here. */
}
