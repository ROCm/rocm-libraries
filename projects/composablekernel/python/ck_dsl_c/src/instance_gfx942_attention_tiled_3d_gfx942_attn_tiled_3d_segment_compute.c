/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx942_attention_tiled_3d_gfx942_attn_tiled_3d_segment_compute.c
 *
 * Chunked C99 port of the segment-kernel online-softmax COMPUTE region of
 * ck_dsl/instances/gfx942/attention_tiled_3d.py (gfx942 narrow-atom variant).
 *
 * SCOPE (this translation unit): the "segment compute" slice of
 *   build_unified_attention_3d_tiled (lines 234-969):
 *
 *   ckc_gfx942_attention_tiled_3d_emit_loop_init        lines 485-536, 721-724
 *       - invariant-hoist cache (hoist_row/qp_r/qh_r/row_ok/causal_lim)
 *       - sinks-conditioned m_inits / l_inits / acc_inits / cur_buf_init
 *       - first K load (_issue_k(tile_start, 0))
 *   ckc_gfx942_attention_tiled_3d_strided_v_b_operand   lines 886-896
 *       - PV B-operand reconstruction from 4 strided V_lds loads
 *   ckc_gfx942_attention_tiled_3d_emit_softmax_loop     lines 726-912
 *       - scf.for over [tile_start, tile_end): buffer swap, QK narrow
 *         16x16x16 MFMA, V/next-K prefetch, alibi/softcap/qq_bias/mask,
 *         online (m,l) update, P_lds store, PV narrow MFMA, carry yield
 *       - stashes m_final/l_final/acc_final into ctx
 *   ckc_gfx942_attention_tiled_3d_emit_epilogue         lines 914-967
 *       - guarded segm_output / segm_max / segm_expsum stores
 *
 * Everything is consumed through the internal header; peers (params/prologue/
 * descriptors/issuers/_mfma_16x16_c_row) are called via that header. The builder
 * call sequence mirrors the Python op-for-op so the emitted IR is byte-identical.
 */

#include "ckc/instance_gfx942_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Local (file-private) helpers
 * ============================================================ */

/* warp_xor_reduce_max(b, v, stages=4): the gfx942 3D kernel uses the *_max
 * sibling of the ported ckc_warp_xor_reduce_sum. The _max variant is not in the
 * shared helper header, so reproduce its byte-faithful op order here (per stage:
 * warp_shuffle_xor(1<<k) then fmax) -- identical to attention.py:303-327. */
static ckc_value_t* ckc__warp_xor_reduce_max(ckc_ir_builder_t* b, ckc_value_t* v)
{
    ckc_value_t* cur = v;
    int k;
    for (k = 0; k < 4; k++)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(b, cur, 1 << k);
        cur = ckc_b_fmax(b, cur, remote);
    }
    return cur;
}

/* _mfma_16x16_c_row peer (lines 79-88) via the internal header. */
static ckc_value_t*
ckc__c_row(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx, int reg)
{
    return ckc_gfx942_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
}

/* ============================================================ *
 * _strided_v_b_operand (lines 886-896)
 *
 * Build the <4 x dtype> PV B-operand from 4 strided V_lds loads reproducing the
 * per-lane (row, col) a 16x16x16 transpose read would deliver:
 *   B[j] = V_lds[cur_buf, k_iter*16 + j + v_k_chunk_base, v_n_col], j in 0..3
 * ============================================================ */
ckc_value_t* ckc_gfx942_attention_tiled_3d_strided_v_b_operand(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    int k_iter,
    ckc_value_t* cur_buf,
    ckc_value_t* v_n_col,
    ckc_value_t* v_k_chunk_base)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->cfg.dtype;
    ckc_value_t* bv = ckc_b_zero_vec(b, dtype, 4);
    int j;
    for (j = 0; j < 4; j++)
    {
        ckc_value_t* v_row =
            ckc_b_add(b, ckc_b_const_i32(b, k_iter * 16 + j), v_k_chunk_base);
        ckc_value_t* idx[3];
        ckc_value_t* loaded;
        ckc_value_t* elem;
        idx[0] = cur_buf;
        idx[1] = v_row;
        idx[2] = v_n_col;
        loaded = ckc_b_smem_load_vN(b, ctx->V_lds, idx, 3, dtype, 1);
        elem = ckc_b_vec_extract(b, loaded, 0);
        bv = ckc_b_vec_insert(b, bv, elem, j);
    }
    return bv;
}

/* ============================================================ *
 * emit_loop_init (lines 485-536, 721-724)
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_emit_loop_init(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const ckc_type_t* dtype = cfg->dtype;
    int reg, n, r;

    /* ---- invariant-hoist cache (lines 485-506) ---- */
    if (cfg->USE_INVARIANT_HOIST)
    {
        for (reg = 0; reg < 4; reg++)
        {
            ckc_value_t* row = ckc__c_row(ctx, reg);
            ckc_value_t* qp_r_div = ckc_b_div(b, row, ckc_b_const_i32(b, cfg->NQK));
            ckc_value_t* qp_r = ckc_b_add(b, ctx->qb_start_pos, qp_r_div);
            ckc_value_t* qh_r_mul =
                ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, cfg->NQK));
            ckc_value_t* qh_r_mod = ckc_b_mod(b, row, ckc_b_const_i32(b, cfg->NQK));
            ckc_value_t* qh_r = ckc_b_add(b, qh_r_mul, qh_r_mod);
            ckc_value_t* row_ok_a = ckc_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
            ckc_value_t* row_ok_b =
                ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, cfg->NUM_QH));
            ckc_value_t* row_ok = ckc_b_land(b, row_ok_a, row_ok_b);
            ctx->hoist_row[reg] = row;
            ctx->hoist_qp_r[reg] = qp_r;
            ctx->hoist_qh_r[reg] = qh_r;
            ctx->hoist_row_ok[reg] = row_ok;
            ctx->hoist_causal_lim[reg] = ckc_b_add(b, ctx->context_len, qp_r);
        }
    }
    else
    {
        for (reg = 0; reg < 4; reg++)
        {
            ctx->hoist_row[reg] = NULL;
            ctx->hoist_qp_r[reg] = NULL;
            ctx->hoist_qh_r[reg] = NULL;
            ctx->hoist_row_ok[reg] = NULL;
            ctx->hoist_causal_lim[reg] = NULL;
        }
    }

    /* ---- m_inits (sinks-conditioned) / l_inits (lines 508-526) ---- */
    if (cfg->USE_SINKS)
    {
        ckc_value_t* seg0 =
            ckc_b_cmp_eq(b, ctx->seg_idx, ckc_b_const_i32(b, 0));
        for (r = 0; r < 4; r++)
        {
            ckc_value_t* qh;
            ckc_value_t* qh_in;
            ckc_value_t* sink_h;
            ckc_value_t* sink_f;
            ckc_value_t* sink_with_mask;
            if (cfg->USE_INVARIANT_HOIST)
            {
                qh = ctx->hoist_qh_r[r];
            }
            else
            {
                ckc_value_t* row = ckc__c_row(ctx, r);
                ckc_value_t* qh_mul =
                    ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, cfg->NQK));
                ckc_value_t* qh_mod = ckc_b_mod(b, row, ckc_b_const_i32(b, cfg->NQK));
                qh = ckc_b_add(b, qh_mul, qh_mod);
            }
            qh_in = ckc_b_cmp_lt(b, qh, ckc_b_const_i32(b, cfg->NUM_QH));
            sink_h = ckc_b_global_load(b, ctx->sinks, qh, dtype, 2);
            sink_f = ckc_b_fmul(b, ckc_b_cast_to_f32(b, sink_h), ctx->rcp_ln2);
            sink_with_mask = ckc_b_select(b, qh_in, sink_f, ctx->neg_inf);
            ctx->m_inits[r] = ckc_b_select(b, seg0, sink_with_mask, ctx->neg_inf);
        }
    }
    else
    {
        for (r = 0; r < 4; r++)
        {
            ctx->m_inits[r] = ctx->neg_inf;
        }
    }
    for (r = 0; r < 4; r++)
    {
        ctx->l_inits[r] = ctx->one_f;
    }

    /* ---- acc_inits (lines 528-529) ---- */
    {
        ckc_value_t* acc_zero = ckc_b_zero_vec_f32(b, 4);
        for (n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_inits[n] = acc_zero;
        }
    }

    /* ---- async DMA infra + paged-KV descriptor (lines 538-602) ----
     * Emitted here (not in emit_prologue) so the SSA op order matches Python's
     * single linear build: acc_zero precedes the buffer rsrc / seq_base / desc. */
    ckc_gfx942_attention_tiled_3d_emit_async_infra(ctx);

    /* ---- first K load + cur_buf_init (lines 721-724) ---- */
    ckc_gfx942_attention_tiled_3d_issue_k(ctx, ctx->tile_start,
                                          ckc_b_const_i32(b, 0));
    ctx->cur_buf_init = ckc_b_const_i32(b, 0);
}

/* ============================================================ *
 * emit_softmax_loop (lines 726-912)
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_emit_softmax_loop(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const ckc_type_t* dtype = cfg->dtype;
    const ckc_type_t* f32 = ckc_f32();
    int r, n, k, reg;

    /* iter_args: m0,l0,m1,l1,m2,l2,m3,l3, acc0..accN-1, cur_buf */
    int num_ml = 8;
    int num_iter = num_ml + cfg->PV_N_TILES + 1;
    ckc_iter_arg_t* iter_args =
        (ckc_iter_arg_t*)ckc_arena_alloc(&b->arena,
                                         (size_t)num_iter * sizeof(*iter_args));
    if (iter_args == NULL)
    {
        return;
    }
    {
        int ai = 0;
        char buf[16];
        for (r = 0; r < 4; r++)
        {
            /* "mR" */
            buf[0] = 'm';
            buf[1] = (char)('0' + r);
            buf[2] = '\0';
            iter_args[ai].name = ckc_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->m_inits[r];
            ai++;
            buf[0] = 'l';
            iter_args[ai].name = ckc_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->l_inits[r];
            ai++;
        }
        for (n = 0; n < cfg->PV_N_TILES; n++)
        {
            /* "accN" -- N up to 15 */
            int p = 0;
            buf[p++] = 'a';
            buf[p++] = 'c';
            buf[p++] = 'c';
            if (n >= 10)
            {
                buf[p++] = (char)('0' + (n / 10));
            }
            buf[p++] = (char)('0' + (n % 10));
            buf[p] = '\0';
            iter_args[ai].name = ckc_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->acc_inits[n];
            ai++;
        }
        iter_args[ai].name = ckc_arena_strdup(&b->arena, "cur_buf");
        iter_args[ai].init = ctx->cur_buf_init;
        ai++;
    }

    ckc_for_t kvloop = ckc_b_scf_for_iter(
        b, ctx->tile_start, ctx->tile_end, ckc_b_const_i32(b, 1),
        iter_args, num_iter, "kv_tile", false, false);

    ckc_b_region_enter(b, kvloop.body);
    {
        ckc_value_t* kv_tile_iv = kvloop.iv;
        ckc_value_t** carry = kvloop.iter_vars;

        ckc_value_t* m_vals[4];
        ckc_value_t* l_vals[4];
        ckc_value_t** acc_vals;
        ckc_value_t* cur_buf;
        ckc_value_t* nxt_buf;
        ckc_value_t* tile_off;
        ckc_value_t* next_tile_iv_raw;
        ckc_value_t* in_range_next;
        ckc_value_t* safe_next_tile;

        ckc_value_t** A_kits;
        ckc_value_t** S_n;
        ckc_value_t* alibi_per_row[4];
        ckc_value_t* m_new[4];
        ckc_value_t* l_local[4];
        ckc_value_t* alpha_regs[4];
        ckc_value_t* new_l_vals[4];
        ckc_value_t** new_acc;

        /* masked[(n,reg)] and s_local[(reg,n)] -- index by n*4+reg / reg*N+n */
        ckc_value_t** masked;

        acc_vals = (ckc_value_t**)ckc_arena_alloc(
            &b->arena, (size_t)cfg->PV_N_TILES * sizeof(ckc_value_t*));
        new_acc = (ckc_value_t**)ckc_arena_alloc(
            &b->arena, (size_t)cfg->PV_N_TILES * sizeof(ckc_value_t*));
        A_kits = (ckc_value_t**)ckc_arena_alloc(
            &b->arena, (size_t)cfg->QK_K_ITERS * sizeof(ckc_value_t*));
        S_n = (ckc_value_t**)ckc_arena_alloc(
            &b->arena, (size_t)cfg->QK_N_TILES * sizeof(ckc_value_t*));
        masked = (ckc_value_t**)ckc_arena_alloc(
            &b->arena, (size_t)(cfg->QK_N_TILES * 4) * sizeof(ckc_value_t*));
        if (acc_vals == NULL || new_acc == NULL || A_kits == NULL ||
            S_n == NULL || masked == NULL)
        {
            ckc_b_region_leave(b);
            return;
        }

        for (r = 0; r < 4; r++)
        {
            m_vals[r] = carry[2 * r];
            l_vals[r] = carry[2 * r + 1];
        }
        for (n = 0; n < cfg->PV_N_TILES; n++)
        {
            acc_vals[n] = carry[8 + n];
        }
        cur_buf = carry[8 + cfg->PV_N_TILES];
        nxt_buf = ckc_b_sub(b, ckc_b_const_i32(b, 1), cur_buf);
        tile_off = ckc_b_mul(b, kv_tile_iv, ckc_b_const_i32(b, cfg->T));

        next_tile_iv_raw = ckc_b_add(b, kv_tile_iv, ckc_b_const_i32(b, 1));
        in_range_next = ckc_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
        safe_next_tile = ckc_b_select(b, in_range_next, next_tile_iv_raw, kv_tile_iv);

        ckc_b_s_waitcnt(b, 0, 0, -1);
        ckc_b_sync(b);

        /* ---------------- QK (narrow 16x16x16, K-step 16) ---------------- */
        for (k = 0; k < cfg->QK_K_ITERS; k++)
        {
            ckc_value_t* q_col_c = ckc_b_const_i32(b, k * 16);
            ckc_value_t* q_col_m = ckc_b_mul(b, ctx->lane_rg, ckc_b_const_i32(b, 4));
            ckc_value_t* q_col_off = ckc_b_add(b, q_col_c, q_col_m);
            ckc_value_t* idx[2];
            idx[0] = ctx->lane_col;
            idx[1] = q_col_off;
            A_kits[k] = ckc_b_smem_load_vN(b, ctx->Q_lds, idx, 2, dtype, 4);
        }
        for (n = 0; n < cfg->QK_N_TILES; n++)
        {
            ckc_value_t* acc_v = ckc_b_zero_vec_f32(b, 4);
            for (k = 0; k < cfg->QK_K_ITERS; k++)
            {
                ckc_value_t* kc_c = ckc_b_const_i32(b, k * 16);
                ckc_value_t* kc_m = ckc_b_mul(b, ctx->lane_rg, ckc_b_const_i32(b, 4));
                ckc_value_t* kc_off = ckc_b_add(b, kc_c, kc_m);
                ckc_value_t* k_row = ckc_b_add(
                    b, ckc_b_const_i32(b, n * 16), ctx->lane_col);
                ckc_value_t* idx[3];
                ckc_value_t* B_v;
                idx[0] = cur_buf;
                idx[1] = k_row;
                idx[2] = kc_off;
                B_v = ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, dtype, 4);
                acc_v = ckc_mfma_16x16x16_for_dtype(b, dtype, A_kits[k], B_v, acc_v);
            }
            S_n[n] = acc_v;
        }

        ckc_gfx942_attention_tiled_3d_issue_v(ctx, kv_tile_iv, cur_buf);
        ckc_gfx942_attention_tiled_3d_issue_k(ctx, safe_next_tile, nxt_buf);

        /* ---------------- alibi slopes (per-row) ---------------- */
        if (cfg->USE_ALIBI)
        {
            for (reg = 0; reg < 4; reg++)
            {
                ckc_value_t* row = ckc__c_row(ctx, reg);
                ckc_value_t* qh_r_mul =
                    ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, cfg->NQK));
                ckc_value_t* qh_r_mod = ckc_b_mod(b, row, ckc_b_const_i32(b, cfg->NQK));
                ckc_value_t* qh_r = ckc_b_add(b, qh_r_mul, qh_r_mod);
                ckc_value_t* qh_ok =
                    ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, cfg->NUM_QH));
                alibi_per_row[reg] = ckc_b_masked_global_load(
                    b, ctx->alibi_slopes_ptr, qh_r, qh_ok,
                    ckc_b_const_f32(b, 0.0), f32, 4);
            }
        }

        /* ---------------- masked scores ---------------- */
        for (reg = 0; reg < 4; reg++)
        {
            ckc_value_t* qp_r;
            ckc_value_t* row_ok;
            ckc_value_t* causal_lim_hoist = NULL;
            if (cfg->USE_INVARIANT_HOIST)
            {
                qp_r = ctx->hoist_qp_r[reg];
                row_ok = ctx->hoist_row_ok[reg];
                causal_lim_hoist = ctx->hoist_causal_lim[reg];
            }
            else
            {
                ckc_value_t* row = ckc__c_row(ctx, reg);
                ckc_value_t* qp_r_div =
                    ckc_b_div(b, row, ckc_b_const_i32(b, cfg->NQK));
                qp_r = ckc_b_add(b, ctx->qb_start_pos, qp_r_div);
                {
                    ckc_value_t* qh_r_mul = ckc_b_mul(
                        b, ctx->kv_head_idx, ckc_b_const_i32(b, cfg->NQK));
                    ckc_value_t* qh_r_mod =
                        ckc_b_mod(b, row, ckc_b_const_i32(b, cfg->NQK));
                    ckc_value_t* qh_r = ckc_b_add(b, qh_r_mul, qh_r_mod);
                    ckc_value_t* row_ok_a =
                        ckc_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
                    ckc_value_t* row_ok_b =
                        ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, cfg->NUM_QH));
                    row_ok = ckc_b_land(b, row_ok_a, row_ok_b);
                }
            }
            for (n = 0; n < cfg->QK_N_TILES; n++)
            {
                ckc_value_t* col_abs = ckc_b_add(
                    b,
                    ckc_b_add(b, tile_off,
                              ckc_b_mul(b, ckc_b_const_i32(b, n),
                                        ckc_b_const_i32(b, 16))),
                    ctx->lane_col);
                ckc_value_t* causal_lim;
                ckc_value_t* causal_ok;
                ckc_value_t* in_prefix;
                ckc_value_t* m_ok;
                ckc_value_t* s_raw;
                ckc_value_t* s_scaled;
                if (cfg->USE_INVARIANT_HOIST)
                {
                    causal_lim = causal_lim_hoist;
                }
                else
                {
                    causal_lim = ckc_b_add(b, ctx->context_len, qp_r);
                }
                causal_ok = ckc_b_cmp_le(b, col_abs, causal_lim);
                in_prefix = ckc_b_cmp_lt(b, col_abs, ctx->max_seq_prefix_len);
                m_ok = ckc_b_land(b, ckc_b_land(b, row_ok, causal_ok), in_prefix);
                if (cfg->SLIDING_WINDOW > 0)
                {
                    ckc_value_t* dist = ckc_b_sub(b, causal_lim, col_abs);
                    m_ok = ckc_b_land(b, m_ok,
                                      ckc_b_cmp_lt(b, dist, ctx->sw_const));
                }
                s_raw = ckc_b_vec_extract(b, S_n[n], reg);
                s_scaled = ckc_b_fmul(b, s_raw, ctx->qk_scale);
                if (cfg->USE_SOFTCAP)
                {
                    s_scaled = ckc_b_fmul(
                        b,
                        ckc_apply_softcap_log2(b, s_scaled, ctx->softcap_p),
                        ctx->rcp_ln2);
                }
                if (cfg->USE_ALIBI)
                {
                    ckc_value_t* pos_off = ckc_b_sub(b, col_abs, ctx->context_len);
                    ckc_value_t* pos_f = ckc_b_sitofp_f32(b, pos_off);
                    ckc_value_t* add_term = ckc_b_fmul(
                        b, ckc_b_fmul(b, alibi_per_row[reg], pos_f), ctx->rcp_ln2);
                    s_scaled = ckc_b_fadd(b, s_scaled, add_term);
                }
                if (cfg->USE_QQ_BIAS)
                {
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
                        b, ctx->qq_bias_ptr, qq_idx, qq_ok,
                        ckc_b_const_f32(b, 0.0), f32, 4);
                    s_scaled = ckc_b_fadd(
                        b, s_scaled, ckc_b_fmul(b, qq_v, ctx->rcp_ln2));
                }
                masked[n * 4 + reg] = ckc_b_select(b, m_ok, s_scaled, ctx->neg_inf);
            }
        }

        /* ---------------- online (m,l) update ---------------- */
        {
            /* s_local[(reg,n)] reuses `masked` values; keep a parallel store. */
            ckc_value_t** s_local = (ckc_value_t**)ckc_arena_alloc(
                &b->arena,
                (size_t)(4 * cfg->QK_N_TILES) * sizeof(ckc_value_t*));
            if (s_local == NULL)
            {
                ckc_b_region_leave(b);
                return;
            }
            for (reg = 0; reg < 4; reg++)
            {
                ckc_value_t* local_max = ctx->neg_inf;
                ckc_value_t* full_max_raw;
                ckc_value_t* ok;
                for (n = 0; n < cfg->QK_N_TILES; n++)
                {
                    ckc_value_t* v = masked[n * 4 + reg];
                    s_local[reg * cfg->QK_N_TILES + n] = v;
                    local_max = ckc_b_fmax(b, local_max, v);
                }
                full_max_raw = ckc__warp_xor_reduce_max(b, local_max);
                ok = ckc_b_fcmp(b, "ogt", full_max_raw, ctx->neg_inf);
                m_new[reg] = ckc_b_select(b, ok, full_max_raw, ctx->zero_f);
            }

            for (reg = 0; reg < 4; reg++)
            {
                ckc_value_t* row = ckc__c_row(ctx, reg);
                ckc_value_t* sum_p = ctx->zero_f;
                for (n = 0; n < cfg->QK_N_TILES; n++)
                {
                    ckc_value_t* p = ckc_b_exp2(
                        b, ckc_b_fsub(b, s_local[reg * cfg->QK_N_TILES + n],
                                      m_new[reg]));
                    ckc_value_t* col = ckc_b_add(
                        b, ckc_b_mul(b, ckc_b_const_i32(b, n),
                                     ckc_b_const_i32(b, 16)),
                        ctx->lane_col);
                    ckc_value_t* idx[2];
                    idx[0] = row;
                    idx[1] = col;
                    ckc_b_smem_store_vN(b, ctx->P_lds, idx, 2,
                                        ckc_b_cast_f32_to(b, p, dtype), 1);
                    sum_p = ckc_b_fadd(b, sum_p, p);
                }
                l_local[reg] = ckc_warp_xor_reduce_sum(b, sum_p, 4);
            }
        }

        for (r = 0; r < 4; r++)
        {
            alpha_regs[r] = ckc_b_exp2(b, ckc_b_fsub(b, m_vals[r], m_new[r]));
        }
        for (r = 0; r < 4; r++)
        {
            new_l_vals[r] = ckc_b_fadd(
                b, ckc_b_fmul(b, l_vals[r], alpha_regs[r]), l_local[r]);
        }

        if (cfg->KV_FP8)
        {
            ckc_b_s_waitcnt(b, 0, 0, -1);
            ckc_b_sync(b);
        }
        else
        {
            ckc_b_s_waitcnt(b, cfg->kv_calls_per_tile, cfg->kv_calls_per_tile, -1);
            ckc_b_sync(b);
        }

        /* ---------------- PV (narrow 16x16x16, strided-V B) ---------------- */
        for (n = 0; n < cfg->PV_N_TILES; n++)
        {
            ckc_value_t* scaled_comps[4];
            ckc_value_t* acc_v;
            ckc_value_t* v_n_col;
            ckc_value_t* v_k_chunk_base;
            for (reg = 0; reg < 4; reg++)
            {
                ckc_value_t* e = ckc_b_vec_extract(b, acc_vals[n], reg);
                scaled_comps[reg] = ckc_b_fmul(b, e, alpha_regs[reg]);
            }
            acc_v = ckc_b_vec_pack(b, scaled_comps, 4, f32);

            {
                ckc_value_t* v_n_mul =
                    ckc_b_mul(b, ckc_b_const_i32(b, n), ckc_b_const_i32(b, 16));
                v_n_col = ckc_b_add(b, v_n_mul, ctx->lane_col);
            }
            v_k_chunk_base = ckc_b_mul(b, ctx->lane_rg, ckc_b_const_i32(b, 4));

            for (k = 0; k < cfg->PV_K_ITERS; k++)
            {
                ckc_value_t* p_off_c = ckc_b_const_i32(b, k * 16);
                ckc_value_t* p_off_m =
                    ckc_b_mul(b, ctx->lane_rg, ckc_b_const_i32(b, 4));
                ckc_value_t* p_off = ckc_b_add(b, p_off_c, p_off_m);
                ckc_value_t* idx[2];
                ckc_value_t* A_p;
                ckc_value_t* B_v;
                idx[0] = ctx->lane_col;
                idx[1] = p_off;
                A_p = ckc_b_smem_load_vN(b, ctx->P_lds, idx, 2, dtype, 4);
                B_v = ckc_gfx942_attention_tiled_3d_strided_v_b_operand(
                    ctx, k, cur_buf, v_n_col, v_k_chunk_base);
                acc_v = ckc_mfma_16x16x16_for_dtype(b, dtype, A_p, B_v, acc_v);
            }
            new_acc[n] = acc_v;
        }

        /* ---------------- carry yield ---------------- */
        {
            ckc_value_t** yields = (ckc_value_t**)ckc_arena_alloc(
                &b->arena, (size_t)num_iter * sizeof(ckc_value_t*));
            int yi = 0;
            if (yields == NULL)
            {
                ckc_b_region_leave(b);
                return;
            }
            for (r = 0; r < 4; r++)
            {
                yields[yi++] = m_new[r];
                yields[yi++] = new_l_vals[r];
            }
            for (n = 0; n < cfg->PV_N_TILES; n++)
            {
                yields[yi++] = new_acc[n];
            }
            yields[yi++] = nxt_buf;
            ckc_b_scf_yield(b, yields, yi);
        }
    }
    ckc_b_region_leave(b);

    /* ---------------- stash final (m,l,acc) into ctx ---------------- */
    {
        ckc_value_t** final = kvloop.op->results;
        for (r = 0; r < 4; r++)
        {
            ctx->m_final[r] = final[2 * r];
            ctx->l_final[r] = final[2 * r + 1];
        }
        for (n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_final[n] = final[8 + n];
        }
    }
}

/* ============================================================ *
 * emit_epilogue (lines 914-967)
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_emit_epilogue(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    int n, reg;
    ckc_value_t* lane_writes_ml;

    /* ---- segm_output stores (lines 920-947) ---- */
    for (n = 0; n < cfg->PV_N_TILES; n++)
    {
        for (reg = 0; reg < 4; reg++)
        {
            ckc_value_t* row;
            ckc_value_t* qp_r;
            ckc_value_t* qh_r;
            ckc_value_t* row_ok;
            ckc_value_t* col;
            ckc_value_t* qtoken;
            ckc_value_t* seg_acc_idx = NULL;
            ckc_value_t* v_acc;
            ckc_if_t guard;
            if (cfg->USE_INVARIANT_HOIST)
            {
                row = ctx->hoist_row[reg];
                qp_r = ctx->hoist_qp_r[reg];
                qh_r = ctx->hoist_qh_r[reg];
                row_ok = ctx->hoist_row_ok[reg];
            }
            else
            {
                ckc_value_t* qp_r_div;
                ckc_value_t* qh_r_mul;
                ckc_value_t* qh_r_mod;
                ckc_value_t* row_ok_a;
                ckc_value_t* row_ok_b;
                row = ckc__c_row(ctx, reg);
                qp_r_div = ckc_b_div(b, row, ckc_b_const_i32(b, cfg->NQK));
                qp_r = ckc_b_add(b, ctx->qb_start_pos, qp_r_div);
                qh_r_mul = ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, cfg->NQK));
                qh_r_mod = ckc_b_mod(b, row, ckc_b_const_i32(b, cfg->NQK));
                qh_r = ckc_b_add(b, qh_r_mul, qh_r_mod);
                row_ok_a = ckc_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
                row_ok_b = ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, cfg->NUM_QH));
                row_ok = ckc_b_land(b, row_ok_a, row_ok_b);
            }
            {
                ckc_value_t* col_mul =
                    ckc_b_mul(b, ckc_b_const_i32(b, n), ckc_b_const_i32(b, 16));
                col = ckc_b_add(b, col_mul, ctx->lane_col);
            }
            qtoken = ckc_b_add(b, ctx->cu_q_start, qp_r);
            {
                const char* names[4];
                ckc_value_t* vals[4];
                names[0] = "token";
                vals[0] = qtoken;
                names[1] = "head";
                vals[1] = qh_r;
                names[2] = "seg";
                vals[2] = ctx->seg_idx;
                names[3] = "dim";
                vals[3] = col;
                ckc_transforms_descriptor_offset(b, ctx->seg_acc_desc, names,
                                                 vals, 4, &seg_acc_idx, NULL);
            }
            v_acc = ckc_b_vec_extract(b, ctx->acc_final[n], reg);
            guard = ckc_b_scf_if(b, row_ok);
            ckc_b_region_enter(b, guard.then_region);
            ckc_b_global_store(b, ctx->segm_output_ptr, seg_acc_idx, v_acc, 4);
            ckc_b_region_leave(b);
        }
    }

    /* ---- segm_max / segm_expsum stores (lines 949-967) ---- */
    {
        ckc_value_t* lwm_mod = ckc_b_mod(b, ctx->tid, ckc_b_const_i32(b, 16));
        lane_writes_ml = ckc_b_cmp_eq(b, lwm_mod, ckc_b_const_i32(b, 0));
    }
    for (reg = 0; reg < 4; reg++)
    {
        ckc_value_t* qp_r;
        ckc_value_t* qh_r;
        ckc_value_t* row_ok;
        ckc_value_t* qtoken;
        ckc_value_t* ml_idx = NULL;
        ckc_value_t* do_write;
        ckc_if_t guard;
        if (cfg->USE_INVARIANT_HOIST)
        {
            qp_r = ctx->hoist_qp_r[reg];
            qh_r = ctx->hoist_qh_r[reg];
            row_ok = ctx->hoist_row_ok[reg];
        }
        else
        {
            ckc_value_t* row = ckc__c_row(ctx, reg);
            ckc_value_t* qp_r_div = ckc_b_div(b, row, ckc_b_const_i32(b, cfg->NQK));
            ckc_value_t* qh_r_mul;
            ckc_value_t* qh_r_mod;
            ckc_value_t* row_ok_a;
            ckc_value_t* row_ok_b;
            qp_r = ckc_b_add(b, ctx->qb_start_pos, qp_r_div);
            qh_r_mul = ckc_b_mul(b, ctx->kv_head_idx, ckc_b_const_i32(b, cfg->NQK));
            qh_r_mod = ckc_b_mod(b, row, ckc_b_const_i32(b, cfg->NQK));
            qh_r = ckc_b_add(b, qh_r_mul, qh_r_mod);
            row_ok_a = ckc_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
            row_ok_b = ckc_b_cmp_lt(b, qh_r, ckc_b_const_i32(b, cfg->NUM_QH));
            row_ok = ckc_b_land(b, row_ok_a, row_ok_b);
        }
        qtoken = ckc_b_add(b, ctx->cu_q_start, qp_r);
        {
            const char* names[3];
            ckc_value_t* vals[3];
            names[0] = "token";
            vals[0] = qtoken;
            names[1] = "head";
            vals[1] = qh_r;
            names[2] = "seg";
            vals[2] = ctx->seg_idx;
            ckc_transforms_descriptor_offset(b, ctx->ml_desc, names, vals, 3,
                                             &ml_idx, NULL);
        }
        do_write = ckc_b_land(b, lane_writes_ml, row_ok);
        guard = ckc_b_scf_if(b, do_write);
        ckc_b_region_enter(b, guard.then_region);
        ckc_b_global_store(b, ctx->segm_max_ptr, ml_idx, ctx->m_final[reg], 4);
        ckc_b_global_store(b, ctx->segm_expsum_ptr, ml_idx, ctx->l_final[reg], 4);
        ckc_b_region_leave(b);
    }
}
