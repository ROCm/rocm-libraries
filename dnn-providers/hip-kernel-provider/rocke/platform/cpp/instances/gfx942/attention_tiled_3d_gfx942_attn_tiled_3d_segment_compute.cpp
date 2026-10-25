// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_3d_gfx942_attn_tiled_3d_segment_compute.c
 *
 * Chunked C99 port of the segment-kernel online-softmax COMPUTE region of
 * rocke/instances/gfx942/attention_tiled_3d.py (gfx942 narrow-atom variant).
 *
 * SCOPE (this translation unit): the "segment compute" slice of
 * build_unified_attention_3d_tiled:
 *
 *   rocke_gfx942_attention_tiled_3d_emit_loop_init
 *       - sinks-conditioned per-lane m_init / acc_inits / cur_buf_init
 *       - first K load (_issue_k(tile_start, 0))
 *   rocke_gfx942_attention_tiled_3d_strided_v_b_operand
 *       - the V^T operand of the PV MFMA from 4 strided V_lds loads
 *   rocke_gfx942_attention_tiled_3d_emit_softmax_loop
 *       - the lane's query-row invariants and its Q^T operand (read once)
 *       - scf.for over [tile_start, tile_end): buffer swap, QK^T
 *         (S^T = K Q^T, 16x16x16), V/next-K prefetch, alibi/softcap/qq_bias/
 *         mask, per-lane online (m,l) update, PV (O^T = V^T P^T, P^T from
 *         registers), carry yield
 *       - stashes m_final/l_final/acc_final into ctx
 *   rocke_gfx942_attention_tiled_3d_emit_epilogue
 *       - guarded segm_output (O^T rows as vectors) / segm_max / segm_expsum
 *
 * Everything is consumed through the internal header; peers (params/prologue/
 * descriptors/issuers) are called via that header. The builder call sequence
 * mirrors the Python op-for-op so the emitted IR is byte-identical.
 */

#include "rocke/instance_gfx942_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Local (file-private) helpers
 * ============================================================ */

/* _row_reduce(x, op): combine x over the 4 lanes {l, l^16, l^32, l^48} that hold
 * the 4 token groups of one query row: ds_swizzle (xor 16), then ds_bpermute
 * (xor 32; gfx942 has no permlane32_swap). op is fmax when is_max, else fadd. */
static rocke_value_t* rocke__row_reduce(rocke_ir_builder_t* b, rocke_value_t* x, bool is_max)
{
    rocke_value_t* sw = rocke_b_warp_shuffle_xor(b, x, 16);
    rocke_value_t* bp;
    x = is_max ? rocke_b_fmax(b, x, sw) : rocke_b_fadd(b, x, sw);
    bp = rocke_b_warp_shuffle_xor(b, x, 32);
    return is_max ? rocke_b_fmax(b, x, bp) : rocke_b_fadd(b, x, bp);
}

/* ============================================================ *
 * _strided_v_b_operand
 *
 * Build the <4 x dtype> V^T operand of the PV MFMA (O^T = V^T P^T) from 4
 * strided V_lds loads (gfx942 has no transpose read):
 *   A[j] = V_lds[v_buf, k_iter*16 + j + v_k_chunk_base, v_n_col], j in 0..3
 * ============================================================ */
rocke_value_t* rocke_gfx942_attention_tiled_3d_strided_v_b_operand(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    int k_iter,
    rocke_value_t* v_buf,
    rocke_value_t* v_n_col,
    rocke_value_t* v_k_chunk_base)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->cfg.dtype;
    rocke_value_t* bv = rocke_b_zero_vec(b, dtype, 4);
    int j;
    for(j = 0; j < 4; j++)
    {
        rocke_value_t* v_row
            = rocke_b_add(b, rocke_b_const_i32(b, k_iter * 16 + j), v_k_chunk_base);
        rocke_value_t* idx[3];
        rocke_value_t* loaded;
        rocke_value_t* elem;
        idx[0] = v_buf;
        idx[1] = v_row;
        idx[2] = rocke_gfx942_attention_tiled_3d_kv_col(ctx, v_row, v_n_col);
        loaded = rocke_b_smem_load_vN(b, ctx->V_lds, idx, 3, dtype, 1);
        elem = rocke_b_vec_extract(b, loaded, 0);
        bv = rocke_b_vec_insert(b, bv, elem, j);
    }
    return bv;
}

/* ============================================================ *
 * emit_loop_init
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_emit_loop_init(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const rocke_type_t* dtype = cfg->dtype;
    int n;

    /* ---- m_init (sinks-conditioned): the lane's query row is lane_col ---- */
    if(cfg->USE_SINKS)
    {
        /* Triton's 3D applies sinks only when segm_idx == 0. */
        rocke_value_t* seg0 = rocke_b_cmp_eq(b, ctx->seg_idx, rocke_b_const_i32(b, 0));
        rocke_value_t* qh_mul = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, cfg->NQK));
        rocke_value_t* qh_mod = rocke_b_mod(b, ctx->lane_col, rocke_b_const_i32(b, cfg->NQK));
        rocke_value_t* qh = rocke_b_add(b, qh_mul, qh_mod);
        rocke_value_t* qh_in = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, cfg->NUM_QH));
        rocke_value_t* sink_h = rocke_b_global_load(b, ctx->sinks, qh, dtype, 2);
        rocke_value_t* sink_f = rocke_b_fmul(b, rocke_b_cast_to_f32(b, sink_h), ctx->rcp_ln2);
        rocke_value_t* sink_with_mask = rocke_b_select(b, qh_in, sink_f, ctx->neg_inf);
        ctx->m_init = rocke_b_select(b, seg0, sink_with_mask, ctx->neg_inf);
    }
    else
    {
        ctx->m_init = ctx->neg_inf;
    }

    /* ---- acc_inits ---- */
    {
        rocke_value_t* acc_zero = rocke_b_zero_vec_f32(b, 4);
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_inits[n] = acc_zero;
        }
    }

    /* ---- async DMA infra + paged-KV descriptor ----
     * Emitted here (not in emit_prologue) so the SSA op order matches Python's
     * single linear build: acc_zero precedes the buffer rsrc / seq_base / desc. */
    rocke_gfx942_attention_tiled_3d_emit_async_infra(ctx);

    /* ---- the first page-id window (WAVE_PAGE_ID_LOAD) ----
     * The wave holds a 64-page window of the segment's page table, one id per
     * lane. A tile lies within one page, and it starts below seq_len, so its
     * page is always one of the sequence's. */
    ctx->first_page_id = NULL;
    if(cfg->WAVE_PAGE_ID_LOAD)
    {
        rocke_value_t* rounded = rocke_b_add(b, ctx->seq_len, rocke_b_const_i32(b, cfg->BS - 1));
        ctx->seq_pages = rocke_b_div(b, rounded, rocke_b_const_i32(b, cfg->BS));
        ctx->seg_first_page = rocke_gfx942_attention_tiled_3d_tile_page(ctx, ctx->tile_start);
        ctx->first_page_id_window
            = rocke_gfx942_attention_tiled_3d_load_page_id_window(ctx, rocke_b_const_i32(b, 0));
        ctx->first_page_id = rocke_gfx942_attention_tiled_3d_window_tile_page_id(
            ctx, ctx->first_page_id_window, ctx->tile_start);
    }

    /* ---- first K load + cur_buf_init ---- */
    rocke_gfx942_attention_tiled_3d_issue_k(
        ctx, ctx->tile_start, rocke_b_const_i32(b, 0), ctx->first_page_id);
    ctx->cur_buf_init = rocke_b_const_i32(b, 0);
}

/* ============================================================ *
 * emit_softmax_loop
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_emit_softmax_loop(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const rocke_type_t* dtype = cfg->dtype;
    const rocke_type_t* f32 = rocke_f32();
    int n, k, reg;

    /* iter_args: m, l, acc0..accN-1, cur_buf (+ pid, pid_window under
     * WAVE_PAGE_ID_LOAD) */
    int num_iter = 2 + cfg->PV_N_TILES + 1 + (cfg->WAVE_PAGE_ID_LOAD ? 2 : 0);
    rocke_iter_arg_t* iter_args
        = (rocke_iter_arg_t*)rocke_arena_alloc(&b->arena, (size_t)num_iter * sizeof(*iter_args));
    if(iter_args == NULL)
    {
        return;
    }
    {
        int ai = 0;
        char buf[16];
        iter_args[ai].name = rocke_arena_strdup(&b->arena, "m");
        iter_args[ai].init = ctx->m_init;
        ai++;
        iter_args[ai].name = rocke_arena_strdup(&b->arena, "l");
        iter_args[ai].init = ctx->one_f;
        ai++;
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            /* "accN" -- N up to 15 */
            int p = 0;
            buf[p++] = 'a';
            buf[p++] = 'c';
            buf[p++] = 'c';
            if(n >= 10)
            {
                buf[p++] = (char)('0' + (n / 10));
            }
            buf[p++] = (char)('0' + (n % 10));
            buf[p] = '\0';
            iter_args[ai].name = rocke_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->acc_inits[n];
            ai++;
        }
        iter_args[ai].name = rocke_arena_strdup(&b->arena, "cur_buf");
        iter_args[ai].init = ctx->cur_buf_init;
        ai++;
        if(cfg->WAVE_PAGE_ID_LOAD)
        {
            iter_args[ai].name = rocke_arena_strdup(&b->arena, "pid");
            iter_args[ai].init = ctx->first_page_id;
            ai++;
            iter_args[ai].name = rocke_arena_strdup(&b->arena, "pid_window");
            iter_args[ai].init = ctx->first_page_id_window;
            ai++;
        }
    }

    /* LICM hoist: the lane's query-row invariants, constant across all KV tiles.
     * qp, qh, row_ok, and causal_lim depend only on CTA-level constants and the
     * lane's query row (lane_col); the ALiBi slope is per query head. */
    {
        rocke_value_t* qp_div = rocke_b_div(b, ctx->lane_col, rocke_b_const_i32(b, cfg->NQK));
        rocke_value_t* qh_mul;
        rocke_value_t* qh_mod;
        rocke_value_t* ok_a;
        rocke_value_t* ok_b;
        ctx->row_qp = rocke_b_add(b, ctx->qb_start_pos, qp_div);
        qh_mul = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, cfg->NQK));
        qh_mod = rocke_b_mod(b, ctx->lane_col, rocke_b_const_i32(b, cfg->NQK));
        ctx->row_qh = rocke_b_add(b, qh_mul, qh_mod);
        ok_a = rocke_b_cmp_lt(b, ctx->row_qp, ctx->cur_batch_q_len);
        ok_b = rocke_b_cmp_lt(b, ctx->row_qh, rocke_b_const_i32(b, cfg->NUM_QH));
        ctx->row_ok = rocke_b_land(b, ok_a, ok_b);
        ctx->row_causal_lim = rocke_b_add(b, ctx->context_len, ctx->row_qp);
        ctx->row_alibi = NULL;
        if(cfg->USE_ALIBI)
        {
            rocke_value_t* qh_ok
                = rocke_b_cmp_lt(b, ctx->row_qh, rocke_b_const_i32(b, cfg->NUM_QH));
            ctx->row_alibi = rocke_b_masked_global_load(
                b, ctx->alibi_slopes_ptr, ctx->row_qh, qh_ok, rocke_b_const_f32(b, 0.0), f32, 4);
        }
    }

    /* The QK B operand (Q^T) is read from LDS once; every tile's MFMAs take it
     * from registers. */
    rocke_value_t** Q_kits = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, (size_t)cfg->QK_K_ITERS * sizeof(rocke_value_t*));
    if(Q_kits == NULL)
    {
        return;
    }
    for(k = 0; k < cfg->QK_K_ITERS; k++)
    {
        rocke_value_t* q_col_c = rocke_b_const_i32(b, k * 16);
        rocke_value_t* q_col_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
        rocke_value_t* q_col_off = rocke_b_add(b, q_col_c, q_col_m);
        rocke_value_t* idx[2];
        idx[0] = ctx->lane_col;
        idx[1] = q_col_off;
        Q_kits[k] = rocke_b_smem_load_vN(b, ctx->Q_lds, idx, 2, dtype, 4);
    }

    rocke_for_t kvloop = rocke_b_scf_for_iter(b,
                                              ctx->tile_start,
                                              ctx->tile_end,
                                              rocke_b_const_i32(b, 1),
                                              iter_args,
                                              num_iter,
                                              "kv_tile",
                                              false,
                                              true);

    rocke_b_region_enter(b, kvloop.body);
    {
        rocke_value_t* kv_tile_iv = kvloop.iv;
        rocke_value_t** carry = kvloop.iter_vars;

        rocke_value_t* m_val;
        rocke_value_t* l_val;
        rocke_value_t** acc_vals;
        rocke_value_t* cur_buf;
        rocke_value_t* nxt_buf;
        rocke_value_t* v_buf;
        rocke_value_t* tile_off;
        rocke_value_t* next_tile_iv_raw;
        rocke_value_t* in_range_next;
        rocke_value_t* safe_next_tile;
        rocke_value_t* cur_page_id = NULL;
        rocke_value_t* page_id_window = NULL;
        rocke_value_t* nxt_page_id = NULL;
        rocke_value_t* nxt_page_id_window = NULL;
        rocke_value_t** S_n;
        rocke_value_t* m_new;
        rocke_value_t* l_local;
        rocke_value_t* alpha;
        rocke_value_t* new_l_val;
        rocke_value_t** new_acc;
        /* masked / p_regs [(n, reg)] indexed n*4+reg; filled reg-major (the
         * Python dict insertion order), which is also the reduction order. */
        rocke_value_t** masked;
        rocke_value_t** p_regs;

        acc_vals = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)cfg->PV_N_TILES * sizeof(rocke_value_t*));
        new_acc = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)cfg->PV_N_TILES * sizeof(rocke_value_t*));
        S_n = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                                 (size_t)cfg->QK_N_TILES * sizeof(rocke_value_t*));
        masked = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)(cfg->QK_N_TILES * 4) * sizeof(rocke_value_t*));
        p_regs = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)(cfg->QK_N_TILES * 4) * sizeof(rocke_value_t*));
        if(acc_vals == NULL || new_acc == NULL || S_n == NULL || masked == NULL || p_regs == NULL)
        {
            rocke_b_region_leave(b);
            return;
        }

        m_val = carry[0];
        l_val = carry[1];
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            acc_vals[n] = carry[2 + n];
        }
        cur_buf = carry[2 + cfg->PV_N_TILES];
        if(cfg->WAVE_PAGE_ID_LOAD)
        {
            cur_page_id = carry[3 + cfg->PV_N_TILES];
        }
        nxt_buf = rocke_b_sub(b, rocke_b_const_i32(b, 1), cur_buf);
        v_buf = rocke_b_const_i32(b, 0);
        tile_off = rocke_b_mul(b, kv_tile_iv, rocke_b_const_i32(b, cfg->T));

        next_tile_iv_raw = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 1));
        in_range_next = rocke_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
        safe_next_tile = rocke_b_select(b, in_range_next, next_tile_iv_raw, kv_tile_iv);

        rocke_b_s_waitcnt(b, 0, 0, -1);
        rocke_b_sync(b);
        /* Page id of the next tile, for its K copies below. */
        if(cfg->WAVE_PAGE_ID_LOAD)
        {
            /* Read from the window in registers: no load in the loop. */
            page_id_window = carry[4 + cfg->PV_N_TILES];
            nxt_page_id = rocke_gfx942_attention_tiled_3d_window_tile_page_id(
                ctx, page_id_window, safe_next_tile);
        }

        /* ---------------- QK: S^T = K Q^T (16x16x16, K-step 16) ---------------- */
        for(n = 0; n < cfg->QK_N_TILES; n++)
        {
            rocke_value_t* acc_v = rocke_b_zero_vec_f32(b, 4);
            for(k = 0; k < cfg->QK_K_ITERS; k++)
            {
                rocke_value_t* kc_c = rocke_b_const_i32(b, k * 16);
                rocke_value_t* kc_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
                rocke_value_t* kc_off = rocke_b_add(b, kc_c, kc_m);
                rocke_value_t* k_row = rocke_b_add(b, rocke_b_const_i32(b, n * 16), ctx->lane_col);
                rocke_value_t* idx[3];
                rocke_value_t* K_v;
                idx[0] = cur_buf;
                idx[1] = k_row;
                idx[2] = rocke_gfx942_attention_tiled_3d_kv_col(ctx, k_row, kc_off);
                K_v = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, dtype, 4);
                acc_v = rocke_mfma_16x16x16_for_dtype(b, dtype, K_v, Q_kits[k], acc_v);
            }
            S_n[n] = acc_v;
        }

        rocke_gfx942_attention_tiled_3d_issue_v(ctx, kv_tile_iv, v_buf, cur_page_id);
        rocke_gfx942_attention_tiled_3d_issue_k(ctx, safe_next_tile, nxt_buf, nxt_page_id);
        if(cfg->WAVE_PAGE_ID_LOAD)
        {
            nxt_page_id_window = rocke_gfx942_attention_tiled_3d_refill_page_id_window(
                ctx, page_id_window, kv_tile_iv);
        }

        /* ---------------- masked scores ---------------- */
        for(reg = 0; reg < 4; reg++)
        {
            /* S^T element reg of block n: token 4 * lane_rg + reg of the block. */
            rocke_value_t* tok_mul = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
            rocke_value_t* tok_in_blk = rocke_b_add(b, tok_mul, rocke_b_const_i32(b, reg));
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                rocke_value_t* ca_n = rocke_b_const_i32(b, n);
                rocke_value_t* ca_16 = rocke_b_const_i32(b, 16);
                rocke_value_t* col_abs = rocke_b_add(
                    b, rocke_b_add(b, tile_off, rocke_b_mul(b, ca_n, ca_16)), tok_in_blk);
                rocke_value_t* causal_ok = rocke_b_cmp_le(b, col_abs, ctx->row_causal_lim);
                rocke_value_t* in_prefix = rocke_b_cmp_lt(b, col_abs, ctx->max_seq_prefix_len);
                rocke_value_t* m_ok
                    = rocke_b_land(b, rocke_b_land(b, ctx->row_ok, causal_ok), in_prefix);
                rocke_value_t* s_raw;
                rocke_value_t* s_scaled;
                if(cfg->SLIDING_WINDOW > 0)
                {
                    rocke_value_t* dist = rocke_b_sub(b, ctx->row_causal_lim, col_abs);
                    m_ok = rocke_b_land(b, m_ok, rocke_b_cmp_lt(b, dist, ctx->sw_const));
                }
                s_raw = rocke_b_vec_extract(b, S_n[n], reg);
                s_scaled = rocke_b_fmul(b, s_raw, ctx->qk_scale);
                if(cfg->USE_SOFTCAP)
                {
                    s_scaled = rocke_b_fmul(
                        b, rocke_apply_softcap_log2(b, s_scaled, ctx->softcap_p), ctx->rcp_ln2);
                }
                if(cfg->USE_ALIBI)
                {
                    rocke_value_t* pos_off = rocke_b_sub(b, col_abs, ctx->context_len);
                    rocke_value_t* pos_f = rocke_b_sitofp_f32(b, pos_off);
                    rocke_value_t* add_term
                        = rocke_b_fmul(b, rocke_b_fmul(b, ctx->row_alibi, pos_f), ctx->rcp_ln2);
                    s_scaled = rocke_b_fadd(b, s_scaled, add_term);
                }
                if(cfg->USE_QQ_BIAS)
                {
                    rocke_value_t* krp = rocke_b_sub(b, col_abs, ctx->context_len);
                    rocke_value_t* krp_ok
                        = rocke_b_land(b,
                                       rocke_b_cmp_ge(b, krp, rocke_b_const_i32(b, 0)),
                                       rocke_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p));
                    rocke_value_t* qq_ok = rocke_b_land(b, ctx->row_ok, krp_ok);
                    rocke_value_t* qp_safe
                        = rocke_b_select(b, ctx->row_ok, ctx->row_qp, rocke_b_const_i32(b, 0));
                    rocke_value_t* qq_idx
                        = rocke_b_add(b, rocke_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    rocke_value_t* qq_v = rocke_b_masked_global_load(
                        b, ctx->qq_bias_ptr, qq_idx, qq_ok, rocke_b_const_f32(b, 0.0), f32, 4);
                    s_scaled = rocke_b_fadd(b, s_scaled, rocke_b_fmul(b, qq_v, ctx->rcp_ln2));
                }
                masked[n * 4 + reg] = rocke_b_select(b, m_ok, s_scaled, ctx->neg_inf);
            }
        }

        /* ---------------- online (m,l) update ----------------
         * Row max and sum: this lane's 4 * QK_N_TILES tokens, then the row's
         * other 3 lanes. */
        {
            rocke_value_t* local_max = ctx->neg_inf;
            rocke_value_t* full_max_raw;
            rocke_value_t* ok;
            rocke_value_t* sum_p = ctx->zero_f;
            for(reg = 0; reg < 4; reg++)
            {
                for(n = 0; n < cfg->QK_N_TILES; n++)
                {
                    local_max = rocke_b_fmax(b, local_max, masked[n * 4 + reg]);
                }
            }
            full_max_raw = rocke__row_reduce(b, local_max, true);
            ok = rocke_b_fcmp(b, "ogt", full_max_raw, ctx->neg_inf);
            m_new = rocke_b_select(b, ok, full_max_raw, ctx->zero_f);
            for(reg = 0; reg < 4; reg++)
            {
                for(n = 0; n < cfg->QK_N_TILES; n++)
                {
                    rocke_value_t* p
                        = rocke_b_exp2_fast(b, rocke_b_fsub(b, masked[n * 4 + reg], m_new));
                    p_regs[n * 4 + reg] = p;
                    sum_p = rocke_b_fadd(b, sum_p, p);
                }
            }
            l_local = rocke__row_reduce(b, sum_p, false);
        }
        alpha = rocke_b_exp2_fast(b, rocke_b_fsub(b, m_val, m_new));
        new_l_val = rocke_b_fadd(b, rocke_b_fmul(b, l_val, alpha), l_local);
        if(cfg->KV_FP8)
        {
            /* FP8 sync loader has no in-flight async work. */
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_sync(b);
        }
        else
        {
            rocke_b_s_waitcnt(b, cfg->kv_calls_per_tile, cfg->kv_calls_per_tile, -1);
            rocke_b_sync(b);
        }

        /* ---------------- PV: O^T = V^T P^T ----------------
         * One 16-token block per MFMA: A = V^T (head-dim rows n*16 + 0..15 via
         * 4 strided LDS loads; gfx942 has no transpose read), B = P^T straight
         * from this lane's registers. */
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            rocke_value_t* scaled_comps[4];
            rocke_value_t* acc_v;
            rocke_value_t* v_col;
            rocke_value_t* v_row_base;
            int kb;
            for(reg = 0; reg < 4; reg++)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, acc_vals[n], reg);
                scaled_comps[reg] = rocke_b_fmul(b, e, alpha);
            }
            acc_v = rocke_b_vec_pack(b, scaled_comps, 4, f32);
            {
                rocke_value_t* vc_n = rocke_b_const_i32(b, n);
                rocke_value_t* vc_16 = rocke_b_const_i32(b, 16);
                v_col = rocke_b_add(b, rocke_b_mul(b, vc_n, vc_16), ctx->lane_col);
            }
            v_row_base = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
            for(kb = 0; kb < cfg->QK_N_TILES; kb++)
            {
                rocke_value_t* A_v = rocke_gfx942_attention_tiled_3d_strided_v_b_operand(
                    ctx, kb, v_buf, v_col, v_row_base);
                rocke_value_t* p_comps[4];
                rocke_value_t* B_p;
                for(reg = 0; reg < 4; reg++)
                {
                    p_comps[reg] = rocke_b_cast_f32_to(b, p_regs[kb * 4 + reg], dtype);
                }
                B_p = rocke_b_vec_pack(b, p_comps, 4, dtype);
                acc_v = rocke_mfma_16x16x16_for_dtype(b, dtype, A_v, B_p, acc_v);
            }
            new_acc[n] = acc_v;
        }

        /* ---------------- carry yield ---------------- */
        {
            rocke_value_t** yields = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, (size_t)num_iter * sizeof(rocke_value_t*));
            int yi = 0;
            if(yields == NULL)
            {
                rocke_b_region_leave(b);
                return;
            }
            yields[yi++] = m_new;
            yields[yi++] = new_l_val;
            for(n = 0; n < cfg->PV_N_TILES; n++)
            {
                yields[yi++] = new_acc[n];
            }
            yields[yi++] = nxt_buf;
            if(cfg->WAVE_PAGE_ID_LOAD)
            {
                yields[yi++] = nxt_page_id;
                yields[yi++] = nxt_page_id_window;
            }
            rocke_b_scf_yield(b, yields, yi);
        }
    }
    rocke_b_region_leave(b);

    /* ---------------- stash final (m,l,acc) into ctx ---------------- */
    {
        rocke_value_t** final = kvloop.op->results;
        ctx->m_final = final[0];
        ctx->l_final = final[1];
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_final[n] = final[2 + n];
        }
    }
}

/* ============================================================ *
 * emit_epilogue
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_emit_epilogue(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    int n;
    rocke_value_t* qtoken = rocke_b_add(b, ctx->cu_q_start, ctx->row_qp);
    rocke_value_t* ml_idx = NULL;
    rocke_if_t guard;

    /* ---- segm_output: O^T tile n, head dims n*16 + 4*lane_rg + 0..3 ---- */
    guard = rocke_b_scf_if(b, ctx->row_ok);
    rocke_b_region_enter(b, guard.then_region);
    for(n = 0; n < cfg->PV_N_TILES; n++)
    {
        rocke_value_t* d0_c = rocke_b_const_i32(b, n * 16);
        rocke_value_t* d0_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
        rocke_value_t* d0 = rocke_b_add(b, d0_c, d0_m);
        rocke_value_t* seg_acc_idx = NULL;
        const char* names[4];
        rocke_value_t* vals[4];
        names[0] = "token";
        vals[0] = qtoken;
        names[1] = "head";
        vals[1] = ctx->row_qh;
        names[2] = "seg";
        vals[2] = ctx->seg_idx;
        names[3] = "dim";
        vals[3] = d0;
        rocke_transforms_descriptor_offset(
            b, ctx->seg_acc_desc, names, vals, 4, &seg_acc_idx, NULL);
        rocke_b_global_store_vN(b, ctx->segm_output_ptr, seg_acc_idx, ctx->acc_final[n], 4, 16);
    }
    rocke_b_region_leave(b);

    /* ---- segm_max / segm_expsum: the lanes with lane_rg == 0 ---- */
    {
        const char* names[3];
        rocke_value_t* vals[3];
        rocke_value_t* lane0;
        names[0] = "token";
        vals[0] = qtoken;
        names[1] = "head";
        vals[1] = ctx->row_qh;
        names[2] = "seg";
        vals[2] = ctx->seg_idx;
        rocke_transforms_descriptor_offset(b, ctx->ml_desc, names, vals, 3, &ml_idx, NULL);
        lane0 = rocke_b_cmp_eq(b, ctx->lane_rg, rocke_b_const_i32(b, 0));
        guard = rocke_b_scf_if(b, rocke_b_land(b, lane0, ctx->row_ok));
    }
    rocke_b_region_enter(b, guard.then_region);
    rocke_b_global_store(b, ctx->segm_max_ptr, ml_idx, ctx->m_final, 4);
    rocke_b_global_store(b, ctx->segm_expsum_ptr, ml_idx, ctx->l_final, 4);
    rocke_b_region_leave(b);
}
