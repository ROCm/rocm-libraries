// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_3d_gfx950_attention_tiled_3d_segment_loop_epilogue.c
 *
 * Chunked C99 port of the segment-kernel online-softmax COMPUTE region of
 * rocke/instances/gfx950/attention_tiled_3d.py (gfx950 WIDE-K variant).
 *
 * SCOPE (this translation unit): the "segment loop + epilogue" slice of
 *   build_unified_attention_3d_tiled (lines 253-950):
 *
 *   rocke_gfx950_attention_tiled_3d_emit_loop_init        lines 536-562, 715-718
 *       - sinks-conditioned m_init / acc_inits / cur_buf_init carry
 *       - async DMA infra + paged-KV descriptor (via emit_async_infra)
 *       - first K load (_issue_k(tile_start, 0))
 *   rocke_gfx950_attention_tiled_3d_emit_softmax_loop     lines 720-898
 *       - scf.for over [tile_start, tile_end): buffer swap, s_waitcnt + sync,
 *         QK wide 16x16x32 MFMA, V/next-K prefetch, alibi/softcap/qq_bias/mask,
 *         online (m,l) update (one query row per lane: in-lane, then ds_swizzle
 *         xor 16 + permlane32_swap), PV MFMA (O^T = V^T P^T, 16x16x16, V^T via
 *         ds_read_tr16_b64, P^T from registers), carry yield
 *       - stashes m_final/l_final/acc_final into ctx
 *   rocke_gfx950_attention_tiled_3d_emit_epilogue         lines 900-948
 *       - guarded segm_output / segm_max / segm_expsum stores
 *
 * Everything is consumed through the internal header; peers (params/prologue/
 * descriptors/issuers/_mfma_16x16_c_row/emit_async_infra) are called via that
 * header. The builder call sequence mirrors the Python op-for-op so the emitted
 * IR is byte-identical.
 */

#include <string.h>

#include "rocke/instance_gfx950_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Local (file-private) helpers
 * ============================================================ */

static bool rocke__dtype_is(const rocke_type_t* dt, const char* nm)
{
    return dt != NULL && dt->name != NULL && nm != NULL && (strcmp(dt->name, nm) == 0);
}

/* mfma_16x16x32_for_dtype (helpers/attention.py): dispatch
 * mfma_f32_16x16x32_<dtype> for fp16 / bf16. Any other dtype is a Python
 * ValueError -> NULL. */
static rocke_value_t* rocke__mfma_16x16x32(rocke_ir_builder_t* b,
                                           const rocke_type_t* dtype,
                                           rocke_value_t* a,
                                           rocke_value_t* bv,
                                           rocke_value_t* c)
{
    if(rocke__dtype_is(dtype, "f16"))
    {
        return rocke_b_mfma_f32_16x16x32_f16(b, a, bv, c);
    }
    if(rocke__dtype_is(dtype, "bf16"))
    {
        return rocke_b_mfma_f32_16x16x32_bf16(b, a, bv, c);
    }
    return NULL;
}

/* mfma_16x16x16_for_dtype (the PV atom for a single 16-token block). The
 * ported C helper lives in helper_helper_rocke.helpers.attention.h. */
static rocke_value_t* rocke__mfma_16x16x16(rocke_ir_builder_t* b,
                                           const rocke_type_t* dtype,
                                           rocke_value_t* a,
                                           rocke_value_t* bv,
                                           rocke_value_t* c)
{
    return rocke_mfma_16x16x16_for_dtype(b, dtype, a, bv, c);
}

/* _row_reduce(x, op): combine x over the 4 lanes {l, l^16, l^32, l^48} that hold
 * the 4 token groups of one query row: ds_swizzle xor 16, then permlane32_swap.
 * op is fmax when is_max, else fadd. */
static rocke_value_t* rocke__row_reduce(rocke_ir_builder_t* b, rocke_value_t* x, bool is_max)
{
    const rocke_type_t* i32 = rocke_i32();
    const rocke_type_t* f32 = rocke_f32();
    rocke_value_t* sw_i = rocke_b_ds_swizzle_xor(b, rocke_b_bitcast(b, x, i32), 16);
    rocke_value_t* sw = rocke_b_bitcast(b, sw_i, f32);
    rocke_value_t* lo_in;
    rocke_value_t* hi_in;
    rocke_value_t* lo = NULL;
    rocke_value_t* hi = NULL;
    rocke_value_t* lo_f;
    rocke_value_t* hi_f;
    x = is_max ? rocke_b_fmax(b, x, sw) : rocke_b_fadd(b, x, sw);
    lo_in = rocke_b_bitcast(b, x, i32);
    hi_in = rocke_b_bitcast(b, x, i32);
    rocke_b_permlane32_swap(b, lo_in, hi_in, &lo, &hi);
    lo_f = rocke_b_bitcast(b, lo, f32);
    hi_f = rocke_b_bitcast(b, hi, f32);
    return is_max ? rocke_b_fmax(b, lo_f, hi_f) : rocke_b_fadd(b, lo_f, hi_f);
}

/* ============================================================ *
 * emit_loop_init (lines 536-562, 715-718)
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_loop_init(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
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

    /* ---- acc_inits (lines 554-555) ---- */
    {
        rocke_value_t* acc_zero = rocke_b_zero_vec_f32(b, 4);
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_inits[n] = acc_zero;
        }
    }

    /* ---- async DMA infra + paged-KV descriptor (lines 564-601) ----
     * Emitted here (not in emit_prologue) so the SSA op order matches Python's
     * single linear build: acc_zero precedes the buffer rsrc / seq_base / desc. */
    rocke_gfx950_attention_tiled_3d_emit_async_infra(ctx);

    /* ---- first K load + cur_buf_init (lines 715-718) ---- */
    if(cfg->WAVE_PAGE_ID_LOAD)
    {
        /* The wave holds a 64-page window of the segment's page table, one id
         * per lane; a tile never straddles two windows. */
        ctx->seg_first_page
            = rocke_b_mul(b, ctx->tile_start, rocke_b_const_i32(b, cfg->N_BLOCKS_PER_TILE));
        ctx->first_page_id_window
            = rocke_gfx950_attention_tiled_3d_load_page_id_window(ctx, rocke_b_const_i32(b, 0));
        rocke_gfx950_attention_tiled_3d_window_tile_ids(
            ctx, ctx->first_page_id_window, ctx->tile_start, ctx->first_page_ids);
    }
    else if(cfg->PAGE_IDS)
    {
        rocke_gfx950_attention_tiled_3d_tile_page_ids(ctx, ctx->tile_start, ctx->first_page_ids);
    }
    rocke_gfx950_attention_tiled_3d_issue_k(
        ctx, ctx->tile_start, rocke_b_const_i32(b, 0), cfg->PAGE_IDS ? ctx->first_page_ids : NULL);
    ctx->cur_buf_init = rocke_b_const_i32(b, 0);
}

/* ============================================================ *
 * emit_softmax_loop (lines 720-898)
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_softmax_loop(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const rocke_type_t* dtype = cfg->dtype;
    const rocke_type_t* f32 = rocke_f32();
    int n, k, reg;

    /* iter_args: m, l, acc0..accN-1, cur_buf
     * (+ pid0..pid{NB-1}: the current tile's page ids when PAGE_IDS)
     * (+ pid_window: the page-id window when WAVE_PAGE_ID_LOAD) */
    int num_ml = 2;
    int num_pid = cfg->PAGE_IDS ? cfg->N_BLOCKS_PER_TILE : 0;
    int num_win = cfg->WAVE_PAGE_ID_LOAD ? 1 : 0;
    int num_iter = num_ml + cfg->PV_N_TILES + 1 + num_pid + num_win;
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
        for(n = 0; n < num_pid; n++)
        {
            buf[0] = 'p';
            buf[1] = 'i';
            buf[2] = 'd';
            buf[3] = (char)('0' + n);
            buf[4] = '\0';
            iter_args[ai].name = rocke_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->first_page_ids[n];
            ai++;
        }
        if(num_win)
        {
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
        rocke_value_t* q_col_c = rocke_b_const_i32(b, k * 32);
        rocke_value_t* q_col_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
        rocke_value_t* q_col_off = rocke_b_add(b, q_col_c, q_col_m);
        rocke_value_t* idx[2];
        idx[0] = ctx->lane_col;
        idx[1] = q_col_off;
        Q_kits[k] = rocke_b_smem_load_vN(b, ctx->Q_lds, idx, 2, dtype, 8);
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
        rocke_value_t* cur_page_ids[8];
        rocke_value_t* nxt_page_ids[8];
        rocke_value_t* page_id_window = NULL;
        rocke_value_t* nxt_page_id_window = NULL;
        rocke_value_t* tile_off;
        rocke_value_t* next_tile_iv_raw;
        rocke_value_t* in_range_next;
        rocke_value_t* safe_next_tile;

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
        nxt_buf = rocke_b_sub(b, rocke_b_const_i32(b, 1), cur_buf);
        v_buf = rocke_b_const_i32(b, 0);
        for(n = 0; n < num_pid; n++)
        {
            cur_page_ids[n] = carry[3 + cfg->PV_N_TILES + n];
        }
        tile_off = rocke_b_mul(b, kv_tile_iv, rocke_b_const_i32(b, cfg->T));

        next_tile_iv_raw = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 1));
        in_range_next = rocke_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
        safe_next_tile = rocke_b_select(b, in_range_next, next_tile_iv_raw, kv_tile_iv);

        rocke_b_s_waitcnt(b, 0, 0, -1);
        rocke_b_sync(b);
        /* Page ids of the next tile, for its K copies below. */
        if(cfg->WAVE_PAGE_ID_LOAD)
        {
            /* Read from the window in registers: no load in the loop. */
            page_id_window = carry[3 + cfg->PV_N_TILES + num_pid];
            rocke_gfx950_attention_tiled_3d_window_tile_ids(
                ctx, page_id_window, safe_next_tile, nxt_page_ids);
        }
        else if(cfg->PAGE_IDS)
        {
            /* Loaded here, before this tile's V copies, so the load overlaps
             * the QK below; the next K copies still wait for it. */
            rocke_gfx950_attention_tiled_3d_tile_page_ids(ctx, safe_next_tile, nxt_page_ids);
        }

        /* ---------------- QK: S^T = K Q^T (16x16x32, K-step 32) ---------------- */
        for(n = 0; n < cfg->QK_N_TILES; n++)
        {
            rocke_value_t* acc_v = rocke_b_zero_vec_f32(b, 4);
            for(k = 0; k < cfg->QK_K_ITERS; k++)
            {
                rocke_value_t* kc_c = rocke_b_const_i32(b, k * 32);
                rocke_value_t* kc_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
                rocke_value_t* kc_off = rocke_b_add(b, kc_c, kc_m);
                rocke_value_t* k_row = rocke_b_add(b, rocke_b_const_i32(b, n * 16), ctx->lane_col);
                rocke_value_t* idx[3];
                rocke_value_t* K_v;
                idx[0] = cur_buf;
                idx[1] = k_row;
                idx[2] = rocke_gfx950_attention_tiled_3d_kv_col(ctx, k_row, kc_off);
                K_v = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, dtype, 8);
                acc_v = rocke__mfma_16x16x32(b, dtype, K_v, Q_kits[k], acc_v);
            }
            S_n[n] = acc_v;
        }

        rocke_gfx950_attention_tiled_3d_issue_v(
            ctx, kv_tile_iv, v_buf, cfg->PAGE_IDS ? cur_page_ids : NULL);
        rocke_gfx950_attention_tiled_3d_issue_k(
            ctx, safe_next_tile, nxt_buf, cfg->PAGE_IDS ? nxt_page_ids : NULL);
        if(cfg->WAVE_PAGE_ID_LOAD)
        {
            nxt_page_id_window = rocke_gfx950_attention_tiled_3d_refill_page_id_window(
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
         * A = V^T (head-dim rows n*16 + 0..15 via the transpose read), B = P^T
         * straight from this lane's registers. Two 16-token blocks share one
         * 16x16x32 MFMA (same cost as one 16x16x16 on gfx950): A and B list the
         * two blocks' tokens in the same order, which is all the k-sum needs. An
         * odd last block uses 16x16x16. */
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            rocke_value_t* scaled_comps[4];
            rocke_value_t* acc_v;
            rocke_value_t* n_col_base;
            int kb;
            for(reg = 0; reg < 4; reg++)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, acc_vals[n], reg);
                scaled_comps[reg] = rocke_b_fmul(b, e, alpha);
            }
            acc_v = rocke_b_vec_pack(b, scaled_comps, 4, f32);
            {
                rocke_value_t* ncb_n = rocke_b_const_i32(b, n);
                rocke_value_t* ncb_16 = rocke_b_const_i32(b, 16);
                n_col_base = rocke_b_add(b, rocke_b_mul(b, ncb_n, ncb_16), ctx->tr_col_lane);
            }
            for(kb = 0; kb < cfg->QK_N_TILES; kb += 2)
            {
                const int nblk = cfg->QK_N_TILES - kb < 2 ? 1 : 2;
                rocke_value_t* A_parts[2];
                rocke_value_t* p_comps[8];
                rocke_value_t* B_p;
                int blk;
                for(blk = 0; blk < nblk; blk++)
                {
                    rocke_value_t* row_lane = rocke_bound_transpose_lds_reader_row(
                        b, ctx->pv_tr_reader, (kb + blk) * 16, 0);
                    rocke_value_t* av_idx[3];
                    av_idx[0] = v_buf;
                    av_idx[1] = row_lane;
                    av_idx[2] = rocke_gfx950_attention_tiled_3d_kv_col(ctx, row_lane, n_col_base);
                    A_parts[blk] = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, av_idx, 3, dtype);
                }
                for(blk = 0; blk < nblk; blk++)
                {
                    for(reg = 0; reg < 4; reg++)
                    {
                        p_comps[blk * 4 + reg]
                            = rocke_b_cast_f32_to(b, p_regs[(kb + blk) * 4 + reg], dtype);
                    }
                }
                B_p = rocke_b_vec_pack(b, p_comps, nblk * 4, dtype);
                if(nblk == 2)
                {
                    rocke_value_t* A_v = rocke_b_vec_concat(b, A_parts[0], A_parts[1]);
                    acc_v = rocke__mfma_16x16x32(b, dtype, A_v, B_p, acc_v);
                }
                else
                {
                    acc_v = rocke__mfma_16x16x16(b, dtype, A_parts[0], B_p, acc_v);
                }
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
            for(n = 0; n < num_pid; n++)
            {
                yields[yi++] = nxt_page_ids[n];
            }
            if(num_win)
            {
                yields[yi++] = nxt_page_id_window;
            }
            rocke_b_scf_yield(b, yields, yi);
        }
    }
    rocke_b_region_leave(b);

    /* ---------------- stash final (m,l,acc) into ctx (lines 901-904) ---------- */
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
 * emit_epilogue (lines 900-948)
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_epilogue(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
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
