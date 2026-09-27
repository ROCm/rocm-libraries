// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_gfx942_attention_tiled_2d_conflict_free_v.c
 *   -- C99 port of the CONFLICT-FREE TRANSPOSED-V (cfv / cfvst) store path of
 *   rocke/instances/gfx942/attention_tiled_2d.py (build body lines 2621-2904).
 *
 * SCOPE (this translation unit).
 *   rocke_gfx942_attn2d_issue_v_transposed         -- vehicle (a): SYNCHRONOUS
 *     token-strided global gather + ONE contiguous vector ds_write per work item
 *     (the proven-correct but ~5x-slower transposed-V fill). Lines 2658-2732.
 *   rocke_gfx942_attn2d_load_token_row_pair        -- vehicle (c) inner closure:
 *     bounded coalesced <2 x f16> dim-pair VMEM load (+ the _CFV_STORE_SEPOFF /
 *     _CFV_STORE_SCALAR_LOAD isolation variants), returned bitcast-to-i32.
 *   rocke_gfx942_attn2d_cfvst_block_coords         -- vehicle (c) inner closure:
 *     block index -> (d0, d1, t0, t1) 2x2 f16 block coordinates.
 *   rocke_gfx942_attn2d_cfvst_load_v_regs          -- vehicle (c): issue the VMEM
 *     loads, keep each thread's 2x2 V tile in VGPRs, return an arena-owned
 *     payload list of (d0, d1, t0, x0, x1).
 *   rocke_gfx942_attn2d_cfvst_store_v_regs         -- vehicle (c): perm_b32 in-
 *     register 2x2 transpose + ONE contiguous 2-half ds_write per dim row (+ the
 *     _CFV_STORE_PREZERO / _CFV_STORE_SCATTER diagnostic variants).
 *   rocke_gfx942_attn2d_issue_v_transposed_store   -- vehicle (c) driver:
 *     _cfvst_store_v_regs(_cfvst_load_v_regs(kv_tile_idx)).
 *
 * The builder-call sequence is byte-identical to the Python body: same ops, same
 * order, same operands. Per-thread tiling counters that are Python function-local
 * (V_T_VEC, V_T_ITEMS_PER_THREAD, _v_t_need_item_guard, _v_t_token_groups,
 * _v_t2_*) are recomputed here from the ctx geometry constants (HD/T/THREADS),
 * exactly as the Python prologue computes them.
 *
 * NOTE (max_seq_prefix_len). Both vehicles bound their HBM read by this CTA's
 * valid KV length, ``max_seq_prefix_len`` (Python local, line 1984). The shared
 * ctx does not carry it as a field, so it is reconstructed here from the ctx
 * locals it derives from (context_len + qb_start_pos + (BLOCK_M-1)//NQK + 1,
 * clamped to seq_len) -- the identical builder-call chain. A header revision that
 * promotes max_seq_prefix_len to a ctx field would let both this bucket and the
 * KV-loop-bounds bucket share the single SSA value.
 *
 * Lifetime: every emitted node + the payload list is arena-owned
 * (rocke_ir_builder_t.arena). Nothing is freed individually.
 */
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"

/* ==========================================================================
 * Local helpers shared by both vehicles.
 * ========================================================================== */

/* paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=) -> (offset, valid).
 * Mirrors the Python keyword call; *out_valid receives Python None as NULL. */
static rocke_value_t* rocke__paged_kv_offset(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* linear_half,
                                             rocke_value_t* kv_head,
                                             rocke_value_t** out_valid)
{
    static const char* names[3] = {"tile_idx", "linear_half", "kv_head"};
    rocke_value_t* in_vals[3];
    rocke_value_t* off = NULL;
    rocke_value_t* vld = NULL;
    in_vals[0] = tile_idx;
    in_vals[1] = linear_half;
    in_vals[2] = kv_head;
    (void)rocke_transforms_descriptor_offset(ctx->b, ctx->kv_desc, names, in_vals, 3, &off, &vld);
    if(out_valid)
        *out_valid = vld;
    return off;
}

/* max_seq_prefix_len = select(msp_raw < seq_len, msp_raw, seq_len)  (line 1984)
 *   msp_raw = context_len + qb_start_pos + ((BLOCK_M-1)//NQK + 1)    (line 1983)
 * Reconstructed from ctx locals (see file header NOTE). */
static rocke_value_t* rocke__max_seq_prefix_len(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int bm1_div_nqk = (ctx->BLOCK_M - 1) / ctx->NQK;
    rocke_value_t* msp_raw = b ? rocke_b_add(b,
                                             rocke_b_add(b, ctx->context_len, ctx->qb_start_pos),
                                             rocke_b_const_i32(b, bm1_div_nqk + 1))
                               : NULL;
    return rocke_b_select(b, rocke_b_cmp_lt(b, msp_raw, ctx->seq_len), msp_raw, ctx->seq_len);
}

/* Per-thread tiling counters (Python function-locals lines 2647-2656). */
static int rocke__v_t_vec(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    int T = ctx->T;
    if(T % 4 == 0)
        return 4;
    if(T % 2 == 0)
        return 2;
    return 1;
}

static int rocke__v_t_items_per_thread(const rocke_gfx942_attn2d_build_ctx_t* ctx, int v_t_vec)
{
    int token_groups = ctx->T / v_t_vec;
    int total_items = ctx->HD * token_groups;
    return (total_items + ctx->THREADS - 1) / ctx->THREADS;
}

/* ==========================================================================
 * rocke_gfx942_attn2d_issue_v_transposed  (vehicle (a), lines 2658-2732)
 *
 * VECTOR-store transpose of V into V_lds[0, col, token] (f16). Per work item
 * ``item = call*THREADS + tid`` -> head-dim row col = item // token_groups and
 * token group g = item % token_groups (token_base = g*V_T_VEC). Reads V_T_VEC
 * consecutive tokens at head-dim col from HBM (token-strided sync gather) and
 * writes them as ONE contiguous vector ds_write.
 * ========================================================================== */
void rocke_gfx942_attn2d_issue_v_transposed(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* kv_tile_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    int V_T_VEC = rocke__v_t_vec(ctx);
    int token_groups = ctx->T / V_T_VEC;
    int total_items = ctx->HD * token_groups;
    int V_T_ITEMS_PER_THREAD = rocke__v_t_items_per_thread(ctx, V_T_VEC);
    bool need_item_guard = (total_items % ctx->THREADS) != 0;
    rocke_value_t* max_seq_prefix_len = rocke__max_seq_prefix_len(ctx);
    int call;
    int j;

    /* _tg = b.const_i32(_v_t_token_groups) */
    rocke_value_t* tg = rocke_b_const_i32(b, token_groups);
    /* zero_h = b.cast_f32_to(b.const_f32(0.0), dtype) */
    rocke_value_t* zero_h = rocke_b_cast_f32_to(b, rocke_b_const_f32(b, 0.0), dtype);

    for(call = 0; call < V_T_ITEMS_PER_THREAD; ++call)
    {
        /* item = call*THREADS + tid */
        rocke_value_t* item = rocke_b_add(
            b,
            rocke_b_mul(b, rocke_b_const_i32(b, call), rocke_b_const_i32(b, ctx->THREADS)),
            ctx->tid);
        if(need_item_guard)
        {
            /* item = select(item < total_items, item, 0) */
            item = rocke_b_select(b,
                                  rocke_b_cmp_lt(b, item, rocke_b_const_i32(b, total_items)),
                                  item,
                                  rocke_b_const_i32(b, 0));
        }
        rocke_value_t* col = rocke_b_div(b, item, tg);
        rocke_value_t* g = rocke_b_mod(b, item, tg);
        rocke_value_t* token_base = rocke_b_mul(b, g, rocke_b_const_i32(b, V_T_VEC));
        /* tile_tok_base = kv_tile_idx * T */
        rocke_value_t* tile_tok_base = rocke_b_mul(b, kv_tile_idx, rocke_b_const_i32(b, ctx->T));

        rocke_value_t* elems[4]; /* V_T_VEC <= 4 */
        for(j = 0; j < V_T_VEC; ++j)
        {
            rocke_value_t* token_j = rocke_b_add(b, token_base, rocke_b_const_i32(b, j));
            /* linear_j = token_j * HD + col */
            rocke_value_t* linear_j
                = rocke_b_add(b, rocke_b_mul(b, token_j, rocke_b_const_i32(b, ctx->HD)), col);
            rocke_value_t* valid_j = NULL;
            rocke_value_t* voff_j
                = rocke__paged_kv_offset(ctx, kv_tile_idx, linear_j, ctx->kv_head_idx, &valid_j);
            /* in_range = (tile_tok_base + token_j) < max_seq_prefix_len */
            rocke_value_t* in_range
                = rocke_b_cmp_lt(b, rocke_b_add(b, tile_tok_base, token_j), max_seq_prefix_len);
            if(valid_j != NULL)
                in_range = rocke_b_land(b, in_range, valid_j);
            rocke_value_t* safe_voff_j
                = rocke_b_select(b, in_range, voff_j, rocke_b_const_i32(b, 0));
            rocke_value_t* safe_elem_j
                = (ctx->KV_BYTES != 1)
                      ? rocke_b_div(b, safe_voff_j, rocke_b_const_i32(b, ctx->KV_BYTES))
                      : safe_voff_j;
            rocke_value_t* v1 = rocke_b_global_load(b, ctx->value, safe_elem_j, dtype, 2);
            v1 = rocke_b_select(b, in_range, v1, zero_h);
            elems[j] = v1;
        }
        /* v_vec = b.vec_pack(elems, dtype) */
        rocke_value_t* v_vec = rocke_b_vec_pack(b, elems, V_T_VEC, dtype);
        /* _v_t_store(col, token_base, v_vec, V_T_VEC) */
        rocke_gfx942_attn2d_v_t_store(ctx, col, token_base, v_vec, V_T_VEC);
    }
}

/* ==========================================================================
 * cfvst (vehicle (c)) payload + inner closures.
 * ========================================================================== */

/* One loaded 2x2 block: the Python tuple (d0, d1, t0, x0, x1). */
typedef struct rocke__cfvst_blk
{
    rocke_value_t* d0;
    rocke_value_t* d1;
    rocke_value_t* t0;
    rocke_value_t* x0; /* i32 = (V[t0,d0], V[t0,d1]) */
    rocke_value_t* x1; /* i32 = (V[t1,d0], V[t1,d1]) */
} rocke__cfvst_blk_t;

/* The arena-owned payload _cfvst_load_v_regs returns + _cfvst_store_v_regs
 * consumes. kv_tile_idx is retained so the store path can re-derive the prezero
 * coverage loop bound (it uses only ctx constants there). */
typedef struct rocke__cfvst_payload
{
    rocke__cfvst_blk_t* blocks;
    int count;
} rocke__cfvst_payload_t;

/* _load_token_row_pair(t_row, d0) -> i32 = (V[t_row,d0], V[t_row,d0+1]), bounded
 * (lines 2773-2832). kv_tile_idx + max_seq_prefix_len are passed via ctx-private
 * statics threaded through _cfvst_load_v_regs (see below); to keep this a free
 * function with the header-declared signature, it recomputes them from ctx. */
rocke_value_t* rocke_gfx942_attn2d_load_token_row_pair(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                       rocke_value_t* t_row,
                                                       rocke_value_t* d0)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    const rocke_type_t* I32 = rocke_i32();
    rocke_value_t* kv_tile_idx = ctx->kv_tile_iv; /* current tile (set by caller) */
    rocke_value_t* zero_h = rocke_b_cast_f32_to(b, rocke_b_const_f32(b, 0.0), dtype);
    rocke_value_t* zero_i32 = rocke_b_const_i32(b, 0);
    rocke_value_t* max_seq_prefix_len = rocke__max_seq_prefix_len(ctx);
    rocke_value_t* tile_tok_base = rocke_b_mul(b, kv_tile_idx, rocke_b_const_i32(b, ctx->T));

    /* linear = t_row * HD + d0 */
    rocke_value_t* linear
        = rocke_b_add(b, rocke_b_mul(b, t_row, rocke_b_const_i32(b, ctx->HD)), d0);
    rocke_value_t* valid = NULL;
    rocke_value_t* voff
        = rocke__paged_kv_offset(ctx, kv_tile_idx, linear, ctx->kv_head_idx, &valid);
    rocke_value_t* in_range
        = rocke_b_cmp_lt(b, rocke_b_add(b, tile_tok_base, t_row), max_seq_prefix_len);
    if(valid != NULL)
        in_range = rocke_b_land(b, in_range, valid);
    rocke_value_t* safe_voff = rocke_b_select(b, in_range, voff, zero_i32);
    rocke_value_t* safe_elem = (ctx->KV_BYTES != 1)
                                   ? rocke_b_div(b, safe_voff, rocke_b_const_i32(b, ctx->KV_BYTES))
                                   : safe_voff;
    rocke_value_t* v2;

    if(ctx->CFV_STORE_SEPOFF)
    {
        /* Separate descriptor call for (t_row, d0+1) instead of voff+1. */
        rocke_value_t* linear1 = rocke_b_add(b,
                                             rocke_b_mul(b, t_row, rocke_b_const_i32(b, ctx->HD)),
                                             rocke_b_add(b, d0, rocke_b_const_i32(b, 1)));
        rocke_value_t* valid1 = NULL;
        rocke_value_t* voff1
            = rocke__paged_kv_offset(ctx, kv_tile_idx, linear1, ctx->kv_head_idx, &valid1);
        (void)valid1;
        rocke_value_t* safe_voff1 = rocke_b_select(b, in_range, voff1, zero_i32);
        rocke_value_t* safe_elem1
            = (ctx->KV_BYTES != 1) ? rocke_b_div(b, safe_voff1, rocke_b_const_i32(b, ctx->KV_BYTES))
                                   : safe_voff1;
        rocke_value_t* e0 = rocke_b_global_load(b, ctx->value, safe_elem, dtype, 2);
        rocke_value_t* e1 = rocke_b_global_load(b, ctx->value, safe_elem1, dtype, 2);
        e0 = rocke_b_select(b, in_range, e0, zero_h);
        e1 = rocke_b_select(b, in_range, e1, zero_h);
        rocke_value_t* pk[2] = {e0, e1};
        v2 = rocke_b_vec_pack(b, pk, 2, dtype);
    }
    else if(ctx->CFV_STORE_SCALAR_LOAD)
    {
        /* Two scalar n=1 loads + manual pack. */
        rocke_value_t* e0 = rocke_b_global_load(b, ctx->value, safe_elem, dtype, 2);
        rocke_value_t* e1 = rocke_b_global_load(
            b, ctx->value, rocke_b_add(b, safe_elem, rocke_b_const_i32(b, 1)), dtype, 2);
        e0 = rocke_b_select(b, in_range, e0, zero_h);
        e1 = rocke_b_select(b, in_range, e1, zero_h);
        rocke_value_t* pk[2] = {e0, e1};
        v2 = rocke_b_vec_pack(b, pk, 2, dtype);
    }
    else
    {
        v2 = rocke_b_global_load_vN(b, ctx->value, safe_elem, dtype, 2, 4);
        /* Zero masked rows (whole pair belongs to one token). */
        rocke_value_t* zpk[2] = {zero_h, zero_h};
        rocke_value_t* zero_vec = rocke_b_vec_pack(b, zpk, 2, dtype);
        v2 = rocke_b_select(b, in_range, v2, zero_vec);
    }
    return rocke_b_bitcast(b, v2, I32);
}

/* _cfvst_block_coords(blk) -> (d0, d1, t0, t1)  (lines 2834-2844).
 * out_a[0..1] = (d0, d1); out_b[0..1] = (t0, t1). */
void rocke_gfx942_attn2d_cfvst_block_coords(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* blk,
                                            rocke_value_t** out_a,
                                            rocke_value_t** out_b)
{
    rocke_ir_builder_t* b = ctx->b;
    int dim_pairs = ctx->HD / 2;
    rocke_value_t* dp = rocke_b_const_i32(b, dim_pairs);
    /* tg = blk // _dp ; dg = blk % _dp */
    rocke_value_t* tg = rocke_b_div(b, blk, dp);
    rocke_value_t* dg = rocke_b_mod(b, blk, dp);
    rocke_value_t* t0 = rocke_b_mul(b, tg, rocke_b_const_i32(b, 2));
    rocke_value_t* t1 = rocke_b_add(b, t0, rocke_b_const_i32(b, 1));
    rocke_value_t* d0 = rocke_b_mul(b, dg, rocke_b_const_i32(b, 2));
    rocke_value_t* d1 = rocke_b_add(b, d0, rocke_b_const_i32(b, 1));
    if(out_a)
    {
        out_a[0] = d0;
        out_a[1] = d1;
    }
    if(out_b)
    {
        out_b[0] = t0;
        out_b[1] = t1;
    }
}

/* _cfvst_load_v_regs(kv_tile_idx)  (lines 2766-2859).
 * Issues the cfvst VMEM loads and keeps each thread's V tile in VGPRs. Returns
 * the arena-owned payload (the per-call (d0, d1, t0, x0, x1) tuples). */
void* rocke_gfx942_attn2d_cfvst_load_v_regs(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* kv_tile_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    int V_T_VEC = rocke__v_t_vec(ctx);
    int V_T_ITEMS_PER_THREAD = rocke__v_t_items_per_thread(ctx, V_T_VEC);
    int token_groups = ctx->T / V_T_VEC;
    int total_items = ctx->HD * token_groups;
    bool need_item_guard = (total_items % ctx->THREADS) != 0;
    int tok_pairs = ctx->T / 2;
    int dim_pairs = ctx->HD / 2;
    int total_blocks = tok_pairs * dim_pairs;
    int call;

    /* _load_token_row_pair reads the *current* tile from ctx->kv_tile_iv; pin it
     * to kv_tile_idx for the duration of this load (the Python closure captures
     * the argument directly). */
    rocke_value_t* saved_iv = ctx->kv_tile_iv;
    ctx->kv_tile_iv = kv_tile_idx;

    rocke__cfvst_payload_t* payload
        = (rocke__cfvst_payload_t*)rocke_arena_alloc(&b->arena, sizeof(rocke__cfvst_payload_t));
    if(payload == NULL)
    {
        ctx->kv_tile_iv = saved_iv;
        return NULL;
    }
    payload->blocks = (rocke__cfvst_blk_t*)rocke_arena_alloc(
        &b->arena,
        sizeof(rocke__cfvst_blk_t) * (size_t)(V_T_ITEMS_PER_THREAD > 0 ? V_T_ITEMS_PER_THREAD : 1));
    payload->count = 0;
    if(payload->blocks == NULL)
    {
        ctx->kv_tile_iv = saved_iv;
        return NULL;
    }

    for(call = 0; call < V_T_ITEMS_PER_THREAD; ++call)
    {
        /* blk = call*THREADS + tid */
        rocke_value_t* blk = rocke_b_add(
            b,
            rocke_b_mul(b, rocke_b_const_i32(b, call), rocke_b_const_i32(b, ctx->THREADS)),
            ctx->tid);
        if(need_item_guard)
        {
            blk = rocke_b_select(b,
                                 rocke_b_cmp_lt(b, blk, rocke_b_const_i32(b, total_blocks)),
                                 blk,
                                 rocke_b_const_i32(b, 0));
        }
        rocke_value_t* dd[2];
        rocke_value_t* tt[2];
        rocke_gfx942_attn2d_cfvst_block_coords(ctx, blk, dd, tt);
        rocke_value_t* d0 = dd[0];
        rocke_value_t* d1 = dd[1];
        rocke_value_t* t0 = tt[0];
        rocke_value_t* t1 = tt[1];
        /* x0 = (V[t0,d0], V[t0,d1]) ; x1 = (V[t1,d0], V[t1,d1]) */
        rocke_value_t* x0 = rocke_gfx942_attn2d_load_token_row_pair(ctx, t0, d0);
        rocke_value_t* x1 = rocke_gfx942_attn2d_load_token_row_pair(ctx, t1, d0);
        payload->blocks[payload->count].d0 = d0;
        payload->blocks[payload->count].d1 = d1;
        payload->blocks[payload->count].t0 = t0;
        payload->blocks[payload->count].x0 = x0;
        payload->blocks[payload->count].x1 = x1;
        payload->count += 1;
    }

    ctx->kv_tile_iv = saved_iv;
    return payload;
}

/* _cfvst_store_v_regs(payload)  (lines 2861-2898).
 * Permute the loaded cfvst VGPRs and publish the transposed V LDS tile. */
void rocke_gfx942_attn2d_cfvst_store_v_regs(rocke_gfx942_attn2d_build_ctx_t* ctx, void* payload_v)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    const rocke_type_t* vec2 = rocke_vector_type(b, dtype, 2);
    rocke__cfvst_payload_t* payload = (rocke__cfvst_payload_t*)payload_v;
    rocke_value_t* zero_h = rocke_b_cast_f32_to(b, rocke_b_const_f32(b, 0.0), dtype);
    int i;
    (void)zero_h;

    if(payload == NULL)
        return;

    if(ctx->CFV_STORE_PREZERO)
    {
        /* DIAGNOSTIC: pre-zero every V_lds slot (dim x token). */
        rocke_for_t pz = rocke_b_scf_for(b,
                                         ctx->tid,
                                         rocke_b_const_i32(b, ctx->HD * ctx->T),
                                         rocke_b_const_i32(b, ctx->THREADS),
                                         "vpz");
        rocke_b_region_enter(b, pz.body);
        {
            rocke_value_t* _pd = rocke_b_div(b, pz.iv, rocke_b_const_i32(b, ctx->T));
            rocke_value_t* _ptk = rocke_b_mod(b, pz.iv, rocke_b_const_i32(b, ctx->T));
            rocke_gfx942_attn2d_v_t_store(ctx, _pd, _ptk, zero_h, 1);
        }
        rocke_b_region_leave(b);
        rocke_b_sync(b);
    }

    for(i = 0; i < payload->count; ++i)
    {
        rocke_value_t* d0 = payload->blocks[i].d0;
        rocke_value_t* d1 = payload->blocks[i].d1;
        rocke_value_t* t0 = payload->blocks[i].t0;
        rocke_value_t* x0 = payload->blocks[i].x0;
        rocke_value_t* x1 = payload->blocks[i].x1;

        if(ctx->CFV_STORE_SCATTER)
        {
            /* DIAGNOSTIC: element-wise scatter (no perm). */
            rocke_value_t* x0v = rocke_b_bitcast(b, x0, vec2); /* (V[t0,d0],V[t0,d1]) */
            rocke_value_t* x1v = rocke_b_bitcast(b, x1, vec2); /* (V[t1,d0],V[t1,d1]) */
            rocke_value_t* t1 = rocke_b_add(b, t0, rocke_b_const_i32(b, 1));
            rocke_value_t* v_t0_d0 = rocke_b_vec_extract(b, x0v, 0);
            rocke_value_t* v_t0_d1 = rocke_b_vec_extract(b, x0v, 1);
            rocke_value_t* v_t1_d0 = rocke_b_vec_extract(b, x1v, 0);
            rocke_value_t* v_t1_d1 = rocke_b_vec_extract(b, x1v, 1);
            rocke_gfx942_attn2d_v_t_store(ctx, d0, t0, v_t0_d0, 1);
            rocke_gfx942_attn2d_v_t_store(ctx, d0, t1, v_t1_d0, 1);
            rocke_gfx942_attn2d_v_t_store(ctx, d1, t0, v_t0_d1, 1);
            rocke_gfx942_attn2d_v_t_store(ctx, d1, t1, v_t1_d1, 1);
        }
        else
        {
            /* 2x2 transpose: each output i32 = 2 consecutive tokens at one dim. */
            rocke_value_t* row_d0
                = rocke_b_perm_b32(b, x0, x1, rocke_b_const_i32(b, 0x01000504)); /* (t0,t1)@d0 */
            rocke_value_t* row_d1
                = rocke_b_perm_b32(b, x0, x1, rocke_b_const_i32(b, 0x03020706)); /* (t0,t1)@d1 */
            /* ONE contiguous 2-half ds_write per dim row (token is inner). */
            rocke_gfx942_attn2d_v_t_store(ctx, d0, t0, rocke_b_bitcast(b, row_d0, vec2), 2);
            rocke_gfx942_attn2d_v_t_store(ctx, d1, t0, rocke_b_bitcast(b, row_d1, vec2), 2);
        }
    }
}

/* _issue_v_transposed_store(kv_tile_idx)  (lines 2900-2903).
 * Register-load + perm_b32 transpose + contiguous LDS store of V into
 * V_lds[0, dim, token] (vehicle (c)). */
void rocke_gfx942_attn2d_issue_v_transposed_store(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* kv_tile_idx)
{
    void* payload = rocke_gfx942_attn2d_cfvst_load_v_regs(ctx, kv_tile_idx);
    rocke_gfx942_attn2d_cfvst_store_v_regs(ctx, payload);
}
