/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx942_attention_tiled_2d_gfx942_attention_tiled_2d_conflict_free_v.c
 *   -- C99 port of the CONFLICT-FREE TRANSPOSED-V (cfv / cfvst) store path of
 *   ck_dsl/instances/gfx942/attention_tiled_2d.py (build body lines 2621-2904).
 *
 * SCOPE (this translation unit).
 *   ckc_gfx942_attn2d_issue_v_transposed         -- vehicle (a): SYNCHRONOUS
 *     token-strided global gather + ONE contiguous vector ds_write per work item
 *     (the proven-correct but ~5x-slower transposed-V fill). Lines 2658-2732.
 *   ckc_gfx942_attn2d_load_token_row_pair        -- vehicle (c) inner closure:
 *     bounded coalesced <2 x f16> dim-pair VMEM load (+ the _CFV_STORE_SEPOFF /
 *     _CFV_STORE_SCALAR_LOAD isolation variants), returned bitcast-to-i32.
 *   ckc_gfx942_attn2d_cfvst_block_coords         -- vehicle (c) inner closure:
 *     block index -> (d0, d1, t0, t1) 2x2 f16 block coordinates.
 *   ckc_gfx942_attn2d_cfvst_load_v_regs          -- vehicle (c): issue the VMEM
 *     loads, keep each thread's 2x2 V tile in VGPRs, return an arena-owned
 *     payload list of (d0, d1, t0, x0, x1).
 *   ckc_gfx942_attn2d_cfvst_store_v_regs         -- vehicle (c): perm_b32 in-
 *     register 2x2 transpose + ONE contiguous 2-half ds_write per dim row (+ the
 *     _CFV_STORE_PREZERO / _CFV_STORE_SCATTER diagnostic variants).
 *   ckc_gfx942_attn2d_issue_v_transposed_store   -- vehicle (c) driver:
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
 * (ckc_ir_builder_t.arena). Nothing is freed individually.
 */
#include "ckc/instance_gfx942_attention_tiled_2d_internal.h"

/* ==========================================================================
 * Local helpers shared by both vehicles.
 * ========================================================================== */

/* paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=) -> (offset, valid).
 * Mirrors the Python keyword call; *out_valid receives Python None as NULL. */
static ckc_value_t* ckc__paged_kv_offset(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                         ckc_value_t* tile_idx,
                                         ckc_value_t* linear_half,
                                         ckc_value_t* kv_head,
                                         ckc_value_t** out_valid)
{
    static const char* names[3] = {"tile_idx", "linear_half", "kv_head"};
    ckc_value_t* in_vals[3];
    ckc_value_t* off = NULL;
    ckc_value_t* vld = NULL;
    in_vals[0] = tile_idx;
    in_vals[1] = linear_half;
    in_vals[2] = kv_head;
    (void)ckc_transforms_descriptor_offset(ctx->b, ctx->kv_desc, names, in_vals, 3, &off, &vld);
    if(out_valid)
        *out_valid = vld;
    return off;
}

/* max_seq_prefix_len = select(msp_raw < seq_len, msp_raw, seq_len)  (line 1984)
 *   msp_raw = context_len + qb_start_pos + ((BLOCK_M-1)//NQK + 1)    (line 1983)
 * Reconstructed from ctx locals (see file header NOTE). */
static ckc_value_t* ckc__max_seq_prefix_len(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    int bm1_div_nqk = (ctx->BLOCK_M - 1) / ctx->NQK;
    ckc_value_t* msp_raw = b ? ckc_b_add(b,
                                         ckc_b_add(b, ctx->context_len, ctx->qb_start_pos),
                                         ckc_b_const_i32(b, bm1_div_nqk + 1))
                             : NULL;
    return ckc_b_select(b, ckc_b_cmp_lt(b, msp_raw, ctx->seq_len), msp_raw, ctx->seq_len);
}

/* Per-thread tiling counters (Python function-locals lines 2647-2656). */
static int ckc__v_t_vec(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    int T = ctx->T;
    if(T % 4 == 0)
        return 4;
    if(T % 2 == 0)
        return 2;
    return 1;
}

static int ckc__v_t_items_per_thread(const ckc_gfx942_attn2d_build_ctx_t* ctx, int v_t_vec)
{
    int token_groups = ctx->T / v_t_vec;
    int total_items = ctx->HD * token_groups;
    return (total_items + ctx->THREADS - 1) / ctx->THREADS;
}

/* ==========================================================================
 * ckc_gfx942_attn2d_issue_v_transposed  (vehicle (a), lines 2658-2732)
 *
 * VECTOR-store transpose of V into V_lds[0, col, token] (f16). Per work item
 * ``item = call*THREADS + tid`` -> head-dim row col = item // token_groups and
 * token group g = item % token_groups (token_base = g*V_T_VEC). Reads V_T_VEC
 * consecutive tokens at head-dim col from HBM (token-strided sync gather) and
 * writes them as ONE contiguous vector ds_write.
 * ========================================================================== */
void ckc_gfx942_attn2d_issue_v_transposed(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                          ckc_value_t* kv_tile_idx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    int V_T_VEC = ckc__v_t_vec(ctx);
    int token_groups = ctx->T / V_T_VEC;
    int total_items = ctx->HD * token_groups;
    int V_T_ITEMS_PER_THREAD = ckc__v_t_items_per_thread(ctx, V_T_VEC);
    bool need_item_guard = (total_items % ctx->THREADS) != 0;
    ckc_value_t* max_seq_prefix_len = ckc__max_seq_prefix_len(ctx);
    int call;
    int j;

    /* _tg = b.const_i32(_v_t_token_groups) */
    ckc_value_t* tg = ckc_b_const_i32(b, token_groups);
    /* zero_h = b.cast_f32_to(b.const_f32(0.0), dtype) */
    ckc_value_t* zero_h = ckc_b_cast_f32_to(b, ckc_b_const_f32(b, 0.0), dtype);

    for(call = 0; call < V_T_ITEMS_PER_THREAD; ++call)
    {
        /* item = call*THREADS + tid */
        ckc_value_t* item = ckc_b_add(
            b, ckc_b_mul(b, ckc_b_const_i32(b, call), ckc_b_const_i32(b, ctx->THREADS)), ctx->tid);
        if(need_item_guard)
        {
            /* item = select(item < total_items, item, 0) */
            item = ckc_b_select(b,
                                ckc_b_cmp_lt(b, item, ckc_b_const_i32(b, total_items)),
                                item,
                                ckc_b_const_i32(b, 0));
        }
        ckc_value_t* col        = ckc_b_div(b, item, tg);
        ckc_value_t* g          = ckc_b_mod(b, item, tg);
        ckc_value_t* token_base = ckc_b_mul(b, g, ckc_b_const_i32(b, V_T_VEC));
        /* tile_tok_base = kv_tile_idx * T */
        ckc_value_t* tile_tok_base = ckc_b_mul(b, kv_tile_idx, ckc_b_const_i32(b, ctx->T));

        ckc_value_t* elems[4]; /* V_T_VEC <= 4 */
        for(j = 0; j < V_T_VEC; ++j)
        {
            ckc_value_t* token_j  = ckc_b_add(b, token_base, ckc_b_const_i32(b, j));
            /* linear_j = token_j * HD + col */
            ckc_value_t* linear_j = ckc_b_add(b, ckc_b_mul(b, token_j, ckc_b_const_i32(b, ctx->HD)), col);
            ckc_value_t* valid_j  = NULL;
            ckc_value_t* voff_j =
                ckc__paged_kv_offset(ctx, kv_tile_idx, linear_j, ctx->kv_head_idx, &valid_j);
            /* in_range = (tile_tok_base + token_j) < max_seq_prefix_len */
            ckc_value_t* in_range = ckc_b_cmp_lt(
                b, ckc_b_add(b, tile_tok_base, token_j), max_seq_prefix_len);
            if(valid_j != NULL)
                in_range = ckc_b_land(b, in_range, valid_j);
            ckc_value_t* safe_voff_j = ckc_b_select(b, in_range, voff_j, ckc_b_const_i32(b, 0));
            ckc_value_t* safe_elem_j =
                (ctx->KV_BYTES != 1)
                    ? ckc_b_div(b, safe_voff_j, ckc_b_const_i32(b, ctx->KV_BYTES))
                    : safe_voff_j;
            ckc_value_t* v1 = ckc_b_global_load(b, ctx->value, safe_elem_j, dtype, 2);
            v1 = ckc_b_select(b, in_range, v1, zero_h);
            elems[j] = v1;
        }
        /* v_vec = b.vec_pack(elems, dtype) */
        ckc_value_t* v_vec = ckc_b_vec_pack(b, elems, V_T_VEC, dtype);
        /* _v_t_store(col, token_base, v_vec, V_T_VEC) */
        ckc_gfx942_attn2d_v_t_store(ctx, col, token_base, v_vec, V_T_VEC);
    }
}

/* ==========================================================================
 * cfvst (vehicle (c)) payload + inner closures.
 * ========================================================================== */

/* One loaded 2x2 block: the Python tuple (d0, d1, t0, x0, x1). */
typedef struct ckc__cfvst_blk
{
    ckc_value_t* d0;
    ckc_value_t* d1;
    ckc_value_t* t0;
    ckc_value_t* x0; /* i32 = (V[t0,d0], V[t0,d1]) */
    ckc_value_t* x1; /* i32 = (V[t1,d0], V[t1,d1]) */
} ckc__cfvst_blk_t;

/* The arena-owned payload _cfvst_load_v_regs returns + _cfvst_store_v_regs
 * consumes. kv_tile_idx is retained so the store path can re-derive the prezero
 * coverage loop bound (it uses only ctx constants there). */
typedef struct ckc__cfvst_payload
{
    ckc__cfvst_blk_t* blocks;
    int count;
} ckc__cfvst_payload_t;

/* _load_token_row_pair(t_row, d0) -> i32 = (V[t_row,d0], V[t_row,d0+1]), bounded
 * (lines 2773-2832). kv_tile_idx + max_seq_prefix_len are passed via ctx-private
 * statics threaded through _cfvst_load_v_regs (see below); to keep this a free
 * function with the header-declared signature, it recomputes them from ctx. */
ckc_value_t* ckc_gfx942_attn2d_load_token_row_pair(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                   ckc_value_t* t_row,
                                                   ckc_value_t* d0)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    const ckc_type_t* I32 = ckc_i32();
    ckc_value_t* kv_tile_idx = ctx->kv_tile_iv; /* current tile (set by caller) */
    ckc_value_t* zero_h = ckc_b_cast_f32_to(b, ckc_b_const_f32(b, 0.0), dtype);
    ckc_value_t* zero_i32 = ckc_b_const_i32(b, 0);
    ckc_value_t* max_seq_prefix_len = ckc__max_seq_prefix_len(ctx);
    ckc_value_t* tile_tok_base = ckc_b_mul(b, kv_tile_idx, ckc_b_const_i32(b, ctx->T));

    /* linear = t_row * HD + d0 */
    ckc_value_t* linear = ckc_b_add(b, ckc_b_mul(b, t_row, ckc_b_const_i32(b, ctx->HD)), d0);
    ckc_value_t* valid = NULL;
    ckc_value_t* voff = ckc__paged_kv_offset(ctx, kv_tile_idx, linear, ctx->kv_head_idx, &valid);
    ckc_value_t* in_range =
        ckc_b_cmp_lt(b, ckc_b_add(b, tile_tok_base, t_row), max_seq_prefix_len);
    if(valid != NULL)
        in_range = ckc_b_land(b, in_range, valid);
    ckc_value_t* safe_voff = ckc_b_select(b, in_range, voff, zero_i32);
    ckc_value_t* safe_elem = (ctx->KV_BYTES != 1)
                                 ? ckc_b_div(b, safe_voff, ckc_b_const_i32(b, ctx->KV_BYTES))
                                 : safe_voff;
    ckc_value_t* v2;

    if(ctx->CFV_STORE_SEPOFF)
    {
        /* Separate descriptor call for (t_row, d0+1) instead of voff+1. */
        ckc_value_t* linear1 = ckc_b_add(
            b, ckc_b_mul(b, t_row, ckc_b_const_i32(b, ctx->HD)), ckc_b_add(b, d0, ckc_b_const_i32(b, 1)));
        ckc_value_t* valid1 = NULL;
        ckc_value_t* voff1 =
            ckc__paged_kv_offset(ctx, kv_tile_idx, linear1, ctx->kv_head_idx, &valid1);
        (void)valid1;
        ckc_value_t* safe_voff1 = ckc_b_select(b, in_range, voff1, zero_i32);
        ckc_value_t* safe_elem1 =
            (ctx->KV_BYTES != 1)
                ? ckc_b_div(b, safe_voff1, ckc_b_const_i32(b, ctx->KV_BYTES))
                : safe_voff1;
        ckc_value_t* e0 = ckc_b_global_load(b, ctx->value, safe_elem, dtype, 2);
        ckc_value_t* e1 = ckc_b_global_load(b, ctx->value, safe_elem1, dtype, 2);
        e0 = ckc_b_select(b, in_range, e0, zero_h);
        e1 = ckc_b_select(b, in_range, e1, zero_h);
        ckc_value_t* pk[2] = {e0, e1};
        v2 = ckc_b_vec_pack(b, pk, 2, dtype);
    }
    else if(ctx->CFV_STORE_SCALAR_LOAD)
    {
        /* Two scalar n=1 loads + manual pack. */
        ckc_value_t* e0 = ckc_b_global_load(b, ctx->value, safe_elem, dtype, 2);
        ckc_value_t* e1 = ckc_b_global_load(
            b, ctx->value, ckc_b_add(b, safe_elem, ckc_b_const_i32(b, 1)), dtype, 2);
        e0 = ckc_b_select(b, in_range, e0, zero_h);
        e1 = ckc_b_select(b, in_range, e1, zero_h);
        ckc_value_t* pk[2] = {e0, e1};
        v2 = ckc_b_vec_pack(b, pk, 2, dtype);
    }
    else
    {
        v2 = ckc_b_global_load_vN(b, ctx->value, safe_elem, dtype, 2, 4);
        /* Zero masked rows (whole pair belongs to one token). */
        ckc_value_t* zpk[2] = {zero_h, zero_h};
        ckc_value_t* zero_vec = ckc_b_vec_pack(b, zpk, 2, dtype);
        v2 = ckc_b_select(b, in_range, v2, zero_vec);
    }
    return ckc_b_bitcast(b, v2, I32);
}

/* _cfvst_block_coords(blk) -> (d0, d1, t0, t1)  (lines 2834-2844).
 * out_a[0..1] = (d0, d1); out_b[0..1] = (t0, t1). */
void ckc_gfx942_attn2d_cfvst_block_coords(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                          ckc_value_t* blk,
                                          ckc_value_t** out_a,
                                          ckc_value_t** out_b)
{
    ckc_ir_builder_t* b = ctx->b;
    int dim_pairs = ctx->HD / 2;
    ckc_value_t* dp = ckc_b_const_i32(b, dim_pairs);
    /* tg = blk // _dp ; dg = blk % _dp */
    ckc_value_t* tg = ckc_b_div(b, blk, dp);
    ckc_value_t* dg = ckc_b_mod(b, blk, dp);
    ckc_value_t* t0 = ckc_b_mul(b, tg, ckc_b_const_i32(b, 2));
    ckc_value_t* t1 = ckc_b_add(b, t0, ckc_b_const_i32(b, 1));
    ckc_value_t* d0 = ckc_b_mul(b, dg, ckc_b_const_i32(b, 2));
    ckc_value_t* d1 = ckc_b_add(b, d0, ckc_b_const_i32(b, 1));
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
void* ckc_gfx942_attn2d_cfvst_load_v_regs(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                          ckc_value_t* kv_tile_idx)
{
    ckc_ir_builder_t* b = ctx->b;
    int V_T_VEC = ckc__v_t_vec(ctx);
    int V_T_ITEMS_PER_THREAD = ckc__v_t_items_per_thread(ctx, V_T_VEC);
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
    ckc_value_t* saved_iv = ctx->kv_tile_iv;
    ctx->kv_tile_iv = kv_tile_idx;

    ckc__cfvst_payload_t* payload =
        (ckc__cfvst_payload_t*)ckc_arena_alloc(&b->arena, sizeof(ckc__cfvst_payload_t));
    if(payload == NULL)
    {
        ctx->kv_tile_iv = saved_iv;
        return NULL;
    }
    payload->blocks =
        (ckc__cfvst_blk_t*)ckc_arena_alloc(&b->arena, sizeof(ckc__cfvst_blk_t) * (size_t)(V_T_ITEMS_PER_THREAD > 0 ? V_T_ITEMS_PER_THREAD : 1));
    payload->count = 0;
    if(payload->blocks == NULL)
    {
        ctx->kv_tile_iv = saved_iv;
        return NULL;
    }

    for(call = 0; call < V_T_ITEMS_PER_THREAD; ++call)
    {
        /* blk = call*THREADS + tid */
        ckc_value_t* blk = ckc_b_add(
            b, ckc_b_mul(b, ckc_b_const_i32(b, call), ckc_b_const_i32(b, ctx->THREADS)), ctx->tid);
        if(need_item_guard)
        {
            blk = ckc_b_select(b,
                               ckc_b_cmp_lt(b, blk, ckc_b_const_i32(b, total_blocks)),
                               blk,
                               ckc_b_const_i32(b, 0));
        }
        ckc_value_t* dd[2];
        ckc_value_t* tt[2];
        ckc_gfx942_attn2d_cfvst_block_coords(ctx, blk, dd, tt);
        ckc_value_t* d0 = dd[0];
        ckc_value_t* d1 = dd[1];
        ckc_value_t* t0 = tt[0];
        ckc_value_t* t1 = tt[1];
        /* x0 = (V[t0,d0], V[t0,d1]) ; x1 = (V[t1,d0], V[t1,d1]) */
        ckc_value_t* x0 = ckc_gfx942_attn2d_load_token_row_pair(ctx, t0, d0);
        ckc_value_t* x1 = ckc_gfx942_attn2d_load_token_row_pair(ctx, t1, d0);
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
void ckc_gfx942_attn2d_cfvst_store_v_regs(ckc_gfx942_attn2d_build_ctx_t* ctx, void* payload_v)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_type_t* dtype = ctx->dtype;
    const ckc_type_t* vec2 = ckc_vector_type(b, dtype, 2);
    ckc__cfvst_payload_t* payload = (ckc__cfvst_payload_t*)payload_v;
    ckc_value_t* zero_h = ckc_b_cast_f32_to(b, ckc_b_const_f32(b, 0.0), dtype);
    int i;
    (void)zero_h;

    if(payload == NULL)
        return;

    if(ctx->CFV_STORE_PREZERO)
    {
        /* DIAGNOSTIC: pre-zero every V_lds slot (dim x token). */
        ckc_for_t pz = ckc_b_scf_for(
            b, ctx->tid, ckc_b_const_i32(b, ctx->HD * ctx->T), ckc_b_const_i32(b, ctx->THREADS), "vpz");
        ckc_b_region_enter(b, pz.body);
        {
            ckc_value_t* _pd  = ckc_b_div(b, pz.iv, ckc_b_const_i32(b, ctx->T));
            ckc_value_t* _ptk = ckc_b_mod(b, pz.iv, ckc_b_const_i32(b, ctx->T));
            ckc_gfx942_attn2d_v_t_store(ctx, _pd, _ptk, zero_h, 1);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
    }

    for(i = 0; i < payload->count; ++i)
    {
        ckc_value_t* d0 = payload->blocks[i].d0;
        ckc_value_t* d1 = payload->blocks[i].d1;
        ckc_value_t* t0 = payload->blocks[i].t0;
        ckc_value_t* x0 = payload->blocks[i].x0;
        ckc_value_t* x1 = payload->blocks[i].x1;

        if(ctx->CFV_STORE_SCATTER)
        {
            /* DIAGNOSTIC: element-wise scatter (no perm). */
            ckc_value_t* x0v = ckc_b_bitcast(b, x0, vec2); /* (V[t0,d0],V[t0,d1]) */
            ckc_value_t* x1v = ckc_b_bitcast(b, x1, vec2); /* (V[t1,d0],V[t1,d1]) */
            ckc_value_t* t1  = ckc_b_add(b, t0, ckc_b_const_i32(b, 1));
            ckc_value_t* v_t0_d0 = ckc_b_vec_extract(b, x0v, 0);
            ckc_value_t* v_t0_d1 = ckc_b_vec_extract(b, x0v, 1);
            ckc_value_t* v_t1_d0 = ckc_b_vec_extract(b, x1v, 0);
            ckc_value_t* v_t1_d1 = ckc_b_vec_extract(b, x1v, 1);
            ckc_gfx942_attn2d_v_t_store(ctx, d0, t0, v_t0_d0, 1);
            ckc_gfx942_attn2d_v_t_store(ctx, d0, t1, v_t1_d0, 1);
            ckc_gfx942_attn2d_v_t_store(ctx, d1, t0, v_t0_d1, 1);
            ckc_gfx942_attn2d_v_t_store(ctx, d1, t1, v_t1_d1, 1);
        }
        else
        {
            /* 2x2 transpose: each output i32 = 2 consecutive tokens at one dim. */
            ckc_value_t* row_d0 =
                ckc_b_perm_b32(b, x0, x1, ckc_b_const_i32(b, 0x01000504)); /* (t0,t1)@d0 */
            ckc_value_t* row_d1 =
                ckc_b_perm_b32(b, x0, x1, ckc_b_const_i32(b, 0x03020706)); /* (t0,t1)@d1 */
            /* ONE contiguous 2-half ds_write per dim row (token is inner). */
            ckc_gfx942_attn2d_v_t_store(ctx, d0, t0, ckc_b_bitcast(b, row_d0, vec2), 2);
            ckc_gfx942_attn2d_v_t_store(ctx, d1, t0, ckc_b_bitcast(b, row_d1, vec2), 2);
        }
    }
}

/* _issue_v_transposed_store(kv_tile_idx)  (lines 2900-2903).
 * Register-load + perm_b32 transpose + contiguous LDS store of V into
 * V_lds[0, dim, token] (vehicle (c)). */
void ckc_gfx942_attn2d_issue_v_transposed_store(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                ckc_value_t* kv_tile_idx)
{
    void* payload = ckc_gfx942_attn2d_cfvst_load_v_regs(ctx, kv_tile_idx);
    ckc_gfx942_attn2d_cfvst_store_v_regs(ctx, payload);
}
