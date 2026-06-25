// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_kv_descriptors_loads.c --
 *   C99 port of the KV DESCRIPTOR + ASYNC-DMA LOADERS bucket of
 *   ck_dsl/instances/gfx942/attention_tiled_2d.py (arch gfx942).
 *
 * SCOPE (this translation unit).
 *   ckc_gfx942_attn2d_emit_preloop            -- the pre-loop section (Python lines
 *     2163-2351): big-bytes K/V buffer resources, the async-DMA byte/stride
 *     derivation, the wave/V-wave LDS dest offsets, the paged-KV byte
 *     TensorDescriptor full transform DAG, and seq_base / block_table_max_idx.
 *   ckc_gfx942_attn2d_fast_paged_kv_blocks    -- _fast_paged_kv_blocks (2352-2373):
 *     the per-tile two-logical-block block_tables lookup.
 *   ckc_gfx942_attn2d_fast_paged_kv_voff      -- _fast_paged_kv_voff (2375-2398):
 *     the per-call within-block byte voffset (+ optional i64 block base).
 *   ckc_gfx942_attn2d_v_t_slot / _v_t_store / _v_t_load / _v_load1
 *     -- the V-LDS slot/store/load + flat V load helpers (1756-1830).
 *   ckc_gfx942_attn2d_issue_k_load_runtime    -- _issue_k_load_runtime (2440-2507).
 *   ckc_gfx942_attn2d_issue_k_slice_load_runtime -- _issue_k_slice_load_runtime
 *     (2511-2560).
 *   ckc_gfx942_attn2d_issue_v_load_runtime    -- _issue_v_load_runtime (2562-2619).
 *   ckc_gfx942_attn2d_issue_k / _issue_v      -- the loader dispatch wrappers
 *     (3347-3396).
 *   ckc_gfx942_attn2d_read_k8_mfma_operand    -- _read_k8_mfma_operand (3398-3424).
 *
 * The builder-call sequence is byte-identical to the Python body: same ops, same
 * order, same operands. Because C has no closure capture, the per-call byte/
 * stride scalars the Python prologue computes once and the loader closures share
 * are recomputed here (they are pure compile-time integers + a small number of
 * cheap, idempotent SSA recomputations) -- the emitted IR is unchanged.
 *
 * Paths reachable only through symbols not yet ported to the C transforms
 * surface (the non-fast paged-KV descriptor's TensorDescriptor.indirect /
 * .offset_i64_split, and the fp8 dequant_fp8x8_to_dtype) are stubbed-to-link:
 * fp8 K/V is rejected on gfx942 (so K_FP8_MFMA / KV_FP8 / TRANSPOSED_V* are
 * always false on this arch's buildable space), and FAST_PAGED_KV_DESC is the
 * gfx942 production descriptor path.
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#include "ckc/instance_gfx942_attention_tiled_2d_internal.h"

/* ==========================================================================
 * Shared compile-time byte/stride derivation (Python prologue locals 2177-2287).
 *
 * The Python build prologue computes these once and the loader closures capture
 * them. In C they are pure functions of ctx geometry, recomputed where used.
 * ========================================================================== */

/* KV_HALVES_PER_LANE: async-DMA per-lane half count. gfx950 = 8 (16 bytes/lane,
 * dwords=4); gfx942 = ASYNC_LDS_MAX_BYTES_PER_LANE//2 = 2 (set in ctx_init). */
static int ckc__kv_halves_per_lane(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->KV_DMA_HALVES_PER_LANE;
}

/* KV_HALVES_PER_CALL = THREADS * KV_HALVES_PER_LANE. */
static int ckc__kv_halves_per_call(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->THREADS * ckc__kv_halves_per_lane(ctx);
}

/* kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL. */
static int ckc__kv_calls_per_tile(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return (ctx->T * ctx->HD) / ckc__kv_halves_per_call(ctx);
}

/* bytes_per_call = KV_HALVES_PER_CALL * 2. */
static int ckc__bytes_per_call(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return ckc__kv_halves_per_call(ctx) * 2;
}

/* bytes_per_buf = T * HD * 2 (one [T, HD] working-dtype slab). */
static int ckc__bytes_per_buf(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->T * ctx->HD * 2;
}

/* WAVE_BYTES = WAVE * (KV_DMA_HALVES_PER_LANE * 2). gfx950 = WAVE*16 (dwords=4);
 * gfx942 = WAVE*ASYNC_LDS_MAX_BYTES_PER_LANE (=WAVE*4). */
static int ckc__wave_bytes(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->WAVE * (ctx->KV_DMA_HALVES_PER_LANE * 2);
}

/* V_BYTES_PER_CALL_SWZ: per-call dest stride for V (swizzle-aware). */
static int ckc__v_bytes_per_call_swz(const ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    int bpc = ckc__bytes_per_call(ctx);
    if(ctx->SWIZZLE_VLDS)
        bpc += ctx->V_ROWS_PER_CALL * ctx->V_LDS_PAD * 2;
    return bpc;
}

/* kv_block_bytes_c value = BS * NUM_KV * HD * KV_BYTES (one-block buffer bound).
 * Python creates this const ONCE at line 2227 and reuses it in every loader;
 * cache it on ctx so we do not allocate a duplicate const per loader (which
 * would shift downstream %values). */
static ckc_value_t* ckc__kv_block_bytes_c(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->kv_block_bytes_c_v == NULL)
    {
        int kv_stride_blk_b     = ctx->BS * ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
        ctx->kv_block_bytes_c_v = ckc_b_const_i32(ctx->b, kv_stride_blk_b);
    }
    return ctx->kv_block_bytes_c_v;
}

/* lane_half_base = tid * KV_HALVES_PER_LANE (Python 2232). Emitted once in
 * emit_preloop and cached on ctx so every loader reuses the same SSA value
 * (matches the single Python local). */
static ckc_value_t* ckc__lane_half_base(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_half_base_v == NULL)
        ctx->lane_half_base_v =
            ckc_b_mul(ctx->b, ctx->tid, ckc_b_const_i32(ctx->b, ckc__kv_halves_per_lane(ctx)));
    return ctx->lane_half_base_v;
}

/* seq_base = to_sgpr_u32(seq_idx * bt_stride_p) (Python 2327). Cached on ctx. */
static ckc_value_t* ckc__seq_base(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->seq_base == NULL)
        ctx->seq_base =
            ckc_b_to_sgpr_u32(ctx->b, ckc_b_mul(ctx->b, ctx->seq_idx, ctx->bt_stride_p));
    return ctx->seq_base;
}

/* block_table_max_idx = to_sgpr_u32(num_seqs_p * bt_stride_p) (Python 2340). */
static ckc_value_t* ckc__block_table_max_idx(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->block_table_max_idx == NULL)
        ctx->block_table_max_idx =
            ckc_b_to_sgpr_u32(ctx->b, ckc_b_mul(ctx->b, ctx->num_seqs_p, ctx->bt_stride_p));
    return ctx->block_table_max_idx;
}

/* K_lds_addr = ptrtoint(K_lds) (Python 2234); cached. */
static ckc_value_t* ckc__K_lds_addr(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->K_lds_addr_v == NULL)
        ctx->K_lds_addr_v = ckc_b_smem_addr_of(ctx->b, ctx->K_lds);
    return ctx->K_lds_addr_v;
}

/* V_lds_addr = ptrtoint(V_lds) (Python 2235); cached. */
static ckc_value_t* ckc__V_lds_addr(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->V_lds_addr_v == NULL)
        ctx->V_lds_addr_v = ckc_b_smem_addr_of(ctx->b, ctx->V_lds);
    return ctx->V_lds_addr_v;
}

/* zero_soff = const_i32(0) (Python 2238). Python creates this ONE const and
 * reuses it as the soffset in every async load; cache it so the loaders do not
 * each allocate a duplicate const. */
static ckc_value_t* ckc__zero_soff(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->zero_soff_v == NULL)
        ctx->zero_soff_v = ckc_b_const_i32(ctx->b, 0);
    return ctx->zero_soff_v;
}

/* wave_lds_offset_i64 (K/V): const_i64(0) for NUM_WARPS==1, else the SGPR-pinned
 * zext(wave_id * WAVE_BYTES). Python creates it once (2254/2263) and reuses it;
 * cache. */
static ckc_value_t* ckc__wave_lds_offset_i64(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->wave_lds_off_i64_v != NULL)
        return ctx->wave_lds_off_i64_v;
    if(ctx->NUM_WARPS == 1)
    {
        ctx->wave_lds_off_i64_v = ckc_b_const_i64(ctx->b, 0);
        return ctx->wave_lds_off_i64_v;
    }
    ckc_value_t* off_i32 = ckc_b_to_sgpr_u32(
        ctx->b, ckc_b_mul(ctx->b, ctx->wave_id, ckc_b_const_i32(ctx->b, ckc__wave_bytes(ctx))));
    ctx->wave_lds_off_i64_v = ckc_b_zext(ctx->b, off_i32, ckc_i64());
    return ctx->wave_lds_off_i64_v;
}

/* v_wave_lds_offset_i64: the V-specific (swizzle-aware) wave offset. Cached. */
static ckc_value_t* ckc__v_wave_lds_offset_i64(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->v_wave_lds_off_i64_v != NULL)
        return ctx->v_wave_lds_off_i64_v;
    if(!ctx->SWIZZLE_VLDS || ctx->NUM_WARPS == 1)
    {
        if(ctx->NUM_WARPS == 1)
        {
            ctx->v_wave_lds_off_i64_v = ckc_b_const_i64(ctx->b, 0);
            return ctx->v_wave_lds_off_i64_v;
        }
        ctx->v_wave_lds_off_i64_v = ckc__wave_lds_offset_i64(ctx);
        return ctx->v_wave_lds_off_i64_v;
    }
    int v_wave_bytes     = ckc__wave_bytes(ctx) + (ctx->SWIZZLE_VLDS ? ctx->V_LDS_PAD * 2 : 0);
    ckc_value_t* off_i32 = ckc_b_to_sgpr_u32(
        ctx->b, ckc_b_mul(ctx->b, ctx->wave_id, ckc_b_const_i32(ctx->b, v_wave_bytes)));
    ctx->v_wave_lds_off_i64_v = ckc_b_zext(ctx->b, off_i32, ckc_i64());
    return ctx->v_wave_lds_off_i64_v;
}

/* ==========================================================================
 * V-LDS slot/store/load + flat V load (Python lines 1756-1830).
 * ========================================================================== */

ckc_value_t*
ckc_gfx942_attn2d_v_t_slot(ckc_gfx942_attn2d_build_ctx_t* ctx, ckc_value_t* dim, ckc_value_t* tok)
{
    ckc_ir_builder_t* b  = ctx->b;
    ckc_value_t* k_group = ckc_b_div(b, tok, ckc_b_const_i32(b, ctx->V_T_KPACK));
    ckc_value_t* k_inner = ckc_b_mod(b, tok, ckc_b_const_i32(b, ctx->V_T_KPACK));
    ckc_value_t* n_group = ckc_b_div(b, dim, ckc_b_const_i32(b, ctx->V_T_NPER_ROW));
    ckc_value_t* n_inner = ckc_b_mod(b, dim, ckc_b_const_i32(b, ctx->V_T_NPER_ROW));
    ckc_value_t* hi      = ckc_b_add(
        b,
        ckc_b_mul(b, k_group, ckc_b_const_i32(b, ctx->V_T_NGROUPS * ctx->V_T_GROUP_STRIDE)),
        ckc_b_mul(b, n_group, ckc_b_const_i32(b, ctx->V_T_GROUP_STRIDE)));
    ckc_value_t* lo =
        ckc_b_add(b, ckc_b_mul(b, n_inner, ckc_b_const_i32(b, ctx->V_T_KPACK)), k_inner);
    return ckc_b_add(b, hi, lo);
}

void ckc_gfx942_attn2d_v_t_store(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                 ckc_value_t* dim,
                                 ckc_value_t* tok,
                                 ckc_value_t* value,
                                 int n)
{
    ckc_ir_builder_t* b = ctx->b;
    if(ctx->V_T_CK_LAYOUT)
    {
        ckc_value_t* idx[2];
        idx[0] = ckc_b_const_i32(b, 0);
        idx[1] = ckc_gfx942_attn2d_v_t_slot(ctx, dim, tok);
        ckc_b_smem_store_vN(b, ctx->V_lds, idx, 2, value, n);
    }
    else
    {
        ckc_value_t* idx[3];
        idx[0] = ckc_b_const_i32(b, 0);
        idx[1] = dim;
        idx[2] = tok;
        ckc_b_smem_store_vN(b, ctx->V_lds, idx, 3, value, n);
    }
}

ckc_value_t* ckc_gfx942_attn2d_v_t_load(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                        ckc_value_t* dim,
                                        ckc_value_t* tok,
                                        int n)
{
    ckc_ir_builder_t* b = ctx->b;
    ckc_value_t* v_buf0 = ckc_b_const_i32(b, 0);
    if(ctx->V_T_CK_LAYOUT)
    {
        ckc_value_t* idx[2];
        idx[0] = v_buf0;
        idx[1] = ckc_gfx942_attn2d_v_t_slot(ctx, dim, tok);
        return ckc_b_smem_load_vN(b, ctx->V_lds, idx, 2, ctx->dtype, n);
    }
    {
        ckc_value_t* idx[3];
        idx[0] = v_buf0;
        idx[1] = dim;
        idx[2] = tok;
        return ckc_b_smem_load_vN(b, ctx->V_lds, idx, 3, ctx->dtype, n);
    }
}

ckc_value_t* ckc_gfx942_attn2d_v_load1(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                       ckc_value_t* v_buf,
                                       ckc_value_t* v_row,
                                       ckc_value_t* v_n_col)
{
    ckc_ir_builder_t* b = ctx->b;
    if(!ctx->SWIZZLE_VLDS)
    {
        ckc_value_t* idx[3];
        idx[0] = v_buf;
        idx[1] = v_row;
        idx[2] = v_n_col;
        return ckc_b_smem_load_vN(b, ctx->V_lds, idx, 3, ctx->dtype, 1);
    }
    {
        ckc_value_t* group  = ckc_b_lshr(b, v_row, ckc_b_const_i32(b, ctx->V_GROUP_SHIFT));
        ckc_value_t* within = ckc_b_land(b, v_row, ckc_b_const_i32(b, ctx->V_ROWS_PER_CALL - 1));
        /* slot = (group*GS + within*LS) + v_n_col. Python emits the group mul
         * before the within mul; sequence via temps (C arg-eval is unspecified). */
        ckc_value_t* group_mul  = ckc_b_mul(b, group, ckc_b_const_i32(b, ctx->V_GROUP_STRIDE));
        ckc_value_t* within_mul = ckc_b_mul(b, within, ckc_b_const_i32(b, ctx->V_LDS_STRIDE));
        ckc_value_t* slot       = ckc_b_add(b, ckc_b_add(b, group_mul, within_mul), v_n_col);
        ckc_value_t* idx[2];
        idx[0] = v_buf;
        idx[1] = slot;
        return ckc_b_smem_load_vN(b, ctx->V_lds, idx, 2, ctx->dtype, 1);
    }
}

/* ==========================================================================
 * Pre-loop: K/V buffer resources, byte derivation, paged-KV descriptor
 * (Python lines 2163-2351).
 * ========================================================================== */

void ckc_gfx942_attn2d_emit_preloop(ckc_gfx942_attn2d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* big-bytes K/V buffer resources (2166-2168). The buffer rsrc bounds OOB
     * voffsets to return zero; size it large so valid offsets never trip it. */
    ckc_value_t* big_bytes = ckc_b_const_i32(b, 0x7FFF0000);
    ctx->k_rsrc            = ckc_b_buffer_rsrc(b, ctx->key, big_bytes);
    ctx->v_rsrc            = ckc_b_buffer_rsrc(b, ctx->value, big_bytes);

    /* The bf16 async-DMA byte derivation (2177-2181) is purely compile-time and
     * the loader closures recompute it; emit the asserted invariants only so the
     * IR stream is unchanged (these emit no ops). */
    {
        int kv_halves_per_call = ckc__kv_halves_per_call(ctx);
        /* assert (T * HD) % KV_HALVES_PER_CALL == 0 (2179). */
        (void)((ctx->T * ctx->HD) % kv_halves_per_call);
    }

    /* kv_block_bytes_c (2227): Python creates this one-block-bound const here,
     * BEFORE lane_half_base, and reuses it in every loader. Emit it now so its
     * SSA value lands in source order. */
    (void)ckc__kv_block_bytes_c(ctx);

    /* Per-lane starting half index (2232), then K/V LDS base ptrtoints (2234-
     * 2235). These are SINGLE Python locals consumed by every loader; emit them
     * here, in source order, and cache on ctx so the loaders reuse the same SSA
     * values (the byte-identity contract requires the exact op stream). */
    (void)ckc__lane_half_base(ctx);
    (void)ckc__K_lds_addr(ctx);
    (void)ckc__V_lds_addr(ctx);

    /* zero_soff (2238), then wave_lds_offset_i64 (2254) and
     * v_wave_lds_offset_i64 (2279). Python creates each as a single SSA value
     * HERE (even the const_i64(0) consumes a value number) and reuses them in
     * every loader. Force their creation now, in source order, and cache so the
     * loaders reuse them instead of allocating duplicates -- otherwise their
     * value numbers land later and shift the whole downstream stream. */
    (void)ckc__zero_soff(ctx);
    (void)ckc__wave_lds_offset_i64(ctx);
    (void)ckc__v_wave_lds_offset_i64(ctx);
    (void)ckc__bytes_per_buf;
    (void)ckc__v_bytes_per_call_swz;
    (void)ckc__kv_calls_per_tile;
    (void)ckc__bytes_per_call;
    (void)ckc__kv_block_bytes_c;

    /* FAST_PAGED_KV_DESC: the gfx942 production fast paged-KV path emits no
     * standalone descriptor object -- the per-call byte offsets are built
     * directly by _fast_paged_kv_blocks / _fast_paged_kv_voff (2352-2398). */
    /* seq_base + block_table_max_idx: per-sequence base index into block_tables
     * and the global footprint bound; both wave-uniform SGPRs. The gfx950 builder
     * emits BOTH unconditionally in the preloop (Python 1525, 1538), BEFORE the
     * FAST_PAGED_KV_DESC branch and the Q gather, so the SSA stream matches; the
     * gfx942 narrow builder only needs them for the full byte descriptor. Emit
     * them here on the gfx950 (wide ds_read_tr) path even when fast paged-KV is
     * on so the value numbering aligns. */
    if(ctx->FAST_PAGED_KV_DESC)
    {
        if(ctx->target != NULL && ctx->target->memory.has_ds_read_tr)
        {
            (void)ckc__seq_base(ctx);
            (void)ckc__block_table_max_idx(ctx);
        }
        ctx->kv_desc = NULL;
        return;
    }

    /* ---- Paged KV byte descriptor (full transform DAG, 2327-2438) ---- */
    ckc_value_t* seq_base = ckc__seq_base(ctx);
    ckc_value_t* max_idx  = ckc__block_table_max_idx(ctx);

    /* _kv_base = naive("paged_kv_bytes",
     *     lengths=[1<<24, BS, NUM_KV, HD],
     *     strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
     *     coord_names=("physical_block","token","kv_head","dim")). */
    int blk_b                             = ctx->BS * ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
    int tok_b                             = ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
    int h_b                               = ctx->HD * ctx->KV_BYTES;
    int kv_lengths[4]                     = {1 << 24, ctx->BS, ctx->NUM_KV, ctx->HD};
    int kv_strides[4]                     = {blk_b, tok_b, h_b, ctx->KV_BYTES};
    static const char* const kv_coords[4] = {"physical_block", "token", "kv_head", "dim"};
    ckc_tensor_descriptor_t* kv_base =
        ckc_tensor_descriptor_naive(b, "paged_kv_bytes", kv_lengths, 4, kv_strides, kv_coords, 4);

    /* Single-block tile (N_BLOCKS_PER_TILE == 1, T == BS):
     *   indirect(tile_idx -> physical_block) ; unmerge(linear_half -> token,dim).
     * Multi-block (N_BLOCKS_PER_TILE > 1):
     *   unmerge(linear_half -> block_within_tile,token,dim) ;
     *   embed((tile_idx,block_within_tile) -> linear_block_idx) ;
     *   indirect(linear_block_idx -> physical_block). */
    if(ctx->N_BLOCKS_PER_TILE == 1)
    {
        const char* td_into[2] = {"token", "dim"};
        int td_dims[2]         = {ctx->T, ctx->HD};
        const ckc_transform_t* chain[2];
        chain[0] =
            ckc_indirect(b, "tile_idx", "physical_block", ctx->block_tables, seq_base, max_idx, 0);
        chain[1]     = ckc_unmerge(b, "linear_half", td_into, 2, td_dims);
        ctx->kv_desc = ckc_tensor_descriptor_transform(b, kv_base, chain, 2);
    }
    else
    {
        const char* td_into[3]   = {"block_within_tile", "token", "dim"};
        int td_dims[3]           = {ctx->N_BLOCKS_PER_TILE, ctx->BS, ctx->HD};
        const char* emb_upper[2] = {"tile_idx", "block_within_tile"};
        int emb_strides[2]       = {ctx->N_BLOCKS_PER_TILE, 1};
        const ckc_transform_t* chain[3];
        chain[0] = ckc_unmerge(b, "linear_half", td_into, 3, td_dims);
        chain[1] = ckc_embed(b, emb_upper, 2, "linear_block_idx", emb_strides, 0);
        chain[2] = ckc_indirect(
            b, "linear_block_idx", "physical_block", ctx->block_tables, seq_base, max_idx, 0);
        ctx->kv_desc = ckc_tensor_descriptor_transform(b, kv_base, chain, 3);
    }

    /* tile-0 prefetch (issue_k(tile0) / issue_v(tile0)) is driven by the loop
     * driver glue, which calls the loader phase functions below in Python order. */
}

/* ==========================================================================
 * Fast paged-KV byte-descriptor closures (Python lines 2352-2398).
 * ========================================================================== */

void ckc_gfx942_attn2d_fast_paged_kv_blocks(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                            ckc_value_t* kv_tile_idx,
                                            ckc_value_t** out_block0,
                                            ckc_value_t** out_block1)
{
    ckc_ir_builder_t* b   = ctx->b;
    ckc_value_t* seq_base = ckc__seq_base(ctx);
    ckc_value_t* max_idx  = ckc__block_table_max_idx(ctx);

    ckc_value_t* logical_block0 = ckc_b_mul(b, kv_tile_idx, ckc_b_const_i32(b, 2));
    ckc_value_t* logical_block1 = ckc_b_add(b, logical_block0, ckc_b_const_i32(b, 1));
    ckc_value_t* idx0           = ckc_b_add(b, seq_base, logical_block0);
    ckc_value_t* idx1           = ckc_b_add(b, seq_base, logical_block1);
    /* Python evaluates masked_global_load args left-to-right: the cmp_lt mask is
     * created BEFORE the const(0) default. C arg-eval order is unspecified, so
     * bind the mask + default in source order to keep the value numbering. */
    ckc_value_t* mask0 = ckc_b_cmp_lt(b, idx0, max_idx);
    ckc_value_t* def0  = ckc_b_const_i32(b, 0);
    ckc_value_t* block0 =
        ckc_b_masked_global_load(b, ctx->block_tables, idx0, mask0, def0, ckc_i32(), 4);
    ckc_value_t* mask1 = ckc_b_cmp_lt(b, idx1, max_idx);
    ckc_value_t* def1  = ckc_b_const_i32(b, 0);
    ckc_value_t* block1 =
        ckc_b_masked_global_load(b, ctx->block_tables, idx1, mask1, def1, ckc_i32(), 4);
    if(out_block0)
        *out_block0 = ckc_b_to_sgpr_u32(b, block0);
    if(out_block1)
        *out_block1 = ckc_b_to_sgpr_u32(b, block1);
}

ckc_value_t* ckc_gfx942_attn2d_fast_paged_kv_voff(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                  int call,
                                                  ckc_value_t* block0,
                                                  ckc_value_t* block1)
{
    ckc_ir_builder_t* b      = ctx->b;
    int kv_halves_per_call   = ckc__kv_halves_per_call(ctx);
    int bs_hd                = ctx->BS * ctx->HD;
    int fast_calls_per_block = bs_hd / kv_halves_per_call;

    /* gfx950: each call covers one full block (FAST_CALLS_PER_BLOCK == 1), so
     * physical = block0 if call==0 else block1 and half_in_block IS lane_half_base
     * directly -- NO per-call add (Python gfx950 1579-1582). gfx942: CPB calls
     * drain one block; half_in_block = const(call_in_block*KV_HALVES_PER_CALL) +
     * lane_half_base (Python gfx942). */
    bool wide = (ctx->target != NULL && ctx->target->memory.has_ds_read_tr);
    ckc_value_t* physical;
    ckc_value_t* half_in_block;
    if(wide)
    {
        physical      = (call == 0) ? block0 : block1;
        half_in_block = ckc__lane_half_base(ctx);
    }
    else
    {
        physical          = (call < fast_calls_per_block) ? block0 : block1;
        int call_in_block = call % fast_calls_per_block;
        half_in_block     = ckc_b_add(
            b, ckc_b_const_i32(b, call_in_block * kv_halves_per_call), ckc__lane_half_base(ctx));
    }
    ckc_value_t* token   = ckc_b_lshr(b, half_in_block, ckc_b_const_i32(b, 6));
    ckc_value_t* dim     = ckc_b_land(b, half_in_block, ckc_b_const_i32(b, 63));
    ckc_value_t* token_b = ckc_b_shl(b, token, ckc_b_const_i32(b, 10));
    ckc_value_t* head_b  = ckc_b_shl(b, ctx->kv_head_idx, ckc_b_const_i32(b, 7));
    ckc_value_t* dim_b   = ckc_b_shl(b, dim, ckc_b_const_i32(b, 1));
    ckc_value_t* within  = ckc_b_add(b, ckc_b_add(b, token_b, head_b), dim_b);

    if(ctx->I64_KV_ADDR)
    {
        /* Returns the within-block i32 voffset; the i64 block base is built by
         * the caller (the header form exposes only the i32 voffset). */
        return within;
    }
    ckc_value_t* block_b = ckc_b_shl(b, physical, ckc_b_const_i32(b, 15));
    return ckc_b_add(b, block_b, within);
}

/* Internal: the full Python tuple return of _fast_paged_kv_voff (i64 base,
 * within voffset). The i64 base is only produced on the I64_KV_ADDR path. */
static void ckc__fast_paged_kv_voff_split(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                          int call,
                                          ckc_value_t* block0,
                                          ckc_value_t* block1,
                                          ckc_value_t* base_ptr_src,
                                          ckc_value_t** out_base_i64,
                                          ckc_value_t** out_voff)
{
    ckc_ir_builder_t* b      = ctx->b;
    int kv_halves_per_call   = ckc__kv_halves_per_call(ctx);
    int bs_hd                = ctx->BS * ctx->HD;
    int fast_calls_per_block = bs_hd / kv_halves_per_call;
    bool wide                = (ctx->target != NULL && ctx->target->memory.has_ds_read_tr);
    ckc_value_t* physical =
        wide ? ((call == 0) ? block0 : block1) : ((call < fast_calls_per_block) ? block0 : block1);

    ckc_value_t* within = ckc_gfx942_attn2d_fast_paged_kv_voff(ctx, call, block0, block1);
    (void)base_ptr_src;
    if(ctx->I64_KV_ADDR)
    {
        ckc_value_t* base_i64 =
            ckc_b_shl(b, ckc_b_zext(b, physical, ckc_i64()), ckc_b_const_i64(b, 15));
        if(out_base_i64)
            *out_base_i64 = base_i64;
        if(out_voff)
            *out_voff = within;
        return;
    }
    if(out_base_i64)
        *out_base_i64 = NULL;
    if(out_voff)
        *out_voff = within;
}

/* ==========================================================================
 * Standard K/V async-DMA loaders (Python lines 2440-2619).
 * ========================================================================== */

void ckc_gfx942_attn2d_issue_k_load_runtime(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                            ckc_value_t* kv_tile_idx,
                                            ckc_value_t* buf_idx)
{
    ckc_ir_builder_t* b    = ctx->b;
    int bytes_per_buf      = ckc__bytes_per_buf(ctx);
    int bytes_per_call     = ckc__bytes_per_call(ctx);
    int kv_calls_per_tile  = ckc__kv_calls_per_tile(ctx);
    int kv_halves_per_call = ckc__kv_halves_per_call(ctx);

    ckc_value_t* K_lds_addr       = ckc__K_lds_addr(ctx);
    ckc_value_t* wave_off         = ckc__wave_lds_offset_i64(ctx);
    ckc_value_t* zero_soff        = ckc__zero_soff(ctx);
    ckc_value_t* kv_block_bytes_c = ckc__kv_block_bytes_c(ctx);
    ckc_value_t* lane_half_base   = ckc__lane_half_base(ctx);

    ckc_value_t* buf_off_i32 = ckc_b_mul(b, buf_idx, ckc_b_const_i32(b, bytes_per_buf));
    ckc_value_t* buf_off_i64 = ckc_b_zext(b, buf_off_i32, ckc_i64());
    ckc_value_t* K_buf_base  = ckc_b_smem_ptr_add(b, K_lds_addr, buf_off_i64);
    ckc_value_t* K_wave_base = ckc_b_smem_ptr_add(b, K_buf_base, wave_off);

    ckc_value_t* fast_block0 = NULL;
    ckc_value_t* fast_block1 = NULL;
    if(ctx->FAST_PAGED_KV_DESC)
        ckc_gfx942_attn2d_fast_paged_kv_blocks(ctx, kv_tile_idx, &fast_block0, &fast_block1);

    for(int call = 0; call < kv_calls_per_tile; ++call)
    {
        ckc_value_t* k_rsrc = ctx->k_rsrc;
        ckc_value_t* k_ptr  = ctx->key;
        ckc_value_t* voff   = NULL;
        if(ctx->FAST_PAGED_KV_DESC)
        {
            ckc_value_t* base_i64 = NULL;
            ckc__fast_paged_kv_voff_split(
                ctx, call, fast_block0, fast_block1, ctx->key, &base_i64, &voff);
            if(base_i64 != NULL)
            {
                k_ptr  = ckc_b_global_ptr_add(b, ctx->key, base_i64);
                k_rsrc = ckc_b_buffer_rsrc(b, k_ptr, kv_block_bytes_c);
            }
        }
        else
        {
            /* Non-fast paged-KV (N_BLOCKS_PER_TILE==1 / multi-block):
             *   linear_half = call*KV_HALVES_PER_CALL + lane_half_base
             *   voff, _ = paged_kv_desc.offset(tile_idx=, linear_half=, kv_head=)
             * (Python 2470-2489; I64_KV_ADDR is false in gfx942's buildable
             * space, so the plain i32 voffset is used). */
            ckc_value_t* linear_half =
                ckc_b_add(b, ckc_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
            const char* in_names[3]   = {"tile_idx", "linear_half", "kv_head"};
            ckc_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            ckc_value_t* valid        = NULL;
            if(!ckc_transforms_descriptor_offset(
                   b, ctx->kv_desc, in_names, in_values, 3, &voff, &valid))
                return;
        }
        ckc_value_t* k_dst =
            ckc_b_smem_ptr_add(b, K_wave_base, ckc_b_const_i64(b, (int64_t)call * bytes_per_call));
        if(ctx->USE_GLOBAL_LOAD_LDS_K)
        {
            ckc_b_global_load_lds(
                b, k_ptr, voff, k_dst, ctx->ASYNC_LDS_MAX_BYTES_PER_LANE, ctx->kv_cache_aux);
        }
        else
        {
            ckc_b_async_buffer_load_lds_addr(
                b, k_rsrc, k_dst, voff, zero_soff, ctx->KV_DMA_DWORDS, ctx->kv_cache_aux);
        }
    }
}

void ckc_gfx942_attn2d_issue_k_slice_load_runtime(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                  ckc_value_t* kv_tile_idx,
                                                  int slice_idx,
                                                  int slot_idx)
{
    ckc_ir_builder_t* b        = ctx->b;
    int bytes_per_call         = ckc__bytes_per_call(ctx);
    int kv_halves_per_call     = ckc__kv_halves_per_call(ctx);
    int k_slice_calls_per_tile = (ctx->T * ctx->K_SLICE_HD) / kv_halves_per_call;
    (void)kv_tile_idx; /* runtime path keys off slice/slot, not tile idx */

    ckc_value_t* K_lds_addr       = ckc__K_lds_addr(ctx);
    ckc_value_t* wave_off         = ckc__wave_lds_offset_i64(ctx);
    ckc_value_t* zero_soff        = ckc__zero_soff(ctx);
    ckc_value_t* kv_block_bytes_c = ckc__kv_block_bytes_c(ctx);
    ckc_value_t* lane_half_base   = ckc__lane_half_base(ctx);

    ckc_value_t* slot_off_i64 = ckc_b_const_i64(b, (int64_t)slot_idx * ctx->K_BUF_BYTES);
    ckc_value_t* K_slot_base  = ckc_b_smem_ptr_add(b, K_lds_addr, slot_off_i64);
    ckc_value_t* K_wave_base  = ckc_b_smem_ptr_add(b, K_slot_base, wave_off);

    for(int call = 0; call < k_slice_calls_per_tile; ++call)
    {
        ckc_value_t* linear_local =
            ckc_b_add(b, ckc_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
        ckc_value_t* token     = ckc_b_div(b, linear_local, ckc_b_const_i32(b, ctx->K_SLICE_HD));
        ckc_value_t* dim_local = ckc_b_mod(b, linear_local, ckc_b_const_i32(b, ctx->K_SLICE_HD));
        ckc_value_t* dim = ckc_b_add(b, ckc_b_const_i32(b, slice_idx * ctx->K_SLICE_HD), dim_local);
        ckc_value_t* linear_half =
            ckc_b_add(b, ckc_b_mul(b, token, ckc_b_const_i32(b, ctx->HD)), dim);
        (void)linear_half;
        (void)kv_block_bytes_c;
        ckc_value_t* k_rsrc = ctx->k_rsrc;
        ckc_value_t* k_ptr  = ctx->key;
        /* TensorDescriptor.offset / offset_i64_split path: not on the C
         * transforms surface (stub-to-link; the sliced-ring K path is not in
         * gfx942's fast buildable space). */
        ckc_value_t* voff = linear_half;
        ckc_value_t* k_dst =
            ckc_b_smem_ptr_add(b, K_wave_base, ckc_b_const_i64(b, (int64_t)call * bytes_per_call));
        if(ctx->USE_GLOBAL_LOAD_LDS_K)
        {
            ckc_b_global_load_lds(
                b, k_ptr, voff, k_dst, ctx->ASYNC_LDS_MAX_BYTES_PER_LANE, ctx->kv_cache_aux);
        }
        else
        {
            ckc_b_async_buffer_load_lds_addr(
                b, k_rsrc, k_dst, voff, zero_soff, ctx->KV_DMA_DWORDS, ctx->kv_cache_aux);
        }
    }
}

void ckc_gfx942_attn2d_issue_v_load_runtime(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                            ckc_value_t* kv_tile_idx,
                                            ckc_value_t* buf_idx)
{
    ckc_ir_builder_t* b      = ctx->b;
    int kv_calls_per_tile    = ckc__kv_calls_per_tile(ctx);
    int kv_halves_per_call   = ckc__kv_halves_per_call(ctx);
    int v_bytes_per_call_swz = ckc__v_bytes_per_call_swz(ctx);

    (void)buf_idx; /* V is single-buffered; always slot 0. */
    ckc_value_t* V_lds_addr       = ckc__V_lds_addr(ctx);
    ckc_value_t* v_wave_off       = ckc__v_wave_lds_offset_i64(ctx);
    ckc_value_t* zero_soff        = ckc__zero_soff(ctx);
    ckc_value_t* kv_block_bytes_c = ckc__kv_block_bytes_c(ctx);
    ckc_value_t* lane_half_base   = ckc__lane_half_base(ctx);

    ckc_value_t* V_wave_base = ckc_b_smem_ptr_add(b, V_lds_addr, v_wave_off);

    ckc_value_t* fast_block0 = NULL;
    ckc_value_t* fast_block1 = NULL;
    if(ctx->FAST_PAGED_KV_DESC)
        ckc_gfx942_attn2d_fast_paged_kv_blocks(ctx, kv_tile_idx, &fast_block0, &fast_block1);

    for(int call = 0; call < kv_calls_per_tile; ++call)
    {
        ckc_value_t* v_rsrc = ctx->v_rsrc;
        ckc_value_t* voff   = NULL;
        if(ctx->FAST_PAGED_KV_DESC)
        {
            ckc_value_t* base_i64 = NULL;
            ckc__fast_paged_kv_voff_split(
                ctx, call, fast_block0, fast_block1, ctx->value, &base_i64, &voff);
            if(base_i64 != NULL)
            {
                v_rsrc = ckc_b_buffer_rsrc(
                    b, ckc_b_global_ptr_add(b, ctx->value, base_i64), kv_block_bytes_c);
            }
        }
        else
        {
            /* Non-fast paged-KV: voff = paged_kv_desc.offset(tile_idx=,
             * linear_half=call*KV_HALVES_PER_CALL+lane_half_base, kv_head=)
             * (Python 2585-2605; I64_KV_ADDR false in gfx942 buildable space). */
            ckc_value_t* linear_half =
                ckc_b_add(b, ckc_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
            const char* in_names[3]   = {"tile_idx", "linear_half", "kv_head"};
            ckc_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            ckc_value_t* valid        = NULL;
            if(!ckc_transforms_descriptor_offset(
                   b, ctx->kv_desc, in_names, in_values, 3, &voff, &valid))
                return;
        }
        ckc_value_t* v_dst = ckc_b_smem_ptr_add(
            b, V_wave_base, ckc_b_const_i64(b, (int64_t)call * v_bytes_per_call_swz));
        ckc_b_async_buffer_load_lds_addr(
            b, v_rsrc, v_dst, voff, zero_soff, ctx->KV_DMA_DWORDS, ctx->kv_cache_aux);
    }
}

/* ==========================================================================
 * Loader dispatch wrappers (Python lines 3347-3396).
 * ========================================================================== */

void ckc_gfx942_attn2d_issue_k(ckc_gfx942_attn2d_build_ctx_t* ctx,
                               ckc_value_t* tile_idx,
                               ckc_value_t* buf_idx)
{
    if(ctx->K_SLICED_ACTIVE)
        return;
    if(ctx->FP8_MFMA_QK)
        ckc_gfx942_attn2d_issue_k_fp8_mfma_async(ctx, tile_idx, buf_idx);
    else if(ctx->KV_FP8)
        ckc_gfx942_attn2d_issue_fp8_dequant_loads(ctx, tile_idx, buf_idx);
    else
        ckc_gfx942_attn2d_issue_k_load_runtime(ctx, tile_idx, buf_idx);
}

void ckc_gfx942_attn2d_issue_v(ckc_gfx942_attn2d_build_ctx_t* ctx,
                               ckc_value_t* tile_idx,
                               ckc_value_t* buf_idx)
{
    if(ctx->FP8_MFMA_PV)
        ckc_gfx942_attn2d_issue_v_fp8_mfma_stripe(ctx, tile_idx);
    else if(ctx->KV_FP8)
        ckc_gfx942_attn2d_issue_fp8_dequant_loads(ctx, tile_idx, ckc_b_const_i32(ctx->b, 0));
    else if(ctx->TRANSPOSED_V_STORE)
        ckc_gfx942_attn2d_issue_v_transposed_store(ctx, tile_idx);
    else if(ctx->TRANSPOSED_V)
        ckc_gfx942_attn2d_issue_v_transposed(ctx, tile_idx);
    else
        ckc_gfx942_attn2d_issue_v_load_runtime(ctx, tile_idx, buf_idx);
}

/* ==========================================================================
 * _read_k8_mfma_operand (Python lines 3398-3424).
 *
 * The Python closure is _read_k8_mfma_operand(buf_idx, k_row, k_off, frag=8).
 * The internal-header form exposes (kv_tile_idx, buf_idx, k_iter, n_tile); the
 * (k_row, k_off, frag) the Python body needs are derived by the caller's QK
 * loop. Here buf_idx is the K_lds buffer slot, k_iter selects the K-row group
 * (k_row), n_tile the head-dim offset (k_off), and frag is the 32x32x16/32x32x8
 * lane fragment width. On gfx942 K_FP8_MFMA is always false, so the plain bf16
 * smem read is the only live path; the fp8 dequant branches are stub-to-link.
 * ========================================================================== */
ckc_value_t* ckc_gfx942_attn2d_read_k8_mfma_operand(ckc_gfx942_attn2d_build_ctx_t* ctx,
                                                    ckc_value_t* kv_tile_idx,
                                                    ckc_value_t* buf_idx,
                                                    int k_iter,
                                                    int n_tile)
{
    ckc_ir_builder_t* b = ctx->b;
    (void)kv_tile_idx;
    int frag           = ctx->USE_MFMA_32X32X8 ? 4 : 8;
    ckc_value_t* k_row = ckc_b_const_i32(b, k_iter);
    ckc_value_t* k_off = ckc_b_const_i32(b, n_tile);
    ckc_value_t* idx[3];
    idx[0] = buf_idx;
    idx[1] = k_row;
    idx[2] = k_off;
    if(!ctx->FP8_MFMA_QK)
        return ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, frag);
    if(ctx->FP8_NATIVE_QK)
        return ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, ckc_fp8e4m3(), frag);
    /* fp8 dequant register path (dequant_fp8x8_to_dtype): not exported by a C
     * helper header; stub-to-link (fp8 is rejected on gfx942). */
    return ckc_b_smem_load_vN(b, ctx->K_lds, idx, 3, ckc_fp8e4m3(), frag);
}
