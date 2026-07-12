// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_kv_descriptors_loads.c --
 *   C99 port of the KV DESCRIPTOR + ASYNC-DMA LOADERS bucket of
 *   rocke/instances/gfx942/attention_tiled_2d.py (arch gfx942).
 *
 * SCOPE (this translation unit).
 *   rocke_gfx942_attn2d_emit_preloop            -- the pre-loop section (Python lines
 *     2163-2351): big-bytes K/V buffer resources, the async-DMA byte/stride
 *     derivation, the wave/V-wave LDS dest offsets, the paged-KV byte
 *     TensorDescriptor full transform DAG, and seq_base / block_table_max_idx.
 *   rocke_gfx942_attn2d_fast_paged_kv_blocks    -- _fast_paged_kv_blocks (2352-2373):
 *     the per-tile two-logical-block block_tables lookup.
 *   rocke_gfx942_attn2d_fast_paged_kv_voff      -- _fast_paged_kv_voff (2375-2398):
 *     the per-call within-block byte voffset (+ optional i64 block base).
 *   rocke_gfx942_attn2d_v_t_slot / _v_t_store / _v_t_load / _v_load1
 *     -- the V-LDS slot/store/load + flat V load helpers (1756-1830).
 *   rocke_gfx942_attn2d_issue_k_load_runtime    -- _issue_k_load_runtime (2440-2507).
 *   rocke_gfx942_attn2d_issue_k_slice_load_runtime -- _issue_k_slice_load_runtime
 *     (2511-2560).
 *   rocke_gfx942_attn2d_issue_v_load_runtime    -- _issue_v_load_runtime (2562-2619).
 *   rocke_gfx942_attn2d_issue_k / _issue_v      -- the loader dispatch wrappers
 *     (3347-3396).
 *   rocke_gfx942_attn2d_read_k8_mfma_operand    -- _read_k8_mfma_operand (3398-3424).
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
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"

/* ==========================================================================
 * Shared compile-time byte/stride derivation (Python prologue locals 2177-2287).
 *
 * The Python build prologue computes these once and the loader closures capture
 * them. In C they are pure functions of ctx geometry, recomputed where used.
 * ========================================================================== */

/* KV_HALVES_PER_LANE: async-DMA per-lane half count. gfx950 = 8 (16 bytes/lane,
 * dwords=4); gfx942 = ASYNC_LDS_MAX_BYTES_PER_LANE//2 = 2 (set in ctx_init). */
static int rocke__kv_halves_per_lane(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->KV_DMA_HALVES_PER_LANE;
}

/* KV_HALVES_PER_CALL = THREADS * KV_HALVES_PER_LANE. */
static int rocke__kv_halves_per_call(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->THREADS * rocke__kv_halves_per_lane(ctx);
}

/* kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL. */
static int rocke__kv_calls_per_tile(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return (ctx->T * ctx->HD) / rocke__kv_halves_per_call(ctx);
}

/* bytes_per_call = KV_HALVES_PER_CALL * 2. */
static int rocke__bytes_per_call(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return rocke__kv_halves_per_call(ctx) * 2;
}

/* bytes_per_buf = T * HD * 2 (one [T, HD] working-dtype slab). */
static int rocke__bytes_per_buf(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->T * ctx->HD * 2;
}

/* WAVE_BYTES = WAVE * (KV_DMA_HALVES_PER_LANE * 2). gfx950 = WAVE*16 (dwords=4);
 * gfx942 = WAVE*ASYNC_LDS_MAX_BYTES_PER_LANE (=WAVE*4). */
static int rocke__wave_bytes(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return ctx->WAVE * (ctx->KV_DMA_HALVES_PER_LANE * 2);
}

/* V_BYTES_PER_CALL_SWZ: per-call dest stride for V (swizzle-aware). */
static int rocke__v_bytes_per_call_swz(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    int bpc = rocke__bytes_per_call(ctx);
    if(ctx->SWIZZLE_VLDS)
        bpc += ctx->V_ROWS_PER_CALL * ctx->V_LDS_PAD * 2;
    return bpc;
}

/* kv_block_bytes_c value = BS * NUM_KV * HD * KV_BYTES (one-block buffer bound).
 * Python creates this const ONCE at line 2227 and reuses it in every loader;
 * cache it on ctx so we do not allocate a duplicate const per loader (which
 * would shift downstream %values). */
static rocke_value_t* rocke__kv_block_bytes_c(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->kv_block_bytes_c_v == NULL)
    {
        int kv_stride_blk_b = ctx->BS * ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
        ctx->kv_block_bytes_c_v = rocke_b_const_i32(ctx->b, kv_stride_blk_b);
    }
    return ctx->kv_block_bytes_c_v;
}

/* lane_half_base = tid * KV_HALVES_PER_LANE (Python 2232). Emitted once in
 * emit_preloop and cached on ctx so every loader reuses the same SSA value
 * (matches the single Python local). */
static rocke_value_t* rocke__lane_half_base(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_half_base_v == NULL)
        ctx->lane_half_base_v = rocke_b_mul(
            ctx->b, ctx->tid, rocke_b_const_i32(ctx->b, rocke__kv_halves_per_lane(ctx)));
    return ctx->lane_half_base_v;
}

/* seq_base = to_sgpr_u32(seq_idx * bt_stride_p) (Python 2327). Cached on ctx. */
static rocke_value_t* rocke__seq_base(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->seq_base == NULL)
        ctx->seq_base
            = rocke_b_to_sgpr_u32(ctx->b, rocke_b_mul(ctx->b, ctx->seq_idx, ctx->bt_stride_p));
    return ctx->seq_base;
}

/* block_table_max_idx = to_sgpr_u32(num_seqs_p * bt_stride_p) (Python 2340). */
static rocke_value_t* rocke__block_table_max_idx(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->block_table_max_idx == NULL)
        ctx->block_table_max_idx
            = rocke_b_to_sgpr_u32(ctx->b, rocke_b_mul(ctx->b, ctx->num_seqs_p, ctx->bt_stride_p));
    return ctx->block_table_max_idx;
}

/* K_lds_addr = ptrtoint(K_lds) (Python 2234); cached. */
static rocke_value_t* rocke__K_lds_addr(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->K_lds_addr_v == NULL)
        ctx->K_lds_addr_v = rocke_b_smem_addr_of(ctx->b, ctx->K_lds);
    return ctx->K_lds_addr_v;
}

/* V_lds_addr = ptrtoint(V_lds) (Python 2235); cached. */
static rocke_value_t* rocke__V_lds_addr(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->V_lds_addr_v == NULL)
        ctx->V_lds_addr_v = rocke_b_smem_addr_of(ctx->b, ctx->V_lds);
    return ctx->V_lds_addr_v;
}

/* zero_soff = const_i32(0) (Python 2238). Python creates this ONE const and
 * reuses it as the soffset in every async load; cache it so the loaders do not
 * each allocate a duplicate const. */
static rocke_value_t* rocke__zero_soff(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->zero_soff_v == NULL)
        ctx->zero_soff_v = rocke_b_const_i32(ctx->b, 0);
    return ctx->zero_soff_v;
}

/* wave_lds_offset_i64 (K/V): const_i64(0) for NUM_WARPS==1, else the SGPR-pinned
 * zext(wave_id * WAVE_BYTES). Python creates it once (2254/2263) and reuses it;
 * cache. */
static rocke_value_t* rocke__wave_lds_offset_i64(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->wave_lds_off_i64_v != NULL)
        return ctx->wave_lds_off_i64_v;
    if(ctx->NUM_WARPS == 1)
    {
        ctx->wave_lds_off_i64_v = rocke_b_const_i64(ctx->b, 0);
        return ctx->wave_lds_off_i64_v;
    }
    rocke_value_t* off_i32 = rocke_b_to_sgpr_u32(
        ctx->b,
        rocke_b_mul(ctx->b, ctx->wave_id, rocke_b_const_i32(ctx->b, rocke__wave_bytes(ctx))));
    ctx->wave_lds_off_i64_v = rocke_b_zext(ctx->b, off_i32, rocke_i64());
    return ctx->wave_lds_off_i64_v;
}

/* v_wave_lds_offset_i64: the V-specific (swizzle-aware) wave offset. Cached. */
static rocke_value_t* rocke__v_wave_lds_offset_i64(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->v_wave_lds_off_i64_v != NULL)
        return ctx->v_wave_lds_off_i64_v;
    if(!ctx->SWIZZLE_VLDS || ctx->NUM_WARPS == 1)
    {
        if(ctx->NUM_WARPS == 1)
        {
            ctx->v_wave_lds_off_i64_v = rocke_b_const_i64(ctx->b, 0);
            return ctx->v_wave_lds_off_i64_v;
        }
        ctx->v_wave_lds_off_i64_v = rocke__wave_lds_offset_i64(ctx);
        return ctx->v_wave_lds_off_i64_v;
    }
    int v_wave_bytes = rocke__wave_bytes(ctx) + (ctx->SWIZZLE_VLDS ? ctx->V_LDS_PAD * 2 : 0);
    rocke_value_t* off_i32 = rocke_b_to_sgpr_u32(
        ctx->b, rocke_b_mul(ctx->b, ctx->wave_id, rocke_b_const_i32(ctx->b, v_wave_bytes)));
    ctx->v_wave_lds_off_i64_v = rocke_b_zext(ctx->b, off_i32, rocke_i64());
    return ctx->v_wave_lds_off_i64_v;
}

/* ==========================================================================
 * V-LDS slot/store/load + flat V load (Python lines 1756-1830).
 * ========================================================================== */

rocke_value_t* rocke_gfx942_attn2d_v_t_slot(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* dim,
                                            rocke_value_t* tok)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* k_group = rocke_b_div(b, tok, rocke_b_const_i32(b, ctx->V_T_KPACK));
    rocke_value_t* k_inner = rocke_b_mod(b, tok, rocke_b_const_i32(b, ctx->V_T_KPACK));
    rocke_value_t* n_group = rocke_b_div(b, dim, rocke_b_const_i32(b, ctx->V_T_NPER_ROW));
    rocke_value_t* n_inner = rocke_b_mod(b, dim, rocke_b_const_i32(b, ctx->V_T_NPER_ROW));
    rocke_value_t* hi = rocke_b_add(
        b,
        rocke_b_mul(b, k_group, rocke_b_const_i32(b, ctx->V_T_NGROUPS * ctx->V_T_GROUP_STRIDE)),
        rocke_b_mul(b, n_group, rocke_b_const_i32(b, ctx->V_T_GROUP_STRIDE)));
    rocke_value_t* lo
        = rocke_b_add(b, rocke_b_mul(b, n_inner, rocke_b_const_i32(b, ctx->V_T_KPACK)), k_inner);
    return rocke_b_add(b, hi, lo);
}

void rocke_gfx942_attn2d_v_t_store(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                   rocke_value_t* dim,
                                   rocke_value_t* tok,
                                   rocke_value_t* value,
                                   int n)
{
    rocke_ir_builder_t* b = ctx->b;
    if(ctx->V_T_CK_LAYOUT)
    {
        rocke_value_t* idx[2];
        idx[0] = rocke_b_const_i32(b, 0);
        idx[1] = rocke_gfx942_attn2d_v_t_slot(ctx, dim, tok);
        rocke_b_smem_store_vN(b, ctx->V_lds, idx, 2, value, n);
    }
    else
    {
        rocke_value_t* idx[3];
        idx[0] = rocke_b_const_i32(b, 0);
        idx[1] = dim;
        idx[2] = tok;
        rocke_b_smem_store_vN(b, ctx->V_lds, idx, 3, value, n);
    }
}

rocke_value_t* rocke_gfx942_attn2d_v_t_load(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* dim,
                                            rocke_value_t* tok,
                                            int n)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* v_buf0 = rocke_b_const_i32(b, 0);
    if(ctx->V_T_CK_LAYOUT)
    {
        rocke_value_t* idx[2];
        idx[0] = v_buf0;
        idx[1] = rocke_gfx942_attn2d_v_t_slot(ctx, dim, tok);
        return rocke_b_smem_load_vN(b, ctx->V_lds, idx, 2, ctx->dtype, n);
    }
    {
        rocke_value_t* idx[3];
        idx[0] = v_buf0;
        idx[1] = dim;
        idx[2] = tok;
        return rocke_b_smem_load_vN(b, ctx->V_lds, idx, 3, ctx->dtype, n);
    }
}

rocke_value_t* rocke_gfx942_attn2d_v_load1(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                           rocke_value_t* v_buf,
                                           rocke_value_t* v_row,
                                           rocke_value_t* v_n_col)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->SWIZZLE_VLDS)
    {
        rocke_value_t* idx[3];
        idx[0] = v_buf;
        idx[1] = v_row;
        idx[2] = v_n_col;
        return rocke_b_smem_load_vN(b, ctx->V_lds, idx, 3, ctx->dtype, 1);
    }
    {
        rocke_value_t* group = rocke_b_lshr(b, v_row, rocke_b_const_i32(b, ctx->V_GROUP_SHIFT));
        rocke_value_t* within
            = rocke_b_land(b, v_row, rocke_b_const_i32(b, ctx->V_ROWS_PER_CALL - 1));
        /* slot = (group*GS + within*LS) + v_n_col. Python emits the group mul
         * before the within mul; sequence via temps (C arg-eval is unspecified). */
        rocke_value_t* group_mul = rocke_b_mul(b, group, rocke_b_const_i32(b, ctx->V_GROUP_STRIDE));
        rocke_value_t* within_mul = rocke_b_mul(b, within, rocke_b_const_i32(b, ctx->V_LDS_STRIDE));
        rocke_value_t* slot = rocke_b_add(b, rocke_b_add(b, group_mul, within_mul), v_n_col);
        rocke_value_t* idx[2];
        idx[0] = v_buf;
        idx[1] = slot;
        return rocke_b_smem_load_vN(b, ctx->V_lds, idx, 2, ctx->dtype, 1);
    }
}

/* ==========================================================================
 * Pre-loop: K/V buffer resources, byte derivation, paged-KV descriptor
 * (Python lines 2163-2351).
 * ========================================================================== */

void rocke_gfx942_attn2d_emit_preloop(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* big-bytes K/V buffer resources (2166-2168). The buffer rsrc bounds OOB
     * voffsets to return zero; size it large so valid offsets never trip it. */
    rocke_value_t* big_bytes = rocke_b_const_i32(b, 0x7FFF0000);
    ctx->k_rsrc = rocke_b_buffer_rsrc(b, ctx->key, big_bytes);
    ctx->v_rsrc = rocke_b_buffer_rsrc(b, ctx->value, big_bytes);

    /* The bf16 async-DMA byte derivation (2177-2181) is purely compile-time and
     * the loader closures recompute it; emit the asserted invariants only so the
     * IR stream is unchanged (these emit no ops). */
    {
        int kv_halves_per_call = rocke__kv_halves_per_call(ctx);
        /* assert (T * HD) % KV_HALVES_PER_CALL == 0 (2179). */
        (void)((ctx->T * ctx->HD) % kv_halves_per_call);
    }

    /* kv_block_bytes_c (2227): Python creates this one-block-bound const here,
     * BEFORE lane_half_base, and reuses it in every loader. Emit it now so its
     * SSA value lands in source order. */
    (void)rocke__kv_block_bytes_c(ctx);

    /* Per-lane starting half index (2232), then K/V LDS base ptrtoints (2234-
     * 2235). These are SINGLE Python locals consumed by every loader; emit them
     * here, in source order, and cache on ctx so the loaders reuse the same SSA
     * values (the byte-identity contract requires the exact op stream). */
    (void)rocke__lane_half_base(ctx);
    (void)rocke__K_lds_addr(ctx);
    (void)rocke__V_lds_addr(ctx);

    /* zero_soff (2238), then wave_lds_offset_i64 (2254) and
     * v_wave_lds_offset_i64 (2279). Python creates each as a single SSA value
     * HERE (even the const_i64(0) consumes a value number) and reuses them in
     * every loader. Force their creation now, in source order, and cache so the
     * loaders reuse them instead of allocating duplicates -- otherwise their
     * value numbers land later and shift the whole downstream stream. */
    (void)rocke__zero_soff(ctx);
    (void)rocke__wave_lds_offset_i64(ctx);
    (void)rocke__v_wave_lds_offset_i64(ctx);
    (void)rocke__bytes_per_buf;
    (void)rocke__v_bytes_per_call_swz;
    (void)rocke__kv_calls_per_tile;
    (void)rocke__bytes_per_call;
    (void)rocke__kv_block_bytes_c;

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
            (void)rocke__seq_base(ctx);
            (void)rocke__block_table_max_idx(ctx);
        }
        ctx->kv_desc = NULL;
        return;
    }

    /* ---- Paged KV byte descriptor (full transform DAG, 2327-2438) ---- */
    rocke_value_t* seq_base = rocke__seq_base(ctx);
    rocke_value_t* max_idx = rocke__block_table_max_idx(ctx);

    /* _kv_base = naive("paged_kv_bytes",
     *     lengths=[1<<24, BS, NUM_KV, HD],
     *     strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
     *     coord_names=("physical_block","token","kv_head","dim")). */
    int blk_b = ctx->BS * ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
    int tok_b = ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
    int h_b = ctx->HD * ctx->KV_BYTES;
    int kv_lengths[4] = {1 << 24, ctx->BS, ctx->NUM_KV, ctx->HD};
    int kv_strides[4] = {blk_b, tok_b, h_b, ctx->KV_BYTES};
    static const char* const kv_coords[4] = {"physical_block", "token", "kv_head", "dim"};
    rocke_tensor_descriptor_t* kv_base = rocke_tensor_descriptor_naive(
        b, "paged_kv_bytes", kv_lengths, 4, kv_strides, kv_coords, 4);

    /* Single-block tile (N_BLOCKS_PER_TILE == 1, T == BS):
     *   indirect(tile_idx -> physical_block) ; unmerge(linear_half -> token,dim).
     * Multi-block (N_BLOCKS_PER_TILE > 1):
     *   unmerge(linear_half -> block_within_tile,token,dim) ;
     *   embed((tile_idx,block_within_tile) -> linear_block_idx) ;
     *   indirect(linear_block_idx -> physical_block). */
    if(ctx->N_BLOCKS_PER_TILE == 1)
    {
        const char* td_into[2] = {"token", "dim"};
        int td_dims[2] = {ctx->T, ctx->HD};
        const rocke_transform_t* chain[2];
        chain[0] = rocke_indirect(
            b, "tile_idx", "physical_block", ctx->block_tables, seq_base, max_idx, 0);
        chain[1] = rocke_unmerge(b, "linear_half", td_into, 2, td_dims);
        ctx->kv_desc = rocke_tensor_descriptor_transform(b, kv_base, chain, 2);
    }
    else
    {
        const char* td_into[3] = {"block_within_tile", "token", "dim"};
        int td_dims[3] = {ctx->N_BLOCKS_PER_TILE, ctx->BS, ctx->HD};
        const char* emb_upper[2] = {"tile_idx", "block_within_tile"};
        int emb_strides[2] = {ctx->N_BLOCKS_PER_TILE, 1};
        const rocke_transform_t* chain[3];
        chain[0] = rocke_unmerge(b, "linear_half", td_into, 3, td_dims);
        chain[1] = rocke_embed(b, emb_upper, 2, "linear_block_idx", emb_strides, 0);
        chain[2] = rocke_indirect(
            b, "linear_block_idx", "physical_block", ctx->block_tables, seq_base, max_idx, 0);
        ctx->kv_desc = rocke_tensor_descriptor_transform(b, kv_base, chain, 3);
    }

    /* tile-0 prefetch (issue_k(tile0) / issue_v(tile0)) is driven by the loop
     * driver glue, which calls the loader phase functions below in Python order. */
}

/* ==========================================================================
 * Fast paged-KV byte-descriptor closures (Python lines 2352-2398).
 * ========================================================================== */

void rocke_gfx942_attn2d_fast_paged_kv_blocks(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t** out_block0,
                                              rocke_value_t** out_block1)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* seq_base = rocke__seq_base(ctx);
    rocke_value_t* max_idx = rocke__block_table_max_idx(ctx);

    rocke_value_t* logical_block0 = rocke_b_mul(b, kv_tile_idx, rocke_b_const_i32(b, 2));
    rocke_value_t* logical_block1 = rocke_b_add(b, logical_block0, rocke_b_const_i32(b, 1));
    rocke_value_t* idx0 = rocke_b_add(b, seq_base, logical_block0);
    rocke_value_t* idx1 = rocke_b_add(b, seq_base, logical_block1);
    /* Python evaluates masked_global_load args left-to-right: the cmp_lt mask is
     * created BEFORE the const(0) default. C arg-eval order is unspecified, so
     * bind the mask + default in source order to keep the value numbering. */
    rocke_value_t* mask0 = rocke_b_cmp_lt(b, idx0, max_idx);
    rocke_value_t* def0 = rocke_b_const_i32(b, 0);
    rocke_value_t* block0
        = rocke_b_masked_global_load(b, ctx->block_tables, idx0, mask0, def0, rocke_i32(), 4);
    rocke_value_t* mask1 = rocke_b_cmp_lt(b, idx1, max_idx);
    rocke_value_t* def1 = rocke_b_const_i32(b, 0);
    rocke_value_t* block1
        = rocke_b_masked_global_load(b, ctx->block_tables, idx1, mask1, def1, rocke_i32(), 4);
    if(out_block0)
        *out_block0 = rocke_b_to_sgpr_u32(b, block0);
    if(out_block1)
        *out_block1 = rocke_b_to_sgpr_u32(b, block1);
}

rocke_value_t* rocke_gfx942_attn2d_fast_paged_kv_voff(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                      int call,
                                                      rocke_value_t* block0,
                                                      rocke_value_t* block1)
{
    rocke_ir_builder_t* b = ctx->b;
    int kv_halves_per_call = rocke__kv_halves_per_call(ctx);
    int bs_hd = ctx->BS * ctx->HD;
    int fast_calls_per_block = bs_hd / kv_halves_per_call;

    /* gfx950: each call covers one full block (FAST_CALLS_PER_BLOCK == 1), so
     * physical = block0 if call==0 else block1 and half_in_block IS lane_half_base
     * directly -- NO per-call add (Python gfx950 1579-1582). gfx942: CPB calls
     * drain one block; half_in_block = const(call_in_block*KV_HALVES_PER_CALL) +
     * lane_half_base (Python gfx942). */
    bool wide = (ctx->target != NULL && ctx->target->memory.has_ds_read_tr);
    rocke_value_t* physical;
    rocke_value_t* half_in_block;
    if(wide)
    {
        physical = (call == 0) ? block0 : block1;
        half_in_block = rocke__lane_half_base(ctx);
    }
    else
    {
        physical = (call < fast_calls_per_block) ? block0 : block1;
        int call_in_block = call % fast_calls_per_block;
        half_in_block = rocke_b_add(b,
                                    rocke_b_const_i32(b, call_in_block * kv_halves_per_call),
                                    rocke__lane_half_base(ctx));
    }
    rocke_value_t* token = rocke_b_lshr(b, half_in_block, rocke_b_const_i32(b, 6));
    rocke_value_t* dim = rocke_b_land(b, half_in_block, rocke_b_const_i32(b, 63));
    rocke_value_t* token_b = rocke_b_shl(b, token, rocke_b_const_i32(b, 10));
    rocke_value_t* head_b = rocke_b_shl(b, ctx->kv_head_idx, rocke_b_const_i32(b, 7));
    rocke_value_t* dim_b = rocke_b_shl(b, dim, rocke_b_const_i32(b, 1));
    rocke_value_t* within = rocke_b_add(b, rocke_b_add(b, token_b, head_b), dim_b);

    if(ctx->I64_KV_ADDR)
    {
        /* Returns the within-block i32 voffset; the i64 block base is built by
         * the caller (the header form exposes only the i32 voffset). */
        return within;
    }
    rocke_value_t* block_b = rocke_b_shl(b, physical, rocke_b_const_i32(b, 15));
    return rocke_b_add(b, block_b, within);
}

/* Internal: the full Python tuple return of _fast_paged_kv_voff (i64 base,
 * within voffset). The i64 base is only produced on the I64_KV_ADDR path. */
static void rocke__fast_paged_kv_voff_split(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            int call,
                                            rocke_value_t* block0,
                                            rocke_value_t* block1,
                                            rocke_value_t* base_ptr_src,
                                            rocke_value_t** out_base_i64,
                                            rocke_value_t** out_voff)
{
    rocke_ir_builder_t* b = ctx->b;
    int kv_halves_per_call = rocke__kv_halves_per_call(ctx);
    int bs_hd = ctx->BS * ctx->HD;
    int fast_calls_per_block = bs_hd / kv_halves_per_call;
    bool wide = (ctx->target != NULL && ctx->target->memory.has_ds_read_tr);
    rocke_value_t* physical = wide ? ((call == 0) ? block0 : block1)
                                   : ((call < fast_calls_per_block) ? block0 : block1);

    rocke_value_t* within = rocke_gfx942_attn2d_fast_paged_kv_voff(ctx, call, block0, block1);
    (void)base_ptr_src;
    if(ctx->I64_KV_ADDR)
    {
        rocke_value_t* base_i64
            = rocke_b_shl(b, rocke_b_zext(b, physical, rocke_i64()), rocke_b_const_i64(b, 15));
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

void rocke_gfx942_attn2d_issue_k_load_runtime(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    int bytes_per_buf = rocke__bytes_per_buf(ctx);
    int bytes_per_call = rocke__bytes_per_call(ctx);
    int kv_calls_per_tile = rocke__kv_calls_per_tile(ctx);
    int kv_halves_per_call = rocke__kv_halves_per_call(ctx);

    rocke_value_t* K_lds_addr = rocke__K_lds_addr(ctx);
    rocke_value_t* wave_off = rocke__wave_lds_offset_i64(ctx);
    rocke_value_t* zero_soff = rocke__zero_soff(ctx);
    rocke_value_t* kv_block_bytes_c = rocke__kv_block_bytes_c(ctx);
    rocke_value_t* lane_half_base = rocke__lane_half_base(ctx);

    rocke_value_t* buf_off_i32 = rocke_b_mul(b, buf_idx, rocke_b_const_i32(b, bytes_per_buf));
    rocke_value_t* buf_off_i64 = rocke_b_zext(b, buf_off_i32, rocke_i64());
    rocke_value_t* K_buf_base = rocke_b_smem_ptr_add(b, K_lds_addr, buf_off_i64);
    rocke_value_t* K_wave_base = rocke_b_smem_ptr_add(b, K_buf_base, wave_off);

    rocke_value_t* fast_block0 = NULL;
    rocke_value_t* fast_block1 = NULL;
    if(ctx->FAST_PAGED_KV_DESC)
        rocke_gfx942_attn2d_fast_paged_kv_blocks(ctx, kv_tile_idx, &fast_block0, &fast_block1);

    for(int call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* k_rsrc = ctx->k_rsrc;
        rocke_value_t* k_ptr = ctx->key;
        rocke_value_t* voff = NULL;
        if(ctx->FAST_PAGED_KV_DESC)
        {
            rocke_value_t* base_i64 = NULL;
            rocke__fast_paged_kv_voff_split(
                ctx, call, fast_block0, fast_block1, ctx->key, &base_i64, &voff);
            if(base_i64 != NULL)
            {
                k_ptr = rocke_b_global_ptr_add(b, ctx->key, base_i64);
                k_rsrc = rocke_b_buffer_rsrc(b, k_ptr, kv_block_bytes_c);
            }
        }
        else
        {
            /* Non-fast paged-KV (N_BLOCKS_PER_TILE==1 / multi-block):
             *   linear_half = call*KV_HALVES_PER_CALL + lane_half_base
             *   voff, _ = paged_kv_desc.offset(tile_idx=, linear_half=, kv_head=)
             * (Python 2470-2489; I64_KV_ADDR is false in gfx942's buildable
             * space, so the plain i32 voffset is used). */
            rocke_value_t* linear_half
                = rocke_b_add(b, rocke_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            rocke_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            rocke_value_t* valid = NULL;
            if(!rocke_transforms_descriptor_offset(
                   b, ctx->kv_desc, in_names, in_values, 3, &voff, &valid))
                return;
        }
        rocke_value_t* k_dst = rocke_b_smem_ptr_add(
            b, K_wave_base, rocke_b_const_i64(b, (int64_t)call * bytes_per_call));
        if(ctx->USE_GLOBAL_LOAD_LDS_K)
        {
            rocke_b_global_load_lds(
                b, k_ptr, voff, k_dst, ctx->ASYNC_LDS_MAX_BYTES_PER_LANE, ctx->kv_cache_aux);
        }
        else
        {
            rocke_b_async_buffer_load_lds_addr(
                b, k_rsrc, k_dst, voff, zero_soff, ctx->KV_DMA_DWORDS, ctx->kv_cache_aux);
        }
    }
}

void rocke_gfx942_attn2d_issue_k_slice_load_runtime(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                    rocke_value_t* kv_tile_idx,
                                                    int slice_idx,
                                                    int slot_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    int bytes_per_call = rocke__bytes_per_call(ctx);
    int kv_halves_per_call = rocke__kv_halves_per_call(ctx);
    int k_slice_calls_per_tile = (ctx->T * ctx->K_SLICE_HD) / kv_halves_per_call;
    (void)kv_tile_idx; /* runtime path keys off slice/slot, not tile idx */

    rocke_value_t* K_lds_addr = rocke__K_lds_addr(ctx);
    rocke_value_t* wave_off = rocke__wave_lds_offset_i64(ctx);
    rocke_value_t* zero_soff = rocke__zero_soff(ctx);
    rocke_value_t* kv_block_bytes_c = rocke__kv_block_bytes_c(ctx);
    rocke_value_t* lane_half_base = rocke__lane_half_base(ctx);

    rocke_value_t* slot_off_i64 = rocke_b_const_i64(b, (int64_t)slot_idx * ctx->K_BUF_BYTES);
    rocke_value_t* K_slot_base = rocke_b_smem_ptr_add(b, K_lds_addr, slot_off_i64);
    rocke_value_t* K_wave_base = rocke_b_smem_ptr_add(b, K_slot_base, wave_off);

    for(int call = 0; call < k_slice_calls_per_tile; ++call)
    {
        rocke_value_t* linear_local
            = rocke_b_add(b, rocke_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
        rocke_value_t* token = rocke_b_div(b, linear_local, rocke_b_const_i32(b, ctx->K_SLICE_HD));
        rocke_value_t* dim_local
            = rocke_b_mod(b, linear_local, rocke_b_const_i32(b, ctx->K_SLICE_HD));
        rocke_value_t* dim
            = rocke_b_add(b, rocke_b_const_i32(b, slice_idx * ctx->K_SLICE_HD), dim_local);
        rocke_value_t* linear_half
            = rocke_b_add(b, rocke_b_mul(b, token, rocke_b_const_i32(b, ctx->HD)), dim);
        (void)linear_half;
        (void)kv_block_bytes_c;
        rocke_value_t* k_rsrc = ctx->k_rsrc;
        rocke_value_t* k_ptr = ctx->key;
        /* TensorDescriptor.offset / offset_i64_split path: not on the C
         * transforms surface (stub-to-link; the sliced-ring K path is not in
         * gfx942's fast buildable space). */
        rocke_value_t* voff = linear_half;
        rocke_value_t* k_dst = rocke_b_smem_ptr_add(
            b, K_wave_base, rocke_b_const_i64(b, (int64_t)call * bytes_per_call));
        if(ctx->USE_GLOBAL_LOAD_LDS_K)
        {
            rocke_b_global_load_lds(
                b, k_ptr, voff, k_dst, ctx->ASYNC_LDS_MAX_BYTES_PER_LANE, ctx->kv_cache_aux);
        }
        else
        {
            rocke_b_async_buffer_load_lds_addr(
                b, k_rsrc, k_dst, voff, zero_soff, ctx->KV_DMA_DWORDS, ctx->kv_cache_aux);
        }
    }
}

void rocke_gfx942_attn2d_issue_v_load_runtime(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    int kv_calls_per_tile = rocke__kv_calls_per_tile(ctx);
    int kv_halves_per_call = rocke__kv_halves_per_call(ctx);
    int v_bytes_per_call_swz = rocke__v_bytes_per_call_swz(ctx);

    (void)buf_idx; /* V is single-buffered; always slot 0. */
    rocke_value_t* V_lds_addr = rocke__V_lds_addr(ctx);
    rocke_value_t* v_wave_off = rocke__v_wave_lds_offset_i64(ctx);
    rocke_value_t* zero_soff = rocke__zero_soff(ctx);
    rocke_value_t* kv_block_bytes_c = rocke__kv_block_bytes_c(ctx);
    rocke_value_t* lane_half_base = rocke__lane_half_base(ctx);

    rocke_value_t* V_wave_base = rocke_b_smem_ptr_add(b, V_lds_addr, v_wave_off);

    rocke_value_t* fast_block0 = NULL;
    rocke_value_t* fast_block1 = NULL;
    if(ctx->FAST_PAGED_KV_DESC)
        rocke_gfx942_attn2d_fast_paged_kv_blocks(ctx, kv_tile_idx, &fast_block0, &fast_block1);

    for(int call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* v_rsrc = ctx->v_rsrc;
        rocke_value_t* voff = NULL;
        if(ctx->FAST_PAGED_KV_DESC)
        {
            rocke_value_t* base_i64 = NULL;
            rocke__fast_paged_kv_voff_split(
                ctx, call, fast_block0, fast_block1, ctx->value, &base_i64, &voff);
            if(base_i64 != NULL)
            {
                v_rsrc = rocke_b_buffer_rsrc(
                    b, rocke_b_global_ptr_add(b, ctx->value, base_i64), kv_block_bytes_c);
            }
        }
        else
        {
            /* Non-fast paged-KV: voff = paged_kv_desc.offset(tile_idx=,
             * linear_half=call*KV_HALVES_PER_CALL+lane_half_base, kv_head=)
             * (Python 2585-2605; I64_KV_ADDR false in gfx942 buildable space). */
            rocke_value_t* linear_half
                = rocke_b_add(b, rocke_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            rocke_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            rocke_value_t* valid = NULL;
            if(!rocke_transforms_descriptor_offset(
                   b, ctx->kv_desc, in_names, in_values, 3, &voff, &valid))
                return;
        }
        rocke_value_t* v_dst = rocke_b_smem_ptr_add(
            b, V_wave_base, rocke_b_const_i64(b, (int64_t)call * v_bytes_per_call_swz));
        rocke_b_async_buffer_load_lds_addr(
            b, v_rsrc, v_dst, voff, zero_soff, ctx->KV_DMA_DWORDS, ctx->kv_cache_aux);
    }
}

/* ==========================================================================
 * Loader dispatch wrappers (Python lines 3347-3396).
 * ========================================================================== */

void rocke_gfx942_attn2d_issue_k(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx)
{
    if(ctx->K_SLICED_ACTIVE)
        return;
    if(ctx->FP8_MFMA_QK)
        rocke_gfx942_attn2d_issue_k_fp8_mfma_async(ctx, tile_idx, buf_idx);
    else if(ctx->KV_FP8)
        rocke_gfx942_attn2d_issue_fp8_dequant_loads(ctx, tile_idx, buf_idx);
    else
        rocke_gfx942_attn2d_issue_k_load_runtime(ctx, tile_idx, buf_idx);
}

void rocke_gfx942_attn2d_issue_v(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx)
{
    if(ctx->FP8_MFMA_PV)
        rocke_gfx942_attn2d_issue_v_fp8_mfma_stripe(ctx, tile_idx);
    else if(ctx->KV_FP8)
        rocke_gfx942_attn2d_issue_fp8_dequant_loads(ctx, tile_idx, rocke_b_const_i32(ctx->b, 0));
    else if(ctx->TRANSPOSED_V_STORE)
        rocke_gfx942_attn2d_issue_v_transposed_store(ctx, tile_idx);
    else if(ctx->TRANSPOSED_V)
        rocke_gfx942_attn2d_issue_v_transposed(ctx, tile_idx);
    else
        rocke_gfx942_attn2d_issue_v_load_runtime(ctx, tile_idx, buf_idx);
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
rocke_value_t* rocke_gfx942_attn2d_read_k8_mfma_operand(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                        rocke_value_t* kv_tile_idx,
                                                        rocke_value_t* buf_idx,
                                                        int k_iter,
                                                        int n_tile)
{
    rocke_ir_builder_t* b = ctx->b;
    (void)kv_tile_idx;
    int frag = ctx->USE_MFMA_32X32X8 ? 4 : 8;
    rocke_value_t* k_row = rocke_b_const_i32(b, k_iter);
    rocke_value_t* k_off = rocke_b_const_i32(b, n_tile);
    rocke_value_t* idx[3];
    idx[0] = buf_idx;
    idx[1] = k_row;
    idx[2] = k_off;
    if(!ctx->FP8_MFMA_QK)
        return rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, frag);
    if(ctx->FP8_NATIVE_QK)
        return rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, rocke_fp8e4m3(), frag);
    /* fp8 dequant register path (dequant_fp8x8_to_dtype): not exported by a C
     * helper header; stub-to-link (fp8 is rejected on gfx942). */
    return rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, rocke_fp8e4m3(), frag);
}
