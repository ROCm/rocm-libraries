// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_kv_descriptors_loads.c
 *   -- C99 port of the PAGED-KV DESCRIPTOR DAG + bf16 K/V ASYNC-DMA LOADERS bucket
 *   of rocke/instances/gfx950/attention_tiled_2d.py (arch gfx950).
 *
 * SCOPE (this translation unit).
 *   rocke_gfx950_attn2d_emit_preloop          -- the pre-loop section (Python lines
 *     1163-1630, the parts relevant to K/V buffer descriptors): the big-bytes K/V
 *     buffer resources, the async-DMA byte/stride derivation, the per-lane half
 *     base, the K/V LDS dest ptrtoints + wave LDS offsets, seq_base /
 *     block_table_max_idx (the SGPR-pinned per-seq block-table base + footprint
 *     bound), and the paged-KV byte TensorDescriptor full transform DAG (the
 *     non-fast path; the fast path emits no standalone descriptor).
 *   rocke_gfx950_attn2d_fast_paged_kv_blocks  -- _fast_paged_kv_blocks (1550-1571):
 *     the per-tile two-logical-block block_tables lookup.
 *   rocke_gfx950_attn2d_fast_paged_kv_voff    -- _fast_paged_kv_voff (1573-1589):
 *     the per-call within-block byte voffset (+ optional i64 block base on
 *     I64_KV_ADDR).
 *   rocke_gfx950_attn2d_issue_k_load_runtime  -- _issue_k_load_runtime (1631-1689).
 *   rocke_gfx950_attn2d_issue_v_load_runtime  -- _issue_v_load_runtime (1691-1739).
 *   rocke_gfx950_attn2d_issue_k / _issue_v    -- the loader dispatch wrappers
 *     (2184-2227).
 *   rocke_gfx950_attn2d_read_k8_mfma_operand  -- _read_k8_mfma_operand (2229-2248).
 *
 * The builder-call sequence is byte-identical to the Python body: same ops, same
 * order, same operands. Because C has no closure capture, the per-call byte/
 * stride scalars the Python prologue computes once and the loader closures share
 * are recomputed here (they are pure compile-time integers + a small number of
 * cheap, idempotent SSA recomputations cached on ctx) -- the emitted IR is
 * unchanged.
 *
 * Paths reachable only through symbols not yet on the C transforms surface (the
 * non-fast paged-KV descriptor's TensorDescriptor.offset_i64_split when
 * I64_KV_ADDR is set) are stubbed-to-link: gfx950 production uses
 * FAST_PAGED_KV_DESC, whose per-call byte offsets are built directly by
 * _fast_paged_kv_blocks / _fast_paged_kv_voff (which DO honour I64_KV_ADDR). The
 * fp8 K/V loaders (_issue_k_fp8_mfma_async, _issue_fp8_dequant_loads,
 * _issue_v_fp8_mfma_stripe) live in a peer TU and are called via the internal
 * header.
 *
 * NOTE gfx950 has NO cfvst / transposed-V-store / k-sliced-ring / strided-V
 * families (those are gfx942-only): the wide atom's CK-Tile LDS transpose reads
 * replace them, so this TU is a strict subset of the gfx942 sibling.
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#include "rocke/instance_gfx950_attention_tiled_2d_internal.h"

/* ==========================================================================
 * Shared compile-time byte/stride derivation (Python prologue locals 1409-1473).
 *
 * The Python build prologue computes these once and the loader closures capture
 * them. In C they are pure functions of ctx geometry, recomputed where used. The
 * gfx950 bf16 path pins dwords=4 (16 bytes/lane) and KV_HALVES_PER_CALL=THREADS*8
 * (8 halves/lane) as literals -- mirror that, not the ASYNC_LDS_MAX_* fields.
 * ========================================================================== */

/* KV_HALVES_PER_CALL = THREADS * 8 (Python 1413). */
static int rocke__kv_halves_per_call(const rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    return ctx->THREADS * 8;
}

/* kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL (Python 1415). */
static int rocke__kv_calls_per_tile(const rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    return (ctx->T * ctx->HD) / rocke__kv_halves_per_call(ctx);
}

/* bytes_per_call = KV_HALVES_PER_CALL * 2 (Python 1416). */
static int rocke__bytes_per_call(const rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    return rocke__kv_halves_per_call(ctx) * 2;
}

/* bytes_per_buf = T * HD * 2 (one [T, HD] working-dtype (bf16) slab) (Python 1464). */
static int rocke__bytes_per_buf(const rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    return ctx->T * ctx->HD * 2;
}

/* WAVE_BYTES = WAVE * 16 (dwords=4 -> 16 bytes/lane) (Python 1473). */
static int rocke__wave_bytes(const rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    return ctx->WAVE * 16;
}

/* kv_block_bytes_c value = BS * NUM_KV * HD * KV_BYTES (one-block buffer bound,
 * Python 1458). Python creates this const ONCE and reuses it in every loader;
 * cache it on ctx so we do not allocate a duplicate const per loader (which would
 * shift downstream %values). */
static rocke_value_t* rocke__kv_block_bytes_c(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->kv_block_bytes_c_v == NULL)
    {
        int kv_stride_blk_b = ctx->BS * ctx->NUM_KV * ctx->HD * ctx->KV_BYTES;
        ctx->kv_block_bytes_c_v = rocke_b_const_i32(ctx->b, kv_stride_blk_b);
    }
    return ctx->kv_block_bytes_c_v;
}

/* lane_half_base = tid * 8 (Python 1460). Emitted once in emit_preloop and cached
 * on ctx so every loader reuses the same SSA value (matches the single Python
 * local). */
static rocke_value_t* rocke__lane_half_base(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_half_base_v == NULL)
        ctx->lane_half_base_v = rocke_b_mul(ctx->b, ctx->tid, rocke_b_const_i32(ctx->b, 8));
    return ctx->lane_half_base_v;
}

/* seq_base = to_sgpr_u32(seq_idx * bt_stride_p) (Python 1525). Cached on ctx. */
static rocke_value_t* rocke__seq_base(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->seq_base == NULL)
        ctx->seq_base
            = rocke_b_to_sgpr_u32(ctx->b, rocke_b_mul(ctx->b, ctx->seq_idx, ctx->bt_stride_p));
    return ctx->seq_base;
}

/* block_table_max_idx = to_sgpr_u32(num_seqs_p * bt_stride_p) (Python 1538). */
static rocke_value_t* rocke__block_table_max_idx(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->block_table_max_idx == NULL)
        ctx->block_table_max_idx
            = rocke_b_to_sgpr_u32(ctx->b, rocke_b_mul(ctx->b, ctx->num_seqs_p, ctx->bt_stride_p));
    return ctx->block_table_max_idx;
}

/* K_lds_addr = ptrtoint(K_lds) (Python 1462); cached. */
static rocke_value_t* rocke__K_lds_addr(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->K_lds_addr_v == NULL)
        ctx->K_lds_addr_v = rocke_b_smem_addr_of(ctx->b, ctx->K_lds);
    return ctx->K_lds_addr_v;
}

/* V_lds_addr = ptrtoint(V_lds) (Python 1463); cached. */
static rocke_value_t* rocke__V_lds_addr(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->V_lds_addr_v == NULL)
        ctx->V_lds_addr_v = rocke_b_smem_addr_of(ctx->b, ctx->V_lds);
    return ctx->V_lds_addr_v;
}

/* zero_soff = const_i32(0) (Python 1466). Python creates this ONE const and
 * reuses it as the soffset in every async load; cache it so the loaders do not
 * each allocate a duplicate const. */
static rocke_value_t* rocke__zero_soff(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->zero_soff_v == NULL)
        ctx->zero_soff_v = rocke_b_const_i32(ctx->b, 0);
    return ctx->zero_soff_v;
}

/* wave_lds_offset_i64: const_i64(0) for NUM_WARPS==1, else the SGPR-pinned
 * zext(wave_id * WAVE_BYTES) (Python 1474-1485). Python creates it once and
 * reuses it; cache. */
static rocke_value_t* rocke__wave_lds_offset_i64(rocke_gfx950_attn2d_build_ctx_t* ctx)
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

/* ==========================================================================
 * Pre-loop: K/V buffer resources, byte derivation, paged-KV descriptor
 * (Python lines 1163-1630, the K/V-descriptor parts).
 * ========================================================================== */

void rocke_gfx950_attn2d_emit_preloop(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* big-bytes K/V buffer resources (1405-1407). The buffer rsrc bounds OOB
     * voffsets to return zero; size it large so valid offsets never trip it. */
    rocke_value_t* big_bytes = rocke_b_const_i32(b, 0x7FFF0000);
    ctx->k_rsrc = rocke_b_buffer_rsrc(b, ctx->key, big_bytes);
    ctx->v_rsrc = rocke_b_buffer_rsrc(b, ctx->value, big_bytes);

    /* The bf16 async-DMA byte derivation (1413-1416) is purely compile-time and
     * the loader closures recompute it; the asserted invariant emits no ops. */
    {
        int kv_halves_per_call = rocke__kv_halves_per_call(ctx);
        /* assert (T * HD) % KV_HALVES_PER_CALL == 0 (1414). */
        (void)((ctx->T * ctx->HD) % kv_halves_per_call);
    }

    /* kv_block_bytes_c (1458): Python creates this one-block-bound const here,
     * after the byte-stride derivation and BEFORE lane_half_base, and reuses it
     * in every loader. Emit it now so its SSA value lands in source order. */
    (void)rocke__kv_block_bytes_c(ctx);

    /* Per-lane starting half index (1460), then K/V LDS base ptrtoints (1462-
     * 1463). These are SINGLE Python locals consumed by every loader; emit them
     * here, in source order, and cache on ctx so the loaders reuse the same SSA
     * values (the byte-identity contract requires the exact op stream). */
    (void)rocke__lane_half_base(ctx);
    (void)rocke__K_lds_addr(ctx);
    (void)rocke__V_lds_addr(ctx);

    /* zero_soff (1466), then wave_lds_offset_i64 (1475/1485). Python creates each
     * as a single SSA value HERE (even the const_i64(0) consumes a value number)
     * and reuses them in every loader. Force their creation now, in source order,
     * and cache so the loaders reuse them instead of allocating duplicates --
     * otherwise their value numbers land later and shift the downstream stream. */
    (void)rocke__zero_soff(ctx);
    (void)rocke__wave_lds_offset_i64(ctx);

    /* seq_base (1525) + block_table_max_idx (1538): the per-sequence base index
     * into block_tables and the global footprint bound; both wave-uniform SGPRs.
     * Python emits these BEFORE the FAST_PAGED_KV_DESC branch, so both paths see
     * them in the same source position. */
    rocke_value_t* seq_base = rocke__seq_base(ctx);
    rocke_value_t* max_idx = rocke__block_table_max_idx(ctx);

    (void)rocke__bytes_per_buf;
    (void)rocke__bytes_per_call;
    (void)rocke__kv_calls_per_tile;

    /* FAST_PAGED_KV_DESC (1539): the gfx950 production fast paged-KV path emits no
     * standalone descriptor object -- the per-call byte offsets are built directly
     * by _fast_paged_kv_blocks / _fast_paged_kv_voff (1550-1589). */
    if(ctx->FAST_PAGED_KV_DESC)
    {
        ctx->kv_desc = NULL;
        /* fp8 async-loader setup (Python 1785-1796) runs after the paged-KV
         * descriptor branch on both paths. */
        rocke_gfx950_attn2d_emit_fp8_async_preloop_setup(ctx);
        return;
    }

    /* ---- Paged KV byte descriptor (full transform DAG, 1591-1629) ----
     * _kv_base = naive("paged_kv_bytes",
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

    /* fp8 async-loader setup (Python 1785-1796), after the paged-KV
     * descriptor DAG. */
    rocke_gfx950_attn2d_emit_fp8_async_preloop_setup(ctx);
}

/* ==========================================================================
 * Fast paged-KV byte-descriptor closures (Python lines 1550-1589).
 * ========================================================================== */

void rocke_gfx950_attn2d_fast_paged_kv_blocks(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t** out_block0,
                                              rocke_value_t** out_block1)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* seq_base = rocke__seq_base(ctx);
    rocke_value_t* max_idx = rocke__block_table_max_idx(ctx);

    /* logical_block0 = kv_tile_idx * 2 ; logical_block1 = logical_block0 + 1. */
    rocke_value_t* logical_block0 = rocke_b_mul(b, kv_tile_idx, rocke_b_const_i32(b, 2));
    rocke_value_t* logical_block1 = rocke_b_add(b, logical_block0, rocke_b_const_i32(b, 1));
    rocke_value_t* idx0 = rocke_b_add(b, seq_base, logical_block0);
    rocke_value_t* idx1 = rocke_b_add(b, seq_base, logical_block1);
    /* Python b.masked_global_load(table, idx, b.cmp_lt(idx, max), const(0))
     * creates the cmp_lt mask BEFORE the const(0) fill (left-to-right). Bind the
     * masks so C's right-to-left arg eval does not allocate const(0) first. */
    rocke_value_t* mask0 = rocke_b_cmp_lt(b, idx0, max_idx);
    rocke_value_t* block0 = rocke_b_masked_global_load(
        b, ctx->block_tables, idx0, mask0, rocke_b_const_i32(b, 0), rocke_i32(), 4);
    rocke_value_t* mask1 = rocke_b_cmp_lt(b, idx1, max_idx);
    rocke_value_t* block1 = rocke_b_masked_global_load(
        b, ctx->block_tables, idx1, mask1, rocke_b_const_i32(b, 0), rocke_i32(), 4);
    if(out_block0)
        *out_block0 = rocke_b_to_sgpr_u32(b, block0);
    if(out_block1)
        *out_block1 = rocke_b_to_sgpr_u32(b, block1);
}

/* Internal: the full Python tuple return of _fast_paged_kv_voff
 * (i64 block base, within-block i32 voffset). The within voffset is identical on
 * both paths; the i64 base is only produced on the I64_KV_ADDR path. */
static void rocke__fast_paged_kv_voff_split(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                            int call,
                                            rocke_value_t* block0,
                                            rocke_value_t* block1,
                                            rocke_value_t** out_base_i64,
                                            rocke_value_t** out_voff)
{
    rocke_ir_builder_t* b = ctx->b;
    /* physical = block0 if call == 0 else block1 (Python 1578). */
    rocke_value_t* physical = (call == 0) ? block0 : block1;
    rocke_value_t* lane_half_base = rocke__lane_half_base(ctx);

    /* token = lane_half_base / HD ; dim = lane_half_base % HD (HD=64 fast path). */
    rocke_value_t* token = rocke_b_lshr(b, lane_half_base, rocke_b_const_i32(b, 6));
    rocke_value_t* dim = rocke_b_land(b, lane_half_base, rocke_b_const_i32(b, 63));
    rocke_value_t* token_b = rocke_b_shl(b, token, rocke_b_const_i32(b, 10)); /* 8*64*2  */
    rocke_value_t* head_b = rocke_b_shl(b, ctx->kv_head_idx, rocke_b_const_i32(b, 7)); /* 64*2 */
    rocke_value_t* dim_b = rocke_b_shl(b, dim, rocke_b_const_i32(b, 1));
    rocke_value_t* within = rocke_b_add(b, rocke_b_add(b, token_b, head_b), dim_b);

    if(ctx->I64_KV_ADDR)
    {
        /* base_i64 = zext(physical, i64) << 15 ; return (base_i64, within). */
        rocke_value_t* base_i64
            = rocke_b_shl(b, rocke_b_zext(b, physical, rocke_i64()), rocke_b_const_i64(b, 15));
        if(out_base_i64)
            *out_base_i64 = base_i64;
        if(out_voff)
            *out_voff = within;
        return;
    }
    /* Legacy single i32 voffset: block_b = physical << 15 ; voff = block_b+within. */
    rocke_value_t* block_b = rocke_b_shl(b, physical, rocke_b_const_i32(b, 15));
    if(out_base_i64)
        *out_base_i64 = NULL;
    if(out_voff)
        *out_voff = rocke_b_add(b, block_b, within);
}

rocke_value_t* rocke_gfx950_attn2d_fast_paged_kv_voff(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                      int call,
                                                      rocke_value_t* block0,
                                                      rocke_value_t* block1)
{
    /* The internal-header form exposes the within/combined i32 voffset only; the
     * i64 block base (I64_KV_ADDR) is built by the loaders via the _split form. */
    rocke_value_t* voff = NULL;
    rocke__fast_paged_kv_voff_split(ctx, call, block0, block1, NULL, &voff);
    return voff;
}

/* ==========================================================================
 * Standard (bf16) K/V async-DMA loaders (Python lines 1631-1739).
 * ========================================================================== */

void rocke_gfx950_attn2d_issue_k_load_runtime(rocke_gfx950_attn2d_build_ctx_t* ctx,
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

    /* buf_off = buf_idx * bytes_per_buf ; K_buf_base = K_lds_addr + buf_off ;
     * K_wave_base = K_buf_base + wave_lds_offset (1646-1649). */
    rocke_value_t* buf_off_i32 = rocke_b_mul(b, buf_idx, rocke_b_const_i32(b, bytes_per_buf));
    rocke_value_t* buf_off_i64 = rocke_b_zext(b, buf_off_i32, rocke_i64());
    rocke_value_t* K_buf_base = rocke_b_smem_ptr_add(b, K_lds_addr, buf_off_i64);
    rocke_value_t* K_wave_base = rocke_b_smem_ptr_add(b, K_buf_base, wave_off);

    rocke_value_t* fast_block0 = NULL;
    rocke_value_t* fast_block1 = NULL;
    if(ctx->FAST_PAGED_KV_DESC)
        rocke_gfx950_attn2d_fast_paged_kv_blocks(ctx, kv_tile_idx, &fast_block0, &fast_block1);

    for(int call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* k_rsrc = ctx->k_rsrc;
        rocke_value_t* voff = NULL;
        if(ctx->FAST_PAGED_KV_DESC)
        {
            rocke_value_t* base_i64 = NULL;
            rocke__fast_paged_kv_voff_split(ctx, call, fast_block0, fast_block1, &base_i64, &voff);
            if(base_i64 != NULL) /* I64_KV_ADDR: per-block 64-bit base */
            {
                k_rsrc = rocke_b_buffer_rsrc(
                    b, rocke_b_global_ptr_add(b, ctx->key, base_i64), kv_block_bytes_c);
            }
        }
        else
        {
            /* Non-fast paged-KV (N_BLOCKS_PER_TILE==1 / multi-block):
             *   linear_half = call*KV_HALVES_PER_CALL + lane_half_base
             *   voff, _ = paged_kv_desc.offset(tile_idx=, linear_half=, kv_head=)
             * (Python 1660-1681). I64_KV_ADDR uses offset_i64_split, folding the
             * physical_block term into a 64-bit buffer base. */
            rocke_value_t* linear_half
                = rocke_b_add(b, rocke_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            rocke_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            rocke_value_t* valid = NULL;
            if(ctx->I64_KV_ADDR)
            {
                rocke_value_t* base_i64 = NULL;
                if(!rocke_transforms_descriptor_offset_i64_split(b,
                                                                 ctx->kv_desc,
                                                                 "physical_block",
                                                                 in_names,
                                                                 in_values,
                                                                 3,
                                                                 &base_i64,
                                                                 &voff,
                                                                 &valid))
                    return;
                k_rsrc = rocke_b_buffer_rsrc(
                    b, rocke_b_global_ptr_add(b, ctx->key, base_i64), kv_block_bytes_c);
            }
            else if(!rocke_transforms_descriptor_offset(
                        b, ctx->kv_desc, in_names, in_values, 3, &voff, &valid))
                return;
        }
        /* k_dst = K_wave_base + call*bytes_per_call (1682). */
        rocke_value_t* k_dst = rocke_b_smem_ptr_add(
            b, K_wave_base, rocke_b_const_i64(b, (int64_t)call * bytes_per_call));
        /* CACHE_STREAM (SLC): one-shot streaming load; dwords=4 (1687-1689). */
        rocke_b_async_buffer_load_lds_addr(
            b, k_rsrc, k_dst, voff, zero_soff, 4, ROCKE_CACHE_STREAM);
    }
}

void rocke_gfx950_attn2d_issue_v_load_runtime(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    int bytes_per_call = rocke__bytes_per_call(ctx);
    int kv_calls_per_tile = rocke__kv_calls_per_tile(ctx);
    int kv_halves_per_call = rocke__kv_halves_per_call(ctx);

    (void)buf_idx; /* V is single-buffered; always slot 0 (Python 1699). */
    rocke_value_t* V_lds_addr = rocke__V_lds_addr(ctx);
    rocke_value_t* wave_off = rocke__wave_lds_offset_i64(ctx);
    rocke_value_t* zero_soff = rocke__zero_soff(ctx);
    rocke_value_t* kv_block_bytes_c = rocke__kv_block_bytes_c(ctx);
    rocke_value_t* lane_half_base = rocke__lane_half_base(ctx);

    /* V_wave_base = V_lds_addr + wave_lds_offset (1700). */
    rocke_value_t* V_wave_base = rocke_b_smem_ptr_add(b, V_lds_addr, wave_off);

    rocke_value_t* fast_block0 = NULL;
    rocke_value_t* fast_block1 = NULL;
    if(ctx->FAST_PAGED_KV_DESC)
        rocke_gfx950_attn2d_fast_paged_kv_blocks(ctx, kv_tile_idx, &fast_block0, &fast_block1);

    for(int call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* v_rsrc = ctx->v_rsrc;
        rocke_value_t* voff = NULL;
        if(ctx->FAST_PAGED_KV_DESC)
        {
            rocke_value_t* base_i64 = NULL;
            rocke__fast_paged_kv_voff_split(ctx, call, fast_block0, fast_block1, &base_i64, &voff);
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
             * (Python 1711-1732). I64_KV_ADDR uses offset_i64_split, folding the
             * physical_block term into a 64-bit buffer base. */
            rocke_value_t* linear_half
                = rocke_b_add(b, rocke_b_const_i32(b, call * kv_halves_per_call), lane_half_base);
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            rocke_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            rocke_value_t* valid = NULL;
            if(ctx->I64_KV_ADDR)
            {
                rocke_value_t* base_i64 = NULL;
                if(!rocke_transforms_descriptor_offset_i64_split(b,
                                                                 ctx->kv_desc,
                                                                 "physical_block",
                                                                 in_names,
                                                                 in_values,
                                                                 3,
                                                                 &base_i64,
                                                                 &voff,
                                                                 &valid))
                    return;
                v_rsrc = rocke_b_buffer_rsrc(
                    b, rocke_b_global_ptr_add(b, ctx->value, base_i64), kv_block_bytes_c);
            }
            else if(!rocke_transforms_descriptor_offset(
                        b, ctx->kv_desc, in_names, in_values, 3, &voff, &valid))
                return;
        }
        /* v_dst = V_wave_base + call*bytes_per_call (1733). */
        rocke_value_t* v_dst = rocke_b_smem_ptr_add(
            b, V_wave_base, rocke_b_const_i64(b, (int64_t)call * bytes_per_call));
        /* CACHE_STREAM (SLC): V consumed once per iter; dwords=4 (1737-1739). */
        rocke_b_async_buffer_load_lds_addr(
            b, v_rsrc, v_dst, voff, zero_soff, 4, ROCKE_CACHE_STREAM);
    }
}

/* ==========================================================================
 * Loader dispatch wrappers (Python lines 2184-2227).
 *
 * Three K paths: bf16 async DMA / fp8 sync dequant (KV_FP8) / fp8 native MFMA
 * (FP8_MFMA_QK). Three V paths: fp8 stripe (FP8_MFMA_PV) / fp8 sync dequant
 * (KV_FP8) / bf16 async DMA. The fp8 loaders live in a peer TU; call them via
 * the internal header.
 * ========================================================================== */

void rocke_gfx950_attn2d_issue_k(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx)
{
    if(ctx->FP8_MFMA_QK) /* K_FP8_MFMA == FP8_MFMA_QK */
        rocke_gfx950_attn2d_issue_k_fp8_mfma_async(ctx, tile_idx, buf_idx);
    else if(ctx->KV_FP8)
        rocke_gfx950_attn2d_issue_fp8_dequant_loads(ctx, tile_idx, buf_idx, false);
    else
        rocke_gfx950_attn2d_issue_k_load_runtime(ctx, tile_idx, buf_idx);
}

void rocke_gfx950_attn2d_issue_v(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx)
{
    if(ctx->FP8_MFMA_PV) /* PV_FP8_MFMA == FP8_MFMA_PV */
        rocke_gfx950_attn2d_issue_v_fp8_mfma_stripe(ctx, tile_idx);
    else if(ctx->KV_FP8)
        /* V is single-buffered: always pin slot 0 (Python 2225). */
        rocke_gfx950_attn2d_issue_fp8_dequant_loads(
            ctx, tile_idx, rocke_b_const_i32(ctx->b, 0), true);
    else
        rocke_gfx950_attn2d_issue_v_load_runtime(ctx, tile_idx, buf_idx);
}

/* ==========================================================================
 * _read_k8_mfma_operand (Python lines 2229-2248).
 *
 * Read 8 K elements from K_lds as the bf16 MFMA operand. For the fp8-in-LDS path
 * (K_FP8_MFMA == FP8_MFMA_QK) K_lds holds raw fp8 bytes; dequant in register via
 * dequant_fp8x8_to_dtype (UNFUSED cvt + explicit fmul k_scale). FP8_NATIVE_QK is
 * a compile-time False on gfx950, so its branch is dead; keep it for parity.
 * Otherwise read bf16 directly. The buf_idx / k_row / k_off are passed by the
 * caller's QK loop (the gfx950 internal-header form takes them as SSA values).
 * ========================================================================== */
rocke_value_t* rocke_gfx950_attn2d_read_k8_mfma_operand(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                        rocke_value_t* buf_idx,
                                                        rocke_value_t* k_row,
                                                        rocke_value_t* k_off)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* idx[3];
    idx[0] = buf_idx;
    idx[1] = k_row;
    idx[2] = k_off;

    /* not K_FP8_MFMA: plain bf16 read (Python 2241-2242). */
    if(!ctx->FP8_MFMA_QK)
        return rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, ctx->dtype, 8);

    /* FP8_NATIVE_QK is always False on gfx950 (Python 793 / 2243-2246): the
     * native-fp8 raw read is dead, but emit it to mirror the source structure. */
    /* (FP8_NATIVE_QK == False -> fall through to the register dequant path.) */

    /* fp8 register dequant: read 8 raw fp8 then cvt + *k_scale + cast (2247-2248). */
    rocke_value_t* k_fp8 = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, rocke_fp8e4m3(), 8);
    return rocke_dequant_fp8x8_to_dtype(b, k_fp8, ctx->k_scale_p, ctx->dtype);
}
