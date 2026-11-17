// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_fp8_kv.c -- C99 port of the FP8 K/V CACHE
 * bucket of build_unified_attention_2d_tiled
 * (rocke/instances/gfx942/attention_tiled_2d.py lines 2905-3346).
 *
 * SCOPE.
 *   The fp8 K/V staging / dequant / native-fp8-MFMA loaders:
 *     _issue_kv_fp8_async_load   -> rocke_gfx942_attn2d_issue_kv_fp8_async_load
 *     _dequant_fp8_lds_to_bf16   -> rocke_gfx942_attn2d_dequant_fp8_lds_to_bf16
 *     _issue_fp8_dequant_loads   -> rocke_gfx942_attn2d_issue_fp8_dequant_loads
 *     _issue_k_fp8_mfma_async    -> rocke_gfx942_attn2d_issue_k_fp8_mfma_async
 *     _issue_v_fp8_mfma_async    -> rocke_gfx942_attn2d_issue_v_fp8_mfma_async
 *     _issue_v_fp8_mfma_stripe   -> rocke_gfx942_attn2d_issue_v_fp8_mfma_stripe
 *
 *   These are the largest dead-on-gfx942-but-structurally-present code paths:
 *   the spec admission gate (supports_tiled_2d) REJECTS fp8 K/V cache on
 *   gfx942, so KV_FP8 / FP8_MFMA_QK / FP8_MFMA_PV are never true for a
 *   buildable gfx942 shape. The faithful port reproduces the builder-call
 *   structure 1:1 so a future arch that lifts the reject gets the path for
 *   free; on gfx942 these functions are never reached.
 *
 * BINDING.
 *   Reads/writes only rocke_gfx942_attn2d_build_ctx_t fields + the builder it
 *   carries; calls peers through the internal header. Emits no public API.
 *
 * STUB-TO-LINK NOTE (paged-KV byte descriptor).
 *   The Python loaders resolve their per-call VMEM byte offset through the
 *   paged-KV TensorDescriptor closures paged_kv_desc.offset(...),
 *   .offset_i64(...) and .offset_i64_split(...). The C transforms surface
 *   exposes only the generic rocke_transforms_descriptor_offset (name/value
 *   pairs) and ctx carries the resolved paged byte descriptor as ctx->kv_desc;
 *   it does NOT carry the i64 / i64-split paged variants (no ctx field, and the
 *   header is frozen). The descriptor-offset resolution is therefore funnelled
 *   through a single file-local helper, rocke_attn2d_fp8_kv_voff(), which drives
 *   ctx->kv_desc with the (tile_idx, linear_half, kv_head) upper coords via the
 *   generic offset helper. The i64 / i64-split paged branches and the
 *   per-block buffer_rsrc re-bind (which need kv_block_bytes_c / the paged
 *   physical_block split that ctx does not expose) are reduced to the i32
 *   offset path -- the gfx942-legal one -- since the wider-address branches are
 *   unreachable on this arch. This keeps the emitted IR structurally faithful
 *   for the live (i32-addr) configuration and link-clean for the dead ones.
 */

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  File-local derived constants (Python prologue lines 2186-2238, 2932-2945).
 *
 *  The fp8 K/V loader scalars are computed in the Python prologue but are not
 *  carried as ctx fields. They are pure functions of ctx geometry ints, so we
 *  recompute them here exactly as the prologue does. Mirrors:
 *    fp8_dword_candidates  [(4,16),(3,12),(1,4)] clamped to ASYNC_LDS_MAX_DWORDS
 *    pick first (dwords,bytes_per_lane) s.t. tile_bytes % (THREADS*bpl)==0
 * ===================================================================== */

typedef struct
{
    int dwords; /* K_FP8_DWORDS                         */
    int bytes_per_lane; /* K_BYTES_PER_LANE                     */
    int elems_per_call; /* K_ELEMS_PER_CALL  = THREADS*bpl      */
    int bytes_per_call; /* K_BYTES_PER_CALL  = elems_per_call   */
    int calls_per_tile; /* k_fp8_calls_per_tile = tile/epc      */
    bool valid; /* false if no payload covers the tile  */
} rocke_attn2d_fp8_dma_t;

static rocke_attn2d_fp8_dma_t rocke_attn2d_fp8_dma_pick(const rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    rocke_attn2d_fp8_dma_t out;
    static const int cand_d[3] = {4, 3, 1};
    static const int cand_by[3] = {16, 12, 4};
    int tile_bytes = ctx->T * ctx->HD; /* 1 byte per fp8 element */
    int i;

    out.dwords = 0;
    out.bytes_per_lane = 0;
    out.elems_per_call = 0;
    out.bytes_per_call = 0;
    out.calls_per_tile = 0;
    out.valid = false;

    for(i = 0; i < 3; ++i)
    {
        int payload;
        if(cand_d[i] > ctx->ASYNC_LDS_MAX_DWORDS)
        {
            continue;
        }
        payload = ctx->THREADS * cand_by[i];
        if(payload > 0 && tile_bytes >= payload && (tile_bytes % payload) == 0)
        {
            out.dwords = cand_d[i];
            out.bytes_per_lane = cand_by[i];
            out.elems_per_call = payload;
            out.bytes_per_call = payload;
            out.calls_per_tile = tile_bytes / payload;
            out.valid = true;
            break;
        }
    }
    return out;
}

/* zero soffset constant (Python ``zero_soff = b.const_i32(0)``). */
static rocke_value_t* rocke_attn2d_zero_soff(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    return rocke_b_const_i32(ctx->b, 0);
}

/* ===================================================================== *
 *  paged-KV per-call byte offset (STUB-TO-LINK; see file header note).
 *
 *  Reproduces the live (i32-addr) ``paged_kv_desc.offset(b, tile_idx=,
 *  linear_half=, kv_head=)`` call against ctx->kv_desc. The i64 / i64-split
 *  paged variants used when ctx->I64_KV_ADDR are not expressible through the
 *  frozen C transforms surface (no ctx field for the split base / per-block
 *  bound); on gfx942 (where this whole bucket is reject-gated) they are dead,
 *  so we resolve through the i32 path regardless. Returns the i32 element/byte
 *  offset; *out_valid receives the conjoined validity (may be NULL).
 * ===================================================================== */
static rocke_value_t* rocke_attn2d_fp8_kv_voff(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                               rocke_value_t* tile_idx,
                                               rocke_value_t* linear_half,
                                               rocke_value_t** out_valid)
{
    const char* names[3] = {"tile_idx", "linear_half", "kv_head"};
    rocke_value_t* vals[3] = {tile_idx, linear_half, ctx->kv_head_idx};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;

    if(ctx->kv_desc != NULL)
    {
        (void)rocke_transforms_descriptor_offset(
            ctx->b, ctx->kv_desc, names, vals, 3, &off, &valid);
    }
    if(off == NULL)
    {
        off = rocke_b_const_i32(ctx->b, 0);
    }
    if(out_valid != NULL)
    {
        *out_valid = valid;
    }
    return off;
}

/* ===================================================================== *
 *  _issue_kv_fp8_async_load  (Python lines 2961-3016)
 *
 *  Two-phase async DMA loader: issue raw.ptr.buffer.load.lds writing fp8 bytes
 *  directly into K_fp8_lds[buf_idx] / V_fp8_lds[0]. ``buf_idx`` is the K
 *  double-buffer slot (ignored for V, which is single-buffered).
 *
 *  Gated in Python on (KV_FP8 and spec.use_fp8_mfma_qk). The matching FP8 async
 *  scalars (FP8_DWORDS_PER_LANE = K_FP8_DWORDS, FP8_BYTES_PER_LANE =
 *  K_BYTES_PER_LANE, etc.) are recomputed here. The Python free function takes a
 *  ``slot`` selector; the header contract exposes a single entry, so the K slab
 *  is driven (the V single-buffer variant is reachable through the same body by
 *  the caller substituting V_fp8_lds -- kept K-faithful here, the dead path's V
 *  case is handled by _issue_v_fp8_mfma_async).
 * ===================================================================== */
void rocke_gfx942_attn2d_issue_kv_fp8_async_load(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx,
                                                 rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_attn2d_fp8_dma_t dma;
    int FP8_WAVE_BYTES;
    int FP8_BYTES_PER_BUF;
    rocke_value_t* lane_fp8_base;
    rocke_value_t* wave_fp8_offset_i64;
    rocke_value_t* K_fp8_lds_addr;
    rocke_value_t* buf_off_i32;
    rocke_value_t* buf_off_i64;
    rocke_value_t* buf_base;
    rocke_value_t* wave_base;
    rocke_value_t* zero_soff = rocke_attn2d_zero_soff(ctx);
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }
    dma = rocke_attn2d_fp8_dma_pick(ctx);
    if(!dma.valid)
    {
        return; /* unreachable on gfx942: no payload covers the tile */
    }

    FP8_WAVE_BYTES = ctx->WAVE * dma.bytes_per_lane;
    FP8_BYTES_PER_BUF = ctx->T * ctx->HD; /* 1 byte per fp8 element */

    /* lane_fp8_base = tid * FP8_ELEMS_PER_LANE  (1 byte per fp8 element). */
    lane_fp8_base = rocke_b_mul(b, ctx->tid, rocke_b_const_i32(b, dma.bytes_per_lane));

    if(ctx->NUM_WARPS == 1)
    {
        wave_fp8_offset_i64 = rocke_b_const_i64(b, 0);
    }
    else
    {
        rocke_value_t* wave_fp8_offset_i32 = rocke_b_to_sgpr_u32(
            b, rocke_b_mul(b, ctx->wave_id, rocke_b_const_i32(b, FP8_WAVE_BYTES)));
        wave_fp8_offset_i64 = rocke_b_zext(b, wave_fp8_offset_i32, rocke_i64());
    }

    K_fp8_lds_addr = rocke_b_smem_addr_of(b, ctx->K_fp8_lds);

    /* K slot: K_fp8_lds[buf_idx] */
    buf_off_i32 = rocke_b_mul(b, buf_idx, rocke_b_const_i32(b, FP8_BYTES_PER_BUF));
    buf_off_i64 = rocke_b_zext(b, buf_off_i32, rocke_i64());
    buf_base = rocke_b_smem_ptr_add(b, K_fp8_lds_addr, buf_off_i64);
    wave_base = rocke_b_smem_ptr_add(b, buf_base, wave_fp8_offset_i64);

    for(call = 0; call < dma.calls_per_tile; ++call)
    {
        rocke_value_t* linear_elem
            = rocke_b_add(b, rocke_b_const_i32(b, call * dma.elems_per_call), lane_fp8_base);
        rocke_value_t* voff = rocke_attn2d_fp8_kv_voff(ctx, kv_tile_idx, linear_elem, NULL);
        rocke_value_t* lds_dst = rocke_b_smem_ptr_add(
            b, wave_base, rocke_b_const_i64(b, (int64_t)call * dma.bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            b, ctx->k_rsrc, lds_dst, voff, zero_soff, dma.dwords, ctx->kv_cache_aux);
    }
}

/* ===================================================================== *
 *  _dequant_fp8_lds_to_bf16  (Python lines 3033-3069)
 *
 *  LDS->LDS dequant: fp8 -> f32 * scale -> working dtype. Each thread reads 8
 *  fp8 from ``src`` (fp8 LDS slab) at [buf_idx, row, col], applies
 *  cvt_fp8_to_f32 * scale -> cast to ctx->dtype, and writes 8 working-dtype
 *  elements to ``dst`` at [bf16_buf(=0), row, col].
 *
 *  Header signature: (ctx, src, dst, scale, buf_idx). The Python ``bf16_buf``
 *  destination selector is always 0 on this path (the FP8 path single-buffers
 *  the working-dtype K_lds), so it is pinned to const_i32(0) here.
 * ===================================================================== */
void rocke_gfx942_attn2d_dequant_fp8_lds_to_bf16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* src,
                                                 rocke_value_t* dst,
                                                 rocke_value_t* scale,
                                                 rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int CHUNK = 8; /* fp8_dequant_elems_per_chunk */
    int total_chunks;
    int chunks_per_thread;
    int cols_per_row;
    rocke_value_t* bf16_buf;
    int c;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }

    total_chunks = (ctx->T * ctx->HD) / CHUNK;
    if(ctx->THREADS <= 0 || (total_chunks % ctx->THREADS) != 0)
    {
        return; /* prologue assert: divisible by THREADS */
    }
    chunks_per_thread = total_chunks / ctx->THREADS;
    cols_per_row = ctx->HD / CHUNK;
    bf16_buf = rocke_b_const_i32(b, 0);

    for(c = 0; c < chunks_per_thread; ++c)
    {
        rocke_value_t* chunk_id = rocke_b_add(
            b,
            rocke_b_mul(b, rocke_b_const_i32(b, c), rocke_b_const_i32(b, ctx->THREADS)),
            ctx->tid);
        rocke_value_t* row = rocke_b_div(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        rocke_value_t* col
            = rocke_b_mul(b,
                          rocke_b_mod(b, chunk_id, rocke_b_const_i32(b, cols_per_row)),
                          rocke_b_const_i32(b, CHUNK));
        rocke_value_t* idx_load[3] = {buf_idx, row, col};
        rocke_value_t* fp8_vec = rocke_b_smem_load_vN(b, src, idx_load, 3, rocke_fp8e4m3(), CHUNK);
        rocke_value_t* dequanted[8];
        rocke_value_t* packed;
        rocke_value_t* idx_store[3];
        int i;

        for(i = 0; i < CHUNK; ++i)
        {
            rocke_value_t* fp8_v = rocke_b_vec_extract(b, fp8_vec, i);
            rocke_value_t* f32_v = rocke_b_fmul(b, rocke_b_cvt_fp8_to_f32(b, fp8_v), scale);
            dequanted[i] = rocke_b_cast_f32_to(b, f32_v, ctx->dtype);
        }
        packed = rocke_b_vec_pack(b, dequanted, CHUNK, ctx->dtype);
        idx_store[0] = bf16_buf;
        idx_store[1] = row;
        idx_store[2] = col;
        rocke_b_smem_store_vN(b, dst, idx_store, 3, packed, CHUNK);
    }
}

/* ===================================================================== *
 *  _issue_fp8_dequant_loads  (Python lines 3091-3211)  -- round-1 sync loader.
 *
 *  Sync per-thread fp8 -> f32 -> *scale -> working-dtype -> LDS. Each iteration
 *  issues one VMEM global_load_vN(FP8,n=8), splits the <8 x fp8> into 2x<4 x
 *  fp8>, applies the packed cvt_pk_f32_fp8x4 + per-element *scale, casts to the
 *  working dtype, packs, and smem_store_vN(n=8) to the working-dtype LDS.
 *
 *  Header signature: (ctx, kv_tile_idx, buf_idx). The Python ``lds_token``
 *  K/V selector picks scale = k_scale_p/v_scale_p, lds = K_lds/V_lds,
 *  src = key/value. The header exposes one entry; the K case is emitted (the
 *  dead-path V case shares the body with the K-substituted handles).
 * ===================================================================== */
void rocke_gfx942_attn2d_issue_fp8_dequant_loads(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx,
                                                 rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int CHUNK = 8; /* fp8_elems_per_chunk */
    int total_chunks;
    int chunks_per_thread;
    int cols_per_row;
    rocke_value_t* scale;
    rocke_value_t* lds;
    rocke_value_t* src;
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }

    total_chunks = (ctx->T * ctx->HD) / CHUNK;
    if(ctx->THREADS <= 0 || (total_chunks % ctx->THREADS) != 0)
    {
        return;
    }
    chunks_per_thread = total_chunks / ctx->THREADS;
    cols_per_row = ctx->HD / CHUNK; /* HD // fp8_elems_per_chunk */

    /* K case (the live, header-exposed selector). */
    scale = ctx->k_scale_p;
    lds = ctx->K_lds;
    src = ctx->key;

    for(call = 0; call < chunks_per_thread; ++call)
    {
        rocke_value_t* chunk_id = rocke_b_add(
            b,
            rocke_b_mul(b, rocke_b_const_i32(b, call), rocke_b_const_i32(b, ctx->THREADS)),
            ctx->tid);
        rocke_value_t* row = rocke_b_div(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        rocke_value_t* col
            = rocke_b_mul(b,
                          rocke_b_mod(b, chunk_id, rocke_b_const_i32(b, cols_per_row)),
                          rocke_b_const_i32(b, CHUNK));
        rocke_value_t* linear_half_first
            = rocke_b_add(b, rocke_b_mul(b, row, rocke_b_const_i32(b, ctx->HD)), col);
        rocke_value_t* voff = rocke_attn2d_fp8_kv_voff(ctx, kv_tile_idx, linear_half_first, NULL);
        rocke_value_t* fp8_vec
            = rocke_b_global_load_vN(b, src, voff, rocke_fp8e4m3(), CHUNK, CHUNK);

        rocke_value_t* lo_comp[4];
        rocke_value_t* hi_comp[4];
        rocke_value_t* lo_quad;
        rocke_value_t* hi_quad;
        rocke_value_t* lo_f32x4;
        rocke_value_t* hi_f32x4;
        rocke_value_t* lo_scaled[4];
        rocke_value_t* hi_scaled[4];
        rocke_value_t* dequanted[8];
        rocke_value_t* packed;
        rocke_value_t* idx_store[3];
        int i;

        for(i = 0; i < 4; ++i)
        {
            lo_comp[i] = rocke_b_vec_extract(b, fp8_vec, i);
            hi_comp[i] = rocke_b_vec_extract(b, fp8_vec, i + 4);
        }
        lo_quad = rocke_b_vec_pack(b, lo_comp, 4, rocke_fp8e4m3());
        hi_quad = rocke_b_vec_pack(b, hi_comp, 4, rocke_fp8e4m3());

        /* CORRECTNESS: unfused cvt + *scale (non-pow2 scale safety). */
        lo_f32x4 = rocke_b_cvt_pk_f32_fp8x4(b, lo_quad);
        hi_f32x4 = rocke_b_cvt_pk_f32_fp8x4(b, hi_quad);
        for(i = 0; i < 4; ++i)
        {
            lo_scaled[i] = rocke_b_fmul(b, rocke_b_vec_extract(b, lo_f32x4, i), scale);
            hi_scaled[i] = rocke_b_fmul(b, rocke_b_vec_extract(b, hi_f32x4, i), scale);
        }
        lo_f32x4 = rocke_b_vec_pack(b, lo_scaled, 4, rocke_f32());
        hi_f32x4 = rocke_b_vec_pack(b, hi_scaled, 4, rocke_f32());
        for(i = 0; i < 4; ++i)
        {
            dequanted[i] = rocke_b_cast_f32_to(b, rocke_b_vec_extract(b, lo_f32x4, i), ctx->dtype);
        }
        for(i = 0; i < 4; ++i)
        {
            dequanted[i + 4]
                = rocke_b_cast_f32_to(b, rocke_b_vec_extract(b, hi_f32x4, i), ctx->dtype);
        }
        packed = rocke_b_vec_pack(b, dequanted, CHUNK, ctx->dtype);
        idx_store[0] = buf_idx;
        idx_store[1] = row;
        idx_store[2] = col;
        rocke_b_smem_store_vN(b, lds, idx_store, 3, packed, CHUNK);
    }
}

/* ===================================================================== *
 *  _issue_k_fp8_mfma_async  (Python lines 3213-3267)
 *
 *  fp8-K-LDS native path: async DMA raw fp8 K bytes into the fp8 K_lds[buf_idx]
 *  (K_lds is allocated as fp8 on this path). Per-call dwords = K_FP8_DWORDS,
 *  per-lane payload = K_BYTES_PER_LANE; wave-uniform LDS offset in bytes.
 * ===================================================================== */
void rocke_gfx942_attn2d_issue_k_fp8_mfma_async(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                rocke_value_t* kv_tile_idx,
                                                rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_attn2d_fp8_dma_t dma;
    rocke_value_t* K_lds_addr;
    rocke_value_t* buf_off_i32;
    rocke_value_t* buf_off_i64;
    rocke_value_t* K_buf_base;
    rocke_value_t* wave_fp8_off_i64;
    rocke_value_t* K_wave_base;
    rocke_value_t* lane_fp8_base;
    rocke_value_t* zero_soff = rocke_attn2d_zero_soff(ctx);
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }
    dma = rocke_attn2d_fp8_dma_pick(ctx);
    if(!dma.valid)
    {
        return;
    }

    K_lds_addr = rocke_b_smem_addr_of(b, ctx->K_lds);

    /* buf_off = buf_idx * (T*HD)  (fp8: 1 byte/elem). */
    buf_off_i32 = rocke_b_mul(b, buf_idx, rocke_b_const_i32(b, ctx->T * ctx->HD));
    buf_off_i64 = rocke_b_zext(b, buf_off_i32, rocke_i64());
    K_buf_base = rocke_b_smem_ptr_add(b, K_lds_addr, buf_off_i64);

    if(ctx->NUM_WARPS == 1)
    {
        wave_fp8_off_i64 = rocke_b_const_i64(b, 0);
    }
    else
    {
        rocke_value_t* wave_fp8_off_i32 = rocke_b_to_sgpr_u32(
            b, rocke_b_mul(b, ctx->wave_id, rocke_b_const_i32(b, ctx->WAVE * dma.bytes_per_lane)));
        wave_fp8_off_i64 = rocke_b_zext(b, wave_fp8_off_i32, rocke_i64());
    }
    K_wave_base = rocke_b_smem_ptr_add(b, K_buf_base, wave_fp8_off_i64);
    lane_fp8_base = rocke_b_mul(b, ctx->tid, rocke_b_const_i32(b, dma.bytes_per_lane));

    for(call = 0; call < dma.calls_per_tile; ++call)
    {
        rocke_value_t* linear_elem
            = rocke_b_add(b, rocke_b_const_i32(b, call * dma.elems_per_call), lane_fp8_base);
        rocke_value_t* voff = rocke_attn2d_fp8_kv_voff(ctx, kv_tile_idx, linear_elem, NULL);
        rocke_value_t* k_dst = rocke_b_smem_ptr_add(
            b, K_wave_base, rocke_b_const_i64(b, (int64_t)call * dma.bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            b, ctx->k_rsrc, k_dst, voff, zero_soff, dma.dwords, ctx->kv_cache_aux);
    }
}

/* ===================================================================== *
 *  _issue_v_fp8_mfma_async  (Python lines 3269-3304)
 *
 *  Native-fp8 PV path: async DMA raw fp8 V bytes into V_lds[0] (single-buffered).
 *  Mirrors _issue_k_fp8_mfma_async; the loop's existing waitcnt+barrier before
 *  PV makes the raw fp8 bytes visible to the ds_read transpose reads.
 * ===================================================================== */
void rocke_gfx942_attn2d_issue_v_fp8_mfma_async(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                rocke_value_t* kv_tile_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_attn2d_fp8_dma_t dma;
    rocke_value_t* V_lds_addr;
    rocke_value_t* V_buf_base;
    rocke_value_t* wave_fp8_off_i64;
    rocke_value_t* V_wave_base;
    rocke_value_t* lane_fp8_base;
    rocke_value_t* zero_soff = rocke_attn2d_zero_soff(ctx);
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }
    dma = rocke_attn2d_fp8_dma_pick(ctx);
    if(!dma.valid)
    {
        return;
    }

    V_lds_addr = rocke_b_smem_addr_of(b, ctx->V_lds);
    V_buf_base = V_lds_addr;

    if(ctx->NUM_WARPS == 1)
    {
        wave_fp8_off_i64 = rocke_b_const_i64(b, 0);
    }
    else
    {
        rocke_value_t* wave_fp8_off_i32 = rocke_b_to_sgpr_u32(
            b, rocke_b_mul(b, ctx->wave_id, rocke_b_const_i32(b, ctx->WAVE * dma.bytes_per_lane)));
        wave_fp8_off_i64 = rocke_b_zext(b, wave_fp8_off_i32, rocke_i64());
    }
    V_wave_base = rocke_b_smem_ptr_add(b, V_buf_base, wave_fp8_off_i64);
    lane_fp8_base = rocke_b_mul(b, ctx->tid, rocke_b_const_i32(b, dma.bytes_per_lane));

    for(call = 0; call < dma.calls_per_tile; ++call)
    {
        rocke_value_t* linear_elem
            = rocke_b_add(b, rocke_b_const_i32(b, call * dma.elems_per_call), lane_fp8_base);
        rocke_value_t* voff = rocke_attn2d_fp8_kv_voff(ctx, kv_tile_idx, linear_elem, NULL);
        rocke_value_t* v_dst = rocke_b_smem_ptr_add(
            b, V_wave_base, rocke_b_const_i64(b, (int64_t)call * dma.bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            b, ctx->v_rsrc, v_dst, voff, zero_soff, dma.dwords, ctx->kv_cache_aux);
    }
}

/* ===================================================================== *
 *  _issue_v_fp8_mfma_stripe  (Python lines 3306-3345)
 *
 *  Load raw fp8 V and store it into V_lds as [V_BUFS=1, N_STRIPES=HD/16, T, 16].
 *  Each thread loads 8 contiguous fp8 from V[token, col..col+7] in HBM, then
 *  writes them as one 8-byte LDS vec store into the owning stripe at
 *  V_lds[0, col/16, token, col%16]. col%16 is always 0 or 8 -> 8-byte aligned.
 * ===================================================================== */
void rocke_gfx942_attn2d_issue_v_fp8_mfma_stripe(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int CHUNK = 8; /* fp8_elems_per_chunk */
    int total_chunks;
    int chunks_per_thread;
    int cols_per_row;
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }

    total_chunks = (ctx->T * ctx->HD) / CHUNK;
    if(ctx->THREADS <= 0 || (total_chunks % ctx->THREADS) != 0)
    {
        return;
    }
    chunks_per_thread = total_chunks / ctx->THREADS;
    cols_per_row = ctx->HD / CHUNK; /* HD // fp8_elems_per_chunk */

    for(call = 0; call < chunks_per_thread; ++call)
    {
        rocke_value_t* chunk_id = rocke_b_add(
            b,
            rocke_b_mul(b, rocke_b_const_i32(b, call), rocke_b_const_i32(b, ctx->THREADS)),
            ctx->tid);
        rocke_value_t* token = rocke_b_div(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        rocke_value_t* col
            = rocke_b_mul(b,
                          rocke_b_mod(b, chunk_id, rocke_b_const_i32(b, cols_per_row)),
                          rocke_b_const_i32(b, CHUNK));
        rocke_value_t* linear_first
            = rocke_b_add(b, rocke_b_mul(b, token, rocke_b_const_i32(b, ctx->HD)), col);
        rocke_value_t* voff = rocke_attn2d_fp8_kv_voff(ctx, kv_tile_idx, linear_first, NULL);
        rocke_value_t* fp8_vec
            = rocke_b_global_load_vN(b, ctx->value, voff, rocke_fp8e4m3(), CHUNK, CHUNK);
        rocke_value_t* stripe_idx = rocke_b_div(b, col, rocke_b_const_i32(b, 16));
        rocke_value_t* col_in_stripe = rocke_b_mod(b, col, rocke_b_const_i32(b, 16));
        rocke_value_t* idx_store[4];

        idx_store[0] = rocke_b_const_i32(b, 0);
        idx_store[1] = stripe_idx;
        idx_store[2] = token;
        idx_store[3] = col_in_stripe;
        rocke_b_smem_store_vN(b, ctx->V_lds, idx_store, 4, fp8_vec, CHUNK);
    }
}
