// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_3d_gfx950_attention_tiled_3d_segment_prologue.c --
 * one part-file of the chunked C99 port of
 * rocke/instances/gfx950/attention_tiled_3d.py (arch gfx950).
 *
 * SCOPE OF THIS PART-FILE: the segment-kernel prologue + load-issuer closures +
 * inner IR helpers of build_unified_attention_3d_tiled:
 *
 *   rocke_gfx950_attention_tiled_3d_declare_params        (params, lines 310-358)
 *   rocke_gfx950_attention_tiled_3d_emit_prologue         (lines 360-534)
 *   rocke_gfx950_attention_tiled_3d_emit_early_zero_fill  (lines 416-461)
 *   rocke_gfx950_attention_tiled_3d_emit_q_to_lds         (lines 482-512)
 *   rocke_gfx950_attention_tiled_3d_emit_async_infra      (lines 564-601)
 *   rocke_gfx950_attention_tiled_3d_mfma_16x16_c_row      (lines 81-96)
 *   rocke_gfx950_attention_tiled_3d_tile_page_ids         (page-id loader)
 *   rocke_gfx950_attention_tiled_3d_issue_fp8_dequant_loads (lines 645-701)
 *   rocke_gfx950_attention_tiled_3d_issue_k               (lines 703-707)
 *   rocke_gfx950_attention_tiled_3d_issue_v               (lines 709-713)
 *
 * GFX950 DELTAS vs the gfx942 segment_setup sibling: WIDE-K config (no wide-b128
 * sync load path, no invariant-hoist); the PV transpose reader is BOUND here
 * (pv_tr_reader + tr_col_lane); the paged-KV descriptor is the single-block
 * (T==BS-style) indirect+unmerge chain with inline byte strides; the async DMA
 * delivers 4 DWORDS (8 halves) per lane and there is no WIDE_* path.
 *
 * The builder-call sequence is a byte-identical translation of those Python
 * spans. Every peer phase (loop init, softmax loop, epilogue, reduce kernel,
 * spec/config helpers) lives in a sibling translation unit and is reached only
 * via the internal header. This part-file writes ONLY ctx fields and reuses the
 * already-ported helper / transforms / atoms / distribution symbols.
 *
 * Lifetime: every emitted node is arena-owned (ctx->b->arena). Nothing is freed
 * individually.
 */

#include <math.h> /* INFINITY */
#include <stdio.h> /* snprintf */
#include <string.h>

#include "rocke/instance_gfx950_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Local conveniences (no IR; mirror the Python builder aliases).
 * ============================================================ */

#define B (ctx->b)
#define CFG (ctx->cfg)

/* ------------------------------------------------------------- *
 * _mfma_16x16_c_row(b, lane, reg) -- lines 81-96.
 *
 *   m_blk = b.div(lane, const_i32(16))
 *   n     = b.mod(lane, const_i32(16))
 *   row, _col = _C16_DIST.calculate_x(b, ys=[const_i32(0), const_i32(reg)],
 *                                     ps=[[m_blk, n]])
 *   return row
 * ------------------------------------------------------------- */
rocke_value_t* rocke_gfx950_attention_tiled_3d_mfma_16x16_c_row(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx, rocke_value_t* lane, int reg)
{
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2] = {NULL, NULL};

    if(!(0 <= reg && reg < 4))
    {
        if(B != NULL && B->status == ROCKE_OK)
        {
            snprintf(B->err, (size_t)ROCKE_ERR_MSG_CAP, "mfma_16x16 reg must be 0..3, got %d", reg);
            B->status = ROCKE_ERR_VALUE;
        }
        return NULL;
    }

    m_blk = rocke_b_div(B, lane, rocke_b_const_i32(B, 16));
    n = rocke_b_mod(B, lane, rocke_b_const_i32(B, 16));

    ys[0] = rocke_b_const_i32(B, 0);
    ys[1] = rocke_b_const_i32(B, reg);
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    if(!rocke_tile_distribution_calculate_x(B, ctx->C16_DIST, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row (col discarded) */
}

/* ============================================================ *
 * Descriptor offset conveniences (Python `idx, _ = desc.offset(...)`).
 * The validity is discarded. On a sticky error these return NULL.
 * ============================================================ */

static rocke_value_t* rocke__ml_offset(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                       rocke_value_t* token,
                                       rocke_value_t* head,
                                       rocke_value_t* seg)
{
    const char* in_names[3] = {"token", "head", "seg"};
    rocke_value_t* in_values[3];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    if(!rocke_transforms_descriptor_offset(B, ctx->ml_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static rocke_value_t* rocke__seg_acc_offset(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                            rocke_value_t* token,
                                            rocke_value_t* head,
                                            rocke_value_t* seg,
                                            rocke_value_t* dim)
{
    const char* in_names[4] = {"token", "head", "seg", "dim"};
    rocke_value_t* in_values[4];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    in_values[3] = dim;
    if(!rocke_transforms_descriptor_offset(
           B, ctx->seg_acc_desc, in_names, in_values, 4, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static rocke_value_t* rocke__q_offset(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                      rocke_value_t* token,
                                      rocke_value_t* head,
                                      rocke_value_t* dim)
{
    const char* in_names[3] = {"token", "head", "dim"};
    rocke_value_t* in_values[3];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = dim;
    if(!rocke_transforms_descriptor_offset(B, ctx->q_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=) -> i32 element off.
 * The Python supplies exactly these three upper coords; the descriptor's
 * remaining lowers resolve through the transform chain. */
static rocke_value_t* rocke__paged_kv_offset(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* linear_half,
                                             rocke_value_t* kv_head)
{
    const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
    rocke_value_t* in_values[3];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    in_values[0] = tile_idx;
    in_values[1] = linear_half;
    in_values[2] = kv_head;
    if(!rocke_transforms_descriptor_offset(
           B, ctx->paged_kv_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* ============================================================ *
 * rocke_gfx950_attention_tiled_3d_declare_params -- lines 303-358.
 *
 * The ~21 params in the load-bearing AITER order (segm_* workspace ptrs FIRST).
 * dtype / kv_io_dtype are taken from cfg (F16/BF16 ; FP8E4M3 when KV_FP8 else
 * dtype). Also sets the kernel attrs (max_workgroup_size, optional waves_per_eu).
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_declare_params(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_param_opts_t o;
    const rocke_type_t* dtype = CFG.dtype;
    const rocke_type_t* kv_io_dtype = CFG.kv_io_dtype;

    /* kernel attrs: max_workgroup_size = THREADS, optional waves_per_eu
     * (lines 303-305). */
    rocke_attr_set_int(B, &ctx->kernel->attrs, "max_workgroup_size", CFG.THREADS);
    if(ctx->spec != NULL && ctx->spec->has_waves_per_eu)
    {
        rocke_attr_set_int(B, &ctx->kernel->attrs, "waves_per_eu", ctx->spec->waves_per_eu);
    }

    /* segm_output_ptr: F32* noalias writeonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.writeonly = true;
    o.writeonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->segm_output_ptr
        = rocke_b_param(B, "segm_output_ptr", rocke_ptr_type(B, rocke_f32(), "global"), &o);

    /* segm_max_ptr: F32* noalias writeonly align4 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.writeonly = true;
    o.writeonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->segm_max_ptr
        = rocke_b_param(B, "segm_max_ptr", rocke_ptr_type(B, rocke_f32(), "global"), &o);

    /* segm_expsum_ptr: F32* noalias writeonly align4 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.writeonly = true;
    o.writeonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->segm_expsum_ptr
        = rocke_b_param(B, "segm_expsum_ptr", rocke_ptr_type(B, rocke_f32(), "global"), &o);

    /* query_ptr: dtype* noalias readonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->query = rocke_b_param(B, "query_ptr", rocke_ptr_type(B, dtype, "global"), &o);

    /* key_cache_ptr: kv_io_dtype* noalias readonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->key = rocke_b_param(B, "key_cache_ptr", rocke_ptr_type(B, kv_io_dtype, "global"), &o);

    /* value_cache_ptr: kv_io_dtype* noalias readonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->value = rocke_b_param(B, "value_cache_ptr", rocke_ptr_type(B, kv_io_dtype, "global"), &o);

    /* sink_ptr: dtype* readonly align16 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->sinks = rocke_b_param(B, "sink_ptr", rocke_ptr_type(B, dtype, "global"), &o);

    /* block_tables_ptr: I32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->block_tables
        = rocke_b_param(B, "block_tables_ptr", rocke_ptr_type(B, rocke_i32(), "global"), &o);

    /* seq_lens_ptr: I32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->seq_lens = rocke_b_param(B, "seq_lens_ptr", rocke_ptr_type(B, rocke_i32(), "global"), &o);

    /* alibi_slopes_ptr: F32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->alibi_slopes_ptr
        = rocke_b_param(B, "alibi_slopes_ptr", rocke_ptr_type(B, rocke_f32(), "global"), &o);

    /* qq_bias_ptr: F32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->qq_bias_ptr
        = rocke_b_param(B, "qq_bias_ptr", rocke_ptr_type(B, rocke_f32(), "global"), &o);

    /* query_start_len_ptr (cu_q): I32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->cu_q
        = rocke_b_param(B, "query_start_len_ptr", rocke_ptr_type(B, rocke_i32(), "global"), &o);

    /* scalar params (no ABI opts) */
    ctx->scale_p = rocke_b_param(B, "scale", rocke_f32(), NULL);
    ctx->k_scale_p = rocke_b_param(B, "k_scale", rocke_f32(), NULL);
    ctx->v_scale_p = rocke_b_param(B, "v_scale", rocke_f32(), NULL);
    ctx->softcap_p = rocke_b_param(B, "softcap", rocke_f32(), NULL);
    ctx->num_seqs_p = rocke_b_param(B, "num_seqs", rocke_i32(), NULL);
    ctx->bt_stride_p = rocke_b_param(B, "block_table_stride", rocke_i32(), NULL);
    ctx->qq_bias_stride0_p = rocke_b_param(B, "qq_bias_stride_0", rocke_i32(), NULL);
    if(ctx->strided_kv)
    {
        ctx->k_strides = rocke_declare_strided_kv_params(B, "k");
        ctx->v_strides = rocke_declare_strided_kv_params(B, "v");
    }
}

/* ============================================================ *
 * rocke_gfx950_attention_tiled_3d_emit_prologue -- lines 360-534.
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_prologue(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int HD = CFG.HD;
    const int T = CFG.T;
    const int BLOCK_M = CFG.BLOCK_M;
    const int BLOCK_Q = CFG.BLOCK_Q;
    const int NUM_QH = CFG.NUM_QH;
    const int NUM_SEG = CFG.NUM_SEG;
    const rocke_type_t* dtype = CFG.dtype;

    /* ---- grid ids + thread (lines 360-363) ---- */
    ctx->q_block_global_idx = rocke_b_block_id_x(B);
    ctx->kv_head_idx = rocke_b_block_id_y(B);
    ctx->seg_idx = rocke_b_block_id_z(B);
    ctx->tid = rocke_b_thread_id_x(B);

    /* ---- binary search seq_idx (lines 365-372) ----
     * per_token=False (the Python helper default for the block-q search). */
    if(CFG.DECODE_GRID)
    {
        /* all-decode grid: one CTA per sequence along x, no search. */
        ctx->seq_idx = ctx->q_block_global_idx;
    }
    else
    {
        ctx->seq_idx = rocke_binary_search_seq_idx(B,
                                                   ctx->cu_q,
                                                   ctx->q_block_global_idx,
                                                   ctx->num_seqs_p,
                                                   BLOCK_Q,
                                                   CFG.binary_search_iters,
                                                   false);
    }

    /* ---- cu_q bounds / per-sequence geometry (lines 373-379) ---- */
    ctx->cu_q_start = rocke_b_global_load_i32(B, ctx->cu_q, ctx->seq_idx, 0);
    ctx->cu_q_stop = rocke_b_global_load_i32(
        B, ctx->cu_q, rocke_b_add(B, ctx->seq_idx, rocke_b_const_i32(B, 1)), 0);
    ctx->cur_batch_q_len = rocke_b_sub(B, ctx->cu_q_stop, ctx->cu_q_start);
    if(CFG.DECODE_GRID)
    {
        ctx->q_block_start_idx = NULL;
        ctx->q_block_local_idx = rocke_b_const_i32(B, 0);
    }
    else
    {
        ctx->q_block_start_idx = rocke_b_add(
            B, rocke_b_div(B, ctx->cu_q_start, rocke_b_const_i32(B, BLOCK_Q)), ctx->seq_idx);
        ctx->q_block_local_idx = rocke_b_sub(B, ctx->q_block_global_idx, ctx->q_block_start_idx);
    }
    ctx->seq_len = rocke_b_global_load_i32(B, ctx->seq_lens, ctx->seq_idx, 0);
    ctx->context_len = rocke_b_sub(B, ctx->seq_len, ctx->cur_batch_q_len);

    /* qb_start_pos = q_block_local_idx * BLOCK_Q (line 381) */
    ctx->qb_start_pos = rocke_b_mul(B, ctx->q_block_local_idx, rocke_b_const_i32(B, BLOCK_Q));

    /* early return guard: if qb_start_pos >= cur_batch_q_len: ret() (lines 382-383) */
    {
        rocke_if_t g
            = rocke_b_scf_if(B, rocke_b_cmp_ge(B, ctx->qb_start_pos, ctx->cur_batch_q_len));
        rocke_b_region_enter(B, g.then_region);
        rocke_b_ret(B);
        rocke_b_region_leave(B);
    }

    /* tps = cdiv(seq_len, NUM_SEG*T) (line 386) */
    {
        rocke_value_t* tps_num
            = rocke_b_add(B, ctx->seq_len, rocke_b_const_i32(B, NUM_SEG * T - 1));
        ctx->tps = rocke_b_div(B, tps_num, rocke_b_const_i32(B, NUM_SEG * T));
    }

    /* ---- descriptors (lines 399-413) ---- */
    {
        const int ml_lengths[3] = {1 << 30, NUM_QH, NUM_SEG};
        const char* ml_coords[3] = {"token", "head", "seg"};
        ctx->ml_desc
            = rocke_tensor_descriptor_naive(B, "segm_ml", ml_lengths, 3, NULL, ml_coords, 3);
    }
    {
        const int sa_lengths[4] = {1 << 30, NUM_QH, NUM_SEG, HD};
        const char* sa_coords[4] = {"token", "head", "seg", "dim"};
        ctx->seg_acc_desc
            = rocke_tensor_descriptor_naive(B, "segm_output", sa_lengths, 4, NULL, sa_coords, 4);
    }
    {
        const int q_lengths[3] = {1 << 30, NUM_QH, HD};
        const char* q_coords[3] = {"token", "head", "dim"};
        ctx->q_desc = rocke_tensor_descriptor_naive(B, "Q", q_lengths, 3, NULL, q_coords, 3);
    }

    /* seg_start_tile_pos = seg_idx*tps*T (line 415) */
    {
        rocke_value_t* sst_inner = rocke_b_mul(B, ctx->seg_idx, ctx->tps);
        ctx->seg_start_tile_pos = rocke_b_mul(B, sst_inner, rocke_b_const_i32(B, T));
    }

    /* early-out zero-fill block (lines 416-461) */
    rocke_gfx950_attention_tiled_3d_emit_early_zero_fill(ctx);

    /* ---- LDS layout (lines 464-467) ---- */
    {
        const int q_shape[2] = {BLOCK_M, HD};
        const int k_shape[3] = {2, T, HD};
        /* One V slot: V(i) is issued and consumed within iteration i. */
        const int v_shape[3] = {1, T, HD};
        ctx->Q_lds = rocke_b_smem_alloc(B, dtype, q_shape, 2, "Qlds");
        ctx->K_lds = rocke_b_smem_alloc(B, dtype, k_shape, 3, "Klds");
        ctx->V_lds = rocke_b_smem_alloc(B, dtype, v_shape, 3, "Vlds");
    }

    /* ---- PV transpose-read state (gfx950-only) ----
     * pv_tr_reader = TransposeLdsReader(K=16, M=16).bind(b, tid)
     * tr_col_lane  = pv_tr_reader.col */
    {
        rocke_transpose_lds_reader_t r;
        r.K = 16;
        r.M = 16;
        ctx->pv_tr_reader = rocke_transpose_lds_reader_bind(B, &r, ctx->tid);
        ctx->tr_col_lane = (ctx->pv_tr_reader != NULL) ? ctx->pv_tr_reader->col : NULL;
    }

    /* ---- SSA constants (lines 474-480) ---- */
    ctx->neg_inf = rocke_b_const_f32(B, -INFINITY);
    ctx->zero_f = rocke_b_const_f32(B, 0.0);
    ctx->one_f = rocke_b_const_f32(B, 1.0);
    ctx->rcp_ln2 = rocke_b_const_f32(B, 1.4426950408889634);
    ctx->qk_scale = rocke_b_fmul(B, ctx->scale_p, ctx->rcp_ln2);
    ctx->sw_const = rocke_b_const_i32(B, CFG.SLIDING_WINDOW);
    ctx->z8 = rocke_b_zero_vec(B, dtype, 8);

    /* ---- Q -> LDS (lines 482-512) ---- */
    rocke_gfx950_attention_tiled_3d_emit_q_to_lds(ctx);

    /* ---- Per-segment tile range (lines 514-523) ---- */
    {
        rocke_value_t* msp_inner = rocke_b_add(B, ctx->context_len, ctx->qb_start_pos);
        rocke_value_t* msp_raw
            = rocke_b_add(B, msp_inner, rocke_b_const_i32(B, CFG.bm1_div_nqk + 1));
        rocke_value_t* msp_cmp = rocke_b_cmp_lt(B, msp_raw, ctx->seq_len);
        rocke_value_t* nt_inner;
        rocke_value_t* tile_end_raw_mul_inner;
        rocke_value_t* tile_end_raw;
        rocke_value_t* tile_end_cmp;
        ctx->max_seq_prefix_len = rocke_b_select(B, msp_cmp, msp_raw, ctx->seq_len);
        nt_inner = rocke_b_add(B, ctx->max_seq_prefix_len, rocke_b_const_i32(B, T - 1));
        ctx->num_tiles = rocke_b_div(B, nt_inner, rocke_b_const_i32(B, T));

        ctx->tile_start = rocke_b_mul(B, ctx->seg_idx, ctx->tps);
        tile_end_raw_mul_inner = rocke_b_add(B, ctx->seg_idx, rocke_b_const_i32(B, 1));
        tile_end_raw = rocke_b_mul(B, tile_end_raw_mul_inner, ctx->tps);
        tile_end_cmp = rocke_b_cmp_lt(B, tile_end_raw, ctx->num_tiles);
        ctx->tile_end = rocke_b_select(B, tile_end_cmp, tile_end_raw, ctx->num_tiles);
    }

    /* ---- lane decode (lines 533-534) ---- */
    ctx->lane_rg = rocke_b_div(B, ctx->tid, rocke_b_const_i32(B, 16));
    ctx->lane_col = rocke_b_mod(B, ctx->tid, rocke_b_const_i32(B, 16));

    /* NOTE: the async DMA infra (lines 564-585) and the paged-KV descriptor
     * (lines 587-601) are emitted AFTER acc_zero in Python's single linear
     * build; they are emitted by emit_loop_init (via
     * rocke_gfx950_attention_tiled_3d_emit_async_infra) so the SSA emission order
     * matches Python byte-for-byte. */
}

/* ============================================================ *
 * rocke_gfx950_attention_tiled_3d_emit_async_infra -- lines 564-601.
 *
 * Emitted from emit_loop_init right after acc_zero so the op order matches the
 * single linear Python build.
 *
 * Builds: big_bytes / key_rsrc / value_rsrc, lane_half_base = tid*8,
 * K/V_lds_addr, zero_soff, seq_base = seq_idx*bt_stride_p, and the paged-KV
 * byte descriptor (naive 4D + indirect("tile_idx") + unmerge("linear_half")).
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_async_infra(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int HD = CFG.HD;
    const int T = CFG.T;
    const int NUM_KV = CFG.NUM_KV;

    /* ---- async DMA infra (lines 565-585) ---- */
    ctx->big_bytes = rocke_b_const_i32(B, 0x7FFF0000);
    if(ctx->strided_kv)
    {
        ctx->key_rsrc = rocke_strided_kv_resource(
            B, ctx->key, &ctx->k_strides, ctx->seq_idx, ctx->kv_head_idx);
        ctx->value_rsrc = rocke_strided_kv_resource(
            B, ctx->value, &ctx->v_strides, ctx->seq_idx, ctx->kv_head_idx);
    }
    else
    {
        ctx->key_rsrc = rocke_b_buffer_rsrc(B, ctx->key, ctx->big_bytes);
        ctx->value_rsrc = rocke_b_buffer_rsrc(B, ctx->value, ctx->big_bytes);
    }
    /* kv_block_bytes_c = const_i32(kv_stride_blk_b): one-block buffer bound for
     * the i64-addressing path (Python creates it unconditionally right after the
     * byte strides; unused in the i32 path, where it is DCE'd). */
    ctx->kv_block_bytes_c = rocke_b_const_i32(B, CFG.kv_stride_blk_b);
    ctx->lane_half_base = rocke_b_mul(B, ctx->tid, rocke_b_const_i32(B, CFG.HALVES_PER_LANE));
    ctx->K_lds_addr = rocke_b_smem_addr_of(B, ctx->K_lds);
    ctx->V_lds_addr = rocke_b_smem_addr_of(B, ctx->V_lds);
    ctx->zero_soff = rocke_b_const_i32(B, 0);

    ctx->seq_pages = NULL;
    memset(ctx->lane_within_b, 0, sizeof(ctx->lane_within_b));
    ctx->paged_kv_desc = NULL;
    /* seq_base = seq_idx * bt_stride_p (line 592) */
    if(ctx->strided_kv)
    {
        return;
    }
    ctx->seq_base = rocke_b_mul(B, ctx->seq_idx, ctx->bt_stride_p);
    if(!CFG.PAGE_IDS)
    {
        /* paged_kv_desc = naive("paged_kv_bytes", [1<<24, T, NUM_KV, HD], byte
         * strides).transform(indirect("tile_idx" -> "physical_block",
         * base=seq_base), unmerge("linear_half" -> ("token","dim"), (T, HD))) */
        const int lengths[4] = {1 << 24, T, NUM_KV, HD};
        const int strides[4]
            = {CFG.kv_stride_blk_b, CFG.kv_stride_tok_b, CFG.kv_stride_h_b, CFG.KV_BYTES};
        const char* coords[4] = {"physical_block", "token", "kv_head", "dim"};
        rocke_tensor_descriptor_t* base
            = rocke_tensor_descriptor_naive(B, "paged_kv_bytes", lengths, 4, strides, coords, 4);
        const rocke_transform_t* chain[2];
        const char* into_td[2] = {"token", "dim"};
        const int dims_td[2] = {T, HD};
        chain[0] = rocke_indirect(
            B, "tile_idx", "physical_block", ctx->block_tables, ctx->seq_base, NULL, 0);
        chain[1] = rocke_unmerge(B, "linear_half", into_td, 2, dims_td);
        ctx->paged_kv_desc = rocke_tensor_descriptor_transform(B, base, chain, 2);
    }
    else if(CFG.N_BLOCKS_PER_TILE > 1 || CFG.WAVE_PAGE_ID_LOAD)
    {
        /* A sequence's last tile can reach past its last page; the table entries
         * there are not the sequence's. Those pages reuse the tile's first page
         * id (always live); the in_prefix mask discards their tokens. */
        rocke_value_t* rounded = rocke_b_add(B, ctx->seq_len, rocke_b_const_i32(B, CFG.BS - 1));
        ctx->seq_pages = rocke_b_div(B, rounded, rocke_b_const_i32(B, CFG.BS));
    }

    /* Page-id loader (fp16/bf16): the per-lane part of the address is
     * loop-invariant; the page base goes into soffset per tile. */
    if(CFG.PAGE_IDS)
    {
        rocke_value_t* lane_tok = rocke_b_div(B, ctx->lane_half_base, rocke_b_const_i32(B, HD));
        rocke_value_t* lane_dim = rocke_b_mod(B, ctx->lane_half_base, rocke_b_const_i32(B, HD));
        rocke_value_t* tok_b = rocke_b_mul(B, lane_tok, rocke_b_const_i32(B, CFG.kv_stride_tok_b));
        rocke_value_t* head_b
            = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, CFG.kv_stride_h_b));
        rocke_value_t* tok_head = rocke_b_add(B, tok_b, head_b);
        if(CFG.KV_SWZ)
        {
            /* A call's LDS rows are ph + lane_tok (mod chunks per row), with ph =
             * its first row; ph is a multiple of tokens_per_call and lane_tok <
             * tokens_per_call, so the sum never wraps. The lane writing LDS
             * chunk c fetches chunk c ^ row. */
            /* n_ph = HD * HD / 4096 <= 16 (HD <= 256, as the PV carry arrays assume). */
            const int n_ph = CFG.KV_CHUNKS_PER_ROW / CFG.tokens_per_call;
            rocke_value_t* lane_chunk = rocke_b_div(B, lane_dim, rocke_b_const_i32(B, 8));
            int ph_i;
            for(ph_i = 0; ph_i < n_ph && ph_i < CFG.kv_calls_per_tile; ++ph_i)
            {
                rocke_value_t* row_ph
                    = rocke_b_add(B, lane_tok, rocke_b_const_i32(B, ph_i * CFG.tokens_per_call));
                rocke_value_t* src_chunk = rocke_b_xor(B, lane_chunk, row_ph);
                rocke_value_t* src_b
                    = rocke_b_mul(B, src_chunk, rocke_b_const_i32(B, 8 * CFG.KV_BYTES));
                ctx->lane_within_b[ph_i] = rocke_b_add(B, tok_head, src_b);
            }
        }
        else
        {
            rocke_value_t* dim_b = rocke_b_mul(B, lane_dim, rocke_b_const_i32(B, CFG.KV_BYTES));
            ctx->lane_within_b[0] = rocke_b_add(B, tok_head, dim_b);
        }
    }
}

rocke_value_t* rocke_gfx950_attention_tiled_3d_kv_col(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx, rocke_value_t* row, rocke_value_t* col)
{
    rocke_value_t* col_chunk;
    rocke_value_t* row_ph;
    rocke_value_t* chunk;
    rocke_value_t* chunk_col;
    rocke_value_t* in_chunk;
    if(!CFG.KV_SWZ)
    {
        return col;
    }
    col_chunk = rocke_b_div(B, col, rocke_b_const_i32(B, 8));
    row_ph = rocke_b_mod(B, row, rocke_b_const_i32(B, CFG.KV_CHUNKS_PER_ROW));
    chunk = rocke_b_xor(B, col_chunk, row_ph);
    chunk_col = rocke_b_mul(B, chunk, rocke_b_const_i32(B, 8));
    in_chunk = rocke_b_mod(B, col, rocke_b_const_i32(B, 8));
    return rocke_b_add(B, chunk_col, in_chunk);
}

/* _kv_src_linear(linear_half): the async copies write LDS in lane order, so
 * under the swizzle each lane fetches the element its LDS position holds
 * (_kv_col is its own inverse). */
rocke_value_t*
    rocke_gfx950_attention_tiled_3d_kv_src_linear(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                                  rocke_value_t* linear_half)
{
    rocke_value_t* row;
    rocke_value_t* row_base;
    rocke_value_t* col;
    if(!CFG.KV_SWZ)
    {
        return linear_half;
    }
    row = rocke_b_div(B, linear_half, rocke_b_const_i32(B, CFG.HD));
    row_base = rocke_b_mul(B, row, rocke_b_const_i32(B, CFG.HD));
    col = rocke_b_mod(B, linear_half, rocke_b_const_i32(B, CFG.HD));
    return rocke_b_add(B, row_base, rocke_gfx950_attention_tiled_3d_kv_col(ctx, row, col));
}

/* _load_page_id_window(rel_page): lane l loads the id of the segment's page
 * rel_page + l; past the sequence's last page it gets 0, never used (see
 * _window_tile_ids). */
rocke_value_t* rocke_gfx950_attention_tiled_3d_load_page_id_window(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx, rocke_value_t* rel_page)
{
    rocke_value_t* first = rocke_b_add(B, ctx->seg_first_page, rel_page);
    rocke_value_t* pg = rocke_b_add(B, first, ctx->tid);
    rocke_value_t* idx = rocke_b_add(B, ctx->seq_base, pg);
    rocke_value_t* live = rocke_b_cmp_lt(B, pg, ctx->seq_pages);
    rocke_value_t* other = rocke_b_const_i32(B, 0);
    return rocke_b_masked_global_load(B, ctx->block_tables, idx, live, other, rocke_i32(), 4);
}

/* _window_tile_ids(page_id_window, kv_tile_idx): the tile's page ids from the
 * 64-page window (v_readlane); dead pages reuse the tile's first id, always live. */
void rocke_gfx950_attention_tiled_3d_window_tile_ids(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
    rocke_value_t* page_id_window,
    rocke_value_t* kv_tile_idx,
    rocke_value_t** out)
{
    const int NB = CFG.N_BLOCKS_PER_TILE;
    int j;
    rocke_value_t* first_blk = rocke_b_mul(B, kv_tile_idx, rocke_b_const_i32(B, NB));
    rocke_value_t* rel = rocke_b_sub(B, first_blk, ctx->seg_first_page);
    rocke_value_t* lane0 = rocke_b_mod(B, rel, rocke_b_const_i32(B, 64));
    out[0] = rocke_b_readlane(B, page_id_window, lane0);
    for(j = 1; j < NB; ++j)
    {
        rocke_value_t* lane = rocke_b_add(B, lane0, rocke_b_const_i32(B, j));
        rocke_value_t* idj = rocke_b_readlane(B, page_id_window, lane);
        rocke_value_t* blk = rocke_b_add(B, first_blk, rocke_b_const_i32(B, j));
        rocke_value_t* live = rocke_b_cmp_lt(B, blk, ctx->seq_pages);
        out[j] = rocke_b_select(B, live, idj, out[0]);
    }
}

/* _refill_page_id_window(page_id_window, kv_tile_idx): the window for the next
 * iteration. If tile kv_tile_idx + 2 starts a new window, load it now (after
 * this tile's copies, so the explicit waits still cover V(t)); the loop-top
 * wait completes it before the next iteration reads it. A wave-uniform 0/1-trip
 * loop is an if with a result (scf_if has none): 0 trips keep the window, 1 trip
 * loads the next one. It compiles to a scalar branch around the load. */
rocke_value_t* rocke_gfx950_attention_tiled_3d_refill_page_id_window(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
    rocke_value_t* page_id_window,
    rocke_value_t* kv_tile_idx)
{
    rocke_value_t* t2 = rocke_b_add(B, kv_tile_idx, rocke_b_const_i32(B, 2));
    rocke_value_t* t2_blk = rocke_b_mul(B, t2, rocke_b_const_i32(B, CFG.N_BLOCKS_PER_TILE));
    rocke_value_t* rel2 = rocke_b_sub(B, t2_blk, ctx->seg_first_page);
    rocke_value_t* rel2_lane = rocke_b_mod(B, rel2, rocke_b_const_i32(B, 64));
    rocke_value_t* at_start = rocke_b_cmp_eq(B, rel2_lane, rocke_b_const_i32(B, 0));
    rocke_value_t* t2_live = rocke_b_cmp_lt(B, t2, ctx->tile_end);
    rocke_value_t* cross = rocke_b_land(B, at_start, t2_live);
    rocke_value_t* lo = rocke_b_const_i32(B, 0);
    rocke_value_t* one = rocke_b_const_i32(B, 1);
    rocke_value_t* zero = rocke_b_const_i32(B, 0);
    rocke_value_t* trips = rocke_b_select(B, cross, one, zero);
    rocke_value_t* step = rocke_b_const_i32(B, 1);
    rocke_iter_arg_t wargs[1];
    rocke_for_t refill;
    rocke_value_t* next_window;
    wargs[0].name = rocke_arena_strdup(&B->arena, "pid_window_r");
    wargs[0].init = page_id_window;
    refill = rocke_b_scf_for_iter(B, lo, trips, step, wargs, 1, "pid_window_load", false, true);
    rocke_b_region_enter(B, refill.body);
    next_window = rocke_gfx950_attention_tiled_3d_load_page_id_window(ctx, rel2);
    rocke_b_scf_yield(B, &next_window, 1);
    rocke_b_region_leave(B);
    return refill.op->results[0];
}

/* _tile_page_ids(kv_tile_idx): one guarded block-table load per page of the
 * tile, made wave-uniform. */
void rocke_gfx950_attention_tiled_3d_tile_page_ids(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                                   rocke_value_t* kv_tile_idx,
                                                   rocke_value_t** out)
{
    const int NB = CFG.N_BLOCKS_PER_TILE;
    int j;
    /* A tile starts below seq_len, so its first page is always live. */
    rocke_value_t* first_blk = rocke_b_mul(B, kv_tile_idx, rocke_b_const_i32(B, NB));
    rocke_value_t* first_pid = rocke_b_global_load_i32(
        B, ctx->block_tables, rocke_b_add(B, ctx->seq_base, first_blk), 0);
    out[0] = rocke_b_to_sgpr_u32(B, first_pid);
    for(j = 1; j < NB; ++j)
    {
        rocke_value_t* blk = rocke_b_add(B, first_blk, rocke_b_const_i32(B, j));
        rocke_value_t* idx = rocke_b_add(B, ctx->seq_base, blk);
        rocke_value_t* live = rocke_b_cmp_lt(B, blk, ctx->seq_pages);
        rocke_value_t* pid = rocke_b_masked_global_load(
            B, ctx->block_tables, idx, live, first_pid, rocke_i32(), 4);
        out[j] = rocke_b_to_sgpr_u32(B, pid);
    }
}

/* _issue_paged_tile(src, lds_base, buf_idx, page_ids): one async DMA call per
 * 64 lanes x 8 halves; the call's page base goes into soffset (i32 path) or a
 * per-page 64-bit buffer base (i64 path). */
static void rocke__issue_paged_tile(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                    rocke_value_t* src,
                                    rocke_value_t* src_rsrc,
                                    rocke_value_t* lds_base,
                                    rocke_value_t* buf_idx,
                                    rocke_value_t* const* page_ids)
{
    int call;

    rocke_value_t* buf_off_i32 = rocke_b_mul(B, buf_idx, rocke_b_const_i32(B, CFG.bytes_per_buf));
    rocke_value_t* buf_off_i64 = rocke_b_zext(B, buf_off_i32, rocke_i64());
    rocke_value_t* buf_base = rocke_b_smem_ptr_add(B, lds_base, buf_off_i64);
    for(call = 0; call < CFG.kv_calls_per_tile; ++call)
    {
        const int blk = call / CFG.calls_per_block;
        const int tok0 = (call % CFG.calls_per_block) * CFG.tokens_per_call;
        const int ph_i = CFG.KV_SWZ ? (call % (CFG.KV_CHUNKS_PER_ROW / CFG.tokens_per_call)) : 0;
        rocke_value_t* voff = rocke_b_add(
            B, ctx->lane_within_b[ph_i], rocke_b_const_i32(B, tok0 * CFG.kv_stride_tok_b));
        rocke_value_t* call_rsrc;
        rocke_value_t* soff;
        if(CFG.I64_KV_ADDR)
        {
            rocke_value_t* pid_i64 = rocke_b_zext(B, page_ids[blk], rocke_i64());
            rocke_value_t* base_i64
                = rocke_b_mul(B, pid_i64, rocke_b_const_i64(B, (int64_t)CFG.kv_stride_blk_b));
            call_rsrc = rocke_b_buffer_rsrc(
                B, rocke_b_global_ptr_add(B, src, base_i64), ctx->kv_block_bytes_c);
            soff = ctx->zero_soff;
        }
        else
        {
            call_rsrc = src_rsrc;
            soff = rocke_b_mul(B, page_ids[blk], rocke_b_const_i32(B, CFG.kv_stride_blk_b));
        }
        rocke_value_t* dst = rocke_b_smem_ptr_add(
            B, buf_base, rocke_b_const_i64(B, (int64_t)call * CFG.bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            B, call_rsrc, dst, voff, soff, CFG.ASYNC_LDS_DWORDS, CFG.KV_CACHE_BITS);
    }
}

/* ============================================================ *
 * rocke_gfx950_attention_tiled_3d_emit_early_zero_fill -- lines 416-461.
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_early_zero_fill(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int NQK = CFG.NQK;
    const int NUM_QH = CFG.NUM_QH;
    const int PV_N_TILES = CFG.PV_N_TILES;
    int n, reg;

    rocke_if_t guard = rocke_b_scf_if(B, rocke_b_cmp_ge(B, ctx->seg_start_tile_pos, ctx->seq_len));
    rocke_b_region_enter(B, guard.then_region);
    {
        rocke_value_t* neg_inf_local = rocke_b_const_f32(B, -INFINITY);
        rocke_value_t* zero_local = rocke_b_const_f32(B, 0.0);
        rocke_value_t* lwm_mod = rocke_b_mod(B, ctx->tid, rocke_b_const_i32(B, 16));
        rocke_value_t* lane_writes_ml_e = rocke_b_cmp_eq(B, lwm_mod, rocke_b_const_i32(B, 0));

        /* ml zero-fill (lines 420-434) */
        for(reg = 0; reg < 4; ++reg)
        {
            rocke_value_t* row
                = rocke_gfx950_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
            rocke_value_t* qp_r_div = rocke_b_div(B, row, rocke_b_const_i32(B, NQK));
            rocke_value_t* qp_r = rocke_b_add(B, ctx->qb_start_pos, qp_r_div);
            rocke_value_t* qh_r_mul = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, NQK));
            rocke_value_t* qh_r_mod = rocke_b_mod(B, row, rocke_b_const_i32(B, NQK));
            rocke_value_t* qh_r = rocke_b_add(B, qh_r_mul, qh_r_mod);
            rocke_value_t* row_ok_a = rocke_b_cmp_lt(B, qp_r, ctx->cur_batch_q_len);
            rocke_value_t* row_ok_b = rocke_b_cmp_lt(B, qh_r, rocke_b_const_i32(B, NUM_QH));
            rocke_value_t* row_ok = rocke_b_land(B, row_ok_a, row_ok_b);
            rocke_value_t* qp_r_safe = rocke_b_select(B, row_ok, qp_r, rocke_b_const_i32(B, 0));
            rocke_value_t* qh_r_safe = rocke_b_select(B, row_ok, qh_r, rocke_b_const_i32(B, 0));
            rocke_value_t* qtoken = rocke_b_add(B, ctx->cu_q_start, qp_r_safe);
            rocke_value_t* ml_idx = rocke__ml_offset(ctx, qtoken, qh_r_safe, ctx->seg_idx);
            rocke_if_t w = rocke_b_scf_if(B, lane_writes_ml_e);
            rocke_b_region_enter(B, w.then_region);
            rocke_b_global_store(B, ctx->segm_max_ptr, ml_idx, neg_inf_local, 4);
            rocke_b_global_store(B, ctx->segm_expsum_ptr, ml_idx, zero_local, 4);
            rocke_b_region_leave(B);
        }

        /* seg_acc zero-fill (lines 436-460) */
        {
            rocke_value_t* lane_col_e = rocke_b_mod(B, ctx->tid, rocke_b_const_i32(B, 16));
            for(n = 0; n < PV_N_TILES; ++n)
            {
                for(reg = 0; reg < 4; ++reg)
                {
                    rocke_value_t* row
                        = rocke_gfx950_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
                    rocke_value_t* col_n = rocke_b_const_i32(B, n);
                    rocke_value_t* col_16 = rocke_b_const_i32(B, 16);
                    rocke_value_t* col_mul = rocke_b_mul(B, col_n, col_16);
                    rocke_value_t* col = rocke_b_add(B, col_mul, lane_col_e);
                    rocke_value_t* qp_r_div = rocke_b_div(B, row, rocke_b_const_i32(B, NQK));
                    rocke_value_t* qp_r = rocke_b_add(B, ctx->qb_start_pos, qp_r_div);
                    rocke_value_t* qh_r_mul
                        = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, NQK));
                    rocke_value_t* qh_r_mod = rocke_b_mod(B, row, rocke_b_const_i32(B, NQK));
                    rocke_value_t* qh_r = rocke_b_add(B, qh_r_mul, qh_r_mod);
                    rocke_value_t* row_ok_a = rocke_b_cmp_lt(B, qp_r, ctx->cur_batch_q_len);
                    rocke_value_t* row_ok_b = rocke_b_cmp_lt(B, qh_r, rocke_b_const_i32(B, NUM_QH));
                    rocke_value_t* row_ok = rocke_b_land(B, row_ok_a, row_ok_b);
                    rocke_value_t* qp_r_safe
                        = rocke_b_select(B, row_ok, qp_r, rocke_b_const_i32(B, 0));
                    rocke_value_t* qh_r_safe
                        = rocke_b_select(B, row_ok, qh_r, rocke_b_const_i32(B, 0));
                    rocke_value_t* qtoken = rocke_b_add(B, ctx->cu_q_start, qp_r_safe);
                    rocke_value_t* seg_acc_idx
                        = rocke__seg_acc_offset(ctx, qtoken, qh_r_safe, ctx->seg_idx, col);
                    rocke_b_global_store(B, ctx->segm_output_ptr, seg_acc_idx, zero_local, 4);
                }
            }
        }
        rocke_b_ret(B);
    }
    rocke_b_region_leave(B);
}

/* ============================================================ *
 * rocke_gfx950_attention_tiled_3d_emit_q_to_lds -- lines 482-512.
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_q_to_lds(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int NQK = CFG.NQK;
    const int NUM_QH = CFG.NUM_QH;
    const int THREADS = CFG.THREADS;
    const int Q_VECS_PER_ROW = CFG.Q_VECS_PER_ROW;
    const int Q_VECS_PER_THREAD = CFG.Q_VECS_PER_THREAD;
    const rocke_type_t* dtype = CFG.dtype;
    int li;

    for(li = 0; li < Q_VECS_PER_THREAD; ++li)
    {
        rocke_value_t* q_vid_li = rocke_b_const_i32(B, li);
        rocke_value_t* q_vid_thr = rocke_b_const_i32(B, THREADS);
        rocke_value_t* q_vid_mul = rocke_b_mul(B, q_vid_li, q_vid_thr);
        rocke_value_t* q_vid = rocke_b_add(B, q_vid_mul, ctx->tid);
        rocke_value_t* Q_row = rocke_b_div(B, q_vid, rocke_b_const_i32(B, Q_VECS_PER_ROW));
        rocke_value_t* Q_col_mod = rocke_b_mod(B, q_vid, rocke_b_const_i32(B, Q_VECS_PER_ROW));
        rocke_value_t* Q_col = rocke_b_mul(B, Q_col_mod, rocke_b_const_i32(B, 8));
        rocke_value_t* q_pos_div = rocke_b_div(B, Q_row, rocke_b_const_i32(B, NQK));
        rocke_value_t* q_pos_t = rocke_b_add(B, ctx->qb_start_pos, q_pos_div);
        rocke_value_t* qh_t_mul = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, NQK));
        rocke_value_t* qh_t_mod = rocke_b_mod(B, Q_row, rocke_b_const_i32(B, NQK));
        rocke_value_t* qh_t = rocke_b_add(B, qh_t_mul, qh_t_mod);
        rocke_value_t* qmask_a = rocke_b_cmp_lt(B, q_pos_t, ctx->cur_batch_q_len);
        rocke_value_t* qmask_b = rocke_b_cmp_lt(B, qh_t, rocke_b_const_i32(B, NUM_QH));
        rocke_value_t* qmask_t = rocke_b_land(B, qmask_a, qmask_b);
        rocke_value_t* q_pos_safe = rocke_b_select(B, qmask_t, q_pos_t, rocke_b_const_i32(B, 0));
        rocke_value_t* qh_safe = rocke_b_select(B, qmask_t, qh_t, rocke_b_const_i32(B, 0));
        rocke_value_t* q_off_tok = rocke_b_add(B, ctx->cu_q_start, q_pos_safe);
        rocke_value_t* q_off_base
            = rocke__q_offset(ctx, q_off_tok, qh_safe, rocke_b_const_i32(B, 0));
        rocke_value_t* v8_idx = rocke_b_add(B, q_off_base, Q_col);
        rocke_value_t* v8 = rocke_b_global_load_vN(B, ctx->query, v8_idx, dtype, 8, 16);
        rocke_value_t* splat = rocke_b_vector_splat(B, qmask_t, 8);
        rocke_value_t* sel = rocke_b_vector_select(B, splat, v8, ctx->z8);
        rocke_value_t* idxs[2] = {Q_row, Q_col};
        rocke_b_smem_store_vN(B, ctx->Q_lds, idxs, 2, sel, 8);
    }
    rocke_b_sync(B);
}

/* ============================================================ *
 * Load issuers (Python closures, lines 603-713).
 * ============================================================ */

/* _issue_strided_tile(src_rsrc, strides, lds_base, kv_tile_idx, buf_idx): the
 * strided-KV loader (fp16/bf16 only), one async DMA call per
 * KV_HALVES_PER_CALL halves at the per-call strided offset. */
static void rocke__issue_strided_tile(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                      rocke_value_t* src_rsrc,
                                      const rocke_strided_kv_params_t* strides,
                                      rocke_value_t* lds_base,
                                      rocke_value_t* kv_tile_idx,
                                      rocke_value_t* buf_idx)
{
    const int KV_HALVES_PER_CALL = CFG.KV_HALVES_PER_CALL;
    const int kv_calls_per_tile = CFG.kv_calls_per_tile;
    const int bytes_per_call = CFG.bytes_per_call;
    const int bytes_per_buf = CFG.bytes_per_buf;
    const int ASYNC_LDS_DWORDS = CFG.ASYNC_LDS_DWORDS;
    int call;

    rocke_value_t* buf_off_i32 = rocke_b_mul(B, buf_idx, rocke_b_const_i32(B, bytes_per_buf));
    rocke_value_t* buf_off_i64 = rocke_b_zext(B, buf_off_i32, rocke_i64());
    rocke_value_t* buf_base = rocke_b_smem_ptr_add(B, lds_base, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* linear_half
            = rocke_b_add(B, rocke_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        rocke_value_t* voff = rocke_strided_kv_offset(
            B,
            strides,
            kv_tile_idx,
            rocke_gfx950_attention_tiled_3d_kv_src_linear(ctx, linear_half),
            ctx->seq_len,
            CFG.HD,
            CFG.T);
        rocke_value_t* dst = rocke_b_smem_ptr_add(
            B, buf_base, rocke_b_const_i64(B, (int64_t)call * bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            B, src_rsrc, dst, voff, ctx->zero_soff, ASYNC_LDS_DWORDS, CFG.KV_CACHE_BITS);
    }
}

/* _issue_fp8_dequant_loads(kv_tile_idx, buf_idx, lds_token) -- lines 645-701.
 * is_value: false => "K" (key, k_scale, K_lds), true => "V".
 *
 * Sync per-thread fp8 -> cvt_pk_f32_fp8x4 -> *scale (UNFUSED fmul) -> dtype ->
 * LDS via rocke_dequant_fp8x8_to_dtype. The unfused fmul is the CRITICAL invariant
 * (the fused E8M0 cvt would truncate non-power-of-two per-tensor scales). */
void rocke_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
    bool is_value,
    rocke_value_t* kv_tile_idx,
    rocke_value_t* buf_idx)
{
    const int HD = CFG.HD;
    const int THREADS = CFG.THREADS;
    const int fp8_elems_per_chunk = CFG.fp8_elems_per_chunk;
    const int fp8_chunks_per_thread = CFG.fp8_chunks_per_thread;
    const rocke_type_t* dtype = CFG.dtype;
    rocke_value_t* scale = is_value ? ctx->v_scale_p : ctx->k_scale_p;
    rocke_value_t* lds = is_value ? ctx->V_lds : ctx->K_lds;
    rocke_value_t* src = is_value ? ctx->value : ctx->key;
    int call;

    for(call = 0; call < fp8_chunks_per_thread; ++call)
    {
        rocke_value_t* chunk_call = rocke_b_const_i32(B, call);
        rocke_value_t* chunk_thr = rocke_b_const_i32(B, THREADS);
        rocke_value_t* chunk_mul = rocke_b_mul(B, chunk_call, chunk_thr);
        rocke_value_t* chunk_id = rocke_b_add(B, chunk_mul, ctx->tid);
        rocke_value_t* row
            = rocke_b_div(B, chunk_id, rocke_b_const_i32(B, HD / fp8_elems_per_chunk));
        rocke_value_t* col_mod
            = rocke_b_mod(B, chunk_id, rocke_b_const_i32(B, HD / fp8_elems_per_chunk));
        rocke_value_t* col = rocke_b_mul(B, col_mod, rocke_b_const_i32(B, fp8_elems_per_chunk));
        rocke_value_t* lhf_mul = rocke_b_mul(B, row, rocke_b_const_i32(B, HD));
        rocke_value_t* linear_half_first = rocke_b_add(B, lhf_mul, col);
        /* I64_KV_ADDR fp8: the Python fp8 sync loader uses the full per-lane
         * offset_i64 here (caches > 2 GiB). That non-split i64 paged variant is
         * not yet on the frozen C transforms surface (same limitation the 2D C
         * fp8 twin documents), so the C fp8 path resolves through the i32 offset
         * regardless. The byte-identity gate never pairs fp8 with i64 KV, so this
         * does not affect the gate; the fp16/bf16 i64 async loaders above are the
         * shipping correctness fix and are fully mirrored. */
        rocke_value_t* voff
            = rocke__paged_kv_offset(ctx, kv_tile_idx, linear_half_first, ctx->kv_head_idx);
        rocke_value_t* fp8_vec = rocke_b_global_load_vN(
            B, src, voff, rocke_fp8e4m3(), fp8_elems_per_chunk, fp8_elems_per_chunk);
        rocke_value_t* packed = rocke_dequant_fp8x8_to_dtype(B, fp8_vec, scale, dtype);
        rocke_value_t* idxs[3];
        idxs[0] = buf_idx;
        idxs[1] = row;
        idxs[2] = col;
        rocke_b_smem_store_vN(B, lds, idxs, 3, packed, fp8_elems_per_chunk);
    }
}

/* _issue_k(tile_idx, buf_idx) -- lines 703-707. */
void rocke_gfx950_attention_tiled_3d_issue_k(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* buf_idx,
                                             rocke_value_t* const* page_ids)
{
    if(CFG.KV_FP8)
    {
        rocke_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(ctx, false, tile_idx, buf_idx);
    }
    else if(ctx->strided_kv)
    {
        rocke__issue_strided_tile(
            ctx, ctx->key_rsrc, &ctx->k_strides, ctx->K_lds_addr, tile_idx, buf_idx);
    }
    else
    {
        rocke__issue_paged_tile(ctx, ctx->key, ctx->key_rsrc, ctx->K_lds_addr, buf_idx, page_ids);
    }
}

/* _issue_v(tile_idx, buf_idx) -- lines 709-713. */
void rocke_gfx950_attention_tiled_3d_issue_v(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* buf_idx,
                                             rocke_value_t* const* page_ids)
{
    if(CFG.KV_FP8)
    {
        rocke_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(ctx, true, tile_idx, buf_idx);
    }
    else if(ctx->strided_kv)
    {
        rocke__issue_strided_tile(
            ctx, ctx->value_rsrc, &ctx->v_strides, ctx->V_lds_addr, tile_idx, buf_idx);
    }
    else
    {
        rocke__issue_paged_tile(
            ctx, ctx->value, ctx->value_rsrc, ctx->V_lds_addr, buf_idx, page_ids);
    }
}
