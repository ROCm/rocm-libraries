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
 *   rocke_gfx950_attention_tiled_3d_issue_k_load          (lines 603-616)
 *   rocke_gfx950_attention_tiled_3d_issue_v_load          (lines 618-631)
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
    const int PV_K_STEP = CFG.PV_K_STEP;
    const rocke_type_t* dtype = CFG.dtype;

    /* ---- grid ids + thread (lines 360-363) ---- */
    ctx->q_block_global_idx = rocke_b_block_id_x(B);
    ctx->kv_head_idx = rocke_b_block_id_y(B);
    ctx->seg_idx = rocke_b_block_id_z(B);
    ctx->tid = rocke_b_thread_id_x(B);

    /* ---- binary search seq_idx (lines 365-372) ----
     * per_token=False (the Python helper default for the block-q search). */
    ctx->seq_idx = rocke_binary_search_seq_idx(B,
                                               ctx->cu_q,
                                               ctx->q_block_global_idx,
                                               ctx->num_seqs_p,
                                               BLOCK_Q,
                                               CFG.binary_search_iters,
                                               false);

    /* ---- cu_q bounds / per-sequence geometry (lines 373-379) ---- */
    ctx->cu_q_start = rocke_b_global_load_i32(B, ctx->cu_q, ctx->seq_idx, 0);
    ctx->cu_q_stop = rocke_b_global_load_i32(
        B, ctx->cu_q, rocke_b_add(B, ctx->seq_idx, rocke_b_const_i32(B, 1)), 0);
    ctx->cur_batch_q_len = rocke_b_sub(B, ctx->cu_q_stop, ctx->cu_q_start);
    ctx->q_block_start_idx = rocke_b_add(
        B, rocke_b_div(B, ctx->cu_q_start, rocke_b_const_i32(B, BLOCK_Q)), ctx->seq_idx);
    ctx->q_block_local_idx = rocke_b_sub(B, ctx->q_block_global_idx, ctx->q_block_start_idx);
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
        ctx->Q_lds = rocke_b_smem_alloc(B, dtype, q_shape, 2, "Qlds");
        ctx->K_lds = rocke_b_smem_alloc(B, dtype, k_shape, 3, "Klds");
        ctx->V_lds = rocke_b_smem_alloc(B, dtype, k_shape, 3, "Vlds");
    }
    {
        const int p_shape[2] = {BLOCK_M, T};
        ctx->P_lds = rocke_b_smem_alloc(B, dtype, p_shape, 2, "Plds");
    }

    /* ---- PV transpose-read state (gfx950-only, lines 471-472) ----
     * pv_tr_reader = TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, tid)
     * tr_col_lane  = pv_tr_reader.col */
    {
        rocke_transpose_lds_reader_t r;
        r.K = PV_K_STEP;
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
    ctx->key_rsrc = rocke_b_buffer_rsrc(B, ctx->key, ctx->big_bytes);
    ctx->value_rsrc = rocke_b_buffer_rsrc(B, ctx->value, ctx->big_bytes);
    /* kv_block_bytes_c = const_i32(kv_stride_blk_b): one-block buffer bound for
     * the i64-addressing path (Python creates it unconditionally right after the
     * byte strides; unused in the i32 path, where it is DCE'd). */
    ctx->kv_block_bytes_c = rocke_b_const_i32(B, CFG.kv_stride_blk_b);
    ctx->lane_half_base = rocke_b_mul(B, ctx->tid, rocke_b_const_i32(B, CFG.HALVES_PER_LANE));
    ctx->K_lds_addr = rocke_b_smem_addr_of(B, ctx->K_lds);
    ctx->V_lds_addr = rocke_b_smem_addr_of(B, ctx->V_lds);
    ctx->zero_soff = rocke_b_const_i32(B, 0);

    /* seq_base = seq_idx * bt_stride_p (line 592) */
    ctx->seq_base = rocke_b_mul(B, ctx->seq_idx, ctx->bt_stride_p);

    /* paged_kv_desc = TensorDescriptor.naive("paged_kv_bytes",
     *   lengths=[1<<24, T, NUM_KV, HD],
     *   strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
     *   coord_names=("physical_block","token","kv_head","dim")
     * ).transform(
     *   indirect("tile_idx", into="physical_block", table=block_tables,
     *            base=seq_base),
     *   unmerge("linear_half", into=("token","dim"), dims=(T, HD)),
     * )  (lines 593-601) */
    {
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

/* _issue_k_load(kv_tile_idx, buf_idx) -- lines 603-616. */
void rocke_gfx950_attention_tiled_3d_issue_k_load(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
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
    rocke_value_t* K_buf_base = rocke_b_smem_ptr_add(B, ctx->K_lds_addr, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* linear_half
            = rocke_b_add(B, rocke_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        rocke_value_t* call_rsrc = ctx->key_rsrc;
        rocke_value_t* voff = NULL;
        if(CFG.I64_KV_ADDR)
        {
            /* offset_i64_split folds the per-block byte base into a 64-bit
             * buffer base (no 2 GiB i32-voffset overflow); only the within-block
             * byte offset stays in the i32 voffset. */
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            rocke_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            rocke_value_t* base_i64 = NULL;
            rocke_value_t* valid = NULL;
            if(!rocke_transforms_descriptor_offset_i64_split(B,
                                                             ctx->paged_kv_desc,
                                                             "physical_block",
                                                             in_names,
                                                             in_values,
                                                             3,
                                                             &base_i64,
                                                             &voff,
                                                             &valid))
                return;
            call_rsrc = rocke_b_buffer_rsrc(
                B, rocke_b_global_ptr_add(B, ctx->key, base_i64), ctx->kv_block_bytes_c);
        }
        else
        {
            voff = rocke__paged_kv_offset(ctx, kv_tile_idx, linear_half, ctx->kv_head_idx);
        }
        rocke_value_t* k_dst = rocke_b_smem_ptr_add(
            B, K_buf_base, rocke_b_const_i64(B, (int64_t)call * bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            B, call_rsrc, k_dst, voff, ctx->zero_soff, ASYNC_LDS_DWORDS, ROCKE_CACHE_ALL);
    }
}

/* _issue_v_load(kv_tile_idx, buf_idx) -- lines 618-631. */
void rocke_gfx950_attention_tiled_3d_issue_v_load(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
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
    rocke_value_t* V_buf_base = rocke_b_smem_ptr_add(B, ctx->V_lds_addr, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* linear_half
            = rocke_b_add(B, rocke_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        rocke_value_t* call_rsrc = ctx->value_rsrc;
        rocke_value_t* voff = NULL;
        if(CFG.I64_KV_ADDR)
        {
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            rocke_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            rocke_value_t* base_i64 = NULL;
            rocke_value_t* valid = NULL;
            if(!rocke_transforms_descriptor_offset_i64_split(B,
                                                             ctx->paged_kv_desc,
                                                             "physical_block",
                                                             in_names,
                                                             in_values,
                                                             3,
                                                             &base_i64,
                                                             &voff,
                                                             &valid))
                return;
            call_rsrc = rocke_b_buffer_rsrc(
                B, rocke_b_global_ptr_add(B, ctx->value, base_i64), ctx->kv_block_bytes_c);
        }
        else
        {
            voff = rocke__paged_kv_offset(ctx, kv_tile_idx, linear_half, ctx->kv_head_idx);
        }
        rocke_value_t* v_dst = rocke_b_smem_ptr_add(
            B, V_buf_base, rocke_b_const_i64(B, (int64_t)call * bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            B, call_rsrc, v_dst, voff, ctx->zero_soff, ASYNC_LDS_DWORDS, ROCKE_CACHE_ALL);
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
                                             rocke_value_t* buf_idx)
{
    if(CFG.KV_FP8)
    {
        rocke_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(ctx, false, tile_idx, buf_idx);
    }
    else
    {
        rocke_gfx950_attention_tiled_3d_issue_k_load(ctx, tile_idx, buf_idx);
    }
}

/* _issue_v(tile_idx, buf_idx) -- lines 709-713. */
void rocke_gfx950_attention_tiled_3d_issue_v(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* buf_idx)
{
    if(CFG.KV_FP8)
    {
        rocke_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(ctx, true, tile_idx, buf_idx);
    }
    else
    {
        rocke_gfx950_attention_tiled_3d_issue_v_load(ctx, tile_idx, buf_idx);
    }
}
