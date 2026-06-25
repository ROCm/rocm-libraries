// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_3d_gfx950_attention_tiled_3d_segment_prologue.c --
 * one part-file of the chunked C99 port of
 * ck_dsl/instances/gfx950/attention_tiled_3d.py (arch gfx950).
 *
 * SCOPE OF THIS PART-FILE: the segment-kernel prologue + load-issuer closures +
 * inner IR helpers of build_unified_attention_3d_tiled:
 *
 *   ckc_gfx950_attention_tiled_3d_declare_params        (params, lines 310-358)
 *   ckc_gfx950_attention_tiled_3d_emit_prologue         (lines 360-534)
 *   ckc_gfx950_attention_tiled_3d_emit_early_zero_fill  (lines 416-461)
 *   ckc_gfx950_attention_tiled_3d_emit_q_to_lds         (lines 482-512)
 *   ckc_gfx950_attention_tiled_3d_emit_async_infra      (lines 564-601)
 *   ckc_gfx950_attention_tiled_3d_mfma_16x16_c_row      (lines 81-96)
 *   ckc_gfx950_attention_tiled_3d_issue_k_load          (lines 603-616)
 *   ckc_gfx950_attention_tiled_3d_issue_v_load          (lines 618-631)
 *   ckc_gfx950_attention_tiled_3d_issue_fp8_dequant_loads (lines 645-701)
 *   ckc_gfx950_attention_tiled_3d_issue_k               (lines 703-707)
 *   ckc_gfx950_attention_tiled_3d_issue_v               (lines 709-713)
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

#include "ckc/instance_gfx950_attention_tiled_3d_internal.h"

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
ckc_value_t* ckc_gfx950_attention_tiled_3d_mfma_16x16_c_row(
    ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx, ckc_value_t* lane, int reg)
{
    ckc_value_t* m_blk;
    ckc_value_t* n;
    ckc_value_t* ys[2];
    ckc_value_t* ps0[2];
    ckc_value_t* const* ps[1];
    int ps_counts[1];
    ckc_value_t* out_x[2] = {NULL, NULL};

    if(!(0 <= reg && reg < 4))
    {
        if(B != NULL && B->status == CKC_OK)
        {
            snprintf(B->err, (size_t)CKC_ERR_MSG_CAP, "mfma_16x16 reg must be 0..3, got %d", reg);
            B->status = CKC_ERR_VALUE;
        }
        return NULL;
    }

    m_blk = ckc_b_div(B, lane, ckc_b_const_i32(B, 16));
    n = ckc_b_mod(B, lane, ckc_b_const_i32(B, 16));

    ys[0] = ckc_b_const_i32(B, 0);
    ys[1] = ckc_b_const_i32(B, reg);
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    if(!ckc_tile_distribution_calculate_x(B, ctx->C16_DIST, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row (col discarded) */
}

/* ============================================================ *
 * Descriptor offset conveniences (Python `idx, _ = desc.offset(...)`).
 * The validity is discarded. On a sticky error these return NULL.
 * ============================================================ */

static ckc_value_t* ckc__ml_offset(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                   ckc_value_t* token,
                                   ckc_value_t* head,
                                   ckc_value_t* seg)
{
    const char* in_names[3] = {"token", "head", "seg"};
    ckc_value_t* in_values[3];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    if(!ckc_transforms_descriptor_offset(B, ctx->ml_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static ckc_value_t* ckc__seg_acc_offset(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                        ckc_value_t* token,
                                        ckc_value_t* head,
                                        ckc_value_t* seg,
                                        ckc_value_t* dim)
{
    const char* in_names[4] = {"token", "head", "seg", "dim"};
    ckc_value_t* in_values[4];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    in_values[3] = dim;
    if(!ckc_transforms_descriptor_offset(
           B, ctx->seg_acc_desc, in_names, in_values, 4, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static ckc_value_t* ckc__q_offset(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                  ckc_value_t* token,
                                  ckc_value_t* head,
                                  ckc_value_t* dim)
{
    const char* in_names[3] = {"token", "head", "dim"};
    ckc_value_t* in_values[3];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = dim;
    if(!ckc_transforms_descriptor_offset(B, ctx->q_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=) -> i32 element off.
 * The Python supplies exactly these three upper coords; the descriptor's
 * remaining lowers resolve through the transform chain. */
static ckc_value_t* ckc__paged_kv_offset(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                         ckc_value_t* tile_idx,
                                         ckc_value_t* linear_half,
                                         ckc_value_t* kv_head)
{
    const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
    ckc_value_t* in_values[3];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = tile_idx;
    in_values[1] = linear_half;
    in_values[2] = kv_head;
    if(!ckc_transforms_descriptor_offset(
           B, ctx->paged_kv_desc, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* ============================================================ *
 * ckc_gfx950_attention_tiled_3d_declare_params -- lines 303-358.
 *
 * The ~21 params in the load-bearing AITER order (segm_* workspace ptrs FIRST).
 * dtype / kv_io_dtype are taken from cfg (F16/BF16 ; FP8E4M3 when KV_FP8 else
 * dtype). Also sets the kernel attrs (max_workgroup_size, optional waves_per_eu).
 * ============================================================ */
void ckc_gfx950_attention_tiled_3d_declare_params(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_param_opts_t o;
    const ckc_type_t* dtype = CFG.dtype;
    const ckc_type_t* kv_io_dtype = CFG.kv_io_dtype;

    /* kernel attrs: max_workgroup_size = THREADS, optional waves_per_eu
     * (lines 303-305). */
    ckc_attr_set_int(B, &ctx->kernel->attrs, "max_workgroup_size", CFG.THREADS);
    if(ctx->spec != NULL && ctx->spec->has_waves_per_eu)
    {
        ckc_attr_set_int(B, &ctx->kernel->attrs, "waves_per_eu", ctx->spec->waves_per_eu);
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
        = ckc_b_param(B, "segm_output_ptr", ckc_ptr_type(B, ckc_f32(), "global"), &o);

    /* segm_max_ptr: F32* noalias writeonly align4 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.writeonly = true;
    o.writeonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->segm_max_ptr = ckc_b_param(B, "segm_max_ptr", ckc_ptr_type(B, ckc_f32(), "global"), &o);

    /* segm_expsum_ptr: F32* noalias writeonly align4 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.writeonly = true;
    o.writeonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->segm_expsum_ptr
        = ckc_b_param(B, "segm_expsum_ptr", ckc_ptr_type(B, ckc_f32(), "global"), &o);

    /* query_ptr: dtype* noalias readonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->query = ckc_b_param(B, "query_ptr", ckc_ptr_type(B, dtype, "global"), &o);

    /* key_cache_ptr: kv_io_dtype* noalias readonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->key = ckc_b_param(B, "key_cache_ptr", ckc_ptr_type(B, kv_io_dtype, "global"), &o);

    /* value_cache_ptr: kv_io_dtype* noalias readonly align16 */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->value = ckc_b_param(B, "value_cache_ptr", ckc_ptr_type(B, kv_io_dtype, "global"), &o);

    /* sink_ptr: dtype* readonly align16 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->sinks = ckc_b_param(B, "sink_ptr", ckc_ptr_type(B, dtype, "global"), &o);

    /* block_tables_ptr: I32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->block_tables
        = ckc_b_param(B, "block_tables_ptr", ckc_ptr_type(B, ckc_i32(), "global"), &o);

    /* seq_lens_ptr: I32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->seq_lens = ckc_b_param(B, "seq_lens_ptr", ckc_ptr_type(B, ckc_i32(), "global"), &o);

    /* alibi_slopes_ptr: F32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->alibi_slopes_ptr
        = ckc_b_param(B, "alibi_slopes_ptr", ckc_ptr_type(B, ckc_f32(), "global"), &o);

    /* qq_bias_ptr: F32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->qq_bias_ptr = ckc_b_param(B, "qq_bias_ptr", ckc_ptr_type(B, ckc_f32(), "global"), &o);

    /* query_start_len_ptr (cu_q): I32* readonly align4 */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->cu_q = ckc_b_param(B, "query_start_len_ptr", ckc_ptr_type(B, ckc_i32(), "global"), &o);

    /* scalar params (no ABI opts) */
    ctx->scale_p = ckc_b_param(B, "scale", ckc_f32(), NULL);
    ctx->k_scale_p = ckc_b_param(B, "k_scale", ckc_f32(), NULL);
    ctx->v_scale_p = ckc_b_param(B, "v_scale", ckc_f32(), NULL);
    ctx->softcap_p = ckc_b_param(B, "softcap", ckc_f32(), NULL);
    ctx->num_seqs_p = ckc_b_param(B, "num_seqs", ckc_i32(), NULL);
    ctx->bt_stride_p = ckc_b_param(B, "block_table_stride", ckc_i32(), NULL);
    ctx->qq_bias_stride0_p = ckc_b_param(B, "qq_bias_stride_0", ckc_i32(), NULL);
}

/* ============================================================ *
 * ckc_gfx950_attention_tiled_3d_emit_prologue -- lines 360-534.
 * ============================================================ */
void ckc_gfx950_attention_tiled_3d_emit_prologue(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int HD = CFG.HD;
    const int T = CFG.T;
    const int BLOCK_M = CFG.BLOCK_M;
    const int BLOCK_Q = CFG.BLOCK_Q;
    const int NUM_QH = CFG.NUM_QH;
    const int NUM_SEG = CFG.NUM_SEG;
    const int PV_K_STEP = CFG.PV_K_STEP;
    const ckc_type_t* dtype = CFG.dtype;

    /* ---- grid ids + thread (lines 360-363) ---- */
    ctx->q_block_global_idx = ckc_b_block_id_x(B);
    ctx->kv_head_idx = ckc_b_block_id_y(B);
    ctx->seg_idx = ckc_b_block_id_z(B);
    ctx->tid = ckc_b_thread_id_x(B);

    /* ---- binary search seq_idx (lines 365-372) ----
     * per_token=False (the Python helper default for the block-q search). */
    ctx->seq_idx = ckc_binary_search_seq_idx(B,
                                             ctx->cu_q,
                                             ctx->q_block_global_idx,
                                             ctx->num_seqs_p,
                                             BLOCK_Q,
                                             CFG.binary_search_iters,
                                             false);

    /* ---- cu_q bounds / per-sequence geometry (lines 373-379) ---- */
    ctx->cu_q_start = ckc_b_global_load_i32(B, ctx->cu_q, ctx->seq_idx, 0);
    ctx->cu_q_stop
        = ckc_b_global_load_i32(B, ctx->cu_q, ckc_b_add(B, ctx->seq_idx, ckc_b_const_i32(B, 1)), 0);
    ctx->cur_batch_q_len = ckc_b_sub(B, ctx->cu_q_stop, ctx->cu_q_start);
    ctx->q_block_start_idx
        = ckc_b_add(B, ckc_b_div(B, ctx->cu_q_start, ckc_b_const_i32(B, BLOCK_Q)), ctx->seq_idx);
    ctx->q_block_local_idx = ckc_b_sub(B, ctx->q_block_global_idx, ctx->q_block_start_idx);
    ctx->seq_len = ckc_b_global_load_i32(B, ctx->seq_lens, ctx->seq_idx, 0);
    ctx->context_len = ckc_b_sub(B, ctx->seq_len, ctx->cur_batch_q_len);

    /* qb_start_pos = q_block_local_idx * BLOCK_Q (line 381) */
    ctx->qb_start_pos = ckc_b_mul(B, ctx->q_block_local_idx, ckc_b_const_i32(B, BLOCK_Q));

    /* early return guard: if qb_start_pos >= cur_batch_q_len: ret() (lines 382-383) */
    {
        ckc_if_t g = ckc_b_scf_if(B, ckc_b_cmp_ge(B, ctx->qb_start_pos, ctx->cur_batch_q_len));
        ckc_b_region_enter(B, g.then_region);
        ckc_b_ret(B);
        ckc_b_region_leave(B);
    }

    /* tps = cdiv(seq_len, NUM_SEG*T) (line 386) */
    {
        ckc_value_t* tps_num = ckc_b_add(B, ctx->seq_len, ckc_b_const_i32(B, NUM_SEG * T - 1));
        ctx->tps = ckc_b_div(B, tps_num, ckc_b_const_i32(B, NUM_SEG * T));
    }

    /* ---- descriptors (lines 399-413) ---- */
    {
        const int ml_lengths[3] = {1 << 30, NUM_QH, NUM_SEG};
        const char* ml_coords[3] = {"token", "head", "seg"};
        ctx->ml_desc = ckc_tensor_descriptor_naive(B, "segm_ml", ml_lengths, 3, NULL, ml_coords, 3);
    }
    {
        const int sa_lengths[4] = {1 << 30, NUM_QH, NUM_SEG, HD};
        const char* sa_coords[4] = {"token", "head", "seg", "dim"};
        ctx->seg_acc_desc
            = ckc_tensor_descriptor_naive(B, "segm_output", sa_lengths, 4, NULL, sa_coords, 4);
    }
    {
        const int q_lengths[3] = {1 << 30, NUM_QH, HD};
        const char* q_coords[3] = {"token", "head", "dim"};
        ctx->q_desc = ckc_tensor_descriptor_naive(B, "Q", q_lengths, 3, NULL, q_coords, 3);
    }

    /* seg_start_tile_pos = seg_idx*tps*T (line 415) */
    {
        ckc_value_t* sst_inner = ckc_b_mul(B, ctx->seg_idx, ctx->tps);
        ctx->seg_start_tile_pos = ckc_b_mul(B, sst_inner, ckc_b_const_i32(B, T));
    }

    /* early-out zero-fill block (lines 416-461) */
    ckc_gfx950_attention_tiled_3d_emit_early_zero_fill(ctx);

    /* ---- LDS layout (lines 464-467) ---- */
    {
        const int q_shape[2] = {BLOCK_M, HD};
        const int k_shape[3] = {2, T, HD};
        ctx->Q_lds = ckc_b_smem_alloc(B, dtype, q_shape, 2, "Qlds");
        ctx->K_lds = ckc_b_smem_alloc(B, dtype, k_shape, 3, "Klds");
        ctx->V_lds = ckc_b_smem_alloc(B, dtype, k_shape, 3, "Vlds");
    }
    {
        const int p_shape[2] = {BLOCK_M, T};
        ctx->P_lds = ckc_b_smem_alloc(B, dtype, p_shape, 2, "Plds");
    }

    /* ---- PV transpose-read state (gfx950-only, lines 471-472) ----
     * pv_tr_reader = TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, tid)
     * tr_col_lane  = pv_tr_reader.col */
    {
        ckc_transpose_lds_reader_t r;
        r.K = PV_K_STEP;
        r.M = 16;
        ctx->pv_tr_reader = ckc_transpose_lds_reader_bind(B, &r, ctx->tid);
        ctx->tr_col_lane = (ctx->pv_tr_reader != NULL) ? ctx->pv_tr_reader->col : NULL;
    }

    /* ---- SSA constants (lines 474-480) ---- */
    ctx->neg_inf = ckc_b_const_f32(B, -INFINITY);
    ctx->zero_f = ckc_b_const_f32(B, 0.0);
    ctx->one_f = ckc_b_const_f32(B, 1.0);
    ctx->rcp_ln2 = ckc_b_const_f32(B, 1.4426950408889634);
    ctx->qk_scale = ckc_b_fmul(B, ctx->scale_p, ctx->rcp_ln2);
    ctx->sw_const = ckc_b_const_i32(B, CFG.SLIDING_WINDOW);
    ctx->z8 = ckc_b_zero_vec(B, dtype, 8);

    /* ---- Q -> LDS (lines 482-512) ---- */
    ckc_gfx950_attention_tiled_3d_emit_q_to_lds(ctx);

    /* ---- Per-segment tile range (lines 514-523) ---- */
    {
        ckc_value_t* msp_inner = ckc_b_add(B, ctx->context_len, ctx->qb_start_pos);
        ckc_value_t* msp_raw = ckc_b_add(B, msp_inner, ckc_b_const_i32(B, CFG.bm1_div_nqk + 1));
        ckc_value_t* msp_cmp = ckc_b_cmp_lt(B, msp_raw, ctx->seq_len);
        ckc_value_t* nt_inner;
        ckc_value_t* tile_end_raw_mul_inner;
        ckc_value_t* tile_end_raw;
        ckc_value_t* tile_end_cmp;
        ctx->max_seq_prefix_len = ckc_b_select(B, msp_cmp, msp_raw, ctx->seq_len);
        nt_inner = ckc_b_add(B, ctx->max_seq_prefix_len, ckc_b_const_i32(B, T - 1));
        ctx->num_tiles = ckc_b_div(B, nt_inner, ckc_b_const_i32(B, T));

        ctx->tile_start = ckc_b_mul(B, ctx->seg_idx, ctx->tps);
        tile_end_raw_mul_inner = ckc_b_add(B, ctx->seg_idx, ckc_b_const_i32(B, 1));
        tile_end_raw = ckc_b_mul(B, tile_end_raw_mul_inner, ctx->tps);
        tile_end_cmp = ckc_b_cmp_lt(B, tile_end_raw, ctx->num_tiles);
        ctx->tile_end = ckc_b_select(B, tile_end_cmp, tile_end_raw, ctx->num_tiles);
    }

    /* ---- lane decode (lines 533-534) ---- */
    ctx->lane_rg = ckc_b_div(B, ctx->tid, ckc_b_const_i32(B, 16));
    ctx->lane_col = ckc_b_mod(B, ctx->tid, ckc_b_const_i32(B, 16));

    /* NOTE: the async DMA infra (lines 564-585) and the paged-KV descriptor
     * (lines 587-601) are emitted AFTER acc_zero in Python's single linear
     * build; they are emitted by emit_loop_init (via
     * ckc_gfx950_attention_tiled_3d_emit_async_infra) so the SSA emission order
     * matches Python byte-for-byte. */
}

/* ============================================================ *
 * ckc_gfx950_attention_tiled_3d_emit_async_infra -- lines 564-601.
 *
 * Emitted from emit_loop_init right after acc_zero so the op order matches the
 * single linear Python build.
 *
 * Builds: big_bytes / key_rsrc / value_rsrc, lane_half_base = tid*8,
 * K/V_lds_addr, zero_soff, seq_base = seq_idx*bt_stride_p, and the paged-KV
 * byte descriptor (naive 4D + indirect("tile_idx") + unmerge("linear_half")).
 * ============================================================ */
void ckc_gfx950_attention_tiled_3d_emit_async_infra(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int HD = CFG.HD;
    const int T = CFG.T;
    const int NUM_KV = CFG.NUM_KV;

    /* ---- async DMA infra (lines 565-585) ---- */
    ctx->big_bytes = ckc_b_const_i32(B, 0x7FFF0000);
    ctx->key_rsrc = ckc_b_buffer_rsrc(B, ctx->key, ctx->big_bytes);
    ctx->value_rsrc = ckc_b_buffer_rsrc(B, ctx->value, ctx->big_bytes);
    /* kv_block_bytes_c = const_i32(kv_stride_blk_b): one-block buffer bound for
     * the i64-addressing path (Python creates it unconditionally right after the
     * byte strides; unused in the i32 path, where it is DCE'd). */
    ctx->kv_block_bytes_c = ckc_b_const_i32(B, CFG.kv_stride_blk_b);
    ctx->lane_half_base = ckc_b_mul(B, ctx->tid, ckc_b_const_i32(B, CFG.HALVES_PER_LANE));
    ctx->K_lds_addr = ckc_b_smem_addr_of(B, ctx->K_lds);
    ctx->V_lds_addr = ckc_b_smem_addr_of(B, ctx->V_lds);
    ctx->zero_soff = ckc_b_const_i32(B, 0);

    /* seq_base = seq_idx * bt_stride_p (line 592) */
    ctx->seq_base = ckc_b_mul(B, ctx->seq_idx, ctx->bt_stride_p);

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
        ckc_tensor_descriptor_t* base
            = ckc_tensor_descriptor_naive(B, "paged_kv_bytes", lengths, 4, strides, coords, 4);

        const ckc_transform_t* chain[2];
        const char* into_td[2] = {"token", "dim"};
        const int dims_td[2] = {T, HD};
        chain[0] = ckc_indirect(
            B, "tile_idx", "physical_block", ctx->block_tables, ctx->seq_base, NULL, 0);
        chain[1] = ckc_unmerge(B, "linear_half", into_td, 2, dims_td);
        ctx->paged_kv_desc = ckc_tensor_descriptor_transform(B, base, chain, 2);
    }
}

/* ============================================================ *
 * ckc_gfx950_attention_tiled_3d_emit_early_zero_fill -- lines 416-461.
 * ============================================================ */
void ckc_gfx950_attention_tiled_3d_emit_early_zero_fill(
    ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int NQK = CFG.NQK;
    const int NUM_QH = CFG.NUM_QH;
    const int PV_N_TILES = CFG.PV_N_TILES;
    int n, reg;

    ckc_if_t guard = ckc_b_scf_if(B, ckc_b_cmp_ge(B, ctx->seg_start_tile_pos, ctx->seq_len));
    ckc_b_region_enter(B, guard.then_region);
    {
        ckc_value_t* neg_inf_local = ckc_b_const_f32(B, -INFINITY);
        ckc_value_t* zero_local = ckc_b_const_f32(B, 0.0);
        ckc_value_t* lwm_mod = ckc_b_mod(B, ctx->tid, ckc_b_const_i32(B, 16));
        ckc_value_t* lane_writes_ml_e = ckc_b_cmp_eq(B, lwm_mod, ckc_b_const_i32(B, 0));

        /* ml zero-fill (lines 420-434) */
        for(reg = 0; reg < 4; ++reg)
        {
            ckc_value_t* row = ckc_gfx950_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
            ckc_value_t* qp_r_div = ckc_b_div(B, row, ckc_b_const_i32(B, NQK));
            ckc_value_t* qp_r = ckc_b_add(B, ctx->qb_start_pos, qp_r_div);
            ckc_value_t* qh_r_mul = ckc_b_mul(B, ctx->kv_head_idx, ckc_b_const_i32(B, NQK));
            ckc_value_t* qh_r_mod = ckc_b_mod(B, row, ckc_b_const_i32(B, NQK));
            ckc_value_t* qh_r = ckc_b_add(B, qh_r_mul, qh_r_mod);
            ckc_value_t* row_ok_a = ckc_b_cmp_lt(B, qp_r, ctx->cur_batch_q_len);
            ckc_value_t* row_ok_b = ckc_b_cmp_lt(B, qh_r, ckc_b_const_i32(B, NUM_QH));
            ckc_value_t* row_ok = ckc_b_land(B, row_ok_a, row_ok_b);
            ckc_value_t* qp_r_safe = ckc_b_select(B, row_ok, qp_r, ckc_b_const_i32(B, 0));
            ckc_value_t* qh_r_safe = ckc_b_select(B, row_ok, qh_r, ckc_b_const_i32(B, 0));
            ckc_value_t* qtoken = ckc_b_add(B, ctx->cu_q_start, qp_r_safe);
            ckc_value_t* ml_idx = ckc__ml_offset(ctx, qtoken, qh_r_safe, ctx->seg_idx);
            ckc_if_t w = ckc_b_scf_if(B, lane_writes_ml_e);
            ckc_b_region_enter(B, w.then_region);
            ckc_b_global_store(B, ctx->segm_max_ptr, ml_idx, neg_inf_local, 4);
            ckc_b_global_store(B, ctx->segm_expsum_ptr, ml_idx, zero_local, 4);
            ckc_b_region_leave(B);
        }

        /* seg_acc zero-fill (lines 436-460) */
        {
            ckc_value_t* lane_col_e = ckc_b_mod(B, ctx->tid, ckc_b_const_i32(B, 16));
            for(n = 0; n < PV_N_TILES; ++n)
            {
                for(reg = 0; reg < 4; ++reg)
                {
                    ckc_value_t* row
                        = ckc_gfx950_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
                    ckc_value_t* col_n = ckc_b_const_i32(B, n);
                    ckc_value_t* col_16 = ckc_b_const_i32(B, 16);
                    ckc_value_t* col_mul = ckc_b_mul(B, col_n, col_16);
                    ckc_value_t* col = ckc_b_add(B, col_mul, lane_col_e);
                    ckc_value_t* qp_r_div = ckc_b_div(B, row, ckc_b_const_i32(B, NQK));
                    ckc_value_t* qp_r = ckc_b_add(B, ctx->qb_start_pos, qp_r_div);
                    ckc_value_t* qh_r_mul = ckc_b_mul(B, ctx->kv_head_idx, ckc_b_const_i32(B, NQK));
                    ckc_value_t* qh_r_mod = ckc_b_mod(B, row, ckc_b_const_i32(B, NQK));
                    ckc_value_t* qh_r = ckc_b_add(B, qh_r_mul, qh_r_mod);
                    ckc_value_t* row_ok_a = ckc_b_cmp_lt(B, qp_r, ctx->cur_batch_q_len);
                    ckc_value_t* row_ok_b = ckc_b_cmp_lt(B, qh_r, ckc_b_const_i32(B, NUM_QH));
                    ckc_value_t* row_ok = ckc_b_land(B, row_ok_a, row_ok_b);
                    ckc_value_t* qp_r_safe = ckc_b_select(B, row_ok, qp_r, ckc_b_const_i32(B, 0));
                    ckc_value_t* qh_r_safe = ckc_b_select(B, row_ok, qh_r, ckc_b_const_i32(B, 0));
                    ckc_value_t* qtoken = ckc_b_add(B, ctx->cu_q_start, qp_r_safe);
                    ckc_value_t* seg_acc_idx
                        = ckc__seg_acc_offset(ctx, qtoken, qh_r_safe, ctx->seg_idx, col);
                    ckc_b_global_store(B, ctx->segm_output_ptr, seg_acc_idx, zero_local, 4);
                }
            }
        }
        ckc_b_ret(B);
    }
    ckc_b_region_leave(B);
}

/* ============================================================ *
 * ckc_gfx950_attention_tiled_3d_emit_q_to_lds -- lines 482-512.
 * ============================================================ */
void ckc_gfx950_attention_tiled_3d_emit_q_to_lds(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    const int NQK = CFG.NQK;
    const int NUM_QH = CFG.NUM_QH;
    const int THREADS = CFG.THREADS;
    const int Q_VECS_PER_ROW = CFG.Q_VECS_PER_ROW;
    const int Q_VECS_PER_THREAD = CFG.Q_VECS_PER_THREAD;
    const ckc_type_t* dtype = CFG.dtype;
    int li;

    for(li = 0; li < Q_VECS_PER_THREAD; ++li)
    {
        ckc_value_t* q_vid_li = ckc_b_const_i32(B, li);
        ckc_value_t* q_vid_thr = ckc_b_const_i32(B, THREADS);
        ckc_value_t* q_vid_mul = ckc_b_mul(B, q_vid_li, q_vid_thr);
        ckc_value_t* q_vid = ckc_b_add(B, q_vid_mul, ctx->tid);
        ckc_value_t* Q_row = ckc_b_div(B, q_vid, ckc_b_const_i32(B, Q_VECS_PER_ROW));
        ckc_value_t* Q_col_mod = ckc_b_mod(B, q_vid, ckc_b_const_i32(B, Q_VECS_PER_ROW));
        ckc_value_t* Q_col = ckc_b_mul(B, Q_col_mod, ckc_b_const_i32(B, 8));
        ckc_value_t* q_pos_div = ckc_b_div(B, Q_row, ckc_b_const_i32(B, NQK));
        ckc_value_t* q_pos_t = ckc_b_add(B, ctx->qb_start_pos, q_pos_div);
        ckc_value_t* qh_t_mul = ckc_b_mul(B, ctx->kv_head_idx, ckc_b_const_i32(B, NQK));
        ckc_value_t* qh_t_mod = ckc_b_mod(B, Q_row, ckc_b_const_i32(B, NQK));
        ckc_value_t* qh_t = ckc_b_add(B, qh_t_mul, qh_t_mod);
        ckc_value_t* qmask_a = ckc_b_cmp_lt(B, q_pos_t, ctx->cur_batch_q_len);
        ckc_value_t* qmask_b = ckc_b_cmp_lt(B, qh_t, ckc_b_const_i32(B, NUM_QH));
        ckc_value_t* qmask_t = ckc_b_land(B, qmask_a, qmask_b);
        ckc_value_t* q_pos_safe = ckc_b_select(B, qmask_t, q_pos_t, ckc_b_const_i32(B, 0));
        ckc_value_t* qh_safe = ckc_b_select(B, qmask_t, qh_t, ckc_b_const_i32(B, 0));
        ckc_value_t* q_off_tok = ckc_b_add(B, ctx->cu_q_start, q_pos_safe);
        ckc_value_t* q_off_base = ckc__q_offset(ctx, q_off_tok, qh_safe, ckc_b_const_i32(B, 0));
        ckc_value_t* v8_idx = ckc_b_add(B, q_off_base, Q_col);
        ckc_value_t* v8 = ckc_b_global_load_vN(B, ctx->query, v8_idx, dtype, 8, 16);
        ckc_value_t* splat = ckc_b_vector_splat(B, qmask_t, 8);
        ckc_value_t* sel = ckc_b_vector_select(B, splat, v8, ctx->z8);
        ckc_value_t* idxs[2] = {Q_row, Q_col};
        ckc_b_smem_store_vN(B, ctx->Q_lds, idxs, 2, sel, 8);
    }
    ckc_b_sync(B);
}

/* ============================================================ *
 * Load issuers (Python closures, lines 603-713).
 * ============================================================ */

/* _issue_k_load(kv_tile_idx, buf_idx) -- lines 603-616. */
void ckc_gfx950_attention_tiled_3d_issue_k_load(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                                ckc_value_t* kv_tile_idx,
                                                ckc_value_t* buf_idx)
{
    const int KV_HALVES_PER_CALL = CFG.KV_HALVES_PER_CALL;
    const int kv_calls_per_tile = CFG.kv_calls_per_tile;
    const int bytes_per_call = CFG.bytes_per_call;
    const int bytes_per_buf = CFG.bytes_per_buf;
    const int ASYNC_LDS_DWORDS = CFG.ASYNC_LDS_DWORDS;
    int call;

    ckc_value_t* buf_off_i32 = ckc_b_mul(B, buf_idx, ckc_b_const_i32(B, bytes_per_buf));
    ckc_value_t* buf_off_i64 = ckc_b_zext(B, buf_off_i32, ckc_i64());
    ckc_value_t* K_buf_base = ckc_b_smem_ptr_add(B, ctx->K_lds_addr, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        ckc_value_t* linear_half
            = ckc_b_add(B, ckc_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        ckc_value_t* call_rsrc = ctx->key_rsrc;
        ckc_value_t* voff = NULL;
        if(CFG.I64_KV_ADDR)
        {
            /* offset_i64_split folds the per-block byte base into a 64-bit
             * buffer base (no 2 GiB i32-voffset overflow); only the within-block
             * byte offset stays in the i32 voffset. */
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            ckc_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            ckc_value_t* base_i64 = NULL;
            ckc_value_t* valid = NULL;
            if(!ckc_transforms_descriptor_offset_i64_split(B,
                                                           ctx->paged_kv_desc,
                                                           "physical_block",
                                                           in_names,
                                                           in_values,
                                                           3,
                                                           &base_i64,
                                                           &voff,
                                                           &valid))
                return;
            call_rsrc = ckc_b_buffer_rsrc(
                B, ckc_b_global_ptr_add(B, ctx->key, base_i64), ctx->kv_block_bytes_c);
        }
        else
        {
            voff = ckc__paged_kv_offset(ctx, kv_tile_idx, linear_half, ctx->kv_head_idx);
        }
        ckc_value_t* k_dst
            = ckc_b_smem_ptr_add(B, K_buf_base, ckc_b_const_i64(B, (int64_t)call * bytes_per_call));
        ckc_b_async_buffer_load_lds_addr(
            B, call_rsrc, k_dst, voff, ctx->zero_soff, ASYNC_LDS_DWORDS, CKC_CACHE_ALL);
    }
}

/* _issue_v_load(kv_tile_idx, buf_idx) -- lines 618-631. */
void ckc_gfx950_attention_tiled_3d_issue_v_load(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                                ckc_value_t* kv_tile_idx,
                                                ckc_value_t* buf_idx)
{
    const int KV_HALVES_PER_CALL = CFG.KV_HALVES_PER_CALL;
    const int kv_calls_per_tile = CFG.kv_calls_per_tile;
    const int bytes_per_call = CFG.bytes_per_call;
    const int bytes_per_buf = CFG.bytes_per_buf;
    const int ASYNC_LDS_DWORDS = CFG.ASYNC_LDS_DWORDS;
    int call;

    ckc_value_t* buf_off_i32 = ckc_b_mul(B, buf_idx, ckc_b_const_i32(B, bytes_per_buf));
    ckc_value_t* buf_off_i64 = ckc_b_zext(B, buf_off_i32, ckc_i64());
    ckc_value_t* V_buf_base = ckc_b_smem_ptr_add(B, ctx->V_lds_addr, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        ckc_value_t* linear_half
            = ckc_b_add(B, ckc_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        ckc_value_t* call_rsrc = ctx->value_rsrc;
        ckc_value_t* voff = NULL;
        if(CFG.I64_KV_ADDR)
        {
            const char* in_names[3] = {"tile_idx", "linear_half", "kv_head"};
            ckc_value_t* in_values[3] = {kv_tile_idx, linear_half, ctx->kv_head_idx};
            ckc_value_t* base_i64 = NULL;
            ckc_value_t* valid = NULL;
            if(!ckc_transforms_descriptor_offset_i64_split(B,
                                                           ctx->paged_kv_desc,
                                                           "physical_block",
                                                           in_names,
                                                           in_values,
                                                           3,
                                                           &base_i64,
                                                           &voff,
                                                           &valid))
                return;
            call_rsrc = ckc_b_buffer_rsrc(
                B, ckc_b_global_ptr_add(B, ctx->value, base_i64), ctx->kv_block_bytes_c);
        }
        else
        {
            voff = ckc__paged_kv_offset(ctx, kv_tile_idx, linear_half, ctx->kv_head_idx);
        }
        ckc_value_t* v_dst
            = ckc_b_smem_ptr_add(B, V_buf_base, ckc_b_const_i64(B, (int64_t)call * bytes_per_call));
        ckc_b_async_buffer_load_lds_addr(
            B, call_rsrc, v_dst, voff, ctx->zero_soff, ASYNC_LDS_DWORDS, CKC_CACHE_ALL);
    }
}

/* _issue_fp8_dequant_loads(kv_tile_idx, buf_idx, lds_token) -- lines 645-701.
 * is_value: false => "K" (key, k_scale, K_lds), true => "V".
 *
 * Sync per-thread fp8 -> cvt_pk_f32_fp8x4 -> *scale (UNFUSED fmul) -> dtype ->
 * LDS via ckc_dequant_fp8x8_to_dtype. The unfused fmul is the CRITICAL invariant
 * (the fused E8M0 cvt would truncate non-power-of-two per-tensor scales). */
void ckc_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(
    ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
    bool is_value,
    ckc_value_t* kv_tile_idx,
    ckc_value_t* buf_idx)
{
    const int HD = CFG.HD;
    const int THREADS = CFG.THREADS;
    const int fp8_elems_per_chunk = CFG.fp8_elems_per_chunk;
    const int fp8_chunks_per_thread = CFG.fp8_chunks_per_thread;
    const ckc_type_t* dtype = CFG.dtype;
    ckc_value_t* scale = is_value ? ctx->v_scale_p : ctx->k_scale_p;
    ckc_value_t* lds = is_value ? ctx->V_lds : ctx->K_lds;
    ckc_value_t* src = is_value ? ctx->value : ctx->key;
    int call;

    for(call = 0; call < fp8_chunks_per_thread; ++call)
    {
        ckc_value_t* chunk_call = ckc_b_const_i32(B, call);
        ckc_value_t* chunk_thr = ckc_b_const_i32(B, THREADS);
        ckc_value_t* chunk_mul = ckc_b_mul(B, chunk_call, chunk_thr);
        ckc_value_t* chunk_id = ckc_b_add(B, chunk_mul, ctx->tid);
        ckc_value_t* row = ckc_b_div(B, chunk_id, ckc_b_const_i32(B, HD / fp8_elems_per_chunk));
        ckc_value_t* col_mod = ckc_b_mod(B, chunk_id, ckc_b_const_i32(B, HD / fp8_elems_per_chunk));
        ckc_value_t* col = ckc_b_mul(B, col_mod, ckc_b_const_i32(B, fp8_elems_per_chunk));
        ckc_value_t* lhf_mul = ckc_b_mul(B, row, ckc_b_const_i32(B, HD));
        ckc_value_t* linear_half_first = ckc_b_add(B, lhf_mul, col);
        /* I64_KV_ADDR fp8: the Python fp8 sync loader uses the full per-lane
         * offset_i64 here (caches > 2 GiB). That non-split i64 paged variant is
         * not yet on the frozen C transforms surface (same limitation the 2D C
         * fp8 twin documents), so the C fp8 path resolves through the i32 offset
         * regardless. The byte-identity gate never pairs fp8 with i64 KV, so this
         * does not affect the gate; the fp16/bf16 i64 async loaders above are the
         * shipping correctness fix and are fully mirrored. */
        ckc_value_t* voff
            = ckc__paged_kv_offset(ctx, kv_tile_idx, linear_half_first, ctx->kv_head_idx);
        ckc_value_t* fp8_vec = ckc_b_global_load_vN(
            B, src, voff, ckc_fp8e4m3(), fp8_elems_per_chunk, fp8_elems_per_chunk);
        ckc_value_t* packed = ckc_dequant_fp8x8_to_dtype(B, fp8_vec, scale, dtype);
        ckc_value_t* idxs[3];
        idxs[0] = buf_idx;
        idxs[1] = row;
        idxs[2] = col;
        ckc_b_smem_store_vN(B, lds, idxs, 3, packed, fp8_elems_per_chunk);
    }
}

/* _issue_k(tile_idx, buf_idx) -- lines 703-707. */
void ckc_gfx950_attention_tiled_3d_issue_k(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                           ckc_value_t* tile_idx,
                                           ckc_value_t* buf_idx)
{
    if(CFG.KV_FP8)
    {
        ckc_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(ctx, false, tile_idx, buf_idx);
    }
    else
    {
        ckc_gfx950_attention_tiled_3d_issue_k_load(ctx, tile_idx, buf_idx);
    }
}

/* _issue_v(tile_idx, buf_idx) -- lines 709-713. */
void ckc_gfx950_attention_tiled_3d_issue_v(ckc_gfx950_attention_tiled_3d_build_ctx_t* ctx,
                                           ckc_value_t* tile_idx,
                                           ckc_value_t* buf_idx)
{
    if(CFG.KV_FP8)
    {
        ckc_gfx950_attention_tiled_3d_issue_fp8_dequant_loads(ctx, true, tile_idx, buf_idx);
    }
    else
    {
        ckc_gfx950_attention_tiled_3d_issue_v_load(ctx, tile_idx, buf_idx);
    }
}
