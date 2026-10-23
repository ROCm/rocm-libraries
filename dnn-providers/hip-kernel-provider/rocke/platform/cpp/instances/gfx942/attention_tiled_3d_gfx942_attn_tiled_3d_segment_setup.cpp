// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_3d_gfx942_attn_tiled_3d_segment_setup.c --
 * one part-file of the chunked C99 port of
 * rocke/instances/gfx942/attention_tiled_3d.py (arch gfx942).
 *
 * SCOPE OF THIS PART-FILE: the segment-kernel prologue + load-issuer closures +
 * inner IR helpers of build_unified_attention_3d_tiled:
 *
 *   rocke_gfx942_attention_tiled_3d_declare_params        (params, lines 281-329)
 *   rocke_gfx942_attention_tiled_3d_emit_prologue         (lines 331-483)
 *   rocke_gfx942_attention_tiled_3d_emit_early_zero_fill  (lines 376-419)
 *   rocke_gfx942_attention_tiled_3d_emit_q_to_lds         (lines 437-467)
 *   rocke_gfx942_attention_tiled_3d_mfma_16x16_c_row      (lines 79-88)
 *   rocke_gfx942_attention_tiled_3d_issue_k_load          (lines 604-617)
 *   rocke_gfx942_attention_tiled_3d_issue_v_load          (lines 619-632)
 *   rocke_gfx942_attention_tiled_3d_issue_wide_load       (lines 645-658)
 *   rocke_gfx942_attention_tiled_3d_issue_fp8_dequant_loads (lines 669-701)
 *   rocke_gfx942_attention_tiled_3d_issue_k               (lines 705-711)
 *   rocke_gfx942_attention_tiled_3d_issue_v               (lines 713-719)
 *   + the wave page-id window helpers (_load_page_id_window,
 *     _window_tile_page_id, _refill_page_id_window).
 *   + the paged-KV descriptor build (kv_base_desc + paged_kv_desc, lines 570-602)
 *     emitted inside emit_prologue.
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

#include "rocke/instance_gfx942_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Local conveniences (no IR; mirror the Python builder aliases).
 * ============================================================ */

#define B (ctx->b)
#define CFG (ctx->cfg)

/* ------------------------------------------------------------- *
 * _mfma_16x16_c_row(b, lane, reg) -- lines 79-88.
 *
 *   m_blk = b.div(lane, const_i32(16))
 *   n     = b.mod(lane, const_i32(16))
 *   row, _col = _C16_DIST.calculate_x(b, ys=[const_i32(0), const_i32(reg)],
 *                                     ps=[[m_blk, n]])
 *   return row
 * ------------------------------------------------------------- */
rocke_value_t* rocke_gfx942_attention_tiled_3d_mfma_16x16_c_row(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx, rocke_value_t* lane, int reg)
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

static rocke_value_t* rocke__ml_offset(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
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

static rocke_value_t* rocke__seg_acc_offset(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
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

static rocke_value_t* rocke__q_offset(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
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
static rocke_value_t* rocke__paged_kv_offset(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
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
 * rocke_gfx942_attention_tiled_3d_declare_params -- lines 281-329.
 *
 * The ~16 params in the load-bearing spec-dependent order. dtype / kv_io_dtype
 * are taken from cfg (F16/BF16 ; FP8E4M3 when KV_FP8 else dtype).
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_declare_params(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_param_opts_t o;
    const rocke_type_t* dtype = CFG.dtype;
    const rocke_type_t* kv_io_dtype = CFG.kv_io_dtype;

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
 * Build kv_base_desc + paged_kv_desc -- lines 570-602.
 *
 * The Python transform chain uses indirect()/embed()/unmerge(); all three are
 * now available in the C transforms surface (helper_rocke.helpers.transforms.h)
 * so the chain is built byte-identically below.
 * ============================================================ */
static void rocke__build_paged_kv_desc(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    const int HD = CFG.HD;
    const int T = CFG.T;
    const int BS = CFG.BS;
    const int NUM_KV = CFG.NUM_KV;

    /* seq_base = seq_idx * bt_stride_p (line 569) */
    if(ctx->strided_kv)
    {
        return;
    }
    ctx->seq_base = rocke_b_mul(B, ctx->seq_idx, ctx->bt_stride_p);

    /* _kv_base = TensorDescriptor.naive("paged_kv_bytes",
     *   lengths=[1<<24, BS, NUM_KV, HD],
     *   strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
     *   coord_names=("physical_block","token","kv_head","dim")) (lines 570-575) */
    {
        const int lengths[4] = {1 << 24, BS, NUM_KV, HD};
        const int strides[4]
            = {CFG.kv_stride_blk_b, CFG.kv_stride_tok_b, CFG.kv_stride_h_b, CFG.KV_BYTES};
        const char* coords[4] = {"physical_block", "token", "kv_head", "dim"};
        ctx->kv_base_desc
            = rocke_tensor_descriptor_naive(B, "paged_kv_bytes", lengths, 4, strides, coords, 4);
    }

    /* Transform chain (lines 576-602). */
    if(T == BS)
    {
        /* paged_kv_desc = _kv_base.transform(
         *     indirect("tile_idx", into="physical_block",
         *              table=block_tables, base=seq_base),
         *     unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
         * ) */
        const rocke_transform_t* chain[2];
        const char* into_td[2] = {"token", "dim"};
        const int dims_td[2] = {T, HD};
        chain[0] = rocke_indirect(
            B, "tile_idx", "physical_block", ctx->block_tables, ctx->seq_base, NULL, 0);
        chain[1] = rocke_unmerge(B, "linear_half", into_td, 2, dims_td);
        ctx->paged_kv_desc = rocke_tensor_descriptor_transform(B, ctx->kv_base_desc, chain, 2);
    }
    else
    {
        /* assert BS % T == 0; BLOCKS_PER_CACHE_BLOCK = BS // T
         * paged_kv_desc = _kv_base.transform(
         *     unmerge("tile_idx",
         *             into=("linear_block_idx", "tile_within_block"),
         *             dims=(1<<24, BLOCKS_PER_CACHE_BLOCK)),
         *     indirect("linear_block_idx", into="physical_block",
         *              table=block_tables, base=seq_base),
         *     unmerge("linear_half", into=("token_in_tile", "dim"), dims=(T, HD)),
         *     embed(("tile_within_block", "token_in_tile"),
         *           into="token", strides=(T, 1)),
         * ) */
        const int BLOCKS_PER_CACHE_BLOCK = BS / T;
        const rocke_transform_t* chain[4];
        const char* into_tib[2] = {"linear_block_idx", "tile_within_block"};
        const int dims_tib[2] = {1 << 24, BLOCKS_PER_CACHE_BLOCK};
        const char* into_ttd[2] = {"token_in_tile", "dim"};
        const int dims_ttd[2] = {T, HD};
        const char* emb_up[2] = {"tile_within_block", "token_in_tile"};
        const int emb_str[2] = {T, 1};
        chain[0] = rocke_unmerge(B, "tile_idx", into_tib, 2, dims_tib);
        chain[1] = rocke_indirect(
            B, "linear_block_idx", "physical_block", ctx->block_tables, ctx->seq_base, NULL, 0);
        chain[2] = rocke_unmerge(B, "linear_half", into_ttd, 2, dims_ttd);
        chain[3] = rocke_embed(B, emb_up, 2, "token", emb_str, 0);
        ctx->paged_kv_desc = rocke_tensor_descriptor_transform(B, ctx->kv_base_desc, chain, 4);
    }
}

/* ============================================================ *
 * rocke_gfx942_attention_tiled_3d_emit_prologue -- lines 331-483.
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_emit_prologue(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    const int HD = CFG.HD;
    const int T = CFG.T;
    const int BLOCK_M = CFG.BLOCK_M;
    const int BLOCK_Q = CFG.BLOCK_Q;
    const int NUM_QH = CFG.NUM_QH;
    const int NUM_SEG = CFG.NUM_SEG;
    const rocke_type_t* dtype = CFG.dtype;

    /* ---- grid ids + thread (lines 331-334) ---- */
    ctx->q_block_global_idx = rocke_b_block_id_x(B);
    ctx->kv_head_idx = rocke_b_block_id_y(B);
    ctx->seg_idx = rocke_b_block_id_z(B);
    ctx->tid = rocke_b_thread_id_x(B);

    /* ---- binary search seq_idx (lines 336-343) ----
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

    /* ---- cu_q bounds / per-sequence geometry (lines 344-350) ---- */
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

    /* qb_start_pos = q_block_local_idx * BLOCK_Q (line 352) */
    ctx->qb_start_pos = rocke_b_mul(B, ctx->q_block_local_idx, rocke_b_const_i32(B, BLOCK_Q));

    /* early return guard: if qb_start_pos >= cur_batch_q_len: ret() (lines 353-354) */
    {
        rocke_if_t g
            = rocke_b_scf_if(B, rocke_b_cmp_ge(B, ctx->qb_start_pos, ctx->cur_batch_q_len));
        rocke_b_region_enter(B, g.then_region);
        rocke_b_ret(B);
        rocke_b_region_leave(B);
    }

    /* tps = cdiv(seq_len, NUM_SEG*T) (line 357) */
    {
        rocke_value_t* tps_num
            = rocke_b_add(B, ctx->seq_len, rocke_b_const_i32(B, NUM_SEG * T - 1));
        ctx->tps = rocke_b_div(B, tps_num, rocke_b_const_i32(B, NUM_SEG * T));
    }

    /* ---- descriptors (lines 359-373) ---- */
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

    /* seg_start_tile_pos = seg_idx*tps*T (line 375) */
    {
        rocke_value_t* sst_inner = rocke_b_mul(B, ctx->seg_idx, ctx->tps);
        ctx->seg_start_tile_pos = rocke_b_mul(B, sst_inner, rocke_b_const_i32(B, T));
    }

    /* early-out zero-fill block (lines 376-419) */
    rocke_gfx942_attention_tiled_3d_emit_early_zero_fill(ctx);

    /* ---- LDS layout (lines 424-427) ---- */
    {
        const int q_shape[2] = {BLOCK_M, HD};
        const int k_shape[3] = {2, T, HD};
        /* One V slot: V(i) is issued and consumed within iteration i. */
        const int v_shape[3] = {1, T, HD};
        ctx->Q_lds = rocke_b_smem_alloc(B, dtype, q_shape, 2, "Qlds");
        ctx->K_lds = rocke_b_smem_alloc(B, dtype, k_shape, 3, "Klds");
        ctx->V_lds = rocke_b_smem_alloc(B, dtype, v_shape, 3, "Vlds");
    }

    /* ---- SSA constants (lines 429-435) ---- */
    ctx->neg_inf = rocke_b_const_f32(B, -INFINITY);
    ctx->zero_f = rocke_b_const_f32(B, 0.0);
    ctx->one_f = rocke_b_const_f32(B, 1.0);
    ctx->rcp_ln2 = rocke_b_const_f32(B, 1.4426950408889634);
    ctx->qk_scale = rocke_b_fmul(B, ctx->scale_p, ctx->rcp_ln2);
    ctx->sw_const = rocke_b_const_i32(B, CFG.SLIDING_WINDOW);
    ctx->z8 = rocke_b_zero_vec(B, dtype, 8);

    /* ---- Q -> LDS (lines 437-467) ---- */
    rocke_gfx942_attention_tiled_3d_emit_q_to_lds(ctx);

    /* ---- Per-segment tile range (lines 470-477) ---- */
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

    /* ---- lane decode (lines 482-483) ---- */
    ctx->lane_rg = rocke_b_div(B, ctx->tid, rocke_b_const_i32(B, 16));
    ctx->lane_col = rocke_b_mod(B, ctx->tid, rocke_b_const_i32(B, 16));

    /* NOTE: the async DMA infra (lines 538-567) and the paged-KV descriptor
     * (lines 569-602) are emitted AFTER acc_zero in Python's single linear
     * build; they are emitted by emit_loop_init (via
     * rocke_gfx942_attention_tiled_3d_emit_async_infra) so the SSA emission order
     * matches Python byte-for-byte. */
}

/* Async DMA infra + paged-KV descriptor (Python lines 538-602). Emitted from
 * emit_loop_init right after acc_zero so the op order matches the single
 * linear Python build. */
void rocke_gfx942_attention_tiled_3d_emit_async_infra(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    /* ---- async DMA infra (lines 538-567) ---- */
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
    ctx->lane_half_base = rocke_b_mul(B, ctx->tid, rocke_b_const_i32(B, CFG.HALVES_PER_LANE));
    ctx->K_lds_addr = rocke_b_smem_addr_of(B, ctx->K_lds);
    ctx->V_lds_addr = rocke_b_smem_addr_of(B, ctx->V_lds);
    ctx->zero_soff = rocke_b_const_i32(B, 0);

    /* ---- paged-KV descriptor (lines 569-602) ---- */
    rocke__build_paged_kv_desc(ctx);
}

/* ============================================================ *
 * rocke_gfx942_attention_tiled_3d_emit_early_zero_fill -- lines 376-419.
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_emit_early_zero_fill(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
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

        /* ml zero-fill (lines 380-395) */
        for(reg = 0; reg < 4; ++reg)
        {
            rocke_value_t* row
                = rocke_gfx942_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
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

        /* seg_acc zero-fill (lines 396-418) */
        {
            rocke_value_t* lane_col_e = rocke_b_mod(B, ctx->tid, rocke_b_const_i32(B, 16));
            for(n = 0; n < PV_N_TILES; ++n)
            {
                for(reg = 0; reg < 4; ++reg)
                {
                    rocke_value_t* row
                        = rocke_gfx942_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
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
 * rocke_gfx942_attention_tiled_3d_emit_q_to_lds -- lines 437-467.
 * ============================================================ */
void rocke_gfx942_attention_tiled_3d_emit_q_to_lds(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx)
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
 * Load issuers (Python closures, lines 604-719).
 * ============================================================ */

/* _issue_k_load(kv_tile_idx, buf_idx) -- lines 604-617. */
void rocke_gfx942_attention_tiled_3d_issue_k_load(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                                  rocke_value_t* kv_tile_idx,
                                                  rocke_value_t* buf_idx,
                                                  rocke_value_t* page_id)
{
    const int KV_HALVES_PER_CALL = CFG.KV_HALVES_PER_CALL;
    const int kv_calls_per_tile = CFG.kv_calls_per_tile;
    const int bytes_per_call = CFG.bytes_per_call;
    const int bytes_per_buf = CFG.bytes_per_buf;
    const int ASYNC_LDS_DWORDS = CFG.ASYNC_LDS_DWORDS;
    int call;
    /* The tile base goes into soffset; each copy adds its lane's offset within
     * the page. */
    rocke_value_t* soff
        = page_id != NULL ? rocke_gfx942_attention_tiled_3d_tile_base(
                                ctx, page_id, kv_tile_idx, CFG.kv_stride_blk_b, CFG.kv_stride_tok_b)
                          : ctx->zero_soff;

    rocke_value_t* buf_off_i32 = rocke_b_mul(B, buf_idx, rocke_b_const_i32(B, bytes_per_buf));
    rocke_value_t* buf_off_i64 = rocke_b_zext(B, buf_off_i32, rocke_i64());
    rocke_value_t* K_buf_base = rocke_b_smem_ptr_add(B, ctx->K_lds_addr, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* linear_half
            = rocke_b_add(B, rocke_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        rocke_value_t* voff;
        if(ctx->strided_kv)
        {
            voff = rocke_strided_kv_offset(
                B,
                &ctx->k_strides,
                kv_tile_idx,
                rocke_gfx942_attention_tiled_3d_kv_src_linear(ctx, linear_half),
                ctx->seq_len,
                CFG.HD,
                CFG.T);
        }
        else if(page_id != NULL)
        {
            rocke_value_t* src_half
                = rocke_gfx942_attention_tiled_3d_kv_src_linear(ctx, linear_half);
            rocke_value_t* tok = rocke_b_div(B, src_half, rocke_b_const_i32(B, CFG.HD));
            rocke_value_t* tok_off = rocke_b_mul(B, tok, rocke_b_const_i32(B, CFG.kv_stride_tok_b));
            rocke_value_t* head_off
                = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, CFG.kv_stride_h_b));
            rocke_value_t* tok_head = rocke_b_add(B, tok_off, head_off);
            rocke_value_t* dim = rocke_b_mod(B, src_half, rocke_b_const_i32(B, CFG.HD));
            rocke_value_t* dim_off = rocke_b_mul(B, dim, rocke_b_const_i32(B, CFG.KV_BYTES));
            voff = rocke_b_add(B, tok_head, dim_off);
        }
        else
        {
            voff = rocke__paged_kv_offset(
                ctx,
                kv_tile_idx,
                rocke_gfx942_attention_tiled_3d_kv_src_linear(ctx, linear_half),
                ctx->kv_head_idx);
        }
        rocke_value_t* k_dst = rocke_b_smem_ptr_add(
            B, K_buf_base, rocke_b_const_i64(B, (int64_t)call * bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            B, ctx->key_rsrc, k_dst, voff, soff, ASYNC_LDS_DWORDS, CFG.KV_CACHE_BITS);
    }
}

/* _issue_v_load(kv_tile_idx, buf_idx) -- lines 619-632. */
void rocke_gfx942_attention_tiled_3d_issue_v_load(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                                  rocke_value_t* kv_tile_idx,
                                                  rocke_value_t* buf_idx,
                                                  rocke_value_t* page_id)
{
    const int KV_HALVES_PER_CALL = CFG.KV_HALVES_PER_CALL;
    const int kv_calls_per_tile = CFG.kv_calls_per_tile;
    const int bytes_per_call = CFG.bytes_per_call;
    const int bytes_per_buf = CFG.bytes_per_buf;
    const int ASYNC_LDS_DWORDS = CFG.ASYNC_LDS_DWORDS;
    int call;
    /* The tile base goes into soffset; each copy adds its lane's offset within
     * the page. */
    rocke_value_t* soff
        = page_id != NULL ? rocke_gfx942_attention_tiled_3d_tile_base(
                                ctx, page_id, kv_tile_idx, CFG.kv_stride_blk_b, CFG.kv_stride_tok_b)
                          : ctx->zero_soff;

    rocke_value_t* buf_off_i32 = rocke_b_mul(B, buf_idx, rocke_b_const_i32(B, bytes_per_buf));
    rocke_value_t* buf_off_i64 = rocke_b_zext(B, buf_off_i32, rocke_i64());
    rocke_value_t* V_buf_base = rocke_b_smem_ptr_add(B, ctx->V_lds_addr, buf_off_i64);

    for(call = 0; call < kv_calls_per_tile; ++call)
    {
        rocke_value_t* linear_half
            = rocke_b_add(B, rocke_b_const_i32(B, call * KV_HALVES_PER_CALL), ctx->lane_half_base);
        rocke_value_t* voff;
        if(ctx->strided_kv)
        {
            voff = rocke_strided_kv_offset(
                B,
                &ctx->v_strides,
                kv_tile_idx,
                rocke_gfx942_attention_tiled_3d_kv_src_linear(ctx, linear_half),
                ctx->seq_len,
                CFG.HD,
                CFG.T);
        }
        else if(page_id != NULL)
        {
            rocke_value_t* src_half
                = rocke_gfx942_attention_tiled_3d_kv_src_linear(ctx, linear_half);
            rocke_value_t* tok = rocke_b_div(B, src_half, rocke_b_const_i32(B, CFG.HD));
            rocke_value_t* tok_off = rocke_b_mul(B, tok, rocke_b_const_i32(B, CFG.kv_stride_tok_b));
            rocke_value_t* head_off
                = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, CFG.kv_stride_h_b));
            rocke_value_t* tok_head = rocke_b_add(B, tok_off, head_off);
            rocke_value_t* dim = rocke_b_mod(B, src_half, rocke_b_const_i32(B, CFG.HD));
            rocke_value_t* dim_off = rocke_b_mul(B, dim, rocke_b_const_i32(B, CFG.KV_BYTES));
            voff = rocke_b_add(B, tok_head, dim_off);
        }
        else
        {
            voff = rocke__paged_kv_offset(
                ctx,
                kv_tile_idx,
                rocke_gfx942_attention_tiled_3d_kv_src_linear(ctx, linear_half),
                ctx->kv_head_idx);
        }
        rocke_value_t* v_dst = rocke_b_smem_ptr_add(
            B, V_buf_base, rocke_b_const_i64(B, (int64_t)call * bytes_per_call));
        rocke_b_async_buffer_load_lds_addr(
            B, ctx->value_rsrc, v_dst, voff, soff, ASYNC_LDS_DWORDS, CFG.KV_CACHE_BITS);
    }
}

/* _kv_col(row, col): LDS column of K/V element (row, col); its 16-byte chunk
 * is permuted by the row under the swizzle. */
rocke_value_t* rocke_gfx942_attention_tiled_3d_kv_col(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx, rocke_value_t* row, rocke_value_t* col)
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
    rocke_gfx942_attention_tiled_3d_kv_src_linear(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
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
    return rocke_b_add(B, row_base, rocke_gfx942_attention_tiled_3d_kv_col(ctx, row, col));
}

/* _tile_page(kv_tile_idx): the segment-local page index of a tile (tiles never
 * straddle pages). */
rocke_value_t*
    rocke_gfx942_attention_tiled_3d_tile_page(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx)
{
    if(CFG.TILES_PER_PAGE == 1)
    {
        return kv_tile_idx;
    }
    return rocke_b_div(B, kv_tile_idx, rocke_b_const_i32(B, CFG.TILES_PER_PAGE));
}

/* _tile_base(page_id, kv_tile_idx, page_stride, token_stride): offset of a
 * tile's first token: the page's, plus the tile's rows before it in the page
 * (sub-page tiles). */
rocke_value_t*
    rocke_gfx942_attention_tiled_3d_tile_base(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                              rocke_value_t* page_id,
                                              rocke_value_t* kv_tile_idx,
                                              int page_stride,
                                              int token_stride)
{
    rocke_value_t* base = rocke_b_mul(B, page_id, rocke_b_const_i32(B, page_stride));
    rocke_value_t* in_page;
    if(CFG.TILES_PER_PAGE == 1)
    {
        return base;
    }
    in_page = rocke_b_mod(B, kv_tile_idx, rocke_b_const_i32(B, CFG.TILES_PER_PAGE));
    return rocke_b_add(
        B, base, rocke_b_mul(B, in_page, rocke_b_const_i32(B, CFG.T * token_stride)));
}

/* _load_page_id_window(rel_page): lane l loads the id of the segment's page
 * rel_page + l; past the sequence's last page it gets 0, never used. */
rocke_value_t* rocke_gfx942_attention_tiled_3d_load_page_id_window(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx, rocke_value_t* rel_page)
{
    rocke_value_t* first = rocke_b_add(B, ctx->seg_first_page, rel_page);
    rocke_value_t* pg = rocke_b_add(B, first, ctx->tid);
    rocke_value_t* idx = rocke_b_add(B, ctx->seq_base, pg);
    rocke_value_t* live = rocke_b_cmp_lt(B, pg, ctx->seq_pages);
    rocke_value_t* other = rocke_b_const_i32(B, 0);
    return rocke_b_masked_global_load(B, ctx->block_tables, idx, live, other, rocke_i32(), 4);
}

/* _window_tile_page_id(page_id_window, kv_tile_idx): the tile's page id from
 * the 64-page window (v_readlane). */
rocke_value_t* rocke_gfx942_attention_tiled_3d_window_tile_page_id(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    rocke_value_t* page_id_window,
    rocke_value_t* kv_tile_idx)
{
    rocke_value_t* page = rocke_gfx942_attention_tiled_3d_tile_page(ctx, kv_tile_idx);
    rocke_value_t* rel = rocke_b_sub(B, page, ctx->seg_first_page);
    rocke_value_t* lane = rocke_b_mod(B, rel, rocke_b_const_i32(B, 64));
    return rocke_b_readlane(B, page_id_window, lane);
}

/* _refill_page_id_window(page_id_window, kv_tile_idx): the window for the next
 * iteration. If tile kv_tile_idx + 2 starts a new window, load it now; the
 * loop-top wait completes it before the next iteration reads it. A wave-uniform
 * 0/1-trip loop is an if with a result (scf_if has none): 0 trips keep the
 * window, 1 trip loads the next one. It compiles to a scalar branch around the
 * load. */
rocke_value_t* rocke_gfx942_attention_tiled_3d_refill_page_id_window(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    rocke_value_t* page_id_window,
    rocke_value_t* kv_tile_idx)
{
    rocke_value_t* t2 = rocke_b_add(B, kv_tile_idx, rocke_b_const_i32(B, 2));
    rocke_value_t* page2 = rocke_gfx942_attention_tiled_3d_tile_page(ctx, t2);
    rocke_value_t* rel2 = rocke_b_sub(B, page2, ctx->seg_first_page);
    rocke_value_t* rel2_lane = rocke_b_mod(B, rel2, rocke_b_const_i32(B, 64));
    rocke_value_t* at_start = rocke_b_cmp_eq(B, rel2_lane, rocke_b_const_i32(B, 0));
    rocke_value_t* t2_live = rocke_b_cmp_lt(B, t2, ctx->tile_end);
    rocke_value_t* cross = rocke_b_land(B, at_start, t2_live);
    if(CFG.TILES_PER_PAGE > 1)
    {
        /* Only the page's first tile starts the window. */
        rocke_value_t* t2_in_page = rocke_b_mod(B, t2, rocke_b_const_i32(B, CFG.TILES_PER_PAGE));
        rocke_value_t* first_in_page = rocke_b_cmp_eq(B, t2_in_page, rocke_b_const_i32(B, 0));
        cross = rocke_b_land(B, cross, first_in_page);
    }
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
    next_window = rocke_gfx942_attention_tiled_3d_load_page_id_window(ctx, rel2);
    rocke_b_scf_yield(B, &next_window, 1);
    rocke_b_region_leave(B);
    return refill.op->results[0];
}

/* _issue_wide_load(src, lds, kv_tile_idx, buf_idx, page_id) -- lines 645-658.
 * is_value: false => (key, K_lds), true => (value, V_lds). page_id: the tile's
 * page id (the copies add the chunk's offset within the page to its base), or
 * NULL (the paged-KV descriptor's per-copy table lookup). */
void rocke_gfx942_attention_tiled_3d_issue_wide_load(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    bool is_value,
    rocke_value_t* kv_tile_idx,
    rocke_value_t* buf_idx,
    rocke_value_t* page_id)
{
    const int HD = CFG.HD;
    const int THREADS = CFG.THREADS;
    const int WIDE_ELEMS = CFG.WIDE_ELEMS;
    const int KV_BYTES = CFG.KV_BYTES;
    const int wide_chunks_per_thread = CFG.wide_chunks_per_thread;
    const rocke_type_t* dtype = CFG.dtype;
    rocke_value_t* src = is_value ? ctx->value : ctx->key;
    rocke_value_t* lds = is_value ? ctx->V_lds : ctx->K_lds;
    int call;

    for(call = 0; call < wide_chunks_per_thread; ++call)
    {
        rocke_value_t* chunk_call = rocke_b_const_i32(B, call);
        rocke_value_t* chunk_thr = rocke_b_const_i32(B, THREADS);
        rocke_value_t* chunk_id = rocke_b_add(B, rocke_b_mul(B, chunk_call, chunk_thr), ctx->tid);
        rocke_value_t* linear_half = rocke_b_mul(B, chunk_id, rocke_b_const_i32(B, WIDE_ELEMS));
        rocke_value_t* row = rocke_b_div(B, linear_half, rocke_b_const_i32(B, HD));
        rocke_value_t* col = rocke_b_mod(B, linear_half, rocke_b_const_i32(B, HD));
        rocke_value_t* elem_off;
        rocke_mem_opts_t opts = ROCKE_MEM_OPTS_INIT;
        if(page_id != NULL)
        {
            rocke_value_t* page_base = rocke_gfx942_attention_tiled_3d_tile_base(
                ctx, page_id, kv_tile_idx, CFG.BS * CFG.NUM_KV * HD, CFG.NUM_KV * HD);
            rocke_value_t* row_off = rocke_b_mul(B, row, rocke_b_const_i32(B, CFG.NUM_KV * HD));
            rocke_value_t* head_off = rocke_b_mul(B, ctx->kv_head_idx, rocke_b_const_i32(B, HD));
            rocke_value_t* in_page = rocke_b_add(B, rocke_b_add(B, row_off, head_off), col);
            elem_off = rocke_b_add(B, page_base, in_page);
        }
        else
        {
            rocke_value_t* voff
                = rocke__paged_kv_offset(ctx, kv_tile_idx, linear_half, ctx->kv_head_idx);
            elem_off = rocke_b_div(B, voff, rocke_b_const_i32(B, KV_BYTES));
        }
        rocke_value_t* vec;
        if(CFG.KV_STREAM)
        {
            opts.temporal_hint = ROCKE_TEMPORAL_STREAMING;
        }
        vec = rocke_b_global_load_vN_ex(B, src, elem_off, dtype, WIDE_ELEMS, 16, &opts);
        rocke_value_t* idxs[3]
            = {buf_idx, row, rocke_gfx942_attention_tiled_3d_kv_col(ctx, row, col)};
        rocke_b_smem_store_vN(B, lds, idxs, 3, vec, WIDE_ELEMS);
    }
}

/* _issue_fp8_dequant_loads(kv_tile_idx, buf_idx, lds_token) -- lines 669-701.
 * is_value: false => "K" (key, k_scale, K_lds), true => "V".
 *
 * The dequant chain (dequant_fp8x8_to_dtype, helpers/attention.py:722-757) is
 * inlined here byte-for-byte: split <8 x fp8> into two <4 x fp8> quads ->
 * cvt_pk_f32_fp8x4 each -> fmul by scale -> cast_f32_to(dtype) -> vec_pack. */
void rocke_gfx942_attention_tiled_3d_issue_fp8_dequant_loads(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
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
    int call, i;

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
        rocke_value_t* voff
            = rocke__paged_kv_offset(ctx, kv_tile_idx, linear_half_first, ctx->kv_head_idx);
        rocke_value_t* fp8_vec = rocke_b_global_load_vN(
            B, src, voff, rocke_fp8e4m3(), fp8_elems_per_chunk, fp8_elems_per_chunk);

        /* dequant_fp8x8_to_dtype inline (attention.py:748-757) */
        rocke_value_t* lo_comps[4];
        rocke_value_t* hi_comps[4];
        rocke_value_t* lo_fp8;
        rocke_value_t* hi_fp8;
        rocke_value_t* lo_f32;
        rocke_value_t* hi_f32;
        rocke_value_t* deq[8];
        rocke_value_t* packed;
        rocke_value_t* idxs[3];

        /* Python: lo_fp8 = vec_pack([vec_extract(i) for i in range(4)]) then
         * hi_fp8 = vec_pack([vec_extract(i) for i in range(4,8)]). The extracts
         * for each quad are emitted immediately before that quad's pack -- NOT
         * all 8 up front -- so interleave extract+pack per quad. */
        for(i = 0; i < 4; ++i)
        {
            lo_comps[i] = rocke_b_vec_extract(B, fp8_vec, i);
        }
        lo_fp8 = rocke_b_vec_pack(B, lo_comps, 4, rocke_fp8e4m3());
        for(i = 0; i < 4; ++i)
        {
            hi_comps[i] = rocke_b_vec_extract(B, fp8_vec, i + 4);
        }
        hi_fp8 = rocke_b_vec_pack(B, hi_comps, 4, rocke_fp8e4m3());
        lo_f32 = rocke_b_cvt_pk_f32_fp8x4(B, lo_fp8);
        hi_f32 = rocke_b_cvt_pk_f32_fp8x4(B, hi_fp8);
        for(i = 0; i < 4; ++i)
        {
            deq[i] = rocke_b_cast_f32_to(
                B, rocke_b_fmul(B, rocke_b_vec_extract(B, lo_f32, i), scale), dtype);
        }
        for(i = 0; i < 4; ++i)
        {
            deq[i + 4] = rocke_b_cast_f32_to(
                B, rocke_b_fmul(B, rocke_b_vec_extract(B, hi_f32, i), scale), dtype);
        }
        packed = rocke_b_vec_pack(B, deq, 8, dtype);

        idxs[0] = buf_idx;
        idxs[1] = row;
        idxs[2] = col;
        rocke_b_smem_store_vN(B, lds, idxs, 3, packed, fp8_elems_per_chunk);
    }
}

/* _issue_k(tile_idx, buf_idx, page_id) -- lines 705-711. */
void rocke_gfx942_attention_tiled_3d_issue_k(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* buf_idx,
                                             rocke_value_t* page_id)
{
    if(CFG.KV_FP8)
    {
        rocke_gfx942_attention_tiled_3d_issue_fp8_dequant_loads(ctx, false, tile_idx, buf_idx);
    }
    else if(CFG.WIDE_KV)
    {
        rocke_gfx942_attention_tiled_3d_issue_wide_load(ctx, false, tile_idx, buf_idx, page_id);
    }
    else
    {
        rocke_gfx942_attention_tiled_3d_issue_k_load(ctx, tile_idx, buf_idx, page_id);
    }
}

/* _issue_v(tile_idx, buf_idx, page_id) -- lines 713-719. */
void rocke_gfx942_attention_tiled_3d_issue_v(rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                             rocke_value_t* tile_idx,
                                             rocke_value_t* buf_idx,
                                             rocke_value_t* page_id)
{
    if(CFG.KV_FP8)
    {
        rocke_gfx942_attention_tiled_3d_issue_fp8_dequant_loads(ctx, true, tile_idx, buf_idx);
    }
    else if(CFG.WIDE_KV)
    {
        rocke_gfx942_attention_tiled_3d_issue_wide_load(ctx, true, tile_idx, buf_idx, page_id);
    }
    else
    {
        rocke_gfx942_attention_tiled_3d_issue_v_load(ctx, tile_idx, buf_idx, page_id);
    }
}

/* ============================================================ *
 * _strided_v_b_operand(k_iter, cur_buf, v_n_col, v_k_chunk_base) -- lines 886-896.
 *
 * NOTE: the single external definition of
 *   rocke_gfx942_attention_tiled_3d_strided_v_b_operand
 * lives in the segment_compute.c part-file (declared in the internal header).
 * It was duplicated here previously, which caused a multiple-definition link
 * error; the copy has been removed so this translation unit relies on the
 * shared declaration instead.
 * ============================================================ */
