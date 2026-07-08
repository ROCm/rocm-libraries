// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_deep_fused_conv_pool_Scatter + maxpool finalization phases.c
 *
 * ONE chunked part-file of the C99 port of
 *   rocke/instances/gfx1151/deep_fused_conv_pool.py  (arch gfx1151).
 *
 * SCOPE (this TU):
 *   SCATTER (Python lines 2279-2452):
 *     _scatter_codes_to_lds            -> rocke_gfx1151_dfcp_scatter_codes_to_lds
 *     _scatter_codes_to_i8_lds         -> rocke_gfx1151_dfcp_scatter_codes_to_i8_lds
 *     _scatter_vec_codes_to_i8_lds     -> rocke_gfx1151_dfcp_scatter_vec_codes_to_i8_lds
 *     _scatter_packed_i4_codes_to_lds  -> rocke_gfx1151_dfcp_scatter_packed_i4_codes_to_lds
 *     _repack_c0_lds_to_packed         -> rocke_gfx1151_dfcp_repack_c0_lds_to_packed
 *   MAXPOOL FINALIZATION (Python lines 2455-2682):
 *     _emit_maxpool_finalquant         -> rocke_gfx1151_dfcp_emit_maxpool_finalquant
 *     _emit_maxpool_finalpack_i8       -> rocke_gfx1151_dfcp_emit_maxpool_finalpack_i8
 *
 * The code-fn selectors (rocke_gfx1151_dfcp_apply_code_fn / _apply_vec_code_fn) are
 * applied to each acc slot exactly where the Python passes its code_fn closure.
 *
 * The builder call sequence is byte-identical to the Python body. Peers
 * (ctx-init, quant primitives, the code-fn appliers) are reached only through the
 * private internal header; no header or peer file is edited here.
 */

#include <stddef.h>

#include "rocke/arch_target.h" /* full rocke_mma_op_t struct (c_layout / c_frag_len) */
#include "rocke/helper_rocke.helpers.epilogues.h" /* full rocke_warp_grid_t struct + property fns */
#include "rocke/instance_gfx1151_deep_fused_conv_pool.h"
#include "rocke/instance_gfx1151_deep_fused_conv_pool_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  SCATTER PHASES (accumulators -> LDS) -- Python lines 2279-2452.
 * ===================================================================== */

/* _scatter_codes_to_lds (Python 2279-2306): apply code_fn(acc_slot_f32) -> f16
 * code to each WMMA accumulator slot and store it at its (row, col) in the
 * row-major dst_smem tile. */
void rocke_gfx1151_dfcp_scatter_codes_to_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                             const rocke_mma_op_t* op,
                                             rocke_value_t* const* accs,
                                             size_t num_accs,
                                             rocke_value_t* dst_smem,
                                             int code_fn)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    const rocke_layout_map_t* c_map = rocke_mma_op_c_layout(op, b);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int frag_len = op->c_frag_len;
    int flat = 0;
    int mi, ni, i;
    (void)num_accs;
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[flat];
            rocke_value_t* m_base;
            rocke_value_t* n_base;
            ++flat;
            m_base = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * ROCKE_GFX1151_DFCP_WMMA));
            n_base = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * ROCKE_GFX1151_DFCP_WMMA));
            for(i = 0; i < frag_len; ++i)
            {
                rocke_value_t* row_off = NULL;
                rocke_value_t* col_off = NULL;
                rocke_value_t* row;
                rocke_value_t* col;
                rocke_value_t* code_h;
                rocke_value_t* idx[2];
                rocke_layout_map_coord(c_map, b, grid->lane, i, &row_off, &col_off);
                row = rocke_b_add(b, m_base, row_off);
                col = rocke_b_add(b, n_base, col_off);
                code_h = rocke_gfx1151_dfcp_apply_code_fn(
                    ctx, code_fn, rocke_b_vec_extract(b, acc, i));
                idx[0] = row;
                idx[1] = col;
                rocke_b_smem_store_f16(b, dst_smem, idx, 2, code_h);
            }
        }
    }
}

/* _scatter_codes_to_i8_lds (Python 2309-2336): apply code_fn(acc_slot) -> i8
 * int4 code and store byte-per-code LDS. */
void rocke_gfx1151_dfcp_scatter_codes_to_i8_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                const rocke_mma_op_t* op,
                                                rocke_value_t* const* accs,
                                                size_t num_accs,
                                                rocke_value_t* dst_smem,
                                                int code_fn)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    const rocke_layout_map_t* c_map = rocke_mma_op_c_layout(op, b);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int frag_len = op->c_frag_len;
    int flat = 0;
    int mi, ni, i;
    (void)num_accs;
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[flat];
            rocke_value_t* m_base;
            rocke_value_t* n_base;
            ++flat;
            m_base = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * ROCKE_GFX1151_DFCP_WMMA));
            n_base = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * ROCKE_GFX1151_DFCP_WMMA));
            for(i = 0; i < frag_len; ++i)
            {
                rocke_value_t* row_off = NULL;
                rocke_value_t* col_off = NULL;
                rocke_value_t* row;
                rocke_value_t* col;
                rocke_value_t* code;
                rocke_value_t* idx[2];
                rocke_layout_map_coord(c_map, b, grid->lane, i, &row_off, &col_off);
                row = rocke_b_add(b, m_base, row_off);
                col = rocke_b_add(b, n_base, col_off);
                code = rocke_gfx1151_dfcp_apply_code_fn(
                    ctx, code_fn, rocke_b_vec_extract(b, acc, i));
                idx[0] = row;
                idx[1] = col;
                rocke_b_smem_store_vN(b, dst_smem, idx, 2, code, 1);
            }
        }
    }
}

/* _scatter_vec_codes_to_i8_lds (Python 2339-2365): vector-quantize one
 * accumulator vector, then scatter i8 code lanes. */
void rocke_gfx1151_dfcp_scatter_vec_codes_to_i8_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                    const rocke_mma_op_t* op,
                                                    rocke_value_t* const* accs,
                                                    size_t num_accs,
                                                    rocke_value_t* dst_smem,
                                                    int vec_code_fn)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    const rocke_layout_map_t* c_map = rocke_mma_op_c_layout(op, b);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int frag_len = op->c_frag_len;
    int flat = 0;
    int mi, ni, i;
    (void)num_accs;
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[flat];
            rocke_value_t* codes;
            rocke_value_t* m_base;
            rocke_value_t* n_base;
            ++flat;
            codes = rocke_gfx1151_dfcp_apply_vec_code_fn(ctx, vec_code_fn, acc);
            m_base = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * ROCKE_GFX1151_DFCP_WMMA));
            n_base = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * ROCKE_GFX1151_DFCP_WMMA));
            for(i = 0; i < frag_len; ++i)
            {
                rocke_value_t* row_off = NULL;
                rocke_value_t* col_off = NULL;
                rocke_value_t* row;
                rocke_value_t* col;
                rocke_value_t* idx[2];
                rocke_layout_map_coord(c_map, b, grid->lane, i, &row_off, &col_off);
                row = rocke_b_add(b, m_base, row_off);
                col = rocke_b_add(b, n_base, col_off);
                idx[0] = row;
                idx[1] = col;
                rocke_b_smem_store_vN(b, dst_smem, idx, 2, rocke_b_vec_extract(b, codes, i), 1);
            }
        }
    }
}

/* _scatter_packed_i4_codes_to_lds (Python 2368-2412): pack adjacent conv0 C
 * columns into one byte and store from even lanes. WMMA C maps adjacent columns
 * to adjacent lanes, so the even lane uses ds_bpermute to read the odd lane's
 * code, packs high/low nibbles, and stores one byte. Odd lanes do not write. */
void rocke_gfx1151_dfcp_scatter_packed_i4_codes_to_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                       const rocke_mma_op_t* op,
                                                       rocke_value_t* const* accs,
                                                       size_t num_accs,
                                                       rocke_value_t* dst_smem,
                                                       int code_fn)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    const rocke_layout_map_t* c_map = rocke_mma_op_c_layout(op, b);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int frag_len = op->c_frag_len;
    int flat = 0;
    int mi, ni, i;
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c4 = rocke_b_const_i32(b, 4);
    rocke_value_t* c0xf = rocke_b_const_i32(b, 0xF);
    (void)num_accs;
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[flat];
            rocke_value_t* m_base;
            rocke_value_t* n_base;
            ++flat;
            m_base = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * ROCKE_GFX1151_DFCP_WMMA));
            n_base = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * ROCKE_GFX1151_DFCP_WMMA));
            for(i = 0; i < frag_len; ++i)
            {
                rocke_value_t* row_off = NULL;
                rocke_value_t* col_off = NULL;
                rocke_value_t* row;
                rocke_value_t* col;
                rocke_value_t* col_is_even;
                rocke_value_t* code;
                rocke_value_t* src_lane;
                rocke_value_t* odd_code;
                rocke_value_t* lo;
                rocke_value_t* hi;
                rocke_value_t* packed;
                rocke_value_t* idx[2];
                rocke_if_t iff;
                rocke_layout_map_coord(c_map, b, grid->lane, i, &row_off, &col_off);
                row = rocke_b_add(b, m_base, row_off);
                col = rocke_b_add(b, n_base, col_off);
                col_is_even = rocke_b_cmp_eq(b, rocke_b_land(b, col, c1), c0);
                code = rocke_b_sext(
                    b,
                    rocke_gfx1151_dfcp_apply_code_fn(ctx, code_fn, rocke_b_vec_extract(b, acc, i)),
                    rocke_i32());
                src_lane = rocke_b_add(b, grid->lane, c1);
                odd_code = rocke_b_ds_bpermute(
                    b, rocke_b_shl(b, src_lane, rocke_b_const_i32(b, 2)), code);
                lo = rocke_b_land(b, code, c0xf);
                hi = rocke_b_shl(b, rocke_b_land(b, odd_code, c0xf), c4);
                packed = rocke_b_trunc(b, rocke_b_lor(b, lo, hi), rocke_i8());
                iff = rocke_b_scf_if(b, col_is_even);
                rocke_b_region_enter(b, iff.then_region);
                idx[0] = row;
                idx[1] = rocke_b_div(b, col, rocke_b_const_i32(b, 2));
                rocke_b_smem_store_vN(b, dst_smem, idx, 2, packed, 1);
                rocke_b_region_leave(b);
            }
        }
    }
}

/* _repack_c0_lds_to_packed (Python 2415-2452): lane-local LDS->LDS repack of
 * byte-per-code C0 into packed bytes. Each thread reads two adjacent-K int4 codes
 * (a plain LDS read, lane-agnostic, NO ds_bpermute) and writes one packed byte
 * (low nibble = even K). */
void rocke_gfx1151_dfcp_repack_c0_lds_to_packed(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                rocke_value_t* c0_smem,
                                                rocke_value_t* c0_packed_smem)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;
    const rocke_warp_grid_t* grid = ctx->grid;
    int k0 = spec->problem.conv.K;
    int bpr = k0 / 2; /* packed bytes per row */
    int total = spec->tile_m * bpr;
    int bs = rocke_gfx1151_dfcp_block_size(spec);
    int ept = (total + bs - 1) / bs;
    rocke_value_t* c_bpr = rocke_b_const_i32(b, bpr);
    rocke_value_t* c_total = rocke_b_const_i32(b, total);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c0xf = rocke_b_const_i32(b, 0xF);
    rocke_value_t* c4 = rocke_b_const_i32(b, 4);
    int e;
    for(e = 0; e < ept; ++e)
    {
        rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), grid->tid);
        rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
        rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
        rocke_value_t* row = rocke_b_div(b, sidx, c_bpr);
        rocke_value_t* kb = rocke_b_mod(b, sidx, c_bpr);
        rocke_value_t* k_lo = rocke_b_mul(b, kb, rocke_b_const_i32(b, 2));
        rocke_value_t* load_idx[2];
        rocke_value_t* pair;
        rocke_value_t* lo;
        rocke_value_t* hi;
        rocke_value_t* packed;
        rocke_value_t* store_idx[2];
        rocke_if_t iff;
        load_idx[0] = row;
        load_idx[1] = k_lo;
        pair = rocke_b_smem_load_vN(b, c0_smem, load_idx, 2, rocke_i8(), 2); /* <2 x i8> */
        lo = rocke_b_land(b, rocke_b_sext(b, rocke_b_vec_extract(b, pair, 0), rocke_i32()), c0xf);
        hi = rocke_b_shl(
            b,
            rocke_b_land(b, rocke_b_sext(b, rocke_b_vec_extract(b, pair, 1), rocke_i32()), c0xf),
            c4);
        packed = rocke_b_trunc(b, rocke_b_lor(b, lo, hi), rocke_i8());
        iff = rocke_b_scf_if(b, in_range);
        rocke_b_region_enter(b, iff.then_region);
        store_idx[0] = row;
        store_idx[1] = kb;
        rocke_b_smem_store_vN(b, c0_packed_smem, store_idx, 2, packed, 1);
        rocke_b_region_leave(b);
    }
}

/* ===================================================================== *
 *  MAXPOOL FINALIZATION PHASES -- Python lines 2455-2682.
 * ===================================================================== */

/* _emit_maxpool_finalquant._emit_body (Python 2484-2543). One thread per pooled
 * pixel: 2x2 max over f16 codes, final int4 quant, pack channels into i32 words,
 * store to y_ptr. `lane_idx` is the (possibly masked) per-thread pixel index. */
static void rocke_gfx1151_dfcp_finalquant_body(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                               rocke_value_t* c1_smem,
                                               rocke_value_t* y_ptr,
                                               int out_k,
                                               int words,
                                               rocke_value_t* c_ptw,
                                               rocke_value_t* c_ctw,
                                               rocke_value_t* c_pool_wo,
                                               rocke_value_t* c_words,
                                               rocke_value_t* c_mf,
                                               rocke_value_t* c_0xf,
                                               rocke_value_t* neg_inf,
                                               rocke_value_t* block_ph,
                                               rocke_value_t* block_pw,
                                               rocke_value_t* lane_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;
    rocke_value_t* local_pho = rocke_b_div(b, lane_idx, c_ptw);
    rocke_value_t* local_pwo = rocke_b_mod(b, lane_idx, c_ptw);
    rocke_value_t* gpho = rocke_b_add(b, block_ph, local_pho);
    rocke_value_t* gpwo = rocke_b_add(b, block_pw, local_pwo);
    rocke_value_t* pix = rocke_b_add(b, rocke_b_mul(b, gpho, c_pool_wo), gpwo);
    rocke_value_t* corners[4];
    /* out_k == conv1_channels is bounded by tile_n (is_valid_spec gate); the
     * encoder_0 shapes keep it small. 512 is generous headroom and keeps the
     * per-channel scratch reentrant (one set of SSA pointers per pooled pixel,
     * mirroring the Python per-call lists). */
    rocke_value_t* chmax[512];
    rocke_value_t* word_vals[64];
    rocke_value_t* base;
    int cw = 8;
    int vec_pool;
    int corner_i = 0;
    int yy, xx, ch, w;

    /* 2x2 corner conv-tile rows for this pooled pixel (used by both paths).
     * Sequenced left-to-right so the const(2)/const(yy)/const(xx) SSA ids match
     * Python's expression evaluation (nested args leave the const order to C's
     * unspecified arg-eval order and offset every value below). */
    for(yy = 0; yy < 2; ++yy)
    {
        rocke_value_t* h0 = rocke_b_mul(b, local_pho, rocke_b_const_i32(b, 2));
        rocke_value_t* ch_h = rocke_b_add(b, h0, rocke_b_const_i32(b, yy));
        for(xx = 0; xx < 2; ++xx)
        {
            rocke_value_t* w0 = rocke_b_mul(b, local_pwo, rocke_b_const_i32(b, 2));
            rocke_value_t* ch_w = rocke_b_add(b, w0, rocke_b_const_i32(b, xx));
            rocke_value_t* cm0 = rocke_b_mul(b, ch_h, c_ctw);
            corners[corner_i++] = rocke_b_add(b, cm0, ch_w);
        }
    }

    /* Vectorized fast path requires the rounded-up channel span to fit inside the
     * tile_n columns so trailing lanes of the last chunk stay in-bounds. */
    vec_pool = spec->vectorize_maxpool && (spec->tile_n % cw == 0)
               && (((out_k + cw - 1) / cw) * cw <= spec->tile_n);

    for(ch = 0; ch < out_k; ++ch)
    {
        chmax[ch] = neg_inf;
    }

    if(vec_pool)
    {
        int n_chunks = (out_k + cw - 1) / cw;
        int ci, ck, j;
        for(ci = 0; ci < 4; ++ci)
        {
            rocke_value_t* conv_m = corners[ci];
            for(ck = 0; ck < n_chunks; ++ck)
            {
                rocke_value_t* idx[2];
                rocke_value_t* vecf;
                idx[0] = conv_m;
                idx[1] = rocke_b_const_i32(b, ck * cw);
                vecf = rocke_b_smem_load_vN_f16(b, c1_smem, idx, 2, cw);
                for(j = 0; j < cw; ++j)
                {
                    int chj = ck * cw + j;
                    rocke_value_t* vf;
                    if(chj >= out_k)
                    {
                        break;
                    }
                    vf = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, vecf, j));
                    chmax[chj] = rocke_b_fmax(b, chmax[chj], vf);
                }
            }
        }
    }
    else
    {
        int ci;
        for(ch = 0; ch < out_k; ++ch)
        {
            for(ci = 0; ci < 4; ++ci)
            {
                rocke_value_t* conv_m = corners[ci];
                rocke_value_t* idx[2];
                rocke_value_t* v;
                idx[0] = conv_m;
                idx[1] = rocke_b_const_i32(b, ch);
                v = rocke_b_vec_extract(b, rocke_b_smem_load_vN_f16(b, c1_smem, idx, 2, 1), 0);
                chmax[ch] = rocke_b_fmax(b, chmax[ch], rocke_b_cast_to_f32(b, v));
            }
        }
    }

    for(w = 0; w < words; ++w)
    {
        word_vals[w] = rocke_b_const_i32(b, 0);
    }
    for(ch = 0; ch < out_k; ++ch)
    {
        rocke_value_t* qf = rocke_gfx1151_dfcp_quant_i4(b, chmax[ch], c_mf); /* i8 int4 code */
        rocke_value_t* nib = rocke_b_land(b, rocke_b_sext(b, qf, rocke_i32()), c_0xf);
        int wi = ch / 8;
        int shift = 4 * (ch % 8);
        if(shift)
        {
            nib = rocke_b_shl(b, nib, rocke_b_const_i32(b, shift));
        }
        word_vals[wi] = rocke_b_lor(b, word_vals[wi], nib);
    }

    base = rocke_b_mul(b, pix, c_words);
    for(w = 0; w < words; ++w)
    {
        rocke_b_global_store(
            b, y_ptr, rocke_b_add(b, base, rocke_b_const_i32(b, w)), word_vals[w], 4);
    }
}

/* _emit_maxpool_finalquant (Python 2455-2556): fp16 path. */
void rocke_gfx1151_dfcp_emit_maxpool_finalquant(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                rocke_value_t* c1_smem,
                                                rocke_value_t* y_ptr)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;
    const rocke_fused_conv_pool_problem_t* p = ctx->p;
    int out_k = rocke_fused_conv_pool_problem_conv1_channels(p);
    int conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    int n_pix = spec->pool_tile_h * spec->pool_tile_w;
    int words = (out_k + 7) / 8;

    rocke_value_t* c_ptw = rocke_b_const_i32(b, spec->pool_tile_w);
    rocke_value_t* c_ctw = rocke_b_const_i32(b, conv_tile_w);
    rocke_value_t* c_pool_wo = rocke_b_const_i32(b, rocke_fused_conv_pool_problem_pool_wo(p));
    rocke_value_t* c_words = rocke_b_const_i32(b, words);
    rocke_value_t* c_mf = rocke_b_const_f32(b, spec->mf);
    rocke_value_t* c_0xf = rocke_b_const_i32(b, 0xF);
    rocke_value_t* neg_inf = rocke_b_const_f32(b, -3.4028234663852886e38);
    rocke_value_t* h_blk = spec->w_fast ? rocke_b_block_id_z(b) : rocke_b_block_id_y(b);
    rocke_value_t* w_blk = spec->w_fast ? rocke_b_block_id_y(b) : rocke_b_block_id_z(b);
    rocke_value_t* block_ph = rocke_b_mul(b, h_blk, rocke_b_const_i32(b, spec->pool_tile_h));
    rocke_value_t* block_pw = rocke_b_mul(b, w_blk, rocke_b_const_i32(b, spec->pool_tile_w));
    rocke_value_t* in_range = rocke_b_cmp_lt(b, ctx->grid->tid, rocke_b_const_i32(b, n_pix));

    if(spec->mask_maxpool)
    {
        /* L3: branch-free tail; clamp out-of-range lanes to the last pooled pixel
         * (idempotent re-store), trading the structured branch for redundant
         * compute. */
        rocke_value_t* sidx
            = rocke_b_select(b, in_range, ctx->grid->tid, rocke_b_const_i32(b, n_pix - 1));
        rocke_gfx1151_dfcp_finalquant_body(ctx,
                                           c1_smem,
                                           y_ptr,
                                           out_k,
                                           words,
                                           c_ptw,
                                           c_ctw,
                                           c_pool_wo,
                                           c_words,
                                           c_mf,
                                           c_0xf,
                                           neg_inf,
                                           block_ph,
                                           block_pw,
                                           sidx);
    }
    else
    {
        rocke_if_t iff = rocke_b_scf_if(b, in_range);
        rocke_b_region_enter(b, iff.then_region);
        rocke_gfx1151_dfcp_finalquant_body(ctx,
                                           c1_smem,
                                           y_ptr,
                                           out_k,
                                           words,
                                           c_ptw,
                                           c_ctw,
                                           c_pool_wo,
                                           c_words,
                                           c_mf,
                                           c_0xf,
                                           neg_inf,
                                           block_ph,
                                           block_pw,
                                           ctx->grid->tid);
        rocke_b_region_leave(b);
    }
}

/* _emit_maxpool_finalpack_i8._emit_body (Python 2591-2674). Native integer
 * maxpool over byte int4 codes; final mf=1 pack is a no-op. */
static void rocke_gfx1151_dfcp_finalpack_i8_body(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                 rocke_value_t* c1_smem,
                                                 rocke_value_t* y_ptr,
                                                 int out_k,
                                                 int words,
                                                 rocke_value_t* c_ptw,
                                                 rocke_value_t* c_ctw,
                                                 rocke_value_t* c_pool_wo,
                                                 rocke_value_t* c_words,
                                                 rocke_value_t* c_0xf,
                                                 rocke_value_t* block_ph,
                                                 rocke_value_t* block_pw,
                                                 rocke_value_t* lane_idx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;
    rocke_value_t* local_pho = rocke_b_div(b, lane_idx, c_ptw);
    rocke_value_t* local_pwo = rocke_b_mod(b, lane_idx, c_ptw);
    rocke_value_t* gpho = rocke_b_add(b, block_ph, local_pho);
    rocke_value_t* gpwo = rocke_b_add(b, block_pw, local_pwo);
    rocke_value_t* pix = rocke_b_add(b, rocke_b_mul(b, gpho, c_pool_wo), gpwo);
    rocke_value_t* corners[4];
    rocke_value_t* chmax[512];
    rocke_value_t* word_vals[64];
    rocke_value_t* base;
    int cw = 8;
    int vec_pool;
    int corner_i = 0;
    int yy, xx, ch, w;

    /* Sequenced left-to-right to match Python's const(2)/const(yy)/const(xx)
     * SSA order (see the finalquant twin above). */
    for(yy = 0; yy < 2; ++yy)
    {
        rocke_value_t* h0 = rocke_b_mul(b, local_pho, rocke_b_const_i32(b, 2));
        rocke_value_t* ch_h = rocke_b_add(b, h0, rocke_b_const_i32(b, yy));
        for(xx = 0; xx < 2; ++xx)
        {
            rocke_value_t* w0 = rocke_b_mul(b, local_pwo, rocke_b_const_i32(b, 2));
            rocke_value_t* ch_w = rocke_b_add(b, w0, rocke_b_const_i32(b, xx));
            rocke_value_t* cm0 = rocke_b_mul(b, ch_h, c_ctw);
            corners[corner_i++] = rocke_b_add(b, cm0, ch_w);
        }
    }

    for(ch = 0; ch < out_k; ++ch)
    {
        chmax[ch] = rocke_b_const_i32(b, -8);
    }

    vec_pool = spec->vectorize_maxpool && (spec->tile_n % cw == 0)
               && (((out_k + cw - 1) / cw) * cw <= spec->tile_n);

    if(vec_pool && spec->pk_maxpool)
    {
        /* Packed-int16 reduction: widen each corner chunk i8 -> <cw x i16> and max
         * across the 4 corners with vector_smax so the gfx11 backend selects
         * v_pk_max_i16 (2 channels/op). Initialising the accumulator from the
         * first corner keeps it bit-exact to the scalar path. */
        int n_chunks = (out_k + cw - 1) / cw;
        int ck, ci, j;
        for(ck = 0; ck < n_chunks; ++ck)
        {
            rocke_value_t* acc16 = NULL;
            for(ci = 0; ci < 4; ++ci)
            {
                rocke_value_t* conv_m = corners[ci];
                rocke_value_t* idx[2];
                rocke_value_t* raw;
                rocke_value_t* w16;
                idx[0] = conv_m;
                idx[1] = rocke_b_const_i32(b, ck * cw);
                raw = rocke_b_smem_load_vN(b, c1_smem, idx, 2, rocke_i8(), cw);
                w16 = rocke_b_vector_sext(b, raw, rocke_i16()); /* signed widen i8 -> i16 */
                acc16 = (acc16 == NULL) ? w16 : rocke_b_vector_smax(b, acc16, w16);
            }
            for(j = 0; j < cw; ++j)
            {
                int chj = ck * cw + j;
                if(chj >= out_k)
                {
                    break;
                }
                chmax[chj] = rocke_b_sext(b, rocke_b_vec_extract(b, acc16, j), rocke_i32());
            }
        }
    }
    else if(vec_pool)
    {
        int n_chunks = (out_k + cw - 1) / cw;
        int ci, ck, j;
        for(ci = 0; ci < 4; ++ci)
        {
            rocke_value_t* conv_m = corners[ci];
            for(ck = 0; ck < n_chunks; ++ck)
            {
                rocke_value_t* idx[2];
                rocke_value_t* vec;
                idx[0] = conv_m;
                idx[1] = rocke_b_const_i32(b, ck * cw);
                vec = rocke_b_smem_load_vN(b, c1_smem, idx, 2, rocke_i8(), cw);
                for(j = 0; j < cw; ++j)
                {
                    int chj = ck * cw + j;
                    rocke_value_t* v;
                    if(chj >= out_k)
                    {
                        break;
                    }
                    v = rocke_b_sext(b, rocke_b_vec_extract(b, vec, j), rocke_i32());
                    chmax[chj] = rocke_b_select(b, rocke_b_cmp_gt(b, v, chmax[chj]), v, chmax[chj]);
                }
            }
        }
    }
    else
    {
        int ci;
        for(ch = 0; ch < out_k; ++ch)
        {
            for(ci = 0; ci < 4; ++ci)
            {
                rocke_value_t* conv_m = corners[ci];
                rocke_value_t* idx[2];
                rocke_value_t* v;
                idx[0] = conv_m;
                idx[1] = rocke_b_const_i32(b, ch);
                v = rocke_b_sext(b,
                                 rocke_b_vec_extract(
                                     b, rocke_b_smem_load_vN(b, c1_smem, idx, 2, rocke_i8(), 1), 0),
                                 rocke_i32());
                chmax[ch] = rocke_b_select(b, rocke_b_cmp_gt(b, v, chmax[ch]), v, chmax[ch]);
            }
        }
    }

    for(w = 0; w < words; ++w)
    {
        word_vals[w] = rocke_b_const_i32(b, 0);
    }
    for(ch = 0; ch < out_k; ++ch)
    {
        /* mf is 1.0; values are already ReLUed int4 codes, so final
         * QuantizeLinear is just nibble packing. */
        rocke_value_t* nib = rocke_b_land(b, chmax[ch], c_0xf);
        int wi = ch / 8;
        int shift = 4 * (ch % 8);
        if(shift)
        {
            nib = rocke_b_shl(b, nib, rocke_b_const_i32(b, shift));
        }
        word_vals[wi] = rocke_b_lor(b, word_vals[wi], nib);
    }

    base = rocke_b_mul(b, pix, c_words);
    for(w = 0; w < words; ++w)
    {
        rocke_b_global_store(
            b, y_ptr, rocke_b_add(b, base, rocke_b_const_i32(b, w)), word_vals[w], 4);
    }
}

/* _emit_maxpool_finalpack_i8 (Python 2559-2682): native path; persistent coords
 * via h_blk/w_blk (NULL => the per-CTA block ids, byte-identical). */
void rocke_gfx1151_dfcp_emit_maxpool_finalpack_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                  rocke_value_t* c1_smem,
                                                  rocke_value_t* y_ptr,
                                                  rocke_value_t* h_blk,
                                                  rocke_value_t* w_blk)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;
    const rocke_fused_conv_pool_problem_t* p = ctx->p;
    int out_k = rocke_fused_conv_pool_problem_conv1_channels(p);
    int conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    int n_pix = spec->pool_tile_h * spec->pool_tile_w;
    int words = (out_k + 7) / 8;

    rocke_value_t* c_ptw = rocke_b_const_i32(b, spec->pool_tile_w);
    rocke_value_t* c_ctw = rocke_b_const_i32(b, conv_tile_w);
    rocke_value_t* c_pool_wo = rocke_b_const_i32(b, rocke_fused_conv_pool_problem_pool_wo(p));
    rocke_value_t* c_words = rocke_b_const_i32(b, words);
    rocke_value_t* c_0xf = rocke_b_const_i32(b, 0xF);
    rocke_value_t* block_ph;
    rocke_value_t* block_pw;
    rocke_value_t* in_range;

    if(h_blk == NULL)
    {
        h_blk = spec->w_fast ? rocke_b_block_id_z(b) : rocke_b_block_id_y(b);
        w_blk = spec->w_fast ? rocke_b_block_id_y(b) : rocke_b_block_id_z(b);
    }
    block_ph = rocke_b_mul(b, h_blk, rocke_b_const_i32(b, spec->pool_tile_h));
    block_pw = rocke_b_mul(b, w_blk, rocke_b_const_i32(b, spec->pool_tile_w));
    in_range = rocke_b_cmp_lt(b, ctx->grid->tid, rocke_b_const_i32(b, n_pix));

    if(spec->mask_maxpool)
    {
        rocke_value_t* sidx
            = rocke_b_select(b, in_range, ctx->grid->tid, rocke_b_const_i32(b, n_pix - 1));
        rocke_gfx1151_dfcp_finalpack_i8_body(ctx,
                                             c1_smem,
                                             y_ptr,
                                             out_k,
                                             words,
                                             c_ptw,
                                             c_ctw,
                                             c_pool_wo,
                                             c_words,
                                             c_0xf,
                                             block_ph,
                                             block_pw,
                                             sidx);
    }
    else
    {
        rocke_if_t iff = rocke_b_scf_if(b, in_range);
        rocke_b_region_enter(b, iff.then_region);
        rocke_gfx1151_dfcp_finalpack_i8_body(ctx,
                                             c1_smem,
                                             y_ptr,
                                             out_k,
                                             words,
                                             c_ptw,
                                             c_ctw,
                                             c_pool_wo,
                                             c_words,
                                             c_0xf,
                                             block_ph,
                                             block_pw,
                                             ctx->grid->tid);
        rocke_b_region_leave(b);
    }
}
