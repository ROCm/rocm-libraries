// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_build_load.c -- one part-file of the chunked C99 port of
 * build_universal_gemm (rocke/instances/common/gemm_universal.py).
 *
 * SCOPE (this TU only):
 *   * rocke_gemm_emit_load_phase    (Python closure emit_load_phase, ~1112)
 *   * rocke_gemm_emit_frag_smem_load(Python closure _emit_frag_smem_load, ~1372)
 *   * rocke_gemm_emit_wmma_phase    (Python closure _emit_wmma_phase, ~1409)
 *
 * Peers (the module-level helpers + the other phase closures) are CALLED via
 * rocke/instance_gemm_internal.h and are defined in sibling part-files; this TU
 * never re-defines them.
 *
 * Faithfulness: every builder call below is in the same order, with the same
 * operands/attrs, as the corresponding Python line. Where the Python relies on
 * host-side helper objects (TensorView / TileWindow), this TU uses the C ports
 * in rocke/helper_rocke.helpers.tensor_view.h, which themselves reproduce the
 * Python builder-call sequence.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.grid.h" /* rocke_chiplet_aware_super_tile_dynamic */
#include "rocke/instance_gemm_internal.h"

/* ===================================================================== *
 *  _swz_col -- the LDS XOR-swizzle column rewrite (Python inner closure).
 *
 *  Python span: gemm_universal.py _swz_col (lines ~895 .. ~898):
 *      def _swz_col(col, row):
 *          if not _SWZ:
 *              return col
 *          return b.xor(col, b.shl(b.mod(b.lshr(row, _c_swr), _c_swmod), _c_swl))
 *
 *  When swizzle is off (`ctx->swz` false) `col` is returned unchanged, exactly
 *  as the Python closure short-circuits. Applied identically to the LDS store
 *  column and the MFMA ds_read column so the physical address agrees on both
 *  sides (correctness preserved). The constants c_swr / c_swmod / c_swl are
 *  pre-materialised in the driver build prologue (rocke_build_universal_gemm
 *  in gemm_universal.cpp).
 * ===================================================================== */
rocke_value_t*
    rocke_gemm_swz_col(rocke_gemm_build_ctx_t* ctx, rocke_value_t* col, rocke_value_t* row)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->swz)
    {
        return col;
    }
    return rocke_b_xor(
        b,
        col,
        rocke_b_shl(b, rocke_b_mod(b, rocke_b_lshr(b, row, ctx->c_swr), ctx->c_swmod), ctx->c_swl));
}

/* _dtl_off_bytes(batch_off, blk_off, row, col): byte offset of one DTL chunk's
 * global source. A chunk smaller than an element (f64 on a 1-dword target)
 * addresses `col` in bytes; otherwise in elements. Operand order is pinned with
 * temporaries so the SSA order matches the Python closure. */
static rocke_value_t* rocke_gemm_dtl_off_bytes(rocke_gemm_build_ctx_t* ctx,
                                               rocke_value_t* c_eb,
                                               rocke_value_t* batch_off,
                                               rocke_value_t* blk_off,
                                               rocke_value_t* row,
                                               rocke_value_t* col,
                                               rocke_value_t* k_off,
                                               bool b_operand)
{
    rocke_ir_builder_t* b = ctx->b;
    /* the global operand row of the tile row (complex B: rocke_gemm_brow) */
    rocke_value_t* g_row = rocke_b_add(b, blk_off, row);
    if(b_operand)
    {
        g_row = rocke_gemm_brow(ctx, g_row);
    }
    rocke_value_t* row_term = rocke_b_mul(b, g_row, ctx->K);
    if(ctx->dtl_sub_elem)
    {
        rocke_value_t* off_elems = rocke_b_add(b, batch_off, rocke_b_add(b, row_term, k_off));
        rocke_value_t* col_b = col;
        if(ctx->swz)
        {
            rocke_value_t* mask = rocke_b_mod(b, rocke_b_lshr(b, row, ctx->c_swr), ctx->c_swmod);
            rocke_value_t* shift = rocke_b_const_i32(b, ctx->swz_l + ctx->dtl_elem_log2);
            col_b = rocke_b_xor(b, col, rocke_b_shl(b, mask, shift));
        }
        return rocke_b_add(b, rocke_b_mul(b, off_elems, c_eb), col_b);
    }
    rocke_value_t* k_term = rocke_b_add(b, k_off, rocke_gemm_swz_col(ctx, col, row));
    rocke_value_t* off_elems = rocke_b_add(b, batch_off, rocke_b_add(b, row_term, k_term));
    return rocke_b_mul(b, off_elems, c_eb);
}

/* ===================================================================== *
 *  emit_load_phase -- one K-tile's coalesced global->LDS copy.
 *
 *  Python span: gemm_universal.py emit_load_phase (lines ~1112 .. ~1370).
 *  lds_parity is int|Value: parity_v != NULL => runtime Value form; else
 *  parity_imm carries the compile-time 0/1 parity.
 * ===================================================================== */
void rocke_gemm_emit_load_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_dst,
                                rocke_value_t* B_dst,
                                rocke_value_t* k_off,
                                int parity_imm,
                                rocke_value_t* parity_v)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* I64 = rocke_i64();
    (void)A_dst; /* Python emit_load_phase's A_dst/B_dst args are unused in body. */
    (void)B_dst;

    /* ----------------------- DirectToLDS path ----------------------- */
    if(spec->trait.direct_to_lds)
    {
        bool parity_is_value = (parity_v != NULL);
        int a_half_bytes = ctx->block_m * ctx->block_k * ctx->dtl_elem_bytes;
        int b_half_bytes = ctx->block_n * ctx->block_k * ctx->dtl_elem_bytes;
        rocke_value_t* a_lds_par_base;
        rocke_value_t* b_lds_par_base;
        if(ctx->prefetch && parity_is_value)
        {
            rocke_value_t* a_par_v = rocke_b_zext(
                b, rocke_b_mul(b, parity_v, rocke_b_const_i32(b, a_half_bytes)), I64);
            rocke_value_t* b_par_v = rocke_b_zext(
                b, rocke_b_mul(b, parity_v, rocke_b_const_i32(b, b_half_bytes)), I64);
            a_lds_par_base = rocke_b_smem_ptr_add(b, ctx->dtl_a_lds_base, a_par_v);
            b_lds_par_base = rocke_b_smem_ptr_add(b, ctx->dtl_b_lds_base, b_par_v);
        }
        else
        {
            a_lds_par_base = ctx->dtl_a_lds_base;
            b_lds_par_base = ctx->dtl_b_lds_base;
        }
        int a_parity_bytes_static
            = (ctx->prefetch && !parity_is_value) ? parity_imm * a_half_bytes : 0;
        int b_parity_bytes_static
            = (ctx->prefetch && !parity_is_value) ? parity_imm * b_half_bytes : 0;
        rocke_value_t* c_eb = rocke_b_const_i32(b, ctx->dtl_elem_bytes);

        /* Per-wave LDS offset (multi-wave WG). */
        int wave_bytes = spec->wave_size * ctx->dtl_bytes_per_lane;
        rocke_value_t* a_lds_wave_base;
        rocke_value_t* b_lds_wave_base;
        if(t->warp_m * t->warp_n * t->warp_k > 1)
        {
            rocke_value_t* warp_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
            rocke_value_t* wave_par_off
                = rocke_b_zext(b, rocke_b_mul(b, warp_id, rocke_b_const_i32(b, wave_bytes)), I64);
            a_lds_wave_base = rocke_b_smem_ptr_add(b, a_lds_par_base, wave_par_off);
            b_lds_wave_base = rocke_b_smem_ptr_add(b, b_lds_par_base, wave_par_off);
        }
        else
        {
            a_lds_wave_base = a_lds_par_base;
            b_lds_wave_base = b_lds_par_base;
        }

        for(int p = 0; p < ctx->dtl_a_passes; ++p)
        {
            int pass_off_bytes = p * ctx->dtl_pass_bytes + a_parity_bytes_static;
            rocke_value_t* pass_lds_a
                = (pass_off_bytes > 0)
                      ? rocke_b_smem_ptr_add(
                            b,
                            a_lds_wave_base,
                            rocke_b_zext(b, rocke_b_const_i32(b, pass_off_bytes), I64))
                      : a_lds_wave_base;
            rocke_value_t* chunk_idx
                = rocke_b_add(b, ctx->tid, rocke_b_const_i32(b, p * spec->block_size));
            rocke_value_t* row = rocke_b_div(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col_v = rocke_b_mod(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col = rocke_b_mul(b, col_v, ctx->dtl_c_cols_per_chunk);
            rocke_value_t* off_bytes = rocke_gemm_dtl_off_bytes(
                ctx, c_eb, ctx->batch_off_a, ctx->block_m_off, row, col, k_off, false);
            rocke_b_async_buffer_load_lds_addr(b,
                                               ctx->dtl_a_rsrc,
                                               pass_lds_a,
                                               off_bytes,
                                               ctx->dtl_zero_soff,
                                               ctx->dtl_dwords,
                                               spec->trait.dtl_cache_a);
        }
        for(int p = 0; p < ctx->dtl_b_passes; ++p)
        {
            int pass_off_bytes = p * ctx->dtl_pass_bytes + b_parity_bytes_static;
            rocke_value_t* pass_lds_b
                = (pass_off_bytes > 0)
                      ? rocke_b_smem_ptr_add(
                            b,
                            b_lds_wave_base,
                            rocke_b_zext(b, rocke_b_const_i32(b, pass_off_bytes), I64))
                      : b_lds_wave_base;
            rocke_value_t* chunk_idx
                = rocke_b_add(b, ctx->tid, rocke_b_const_i32(b, p * spec->block_size));
            rocke_value_t* row = rocke_b_div(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col_v = rocke_b_mod(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col = rocke_b_mul(b, col_v, ctx->dtl_c_cols_per_chunk);
            rocke_value_t* off_bytes = rocke_gemm_dtl_off_bytes(
                ctx, c_eb, ctx->batch_off_b, ctx->block_n_off, row, col, k_off, true);
            rocke_b_async_buffer_load_lds_addr(b,
                                               ctx->dtl_b_rsrc,
                                               pass_lds_b,
                                               off_bytes,
                                               ctx->dtl_zero_soff,
                                               ctx->dtl_dwords,
                                               spec->trait.dtl_cache_b);
        }
        return;
    }

    /* ----------------------- canonical / preshuffle path ----------------------- */
    int load_vec = ctx->load_vec;
    bool db = ctx->db;

    /* a_global_tile / b_global_tile / a_lds_tile / b_lds_tile. */
    rocke_tile_window_t a_global_tile, b_global_tile, a_lds_tile, b_lds_tile;
    {
        int a_glen[3] = {1, ctx->block_m, ctx->block_k};
        int b_glen[3] = {1, ctx->block_n, ctx->block_k};
        rocke_value_t* a_gorigin[3] = {ctx->batch_off_a, ctx->block_m_off, k_off};
        rocke_value_t* b_gorigin[3] = {ctx->batch_off_b, ctx->block_n_off, k_off};
        rocke_make_tile_window(&a_global_tile, &ctx->a_view, a_glen, a_gorigin, 3);
        rocke_make_tile_window(&b_global_tile, &ctx->b_view, b_glen, b_gorigin, 3);

        int a_llen[2] = {ctx->block_m, ctx->block_k};
        int b_llen[2] = {ctx->block_n, ctx->block_k};
        rocke_value_t* zero_a = rocke_b_const_i32(b, 0);
        rocke_value_t* zero_b = rocke_b_const_i32(b, 0);
        rocke_value_t* a_lorigin[2] = {zero_a, zero_b};
        rocke_value_t* zero_c = rocke_b_const_i32(b, 0);
        rocke_value_t* zero_d = rocke_b_const_i32(b, 0);
        rocke_value_t* b_lorigin[2] = {zero_c, zero_d};
        rocke_make_tile_window(&a_lds_tile, &ctx->a_lds_view, a_llen, a_lorigin, 2);
        rocke_make_tile_window(&b_lds_tile, &ctx->b_lds_view, b_llen, b_lorigin, 2);
    }

    /* _ld_a_row / _ld_b_row : double-buffer parity row fold (compv4 non-DTL).
     * Implemented as inline lambdas via a helper macro-free local routine. */
#define LD_ROW(rvar, blk)                                                                        \
    (!db ? (rvar)                                                                                \
         : (parity_v != NULL                                                                     \
                ? rocke_b_add(b, (rvar), rocke_b_mul(b, parity_v, rocke_b_const_i32(b, (blk))))  \
                : (parity_imm ? rocke_b_add(b, (rvar), rocke_b_const_i32(b, parity_imm * (blk))) \
                              : (rvar))))

#define PADK_VALID(elem_col) rocke_b_cmp_lt(b, rocke_b_add(b, k_off, (elem_col)), ctx->K)
/* _safe_k0(): the column a masked lane reads instead (any in-bounds element):
 * the tile's first column, or with split_k_aligned -- whose empty slices sit
 * past K -- the row's first column (k_off + (-k_off)) */
#define SAFE_K0()                                                                 \
    (spec->trait.split_k_aligned ? rocke_b_sub(b, rocke_b_const_i32(b, 0), k_off) \
                                 : rocke_b_const_i32(b, 0))
#define STORAGE_ZERO()                              \
    (rocke_type_eq(ctx->storage_dtype, rocke_f64()) \
         ? rocke_b_const_f64(b, 0.0)                \
         : rocke_b_cast_f32_to(b, rocke_b_const_f32(b, 0.0), ctx->storage_dtype))
#define MASK_STORAGE(value, valid) rocke_b_select(b, (valid), (value), STORAGE_ZERO())

    /* A-load loop. */
    for(int e = 0; e < ctx->a_vecs_per_thread; ++e)
    {
        rocke_value_t* vec_idx
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
        /* _vec_rc(vec_idx). */
        rocke_value_t* row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
        rocke_value_t* col_v = rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec);
        rocke_value_t* col = (load_vec > 1) ? rocke_b_mul(b, col_v, ctx->c_load_vec) : col_v;
        if(spec->trait.pad_k)
        {
            rocke_value_t* comps[16];
            rocke_value_t* a_val;
            for(int i = 0; i < load_vec; ++i)
            {
                rocke_value_t* elem_col = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                rocke_value_t* valid = PADK_VALID(elem_col);
                rocke_value_t* safe_col = rocke_b_select(b, valid, elem_col, SAFE_K0());
                rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, safe_col};
                rocke_value_t* raw = rocke_tile_window_load_scalar(b, &a_global_tile, gidx, 3);
                comps[i] = MASK_STORAGE(raw, valid);
            }
            a_val = (load_vec == 1) ? comps[0]
                                    : rocke_b_vec_pack(b, comps, load_vec, ctx->storage_dtype);
            rocke_value_t* lidx[2] = {LD_ROW(row, ctx->block_m), rocke_gemm_swz_col(ctx, col, row)};
            if(load_vec == 1)
                rocke_tile_window_store_scalar(b, &a_lds_tile, lidx, 2, a_val, 0);
            else
                rocke_tile_window_store_vec(b, &a_lds_tile, lidx, 2, a_val, load_vec);
        }
        else if(load_vec == 1)
        {
            rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
            rocke_value_t* a_val = rocke_tile_window_load_scalar(b, &a_global_tile, gidx, 3);
            rocke_value_t* lidx[2] = {LD_ROW(row, ctx->block_m), rocke_gemm_swz_col(ctx, col, row)};
            rocke_tile_window_store_scalar(b, &a_lds_tile, lidx, 2, a_val, 0);
        }
        else
        {
            rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
            rocke_value_t* a_val = rocke_tile_window_load_vec(b, &a_global_tile, gidx, 3, load_vec);
            rocke_value_t* lidx[2] = {LD_ROW(row, ctx->block_m), rocke_gemm_swz_col(ctx, col, row)};
            rocke_tile_window_store_vec(b, &a_lds_tile, lidx, 2, a_val, load_vec);
        }
    }

    /* B-load branch. */
    if(spec->trait.preshuffle_b)
    {
        rocke_value_t* n_tile_idx = rocke_b_div(b, ctx->block_n_off, ctx->c_block_n);
        rocke_value_t* k_tile_idx = rocke_b_div(b, k_off, ctx->c_block_k);
        rocke_value_t* n_tile_count = rocke_b_div(b, ctx->N, ctx->c_block_n);
        /* Python evaluates rocke_b_mul args left-to-right: the inner add/mul are
         * built BEFORE the const. C evaluates call args right-to-left, so bind
         * each sub-expression to a temp in Python order to match SSA ids. */
        rocke_value_t* tile_off_inner
            = rocke_b_add(b, rocke_b_mul(b, k_tile_idx, n_tile_count), n_tile_idx);
        rocke_value_t* tile_off_const = rocke_b_const_i32(b, ctx->block_n * ctx->block_k);
        rocke_value_t* tile_offset_elements = rocke_b_mul(b, tile_off_inner, tile_off_const);
        rocke_value_t* base_off = rocke_b_add(b, ctx->batch_off_b, tile_offset_elements);
        for(int e = 0; e < ctx->b_vecs_per_thread; ++e)
        {
            rocke_value_t* vec_idx
                = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
            rocke_value_t* glob_off
                = rocke_b_add(b, base_off, rocke_b_mul(b, vec_idx, ctx->c_load_vec));
            rocke_value_t* row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col_v = rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col = (load_vec > 1) ? rocke_b_mul(b, col_v, ctx->c_load_vec) : col_v;
            if(spec->trait.pad_k)
            {
                rocke_value_t* comps[16];
                rocke_value_t* b_val;
                for(int i = 0; i < load_vec; ++i)
                {
                    rocke_value_t* elem_col
                        = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                    rocke_value_t* valid = PADK_VALID(elem_col);
                    rocke_value_t* raw_off = rocke_b_add(
                        b,
                        base_off,
                        rocke_b_add(
                            b, rocke_b_mul(b, vec_idx, ctx->c_load_vec), rocke_b_const_i32(b, i)));
                    rocke_value_t* safe_off = rocke_b_select(b, valid, raw_off, base_off);
                    rocke_value_t* raw
                        = rocke_b_global_load(b, ctx->Bp, safe_off, ctx->storage_dtype, 0);
                    comps[i] = MASK_STORAGE(raw, valid);
                }
                b_val = (load_vec == 1) ? comps[0]
                                        : rocke_b_vec_pack(b, comps, load_vec, ctx->storage_dtype);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                if(load_vec == 1)
                    rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
                else
                    rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
            else if(load_vec == 1)
            {
                rocke_value_t* b_val
                    = rocke_b_global_load(b, ctx->Bp, glob_off, ctx->storage_dtype, 0);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
            }
            else
            {
                rocke_value_t* b_val
                    = rocke_b_global_load_vN(b, ctx->Bp, glob_off, ctx->storage_dtype, load_vec, 0);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
        }
    }
    else
    {
        for(int e = 0; e < ctx->b_vecs_per_thread; ++e)
        {
            rocke_value_t* vec_idx
                = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
            rocke_value_t* row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col_v = rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col = (load_vec > 1) ? rocke_b_mul(b, col_v, ctx->c_load_vec) : col_v;
            if(spec->trait.pad_k)
            {
                rocke_value_t* comps[16];
                rocke_value_t* b_val;
                for(int i = 0; i < load_vec; ++i)
                {
                    rocke_value_t* elem_col
                        = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                    rocke_value_t* valid = PADK_VALID(elem_col);
                    rocke_value_t* safe_col = rocke_b_select(b, valid, elem_col, SAFE_K0());
                    rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, safe_col};
                    rocke_value_t* raw = rocke_tile_window_load_scalar(b, &b_global_tile, gidx, 3);
                    comps[i] = MASK_STORAGE(raw, valid);
                }
                b_val = (load_vec == 1) ? comps[0]
                                        : rocke_b_vec_pack(b, comps, load_vec, ctx->storage_dtype);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                if(load_vec == 1)
                    rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
                else
                    rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
            else if(load_vec == 1)
            {
                rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
                rocke_value_t* b_val = rocke_tile_window_load_scalar(b, &b_global_tile, gidx, 3);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
            }
            else
            {
                rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
                rocke_value_t* b_val
                    = rocke_tile_window_load_vec(b, &b_global_tile, gidx, 3, load_vec);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
        }
    }
#undef MASK_STORAGE
#undef STORAGE_ZERO
#undef SAFE_K0
#undef PADK_VALID
#undef LD_ROW
}

/* ===================================================================== *
 *  _emit_frag_smem_load -- one frag_len-wide operand fragment from a
 *  row-major LDS tile (8-wide chunked + vec_concat for wide WMMA frags).
 *
 *  Python span: gemm_universal.py _emit_frag_smem_load (lines ~1372 .. ~1407).
 * ===================================================================== */
rocke_value_t* rocke_gemm_emit_frag_smem_load(rocke_gemm_build_ctx_t* ctx,
                                              rocke_value_t* src,
                                              rocke_value_t* mn_in_atom,
                                              rocke_value_t* k_in_atom,
                                              rocke_value_t* atom_mn_base,
                                              rocke_value_t* k_tile_base,
                                              int frag_len)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* lds_row = rocke_b_add(b, atom_mn_base, mn_in_atom);
    rocke_value_t* lds_col = rocke_b_add(b, k_tile_base, k_in_atom);
    /* max_vec = 8 if storage_dtype in (F16, BF16) else frag_len */
    bool is_half = rocke_type_eq(ctx->storage_dtype, rocke_f16())
                   || rocke_type_eq(ctx->storage_dtype, rocke_bf16());
    int max_vec = is_half ? 8 : frag_len;
    if(frag_len <= max_vec)
    {
        return rocke_gemm_emit_smem_load(b, src, lds_row, lds_col, frag_len, ctx->storage_dtype);
    }
    rocke_value_t* frag = NULL;
    for(int off = 0; off < frag_len; off += max_vec)
    {
        rocke_value_t* chunk
            = rocke_gemm_emit_smem_load(b,
                                        src,
                                        lds_row,
                                        rocke_b_add(b, lds_col, rocke_b_const_i32(b, off)),
                                        max_vec,
                                        ctx->storage_dtype);
        frag = (frag == NULL) ? chunk : rocke_b_vec_concat(b, frag, chunk);
    }
    return frag;
}

/* ===================================================================== *
 *  _emit_wmma_phase -- one K-tile of WMMA atoms, fully contract-driven.
 *
 *  Python span: gemm_universal.py _emit_wmma_phase (lines ~1409 .. ~1464).
 *  Reads iter_vars (length ctx->num_accs), writes the new accs into out_accs.
 * ===================================================================== */
void rocke_gemm_emit_wmma_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    (void)num_iter_vars;

    const rocke_arch_layout_map_t* a_map = rocke_mmaop_a_layout(ctx->op, b);
    const rocke_arch_layout_map_t* b_map = rocke_mmaop_b_layout(ctx->op, b);

    /* a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
     * b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0) */
    rocke_value_t* a_row_in_atom = NULL;
    rocke_value_t* a_k_in_atom = NULL;
    rocke_value_t* b_k_in_atom = NULL;
    rocke_value_t* b_col_in_atom = NULL;
    rocke_arch_layout_map_coord(a_map, b, ctx->lane, 0, &a_row_in_atom, &a_k_in_atom);
    rocke_arch_layout_map_coord(b_map, b, ctx->lane, 0, &b_k_in_atom, &b_col_in_atom);

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, ctx->mfmas_m * t->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, ctx->mfmas_n * t->warp_tile_n));

    /* new_accs = list(iter_vars) */
    for(int i = 0; i < ctx->num_accs; ++i)
        out_accs[i] = iter_vars[i];

    /* Bounded scratch for per-K-step fragment vectors. */
    rocke_value_t* a_rows[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* b_cols[ROCKE_GEMM_MAX_ACCS];

    for(int kk = 0; kk < ctx->k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * t->warp_tile_k);
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * t->warp_tile_m));
            a_rows[mi] = rocke_gemm_emit_frag_smem_load(
                ctx, A_src, a_row_in_atom, a_k_in_atom, atom_row, k_tile_base, ctx->a_per_lane);
        }
        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * t->warp_tile_n));
            b_cols[ni] = rocke_gemm_emit_frag_smem_load(
                ctx, B_src, b_col_in_atom, b_k_in_atom, atom_row, k_tile_base, ctx->b_per_lane);
        }
        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat]
                    = rocke_gemm_emit_mma(b, ctx->op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        }
    }
}
