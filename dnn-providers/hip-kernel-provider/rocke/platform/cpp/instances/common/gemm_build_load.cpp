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
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.loads.h"
#include "rocke/helper_rocke.helpers.preshuffle.h"
#include "rocke/helper_rocke.helpers.tiling.mma.h"
#include "rocke/instance_gemm_internal.h"

/* storage_dtype in (FP8E4M3, BF8E5M2) -- the 8-bit A/B operands of the gfx1250
 * low-bit WMMA path, which the f16/bf16 storage paths never produce. */
static bool rocke_gemm_storage_is_fp8(const rocke_type_t* ty)
{
    return rocke_type_eq(ty, rocke_fp8e4m3()) || rocke_type_eq(ty, rocke_bf8e5m2());
}

/* ===================================================================== *
 *  _swz_col -- the LDS XOR-swizzle column rewrite (Python inner closure).
 *
 *  Python span: gemm_universal.py _swz_col (lines ~895 .. ~898):
 *      def _swz_col(col, row):
 *          if not _SWZ:
 *              return col
 *          return emit_lds_col_xor(b, row, col, row_shift=_c_swr,
 *                                  chunk_mask=_c_swmask, chunk_log2=_c_swl)
 *
 *  When swizzle is off (`ctx->swz` false) `col` is returned unchanged, exactly
 *  as the Python closure short-circuits. Applied identically to the LDS store
 *  column and the MFMA ds_read column so the physical address agrees on both
 *  sides (correctness preserved). The constants c_swr / c_swmask / c_swl are
 *  pre-materialised in the driver build prologue (rocke_build_universal_gemm
 *  in gemm_universal.cpp).
 * ===================================================================== */
rocke_value_t*
    rocke_gemm_swz_col(rocke_gemm_build_ctx_t* ctx, rocke_value_t* col, rocke_value_t* row)
{
    rocke_ir_builder_t* b = ctx->b;
    if(rocke_lds_column_xor_is_identity(&ctx->lds_xor))
        return col;
    return rocke_emit_lds_col_xor(b, row, col, ctx->c_swr, ctx->c_swmask, ctx->c_swl);
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
                                               rocke_value_t* k_off)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* row_term = rocke_b_mul(b, rocke_b_add(b, blk_off, row), ctx->K);
    if(ctx->dtl_sub_elem)
    {
        rocke_value_t* off_elems = rocke_b_add(b, batch_off, rocke_b_add(b, row_term, k_off));
        rocke_value_t* col_b = col;
        if(!rocke_lds_column_xor_is_identity(&ctx->lds_xor))
        {
            rocke_value_t* shift = rocke_b_const_i32(b, ctx->lds_xor.chunk_log2_bytes);
            col_b = rocke_emit_lds_col_xor(b, row, col, ctx->c_swr, ctx->c_swmask, shift);
        }
        return rocke_b_add(b, rocke_b_mul(b, off_elems, c_eb), col_b);
    }
    rocke_value_t* k_term = rocke_b_add(b, k_off, rocke_gemm_swz_col(ctx, col, row));
    rocke_value_t* off_elems = rocke_b_add(b, batch_off, rocke_b_add(b, row_term, k_term));
    return rocke_b_mul(b, off_elems, c_eb);
}

struct rocke_gemm_dtl_source
{
    rocke_gemm_build_ctx_t* ctx;
    rocke_value_t* elem_bytes;
    rocke_value_t* batch_off;
    rocke_value_t* block_off;
    rocke_value_t* k_off;
};

static rocke_value_t* rocke_gemm_dtl_descriptor(rocke_ir_builder_t*,
                                                rocke_value_t* row,
                                                rocke_value_t* col,
                                                void* user)
{
    const auto& src = *static_cast<rocke_gemm_dtl_source*>(user);
    return rocke_gemm_dtl_off_bytes(
        src.ctx, src.elem_bytes, src.batch_off, src.block_off, row, col, src.k_off);
}

static void rocke_gemm_issue_dtl_operand(rocke_gemm_dtl_source* src,
                                         rocke_value_t* dest,
                                         rocke_value_t* wave_base,
                                         rocke_value_t* rsrc,
                                         int passes,
                                         int slot_bytes,
                                         int coherency)
{
    const auto* ctx = src->ctx;
    rocke_async_tile_loader_slot_t slot = {};
    slot.loader.block_size = ctx->spec->block_size;
    slot.loader.dwords = ctx->dtl_dwords;
    slot.loader.passes = passes;
    slot.smem_dst = dest;
    slot.per_wave_lds_base = wave_base;
    rocke_async_tile_loader_slot_issue_bytes(ctx->b,
                                             &slot,
                                             ctx->tid,
                                             rsrc,
                                             rocke_gemm_dtl_descriptor,
                                             src,
                                             ctx->dtl_c_chunks_per_row,
                                             ctx->dtl_c_cols_per_chunk,
                                             ctx->dtl_zero_soff,
                                             slot_bytes,
                                             coherency);
}

/* ===================================================================== *
 *  emit_load_phase -- one K-tile's coalesced global->LDS copy.
 *
 *  Python span: gemm_universal.py emit_load_phase (lines ~1112 .. ~1370).
 *  lds_parity is int|Value: parity_v != NULL => runtime Value form; else
 *  parity_imm carries the compile-time 0/1 parity.
 * ===================================================================== */
/* _issue: one TDM descriptor moving a [rows, block_k] tile of `ptr` whose
 * first row is `row_off` into LDS at `lds_base`. Mirrors Python _issue
 * closure inside emit_load_phase. */
static void rocke_gemm_tdm_issue(rocke_gemm_build_ctx_t* ctx,
                                 rocke_value_t* ptr,
                                 rocke_value_t* row_off,
                                 int rows,
                                 rocke_value_t* extent,
                                 rocke_value_t* lds_base,
                                 rocke_value_t* batch_off,
                                 rocke_value_t* k_off,
                                 rocke_value_t* k_remaining,
                                 rocke_value_t* zero,
                                 rocke_gemm_emit_hook_t before_issue)
{
    const rocke_tdm_source_window_2d_t source = {
        .ptr = ptr,
        .base_offset = batch_off,
        .row_origin = row_off,
        .col_origin = k_off,
        .row_extent = extent,
        .cols_remaining = k_remaining,
        .row_stride = ctx->K,
    };
    const rocke_tdm_tile_loader_2d_t loader = {
        .tile_rows = rows,
        .tile_cols = ctx->block_k,
        .elem_bytes = ctx->tdm_elem_bytes,
        .pad_interval = ctx->tdm_pad_enable ? ctx->tdm_pad_interval : -1,
        .pad_amount = ctx->tdm_pad_amount,
    };
    rocke_tdm_tile_load_t transfer;
    if(!rocke_tdm_tile_loader_2d_prepare(ctx->b, &loader, &source, lds_base, zero, &transfer))
        return;
    if(before_issue.fn != NULL)
        before_issue.fn(ctx, before_issue.user);
    rocke_tdm_tile_load_issue(ctx->b, &transfer, /*cachepolicy=*/0);
}

struct rocke_gemm_sync_operand
{
    rocke_gemm_build_ctx_t* ctx;
    const rocke_tile_window_t* source;
    const rocke_tile_window_t* dest;
    rocke_value_t* k_off;
    int rows;
    int parity_imm;
    rocke_value_t* parity_v;
    rocke_value_t* packed_offset = nullptr;
};

static void
    rocke_gemm_prepare_packed_offset(rocke_ir_builder_t* b, rocke_value_t* chunk_id, void* user)
{
    auto& copy = *static_cast<rocke_gemm_sync_operand*>(user);
    auto* chunk_start = rocke_b_mul(b, chunk_id, copy.ctx->c_load_vec);
    copy.packed_offset = rocke_b_add(b, copy.source->origin[0], chunk_start);
}

struct rocke_gemm_clipped_operand
{
    const rocke_gemm_sync_operand& copy;
    rocke_value_t* row;
    rocke_value_t* chunk_id;
};

static rocke_value_t* rocke_gemm_read_clipped_scalar(
    rocke_ir_builder_t* b, rocke_value_t* local, rocke_value_t* valid, int i, const void* user)
{
    const auto& chunk = *static_cast<const rocke_gemm_clipped_operand*>(user);
    const auto* source = chunk.copy.source;
    const auto* ctx = chunk.copy.ctx;
    if(source->rank == 2)
    {
        rocke_value_t* base = source->origin[0];
        rocke_value_t* chunk_start = rocke_b_mul(b, chunk.chunk_id, ctx->c_load_vec);
        rocke_value_t* element = rocke_b_add(b, chunk_start, rocke_b_const_i32(b, i));
        rocke_value_t* offset = rocke_b_add(b, base, element);
        rocke_value_t* safe = rocke_b_select(b, valid, offset, base);
        return rocke_b_global_load(b, source->view->base, safe, ctx->storage_dtype, 0);
    }
    rocke_value_t* safe = rocke_b_select(b, valid, local, rocke_b_const_i32(b, 0));
    rocke_value_t* indices[3] = {rocke_b_const_i32(b, 0), chunk.row, safe};
    return rocke_tile_window_load_scalar(b, source, indices, 3);
}

static void rocke_gemm_copy_sync_chunk(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t* chunk_id,
                                       void* user)
{
    const auto& copy = *static_cast<rocke_gemm_sync_operand*>(user);
    auto* ctx = copy.ctx;
    int width = ctx->load_vec;
    rocke_value_t* value;
    if(ctx->spec->trait.pad_k)
    {
        const rocke_gemm_clipped_operand clipped{copy, row, chunk_id};
        value = rocke_b_load_clipped_global_fragment(b,
                                                     copy.k_off,
                                                     col,
                                                     ctx->K,
                                                     ctx->storage_dtype,
                                                     width,
                                                     rocke_gemm_read_clipped_scalar,
                                                     &clipped);
    }
    else if(copy.source->rank == 2)
    {
        auto* ptr = copy.source->view->base;
        value = width == 1 ? rocke_b_global_load(b, ptr, copy.packed_offset, ctx->storage_dtype, 0)
                           : rocke_b_global_load_vN(
                                 b, ptr, copy.packed_offset, ctx->storage_dtype, width, 0);
    }
    else
    {
        rocke_value_t* src_idx[3] = {rocke_b_const_i32(b, 0), row, col};
        value = width == 1 ? rocke_tile_window_load_scalar(b, copy.source, src_idx, 3)
                           : rocke_tile_window_load_vec(b, copy.source, src_idx, 3, width);
    }

    rocke_value_t* dst_row = row;
    if(ctx->db && copy.parity_v != NULL)
        dst_row
            = rocke_b_add(b, row, rocke_b_mul(b, copy.parity_v, rocke_b_const_i32(b, copy.rows)));
    else if(ctx->db && copy.parity_imm)
        dst_row = rocke_b_add(b, row, rocke_b_const_i32(b, copy.parity_imm * copy.rows));
    rocke_value_t* dst_idx[2] = {dst_row, rocke_gemm_swz_col(ctx, col, row)};
    if(width == 1)
        rocke_tile_window_store_scalar(b, copy.dest, dst_idx, 2, value, 0);
    else
        rocke_tile_window_store_vec(b, copy.dest, dst_idx, 2, value, width);
}

static void rocke_gemm_copy_sync_operand(rocke_gemm_sync_operand* copy)
{
    const auto* ctx = copy->ctx;
    rocke_coalesced_tile_loader_t loader = {};
    loader.tile_rows = copy->rows;
    loader.tile_cols = ctx->block_k;
    loader.block_size = ctx->threads;
    loader.load_vec = ctx->load_vec;
    const bool prepare_packed = copy->source->rank == 2 && !ctx->spec->trait.pad_k;
    const rocke_coalesced_chunk_grid_t grid{ctx->c_threads,
                                            ctx->c_load_vec,
                                            ctx->c_block_k_div_vec,
                                            prepare_packed ? rocke_gemm_prepare_packed_offset
                                                           : NULL};
    rocke_coalesced_tile_loader_visit_chunks(
        ctx->b, &loader, ctx->tid, &grid, rocke_gemm_copy_sync_chunk, copy);
}

void rocke_gemm_emit_load_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_dst,
                                rocke_value_t* B_dst,
                                rocke_value_t* k_off,
                                int parity_imm,
                                rocke_value_t* parity_v)
{
    rocke_gemm_emit_hook_t none = {NULL, NULL};
    rocke_gemm_emit_load_phase_ex(ctx, A_dst, B_dst, k_off, parity_imm, parity_v, none);
}

void rocke_gemm_emit_load_phase_ex(rocke_gemm_build_ctx_t* ctx,
                                   rocke_value_t* A_dst,
                                   rocke_value_t* B_dst,
                                   rocke_value_t* k_off,
                                   int parity_imm,
                                   rocke_value_t* parity_v,
                                   rocke_gemm_emit_hook_t before_issue)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* I64 = rocke_i64();

    /* ----------------------- TDM path ----------------------- *
     * Mirrors Python's `if spec.trait.tdm:` branch at the top of emit_load_phase.
     * One descriptor per operand moves the whole tile; returns early. */
    if(spec->trait.tdm)
    {
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* a_lds;
        rocke_value_t* b_lds;
        rocke_value_t* k_remaining;
        /* Every builder call gets its own statement. C leaves the evaluation
         * order of function arguments unspecified. */
        if(parity_v != NULL)
        {
            rocke_value_t* a_off = rocke_b_zext(
                b, rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->tdm_a_buf_bytes)), I64);
            a_lds = rocke_b_smem_ptr_add(b, ctx->tdm_a_lds_base, a_off);
            rocke_value_t* b_off = rocke_b_zext(
                b, rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->tdm_b_buf_bytes)), I64);
            b_lds = rocke_b_smem_ptr_add(b, ctx->tdm_b_lds_base, b_off);
        }
        else if(parity_imm)
        {
            rocke_value_t* a_off
                = rocke_b_zext(b, rocke_b_const_i32(b, parity_imm * ctx->tdm_a_buf_bytes), I64);
            a_lds = rocke_b_smem_ptr_add(b, ctx->tdm_a_lds_base, a_off);
            rocke_value_t* b_off
                = rocke_b_zext(b, rocke_b_const_i32(b, parity_imm * ctx->tdm_b_buf_bytes), I64);
            b_lds = rocke_b_smem_ptr_add(b, ctx->tdm_b_lds_base, b_off);
        }
        else
        {
            a_lds = ctx->tdm_a_lds_base;
            b_lds = ctx->tdm_b_lds_base;
        }

        k_remaining = rocke_b_smax(b, rocke_b_sub(b, ctx->K, k_off), zero);

        /* tensor_load_to_lds is wave-level; one wave owns each operand. */
        if(ctx->tdm_waves > 1)
        {
            /* Waves 2p / 2p+1 own the fills of half p with
             * tdm_alternate_issuers; waves 0 / 1 otherwise. The default emits
             * the B-wave constant after the A gate, where it always has. */
            rocke_value_t* a_wave = zero;
            rocke_value_t* b_wave = NULL;
            if(spec->trait.tdm_alternate_issuers)
            {
                if(parity_v != NULL)
                {
                    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
                    a_wave = rocke_b_mul(b, parity_v, c2);
                }
                else
                {
                    a_wave = rocke_b_const_i32(b, 2 * parity_imm);
                }
                rocke_value_t* c1 = rocke_b_const_i32(b, 1);
                b_wave = rocke_b_add(b, a_wave, c1);
            }
            rocke_if_t a_gate = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tdm_wave_id, a_wave));
            rocke_b_region_enter(b, a_gate.then_region);
            rocke_gemm_tdm_issue(ctx,
                                 ctx->A,
                                 ctx->block_m_off,
                                 ctx->block_m,
                                 ctx->M,
                                 a_lds,
                                 ctx->tdm_batch_a,
                                 k_off,
                                 k_remaining,
                                 zero,
                                 before_issue);
            rocke_b_region_leave(b);
            if(b_wave == NULL)
                b_wave = rocke_b_const_i32(b, 1);
            rocke_if_t b_gate = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tdm_wave_id, b_wave));
            rocke_b_region_enter(b, b_gate.then_region);
            rocke_gemm_tdm_issue(ctx,
                                 ctx->Bp,
                                 ctx->block_n_off,
                                 ctx->block_n,
                                 ctx->N,
                                 b_lds,
                                 ctx->tdm_batch_b,
                                 k_off,
                                 k_remaining,
                                 zero,
                                 before_issue);
            rocke_b_region_leave(b);
        }
        else
        {
            rocke_gemm_tdm_issue(ctx,
                                 ctx->A,
                                 ctx->block_m_off,
                                 ctx->block_m,
                                 ctx->M,
                                 a_lds,
                                 ctx->tdm_batch_a,
                                 k_off,
                                 k_remaining,
                                 zero,
                                 before_issue);
            rocke_gemm_tdm_issue(ctx,
                                 ctx->Bp,
                                 ctx->block_n_off,
                                 ctx->block_n,
                                 ctx->N,
                                 b_lds,
                                 ctx->tdm_batch_b,
                                 k_off,
                                 k_remaining,
                                 zero,
                                 before_issue);
        }
        return;
    }

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

        rocke_gemm_dtl_source src_a{ctx, c_eb, ctx->batch_off_a, ctx->block_m_off, k_off};
        rocke_gemm_dtl_source src_b{ctx, c_eb, ctx->batch_off_b, ctx->block_n_off, k_off};
        rocke_gemm_issue_dtl_operand(&src_a,
                                     A_dst,
                                     a_lds_wave_base,
                                     ctx->dtl_a_rsrc,
                                     ctx->dtl_a_passes,
                                     a_parity_bytes_static,
                                     spec->trait.dtl_cache_a);
        rocke_gemm_issue_dtl_operand(&src_b,
                                     B_dst,
                                     b_lds_wave_base,
                                     ctx->dtl_b_rsrc,
                                     ctx->dtl_b_passes,
                                     b_parity_bytes_static,
                                     spec->trait.dtl_cache_b);
        return;
    }

    /* Synchronous copy: ordinary and packed operands differ only in the
     * source view and origin. Both use the coalesced visitor. */
    rocke_tile_window_t a_global_tile, b_global_tile, a_lds_tile, b_lds_tile;
    const int a_glen[3] = {1, ctx->block_m, ctx->block_k};
    int b_glen[3] = {1, ctx->block_n, ctx->block_k};
    rocke_value_t* a_gorigin[3] = {ctx->batch_off_a, ctx->block_m_off, k_off};
    rocke_value_t* b_gorigin[3] = {ctx->batch_off_b, ctx->block_n_off, k_off};
    rocke_make_tile_window(&a_global_tile, &ctx->a_view, a_glen, a_gorigin, 3);

    const int a_llen[2] = {ctx->block_m, ctx->block_k};
    const int b_llen[2] = {ctx->block_n, ctx->block_k};
    rocke_value_t* a_lorigin[2] = {rocke_b_const_i32(b, 0), rocke_b_const_i32(b, 0)};
    rocke_value_t* b_lorigin[2] = {rocke_b_const_i32(b, 0), rocke_b_const_i32(b, 0)};
    rocke_tensor_view_t a_lds_view = ctx->a_lds_view;
    rocke_tensor_view_t b_lds_view = ctx->b_lds_view;
    a_lds_view.base = A_dst;
    b_lds_view.base = B_dst;
    rocke_make_tile_window(&a_lds_tile, &a_lds_view, a_llen, a_lorigin, 2);
    rocke_make_tile_window(&b_lds_tile, &b_lds_view, b_llen, b_lorigin, 2);

    rocke_gemm_sync_operand a{
        ctx, &a_global_tile, &a_lds_tile, k_off, ctx->block_m, parity_imm, parity_v};
    rocke_gemm_copy_sync_operand(&a);
    if(spec->trait.preshuffle_b)
    {
        rocke_value_t* n_tile = rocke_b_div(b, ctx->block_n_off, ctx->c_block_n);
        rocke_value_t* k_tile = rocke_b_div(b, k_off, ctx->c_block_k);
        rocke_value_t* n_tile_count = rocke_b_div(b, ctx->N, ctx->c_block_n);
        rocke_value_t* tile_id = rocke_emit_preshuffleb_tile_id(b, n_tile, k_tile, n_tile_count);
        rocke_value_t* tile_elems = rocke_b_const_i32(b, ctx->block_n * ctx->block_k);
        b_gorigin[0] = rocke_b_add(b, ctx->batch_off_b, rocke_b_mul(b, tile_id, tile_elems));
        b_gorigin[1] = b_lorigin[0];
        b_glen[1] = ctx->block_n * ctx->block_k;
    }
    rocke_make_tile_window(&b_global_tile, &ctx->b_view, b_glen, b_gorigin, ctx->b_view.desc.rank);
    rocke_gemm_sync_operand bv{
        ctx, &b_global_tile, &b_lds_tile, k_off, ctx->block_n, parity_imm, parity_v};
    rocke_gemm_copy_sync_operand(&bv);
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
    /* fp8/bf8 is the one case where frag_len is not an element count: the
     * gfx1250 K=64 WMMA carries its operands as <frag_len x i32>, one dword per
     * slot covering four K-contiguous bytes. 8-bit LDS loads cap at 16 elements
     * (ds_read_b128), so read the 4*frag_len bytes as 16-byte chunks, bitcast
     * each to <4 x i32>, and concatenate into the <frag_len x i32> operand. */
    if(rocke_gemm_storage_is_fp8(ctx->storage_dtype))
    {
        const rocke_type_t* dword4 = rocke_vector_type(b, rocke_i32(), 4);
        rocke_value_t* fp8_frag = NULL;

        for(int off = 0; off < frag_len * 4; off += 16)
        {
            rocke_value_t* chunk
                = rocke_gemm_emit_smem_load(b,
                                            src,
                                            lds_row,
                                            rocke_b_add(b, lds_col, rocke_b_const_i32(b, off)),
                                            16,
                                            ctx->storage_dtype);
            rocke_value_t* dwords = rocke_b_vec_bitcast(b, chunk, dword4);
            fp8_frag = (fp8_frag == NULL) ? dwords : rocke_b_vec_concat(b, fp8_frag, dwords);
        }
        return fp8_frag;
    }
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
                                int parity_imm,
                                rocke_value_t* parity_v,
                                rocke_value_t** out_accs)
{
    rocke_gemm_emit_hook_t none = {NULL, NULL};
    rocke_gemm_emit_wmma_phase_ex(
        ctx, A_src, B_src, iter_vars, num_iter_vars, parity_imm, parity_v, out_accs, none, none);
}

struct rocke_gemm_wmma_operand
{
    rocke_value_t* lds;
    rocke_value_t* free_in_atom;
    rocke_value_t* k_in_atom;
    rocke_value_t* warp_off;
    rocke_value_t* slot_off;
    int static_slot_off;
    int atom_rows;
    int frag_len;
};

struct rocke_gemm_wmma_tile
{
    rocke_gemm_build_ctx_t* ctx;
    rocke_gemm_wmma_operand a;
    rocke_gemm_wmma_operand bb;
    rocke_value_t* k_tile_base;
    rocke_gemm_emit_hook_t after_last_reads;
    rocke_gemm_emit_hook_t after_first_row;
};

static rocke_value_t* rocke_gemm_read_wmma_operand(rocke_ir_builder_t* b,
                                                   const rocke_gemm_wmma_tile& tile,
                                                   const rocke_gemm_wmma_operand& op,
                                                   int atom)
{
    rocke_value_t* row = rocke_b_add(
        b, op.warp_off, rocke_b_const_i32(b, atom * op.atom_rows + op.static_slot_off));
    if(op.slot_off != NULL)
        row = rocke_b_add(b, row, op.slot_off);
    return rocke_gemm_emit_frag_smem_load(
        tile.ctx, op.lds, op.free_in_atom, op.k_in_atom, row, tile.k_tile_base, op.frag_len);
}

static rocke_value_t* rocke_gemm_read_wmma_a(rocke_ir_builder_t* b, int ki, int mi, void* user)
{
    auto& tile = *static_cast<rocke_gemm_wmma_tile*>(user);
    if(mi == 0)
        tile.k_tile_base = rocke_b_const_i32(b, ki * tile.ctx->spec->tile.warp_tile_k);
    return rocke_gemm_read_wmma_operand(b, tile, tile.a, mi);
}

static rocke_value_t* rocke_gemm_read_wmma_b(rocke_ir_builder_t* b, int, int ni, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_wmma_tile*>(user);
    return rocke_gemm_read_wmma_operand(b, tile, tile.bb, ni);
}

static rocke_value_t* rocke_gemm_issue_wmma(
    rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb, rocke_value_t* acc, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_wmma_tile*>(user);
    return rocke_gemm_emit_mma(b, tile.ctx->op, a, bb, acc);
}

static void rocke_gemm_finish_wmma_reads(rocke_ir_builder_t*, int ki, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_wmma_tile*>(user);
    if(ki == tile.ctx->k_atoms - 1)
        tile.after_last_reads.fn(tile.ctx, tile.after_last_reads.user);
}

static void rocke_gemm_finish_wmma_row(rocke_ir_builder_t*, int ki, int mi, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_wmma_tile*>(user);
    if(ki == tile.ctx->k_atoms - 1 && mi == 0)
        tile.after_first_row.fn(tile.ctx, tile.after_first_row.user);
}

/* The after_last_reads hook runs before the last K atom's first MMA;
 * after_first_row follows its first complete MMA row. */
void rocke_gemm_emit_wmma_phase_ex(rocke_gemm_build_ctx_t* ctx,
                                   rocke_value_t* A_src,
                                   rocke_value_t* B_src,
                                   rocke_value_t* const* iter_vars,
                                   int num_iter_vars,
                                   int parity_imm,
                                   rocke_value_t* parity_v,
                                   rocke_value_t** out_accs,
                                   rocke_gemm_emit_hook_t after_last_reads,
                                   rocke_gemm_emit_hook_t after_first_row)
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

    /* The ring/ping-pong slot this K-tile was filled into. The load side already
     * offsets its LDS destination by the parity; without the same shift here
     * every WMMA reads slot 0 while the mover is filling another -- a read/write
     * race, not a wrong-address bug. */
    bool parity_is_value = parity_v != NULL;
    rocke_value_t* a_par_row_v;
    rocke_value_t* b_par_row_v;
    int a_par_row_static;
    int b_par_row_static;
    if(ctx->prefetch && parity_is_value)
    {
        a_par_row_v = rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->block_m));
        b_par_row_v = rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->block_n));
        a_par_row_static = 0;
        b_par_row_static = 0;
    }
    else
    {
        a_par_row_v = NULL;
        b_par_row_v = NULL;
        a_par_row_static = ctx->prefetch ? parity_imm * ctx->block_m : 0;
        b_par_row_static = ctx->prefetch ? parity_imm * ctx->block_n : 0;
    }

    rocke_gemm_wmma_tile tile{
        ctx,
        {A_src,
         a_row_in_atom,
         a_k_in_atom,
         warp_m_off,
         a_par_row_v,
         a_par_row_static,
         t->warp_tile_m,
         ctx->a_per_lane},
        {B_src,
         b_col_in_atom,
         b_k_in_atom,
         warp_n_off,
         b_par_row_v,
         b_par_row_static,
         t->warp_tile_n,
         ctx->b_per_lane},
        nullptr,
        after_last_reads,
        after_first_row,
    };
    const rocke_tile_mma_readers_t readers{
        .read_a = rocke_gemm_read_wmma_a,
        .read_b = rocke_gemm_read_wmma_b,
        .mma = rocke_gemm_issue_wmma,
        .after_reads = after_last_reads.fn ? rocke_gemm_finish_wmma_reads : nullptr,
        .before_row = nullptr,
        .after_row = after_first_row.fn ? rocke_gemm_finish_wmma_row : nullptr,
        .user = &tile,
    };
    rocke_tile_mma_accumulate_staged(b,
                                     {ctx->mfmas_m, ctx->mfmas_n, ctx->k_atoms},
                                     iter_vars,
                                     static_cast<size_t>(ctx->num_accs),
                                     &readers,
                                     out_accs);
}
