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
#include "rocke/tdm.h" /* rocke_tdm_build_descriptor_2d */

/* Widest global A/B load vector, in elements: the 4-dword buffer_load limit is
 * 16 bytes, so a 1-byte operand tops out at 16 elements. The pad_k gather
 * arrays below are sized to this. */
#define ROCKE_GEMM_MAX_LOAD_VEC 16

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
                                               rocke_value_t* k_off)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* row_term = rocke_b_mul(b, rocke_b_add(b, blk_off, row), ctx->K);
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
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* origin
        = rocke_b_add(b, batch_off, rocke_b_add(b, rocke_b_mul(b, row_off, ctx->K), k_off));
    rocke_tdm_descriptor_2d_args_t args;
    rocke_value_t* groups[5];
    memset(&args, 0, sizeof(args));
    args.global_addr = rocke_b_global_addr_of(b, ptr, origin);
    args.lds_addr = lds_base;
    args.elem_bytes = ctx->tdm_elem_bytes;
    args.tensor_dim0 = k_remaining;
    /* The TENSOR's remaining extent from this tile's origin, not the tile
     * height. The mover bounds its address generation on this, so a constant
     * tile height describes a tensor ending `rows` past every origin -- wrong
     * for every tile but the first. pad_m/pad_n are not consulted precisely
     * because OOB is handled here, by clipping the extent. */
    args.tensor_dim1 = rocke_b_smax(b, rocke_b_sub(b, extent, row_off), zero);
    args.tile_dim0 = ctx->block_k;
    args.tile_dim1 = rows;
    /* dim0_stride_value = K (runtime row pitch). Mirrors Python: dim0_stride=K. */
    args.dim0_stride_value = ctx->K;
    args.dim1_stride = 1;
    args.pad_enable = ctx->tdm_pad_enable;
    args.pad_interval = ctx->tdm_pad_interval;
    args.pad_amount = ctx->tdm_pad_amount;
    if(rocke_tdm_build_descriptor_2d(b, &args, groups) != ROCKE_OK)
    {
        return;
    }
    if(before_issue.fn != NULL)
        before_issue.fn(ctx, before_issue.user);
    rocke_b_tensor_load_to_lds(b, groups[0], groups[1], groups[2], groups[3], groups[4], 0);
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
    (void)A_dst; /* Python emit_load_phase's A_dst/B_dst args are unused in body. */
    (void)B_dst;

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
         * order of function arguments unspecified (see tdm.cpp). */
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
                                 ctx->batch_off_a,
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
                                 ctx->batch_off_b,
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
                                 ctx->batch_off_a,
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
                                 ctx->batch_off_b,
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
                ctx, c_eb, ctx->batch_off_a, ctx->block_m_off, row, col, k_off);
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
                ctx, c_eb, ctx->batch_off_b, ctx->block_n_off, row, col, k_off);
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

    /* The pad_k sub-paths below gather load_vec scalars into a fixed
     * ROCKE_GEMM_MAX_LOAD_VEC-element array before packing. load_vec is capped
     * at 16/elem_bytes by rocke_choose_load_vec, so with a 1-byte operand it is
     * exactly 16 -- the array is full with no headroom. Clamp rather than
     * overrun if that invariant ever moves. */
    if(load_vec > ROCKE_GEMM_MAX_LOAD_VEC)
    {
        load_vec = ROCKE_GEMM_MAX_LOAD_VEC;
    }

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
/* _zero_storage_scalar(): cast_f32_to has no 8-bit arm, and the cvt_f32_to_fp8
 * route would burn a conversion intrinsic to produce a byte we already know:
 * +0.0 is all-zero in both e4m3 and e5m2, and both lower to i8, so truncating a
 * zero i32 is the whole job. fp64 takes an f64 constant. */
#define STORAGE_ZERO()                                                   \
    (rocke_gemm_storage_is_fp8(ctx->storage_dtype)                       \
         ? rocke_b_trunc(b, rocke_b_const_i32(b, 0), ctx->storage_dtype) \
     : rocke_type_eq(ctx->storage_dtype, rocke_f64())                    \
         ? rocke_b_const_f64(b, 0.0)                                     \
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
            rocke_value_t* comps[ROCKE_GEMM_MAX_LOAD_VEC];
            rocke_value_t* a_val;
            for(int i = 0; i < load_vec; ++i)
            {
                rocke_value_t* elem_col = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                rocke_value_t* valid = PADK_VALID(elem_col);
                rocke_value_t* safe_col
                    = rocke_b_select(b, valid, elem_col, rocke_b_const_i32(b, 0));
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
                rocke_value_t* comps[ROCKE_GEMM_MAX_LOAD_VEC];
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
                rocke_value_t* comps[ROCKE_GEMM_MAX_LOAD_VEC];
                rocke_value_t* b_val;
                for(int i = 0; i < load_vec; ++i)
                {
                    rocke_value_t* elem_col
                        = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                    rocke_value_t* valid = PADK_VALID(elem_col);
                    rocke_value_t* safe_col
                        = rocke_b_select(b, valid, elem_col, rocke_b_const_i32(b, 0));
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

/* after_last_reads is emitted once the last k-atom's fragment loads are issued
 * and before its WMMAs -- the earliest point at which this wave has no further
 * reads of the tile (the split-barrier signal). */
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
            rocke_value_t* atom_row = rocke_b_add(
                b, warp_m_off, rocke_b_const_i32(b, mi * t->warp_tile_m + a_par_row_static));
            if(a_par_row_v != NULL)
                atom_row = rocke_b_add(b, atom_row, a_par_row_v);
            a_rows[mi] = rocke_gemm_emit_frag_smem_load(
                ctx, A_src, a_row_in_atom, a_k_in_atom, atom_row, k_tile_base, ctx->a_per_lane);
        }
        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            rocke_value_t* atom_row = rocke_b_add(
                b, warp_n_off, rocke_b_const_i32(b, ni * t->warp_tile_n + b_par_row_static));
            if(b_par_row_v != NULL)
                atom_row = rocke_b_add(b, atom_row, b_par_row_v);
            b_cols[ni] = rocke_gemm_emit_frag_smem_load(
                ctx, B_src, b_col_in_atom, b_k_in_atom, atom_row, k_tile_base, ctx->b_per_lane);
        }
        bool last_k_atom = kk == ctx->k_atoms - 1;
        if(after_last_reads.fn != NULL && last_k_atom)
            after_last_reads.fn(ctx, after_last_reads.user);
        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat]
                    = rocke_gemm_emit_mma(b, ctx->op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
            if(after_first_row.fn != NULL && last_k_atom && mi == 0)
                after_first_row.fn(ctx, after_first_row.user);
        }
    }
}
