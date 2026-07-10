// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_load_phase.c -- one part-file of the chunked
 * C99 port of build_implicit_gemm_conv
 * (rocke/instances/common/conv_implicit_gemm.py).
 *
 * SCOPE (this TU only):
 *   * rocke_conv_a_descriptor / rocke_conv_b_descriptor -- the k_off_capture-reading
 *     address closures (Python a_descriptor / b_descriptor, lines 962-984). Each
 *     maps a tile-local (row, col) to a (linear element offset, valid predicate)
 *     via the coord-transform DAG descriptors A_desc / B_desc.
 *   * rocke_conv_emit_load_phase -- the one-K-tile global->LDS copy closure (Python
 *     emit_load_phase, lines 1034-1106): sets ctx->k_off_capture then dispatches
 *     the async (AsyncTileLoader.bind/issue + raw_ptr_buffer_load_lds, with
 *     CACHE_STREAM) vs sync (CoalescedTileLoader.load) path, honouring the
 *     a_load_override hook on the sync path.
 *   * rocke_conv_choose_load_vec -- the load-side module helper (Python
 *     _choose_load_vec, lines 722-727): thin adapter over rocke_choose_load_vec.
 *   * rocke_conv_emit_smem_load -- the load-side module helper (Python
 *     _emit_smem_load, lines 641-644): f16 n==4 fast path else smem_load_vN_f16.
 *
 * Peers (the descriptor builders, the prologue populate, the compute/MFMA
 * phases, the K-loop drivers, and the epilogue) are CALLED via
 * rocke/instance_conv_implicit_gemm_internal.h and defined in sibling part-files;
 * this TU never re-defines them, nor does it edit any header.
 *
 * Faithfulness: every builder call below is in the same order, with the same
 * operands, as the corresponding Python line so the emitted IR op stream
 * matches byte-for-byte.
 */

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_choose_load_vec */
#include "rocke/instance_conv_implicit_gemm_internal.h"

/* ===================================================================== *
 *  _choose_load_vec -- pick the widest fp16 load vector width.
 *
 *  Python span: conv_implicit_gemm.py lines 722-727:
 *      def _choose_load_vec(spec):
 *          return choose_load_vec(spec.tile_m, spec.tile_n, spec.tile_k,
 *                                 spec.block_size)
 * ===================================================================== */
int rocke_conv_choose_load_vec(const rocke_implicit_gemm_conv_spec_t* spec)
{
    int out_vec = 0;
    /* spec.block_size = warp_m * warp_n * wave_size (the @property). */
    int block_size = rocke_implicit_gemm_conv_spec_block_size(spec);
    rocke_status_t st
        = rocke_choose_load_vec(spec->tile_m, spec->tile_n, spec->tile_k, block_size, &out_vec);
    /* On the Python ValueError path choose_load_vec raises; here the status is
     * ROCKE_ERR_VALUE and out_vec is left untouched (0). The prologue is the
     * gate that surfaces the spec validity error before this is reached. */
    if(st != ROCKE_OK)
        return out_vec;
    return out_vec;
}

/* ===================================================================== *
 *  _emit_smem_load -- one fp16 LDS read (v4 fast path else vN).
 *
 *  Python span: conv_implicit_gemm.py lines 641-644:
 *      def _emit_smem_load(b, smem, row, col, n):
 *          if n == 4:
 *              return b.smem_load_v4_f16(smem, row, col)
 *          return b.smem_load_vN_f16(smem, row, col, n=n)
 *
 *  The C smem_load_vN_f16 takes an explicit (row, col) index pair as an indices
 *  array of length 2, matching the Python (smem, row, col) call.
 * ===================================================================== */
rocke_value_t* rocke_conv_emit_smem_load(
    rocke_ir_builder_t* b, rocke_value_t* smem, rocke_value_t* row, rocke_value_t* col, int n)
{
    if(n == 4)
        return rocke_b_smem_load_v4_f16(b, smem, row, col);
    {
        rocke_value_t* idx[2];
        idx[0] = row;
        idx[1] = col;
        return rocke_b_smem_load_vN_f16(b, smem, idx, 2, n);
    }
}

/* ===================================================================== *
 *  a_descriptor -- the A global-address closure.
 *
 *  Python span: conv_implicit_gemm.py lines 962-974:
 *      def a_descriptor(b_, row, col):
 *          k_val = b_.add(k_off_capture[0], col)
 *          if a_mhw_index_fn is not None:
 *              n_v, ho_v, wo_v = a_mhw_index_fn(b_, row, grid)
 *              return A_desc.offset(b_, n=n_v, ho=ho_v, wo=wo_v, k=k_val)
 *          m_val = (m_index_fn(b_, row, grid)
 *                   if m_index_fn is not None
 *                   else b_.add(block_m_off_v, row))
 *          return A_desc.offset(b_, m=m_val, k=k_val)
 *
 *  ctx is threaded as the loader's `user`; the K-loop's emit_load_phase writes
 *  ctx->k_off_capture before the loaders fire so this closure reads the current
 *  k0 without recompiling. Writes the valid predicate through *out_valid (NULL
 *  == Python None "always in-bounds").
 * ===================================================================== */
rocke_value_t* rocke_conv_a_descriptor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* ctx_user)
{
    rocke_conv_build_ctx_t* ctx = (rocke_conv_build_ctx_t*)ctx_user;
    const rocke_conv_build_overrides_t* ov = ctx->ov;

    /* k_val = b_.add(k_off_capture[0], col) */
    rocke_value_t* k_val = rocke_b_add(b, ctx->k_off_capture, col);

    if(ov != NULL && ov->a_mhw_index_fn != NULL)
    {
        /* Decomposed A descriptor: feed (n, ho, wo) straight in, skipping the
         * m-flatten -> magic-unmerge round-trip (see make_a_descriptor). */
        rocke_value_t* n_v = NULL;
        rocke_value_t* ho_v = NULL;
        rocke_value_t* wo_v = NULL;
        ov->a_mhw_index_fn(b, row, &ctx->grid, &n_v, &ho_v, &wo_v, ov->user);

        /* A_desc.offset(b_, n=n_v, ho=ho_v, wo=wo_v, k=k_val) */
        const char* names[4] = {"n", "ho", "wo", "k"};
        rocke_value_t* vals[4] = {n_v, ho_v, wo_v, k_val};
        rocke_value_t* off = NULL;
        rocke_value_t* valid = NULL;
        rocke_transforms_descriptor_offset(b, ctx->A_desc, names, vals, 4, &off, &valid);
        *out_valid = valid;
        return off;
    }

    /* m_val = m_index_fn(b_, row, grid) if m_index_fn else b_.add(block_m_off_v, row) */
    rocke_value_t* m_val;
    if(ov != NULL && ov->m_index_fn != NULL)
        m_val = ov->m_index_fn(b, row, &ctx->grid, ov->user);
    else
        m_val = rocke_b_add(b, ctx->block_m_off_v, row);

    /* A_desc.offset(b_, m=m_val, k=k_val) */
    {
        const char* names[2] = {"m", "k"};
        rocke_value_t* vals[2] = {m_val, k_val};
        rocke_value_t* off = NULL;
        rocke_value_t* valid = NULL;
        rocke_transforms_descriptor_offset(b, ctx->A_desc, names, vals, 2, &off, &valid);
        *out_valid = valid;
        return off;
    }
}

/* ===================================================================== *
 *  b_descriptor -- the B global-address closure.
 *
 *  Python span: conv_implicit_gemm.py lines 976-979:
 *      def b_descriptor(b_, row, col):
 *          k_out = b_.add(block_n_off_v, row)
 *          kg = b_.add(k_off_capture[0], col)
 *          return B_desc.offset(b_, k_out=k_out, k_gemm=kg)
 * ===================================================================== */
rocke_value_t* rocke_conv_b_descriptor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* ctx_user)
{
    rocke_conv_build_ctx_t* ctx = (rocke_conv_build_ctx_t*)ctx_user;

    rocke_value_t* k_out = rocke_b_add(b, ctx->block_n_off_v, row);
    rocke_value_t* kg = rocke_b_add(b, ctx->k_off_capture, col);

    const char* names[2] = {"k_out", "k_gemm"};
    rocke_value_t* vals[2] = {k_out, kg};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, ctx->B_desc, names, vals, 2, &off, &valid);
    *out_valid = valid;
    return off;
}

/* ===================================================================== *
 *  emit_load_phase -- global -> LDS copy for one K tile via the descriptor DAG.
 *
 *  Python span: conv_implicit_gemm.py lines 1034-1106.
 *
 *  Sets ctx->k_off_capture = k_off (the Python `k_off_capture[0] = k_off`) so
 *  the descriptor closures pick up the current k0, then dispatches:
 *    async_dma=True : AsyncTileLoader.bind(...).issue(...) for A and B, each with
 *                     CACHE_STREAM coherency + raw_ptr_buffer_load_lds inside.
 *    async_dma=False: CoalescedTileLoader.load(...) for A (or the a_load_override
 *                     hook) and B.
 * ===================================================================== */
void rocke_conv_emit_load_phase(rocke_conv_build_ctx_t* ctx,
                                rocke_value_t* k_off,
                                rocke_value_t* A_dst,
                                rocke_value_t* B_dst)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const rocke_conv_build_overrides_t* ov = ctx->ov;

    /* k_off_capture[0] = k_off */
    ctx->k_off_capture = k_off;

    if(spec->async_dma)
    {
        /* CACHE_STREAM (SLC=1): each K-tile is consumed exactly once by the
         * MFMA phase of the same iter then overwritten by the next iter's
         * prefetch; streaming keeps the loads from evicting useful cache lines.
         * (Python imports CACHE_STREAM from ...core.ir; the C enum value is
         * ROCKE_CACHE_STREAM.) */
        rocke_async_tile_loader_slot_t a_slot;
        rocke_async_tile_loader_bind(b, &ctx->a_loader, A_dst, ctx->warp_id, &a_slot);
        rocke_async_tile_loader_slot_issue(b,
                                           &a_slot,
                                           ctx->tid,
                                           ctx->a_rsrc,
                                           rocke_conv_a_descriptor,
                                           ctx,
                                           0x7FFFFFFF, /* oob_sentinel default = (1 << 31) - 1 */
                                           ROCKE_CACHE_STREAM);

        rocke_async_tile_loader_slot_t b_slot;
        rocke_async_tile_loader_bind(b, &ctx->b_loader, B_dst, ctx->warp_id, &b_slot);
        rocke_async_tile_loader_slot_issue(b,
                                           &b_slot,
                                           ctx->tid,
                                           ctx->b_rsrc,
                                           rocke_conv_b_descriptor,
                                           ctx,
                                           0x7FFFFFFF,
                                           ROCKE_CACHE_STREAM);
        return;
    }

    /* Sync path: CoalescedTileLoader.load emits the per-thread
     * buffer_load_vN_f16 -> smem_store_vN_f16 chunks. The descriptor callback
     * gets (row, col) inside the tile-local frame and returns the global element
     * offset + validity, so the conv-coord-transform DAG drives the address
     * arithmetic while the loader owns the thread distribution. */
    if(ov != NULL && ov->a_load_override != NULL)
    {
        ov->a_load_override(b, spec, k_off, A_dst, &ctx->grid, ctx->input_cache_context, ov->user);
    }
    else
    {
        rocke_coalesced_tile_loader_load(b,
                                         &ctx->a_sync_loader,
                                         ctx->tid,
                                         A_dst,
                                         rocke_conv_a_descriptor,
                                         ctx,
                                         ctx->a_rsrc,
                                         NULL); /* use_buffer_rsrc => ptr unused */
    }
    rocke_coalesced_tile_loader_load(
        b, &ctx->b_sync_loader, ctx->tid, B_dst, rocke_conv_b_descriptor, ctx, ctx->b_rsrc, NULL);
}
