// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_fp8_moe_fused_mega_fp8_loads_dtla.c -- one part-file of
 * the chunked C99 port of build_moe_fused_mega_gemm_fp8
 * (rocke/instances/common/moe_fused_mega_fp8.py).
 *
 * SCOPE (this TU only): FP8 OPERAND LOADS + DIRECT-TO-LDS STAGING.
 *   Python lines 983-1216:
 *     _global_load_fp8_vec   -> rocke_moe_fp8_global_load_fp8_vec
 *     _load_a_fp8            -> rocke_moe_fp8_load_a_fp8
 *     _load_b_fp8            -> rocke_moe_fp8_load_b_fp8
 *     (_load_a_fp8_lds       -> rocke_moe_fp8_load_a_fp8_lds  -- see note below)
 *     _dtla_stage_b_fp8      -> rocke_moe_fp8_dtla_stage_b_fp8
 *     _dtla_read_b_fp8       -> rocke_moe_fp8_dtla_read_b_fp8
 *     _xdtla_stage_a_fp8     -> rocke_moe_fp8_xdtla_stage_a_fp8
 *     _xdtla_read_a_fp8      -> rocke_moe_fp8_xdtla_read_a_fp8
 *
 * These are the shared toolkit consumed by the gate/up fused K-loop and the down
 * GEMM. Peers (the fused K-loop, the down GEMM, the body driver, etc.) are CALLED
 * via rocke/instance_moe_fused_mega_fp8_internal.h and live in sibling part-files;
 * this TU never re-defines them.
 *
 * Faithfulness: every builder call below is in the SAME order, with the same
 * operands/attrs, as the corresponding Python line. Where Python keyword-only
 * args are loop-local they are passed explicitly; where they are enclosing-scope
 * locals (the resolved atom, the lane decode) they are read from ctx.
 *
 * NOTE on _load_a_fp8_lds: the internal-header prototype declares it in this
 * bucket's surface, but its Python body lives in the STAGE-2 down-GEMM span
 * (lines 1224-1255), outside the 983-1216 scope assigned here. It is stubbed
 * to-link below so the gate/up + down peers can bind; the down-GEMM part-file
 * owns its faithful body.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/instance_moe_fused_mega_fp8.h"
#include "rocke/instance_moe_fused_mega_fp8_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 * _global_load_fp8_vec (Python lines 983-1002)
 *
 *   if n <= 16:
 *       return b.global_load_vN(ptr, addr, FP8E4M3, n)
 *   acc = None
 *   off = 0
 *   while off < n:
 *       chunk = min(16, n - off)
 *       a = addr if off == 0 else b.add(addr, b.const_i32(off))
 *       v = b.global_load_vN(ptr, a, FP8E4M3, chunk)
 *       acc = v if acc is None else b.vec_concat(acc, v)
 *       off += chunk
 *   return acc
 * ===================================================================== */
rocke_value_t* rocke_moe_fp8_global_load_fp8_vec(rocke_moe_fp8_build_ctx_t* ctx,
                                                 rocke_value_t* ptr,
                                                 rocke_value_t* addr,
                                                 int n)
{
    rocke_ir_builder_t* b = ctx->b;
    if(n <= 16)
    {
        return rocke_b_global_load_vN(b, ptr, addr, rocke_fp8e4m3(), n, 0);
    }

    rocke_value_t* acc = NULL;
    int off = 0;
    while(off < n)
    {
        int chunk = (16 < (n - off)) ? 16 : (n - off);
        rocke_value_t* a = (off == 0) ? addr : rocke_b_add(b, addr, rocke_b_const_i32(b, off));
        rocke_value_t* v = rocke_b_global_load_vN(b, ptr, a, rocke_fp8e4m3(), chunk, 0);
        acc = (acc == NULL) ? v : rocke_b_vec_concat(b, acc, v);
        off += chunk;
    }
    return acc;
}

/* ===================================================================== *
 * _load_a_fp8 (Python lines 1005-1017)
 *
 *   m_row = b.add(m_tile_base, lane_decode.m_in_atom)
 *   k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
 *   k_base = b.add(k_tile_base, k_lane_start)
 *   a_addr = b.add(b.mul(m_row, K), k_base)
 *   return _global_load_fp8_vec(b, A, a_addr, atom.a_per_lane)
 * ===================================================================== */
rocke_value_t* rocke_moe_fp8_load_a_fp8(rocke_moe_fp8_build_ctx_t* ctx,
                                        rocke_value_t* A,
                                        rocke_value_t* m_tile_base,
                                        rocke_value_t* k_tile_base,
                                        rocke_value_t* K)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;

    rocke_value_t* m_row = rocke_b_add(b, m_tile_base, lane_decode->m_in_atom);
    rocke_value_t* k_lane_start
        = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->a_per_lane));
    rocke_value_t* k_base = rocke_b_add(b, k_tile_base, k_lane_start);
    rocke_value_t* a_addr = rocke_b_add(b, rocke_b_mul(b, m_row, K), k_base);
    return rocke_moe_fp8_global_load_fp8_vec(ctx, A, a_addr, atom->a_per_lane);
}

/* ===================================================================== *
 * _load_b_fp8 (Python lines 1020-1039)
 *
 *   n_col = b.add(n_tile_base, lane_decode.n_in_atom)
 *   k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
 *   k_base = b.add(k_tile_base, k_lane_start)
 *   row_base = b.mul(n_col, N)
 *   b_addr = b.add(row_base, k_base)
 *   return _global_load_fp8_vec(b, B, b_addr, atom.b_per_lane)
 * ===================================================================== */
rocke_value_t* rocke_moe_fp8_load_b_fp8(rocke_moe_fp8_build_ctx_t* ctx,
                                        rocke_value_t* B,
                                        rocke_value_t* n_tile_base,
                                        rocke_value_t* k_tile_base,
                                        rocke_value_t* N)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;

    rocke_value_t* n_col = rocke_b_add(b, n_tile_base, lane_decode->n_in_atom);
    rocke_value_t* k_lane_start
        = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->b_per_lane));
    rocke_value_t* k_base = rocke_b_add(b, k_tile_base, k_lane_start);
    rocke_value_t* row_base = rocke_b_mul(b, n_col, N);
    rocke_value_t* b_addr = rocke_b_add(b, row_base, k_base);
    return rocke_moe_fp8_global_load_fp8_vec(ctx, B, b_addr, atom->b_per_lane);
}

/* ===================================================================== *
 * _load_a_fp8_lds  (Python lines 1224-1255 -- STAGE-2 down-GEMM span)
 *
 *   m_row = b.add(m_tile_base, lane_decode.m_in_atom)
 *   k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
 *   k_col = b.add(k_tile_base, k_lane_start)
 *   n = atom.a_per_lane
 *   if n <= 16:
 *       return b.smem_load_vN(a_view.base, m_row, k_col, dtype=FP8E4M3, n=n)
 *   acc = None
 *   off = 0
 *   while off < n:
 *       chunk = min(16, n - off)
 *       c = k_col if off == 0 else b.add(k_col, b.const_i32(off))
 *       v = b.smem_load_vN(a_view.base, m_row, c, dtype=FP8E4M3, n=chunk)
 *       acc = v if acc is None else b.vec_concat(acc, v)
 *       off += chunk
 *   return acc
 *
 * smem_load_vN here is the 2-D LDS view read: indices [m_row, k_col].
 * ===================================================================== */
rocke_value_t* rocke_moe_fp8_load_a_fp8_lds(rocke_moe_fp8_build_ctx_t* ctx,
                                            const rocke_tensor_view_t* a_view,
                                            rocke_value_t* m_tile_base,
                                            rocke_value_t* k_tile_base)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;

    rocke_value_t* m_row = rocke_b_add(b, m_tile_base, lane_decode->m_in_atom);
    rocke_value_t* k_lane_start
        = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->a_per_lane));
    rocke_value_t* k_col = rocke_b_add(b, k_tile_base, k_lane_start);

    int n = atom->a_per_lane;
    if(n <= 16)
    {
        rocke_value_t* indices[2] = {m_row, k_col};
        return rocke_b_smem_load_vN(b, a_view->base, indices, 2, rocke_fp8e4m3(), n);
    }

    rocke_value_t* acc = NULL;
    int off = 0;
    while(off < n)
    {
        int chunk = (16 < (n - off)) ? 16 : (n - off);
        rocke_value_t* c = (off == 0) ? k_col : rocke_b_add(b, k_col, rocke_b_const_i32(b, off));
        rocke_value_t* indices[2] = {m_row, c};
        rocke_value_t* v
            = rocke_b_smem_load_vN(b, a_view->base, indices, 2, rocke_fp8e4m3(), chunk);
        acc = (acc == NULL) ? v : rocke_b_vec_concat(b, acc, v);
        off += chunk;
    }
    return acc;
}

/* ===================================================================== *
 * _dtla_stage_b_fp8 (Python lines 1072-1114)
 *
 *   n_col = b.add(n_tile_base, lane_decode.n_in_atom)
 *   k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
 *   k_base = b.add(k_tile_base, k_lane_start)
 *   row_base = b.mul(n_col, N)
 *   src_elem = b.add(row_base, k_base)
 *   frag_bytes = atom.b_per_lane
 *   chunks = (frag_bytes + DTLA_CHUNK - 1) // DTLA_CHUNK
 *   block_bytes = wave_size * DTLA_CHUNK
 *   slot_base_off = slot * chunks * block_bytes
 *   for c in range(chunks):
 *       chunk = min(DTLA_CHUNK, frag_bytes - c*DTLA_CHUNK)
 *       src = src_elem if c == 0 else b.add(src_elem, b.const_i32(c*DTLA_CHUNK))
 *       dst = b.smem_ptr_add(wave_lds_base, b.const_i64(slot_base_off + c*block_bytes))
 *       b.global_load_lds(B, src, dst, chunk, CACHE_ALL)
 *
 * lane is captured by the Python signature but unused in the body (the per-lane
 * spread is implicit in the hardware DMA); kept in the C signature for contract.
 * ===================================================================== */
void rocke_moe_fp8_dtla_stage_b_fp8(rocke_moe_fp8_build_ctx_t* ctx,
                                    rocke_value_t* B,
                                    rocke_value_t* n_tile_base,
                                    rocke_value_t* k_tile_base,
                                    rocke_value_t* N,
                                    const rocke_tensor_view_t* stage_view,
                                    int slot,
                                    rocke_value_t* wave_lds_base,
                                    rocke_value_t* lane,
                                    int wave_size)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;
    (void)stage_view;
    (void)lane;

    rocke_value_t* n_col = rocke_b_add(b, n_tile_base, lane_decode->n_in_atom);
    rocke_value_t* k_lane_start
        = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->b_per_lane));
    rocke_value_t* k_base = rocke_b_add(b, k_tile_base, k_lane_start);
    rocke_value_t* row_base = rocke_b_mul(b, n_col, N);
    rocke_value_t* src_elem = rocke_b_add(b, row_base, k_base);

    int frag_bytes = atom->b_per_lane;
    int chunks = (frag_bytes + ROCKE_MOE_FP8_DTLA_CHUNK - 1) / ROCKE_MOE_FP8_DTLA_CHUNK;
    int block_bytes = wave_size * ROCKE_MOE_FP8_DTLA_CHUNK;
    int slot_base_off = slot * chunks * block_bytes;

    for(int c = 0; c < chunks; ++c)
    {
        int rem = frag_bytes - c * ROCKE_MOE_FP8_DTLA_CHUNK;
        int chunk = (ROCKE_MOE_FP8_DTLA_CHUNK < rem) ? ROCKE_MOE_FP8_DTLA_CHUNK : rem;
        rocke_value_t* src
            = (c == 0)
                  ? src_elem
                  : rocke_b_add(b, src_elem, rocke_b_const_i32(b, c * ROCKE_MOE_FP8_DTLA_CHUNK));
        rocke_value_t* dst = rocke_b_smem_ptr_add(
            b, wave_lds_base, rocke_b_const_i64(b, slot_base_off + c * block_bytes));
        rocke_b_global_load_lds(b, B, src, dst, chunk, ROCKE_CACHE_ALL);
    }
}

/* ===================================================================== *
 * _dtla_read_b_fp8 (Python lines 1117-1145)
 *
 *   frag = atom.b_per_lane
 *   chunks = (frag + DTLA_CHUNK - 1) // DTLA_CHUNK
 *   acc = None
 *   for c in range(chunks):
 *       chunk = min(DTLA_CHUNK, frag - c*DTLA_CHUNK)
 *       row = b.add(warp_row_base,
 *                   b.add(b.const_i32((slot*chunks + c)*wave_size), lane))
 *       v = b.smem_load_vN(stage_view.base, row, b.const_i32(0), dtype=FP8E4M3, n=chunk)
 *       acc = v if acc is None else b.vec_concat(acc, v)
 *   return acc
 * ===================================================================== */
rocke_value_t* rocke_moe_fp8_dtla_read_b_fp8(rocke_moe_fp8_build_ctx_t* ctx,
                                             const rocke_tensor_view_t* stage_view,
                                             int slot,
                                             rocke_value_t* lane,
                                             rocke_value_t* warp_row_base,
                                             int wave_size)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;

    int frag = atom->b_per_lane;
    int chunks = (frag + ROCKE_MOE_FP8_DTLA_CHUNK - 1) / ROCKE_MOE_FP8_DTLA_CHUNK;
    rocke_value_t* acc = NULL;
    for(int c = 0; c < chunks; ++c)
    {
        int rem = frag - c * ROCKE_MOE_FP8_DTLA_CHUNK;
        int chunk = (ROCKE_MOE_FP8_DTLA_CHUNK < rem) ? ROCKE_MOE_FP8_DTLA_CHUNK : rem;
        rocke_value_t* row = rocke_b_add(
            b,
            warp_row_base,
            rocke_b_add(b, rocke_b_const_i32(b, (slot * chunks + c) * wave_size), lane));
        rocke_value_t* indices[2] = {row, rocke_b_const_i32(b, 0)};
        rocke_value_t* v
            = rocke_b_smem_load_vN(b, stage_view->base, indices, 2, rocke_fp8e4m3(), chunk);
        acc = (acc == NULL) ? v : rocke_b_vec_concat(b, acc, v);
    }
    return acc;
}

/* ===================================================================== *
 * _xdtla_stage_a_fp8 (Python lines 1159-1191; _USE_X_DTLA path)
 *
 *   m_row = b.add(m_tile_base, lane_decode.m_in_atom)
 *   k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
 *   k_base = b.add(k_tile_base, k_lane_start)
 *   row_base = b.mul(m_row, K)
 *   src_elem = b.add(row_base, k_base)
 *   frag_bytes = atom.a_per_lane
 *   chunks = (frag_bytes + DTLA_CHUNK - 1) // DTLA_CHUNK
 *   block_bytes = wave_size * DTLA_CHUNK
 *   slot_base_off = slot * chunks * block_bytes
 *   for c in range(chunks):
 *       chunk = min(DTLA_CHUNK, frag_bytes - c*DTLA_CHUNK)
 *       src = src_elem if c == 0 else b.add(src_elem, b.const_i32(c*DTLA_CHUNK))
 *       dst = b.smem_ptr_add(wave_lds_base, b.const_i64(slot_base_off + c*block_bytes))
 *       b.global_load_lds(A, src, dst, chunk, CACHE_ALL)
 * ===================================================================== */
void rocke_moe_fp8_xdtla_stage_a_fp8(rocke_moe_fp8_build_ctx_t* ctx,
                                     rocke_value_t* A,
                                     rocke_value_t* m_tile_base,
                                     rocke_value_t* k_tile_base,
                                     rocke_value_t* K,
                                     const rocke_tensor_view_t* stage_view,
                                     int slot,
                                     rocke_value_t* wave_lds_base,
                                     rocke_value_t* lane,
                                     int wave_size)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;
    const rocke_lane_decode_t* lane_decode = &ctx->lane_decode;
    (void)stage_view;
    (void)lane;

    rocke_value_t* m_row = rocke_b_add(b, m_tile_base, lane_decode->m_in_atom);
    rocke_value_t* k_lane_start
        = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->a_per_lane));
    rocke_value_t* k_base = rocke_b_add(b, k_tile_base, k_lane_start);
    rocke_value_t* row_base = rocke_b_mul(b, m_row, K);
    rocke_value_t* src_elem = rocke_b_add(b, row_base, k_base);

    int frag_bytes = atom->a_per_lane;
    int chunks = (frag_bytes + ROCKE_MOE_FP8_DTLA_CHUNK - 1) / ROCKE_MOE_FP8_DTLA_CHUNK;
    int block_bytes = wave_size * ROCKE_MOE_FP8_DTLA_CHUNK;
    int slot_base_off = slot * chunks * block_bytes;

    for(int c = 0; c < chunks; ++c)
    {
        int rem = frag_bytes - c * ROCKE_MOE_FP8_DTLA_CHUNK;
        int chunk = (ROCKE_MOE_FP8_DTLA_CHUNK < rem) ? ROCKE_MOE_FP8_DTLA_CHUNK : rem;
        rocke_value_t* src
            = (c == 0)
                  ? src_elem
                  : rocke_b_add(b, src_elem, rocke_b_const_i32(b, c * ROCKE_MOE_FP8_DTLA_CHUNK));
        rocke_value_t* dst = rocke_b_smem_ptr_add(
            b, wave_lds_base, rocke_b_const_i64(b, slot_base_off + c * block_bytes));
        rocke_b_global_load_lds(b, A, src, dst, chunk, ROCKE_CACHE_ALL);
    }
}

/* ===================================================================== *
 * _xdtla_read_a_fp8 (Python lines 1194-1216; _USE_X_DTLA path)
 *
 *   frag = atom.a_per_lane
 *   chunks = (frag + DTLA_CHUNK - 1) // DTLA_CHUNK
 *   acc = None
 *   for c in range(chunks):
 *       chunk = min(DTLA_CHUNK, frag - c*DTLA_CHUNK)
 *       row = b.add(warp_row_base,
 *                   b.add(b.const_i32((slot*chunks + c)*wave_size), lane))
 *       v = b.smem_load_vN(stage_view.base, row, b.const_i32(0), dtype=FP8E4M3, n=chunk)
 *       acc = v if acc is None else b.vec_concat(acc, v)
 *   return acc
 * ===================================================================== */
rocke_value_t* rocke_moe_fp8_xdtla_read_a_fp8(rocke_moe_fp8_build_ctx_t* ctx,
                                              const rocke_tensor_view_t* stage_view,
                                              int slot,
                                              rocke_value_t* lane,
                                              rocke_value_t* warp_row_base,
                                              int wave_size)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_mfma_atom_t* atom = ctx->atom;

    int frag = atom->a_per_lane;
    int chunks = (frag + ROCKE_MOE_FP8_DTLA_CHUNK - 1) / ROCKE_MOE_FP8_DTLA_CHUNK;
    rocke_value_t* acc = NULL;
    for(int c = 0; c < chunks; ++c)
    {
        int rem = frag - c * ROCKE_MOE_FP8_DTLA_CHUNK;
        int chunk = (ROCKE_MOE_FP8_DTLA_CHUNK < rem) ? ROCKE_MOE_FP8_DTLA_CHUNK : rem;
        rocke_value_t* row = rocke_b_add(
            b,
            warp_row_base,
            rocke_b_add(b, rocke_b_const_i32(b, (slot * chunks + c) * wave_size), lane));
        rocke_value_t* indices[2] = {row, rocke_b_const_i32(b, 0)};
        rocke_value_t* v
            = rocke_b_smem_load_vN(b, stage_view->base, indices, 2, rocke_fp8e4m3(), chunk);
        acc = (acc == NULL) ? v : rocke_b_vec_concat(b, acc, v);
    }
    return acc;
}
