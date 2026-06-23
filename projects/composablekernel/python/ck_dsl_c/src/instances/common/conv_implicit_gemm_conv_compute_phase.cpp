// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_compute_phase.c -- C99 port of the MFMA/WMMA
 * K-tile compute closures + small math helpers of build_implicit_gemm_conv
 * (ck_dsl/instances/common/conv_implicit_gemm.py).
 *
 * SCOPE (this TU only):
 *   _emit_mfma           (py 637-638)  -> ckc_conv_emit_mfma
 *   _emit_frag_smem_load (py 692-719)  -> ckc_conv_emit_frag_smem_load
 *   emit_wmma_phase      (py 1108-1162)-> ckc_conv_emit_wmma_phase
 *   emit_mfma_phase      (py 1164-1255)-> ckc_conv_emit_mfma_phase
 *
 * Every IR op is emitted in the byte-identical builder-call order of the Python
 * source. Peers (emit_smem_load, descriptors, loaders, k-loop, epilogue) are
 * declared in the internal header and resolved at link time.
 */

#include <stddef.h>

#include "ckc/instance_conv_implicit_gemm_internal.h"
#include "ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h" /* ckc_lane_decode_t, ckc_decode_mfma_lanes */

/* ====================================================================== *
 * _emit_mfma(b, atom, a, bv, c)  (py 637-638)
 *
 *   def _emit_mfma(b, atom, a, bv, c):
 *       return atom.emit(b, a, bv, c)
 *
 * MfmaAtom.emit dispatches to the ISA-named builder method keyed by the atom's
 * backend op_id. The atoms.h port does not expose atom.emit() as a symbol; the
 * faithful inline reproduction (mfma_gemm_inner.h:44) is
 *     emit -> ckc_b_mma(b, atom->name, a, b, c)
 * so the conv _emit_mfma is the same single mma() call by atom name.
 * ====================================================================== */
ckc_value_t* ckc_conv_emit_mfma(ckc_ir_builder_t* b,
                                const ckc_mfma_atom_t* atom,
                                ckc_value_t* a,
                                ckc_value_t* bv,
                                ckc_value_t* c)
{
    return ckc_b_mma(b, atom->name, a, bv, c, NULL, 0);
}

/* ====================================================================== *
 * _emit_frag_smem_load(b, src, mn_in_atom, k_in_atom, atom_mn_base,
 *                      k_tile_base, frag_len)  (py 692-719)
 *
 *   lds_row = b.add(atom_mn_base, mn_in_atom)
 *   lds_col = b.add(k_tile_base, k_in_atom)
 *   if frag_len <= 8:
 *       return _emit_smem_load(b, src, lds_row, lds_col, frag_len)
 *   frag = None
 *   for off in range(0, frag_len, 8):
 *       chunk = _emit_smem_load(b, src, lds_row, b.add(lds_col, b.const_i32(off)), 8)
 *       frag = chunk if frag is None else b.vec_concat(frag, chunk)
 *   return frag
 * ====================================================================== */
ckc_value_t* ckc_conv_emit_frag_smem_load(ckc_ir_builder_t* b,
                                          ckc_value_t* src,
                                          ckc_value_t* mn_in_atom,
                                          ckc_value_t* k_in_atom,
                                          ckc_value_t* atom_mn_base,
                                          ckc_value_t* k_tile_base,
                                          int frag_len)
{
    ckc_value_t* lds_row = ckc_b_add(b, atom_mn_base, mn_in_atom);
    ckc_value_t* lds_col = ckc_b_add(b, k_tile_base, k_in_atom);
    if(frag_len <= 8)
        return ckc_conv_emit_smem_load(b, src, lds_row, lds_col, frag_len);

    ckc_value_t* frag = NULL;
    for(int off = 0; off < frag_len; off += 8)
    {
        ckc_value_t* col   = ckc_b_add(b, lds_col, ckc_b_const_i32(b, off));
        ckc_value_t* chunk = ckc_conv_emit_smem_load(b, src, lds_row, col, 8);
        frag               = (frag == NULL) ? chunk : ckc_b_vec_concat(b, frag, chunk);
    }
    return frag;
}

/* ====================================================================== *
 * emit_wmma_phase(A_src, B_src, iter_vars) -> new_accs  (py 1108-1162)
 *
 * One K-tile of WMMA atoms, fully MMA-contract driven (gfx1151). Operand
 * fragments come from the op's A/B layout maps; the matmul is emitted
 * target-neutrally via b.mma(op, ...). Reads iter_vars (length ctx->num_accs);
 * writes the new accs into out_accs.
 * ====================================================================== */
void ckc_conv_emit_wmma_phase(ckc_conv_build_ctx_t* ctx,
                              ckc_value_t* A_src,
                              ckc_value_t* B_src,
                              ckc_value_t* const* iter_vars,
                              int num_iter_vars,
                              ckc_value_t** out_accs)
{
    ckc_ir_builder_t* b                       = ctx->b;
    const ckc_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const ckc_mmaop_t* op                     = ctx->op;

    /* a_map = op.a_layout(); b_map = op.b_layout() */
    const ckc_arch_layout_map_t* a_map = ckc_mmaop_a_layout(op, b);
    const ckc_arch_layout_map_t* b_map = ckc_mmaop_b_layout(op, b);

    /* a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
     * b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0) */
    ckc_value_t* a_row_in_atom = NULL;
    ckc_value_t* a_k_in_atom   = NULL;
    ckc_value_t* b_k_in_atom   = NULL;
    ckc_value_t* b_col_in_atom = NULL;
    ckc_arch_layout_map_coord(a_map, b, ctx->lane, 0, &a_row_in_atom, &a_k_in_atom);
    ckc_arch_layout_map_coord(b_map, b, ctx->lane, 0, &b_k_in_atom, &b_col_in_atom);

    ckc_value_t* warp_m_off = ckc_warp_grid_warp_m_off(b, &ctx->grid);
    ckc_value_t* warp_n_off = ckc_warp_grid_warp_n_off(b, &ctx->grid);

    /* new_accs = list(iter_vars) */
    for(int i = 0; i < num_iter_vars; ++i)
        out_accs[i] = iter_vars[i];

    /* fragment-row scratch (max mfmas_m / mfmas_n bounded by acc geometry). */
    ckc_value_t* a_rows[CKC_CONV_MAX_ACCS];
    ckc_value_t* b_cols[CKC_CONV_MAX_ACCS];

    for(int kk = 0; kk < ctx->k_atoms; ++kk)
    {
        ckc_value_t* k_tile_base = ckc_b_const_i32(b, kk * spec->warp_tile_k);

        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            ckc_value_t* atom_row =
                ckc_b_add(b, warp_m_off, ckc_b_const_i32(b, mi * spec->warp_tile_m));
            a_rows[mi] = ckc_conv_emit_frag_smem_load(
                b, A_src, a_row_in_atom, a_k_in_atom, atom_row, k_tile_base, ctx->a_per_lane);
        }

        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            ckc_value_t* atom_row =
                ckc_b_add(b, warp_n_off, ckc_b_const_i32(b, ni * spec->warp_tile_n));
            b_cols[ni] = ckc_conv_emit_frag_smem_load(
                b, B_src, b_col_in_atom, b_k_in_atom, atom_row, k_tile_base, ctx->b_per_lane);
        }

        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat] =
                    ckc_b_mma(b, op->op_id, a_rows[mi], b_cols[ni], out_accs[flat], NULL, 0);
                ++flat;
            }
        }
    }
}

/* ====================================================================== *
 * emit_mfma_phase(A_src, B_src, iter_vars) -> new_accs  (py 1164-1255)
 *
 * One K-tile worth of MFMAs across all per-warp atom positions. Delegates to
 * emit_wmma_phase on the WMMA family, otherwise decodes the lane, walks the
 * K-atom loop, honours the a_operand_override hook, and emits the compv3/compv4
 * sched_group_barrier hints at each kk step's tail.
 * ====================================================================== */
void ckc_conv_emit_mfma_phase(ckc_conv_build_ctx_t* ctx,
                              ckc_value_t* A_src,
                              ckc_value_t* B_src,
                              ckc_value_t* const* iter_vars,
                              int num_iter_vars,
                              ckc_value_t** out_accs)
{
    ckc_ir_builder_t* b                       = ctx->b;
    const ckc_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const ckc_mfma_atom_t* atom               = ctx->atom;

    /* if op.family == "wmma": return emit_wmma_phase(...) */
    if(ctx->is_wmma)
    {
        ckc_conv_emit_wmma_phase(ctx, A_src, B_src, iter_vars, num_iter_vars, out_accs);
        return;
    }

    /* decoded = decode_mfma_lanes(b, atom, lane) */
    ckc_lane_decode_t decoded = ckc_decode_mfma_lanes(b, atom, ctx->lane);
    ckc_value_t* m_in_atom    = decoded.m_in_atom;
    ckc_value_t* n_in_atom    = decoded.n_in_atom;
    ckc_value_t* k_blk        = decoded.k_blk;

    ckc_value_t* warp_m_off = ckc_warp_grid_warp_m_off(b, &ctx->grid);
    ckc_value_t* warp_n_off = ckc_warp_grid_warp_n_off(b, &ctx->grid);

    /* new_accs = list(iter_vars) */
    for(int i = 0; i < num_iter_vars; ++i)
        out_accs[i] = iter_vars[i];

    ckc_value_t* a_rows[CKC_CONV_MAX_ACCS];
    ckc_value_t* b_cols[CKC_CONV_MAX_ACCS];

    /* a_operand_override hook + its opaque user (Python `a_operand_override`). */
    const ckc_conv_build_overrides_t* ov = ctx->ov;

    for(int kk = 0; kk < ctx->k_atoms; ++kk)
    {
        /* col_base = b.add(b.mul(k_blk, const(a_per_lane)),
         *                  const(kk * warp_tile_k))
         *
         * Python evaluates the builder calls strictly left-to-right: the inner
         * b.const_i32(a_per_lane) and b.mul(...) run (consuming two SSA ids)
         * before b.const_i32(kk*warp_tile_k). C argument evaluation order is
         * unspecified, so the operands must be sequenced into temporaries to
         * keep the SSA numbering byte-identical with the Python emitter. */
        ckc_value_t* col_mul  = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, ctx->a_per_lane));
        ckc_value_t* col_off  = ckc_b_const_i32(b, kk * spec->warp_tile_k);
        ckc_value_t* col_base = ckc_b_add(b, col_mul, col_off);

        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            /* a_row = warp_m_off + (mi*warp_tile_m + m_in_atom) */
            ckc_value_t* a_row = ckc_b_add(
                b, warp_m_off, ckc_b_add(b, ckc_b_const_i32(b, mi * spec->warp_tile_m), m_in_atom));
            if(ov != NULL && ov->a_operand_override != NULL)
            {
                a_rows[mi] = ov->a_operand_override(b,
                                                    spec,
                                                    a_row,
                                                    ctx->k_off_capture,
                                                    col_base,
                                                    ctx->a_per_lane,
                                                    &ctx->grid,
                                                    ctx->input_cache_context,
                                                    ov->user);
            }
            else
            {
                a_rows[mi] = ckc_conv_emit_smem_load(b, A_src, a_row, col_base, ctx->a_per_lane);
            }
        }

        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            /* b_row = warp_n_off + (ni*warp_tile_n + n_in_atom) */
            ckc_value_t* b_row = ckc_b_add(
                b, warp_n_off, ckc_b_add(b, ckc_b_const_i32(b, ni * spec->warp_tile_n), n_in_atom));
            b_cols[ni] = ckc_conv_emit_smem_load(b, B_src, b_row, col_base, ctx->b_per_lane);
        }

        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat] =
                    ckc_conv_emit_mfma(b, atom, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        }

        /* schedule.emit_after_mfma_step(b, ds_read_count=mfmas_m + mfmas_n,
         *                               mfma_count=mfmas_m * mfmas_n) */
        ckc_schedule_policy_emit_after_mfma_step(
            &ctx->schedule, b, ctx->mfmas_m + ctx->mfmas_n, ctx->mfmas_m * ctx->mfmas_n);
    }
}
