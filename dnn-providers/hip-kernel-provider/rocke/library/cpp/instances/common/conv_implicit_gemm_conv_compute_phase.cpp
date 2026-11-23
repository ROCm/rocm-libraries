// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_compute_phase.c -- C99 port of the MFMA/WMMA
 * K-tile compute closures + small math helpers of build_implicit_gemm_conv
 * (rocke/instances/common/conv_implicit_gemm.py).
 *
 * SCOPE (this TU only):
 *   _emit_mfma           (py 637-638)  -> rocke_conv_emit_mfma
 *   _emit_frag_smem_load (py 692-719)  -> rocke_conv_emit_frag_smem_load
 *   emit_wmma_phase      (py 1108-1162)-> rocke_conv_emit_wmma_phase
 *   emit_mfma_phase      (py 1164-1255)-> rocke_conv_emit_mfma_phase
 *
 * Every IR op is emitted in the byte-identical builder-call order of the Python
 * source. Peers (emit_smem_load, descriptors, loaders, k-loop, epilogue) are
 * declared in the internal header and resolved at link time.
 */

#include <stddef.h>
#include <string.h> /* strcmp -- operand dtype string -> transpose-read elem type */

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h" /* rocke_lane_decode_t, rocke_decode_mfma_lanes */
#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/instance_conv_implicit_gemm_internal.h"
#include "rocke/instance_conv_layouts.h" /* rocke_conv_tr_lane_offsets32 */
#include "rocke/ir_internal.h"

/* ====================================================================== *
 * _emit_mfma(b, atom, a, bv, c)  (py 637-638)
 *
 *   def _emit_mfma(b, atom, a, bv, c):
 *       return atom.emit(b, a, bv, c)
 *
 * MfmaAtom.emit dispatches to the ISA-named builder method keyed by the atom's
 * backend op_id. The atoms.h port does not expose atom.emit() as a symbol; the
 * faithful inline reproduction (mfma_gemm_inner.h:44) is
 *     emit -> rocke_b_mma(b, atom->name, a, b, c)
 * so the conv _emit_mfma is the same single mma() call by atom name.
 * ====================================================================== */
rocke_value_t* rocke_conv_emit_mfma(rocke_ir_builder_t* b,
                                    const rocke_mfma_atom_t* atom,
                                    rocke_value_t* a,
                                    rocke_value_t* bv,
                                    rocke_value_t* c)
{
    return rocke_b_mma(b, atom->name, a, bv, c, NULL, 0);
}

/* ====================================================================== *
 * _emit_frag_smem_load(b, src, mn_in_atom, k_in_atom, atom_mn_base,
 *                      k_tile_base, frag_len)  (py 692-719)
 *
 *   lds_row = b.add(atom_mn_base, mn_in_atom)
 *   lds_col = b.add(k_tile_base, k_in_atom)
 *   if frag_len <= 8:
 *       return _emit_smem_load(b, src, lds_row, lds_col, frag_len, smem_dtype=smem_dtype)
 *   frag = None
 *   for off in range(0, frag_len, 8):
 *       chunk = _emit_smem_load(b, src, lds_row, b.add(lds_col, b.const_i32(off)), 8,
 *                               smem_dtype=smem_dtype)
 *       frag = chunk if frag is None else b.vec_concat(frag, chunk)
 *   return frag
 * ====================================================================== */

/* K-outer transpose-read fragment feed (wgrad's lds_k_outer path).
 *
 * For a K-outer tile T[k][mn] the MFMA operand of lane l is n/4
 * ds_read_b64_tr_b16 at
 *     row(r) = k_base  + (l // MN)*n + ((l % 16)//4) + 4*r    r in [0, n/4)
 *     col    = mn_base + ((l % MN)//16)*16 + (l % 4)*4
 * after which lane l holds T[k_base .. k_base+n-1][mn_base + l % MN].
 *
 * n is the per-lane operand length, which is what sets the k-stride between
 * lane groups: MFMA lane l owns k = (l // MN)*n .. +n-1. It is 8 for 32x32x16
 * and 16x16x32, and 4 for 16x16x16. Hardcoding the stride at 8 made the
 * 16x16x16 atom read k rows 8..27 of a 16-row tile -- past the end of the
 * K-outer tile. The emitted IR is unchanged for the two 8-element atoms.
 * Mirrors _tr_frag() in conv_implicit_gemm_wgrad.py, a thin binding of
 * ConvKOuterFragmentReader.fragment (library/kernels/common/_conv_layouts.py). */
/* Element type for the K-outer transpose read, from the operand dtype string.
 * Mirrors Python's
 *   _smem_dtype = BF16 if a_dtype == "bf16" else F32 if a_dtype == "fp32" else None
 *   tr_dtype    = _smem_dtype if _smem_dtype is not None else F16
 * Never returns NULL: rocke_b_ds_read_tr16_b128 defaults a NULL dtype to f16,
 * and the gfx1250 opcode is element-typed, so a bf16 kernel would silently
 * select .v8f16 and feed half fragments to a bf16 WMMA. The wave64
 * ds_read_b64_tr_b16 is type-agnostic, which is why gfx950 parity never caught
 * a NULL here. */
const rocke_type_t* rocke_conv_tr_elem_dtype(const char* a_dtype)
{
    if(a_dtype != NULL && strcmp(a_dtype, "bf16") == 0)
        return rocke_bf16();
    if(a_dtype != NULL && strcmp(a_dtype, "fp32") == 0)
        return rocke_f32();
    return rocke_f16();
}

rocke_value_t* rocke_conv_tr_frag(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  rocke_value_t* tr_lane_mod4,
                                  rocke_value_t* tr_grp16,
                                  rocke_value_t* smem,
                                  rocke_value_t* mn_base,
                                  rocke_value_t* k_base,
                                  int mn_atom,
                                  int n,
                                  int wave_size,
                                  const rocke_type_t* dtype)
{
    /* wave32 (gfx1250 WMMA 16x16x32): the RESULT layout is lane l owning column
     * l % 16 and K-half l // 16 -- but that is what the lane must end up
     * holding, not the address it supplies. ds_load_tr16_b128 transposes an 8x8
     * element block within each group of 8 lanes: the 8 lanes of a group each
     * read 8 contiguous elements, and lane j of the group receives element j
     * from all 8 of those runs. So the group addresses the 8-column block
     * containing l % 16, and lane l supplies the (l % 8)-th K row of the run:
     *     col  = mn_base + ((l % 16) / 8) * 8
     *     row0 = k_base  + (l / 16) * n + (l % 8)
     * Verified on silicon. Mirrors the wave32 branch of
     * _BoundConvKOuterFragmentReader.fragment() in Python; the lane map itself
     * is the shared rocke_conv_tr_lane_offsets32. */
    if(wave_size == 32)
    {
        rocke_value_t* col32;
        rocke_value_t* row032;
        rocke_conv_tr_lane_offsets32(b, lane, n, mn_base, k_base, &col32, &row032);
        rocke_value_t* out32 = NULL;
        for(int r = 0; r < n / 8; ++r)
        {
            rocke_value_t* row = rocke_b_add(b, row032, rocke_b_const_i32(b, 8 * r));
            rocke_value_t* idx32[2];
            idx32[0] = row;
            idx32[1] = col32;
            rocke_value_t* part = rocke_b_ds_read_tr16_b128(b, smem, idx32, 2, dtype);
            out32 = (out32 == NULL) ? part : rocke_b_vec_concat(b, out32, part);
        }
        return out32;
    }
    /* Every operand is sequenced into a temporary: Python evaluates these
     * builder calls strictly left-to-right (innermost first) and C argument
     * evaluation order is unspecified, so nesting them would drift the SSA ids. */
    rocke_value_t* c_mn = rocke_b_const_i32(b, mn_atom);

    /* col = mn_base + (((lane % MN) / 16) * 16 + tr_lane_mod4) */
    rocke_value_t* lane_mod_mn = rocke_b_mod(b, lane, c_mn);
    rocke_value_t* c16a = rocke_b_const_i32(b, 16);
    rocke_value_t* grp = rocke_b_div(b, lane_mod_mn, c16a);
    rocke_value_t* c16b = rocke_b_const_i32(b, 16);
    rocke_value_t* col_mul = rocke_b_mul(b, grp, c16b);
    rocke_value_t* col_inner = rocke_b_add(b, col_mul, tr_lane_mod4);
    rocke_value_t* col = rocke_b_add(b, mn_base, col_inner);

    /* row0 = k_base + ((lane / MN) * n + tr_grp16) */
    rocke_value_t* lane_div_mn = rocke_b_div(b, lane, c_mn);
    rocke_value_t* c_n = rocke_b_const_i32(b, n);
    rocke_value_t* row_mul = rocke_b_mul(b, lane_div_mn, c_n);
    rocke_value_t* row_inner = rocke_b_add(b, row_mul, tr_grp16);
    rocke_value_t* row0 = rocke_b_add(b, k_base, row_inner);

    rocke_value_t* out = NULL;
    for(int r = 0; r < n / 4; ++r)
    {
        rocke_value_t* row = rocke_b_add(b, row0, rocke_b_const_i32(b, 4 * r));
        rocke_value_t* idx[2];
        idx[0] = row;
        idx[1] = col;
        rocke_value_t* part = rocke_b_ds_read_tr16_b64(b, smem, idx, 2, dtype);
        out = (out == NULL) ? part : rocke_b_vec_concat(b, out, part);
    }
    return out;
}

rocke_value_t* rocke_conv_emit_frag_smem_load(rocke_ir_builder_t* b,
                                              rocke_value_t* src,
                                              rocke_value_t* mn_in_atom,
                                              rocke_value_t* k_in_atom,
                                              rocke_value_t* atom_mn_base,
                                              rocke_value_t* k_tile_base,
                                              int frag_len,
                                              const rocke_type_t* smem_dtype)
{
    rocke_value_t* lds_row = rocke_b_add(b, atom_mn_base, mn_in_atom);
    rocke_value_t* lds_col = rocke_b_add(b, k_tile_base, k_in_atom);
    if(frag_len <= 8)
        return rocke_conv_emit_smem_load(b, src, lds_row, lds_col, frag_len, smem_dtype);

    rocke_value_t* frag = NULL;
    for(int off = 0; off < frag_len; off += 8)
    {
        rocke_value_t* col = rocke_b_add(b, lds_col, rocke_b_const_i32(b, off));
        rocke_value_t* chunk = rocke_conv_emit_smem_load(b, src, lds_row, col, 8, smem_dtype);
        frag = (frag == NULL) ? chunk : rocke_b_vec_concat(b, frag, chunk);
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
void rocke_conv_emit_wmma_phase(rocke_conv_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const rocke_mmaop_t* op = ctx->op;

    /* a_map = op.a_layout(); b_map = op.b_layout() */
    const rocke_arch_layout_map_t* a_map = rocke_mmaop_a_layout(op, b);
    const rocke_arch_layout_map_t* b_map = rocke_mmaop_b_layout(op, b);

    /* a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
     * b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0) */
    rocke_value_t* a_row_in_atom = NULL;
    rocke_value_t* a_k_in_atom = NULL;
    rocke_value_t* b_k_in_atom = NULL;
    rocke_value_t* b_col_in_atom = NULL;
    rocke_arch_layout_map_coord(a_map, b, ctx->lane, 0, &a_row_in_atom, &a_k_in_atom);
    rocke_arch_layout_map_coord(b_map, b, ctx->lane, 0, &b_k_in_atom, &b_col_in_atom);

    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, &ctx->grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, &ctx->grid);

    /* new_accs = list(iter_vars) */
    for(int i = 0; i < num_iter_vars; ++i)
        out_accs[i] = iter_vars[i];

    /* fragment-row scratch (max mfmas_m / mfmas_n bounded by acc geometry). */
    rocke_value_t* a_rows[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* b_cols[ROCKE_CONV_MAX_ACCS];

    for(int kk = 0; kk < ctx->k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * spec->warp_tile_k);

        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));
            if(ctx->lds_k_outer)
            {
                /* Python skips the frag load entirely on this path. */
                a_rows[mi] = rocke_conv_tr_frag(b,
                                                ctx->lane,
                                                ctx->tr_lane_mod4,
                                                ctx->tr_grp16,
                                                A_src,
                                                atom_row,
                                                k_tile_base,
                                                spec->warp_tile_m,
                                                ctx->a_per_lane,
                                                spec->wave_size,
                                                ctx->tr_dtype);
                continue;
            }
            a_rows[mi] = rocke_conv_emit_frag_smem_load(b,
                                                        A_src,
                                                        a_row_in_atom,
                                                        a_k_in_atom,
                                                        atom_row,
                                                        k_tile_base,
                                                        ctx->a_per_lane,
                                                        ctx->smem_dtype);
        }

        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));
            if(ctx->lds_k_outer)
            {
                b_cols[ni] = rocke_conv_tr_frag(b,
                                                ctx->lane,
                                                ctx->tr_lane_mod4,
                                                ctx->tr_grp16,
                                                B_src,
                                                atom_row,
                                                k_tile_base,
                                                spec->warp_tile_n,
                                                ctx->b_per_lane,
                                                spec->wave_size,
                                                ctx->tr_dtype);
                continue;
            }
            b_cols[ni] = rocke_conv_emit_frag_smem_load(b,
                                                        B_src,
                                                        b_col_in_atom,
                                                        b_k_in_atom,
                                                        atom_row,
                                                        k_tile_base,
                                                        ctx->b_per_lane,
                                                        ctx->smem_dtype);
        }

        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat]
                    = rocke_b_mma(b, op->op_id, a_rows[mi], b_cols[ni], out_accs[flat], NULL, 0);
                ++flat;
            }
        }
    }
}

/* Python _ir_dtype(spec.data.dtype_a): the vec_pack element type of the
 * k-packed fragment halves. */
static const rocke_type_t* conv_ir_dtype_a(const rocke_implicit_gemm_conv_spec_t* spec)
{
    if(spec->dtype_a != NULL && strcmp(spec->dtype_a, "bf16") == 0)
        return rocke_bf16();
    if(spec->dtype_a != NULL && strcmp(spec->dtype_a, "fp32") == 0)
        return rocke_f32();
    return rocke_f16();
}

/* Python lds_swz: the lds_xor column of tile element (row, col) -- its 16-byte
 * chunk XOR the row key (the in-chunk offset is kept). The key is the row's
 * 256-byte line (128-byte span under lds_kpack_read; the row itself under
 * lds_xor_rowkey). */
rocke_value_t*
    rocke_conv_lds_swz(rocke_conv_build_ctx_t* ctx, rocke_value_t* row, rocke_value_t* col)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const int block_k = ctx->block_k;
    int sh;
    if(spec->lds_kpack_read)
        sh = spec->lds_xor_rowkey ? 0 : (block_k == 32 ? 1 : 0);
    else
        sh = block_k == 32 ? 2 : (block_k == 64 ? 1 : 0);
    /* The shared swizzle without its bit-3 flip term (flip_shift -1). */
    const int col_xor[4] = {sh, block_k / 8 - 1, 3, -1};
    return rocke_swizzle_lds_col(b, row, col, col_xor);
}

/* ====================================================================== *
 * emit_mfma_phase(A_src, B_src, iter_vars) -> new_accs  (py 1164-1255)
 *
 * One K-tile worth of MFMAs across all per-warp atom positions. Delegates to
 * emit_wmma_phase on the WMMA family, otherwise decodes the lane, walks the
 * K-atom loop, honours the a_operand_override hook, and emits the compv3/compv4
 * sched_group_barrier hints at each kk step's tail.
 * ====================================================================== */
/* Python _xa (xh_stride_m): the LDS (row, col) of x-halo A element (row,
 * col): row bit 0 flipped by row bit 2 and the 16-byte chunk XOR (row/4) % 4
 * (+ the xh_yring region base row). */
void rocke_conv_xa(rocke_conv_build_ctx_t* ctx,
                   rocke_value_t* row,
                   rocke_value_t* col,
                   rocke_value_t** prow_out,
                   rocke_value_t** col_out)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* hi = rocke_b_lshr(b, row, rocke_b_const_i32(b, 2));
    rocke_value_t* prow = rocke_b_xor(b, row, rocke_b_land(b, hi, rocke_b_const_i32(b, 1)));
    rocke_value_t* key = rocke_b_land(b, hi, rocke_b_const_i32(b, 3));
    if(ctx->xa_ring_off != NULL)
        prow = rocke_b_add(b, prow, ctx->xa_ring_off);
    *prow_out = prow;
    *col_out = rocke_b_xor(b, col, rocke_b_shl(b, key, rocke_b_const_i32(b, 3)));
}

/* Python emit_mfma_phase's load_frags(kk): the A rows and B columns one kk
 * step of MFMAs consumes (k-packed, lds_xor and the rpf_pingpong partial
 * reads included). */
typedef struct
{
    rocke_conv_build_ctx_t* ctx;
    rocke_value_t* A_src;
    rocke_value_t* B_src;
    rocke_value_t* m_in_atom;
    rocke_value_t* n_in_atom;
    rocke_value_t* k_blk;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    bool kprd;
    const rocke_type_t* ir_dtype_a;
    /* _kp_cache[kk + 1]: the odd kk step's halves. */
    rocke_value_t* kp_a[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* kp_b[ROCKE_CONV_MAX_ACCS];
    int kp_na;
    int kp_nb;
} conv_frag_env_t;

static void conv_load_frags(
    conv_frag_env_t* e, int kk, rocke_value_t** a_rows, int* na, rocke_value_t** b_cols, int* nb)
{
    rocke_conv_build_ctx_t* ctx = e->ctx;
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const rocke_conv_build_overrides_t* ov = ctx->ov;
    const bool has_ov = ov != NULL && ov->a_operand_override != NULL;
    *na = 0;
    *nb = 0;
    if(e->kprd)
    {
        if(kk % 2)
        {
            for(int i = 0; i < e->kp_na; ++i)
                a_rows[i] = e->kp_a[i];
            for(int i = 0; i < e->kp_nb; ++i)
                b_cols[i] = e->kp_b[i];
            *na = e->kp_na;
            *nb = e->kp_nb;
            return;
        }
        /* k-packed: lane group k_blk owns 2*per_lane contiguous k of the kk
         * pair (2p, 2p+1); one 16-byte read, split in halves. */
        rocke_value_t* cb_mul = rocke_b_mul(b, e->k_blk, rocke_b_const_i32(b, 2 * ctx->a_per_lane));
        rocke_value_t* cb_off = rocke_b_const_i32(b, kk * spec->warp_tile_k);
        rocke_value_t* cb = rocke_b_add(b, cb_mul, cb_off);
        const bool skip_a = ctx->pp_b_only;
        const bool skip_b = ctx->pp_b_from_p1 > 0 && kk >= ctx->pp_b_from_p1 - 1;
        rocke_value_t* pa[ROCKE_CONV_MAX_ACCS];
        rocke_value_t* pb[ROCKE_CONV_MAX_ACCS];
        int npa = 0, npb = 0;
        for(int mi = 0; mi < (skip_a ? 0 : ctx->mfmas_m); ++mi)
        {
            if(ctx->xh_prev != NULL && mi < ctx->mfmas_m - 1)
            {
                /* xh_stride_m: the previous tap's fragment mi + 1 is this one. */
                pa[npa++] = ctx->xh_prev[kk][mi + 1];
                continue;
            }
            rocke_value_t* a_row;
            if(ctx->xh_a_rows != NULL)
                a_row = ctx->xh_a_rows[mi];
            else
            {
                rocke_value_t* mc = rocke_b_const_i32(b, mi * spec->warp_tile_m);
                a_row = rocke_b_add(b, e->warp_m_off, rocke_b_add(b, mc, e->m_in_atom));
            }
            if(spec->xh_stride_m && ctx->xh_a_rows != NULL)
            {
                rocke_value_t *pr, *pc;
                rocke_conv_xa(ctx, a_row, cb, &pr, &pc);
                pa[npa++] = rocke_conv_emit_smem_load(
                    b, e->A_src, pr, pc, 2 * ctx->a_per_lane, ctx->smem_dtype);
                continue;
            }
            rocke_value_t* col = spec->lds_xor ? rocke_conv_lds_swz(ctx, a_row, cb) : cb;
            pa[npa++] = rocke_conv_emit_smem_load(
                b, e->A_src, a_row, col, 2 * ctx->a_per_lane, ctx->smem_dtype);
        }
        if(ctx->xh_rec != NULL)
            for(int i = 0; i < npa; ++i)
                ctx->xh_rec[kk][i] = pa[i];
        for(int ni = 0; ni < (skip_b ? 0 : ctx->mfmas_n); ++ni)
        {
            rocke_value_t* nc = rocke_b_const_i32(b, ni * spec->warp_tile_n + ctx->xh_b_row_off);
            rocke_value_t* b_row = rocke_b_add(b, e->warp_n_off, rocke_b_add(b, nc, e->n_in_atom));
            rocke_value_t* col = spec->lds_xor ? rocke_conv_lds_swz(ctx, b_row, cb) : cb;
            pb[npb++] = rocke_conv_emit_smem_load(
                b, e->B_src, b_row, col, 2 * ctx->b_per_lane, ctx->smem_dtype);
        }
        /* half(v, n, h) = vec_pack([vec_extract(v, h*n + i)], ir_dtype_a): the
         * cached odd halves first (A then B), then the even ones. */
        for(int h = 1; h >= 0; --h)
        {
            for(int mi = 0; mi < npa; ++mi)
            {
                rocke_value_t* el[16];
                for(int i = 0; i < ctx->a_per_lane; ++i)
                    el[i] = rocke_b_vec_extract(b, pa[mi], h * ctx->a_per_lane + i);
                rocke_value_t* v = rocke_b_vec_pack(b, el, ctx->a_per_lane, e->ir_dtype_a);
                if(h)
                    e->kp_a[mi] = v;
                else
                    a_rows[mi] = v;
            }
            for(int ni = 0; ni < npb; ++ni)
            {
                rocke_value_t* el[16];
                for(int i = 0; i < ctx->b_per_lane; ++i)
                    el[i] = rocke_b_vec_extract(b, pb[ni], h * ctx->b_per_lane + i);
                rocke_value_t* v = rocke_b_vec_pack(b, el, ctx->b_per_lane, e->ir_dtype_a);
                if(h)
                    e->kp_b[ni] = v;
                else
                    b_cols[ni] = v;
            }
        }
        e->kp_na = npa;
        e->kp_nb = npb;
        *na = npa;
        *nb = npb;
        return;
    }

    /* col_base = b.add(b.mul(k_blk, const(a_per_lane)), const(kk * warp_tile_k))
     *
     * Python evaluates the builder calls strictly left-to-right: the inner
     * b.const_i32(a_per_lane) and b.mul(...) run (consuming two SSA ids)
     * before b.const_i32(kk*warp_tile_k). C argument evaluation order is
     * unspecified, so the operands must be sequenced into temporaries to keep
     * the SSA numbering byte-identical with the Python emitter. */
    rocke_value_t* col_mul = rocke_b_mul(b, e->k_blk, rocke_b_const_i32(b, ctx->a_per_lane));
    rocke_value_t* col_off = rocke_b_const_i32(b, kk * spec->warp_tile_k);
    rocke_value_t* col_base = rocke_b_add(b, col_mul, col_off);
    /* pp_b_double without k-packed reads: [mem B] reads the B columns only
     * (b_only); [mem A] the B columns of kk < b_from only. */
    const bool np_skip_a = ctx->pp_b_only && !has_ov;
    const bool np_skip_b = ctx->pp_b_from_p1 > 0 && kk >= ctx->pp_b_from_p1 - 1;

    for(int mi = 0; mi < (np_skip_a ? 0 : ctx->mfmas_m); ++mi)
    {
        if(ctx->lds_k_outer)
        {
            /* Python skips the a_row computation entirely on this path (it
             * `continue`s before it), so emitting it here would add ops the
             * Python engine never emits. Operands sequenced into temporaries to
             * preserve left-to-right evaluation. */
            rocke_value_t* mn_c = rocke_b_const_i32(b, mi * spec->warp_tile_m);
            rocke_value_t* mn_base = rocke_b_add(b, e->warp_m_off, mn_c);
            rocke_value_t* k_c = rocke_b_const_i32(b, kk * spec->warp_tile_k);
            a_rows[(*na)++] = rocke_conv_tr_frag(b,
                                                 ctx->lane,
                                                 ctx->tr_lane_mod4,
                                                 ctx->tr_grp16,
                                                 e->A_src,
                                                 mn_base,
                                                 k_c,
                                                 spec->warp_tile_m,
                                                 ctx->a_per_lane,
                                                 spec->wave_size,
                                                 ctx->tr_dtype);
            continue;
        }
        /* a_row = warp_m_off + (mi*warp_tile_m + m_in_atom) */
        rocke_value_t* a_row;
        if(ctx->xh_a_rows != NULL)
            a_row = ctx->xh_a_rows[mi];
        else
            a_row = rocke_b_add(
                b,
                e->warp_m_off,
                rocke_b_add(b, rocke_b_const_i32(b, mi * spec->warp_tile_m), e->m_in_atom));
        if(!has_ov && ctx->xh_prev != NULL && mi < ctx->mfmas_m - 1)
        {
            a_rows[(*na)++] = ctx->xh_prev[kk][mi + 1];
        }
        else if(!has_ov && spec->xh_stride_m && ctx->xh_a_rows != NULL)
        {
            rocke_value_t *pr, *pc;
            rocke_conv_xa(ctx, a_row, col_base, &pr, &pc);
            a_rows[(*na)++]
                = rocke_conv_emit_smem_load(b, e->A_src, pr, pc, ctx->a_per_lane, ctx->smem_dtype);
        }
        else if(has_ov)
        {
            a_rows[(*na)++] = ov->a_operand_override(b,
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
            rocke_value_t* a_col
                = spec->lds_xor ? rocke_conv_lds_swz(ctx, a_row, col_base) : col_base;
            a_rows[(*na)++] = rocke_conv_load_mma_operand(
                b, e->A_src, a_row, a_col, ctx->a_per_lane, ctx->smem_dtype, ctx->is_tf32);
        }
    }

    if(ctx->xh_rec != NULL)
        for(int i = 0; i < *na; ++i)
            ctx->xh_rec[kk][i] = a_rows[i];

    for(int ni = 0; ni < (np_skip_b ? 0 : ctx->mfmas_n); ++ni)
    {
        if(ctx->lds_k_outer)
        {
            rocke_value_t* mn_c = rocke_b_const_i32(b, ni * spec->warp_tile_n);
            rocke_value_t* mn_base = rocke_b_add(b, e->warp_n_off, mn_c);
            rocke_value_t* k_c = rocke_b_const_i32(b, kk * spec->warp_tile_k);
            b_cols[(*nb)++] = rocke_conv_tr_frag(b,
                                                 ctx->lane,
                                                 ctx->tr_lane_mod4,
                                                 ctx->tr_grp16,
                                                 e->B_src,
                                                 mn_base,
                                                 k_c,
                                                 spec->warp_tile_n,
                                                 ctx->b_per_lane,
                                                 spec->wave_size,
                                                 ctx->tr_dtype);
            continue;
        }
        /* b_row = warp_n_off + (ni*warp_tile_n + b_row_off + n_in_atom) */
        rocke_value_t* b_row = rocke_b_add(
            b,
            e->warp_n_off,
            rocke_b_add(
                b, rocke_b_const_i32(b, ni * spec->warp_tile_n + ctx->xh_b_row_off), e->n_in_atom));
        rocke_value_t* b_col = spec->lds_xor ? rocke_conv_lds_swz(ctx, b_row, col_base) : col_base;
        b_cols[(*nb)++] = rocke_conv_load_mma_operand(
            b, e->B_src, b_row, b_col, ctx->b_per_lane, ctx->smem_dtype, ctx->is_tf32);
    }
}

void rocke_conv_emit_mfma_phase(rocke_conv_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const rocke_mfma_atom_t* atom = ctx->atom;

    /* if op.family == "wmma": return emit_wmma_phase(...) */
    if(ctx->is_wmma)
    {
        rocke_conv_emit_wmma_phase(ctx, A_src, B_src, iter_vars, num_iter_vars, out_accs);
        return;
    }

    /* decoded = decode_mfma_lanes(b, atom, lane) */
    rocke_lane_decode_t decoded = rocke_decode_mfma_lanes(b, atom, ctx->lane);

    conv_frag_env_t* e = (conv_frag_env_t*)rocke_arena_alloc(&b->arena, sizeof(conv_frag_env_t));
    if(e == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "conv mfma phase: arena alloc failed");
        return;
    }
    memset(e, 0, sizeof(*e));
    e->ctx = ctx;
    e->A_src = A_src;
    e->B_src = B_src;
    e->m_in_atom = decoded.m_in_atom;
    e->n_in_atom = decoded.n_in_atom;
    e->k_blk = decoded.k_blk;
    e->warp_m_off = rocke_warp_grid_warp_m_off(b, &ctx->grid);
    e->warp_n_off = rocke_warp_grid_warp_n_off(b, &ctx->grid);
    if(ctx->wg_a_row_off != NULL)
        e->warp_m_off = rocke_b_add(b, e->warp_m_off, ctx->wg_a_row_off);
    if(ctx->wg_b_row_off != NULL)
        e->warp_n_off = rocke_b_add(b, e->warp_n_off, ctx->wg_b_row_off);
    const rocke_conv_build_overrides_t* ov = ctx->ov;
    e->kprd = spec->lds_kpack_read && !(ov != NULL && ov->a_operand_override != NULL);
    e->ir_dtype_a = conv_ir_dtype_a(spec);

    /* new_accs = list(iter_vars) */
    for(int i = 0; i < num_iter_vars; ++i)
        out_accs[i] = iter_vars[i];

    if(ctx->pp_read != NULL)
    {
        /* rpf_pingpong [mem A]: every fragment of the tile, no MFMA. */
        for(int kk = ctx->pp_kk_from; kk < ctx->k_atoms; ++kk)
        {
            rocke_conv_frag_t* f = &ctx->pp_read[ctx->pp_read_n++];
            conv_load_frags(e, kk, f->a, &f->na, f->b, &f->nb);
        }
        return;
    }
    if(ctx->pp_use != NULL)
    {
        /* rpf_pingpong [compute A/B]: MFMAs [i0, i1) of the tile, kk-major. */
        const int per_kk = ctx->mfmas_m * ctx->mfmas_n;
        for(int i = ctx->pp_i0; i < ctx->pp_i1; ++i)
        {
            const int kk = i / per_kk;
            const int flat = i % per_kk;
            const int mi = flat / ctx->mfmas_n;
            const int ni = flat % ctx->mfmas_n;
            const rocke_conv_frag_t* f = &ctx->pp_use[kk];
            out_accs[flat] = rocke_conv_emit_mfma(b, atom, f->a[mi], f->b[ni], out_accs[flat]);
        }
        return;
    }

    /* lds_ahead: step kk+1's fragment reads issue before step kk's MFMAs. */
    const bool lds_ahead = spec->lds_ahead && !(ov != NULL && ov->a_operand_override != NULL);
    rocke_conv_frag_t* fr2
        = (rocke_conv_frag_t*)rocke_arena_alloc(&b->arena, 2 * sizeof(rocke_conv_frag_t));
    if(fr2 == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "conv mfma phase: arena alloc failed");
        return;
    }
    int cur = 0;
    if(lds_ahead)
        conv_load_frags(e, 0, fr2[0].a, &fr2[0].na, fr2[0].b, &fr2[0].nb);
    for(int kk = 0; kk < ctx->k_atoms; ++kk)
    {
        rocke_value_t** a_rows;
        rocke_value_t** b_cols;
        if(lds_ahead)
        {
            a_rows = fr2[cur].a;
            b_cols = fr2[cur].b;
            if(kk + 1 < ctx->k_atoms)
            {
                rocke_conv_frag_t* nx = &fr2[cur ^ 1];
                conv_load_frags(e, kk + 1, nx->a, &nx->na, nx->b, &nx->nb);
            }
            cur ^= 1;
        }
        else
        {
            conv_load_frags(e, kk, fr2[0].a, &fr2[0].na, fr2[0].b, &fr2[0].nb);
            a_rows = fr2[0].a;
            b_cols = fr2[0].b;
        }
        if(kk == 0 && ctx->after_first_reads != NULL)
        {
            /* rpf_fetch_late: the next tile's global reads go here, behind the
             * first LDS fragment reads and ahead of every MFMA. */
            rocke_b_sched_barrier(b, 0);
            ctx->after_first_reads(ctx->after_first_reads_user);
        }
        if(ctx->defer_on && kk >= ctx->k_atoms - ctx->defer_n)
        {
            /* defer_mfma: the fragments are read now; their MFMAs are emitted
             * by the caller after the barrier, interleaved with the next
             * tile's LDS store. */
            rocke_conv_frag_t* d = &ctx->defer_frags[ctx->defer_cnt++];
            for(int i = 0; i < ctx->mfmas_m; ++i)
                d->a[i] = a_rows[i];
            for(int i = 0; i < ctx->mfmas_n; ++i)
                d->b[i] = b_cols[i];
            d->na = ctx->mfmas_m;
            d->nb = ctx->mfmas_n;
            continue;
        }
        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat]
                    = rocke_conv_emit_mfma(b, atom, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        }

        /* schedule.emit_after_mfma_step(
         *     b, ds_read_count=_n_dsr(kk) * _ds_reads_per_frag,
         *     mfma_count=mfmas_m * mfmas_n * _mfmas_per_mma) */
        /* _n_dsr(kk): k-packed reads issue both steps' reads at the even kk. */
        const int na_r = (spec->xh_stride_m && ctx->xh_prev != NULL) ? 1 : ctx->mfmas_m;
        const int n_dsr = (e->kprd && (kk % 2)) ? 0 : na_r + ctx->mfmas_n;
        int n_ld = ctx->vmi_pending;
        if(n_ld && (ctx->schedule.emit_hints || spec->uk_xhalo))
        {
            /* vmem_interleave: the next tile's buffer_loads, issued just before
             * this phase, go one per vmem_interleave MFMAs. */
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_READ, n_dsr * ctx->ds_reads_per_frag, 0);
            int n_mf = ctx->mfmas_m * ctx->mfmas_n * ctx->mfmas_per_mma;
            while(n_ld && n_mf)
            {
                rocke_b_sched_group_barrier(b, ROCKE_SCHED_VMEM_READ, 1, 0);
                const int take = spec->vmem_interleave < n_mf ? spec->vmem_interleave : n_mf;
                rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, take, 0);
                n_mf -= take;
                n_ld -= 1;
            }
            if(n_mf)
                rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, n_mf, 0);
            ctx->vmi_pending = n_ld;
        }
        else
        {
            rocke_schedule_policy_emit_after_mfma_step(&ctx->schedule,
                                                       b,
                                                       n_dsr * ctx->ds_reads_per_frag,
                                                       ctx->mfmas_m * ctx->mfmas_n
                                                           * ctx->mfmas_per_mma);
        }
    }
    ctx->vmi_pending = 0;
}
