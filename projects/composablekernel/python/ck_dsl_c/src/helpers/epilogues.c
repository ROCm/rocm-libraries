/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of DirectEpilogue and CShuffleEpilogue from
 * ck_dsl/helpers/epilogues.py. See the header for the scope and the AddrFn /
 * WarpGrid translation notes.
 *
 * Builder-call fidelity: every IR op is emitted in the same order as the
 * Python, with the same operands. The Python `raise` paths map to
 * ckc_i_set_err (the sticky-error spelling used across the C port).
 */
#include "ckc/helper_ck_dsl.helpers.epilogues.h"

#include "ckc/helper_ck_dsl.helpers.tensor_view.h" /* make_lds_view, ckc_tensor_view_t */
#include "ckc/ir_internal.h"                        /* ckc_i_set_err */

/* (1 << 31) - 1, the Python OOB sentinel byte offset. Kept as int64 so the
 * const_i32 argument is exact regardless of int width. */
#define CKC_EPI_OOB_SENTINEL ((int64_t)(((int64_t)1 << 31) - 1))

/* ============================ WarpGrid helpers ============================ */

bool ckc_warp_grid_is_bound(const ckc_warp_grid_t* grid)
{
    return grid != NULL && grid->tid != NULL;
}

int ckc_warp_grid_block_size(const ckc_warp_grid_t* grid)
{
    /* warp_m * warp_n * warp_k * wave_size */
    return grid->warp_m * grid->warp_n * grid->warp_k * grid->wave_size;
}

int ckc_warp_grid_mfmas_per_warp_m(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid)
{
    int div = grid->warp_m * grid->warp_tile_m;
    if (div == 0 || (grid->tile_m % div) != 0)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "tile_m %d not divisible by warp_m * warp_tile_m = %d",
                      grid->tile_m, div);
        return -1;
    }
    return grid->tile_m / div;
}

int ckc_warp_grid_mfmas_per_warp_n(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid)
{
    int div = grid->warp_n * grid->warp_tile_n;
    if (div == 0 || (grid->tile_n % div) != 0)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "tile_n %d not divisible by warp_n * warp_tile_n = %d",
                      grid->tile_n, div);
        return -1;
    }
    return grid->tile_n / div;
}

ckc_value_t* ckc_warp_grid_warp_m_off(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid)
{
    int mfmas_m;
    if (!ckc_warp_grid_is_bound(grid))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "WarpGrid is not bound; call .bind(b) first");
        return NULL;
    }
    mfmas_m = ckc_warp_grid_mfmas_per_warp_m(b, grid);
    if (mfmas_m < 0) return NULL;
    return ckc_b_mul(b, grid->warp_m_idx,
                     ckc_b_const_i32(b, (int64_t)mfmas_m * grid->warp_tile_m));
}

ckc_value_t* ckc_warp_grid_warp_n_off(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid)
{
    int mfmas_n;
    if (!ckc_warp_grid_is_bound(grid))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "WarpGrid is not bound; call .bind(b) first");
        return NULL;
    }
    mfmas_n = ckc_warp_grid_mfmas_per_warp_n(b, grid);
    if (mfmas_n < 0) return NULL;
    return ckc_b_mul(b, grid->warp_n_idx,
                     ckc_b_const_i32(b, (int64_t)mfmas_n * grid->warp_tile_n));
}

/* -------------------------------------------------------------- lane_to_output *
 *
 * File-local port of MfmaAtom.lane_to_output (ck_dsl/helpers/atoms.py): the
 * per-lane (row_in_atom, col_in_atom) of accumulator slot `i` within one atom.
 * The two epilogues are the only consumers, so it lives here as a static helper
 * (atoms.h does not yet export it). Returns 0 on success and writes *out_row /
 * *out_col; returns -1 + sets the builder error for the NotImplementedError
 * dispatch miss. */
static int ckc_epi_lane_to_output(ckc_ir_builder_t* b,
                                  const ckc_mfma_atom_t* atom,
                                  ckc_value_t* lane,
                                  int i,
                                  ckc_value_t** out_row,
                                  ckc_value_t** out_col)
{
    if (atom->m == 16 && atom->n == 16)
    {
        ckc_value_t* c_atom_n = ckc_b_const_i32(b, atom->n);
        ckc_value_t* n_in_atom = ckc_b_mod(b, lane, c_atom_n);
        ckc_value_t* m_blk = ckc_b_div(b, lane, c_atom_n);
        /* Python (atoms.py:572):
         *   row = b.add(b.mul(m_blk, b.const_i32(c_per_lane)), b.const_i32(i))
         * The b.mul subtree (and its inner const) runs before b.const_i32(i).
         * Sequence into temporaries so C arg-evaluation order does not perturb
         * the SSA numbering. */
        ckc_value_t* row_mul =
            ckc_b_mul(b, m_blk, ckc_b_const_i32(b, atom->c_per_lane));
        ckc_value_t* row_ci = ckc_b_const_i32(b, i);
        ckc_value_t* row    = ckc_b_add(b, row_mul, row_ci);
        *out_row = row;
        *out_col = n_in_atom;
        return 0;
    }
    if (atom->m == 32 && atom->n == 32)
    {
        ckc_value_t* c_atom_n = ckc_b_const_i32(b, atom->n);
        ckc_value_t* n_in_atom = ckc_b_mod(b, lane, c_atom_n);
        ckc_value_t* m_blk = ckc_b_div(b, lane, c_atom_n);
        int rb = i / 4;
        int ri = i % 4;
        /* Python (atoms.py:580-583):
         *   row = b.add(
         *       b.add(b.const_i32(rb * 8), b.mul(m_blk, b.const_i32(4))),
         *       b.const_i32(ri))
         * Evaluation order is strictly left-to-right:
         *   const(rb*8); const(4); mul; inner add; const(ri); outer add.
         * Sequence into temporaries to keep the SSA numbering byte-identical. */
        ckc_value_t* rb_ci   = ckc_b_const_i32(b, rb * 8);
        ckc_value_t* row_mul = ckc_b_mul(b, m_blk, ckc_b_const_i32(b, 4));
        ckc_value_t* row_in  = ckc_b_add(b, rb_ci, row_mul);
        ckc_value_t* ri_ci   = ckc_b_const_i32(b, ri);
        ckc_value_t* row     = ckc_b_add(b, row_in, ri_ci);
        *out_row = row;
        *out_col = n_in_atom;
        return 0;
    }
    if (atom->m == 4 && atom->n == 4)
    {
        ckc_value_t* c4 = ckc_b_const_i32(b, 4);
        ckc_value_t* lane_in_batch = ckc_b_mod(b, lane, c4);
        *out_row = ckc_b_const_i32(b, i);
        *out_col = lane_in_batch;
        return 0;
    }
    ckc_i_set_err(b, CKC_ERR_NOTIMPL, "no lane_to_output dispatch for atom %dx%d",
                  atom->m, atom->n);
    return -1;
}

/* ============================ DirectEpilogue ============================ */

int ckc_direct_epilogue_row_stride_per_slot(const ckc_direct_epilogue_t* epi)
{
    const ckc_mfma_atom_t* atom = epi->atom;
    if (atom->m == 16 && atom->n == 16) return 4; /* 4 consecutive M rows per lane */
    if (atom->m == 4 && atom->n == 4) return 4;   /* 4 consecutive M rows per lane */
    return 0;                                      /* 32x32: scattered */
}

bool ckc_direct_epilogue_is_col_contiguous(const ckc_direct_epilogue_t* epi)
{
    (void)epi;
    return false; /* Python keeps the helper simple */
}

ckc_value_t* ckc_direct_epilogue_bounds_check(ckc_ir_builder_t* b,
                                              ckc_value_t* m,
                                              ckc_value_t* n,
                                              ckc_value_t* bounds_m,
                                              ckc_value_t* bounds_n,
                                              int vec_n)
{
    ckc_value_t* m_ok;
    ckc_value_t* n_ok;
    if (bounds_m == NULL || bounds_n == NULL) return NULL; /* Python bounds is None */
    m_ok = ckc_b_cmp_lt(b, m, bounds_m);
    if (vec_n > 1)
    {
        ckc_value_t* n_end = ckc_b_add(b, n, ckc_b_const_i32(b, vec_n));
        n_ok = ckc_b_cmp_le(b, n_end, bounds_n);
    }
    else
    {
        n_ok = ckc_b_cmp_lt(b, n, bounds_n);
    }
    return ckc_b_land(b, m_ok, n_ok);
}

void ckc_direct_epilogue_store(ckc_ir_builder_t* b,
                               const ckc_direct_epilogue_t* epi,
                               ckc_value_t* const* accs,
                               int num_accs,
                               ckc_epilogue_addr_fn addr_fn,
                               void* addr_user,
                               ckc_value_t* d_rsrc,
                               ckc_value_t* bounds_m,
                               ckc_value_t* bounds_n,
                               bool vec_in_acc)
{
    const ckc_mfma_atom_t* atom = epi->atom;
    const ckc_warp_grid_t* grid = &epi->grid;
    int mfmas_m, mfmas_n, mi, ni;
    ckc_value_t* warp_m_off;
    ckc_value_t* warp_n_off;
    ckc_value_t* c_half_bytes;
    ckc_value_t* oob_sentinel;

    if (!ckc_warp_grid_is_bound(grid))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "DirectEpilogue: grid must be bound first");
        return;
    }

    mfmas_m = ckc_warp_grid_mfmas_per_warp_m(b, grid);
    mfmas_n = ckc_warp_grid_mfmas_per_warp_n(b, grid);
    if (mfmas_m < 0 || mfmas_n < 0) return;
    if (num_accs != mfmas_m * mfmas_n)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "DirectEpilogue: expected %d accs, got %d",
                      mfmas_m * mfmas_n, num_accs);
        return;
    }

    warp_m_off = ckc_warp_grid_warp_m_off(b, grid);
    warp_n_off = ckc_warp_grid_warp_n_off(b, grid);
    if (warp_m_off == NULL || warp_n_off == NULL) return;
    c_half_bytes = ckc_b_const_i32(b, 2);
    oob_sentinel = ckc_b_const_i32(b, CKC_EPI_OOB_SENTINEL);

    for (mi = 0; mi < mfmas_m; ++mi)
    {
        for (ni = 0; ni < mfmas_n; ++ni)
        {
            ckc_value_t* acc = accs[mi * mfmas_n + ni];
            /* Python (epilogues.py:189-196):
             *   atom_m_off = b.add(b.add(block_m_off, warp_m_off),
             *                      b.const_i32(mi * atom.m))
             *   atom_n_off = b.add(b.add(block_n_off, warp_n_off),
             *                      b.const_i32(ni * atom.n))
             * The inner b.add() runs (consuming an SSA id) before the
             * b.const_i32() second operand. C argument evaluation order is
             * unspecified, so sequence the inner add and the const into
             * temporaries to keep the SSA numbering byte-identical. */
            ckc_value_t* m_inner = ckc_b_add(b, grid->block_m_off, warp_m_off);
            ckc_value_t* m_const = ckc_b_const_i32(b, (int64_t)mi * atom->m);
            ckc_value_t* atom_m_off = ckc_b_add(b, m_inner, m_const);
            ckc_value_t* n_inner = ckc_b_add(b, grid->block_n_off, warp_n_off);
            ckc_value_t* n_const = ckc_b_const_i32(b, (int64_t)ni * atom->n);
            ckc_value_t* atom_n_off = ckc_b_add(b, n_inner, n_const);

            if (vec_in_acc)
            {
                /* one wide vec store per lane (4 halves per 4x4 atom). */
                ckc_value_t* row_off;
                ckc_value_t* col_off;
                ckc_value_t* m_val;
                ckc_value_t* n_val;
                ckc_value_t* ok;
                ckc_value_t* valid = NULL;
                ckc_value_t* off_elems;
                ckc_value_t* off_bytes;
                ckc_value_t* safe;
                ckc_value_t* acc_h;
                if (ckc_epi_lane_to_output(b, atom, grid->lane, 0, &row_off, &col_off) != 0)
                    return;
                m_val = ckc_b_add(b, atom_m_off, row_off);
                n_val = ckc_b_add(b, atom_n_off, col_off);
                ok = ckc_direct_epilogue_bounds_check(b, m_val, n_val, bounds_m, bounds_n,
                                                      atom->c_per_lane);
                off_elems = addr_fn(b, m_val, n_val, &valid, addr_user);
                if (ok != NULL)
                    ok = (valid != NULL) ? ckc_b_land(b, ok, valid) : ok;
                else
                    ok = valid;
                off_bytes = ckc_b_mul(b, off_elems, c_half_bytes);
                safe = (ok != NULL) ? ckc_b_select(b, ok, off_bytes, oob_sentinel)
                                    : off_bytes;
                acc_h = ckc_b_vec_trunc_f32_to_f16(b, acc);
                /* dword width from c_per_lane: 4 halves -> 2 dwords; 8 -> 4. */
                if (atom->c_per_lane == 4)
                    ckc_b_buffer_store_vN_f16(b, d_rsrc, safe, ckc_b_const_i32(b, 0),
                                              acc_h, 2);
                else if (atom->c_per_lane == 8)
                    ckc_b_buffer_store_vN_f16(b, d_rsrc, safe, ckc_b_const_i32(b, 0),
                                              acc_h, 4);
                else
                {
                    ckc_i_set_err(b, CKC_ERR_VALUE,
                                  "vec_in_acc=True with c_per_lane=%d unsupported",
                                  atom->c_per_lane);
                    return;
                }
            }
            else
            {
                int i;
                for (i = 0; i < atom->c_per_lane; ++i)
                {
                    ckc_value_t* row_off;
                    ckc_value_t* col_off;
                    ckc_value_t* m_val;
                    ckc_value_t* n_val;
                    ckc_value_t* ok;
                    ckc_value_t* valid = NULL;
                    ckc_value_t* v_f32;
                    ckc_value_t* v_f16;
                    ckc_value_t* off_elems;
                    ckc_value_t* off_bytes;
                    ckc_value_t* safe;
                    if (ckc_epi_lane_to_output(b, atom, grid->lane, i, &row_off,
                                               &col_off) != 0)
                        return;
                    m_val = ckc_b_add(b, atom_m_off, row_off);
                    n_val = ckc_b_add(b, atom_n_off, col_off);
                    ok = ckc_direct_epilogue_bounds_check(b, m_val, n_val, bounds_m,
                                                          bounds_n, 1);
                    off_elems = addr_fn(b, m_val, n_val, &valid, addr_user);
                    if (ok != NULL)
                        ok = (valid != NULL) ? ckc_b_land(b, ok, valid) : ok;
                    else
                        ok = valid;
                    v_f32 = ckc_b_vec_extract(b, acc, i);
                    v_f16 = ckc_b_trunc_f32_to_f16(b, v_f32);
                    off_bytes = ckc_b_mul(b, off_elems, c_half_bytes);
                    safe = (ok != NULL) ? ckc_b_select(b, ok, off_bytes, oob_sentinel)
                                        : off_bytes;
                    ckc_b_buffer_store_f16(b, d_rsrc, safe, ckc_b_const_i32(b, 0), v_f16);
                }
            }
        }
    }
}

/* ============================ CShuffleEpilogue ============================ */

ckc_cshuffle_epilogue_t ckc_cshuffle_epilogue_make(const ckc_mfma_atom_t* atom,
                                                   const ckc_warp_grid_t* grid)
{
    ckc_cshuffle_epilogue_t epi;
    epi.atom = atom;
    epi.grid = *grid;
    epi.store_vec = 8;          /* Python default */
    epi.smem_name_hint = "C_smem";
    epi.out_dtype = "f16";
    return epi;
}

ckc_cshuffle_epilogue_t ckc_cshuffle_epilogue_from_grid(const ckc_mfma_atom_t* atom,
                                                        const ckc_warp_grid_t* grid,
                                                        int max_store_vec)
{
    ckc_cshuffle_epilogue_t epi = ckc_cshuffle_epilogue_make(atom, grid);
    int v = max_store_vec;
    int block_size = ckc_warp_grid_block_size(grid);
    while (v > 1)
    {
        bool ok = (grid->tile_n % v == 0)
                  && ((grid->tile_m * grid->tile_n) / v >= block_size)
                  && (((grid->tile_m * grid->tile_n) / v) % block_size == 0);
        if (ok) break;
        v /= 2;
    }
    epi.store_vec = v;
    return epi;
}

void ckc_cshuffle_epilogue_store(ckc_ir_builder_t* b,
                                 const ckc_cshuffle_epilogue_t* epi,
                                 ckc_value_t* const* accs,
                                 int num_accs,
                                 ckc_epilogue_addr_fn addr_fn,
                                 void* addr_user,
                                 ckc_value_t* d_rsrc,
                                 ckc_value_t* bounds_m,
                                 ckc_value_t* bounds_n)
{
    const ckc_mfma_atom_t* atom = epi->atom;
    const ckc_warp_grid_t* grid = &epi->grid;
    int mfmas_m, mfmas_n, mi, ni;
    ckc_value_t* warp_m_off;
    ckc_value_t* warp_n_off;
    ckc_tensor_view_t c_view;
    ckc_value_t* c_smem;
    int lds_shape[2];
    int threads, sv, vecs_per_thread, e;
    ckc_value_t* c_threads;
    ckc_value_t* c_tile_n_div_vec;
    ckc_value_t* c_half_bytes;
    ckc_value_t* oob_sentinel;

    if (!ckc_warp_grid_is_bound(grid))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "CShuffleEpilogue: grid must be bound first");
        return;
    }

    mfmas_m = ckc_warp_grid_mfmas_per_warp_m(b, grid);
    mfmas_n = ckc_warp_grid_mfmas_per_warp_n(b, grid);
    if (mfmas_m < 0 || mfmas_n < 0) return;
    if (num_accs != mfmas_m * mfmas_n)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "CShuffleEpilogue: expected %d accs, got %d",
                      mfmas_m * mfmas_n, num_accs);
        return;
    }

    warp_m_off = ckc_warp_grid_warp_m_off(b, grid);
    warp_n_off = ckc_warp_grid_warp_n_off(b, grid);
    if (warp_m_off == NULL || warp_n_off == NULL) return;

    /* ---- step 1: publish accs to LDS at the MFMA output layout. ----
     *
     * The cshuffle LDS layout (LdsLayout.cshuffle) is a plain [tile_m, tile_n]
     * row-major region (logical_cols = tile_n, no swizzle, k_pad=0); see
     * helpers/layouts.py. The Python routes the publish through a
     * StaticDistributedTensor + store_tile_cshuffle(coord_fn=...) where
     * coord_fn(i) == lane_to_output(lane, i). Those distribution / tile-window
     * abstractions are not yet ported; the equivalent emitted op is one
     * ds_write_b16 per accumulator slot at the [ld_m, ld_n] coordinate, which we
     * emit directly here.
     *
     * TODO(port): route through store_tile_cshuffle once the distribution
     * tile-window machinery is ported, to keep the publish path byte-identical
     * to the StaticDistributedTensor spelling. */
    if (epi->out_dtype != NULL && epi->out_dtype[0] != '\0'
        && !(epi->out_dtype[0] == 'f' && epi->out_dtype[1] == '1'
             && epi->out_dtype[2] == '6' && epi->out_dtype[3] == '\0'))
    {
        /* TODO(port): bf16 / fp8e4m3 / bf8e5m2 staging element types are not yet
         * wired; only the default f16 path is emitted. */
        ckc_i_set_err(b, CKC_ERR_NOTIMPL,
                      "CShuffleEpilogue out_dtype=%s not yet ported (f16 only)",
                      epi->out_dtype);
        return;
    }

    lds_shape[0] = grid->tile_m; /* storage_shape(tile_m) == (tile_m, tile_n) */
    lds_shape[1] = grid->tile_n;
    if (ckc_make_lds_view(b, &c_view, ckc_f16(), lds_shape, 2, epi->smem_name_hint,
                          NULL) != CKC_OK)
        return;
    c_smem = c_view.base;

    /* Python (epilogues.py:399-402) builds the full-extent LDS window:
     *   c_window = c_view.tile(storage_shape(tile_m),
     *                          [b.const_i32(0), b.const_i32(0)])
     * The TileWindow itself carries no IR, but the two b.const_i32(0) origin
     * values ARE emitted here. The C publish path writes through the raw
     * c_smem base rather than the (unported) TileWindow, but the two origin
     * constants must still be emitted to keep the SSA numbering byte-identical
     * with the Python emitter. */
    (void) ckc_b_const_i32(b, 0);
    (void) ckc_b_const_i32(b, 0);

    for (mi = 0; mi < mfmas_m; ++mi)
    {
        for (ni = 0; ni < mfmas_n; ++ni)
        {
            ckc_value_t* acc = accs[mi * mfmas_n + ni];
            /* Python (epilogues.py:415-431) emits, per warp tile:
             *   acc_h = b.vec_trunc_f32_to_f16(acc)
             *   for i in range(c_per_lane):           # all extracts up front
             *       dt.set([i,0], b.vec_extract(acc_h, i))
             *   tile_m_base = b.add(warp_m_off, b.const_i32(mi * atom.m))
             *   tile_n_base = b.add(warp_n_off, b.const_i32(ni * atom.n))
             *   store_tile_cshuffle(...)             # per slot: coord_fn then store
             * where store_tile_cshuffle (distribution.py:1112-1133) does, per
             * slot: coord_fn (lane_to_output + the two ld_m/ld_n adds) then the
             * smem store of the already-extracted scalar. The extracts must
             * therefore all precede the tile bases and the per-slot coord/store
             * loop to keep the SSA emission order byte-identical. */
            ckc_value_t* acc_h = ckc_b_vec_trunc_f32_to_f16(b, acc);
            int i;
            ckc_value_t* halves[16]; /* c_per_lane <= 16 (32x32 atom) */
            for (i = 0; i < atom->c_per_lane; ++i)
                halves[i] = ckc_b_vec_extract(b, acc_h, i);
            ckc_value_t* tile_m_base =
                ckc_b_add(b, warp_m_off, ckc_b_const_i32(b, (int64_t)mi * atom->m));
            ckc_value_t* tile_n_base =
                ckc_b_add(b, warp_n_off, ckc_b_const_i32(b, (int64_t)ni * atom->n));
            for (i = 0; i < atom->c_per_lane; ++i)
            {
                /* coord_fn: (i, 0) -> lane_to_output(lane, i) offset by tile base */
                ckc_value_t* row_in_atom;
                ckc_value_t* col_in_atom;
                ckc_value_t* ld_m;
                ckc_value_t* ld_n;
                ckc_value_t* idx[2];
                if (ckc_epi_lane_to_output(b, atom, grid->lane, i, &row_in_atom,
                                           &col_in_atom) != 0)
                    return;
                ld_m = ckc_b_add(b, tile_m_base, row_in_atom);
                ld_n = ckc_b_add(b, tile_n_base, col_in_atom);
                idx[0] = ld_m;
                idx[1] = ld_n;
                ckc_b_smem_store_vN_f16(b, c_smem, idx, 2, halves[i], 1);
            }
        }
    }

    /* ---- step 2: barrier. ---- */
    ckc_b_sync(b);

    /* ---- step 3: wide global stores from LDS. ---- */
    threads = ckc_warp_grid_block_size(grid);
    sv = epi->store_vec;
    if ((grid->tile_n % sv)
        || ((grid->tile_m * grid->tile_n) / sv < threads)
        || (((grid->tile_m * grid->tile_n) / sv) % threads))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "store_vec %d does not distribute over tile %dx%d and block_size %d",
                      sv, grid->tile_m, grid->tile_n, threads);
        return;
    }
    vecs_per_thread = (grid->tile_m * grid->tile_n / sv) / threads;
    c_threads = ckc_b_const_i32(b, threads);
    c_tile_n_div_vec = ckc_b_const_i32(b, grid->tile_n / sv);
    c_half_bytes = ckc_b_const_i32(b, 2);
    oob_sentinel = ckc_b_const_i32(b, CKC_EPI_OOB_SENTINEL);

    for (e = 0; e < vecs_per_thread; ++e)
    {
        ckc_value_t* vec_idx =
            ckc_b_add(b, ckc_b_mul(b, ckc_b_const_i32(b, e), c_threads), grid->tid);
        ckc_value_t* row = ckc_b_div(b, vec_idx, c_tile_n_div_vec);
        ckc_value_t* col_v = ckc_b_mod(b, vec_idx, c_tile_n_div_vec);
        ckc_value_t* col =
            (sv > 1) ? ckc_b_mul(b, col_v, ckc_b_const_i32(b, sv)) : col_v;
        ckc_value_t* m_val = ckc_b_add(b, grid->block_m_off, row);
        ckc_value_t* n_val = ckc_b_add(b, grid->block_n_off, col);
        ckc_value_t* ok = ckc_direct_epilogue_bounds_check(b, m_val, n_val, bounds_m,
                                                           bounds_n, sv);
        ckc_value_t* valid = NULL;
        ckc_value_t* off_elems = addr_fn(b, m_val, n_val, &valid, addr_user);
        ckc_value_t* off_bytes;
        ckc_value_t* safe;
        ckc_value_t* idx[2];

        if (ok != NULL && valid != NULL)
            ok = ckc_b_land(b, ok, valid);
        else if (ok == NULL)
            ok = valid;
        /* (ok != NULL && valid == NULL) -> ok stays as-is */

        off_bytes = ckc_b_mul(b, off_elems, c_half_bytes);
        safe = (ok != NULL) ? ckc_b_select(b, ok, off_bytes, oob_sentinel) : off_bytes;

        idx[0] = row;
        idx[1] = col;
        if (sv == 1)
        {
            ckc_value_t* v = ckc_b_smem_load_vN_f16(b, c_smem, idx, 2, 2);
            ckc_value_t* h = ckc_b_vec_extract(b, v, 0);
            ckc_b_buffer_store_f16(b, d_rsrc, safe, ckc_b_const_i32(b, 0), h);
        }
        else
        {
            ckc_value_t* v;
            int dwords;
            if (sv == 4)
                v = ckc_b_smem_load_v4_f16(b, c_smem, row, col);
            else
                v = ckc_b_smem_load_vN_f16(b, c_smem, idx, 2, sv);
            dwords = sv / 2;
            ckc_b_buffer_store_vN_f16(b, d_rsrc, safe, ckc_b_const_i32(b, 0), v, dwords);
        }
    }
}
