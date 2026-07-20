// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_epilogue.c -- C99 port of the epilogue chunk of
 * rocke/instances/common/gemm_universal.py (lines ~1783-2302):
 *
 *   _emit_epilogue            (rocke_gemm_emit_epilogue)
 *   _emit_mfma_acc_scatter    (rocke_gemm_emit_mfma_acc_scatter)
 *   _emit_epilogue_default    (rocke_gemm_emit_epilogue_default)
 *   _emit_epilogue_cshuffle   (rocke_gemm_emit_epilogue_cshuffle)
 *   _load_smem_scalar         (rocke_gemm_load_smem_scalar)
 *   _load_smem_vec            (rocke_gemm_load_smem_vec)
 *
 * Faithful, byte-identical builder-call sequence: each emitted op mirrors the
 * Python op order/attrs. Peer phase functions are called via the internal
 * header; only this scope's functions are defined here.
 */
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.atoms.h" /* mfma_atom / c_warp_params*/
#include "rocke/helper_rocke.helpers.distribution.h" /* tile distribution        */
#include "rocke/helper_rocke.helpers.fuse.h" /* FusedEpilogue apply/decl */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_io_ir_type           */
#include "rocke/instance_gemm_internal.h"

/* Dispatch helpers for the attached fused epilogue (Python's runtime method
 * resolution on the _fused_epilogue object). The pointer is the FusedEpilogue
 * base for both the stock chain and a _MultiDEpilogue (whose first member is
 * that base); `is_mde` selects the apply_vec override. declare_params /
 * record_runtime / apply_scalar are inherited from FusedEpilogue in both. */
static void
    rocke_gemm_fe_declare_and_record(rocke_ir_builder_t* b, void* fused_epilogue, rocke_value_t* N)
{
    rocke_fused_epilogue_t* fe = (rocke_fused_epilogue_t*)fused_epilogue;
    if(fe == NULL)
    {
        return;
    }
    /* fused_epilogue.declare_params(b)
     * record_runtime(b, N=N)   (FusedEpilogue.record_runtime exists -> called) */
    rocke_fe_declare_params(b, fe);
    rocke_fe_record_runtime(fe, N, NULL);
}

static rocke_value_t* rocke_gemm_fe_apply_scalar(rocke_ir_builder_t* b,
                                                 void* fused_epilogue,
                                                 rocke_value_t* v,
                                                 rocke_value_t* m,
                                                 rocke_value_t* n)
{
    /* _MultiDEpilogue does not override apply_scalar -> base path either way. */
    return rocke_fe_apply_scalar(b, (rocke_fused_epilogue_t*)fused_epilogue, v, m, n);
}

static rocke_value_t* rocke_gemm_fe_apply_vec(rocke_ir_builder_t* b,
                                              void* fused_epilogue,
                                              bool is_mde,
                                              rocke_value_t* v,
                                              rocke_value_t* m,
                                              rocke_value_t* n,
                                              int n_elems)
{
    if(is_mde)
    {
        /* _MultiDEpilogue.apply_vec(b, v, m, n, n_elems=...) */
        return rocke_mde_apply_vec(b, (rocke_multi_d_epilogue_t*)fused_epilogue, v, m, n, n_elems);
    }
    /* FusedEpilogue.apply_vec(b, v, m, n, n_elems=...) */
    return rocke_fe_apply_vec(b, (rocke_fused_epilogue_t*)fused_epilogue, v, m, n, n_elems);
}

/* =====================================================================
 * file-static helpers (module-level Python helpers not exposed by the
 * shared internal header; reproduced here as TU-local statics so the
 * epilogue functions stay self-contained, matching the Python module).
 * ===================================================================== */

/* _dtype_ir(name) -> io_ir_type(name). */
static const rocke_type_t* gemm_dtype_ir(rocke_ir_builder_t* b, const char* name)
{
    return rocke_b_io_ir_type(b, name);
}

/* _storage_dtype(spec): validates homogeneous A/B/C dtypes, fp32 acc, RCR
 * layout, then resolves the A dtype to its IR type. Returns NULL with the
 * builder sticky error set on failure.
 *
 * Note: in Python the three validation branches raise ValueError directly. The
 * C port has no public sticky-error setter, so on a homogeneity / acc / layout
 * mismatch we route through rocke_b_io_ir_type with the offending A-dtype: for the
 * (already-validated upstream) GEMM call sites the dtypes are homogeneous fp16 /
 * bf16 with fp32 acc + RCR, so this reduces to io_ir_type(dtype_a) exactly as
 * Python and the emitted IR is byte-identical. NAMED GAP (blocked on shared
 * infra): the heterogeneous-dtype ValueError text is NOT reproduced here because
 * no public _storage_dtype validator exists in a shared header; the ck_gemm_
 * storage_dtype validator in gemm_spec.cpp is the host-side reject surface and is
 * reached before emission, so these emit-path call sites only need the resolved
 * IR type. When a shared _storage_dtype validator lands, call it here so the
 * ValueError messages match byte-for-byte. */
static const rocke_type_t* gemm_storage_dtype(rocke_ir_builder_t* b,
                                              const rocke_gemm_universal_spec_t* spec)
{
    const rocke_gemm_data_spec_t* d = &spec->data;
    const char* a = d->dtype_a;
    const char* bb = d->dtype_b;
    const char* c = d->dtype_c;
    const char* acc = d->dtype_acc;
    bool ab_ne;
    bool ac_ne;
    bool acc_ok;
    bool layout_ok;

    if(!rocke_ir_builder_ok(b))
        return NULL;

    ab_ne = (a == NULL || bb == NULL) ? (a != bb) : (strcmp(a, bb) != 0);
    ac_ne = (a == NULL || c == NULL) ? (a != c) : (strcmp(a, c) != 0);
    acc_ok = acc != NULL && (strcmp(acc, "fp32") == 0 || strcmp(acc, "f32") == 0);
    layout_ok = d->layout != NULL && strcmp(d->layout, "RCR") == 0;

    if(ab_ne || ac_ne || !acc_ok || !layout_ok)
    {
        /* Force the ValueError-equivalent sticky error through the io helper. */
        return rocke_b_io_ir_type(b, NULL);
    }
    return gemm_dtype_ir(b, d->dtype_a);
}

/* =====================================================================
 * _load_smem_scalar / _load_smem_vec
 * ===================================================================== */

rocke_value_t* rocke_gemm_load_smem_scalar(rocke_ir_builder_t* b,
                                           rocke_value_t* smem,
                                           rocke_value_t* row,
                                           rocke_value_t* col,
                                           const rocke_type_t* dtype)
{
    /* v = b.smem_load_vN(smem, row, col, dtype=dtype, n=2)
     * return b.vec_extract(v, 0) */
    rocke_value_t* idx[2] = {row, col};
    rocke_value_t* v = rocke_b_smem_load_vN(b, smem, idx, 2, dtype, 2);
    return rocke_b_vec_extract(b, v, 0);
}

rocke_value_t* rocke_gemm_load_smem_vec(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* row,
                                        rocke_value_t* col,
                                        int n,
                                        const rocke_type_t* dtype)
{
    /* if dtype == F16 and n == 4: return b.smem_load_v4_f16(smem, row, col)
     * return b.smem_load_vN(smem, row, col, dtype=dtype, n=n) */
    if(rocke_type_eq(dtype, rocke_f16()) && n == 4)
        return rocke_b_smem_load_v4_f16(b, smem, row, col);
    {
        rocke_value_t* idx[2] = {row, col};
        return rocke_b_smem_load_vN(b, smem, idx, 2, dtype, n);
    }
}

/* =====================================================================
 * _emit_mfma_acc_scatter
 * ===================================================================== */

void rocke_gemm_emit_mfma_acc_scatter(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      rocke_value_t* lane,
                                      rocke_value_t* const* accs,
                                      int num_accs,
                                      rocke_value_t* m_base_off,
                                      rocke_value_t* n_base_off,
                                      int c_per_lane,
                                      const rocke_type_t* storage_dtype,
                                      rocke_gemm_per_cell_fn per_cell,
                                      void* user,
                                      bool n_base_first)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);

    (void)num_accs;

    /* atom = mfma_atom(dtype_a, warp_tile_m, warp_tile_n, warp_tile_k)
     * _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
     * c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom)) */
    const rocke_mfma_atom_t* atom
        = rocke_b_mfma_atom(b, spec->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    if(!rocke_ir_builder_ok(b))
        return;

    int kc_m1 = 0;
    int kc_nlane = 0;
    if(rocke_b_c_warp_params(b, atom, NULL, NULL, &kc_m1, &kc_nlane) != ROCKE_OK)
        return;

    rocke_tile_distribution_encoding_t* c_enc = rocke_make_c_warp_dstr_encoding(b, atom);
    rocke_tile_distribution_t* c_dist = rocke_make_static_tile_distribution(b, c_enc);
    (void)c_dist;
    if(!rocke_ir_builder_ok(b))
        return;

    /* c_nlane = b.const_i32(kc_nlane)
     * n_in_atom = b.mod(lane, c_nlane)
     * m_blk     = b.div(lane, c_nlane)
     * p_lane = [m_blk, n_in_atom] */
    rocke_value_t* c_nlane = rocke_b_const_i32(b, kc_nlane);
    rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_nlane);
    rocke_value_t* m_blk = rocke_b_div(b, lane, c_nlane);
    rocke_value_t* p_lane[2] = {m_blk, n_in_atom};

    /* Per accumulator slot i, the two Y dims decompose row-major over
     * (kCM0PerLane, kCM1PerLane): y0 = i // kCM1PerLane, y1 = i % kCM1PerLane.
     * calculate_x reconstructs row_in_atom (lane + slot) and col_in_atom
     * (= n_in_atom) from these. This mirrors the Python:
     *
     *   for i in range(c_per_lane):
     *     ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
     *     x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
     *     row_in_atom.append(x_row); col_in_atom = x_col
     */
    rocke_value_t* row_in_atom[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* col_in_atom = n_in_atom;
    {
        int i;
        rocke_value_t* const* ps[1] = {p_lane};
        int ps_counts[1] = {2};
        for(i = 0; i < c_per_lane; ++i)
        {
            rocke_value_t* ys[2]
                = {rocke_b_const_i32(b, i / kc_m1), rocke_b_const_i32(b, i % kc_m1)};
            rocke_value_t* x_out[2] = {NULL, NULL};
            if(!rocke_tile_distribution_calculate_x(b, c_dist, ys, 2, ps, ps_counts, 1, x_out, 2))
                return;
            row_in_atom[i] = x_out[0];
            col_in_atom = x_out[1]; /* constant across i (depends only on n_in_atom) */
        }
    }

    /* flat = 0
     * for mi in range(mfmas_m):
     *   base_m = add(m_base_off, mi*warp_tile_m)
     *   for ni in range(mfmas_n):
     *     acc = accs[flat]; flat += 1
     *     acc_h = vec_cast_f32_to(acc, storage_dtype)
     *     c_n = (n_base_first) ? add(add(n_base_off, ni*wtn), col_in_atom)
     *                          : add(n_base_off, add(ni*wtn, col_in_atom))
     *     for i in range(c_per_lane):
     *       c_m = add(base_m, row_in_atom[i])
     *       per_cell(c_m, c_n, acc_h, i) */
    {
        int flat = 0;
        int mi, ni, i;
        for(mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* base_m
                = rocke_b_add(b, m_base_off, rocke_b_const_i32(b, mi * t->warp_tile_m));
            for(ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* acc = accs[flat];
                rocke_value_t* acc_h;
                rocke_value_t* c_n;
                flat += 1;
                acc_h = rocke_b_vec_cast_f32_to(b, acc, storage_dtype);
                if(n_base_first)
                {
                    c_n = rocke_b_add(
                        b,
                        rocke_b_add(b, n_base_off, rocke_b_const_i32(b, ni * t->warp_tile_n)),
                        col_in_atom);
                }
                else
                {
                    c_n = rocke_b_add(
                        b,
                        n_base_off,
                        rocke_b_add(b, rocke_b_const_i32(b, ni * t->warp_tile_n), col_in_atom));
                }
                for(i = 0; i < c_per_lane; ++i)
                {
                    rocke_value_t* c_m = rocke_b_add(b, base_m, row_in_atom[i]);
                    per_cell(b, c_m, c_n, acc_h, i, user);
                }
            }
        }
    }
}

/* =====================================================================
 * _emit_epilogue_default
 * ===================================================================== */

/* Shared per-cell + per-element-store closure state for the default epilogue. */
typedef struct gemm_default_store_user
{
    rocke_ir_builder_t* b;
    const rocke_gemm_universal_spec_t* spec;
    rocke_value_t* M;
    rocke_value_t* N;
    rocke_value_t* C;
    rocke_value_t* batch_off_c;
    void* fused_epilogue; /* opaque (NULL = matmul-only) */
    bool fused_is_mde; /* fused_epilogue is a _MultiDEpilogue base */
    bool pad_m;
    bool pad_n;
} gemm_default_store_user_t;

/* _store_masked(c_m, c_n, c_off, h). */
static void gemm_default_store_masked(gemm_default_store_user_t* u,
                                      rocke_value_t* c_m,
                                      rocke_value_t* c_n,
                                      rocke_value_t* c_off,
                                      rocke_value_t* h)
{
    rocke_ir_builder_t* b = u->b;
    if(!(u->pad_m || u->pad_n))
    {
        rocke_b_global_store(b, u->C, c_off, h, 2);
        return;
    }
    {
        rocke_value_t* checks[2];
        int nchecks = 0;
        rocke_value_t* in_bounds;
        if(u->pad_m)
            checks[nchecks++] = rocke_b_cmp_lt(b, c_m, u->M);
        if(u->pad_n)
            checks[nchecks++] = rocke_b_cmp_lt(b, c_n, u->N);
        in_bounds = (nchecks == 1) ? checks[0] : rocke_b_land(b, checks[0], checks[1]);
        {
            rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_global_store(b, u->C, c_off, h, 2);
            rocke_b_region_leave(b);
        }
    }
}

/* _store_cell(c_m, c_n, acc_h, i): the per-cell callback for the MFMA scatter. */
static void gemm_default_store_cell(rocke_ir_builder_t* b,
                                    rocke_value_t* c_m,
                                    rocke_value_t* c_n,
                                    rocke_value_t* acc_h,
                                    int i,
                                    void* user)
{
    gemm_default_store_user_t* u = (gemm_default_store_user_t*)user;
    rocke_value_t* c_off = rocke_b_add(b, rocke_b_mul(b, c_m, u->N), c_n);
    rocke_value_t* h;
    if(u->batch_off_c != NULL)
        c_off = rocke_b_add(b, u->batch_off_c, c_off);
    h = rocke_b_vec_extract(b, acc_h, i);
    if(u->fused_epilogue != NULL)
    {
        /* h = fused_epilogue.apply_scalar(b, h, c_m, c_n) */
        h = rocke_gemm_fe_apply_scalar(b, u->fused_epilogue, h, c_m, c_n);
    }
    gemm_default_store_masked(u, c_m, c_n, c_off, h);
}

void rocke_gemm_emit_epilogue_default(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      const rocke_mmaop_t* op,
                                      rocke_value_t* const* accs,
                                      int num_accs,
                                      rocke_value_t* warp_m_idx,
                                      rocke_value_t* warp_n_idx,
                                      rocke_value_t* lane,
                                      rocke_value_t* block_m_off,
                                      rocke_value_t* block_n_off,
                                      rocke_value_t* M,
                                      rocke_value_t* N,
                                      rocke_value_t* C,
                                      int c_per_lane,
                                      rocke_value_t* batch_off_c,
                                      void* fused_epilogue,
                                      bool fused_is_mde)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* storage_dtype = gemm_storage_dtype(b, spec);
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    bool pad_m = spec->trait.pad_m;
    bool pad_n = spec->trait.pad_n;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;

    (void)num_accs;
    if(!rocke_ir_builder_ok(b))
        return;

    warp_m_off = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    warp_n_off = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));

    if(fused_epilogue != NULL)
    {
        /* fused_epilogue.declare_params(b); record_runtime(b, N=N) */
        rocke_gemm_fe_declare_and_record(b, fused_epilogue, N);
    }

    /* ---- WMMA (RDNA) accumulator scatter (op.family == "wmma"). ---- */
    if(op->family != NULL && strcmp(op->family, "wmma") == 0)
    {
        const rocke_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);
        rocke_value_t* block_warp_m_off = rocke_b_add(b, block_m_off, warp_m_off);
        rocke_value_t* block_warp_n_off = rocke_b_add(b, block_n_off, warp_n_off);
        gemm_default_store_user_t u;
        int flat = 0;
        int mi, ni, i;

        u.b = b;
        u.spec = spec;
        u.M = M;
        u.N = N;
        u.C = C;
        u.batch_off_c = batch_off_c;
        u.fused_epilogue = fused_epilogue;
        u.fused_is_mde = fused_is_mde;
        u.pad_m = pad_m;
        u.pad_n = pad_n;

        for(mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* atom_m
                = rocke_b_add(b, block_warp_m_off, rocke_b_const_i32(b, mi * t->warp_tile_m));
            for(ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* acc = accs[flat];
                rocke_value_t* atom_n;
                rocke_value_t* acc_h;
                flat += 1;
                atom_n
                    = rocke_b_add(b, block_warp_n_off, rocke_b_const_i32(b, ni * t->warp_tile_n));
                acc_h = rocke_b_vec_cast_f32_to(b, acc, storage_dtype);
                for(i = 0; i < c_per_lane; ++i)
                {
                    rocke_value_t* row_in_atom = NULL;
                    rocke_value_t* col_in_atom = NULL;
                    rocke_value_t* c_m;
                    rocke_value_t* c_n;
                    rocke_value_t* c_off;
                    rocke_value_t* h;
                    rocke_layout_map_coord(c_map, b, lane, i, &row_in_atom, &col_in_atom);
                    c_m = rocke_b_add(b, atom_m, row_in_atom);
                    c_n = rocke_b_add(b, atom_n, col_in_atom);
                    c_off = rocke_b_add(b, rocke_b_mul(b, c_m, N), c_n);
                    if(batch_off_c != NULL)
                        c_off = rocke_b_add(b, batch_off_c, c_off);
                    h = rocke_b_vec_extract(b, acc_h, i);
                    if(fused_epilogue != NULL)
                    {
                        /* h = fused_epilogue.apply_scalar(b, h, c_m, c_n) */
                        h = rocke_gemm_fe_apply_scalar(b, fused_epilogue, h, c_m, c_n);
                    }
                    gemm_default_store_masked(&u, c_m, c_n, c_off, h);
                }
            }
        }
        return;
    }

    /* ---- MFMA (CDNA): shared accumulator -> (row, col) scatter. ---- */
    {
        rocke_value_t* block_warp_m_off = rocke_b_add(b, block_m_off, warp_m_off);
        rocke_value_t* block_warp_n_off = rocke_b_add(b, block_n_off, warp_n_off);
        gemm_default_store_user_t u;

        u.b = b;
        u.spec = spec;
        u.M = M;
        u.N = N;
        u.C = C;
        u.batch_off_c = batch_off_c;
        u.fused_epilogue = fused_epilogue;
        u.fused_is_mde = fused_is_mde;
        u.pad_m = pad_m;
        u.pad_n = pad_n;

        rocke_gemm_emit_mfma_acc_scatter(b,
                                         spec,
                                         lane,
                                         accs,
                                         num_accs,
                                         block_warp_m_off,
                                         block_warp_n_off,
                                         c_per_lane,
                                         storage_dtype,
                                         gemm_default_store_cell,
                                         &u,
                                         false);
    }
}

/* =====================================================================
 * _emit_epilogue_cshuffle
 * ===================================================================== */

/* _smem_cell closure state: the LDS staging tile handle. */
typedef struct gemm_cshuffle_smem_user
{
    rocke_value_t* Cs;
} gemm_cshuffle_smem_user_t;

/* _smem_cell(ld_m, ld_n, acc_h, i): extract one half + scalar smem store. */
static void gemm_cshuffle_smem_cell(rocke_ir_builder_t* b,
                                    rocke_value_t* ld_m,
                                    rocke_value_t* ld_n,
                                    rocke_value_t* acc_h,
                                    int i,
                                    void* user)
{
    gemm_cshuffle_smem_user_t* u = (gemm_cshuffle_smem_user_t*)user;
    rocke_value_t* h = rocke_b_vec_extract(b, acc_h, i);
    rocke_value_t* idx[2] = {ld_m, ld_n};
    rocke_b_smem_store_vN(b, u->Cs, idx, 2, h, 1);
}

/* =====================================================================
 * _emit_epilogue_split_k
 *
 * Faithful, byte-identical port of the Python _emit_epilogue_split_k. Each warp
 * owns a per-lane <c_per_lane x f32> accumulator; every slot is scattered to its
 * output (c_m, c_n) using the atom's CWarpDstrEncoding (exactly as the default
 * epilogue scatter) and the raw f32 value is atomic-added into Cf32[c_m, c_n].
 * Only the per-cell write differs from the direct epilogue (f32 atomicrmw fadd
 * vs bf16 global store).
 * ===================================================================== */
void rocke_gemm_emit_epilogue_split_k(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      rocke_value_t* const* accs,
                                      int num_accs,
                                      rocke_value_t* warp_m_idx,
                                      rocke_value_t* warp_n_idx,
                                      rocke_value_t* lane,
                                      rocke_value_t* block_m_off,
                                      rocke_value_t* block_n_off,
                                      rocke_value_t* M,
                                      rocke_value_t* N,
                                      rocke_value_t* Cf32,
                                      int c_per_lane)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    rocke_value_t* block_warp_m_off;
    rocke_value_t* block_warp_n_off;
    bool pad_m;
    bool pad_n;

    (void)num_accs;

    /* warp_m_off = mul(warp_m_idx, mfmas_m*warp_tile_m)
     * warp_n_off = mul(warp_n_idx, mfmas_n*warp_tile_n)
     * block_warp_m_off = add(block_m_off, warp_m_off)
     * block_warp_n_off = add(block_n_off, warp_n_off) */
    warp_m_off = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    warp_n_off = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));
    block_warp_m_off = rocke_b_add(b, block_m_off, warp_m_off);
    block_warp_n_off = rocke_b_add(b, block_n_off, warp_n_off);

    /* atom = mfma_atom(dtype_a, warp_tile_m, warp_tile_n, warp_tile_k)
     * _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
     * c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom)) */
    const rocke_mfma_atom_t* atom
        = rocke_b_mfma_atom(b, spec->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    if(!rocke_ir_builder_ok(b))
        return;

    int kc_m1 = 0;
    int kc_nlane = 0;
    if(rocke_b_c_warp_params(b, atom, NULL, NULL, &kc_m1, &kc_nlane) != ROCKE_OK)
        return;

    rocke_tile_distribution_encoding_t* c_enc = rocke_make_c_warp_dstr_encoding(b, atom);
    rocke_tile_distribution_t* c_dist = rocke_make_static_tile_distribution(b, c_enc);
    if(!rocke_ir_builder_ok(b))
        return;

    /* c_nlane = const(kc_nlane); n_in_atom = mod(lane, c_nlane);
     * m_blk = div(lane, c_nlane); p_lane = [m_blk, n_in_atom] */
    rocke_value_t* c_nlane = rocke_b_const_i32(b, kc_nlane);
    rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_nlane);
    rocke_value_t* m_blk = rocke_b_div(b, lane, c_nlane);
    rocke_value_t* p_lane[2] = {m_blk, n_in_atom};

    /* row_in_atom[i], col_in_atom via c_dist.calculate_x for each slot. */
    rocke_value_t* row_in_atom[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* col_in_atom = n_in_atom;
    {
        int i;
        rocke_value_t* const* ps[1] = {p_lane};
        int ps_counts[1] = {2};
        for(i = 0; i < c_per_lane; ++i)
        {
            rocke_value_t* ys[2]
                = {rocke_b_const_i32(b, i / kc_m1), rocke_b_const_i32(b, i % kc_m1)};
            rocke_value_t* x_out[2] = {NULL, NULL};
            if(!rocke_tile_distribution_calculate_x(b, c_dist, ys, 2, ps, ps_counts, 1, x_out, 2))
                return;
            row_in_atom[i] = x_out[0];
            col_in_atom = x_out[1];
        }
    }

    pad_m = spec->trait.pad_m;
    pad_n = spec->trait.pad_n;

    /* flat = 0
     * for mi in range(mfmas_m):
     *   base_m = add(block_warp_m_off, mi*warp_tile_m)
     *   for ni in range(mfmas_n):
     *     acc = accs[flat]; flat += 1
     *     c_n = add(block_warp_n_off, add(ni*warp_tile_n, col_in_atom))
     *     for i in range(c_per_lane):
     *       c_m = add(base_m, row_in_atom[i])
     *       c_off = add(mul(c_m, N), c_n)
     *       val = vec_extract(acc, i)
     *       [guard c_m<M / c_n<N] -> global_atomic_add_f32(Cf32, c_off, val) */
    {
        int flat = 0;
        int mi, ni, i;
        for(mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* base_m
                = rocke_b_add(b, block_warp_m_off, rocke_b_const_i32(b, mi * t->warp_tile_m));
            for(ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* acc = accs[flat];
                rocke_value_t* c_n;
                flat += 1;
                c_n = rocke_b_add(
                    b,
                    block_warp_n_off,
                    rocke_b_add(b, rocke_b_const_i32(b, ni * t->warp_tile_n), col_in_atom));
                for(i = 0; i < c_per_lane; ++i)
                {
                    rocke_value_t* c_m = rocke_b_add(b, base_m, row_in_atom[i]);
                    rocke_value_t* c_off = rocke_b_add(b, rocke_b_mul(b, c_m, N), c_n);
                    rocke_value_t* val = rocke_b_vec_extract(b, acc, i);
                    if(pad_m || pad_n)
                    {
                        rocke_value_t* checks[2];
                        int nchecks = 0;
                        rocke_value_t* in_bounds;
                        if(pad_m)
                            checks[nchecks++] = rocke_b_cmp_lt(b, c_m, M);
                        if(pad_n)
                            checks[nchecks++] = rocke_b_cmp_lt(b, c_n, N);
                        in_bounds
                            = (nchecks == 1) ? checks[0] : rocke_b_land(b, checks[0], checks[1]);
                        {
                            rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
                            rocke_b_region_enter(b, iff.then_region);
                            rocke_b_global_atomic_add_f32(b, Cf32, c_off, val);
                            rocke_b_region_leave(b);
                        }
                    }
                    else
                    {
                        rocke_b_global_atomic_add_f32(b, Cf32, c_off, val);
                    }
                }
            }
        }
    }
}

void rocke_gemm_emit_epilogue_cshuffle(rocke_ir_builder_t* b,
                                       const rocke_gemm_universal_spec_t* spec,
                                       rocke_value_t* smem_unused,
                                       rocke_value_t* const* accs,
                                       int num_accs,
                                       rocke_value_t* warp_m_idx,
                                       rocke_value_t* warp_n_idx,
                                       rocke_value_t* lane,
                                       rocke_value_t* block_m_off,
                                       rocke_value_t* block_n_off,
                                       rocke_value_t* M,
                                       rocke_value_t* N,
                                       rocke_value_t* C,
                                       int a_per_lane,
                                       int b_per_lane,
                                       int c_per_lane,
                                       rocke_value_t* batch_off_c,
                                       void* fused_epilogue,
                                       bool fused_is_mde)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* storage_dtype = gemm_storage_dtype(b, spec);
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    rocke_value_t* Cs;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    gemm_cshuffle_smem_user_t smem_user;

    int threads;
    int store_vec;
    rocke_value_t* tid;
    rocke_value_t* c_threads;
    rocke_value_t* c_tile_n_div_vec;
    int vecs_per_thread;
    bool pad_m;
    bool pad_n;

    (void)smem_unused;
    (void)a_per_lane;
    (void)b_per_lane;
    (void)num_accs;
    if(!rocke_ir_builder_ok(b))
        return;

    if(fused_epilogue != NULL)
    {
        /* fused_epilogue.declare_params(b)
         * record_runtime = getattr(fused_epilogue, "record_runtime", None)
         * if record_runtime is not None: record_runtime(b, N=N) */
        rocke_gemm_fe_declare_and_record(b, fused_epilogue, N);
    }

    /* Cs = b.smem_alloc(storage_dtype, [tile_m, tile_n], name_hint="C_smem") */
    {
        int shape[2];
        shape[0] = t->tile_m;
        shape[1] = t->tile_n;
        Cs = rocke_b_smem_alloc(b, storage_dtype, shape, 2, "C_smem");
    }

    warp_m_off = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    warp_n_off = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));

    /* ---- step 1+2: warp accs -> LDS at the MFMA layout. ---- */
    smem_user.Cs = Cs;
    rocke_gemm_emit_mfma_acc_scatter(b,
                                     spec,
                                     lane,
                                     accs,
                                     num_accs,
                                     warp_m_off,
                                     warp_n_off,
                                     c_per_lane,
                                     storage_dtype,
                                     gemm_cshuffle_smem_cell,
                                     &smem_user,
                                     true);

    /* ---- step 3: barrier. ---- */
    rocke_b_sync(b);

    /* ---- step 4: wide global stores. ---- */
    threads = spec->block_size;
    store_vec = 8;
    while(store_vec > 1
          && ((t->tile_n % store_vec != 0) || ((t->tile_m * t->tile_n) / store_vec < threads)
              || (((t->tile_m * t->tile_n) / store_vec) % threads)))
    {
        store_vec /= 2;
    }

    if(store_vec == 1)
    {
        /* Pathological: fall back to scalar stores. */
        store_vec = 1;
    }

    tid = rocke_b_thread_id_x(b);
    c_threads = rocke_b_const_i32(b, threads);
    c_tile_n_div_vec = rocke_b_const_i32(b, t->tile_n / store_vec);
    vecs_per_thread = (t->tile_m * t->tile_n / store_vec) / threads;
    pad_m = spec->trait.pad_m;
    pad_n = spec->trait.pad_n;

    {
        int e;
        for(e = 0; e < vecs_per_thread; ++e)
        {
            rocke_value_t* vec_idx
                = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads), tid);
            /* _vec_rc(vec_idx): row, col_v, col */
            rocke_value_t* row = rocke_b_div(b, vec_idx, c_tile_n_div_vec);
            rocke_value_t* col_v = rocke_b_mod(b, vec_idx, c_tile_n_div_vec);
            rocke_value_t* col
                = (store_vec > 1) ? rocke_b_mul(b, col_v, rocke_b_const_i32(b, store_vec)) : col_v;

            rocke_value_t* c_m = rocke_b_add(b, block_m_off, row);
            rocke_value_t* c_n = rocke_b_add(b, block_n_off, col);
            rocke_value_t* c_off = rocke_b_add(b, rocke_b_mul(b, c_m, N), c_n);
            rocke_value_t* in_bounds = NULL;

            if(batch_off_c != NULL)
                c_off = rocke_b_add(b, batch_off_c, c_off);

            if(pad_m || pad_n)
            {
                rocke_value_t* checks[2];
                int nchecks = 0;
                if(pad_m)
                    checks[nchecks++] = rocke_b_cmp_lt(b, c_m, M);
                if(pad_n)
                {
                    if(store_vec == 1)
                    {
                        checks[nchecks++] = rocke_b_cmp_lt(b, c_n, N);
                    }
                    else
                    {
                        rocke_value_t* c_n_last
                            = rocke_b_add(b, c_n, rocke_b_const_i32(b, store_vec - 1));
                        checks[nchecks++] = rocke_b_cmp_lt(b, c_n_last, N);
                    }
                }
                in_bounds = (nchecks == 1) ? checks[0] : rocke_b_land(b, checks[0], checks[1]);
            }

            if(store_vec == 1)
            {
                rocke_value_t* h = rocke_gemm_load_smem_scalar(b, Cs, row, col, storage_dtype);
                if(fused_epilogue != NULL)
                {
                    /* h = fused_epilogue.apply_scalar(b, h, c_m, c_n) */
                    h = rocke_gemm_fe_apply_scalar(b, fused_epilogue, h, c_m, c_n);
                }
                if(in_bounds != NULL)
                {
                    rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
                    rocke_b_region_enter(b, iff.then_region);
                    rocke_b_global_store(b, C, c_off, h, 2);
                    rocke_b_region_leave(b);
                }
                else
                {
                    rocke_b_global_store(b, C, c_off, h, 2);
                }
            }
            else
            {
                rocke_value_t* hv
                    = rocke_gemm_load_smem_vec(b, Cs, row, col, store_vec, storage_dtype);
                if(pad_n)
                {
                    int i;
                    for(i = 0; i < store_vec; ++i)
                    {
                        rocke_value_t* c_n_i
                            = i ? rocke_b_add(b, c_n, rocke_b_const_i32(b, i)) : c_n;
                        rocke_value_t* c_off_i
                            = i ? rocke_b_add(b, c_off, rocke_b_const_i32(b, i)) : c_off;
                        rocke_value_t* h = rocke_b_vec_extract(b, hv, i);
                        rocke_value_t* checks[2];
                        int nchecks = 0;
                        rocke_value_t* elem_in_bounds;
                        if(fused_epilogue != NULL)
                        {
                            h = rocke_gemm_fe_apply_scalar(b, fused_epilogue, h, c_m, c_n_i);
                        }
                        if(pad_m)
                            checks[nchecks++] = rocke_b_cmp_lt(b, c_m, M);
                        checks[nchecks++] = rocke_b_cmp_lt(b, c_n_i, N);
                        elem_in_bounds
                            = (nchecks == 1) ? checks[0] : rocke_b_land(b, checks[0], checks[1]);
                        {
                            rocke_if_t iff = rocke_b_scf_if(b, elem_in_bounds);
                            rocke_b_region_enter(b, iff.then_region);
                            rocke_b_global_store(b, C, c_off_i, h, 2);
                            rocke_b_region_leave(b);
                        }
                    }
                }
                else
                {
                    if(fused_epilogue != NULL)
                    {
                        /* hv = fused_epilogue.apply_vec(b, hv, c_m, c_n, n_elems=store_vec) */
                        hv = rocke_gemm_fe_apply_vec(
                            b, fused_epilogue, fused_is_mde, hv, c_m, c_n, store_vec);
                    }
                    if(in_bounds != NULL)
                    {
                        rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
                        rocke_b_region_enter(b, iff.then_region);
                        rocke_b_global_store_vN(b, C, c_off, hv, store_vec, 0);
                        rocke_b_region_leave(b);
                    }
                    else
                    {
                        rocke_b_global_store_vN(b, C, c_off, hv, store_vec, 0);
                    }
                }
            }
        }
    }
}

/* =====================================================================
 * _emit_epilogue
 * ===================================================================== */

void rocke_gemm_emit_epilogue(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    /* fused_ep = getattr(spec, "_fused_epilogue", None) -- the side-channel the
     * multi-D builder populates via rocke_gemm_universal_spec_set_fused_epilogue.
     * NULL keeps the matmul-only path byte-identical. */
    void* fused_ep = spec->_fused_epilogue;
    bool fused_is_mde = spec->_fused_epilogue_is_mde;

    if(ctx->is_split_k)
    {
        /* Split-K: atomic-add each warp's f32 accumulator into the Cf32[M, N]
         * workspace. Reuses the same MFMA acc -> (row, col) scatter as the
         * default epilogue, but the per-cell write is an f32 atomicrmw fadd
         * instead of a bf16 global store. */
        rocke_gemm_emit_epilogue_split_k(b,
                                         spec,
                                         ctx->for_results,
                                         ctx->num_for_results,
                                         ctx->warp_m_idx,
                                         ctx->warp_n_idx,
                                         ctx->lane,
                                         ctx->block_m_off,
                                         ctx->block_n_off,
                                         ctx->M,
                                         ctx->N,
                                         ctx->C,
                                         ctx->c_per_lane);
        return;
    }

    if(spec->trait.epilogue != NULL && strcmp(spec->trait.epilogue, "cshuffle") == 0)
    {
        rocke_gemm_emit_epilogue_cshuffle(b,
                                          spec,
                                          ctx->A_smem,
                                          ctx->for_results,
                                          ctx->num_for_results,
                                          ctx->warp_m_idx,
                                          ctx->warp_n_idx,
                                          ctx->lane,
                                          ctx->block_m_off,
                                          ctx->block_n_off,
                                          ctx->M,
                                          ctx->N,
                                          ctx->C,
                                          ctx->a_per_lane,
                                          ctx->b_per_lane,
                                          ctx->c_per_lane,
                                          ctx->batch_off_c,
                                          fused_ep,
                                          fused_is_mde);
    }
    else
    {
        rocke_gemm_emit_epilogue_default(b,
                                         spec,
                                         ctx->op,
                                         ctx->for_results,
                                         ctx->num_for_results,
                                         ctx->warp_m_idx,
                                         ctx->warp_n_idx,
                                         ctx->lane,
                                         ctx->block_m_off,
                                         ctx->block_n_off,
                                         ctx->M,
                                         ctx->N,
                                         ctx->C,
                                         ctx->c_per_lane,
                                         ctx->batch_off_c,
                                         fused_ep,
                                         fused_is_mde);
    }
}
