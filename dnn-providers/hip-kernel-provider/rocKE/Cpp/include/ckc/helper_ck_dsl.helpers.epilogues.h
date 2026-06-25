/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.epilogues.h -- C99 port of two symbols from
 * ck_dsl/helpers/epilogues.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   DirectEpilogue                      ckc_direct_epilogue_t + ckc_direct_epilogue_store
 *   CShuffleEpilogue                    ckc_cshuffle_epilogue_t + ckc_cshuffle_epilogue_*
 *
 * SCOPE: only these two epilogue helpers (the per-lane direct global store and
 * the LDS-staged C-shuffle store). Both are *frozen dataclasses* in Python with
 * a single public `store(...)` method; here each maps to a small value struct
 * plus free functions whose first argument is the struct.
 *
 * AddrFn callback
 * ---------------
 * The Python authoring input
 *     addr_fn : (b, m_global, n_global) -> (off_elements, valid)
 * is translated to a C function pointer that returns `off_elements` directly and
 * writes `valid` (the optional i1 OOB SSA, Python `None` => write NULL) through
 * an out-param. `user` carries the closure environment (the kernel's D
 * descriptor etc.).
 *
 * The op sequence emitted into the IRBuilder is byte-faithful to the Python:
 * the same const_i32 / add / mul / mod / div / select / cmp / land / trunc /
 * buffer_store / smem_load ops in the same order. Error paths (`raise
 * RuntimeError` for an unbound grid, `raise ValueError` for a bad acc count /
 * unsupported store width) map onto the builder sticky error (ckc_i_set_err)
 * exactly like the rest of the C port.
 *
 * Grid binding
 * ------------
 * The Python `WarpGrid` is not yet a standalone C type; the fields the
 * epilogues read are bundled into ckc_warp_grid_t below. A grid is "bound" when
 * `tid != NULL` (mirrors WarpGrid.is_bound == (self.tid is not None)).
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_EPILOGUES_H
#define CKC_HELPER_CK_DSL_HELPERS_EPILOGUES_H

#include <stdbool.h>

#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom_t */
#include "ckc/ir.h" /* ckc_ir_builder_t, ckc_value_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ WarpGrid *
 *
 * The subset of ck_dsl.helpers.geometry.WarpGrid that the epilogues consume.
 * Compile-time geometry is plain ints; the runtime SSA values are populated by
 * WarpGrid.bind in the kernel author's code (here the caller fills them).
 *
 * Mirrors the Python @property derivations the epilogues use:
 *   mfmas_per_warp_m = tile_m / (warp_m * warp_tile_m)   (must divide evenly)
 *   mfmas_per_warp_n = tile_n / (warp_n * warp_tile_n)   (must divide evenly)
 *   block_size       = warp_m * warp_n * warp_k * wave_size
 *   is_bound         = (tid != NULL)
 *   warp_m_off(b)    = warp_m_idx * (mfmas_per_warp_m * warp_tile_m)
 *   warp_n_off(b)    = warp_n_idx * (mfmas_per_warp_n * warp_tile_n)
 */
typedef struct ckc_warp_grid
{
    /* compile-time geometry */
    int tile_m;
    int tile_n;
    int tile_k;
    int warp_m;
    int warp_n;
    int warp_k;
    int warp_tile_m;
    int warp_tile_n;
    int warp_tile_k;
    int wave_size;

    /* runtime IR values (populated by WarpGrid.bind; NULL when unbound) */
    ckc_value_t* tid;
    ckc_value_t* lane;
    ckc_value_t* warp_id;
    ckc_value_t* warp_m_idx;
    ckc_value_t* warp_n_idx;
    ckc_value_t* warp_k_idx;
    ckc_value_t* block_m_off;
    ckc_value_t* block_n_off;
    ckc_value_t* block_k_off;
} ckc_warp_grid_t;

/* WarpGrid @property analogues. The mfmas_per_warp_* functions return -1 and
 * set the builder sticky error (CKC_ERR_VALUE) when the tile is not divisible by
 * warp * warp_tile, matching the Python ValueError. `block_size`/`is_bound`
 * never fail. The warp_*_off functions emit the same mul as the Python and
 * return NULL with the unbound RuntimeError set when the grid is not bound. */
bool ckc_warp_grid_is_bound(const ckc_warp_grid_t* grid);
int ckc_warp_grid_block_size(const ckc_warp_grid_t* grid);
int ckc_warp_grid_mfmas_per_warp_m(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid);
int ckc_warp_grid_mfmas_per_warp_n(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid);
ckc_value_t* ckc_warp_grid_warp_m_off(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid);
ckc_value_t* ckc_warp_grid_warp_n_off(ckc_ir_builder_t* b, const ckc_warp_grid_t* grid);

/* ------------------------------------------------------------------ AddrFn *
 *
 * Python: addr_fn(b, m_global, n_global) -> (off_elements, valid)
 *   - return value          == off_elements (i32 SSA linear element offset)
 *   - *out_valid (optional) == valid (i1 SSA, or NULL for Python None)
 *   - user                  == closure environment
 * `out_valid` is never NULL itself (callee always writes it); the *written*
 * value may be NULL to mean "no per-element validity". */
typedef ckc_value_t* (*ckc_epilogue_addr_fn)(ckc_ir_builder_t* b,
                                             ckc_value_t* m_global,
                                             ckc_value_t* n_global,
                                             ckc_value_t** out_valid,
                                             void* user);

/* ------------------------------------------------------------- DirectEpilogue *
 *
 * Python @dataclass(frozen=True) DirectEpilogue(atom, grid).
 * `atom` points at a static MFMA catalog entry (see atoms.h); `grid` is a bound
 * WarpGrid subset. */
typedef struct ckc_direct_epilogue
{
    const ckc_mfma_atom_t* atom;
    ckc_warp_grid_t grid;
} ckc_direct_epilogue_t;

/* @property _row_stride_per_slot: 4 for 16x16 / 4x4, 0 for 32x32 (scattered). */
int ckc_direct_epilogue_row_stride_per_slot(const ckc_direct_epilogue_t* epi);
/* @property _is_col_contiguous: always false (the Python keeps it simple). */
bool ckc_direct_epilogue_is_col_contiguous(const ckc_direct_epilogue_t* epi);

/* DirectEpilogue.store(b, accs, addr_fn, d_rsrc, bounds=None, vec_in_acc=False).
 *
 * `accs` is the flat row-major (mi, ni) accumulator list of length
 * num_accs == mfmas_per_warp_m * mfmas_per_warp_n. `bounds_m`/`bounds_n` are the
 * (M, N) i32 SSA values; pass both NULL to skip the OOB mask (Python
 * bounds=None). `addr_fn`/`addr_user` is the AddrFn closure. On error (unbound
 * grid, wrong acc count, unsupported vec width) the builder sticky error is set
 * and the function returns early. */
void ckc_direct_epilogue_store(ckc_ir_builder_t* b,
                               const ckc_direct_epilogue_t* epi,
                               ckc_value_t* const* accs,
                               int num_accs,
                               ckc_epilogue_addr_fn addr_fn,
                               void* addr_user,
                               ckc_value_t* d_rsrc,
                               ckc_value_t* bounds_m, /* NULL => no bounds */
                               ckc_value_t* bounds_n, /* NULL => no bounds */
                               bool vec_in_acc);

/* DirectEpilogue._bounds_check(b, m, n, bounds, vec_n=1) -> Optional[Value].
 * Returns NULL for the Python `None` (bounds_m/bounds_n NULL) path, else the i1
 * `m < M && (n[+vec_n] cmp) N` SSA. Static method in Python; exposed because the
 * CShuffleEpilogue store calls it too. */
ckc_value_t* ckc_direct_epilogue_bounds_check(ckc_ir_builder_t* b,
                                              ckc_value_t* m,
                                              ckc_value_t* n,
                                              ckc_value_t* bounds_m,
                                              ckc_value_t* bounds_n,
                                              int vec_n);

/* ----------------------------------------------------------- CShuffleEpilogue *
 *
 * Python @dataclass(frozen=True) CShuffleEpilogue(atom, grid, store_vec=8,
 *   smem_name_hint="C_smem", out_dtype="f16").
 *
 * NOTE: out_dtype other than "f16" (the bf16/fp8 staging variants documented in
 * the Python) is NOT yet wired in this phase; only the default f16 path is
 * emitted. A non-"f16" out_dtype is accepted on the struct but the store path
 * treats it as the f16 path (see TODO(port) in the .c). */
typedef struct ckc_cshuffle_epilogue
{
    const ckc_mfma_atom_t* atom;
    ckc_warp_grid_t grid;
    int store_vec; /* halves per wide store; default 8 */
    const char* smem_name_hint; /* default "C_smem" */
    const char* out_dtype; /* default "f16" */
} ckc_cshuffle_epilogue_t;

/* Construct with the Python defaults (store_vec=8, smem_name_hint="C_smem",
 * out_dtype="f16"). */
ckc_cshuffle_epilogue_t ckc_cshuffle_epilogue_make(const ckc_mfma_atom_t* atom,
                                                   const ckc_warp_grid_t* grid);

/* CShuffleEpilogue.from_grid(atom, grid, max_store_vec=8): pick the widest
 * store_vec that distributes the tile evenly over block_size. */
ckc_cshuffle_epilogue_t ckc_cshuffle_epilogue_from_grid(const ckc_mfma_atom_t* atom,
                                                        const ckc_warp_grid_t* grid,
                                                        int max_store_vec);

/* CShuffleEpilogue.store(b, accs, addr_fn, d_rsrc, bounds=None).
 * Same acc/bounds/addr_fn contract as the direct store. */
void ckc_cshuffle_epilogue_store(ckc_ir_builder_t* b,
                                 const ckc_cshuffle_epilogue_t* epi,
                                 ckc_value_t* const* accs,
                                 int num_accs,
                                 ckc_epilogue_addr_fn addr_fn,
                                 void* addr_user,
                                 ckc_value_t* d_rsrc,
                                 ckc_value_t* bounds_m, /* NULL => no bounds */
                                 ckc_value_t* bounds_n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_EPILOGUES_H */
