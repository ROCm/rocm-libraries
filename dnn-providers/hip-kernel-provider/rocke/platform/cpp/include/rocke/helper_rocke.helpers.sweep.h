/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.sweep.h -- C99 port of the
 * ``rocke.helpers.sweep`` module.
 *
 * Ported symbols (per phase scope):
 *   sweep_row_chunks, pass2_row_chunks, RowChunkSweepResult.
 *
 * These mirror CK Tile's "load X once, sweep Y positions" idiom (see
 * ``include/ck_tile/core/tensor/sweep_tile.hpp``). Both functions emit IR
 * via the C builder (rocke_b_*, rocke/ir.h) and the TileWindow load/store-as-f32
 * peers (rocke/helper_rocke.helpers.tensor_view.h). The builder-call sequence
 * reproduces the Python module byte-for-byte: per chunk the helper emits
 *
 *   n_off = b.add(b.mul(b.const_i32(k * block_size), c_vec),
 *                 b.mul(tid, c_vec))
 *
 * then the f32 vec load (pass 1) or f32 demote + vec store (pass 2).
 *
 * Modelling choices for the port:
 *
 *   * Python lambda bodies become C function pointers carrying an explicit
 *     ``void* user`` context (the codebase convention -- cf. persistent /
 *     mfma_gemm_inner). ``body`` may be NULL in sweep_row_chunks (the
 *     "just populate cached" case); it is required in pass2_row_chunks.
 *   * The pass-1 body receives the freshly loaded ``vec`` f32 scalars as a
 *     read-only array. The pass-2 body fills an ``out[vec]`` array with the
 *     f32 scalars to store back (the Python ``return`` value).
 *   * ``RowChunkSweepResult.cached`` is the per-thread f32 scalar list. The
 *     C struct holds an arena-owned pointer + count; when ``cache`` is
 *     false the list is empty (NULL / 0), matching Python's empty list.
 *
 * Errors: where Python raises ValueError, the C port records the sticky
 * error on the builder (rocke_i_set_err) and returns an empty result / void.
 */
#ifndef ROCKE_HELPER_SWEEP_H
#define ROCKE_HELPER_SWEEP_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------- RowChunkSweepResult */

/* Bookkeeping returned by rocke_sweep_row_chunks (Python RowChunkSweepResult).
 *
 *   * cached / num_cached : the per-thread list of f32 SSA Values loaded
 *     during pass 1 when the caller asked for cache=true. The array is
 *     arena-owned (lives as long as the builder). Empty (NULL / 0) when
 *     cache was false -- matching Python's empty list.
 *
 *   * chunks_per_thread : the compile-time chunk count
 *     (elems_per_thread / vec).
 */
typedef struct rocke_row_chunk_sweep_result
{
    rocke_value_t** cached;
    int num_cached;
    int chunks_per_thread;
} rocke_row_chunk_sweep_result_t;

/* ---------------------------------------------------------------- callbacks */

/* Pass-1 body callback for rocke_sweep_row_chunks.
 *
 * Python ``body(n_off, x_scalars)`` is a closure; in C we thread an explicit
 * user-data pointer. ``n_off`` is the current chunk's starting column index
 * (i32 SSA). ``x_scalars`` is a read-only array of ``vec`` f32 SSA scalars
 * freshly loaded for this chunk. Invoked once per chunk.
 */
typedef void (*rocke_sweep_row_body_fn)(rocke_ir_builder_t* b,
                                        rocke_value_t* n_off,
                                        rocke_value_t* const* x_scalars,
                                        int vec,
                                        void* user);

/* Pass-2 body callback for rocke_pass2_row_chunks.
 *
 * Python ``body(n_off, k, x_scalars) -> list`` is a closure; in C we thread
 * an explicit user-data pointer. ``n_off`` is the current chunk's starting
 * column index (i32 SSA); ``k`` is the chunk index; ``x_scalars`` is the
 * (possibly empty) array of ``vec`` cached f32 scalars for this chunk. The
 * callback MUST fill ``out`` with exactly ``vec`` f32 SSA scalars to store
 * back; the helper then truncates and stores them. (Python returned the
 * list; the C contract is the out-array of fixed length ``vec``.)
 */
typedef void (*rocke_pass2_row_body_fn)(rocke_ir_builder_t* b,
                                        rocke_value_t* n_off,
                                        int k,
                                        rocke_value_t* const* x_scalars,
                                        int num_x,
                                        rocke_value_t** out,
                                        int vec,
                                        void* user);

/* -------------------------------------------------------- TileWindow peers *
 *
 * Provided by the tensor_view port (helper_rocke.helpers.tensor_view.c).
 * Declared here so this translation unit compiles standalone; resolved at
 * link time once the peer lands.
 *
 * rocke_tile_window_load_vec_as_f32: TileWindow.load_vec_as_f32 -- vector load
 *   + per-lane f32 promotion. Writes the ``n`` f32 SSA scalars to out[0..n)
 *   (caller-provided, length >= n).
 *
 * rocke_tile_window_store_vec_from_f32: TileWindow.store_vec_from_f32 -- f32
 *   demote + pack + vector store of the ``num_values`` scalars.
 */
void rocke_tile_window_load_vec_as_f32(rocke_ir_builder_t* b,
                                       const rocke_tile_window_t* w,
                                       rocke_value_t* const* local_indices,
                                       int num_indices,
                                       int n,
                                       rocke_value_t** out);

void rocke_tile_window_store_vec_from_f32(rocke_ir_builder_t* b,
                                          const rocke_tile_window_t* w,
                                          rocke_value_t* const* local_indices,
                                          int num_indices,
                                          rocke_value_t* const* values,
                                          int num_values);

/* ----------------------------------------------------------- sweep_row_chunks */

/* sweep_row_chunks(b, tile, *, tid, block_size, vec, elems_per_thread,
 *                  row=None, body=None, cache=False) -> RowChunkSweepResult
 *
 * Sweep one block's worth of (block_size * vec)-element chunks along the row
 * at ``row``, invoking ``body(n_off, x_scalars)`` per chunk.
 *
 *   * tile must be a 2D TileWindow (row dim first).
 *   * tid is the i32 SSA thread index.
 *   * block_size, vec, elems_per_thread are compile-time ints. body is
 *     called elems_per_thread / vec times.
 *   * row is an i32 SSA Value or NULL (Python None): when non-NULL the tile
 *     origin is shifted to (row, origin[1]) for the sweep, leaving the
 *     column origin untouched.
 *   * body / user form the per-chunk callback; pass body=NULL for an empty
 *     body (just populate ``cached``).
 *   * cache: when true, the loaded f32 scalars are recorded (in element
 *     order across chunks) into the result's ``cached`` list.
 *
 * On a Python-ValueError condition (non-2D tile, elems_per_thread not
 * divisible by vec) the sticky error is recorded on the builder and a
 * zeroed result is returned.
 */
rocke_row_chunk_sweep_result_t rocke_sweep_row_chunks(rocke_ir_builder_t* b,
                                                      const rocke_tile_window_t* tile,
                                                      rocke_value_t* tid,
                                                      int block_size,
                                                      int vec,
                                                      int elems_per_thread,
                                                      rocke_value_t* row,
                                                      rocke_sweep_row_body_fn body,
                                                      void* user,
                                                      bool cache);

/* ---------------------------------------------------------- pass2_row_chunks */

/* pass2_row_chunks(b, tile, *, tid, block_size, vec, elems_per_thread,
 *                  row=None, body=<required>, cached_f32=()) -> None
 *
 * Second-pass sweep that *writes* one block's worth of chunks.
 * ``body(n_off, k, x_scalars) -> [vec f32]`` is invoked elems_per_thread /
 * vec times; the helper truncates the returned scalars to the tile dtype and
 * stores them via TileWindow.store_vec_from_f32.
 *
 *   * cached_f32 / num_cached_f32 is the optional per-thread cache returned
 *     by rocke_sweep_row_chunks; when non-empty, chunk k receives the slice
 *     cached_f32[k*vec : (k+1)*vec] as its x_scalars. Pass NULL / 0 for the
 *     Python default empty tuple (the body then receives an empty slice).
 *   * row / tid / block_size / vec / elems_per_thread match
 *     rocke_sweep_row_chunks.
 *
 * On a Python-ValueError condition (non-2D tile, NULL body,
 * elems_per_thread not divisible by vec) the sticky error is recorded on the
 * builder and the function returns early.
 */
void rocke_pass2_row_chunks(rocke_ir_builder_t* b,
                            const rocke_tile_window_t* tile,
                            rocke_value_t* tid,
                            int block_size,
                            int vec,
                            int elems_per_thread,
                            rocke_value_t* row,
                            rocke_pass2_row_body_fn body,
                            void* user,
                            rocke_value_t* const* cached_f32,
                            int num_cached_f32);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_SWEEP_H */
