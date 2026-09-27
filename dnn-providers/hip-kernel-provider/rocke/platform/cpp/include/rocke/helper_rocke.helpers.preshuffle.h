/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.preshuffle.h -- C99 port of three symbols from
 * rocke/helpers/preshuffle.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   PreshuffleBSpec (frozen dataclass)  rocke_preshuffleb_spec_t + .tile_bytes
 *                                         rocke_preshuffleb_spec_tile_bytes()
 *   emit_preshuffleb_offset(...)        rocke_emit_preshuffleb_offset(...)
 *   host_preshuffle_layout(...)         rocke_host_preshuffle_layout(...)
 *
 * Preshuffled-B layout helper. The preshuffled-B GEMM family reorders the
 * B-matrix tiles host-side into tile-major layout so each per-K-iter load is
 * one aligned buffer_load_dwordx4. This module is the layout descriptor + the
 * per-lane byte-offset producer.
 *
 * Of the three symbols:
 *
 *   - PreshuffleBSpec / .tile_bytes is a pure value type: tile_bytes is the
 *     block_n * block_k * elem_bytes product. No builder.
 *   - emit_preshuffleb_offset IS the only builder-emitting symbol: it issues
 *     the exact const_i32 / add / mul sequence of the Python, in order, so the
 *     resulting IR is byte-identical.
 *   - host_preshuffle_layout is a pure host-side (shape, strides) producer with
 *     a divisibility precondition. The Python `raise ValueError` maps to the
 *     ROCKE_ERR_VALUE status return; the builder-aware spelling records the
 *     Python-matching message on the builder sticky error.
 *
 * Error model mirrors the rest of the C port: an out-param + rocke_status_t for
 * the builder-free spelling, and a sticky-error builder (rocke_b_*) for the
 * builder-aware one.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_PRESHUFFLE_H
#define ROCKE_HELPER_ROCKE_HELPERS_PRESHUFFLE_H

#include <stddef.h>

#include "rocke/ir.h" /* rocke_status_t, rocke_ir_builder_t, rocke_value_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ PreshuffleBSpec *
 *
 * Value type mirroring rocke.helpers.preshuffle.PreshuffleBSpec (frozen
 * dataclass). One concrete preshuffled-B tile shape.
 *
 * Fields are 1:1 with the Python dataclass declaration order:
 *   block_n, block_k, elem_bytes (default 1).
 *
 * elem_bytes: 1 for fp8/bf8/i8; 2 for f16/bf16; for i4 use 1 with 2-per-byte
 * packing (the Python notes 0.5 conceptually but the field is an int).
 */
typedef struct rocke_preshuffleb_spec
{
    int block_n;
    int block_k;
    int elem_bytes; /* Python default 1 */
} rocke_preshuffleb_spec_t;

/* PreshuffleBSpec.tile_bytes property:
 *   block_n * block_k * elem_bytes -- bytes per preshuffled tile. */
int rocke_preshuffleb_spec_tile_bytes(const rocke_preshuffleb_spec_t* spec);

/* ------------------------------------------------- emit_preshuffleb_offset *
 *
 * Python:
 *
 *     def emit_preshuffleb_offset(b, spec, *, n_tile, k_tile, n_in_tile,
 *                                 k_in_tile, n_tile_count) -> Value:
 *         c_tile_bytes = b.const_i32(spec.tile_bytes)
 *         c_block_k    = b.const_i32(spec.block_k)
 *         c_elem_bytes = b.const_i32(spec.elem_bytes)
 *         tile_id     = b.add(b.mul(k_tile, n_tile_count), n_tile)
 *         tile_base   = b.mul(tile_id, c_tile_bytes)
 *         inner       = b.add(b.mul(n_in_tile, c_block_k), k_in_tile)
 *         inner_bytes = b.mul(inner, c_elem_bytes)
 *         return b.add(tile_base, inner_bytes)
 *
 * Per-lane byte offset for one (n_tile, k_tile, n_in_tile, k_in_tile) quad:
 *
 *   offset = (k_tile * n_tile_count + n_tile) * tile_bytes
 *          + (n_in_tile * block_k + k_in_tile) * elem_bytes
 *
 * Emits the const_i32 / mul / add ops in the exact Python order onto `b`.
 * Returns the resulting offset Value, or NULL if `b` is already in an error
 * state (or `spec` is NULL, which records ROCKE_ERR_VALUE on `b`). */
rocke_value_t* rocke_emit_preshuffleb_offset(rocke_ir_builder_t* b,
                                             const rocke_preshuffleb_spec_t* spec,
                                             rocke_value_t* n_tile,
                                             rocke_value_t* k_tile,
                                             rocke_value_t* n_in_tile,
                                             rocke_value_t* k_in_tile,
                                             rocke_value_t* n_tile_count);

/* ------------------------------------------------- host_preshuffle_layout *
 *
 * Python:
 *
 *     def host_preshuffle_layout(spec, *, n, k) -> (shape, strides):
 *         n_tiles = (n + spec.block_n - 1) // spec.block_n
 *         k_tiles = (k + spec.block_k - 1) // spec.block_k
 *         if n_tiles*block_n != n or k_tiles*block_k != k: raise ValueError(...)
 *         shape   = (k_tiles, n_tiles, block_n, block_k)
 *         strides = (n_tiles*block_n*block_k, block_n*block_k, block_k, 1)
 *         return shape, strides
 *
 * Pure host-side layout descriptor. On success writes the 4-element shape and
 * strides tuples into the caller-provided arrays (each of length 4; any may be
 * NULL to skip) and returns ROCKE_OK.
 *
 * Returns ROCKE_ERR_VALUE if N / K do not divide block_n / block_k (the Python
 * ValueError path), leaving the out arrays untouched. `spec` must be non-NULL
 * (a NULL spec also returns ROCKE_ERR_VALUE). */
rocke_status_t rocke_host_preshuffle_layout(
    const rocke_preshuffleb_spec_t* spec, int n, int k, int out_shape[4], int out_strides[4]);

/* Builder-aware variant: identical computation; on the ValueError path it sets
 * the builder sticky error (ROCKE_ERR_VALUE) with a Python-matching message and
 * returns that status. No-op returning b->status if `b` is already in error. */
rocke_status_t rocke_b_host_preshuffle_layout(rocke_ir_builder_t* b,
                                              const rocke_preshuffleb_spec_t* spec,
                                              int n,
                                              int k,
                                              int out_shape[4],
                                              int out_strides[4]);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_PRESHUFFLE_H */
