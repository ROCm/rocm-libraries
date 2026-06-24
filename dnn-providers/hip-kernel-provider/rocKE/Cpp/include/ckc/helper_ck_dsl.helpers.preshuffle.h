/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.preshuffle.h -- C99 port of three symbols from
 * ck_dsl/helpers/preshuffle.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   PreshuffleBSpec (frozen dataclass)  ckc_preshuffleb_spec_t + .tile_bytes
 *                                         ckc_preshuffleb_spec_tile_bytes()
 *   emit_preshuffleb_offset(...)        ckc_emit_preshuffleb_offset(...)
 *   host_preshuffle_layout(...)         ckc_host_preshuffle_layout(...)
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
 *     CKC_ERR_VALUE status return; the builder-aware spelling records the
 *     Python-matching message on the builder sticky error.
 *
 * Error model mirrors the rest of the C port: an out-param + ckc_status_t for
 * the builder-free spelling, and a sticky-error builder (ckc_b_*) for the
 * builder-aware one.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_PRESHUFFLE_H
#define CKC_HELPER_CK_DSL_HELPERS_PRESHUFFLE_H

#include <stddef.h>

#include "ckc/ir.h" /* ckc_status_t, ckc_ir_builder_t, ckc_value_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ PreshuffleBSpec *
 *
 * Value type mirroring ck_dsl.helpers.preshuffle.PreshuffleBSpec (frozen
 * dataclass). One concrete preshuffled-B tile shape.
 *
 * Fields are 1:1 with the Python dataclass declaration order:
 *   block_n, block_k, elem_bytes (default 1).
 *
 * elem_bytes: 1 for fp8/bf8/i8; 2 for f16/bf16; for i4 use 1 with 2-per-byte
 * packing (the Python notes 0.5 conceptually but the field is an int).
 */
typedef struct ckc_preshuffleb_spec
{
    int block_n;
    int block_k;
    int elem_bytes; /* Python default 1 */
} ckc_preshuffleb_spec_t;

/* PreshuffleBSpec.tile_bytes property:
 *   block_n * block_k * elem_bytes -- bytes per preshuffled tile. */
int ckc_preshuffleb_spec_tile_bytes(const ckc_preshuffleb_spec_t* spec);

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
 * state (or `spec` is NULL, which records CKC_ERR_VALUE on `b`). */
ckc_value_t* ckc_emit_preshuffleb_offset(ckc_ir_builder_t* b,
                                         const ckc_preshuffleb_spec_t* spec,
                                         ckc_value_t* n_tile,
                                         ckc_value_t* k_tile,
                                         ckc_value_t* n_in_tile,
                                         ckc_value_t* k_in_tile,
                                         ckc_value_t* n_tile_count);

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
 * NULL to skip) and returns CKC_OK.
 *
 * Returns CKC_ERR_VALUE if N / K do not divide block_n / block_k (the Python
 * ValueError path), leaving the out arrays untouched. `spec` must be non-NULL
 * (a NULL spec also returns CKC_ERR_VALUE). */
ckc_status_t ckc_host_preshuffle_layout(
    const ckc_preshuffleb_spec_t* spec, int n, int k, int out_shape[4], int out_strides[4]);

/* Builder-aware variant: identical computation; on the ValueError path it sets
 * the builder sticky error (CKC_ERR_VALUE) with a Python-matching message and
 * returns that status. No-op returning b->status if `b` is already in error. */
ckc_status_t ckc_b_host_preshuffle_layout(ckc_ir_builder_t* b,
                                          const ckc_preshuffleb_spec_t* spec,
                                          int n,
                                          int k,
                                          int out_shape[4],
                                          int out_strides[4]);

#ifdef __cplusplus
}
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_PRESHUFFLE_H */
