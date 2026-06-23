/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.streamk.h -- C99 port of four symbols from
 * ck_dsl/helpers/streamk.py (StreamK tile partitioner helpers, CK Tile
 * parity):
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   class StreamKReductionStrategy      ckc_streamk_reduction_strategy_t (enum)
 *   class StreamKPartition (dataclass)  ckc_streamk_partition_t (struct) +
 *                                         ckc_streamk_partition_num_macro_tiles()
 *                                         ckc_streamk_partition_k_iters_per_output_tile()
 *   compute_streamk_grid_size(...)      ckc_compute_streamk_grid_size(...)
 *   emit_streamk_decode(...)            ckc_emit_streamk_decode(...)
 *
 * StreamKPartition is a frozen dataclass of three compile-time ints; its two
 * properties (num_macro_tiles, k_iters_per_output_tile) become pure accessor
 * functions. compute_streamk_grid_size is a pure-int compute. Only
 * emit_streamk_decode touches the IR builder; its builder-call sequence is
 * byte-faithful to the Python so the emitted op stream matches exactly.
 *
 * Error model mirrors the rest of the C port: the Python `raise ValueError`
 * in compute_streamk_grid_size is reported via an out-param status + a
 * sentinel return; emit_streamk_decode leaves NULL result fields and sets the
 * builder's sticky error on a degenerate spec.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_STREAMK_H
#define CKC_HELPER_CK_DSL_HELPERS_STREAMK_H

#include "ckc/ir.h" /* ckc_ir_builder_t, ckc_value_t, ckc_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Default chiplet / CU count for compute_streamk_grid_size (MI300X). */
#define CKC_STREAMK_DEFAULT_NUM_CUS 304
#define CKC_STREAMK_DEFAULT_BLOCKS_PER_CU 1

/* ------------------------------------------------------------------ *
 * StreamKReductionStrategy
 * ------------------------------------------------------------------ *
 * Python enum (values are the strategy strings, per CK Tile naming):
 *   Atomic    = "atomic"     # global_atomic_add_f32 into workspace
 *   Reduction = "reduction"  # cooperative + flag-table reduction
 */
typedef enum ckc_streamk_reduction_strategy
{
    CKC_STREAMK_REDUCTION_ATOMIC    = 0, /* "atomic"    */
    CKC_STREAMK_REDUCTION_REDUCTION = 1  /* "reduction" */
} ckc_streamk_reduction_strategy_t;

/* The enum value string ("atomic" / "reduction"), or NULL if out of range. */
const char* ckc_streamk_reduction_strategy_value(ckc_streamk_reduction_strategy_t s);

/* ------------------------------------------------------------------ *
 * StreamKPartition
 * ------------------------------------------------------------------ *
 * Python: @dataclass(frozen=True) class StreamKPartition
 *   m_tiles: int   # ceil(M / tile_m); the number of M tiles
 *   n_tiles: int   # ceil(N / tile_n)
 *   k_iters: int   # K / tile_k
 *
 * The compile-time-fixed shape inputs to the StreamK partitioner; the actual
 * (m_tile, n_tile, k_iter, is_first, is_last) decoding happens at runtime via
 * ckc_emit_streamk_decode.
 */
typedef struct ckc_streamk_partition
{
    int m_tiles; /* ceil(M / tile_m) */
    int n_tiles; /* ceil(N / tile_n) */
    int k_iters; /* K / tile_k       */
} ckc_streamk_partition_t;

/* @property num_macro_tiles: m_tiles * n_tiles * k_iters -- total chunk count.
 * (Also exposed under the module name streamk_num_macro_tiles in Python.) */
int ckc_streamk_partition_num_macro_tiles(const ckc_streamk_partition_t* spec);

/* @property k_iters_per_output_tile: k_iters -- the number of CTAs that touch
 * each (m, n) output tile in the Atomic strategy. */
int ckc_streamk_partition_k_iters_per_output_tile(const ckc_streamk_partition_t* spec);

/* Module-level streamk_num_macro_tiles(spec): plain Python view of the
 * num_macro_tiles property. */
int ckc_streamk_num_macro_tiles(const ckc_streamk_partition_t* spec);

/* ------------------------------------------------------------------ *
 * compute_streamk_grid_size
 * ------------------------------------------------------------------ *
 * Python:
 *   def compute_streamk_grid_size(spec, *, num_cus=304, blocks_per_cu=1):
 *       if spec.num_macro_tiles <= 0:
 *           raise ValueError("spec has zero macro tiles")
 *       return min(spec.num_macro_tiles, num_cus * blocks_per_cu)
 *
 * Pass CKC_STREAMK_DEFAULT_NUM_CUS / CKC_STREAMK_DEFAULT_BLOCKS_PER_CU for the
 * keyword defaults. On the ValueError path, *out_status (if non-NULL) is set
 * to CKC_ERR_VALUE and the function returns -1; otherwise *out_status (if
 * non-NULL) is set to CKC_OK.
 */
int ckc_compute_streamk_grid_size(const ckc_streamk_partition_t* spec,
                                  int num_cus,
                                  int blocks_per_cu,
                                  ckc_status_t* out_status);

/* ------------------------------------------------------------------ *
 * emit_streamk_decode
 * ------------------------------------------------------------------ *
 * Python NamedTuple returned by emit_streamk_decode:
 *   m_tile:   Value  # i32
 *   n_tile:   Value  # i32
 *   k_iter:   Value  # i32 in [0, k_iters)
 *   is_first: Value  # i1: k_iter == 0
 *   is_last:  Value  # i1: k_iter == k_iters - 1
 */
typedef struct ckc_streamk_decoded_tile
{
    ckc_value_t* m_tile;
    ckc_value_t* n_tile;
    ckc_value_t* k_iter;
    ckc_value_t* is_first;
    ckc_value_t* is_last;
} ckc_streamk_decoded_tile_t;

/* Decode a linear macro-tile id into (m_tile, n_tile, k_iter, is_first,
 * is_last). Layout matches CK Tile's "K-major within a (m, n) tile" walk:
 *
 *   c_k_iters = b.const_i32(spec.k_iters)
 *   c_n_tiles = b.const_i32(spec.n_tiles)
 *   k_iter    = b.mod(linear_id, c_k_iters)
 *   nn        = b.div(linear_id, c_k_iters)
 *   n_tile    = b.mod(nn, c_n_tiles)
 *   m_tile    = b.div(nn, c_n_tiles)
 *   is_first  = b.cmp_eq(k_iter, b.const_i32(0))
 *   is_last   = b.cmp_eq(k_iter, b.const_i32(spec.k_iters - 1))
 *
 * The builder-call sequence is byte-identical to the Python.
 */
ckc_streamk_decoded_tile_t ckc_emit_streamk_decode(ckc_ir_builder_t* b,
                                                   ckc_value_t* linear_id,
                                                   const ckc_streamk_partition_t* spec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_STREAMK_H */
