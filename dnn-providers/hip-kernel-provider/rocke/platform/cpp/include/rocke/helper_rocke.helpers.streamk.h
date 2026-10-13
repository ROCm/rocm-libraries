/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.streamk.h -- C99 port of four symbols from
 * rocke/helpers/streamk.py (StreamK tile partitioner helpers, CK Tile
 * parity):
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   class StreamKReductionStrategy      rocke_streamk_reduction_strategy_t (enum)
 *   class StreamKPartition (dataclass)  rocke_streamk_partition_t (struct) +
 *                                         rocke_streamk_partition_num_macro_tiles()
 *                                         rocke_streamk_partition_k_iters_per_output_tile()
 *   compute_streamk_grid_size(...)      rocke_compute_streamk_grid_size(...)
 *   emit_streamk_decode(...)            rocke_emit_streamk_decode(...)
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
#ifndef ROCKE_HELPER_ROCKE_HELPERS_STREAMK_H
#define ROCKE_HELPER_ROCKE_HELPERS_STREAMK_H

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Default chiplet / CU count for compute_streamk_grid_size (MI300X). */
#define ROCKE_STREAMK_DEFAULT_NUM_CUS 304
#define ROCKE_STREAMK_DEFAULT_BLOCKS_PER_CU 1

/* ------------------------------------------------------------------ *
 * StreamKReductionStrategy
 * ------------------------------------------------------------------ *
 * Python enum (values are the strategy strings, per CK Tile naming):
 *   Atomic    = "atomic"     # global_atomic_add_f32 into workspace
 *   Reduction = "reduction"  # cooperative + flag-table reduction
 */
typedef enum rocke_streamk_reduction_strategy
{
    ROCKE_STREAMK_REDUCTION_ATOMIC = 0, /* "atomic"    */
    ROCKE_STREAMK_REDUCTION_REDUCTION = 1 /* "reduction" */
} rocke_streamk_reduction_strategy_t;

/* The enum value string ("atomic" / "reduction"), or NULL if out of range. */
const char* rocke_streamk_reduction_strategy_value(rocke_streamk_reduction_strategy_t s);

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
 * rocke_emit_streamk_decode.
 */
typedef struct rocke_streamk_partition
{
    int m_tiles; /* ceil(M / tile_m) */
    int n_tiles; /* ceil(N / tile_n) */
    int k_iters; /* K / tile_k       */
} rocke_streamk_partition_t;

/* @property num_macro_tiles: m_tiles * n_tiles * k_iters -- total chunk count.
 * (Also exposed under the module name streamk_num_macro_tiles in Python.) */
int rocke_streamk_partition_num_macro_tiles(const rocke_streamk_partition_t* spec);

/* @property k_iters_per_output_tile: k_iters -- the number of CTAs that touch
 * each (m, n) output tile in the Atomic strategy. */
int rocke_streamk_partition_k_iters_per_output_tile(const rocke_streamk_partition_t* spec);

/* Module-level streamk_num_macro_tiles(spec): plain Python view of the
 * num_macro_tiles property. */
int rocke_streamk_num_macro_tiles(const rocke_streamk_partition_t* spec);

/* ------------------------------------------------------------------ *
 * compute_streamk_grid_size
 * ------------------------------------------------------------------ *
 * Python:
 *   def compute_streamk_grid_size(spec, *, num_cus=304, blocks_per_cu=1):
 *       if spec.num_macro_tiles <= 0:
 *           raise ValueError("spec has zero macro tiles")
 *       return min(spec.num_macro_tiles, num_cus * blocks_per_cu)
 *
 * Pass ROCKE_STREAMK_DEFAULT_NUM_CUS / ROCKE_STREAMK_DEFAULT_BLOCKS_PER_CU for the
 * keyword defaults. On the ValueError path, *out_status (if non-NULL) is set
 * to ROCKE_ERR_VALUE and the function returns -1; otherwise *out_status (if
 * non-NULL) is set to ROCKE_OK.
 */
int rocke_compute_streamk_grid_size(const rocke_streamk_partition_t* spec,
                                    int num_cus,
                                    int blocks_per_cu,
                                    rocke_status_t* out_status);

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
typedef struct rocke_streamk_decoded_tile
{
    rocke_value_t* m_tile;
    rocke_value_t* n_tile;
    rocke_value_t* k_iter;
    rocke_value_t* is_first;
    rocke_value_t* is_last;
} rocke_streamk_decoded_tile_t;

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
rocke_streamk_decoded_tile_t rocke_emit_streamk_decode(rocke_ir_builder_t* b,
                                                       rocke_value_t* linear_id,
                                                       const rocke_streamk_partition_t* spec);

/* ------------------------------------------------------------------ *
 * Iteration-balanced stream-K (CK Tile StreamKTilePartitioner)
 * ------------------------------------------------------------------ *
 *   Python                                C99 (this header)
 *   -----------------------------------   -----------------------------------
 *   class StreamKIterPartition            rocke_streamk_iter_partition_t
 *     (its @property values)                + rocke_streamk_iter_plan_t, resolved
 *                                             once by rocke_streamk_iter_plan()
 *   streamk_iter_partition(...)           rocke_streamk_iter_partition_make(...)
 *   StreamKIterPartition.flags_bytes()    rocke_streamk_flags_bytes(...)
 *   streamk_start_iter / streamk_end_iter rocke_streamk_start_iter / _end_iter
 *   StreamKIterArgs                       rocke_streamk_iter_args_t
 *   emit_streamk_sk_start_iter(...)       rocke_emit_streamk_sk_start_iter(...)
 *   emit_streamk_iter_range(...)          rocke_emit_streamk_iter_range(...)
 *
 * The derived quantities are returned as one resolved plan by value rather
 * than as per-property accessors, so the C side cannot recompute any of them
 * with different rounding than the Python @property it mirrors.
 */
typedef struct rocke_streamk_iter_partition
{
    int m_tiles;
    int n_tiles;
    int iters_per_tile;
    int max_active_wgs;
    bool persistent;
} rocke_streamk_iter_partition_t;

typedef struct rocke_streamk_iter_plan
{
    int num_tiles;
    int sk_tiles;
    int sk_ctas;
    int total_sk_iters;
    int iters_per_sk_cta; /* >= 1; clamped to 1 for an all-DP plan (a divisor) */
    int extra_iters;
    int dp_tiles;
    int total_dp_iters;
    int grid_size;
    int max_linear_partners;
    int tree_rounds;
} rocke_streamk_iter_plan_t;

/* streamk_iter_partition: every count must be > 0. On violation *out_status
 * (if non-NULL) is ROCKE_ERR_VALUE and the returned partition is zeroed. */
rocke_streamk_iter_partition_t rocke_streamk_iter_partition_make(int m_tiles,
                                                                 int n_tiles,
                                                                 int iters_per_tile,
                                                                 int max_active_wgs,
                                                                 bool persistent,
                                                                 rocke_status_t* out_status);

/* Every StreamKIterPartition @property, resolved once. */
rocke_streamk_iter_plan_t rocke_streamk_iter_plan(const rocke_streamk_iter_partition_t* p);

/* StreamKIterPartition.flags_bytes(n_waves): bytes of the stream-K fixup flag
 * table, one i32 flag per (stream-K CTA, wave) padded to 256 bytes; 0 for an
 * all-DP plan. n_waves is the waves per CTA that signal independently
 * (warp_m * warp_n for wgrad). n_waves <= 0 (Python ValueError) sets
 * *out_status (if non-NULL) to ROCKE_ERR_VALUE and returns 0. */
size_t rocke_streamk_flags_bytes(const rocke_streamk_iter_plan_t* r,
                                 int n_waves,
                                 rocke_status_t* out_status);

int rocke_streamk_start_iter(const rocke_streamk_iter_partition_t* p, int sk_cta);
int rocke_streamk_end_iter(const rocke_streamk_iter_partition_t* p, int sk_cta);

/* The runtime stream-K partition a shape-agnostic kernel reads (Python
 * StreamKIterArgs). The host computes a partition per launch and passes these
 * as kernel arguments; the emitters below only see them as SSA values. */
typedef struct rocke_streamk_iter_args
{
    rocke_value_t* iters_per_tile; /* i32 MAC iterations per output tile */
    rocke_value_t* dp_tiles; /* i32 tiles owned whole by data-parallel CTAs */
    rocke_value_t* total_dp_iters; /* i32 dp_tiles * iters_per_tile */
    rocke_value_t* iters_per_sk_cta; /* i32 base iterations per stream-K CTA (>= 1) */
    rocke_value_t* extra_iters; /* i32 stream-K CTAs that own one more */
} rocke_streamk_iter_args_t;

/* emit_streamk_sk_start_iter: SSA start iteration of a runtime SK CTA index. */
rocke_value_t* rocke_emit_streamk_sk_start_iter(rocke_ir_builder_t* b,
                                                rocke_value_t* sk_cta,
                                                const rocke_streamk_iter_args_t* a);

/* emit_streamk_iter_range result (Python _IterRange). */
typedef struct rocke_streamk_iter_range
{
    rocke_value_t* start; /* i32 global first iteration (SGPR) */
    rocke_value_t* end; /* i32 one past the last iteration (SGPR) */
    rocke_value_t* sk_cta; /* i32 stream-K CTA index; meaningless for a DP CTA */
} rocke_streamk_iter_range_t;

/* emit_streamk_iter_range(b, cta, args, *, dp). */
rocke_streamk_iter_range_t rocke_emit_streamk_iter_range(rocke_ir_builder_t* b,
                                                         rocke_value_t* cta,
                                                         const rocke_streamk_iter_args_t* a,
                                                         bool dp);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_STREAMK_H */
