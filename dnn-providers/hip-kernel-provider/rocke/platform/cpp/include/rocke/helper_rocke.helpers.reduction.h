/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.reduction.h -- C99 port of selected symbols from
 * rocke/helpers/reduction.py.
 *
 * Block-level reductions, lifted from the inline copies in norm/reduce kernels.
 * The DSL counterpart is a thin LDS tree reduction over a single f32 broadcast
 * value: each thread writes its partial to an LDS scratch buffer, the reduction
 * halves the active lane set each step, and the final value at index 0 is
 * broadcast back to every lane.
 *
 * PORTED SYMBOLS (this phase):
 *   - REGISTER_TILE_MAX_ELEMS_PER_THREAD  (compile-time constant)
 *   - ReduceCombine                        (combiner enum: sum/max/min/prod)
 *   - row_norm_needs_two_pass              (pure host-side BUILD-time selector)
 *   - tree_reduce                          (balanced binary-tree fold)
 *   - block_lds_reduce                     (LDS tree reduce, broadcast)
 *   - block_lds_reduce_pair                (twin-channel, one barrier schedule)
 *   - block_lds_reduce_with_wave_prologue  (wave-XOR + cross-warp LDS)
 *   - block_lds_reduce_with_index          (argmax / argmin tree)
 *   - welford_block_reduce                 (mean/var via fused pair fold)
 *   - welford_block_reduce_stable          (count-weighted Welford triple)
 *
 * The IndexCombine ("argmax"/"argmin") combiner needed by
 * block_lds_reduce_with_index is also exposed here.
 *
 * The combiners are applied in f32 regardless of the storage dtype the caller
 * is accumulating from. The barrier between halving steps is rocke_b_sync().
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). These
 * helpers emit IR via the rocke_b_* builder API and bind only to rocke/ir.h's public
 * surface, byte-faithfully reproducing the Python builder-call sequence.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_REDUCTION_H
#define ROCKE_HELPER_ROCKE_HELPERS_REDUCTION_H

#include <stdbool.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- combiners */

/* ReduceCombine = Literal["sum", "max", "min", "prod"]. */
typedef enum rocke_reduce_combine
{
    ROCKE_REDUCE_SUM = 0,
    ROCKE_REDUCE_MAX,
    ROCKE_REDUCE_MIN,
    ROCKE_REDUCE_PROD
} rocke_reduce_combine_t;

/* IndexCombine = Literal["argmax", "argmin"]. */
typedef enum rocke_index_combine
{
    ROCKE_INDEX_ARGMAX = 0,
    ROCKE_INDEX_ARGMIN
} rocke_index_combine_t;

/* ------------------------------------------------------- compile-time const */

/* Per-thread register-tile capacity for the row-norm family
 * (REGISTER_TILE_MAX_ELEMS_PER_THREAD = 64). */
#define ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD 64

/* --------------------------------------------------------- host-side select */

/* row_norm_needs_two_pass(elems_per_thread, max_cached=64) analogue.
 *
 * Returns true when elems_per_thread exceeds the per-thread register-tile
 * capacity (so caching the whole row in VGPRs would overflow the budget) and
 * the kernel must re-stream X from HBM in pass 2. Emits NO IR. */
bool rocke_row_norm_needs_two_pass(int elems_per_thread, int max_cached);

/* ------------------------------------------------------------ tree_reduce */

/* The binary combiner callback emitted at each tree node, e.g. rocke_b_fadd /
 * rocke_b_fmax. ``user`` is an opaque cookie passed through unchanged. */
typedef rocke_value_t* (*rocke_combine_fn)(rocke_ir_builder_t* b,
                                           rocke_value_t* a,
                                           rocke_value_t* c,
                                           void* user);

/* tree_reduce(b, combine, xs) analogue: balanced binary-tree fold of n scalars
 * (depth ~ log2 n). Pairs xs[i]/xs[i+1] left-to-right, carrying any odd tail
 * element forward unchanged. Returns NULL (and sets sticky error) when n < 1.
 * ``user`` is forwarded to every ``combine`` invocation. */
rocke_value_t* rocke_tree_reduce(
    rocke_ir_builder_t* b, rocke_combine_fn combine, void* user, rocke_value_t* const* xs, int n);

/* --------------------------------------------------------- block_lds_reduce */

/* block_lds_reduce(b, val, lds_buf, tid, block_size, combine) analogue.
 *
 * LDS tree reduction across all ``block_size`` lanes. ``val`` is the per-thread
 * f32 partial; ``lds_buf`` is a block_size x f32 LDS allocation owned by the
 * caller. The reduced value is broadcast back to every lane. Returns NULL (and
 * sets sticky error) on a bad combine or non-f32 input. */
rocke_value_t* rocke_block_lds_reduce(rocke_ir_builder_t* b,
                                      rocke_value_t* val,
                                      rocke_value_t* lds_buf,
                                      rocke_value_t* tid,
                                      int block_size,
                                      rocke_reduce_combine_t combine);

/* block_lds_reduce_pair(...) analogue: twin-channel block reduction sharing one
 * barrier schedule. Functionally two back-to-back block_lds_reduce calls, but
 * interleaved inside a single halving loop. Writes the two broadcast results to
 * *out_a / *out_c. Returns 1 on success, 0 (sticky error) on non-f32 input. */
int rocke_block_lds_reduce_pair(rocke_ir_builder_t* b,
                                rocke_value_t* val_a,
                                rocke_value_t* val_c,
                                rocke_value_t* lds_a,
                                rocke_value_t* lds_c,
                                rocke_value_t* tid,
                                int block_size,
                                rocke_reduce_combine_t combine_a,
                                rocke_reduce_combine_t combine_c,
                                rocke_value_t** out_a,
                                rocke_value_t** out_c);

/* block_lds_reduce_with_wave_prologue(...) analogue: wave-XOR butterfly +
 * cross-warp LDS. Six (for wave_size=64) cross-lane shuffle stages with no LDS,
 * then one sync over a num_warps-slot scratch in ``lds_buf``. Returns NULL (and
 * sets sticky error) on non-f32 input. */
rocke_value_t* rocke_block_lds_reduce_with_wave_prologue(rocke_ir_builder_t* b,
                                                         rocke_value_t* val,
                                                         rocke_value_t* lds_buf,
                                                         rocke_value_t* tid,
                                                         int block_size,
                                                         rocke_reduce_combine_t combine,
                                                         int wave_size);

/* welford_block_reduce(...) analogue: numerically-stable mean/variance via the
 * fused (sum, sum_sq) pair fold. ``count_val`` is the compile-time per-thread
 * element count. Writes mean to *out_mean, var to *out_var. Returns 1 on
 * success, 0 (sticky error) on failure. */
int rocke_welford_block_reduce(rocke_ir_builder_t* b,
                               rocke_value_t* sum_val,
                               rocke_value_t* sum_sq_val,
                               int count_val,
                               rocke_value_t* lds_sum,
                               rocke_value_t* lds_sumsq,
                               rocke_value_t* tid,
                               int block_size,
                               rocke_value_t** out_mean,
                               rocke_value_t** out_var);

/* welford_block_reduce_stable(...) analogue: count-weighted (mean, M2, count)
 * parallel merge (CK BlockwiseWelford::Merge). Each thread supplies its own
 * partial Welford triple (all f32, count as an f32 Value). Writes mean to
 * *out_mean, var = M2_total/count_total to *out_var. Returns 1 on success,
 * 0 (sticky error) on non-f32 input. */
int rocke_welford_block_reduce_stable(rocke_ir_builder_t* b,
                                      rocke_value_t* mean_val,
                                      rocke_value_t* m2_val,
                                      rocke_value_t* count_val,
                                      rocke_value_t* lds_mean,
                                      rocke_value_t* lds_m2,
                                      rocke_value_t* lds_count,
                                      rocke_value_t* tid,
                                      int block_size,
                                      rocke_value_t** out_mean,
                                      rocke_value_t** out_var);

/* block_lds_reduce_with_index(...) analogue: LDS tree reduction carrying both
 * value (f32) and index (i32) for argmax / argmin. Uses the CK doubling tree so
 * ties resolve to the LOWEST index. Writes value to *out_val, index to
 * *out_idx. Returns 1 on success, 0 (sticky error) on bad combine, non-f32 val,
 * non-i32 idx, or non-power-of-two block_size. */
int rocke_block_lds_reduce_with_index(rocke_ir_builder_t* b,
                                      rocke_value_t* val,
                                      rocke_value_t* idx,
                                      rocke_value_t* lds_val,
                                      rocke_value_t* lds_idx,
                                      rocke_value_t* tid,
                                      int block_size,
                                      rocke_index_combine_t combine,
                                      rocke_value_t** out_val,
                                      rocke_value_t** out_idx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_REDUCTION_H */
