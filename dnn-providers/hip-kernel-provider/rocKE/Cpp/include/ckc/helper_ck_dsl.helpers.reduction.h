/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.reduction.h -- C99 port of selected symbols from
 * ck_dsl/helpers/reduction.py.
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
 * is accumulating from. The barrier between halving steps is ckc_b_sync().
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). These
 * helpers emit IR via the ckc_b_* builder API and bind only to ckc/ir.h's public
 * surface, byte-faithfully reproducing the Python builder-call sequence.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_REDUCTION_H
#define CKC_HELPER_CK_DSL_HELPERS_REDUCTION_H

#include <stdbool.h>

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- combiners */

/* ReduceCombine = Literal["sum", "max", "min", "prod"]. */
typedef enum ckc_reduce_combine
{
    CKC_REDUCE_SUM = 0,
    CKC_REDUCE_MAX,
    CKC_REDUCE_MIN,
    CKC_REDUCE_PROD
} ckc_reduce_combine_t;

/* IndexCombine = Literal["argmax", "argmin"]. */
typedef enum ckc_index_combine
{
    CKC_INDEX_ARGMAX = 0,
    CKC_INDEX_ARGMIN
} ckc_index_combine_t;

/* ------------------------------------------------------- compile-time const */

/* Per-thread register-tile capacity for the row-norm family
 * (REGISTER_TILE_MAX_ELEMS_PER_THREAD = 64). */
#define CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD 64

/* --------------------------------------------------------- host-side select */

/* row_norm_needs_two_pass(elems_per_thread, max_cached=64) analogue.
 *
 * Returns true when elems_per_thread exceeds the per-thread register-tile
 * capacity (so caching the whole row in VGPRs would overflow the budget) and
 * the kernel must re-stream X from HBM in pass 2. Emits NO IR. */
bool ckc_row_norm_needs_two_pass(int elems_per_thread, int max_cached);

/* ------------------------------------------------------------ tree_reduce */

/* The binary combiner callback emitted at each tree node, e.g. ckc_b_fadd /
 * ckc_b_fmax. ``user`` is an opaque cookie passed through unchanged. */
typedef ckc_value_t* (*ckc_combine_fn)(ckc_ir_builder_t* b,
                                       ckc_value_t* a,
                                       ckc_value_t* c,
                                       void* user);

/* tree_reduce(b, combine, xs) analogue: balanced binary-tree fold of n scalars
 * (depth ~ log2 n). Pairs xs[i]/xs[i+1] left-to-right, carrying any odd tail
 * element forward unchanged. Returns NULL (and sets sticky error) when n < 1.
 * ``user`` is forwarded to every ``combine`` invocation. */
ckc_value_t* ckc_tree_reduce(
    ckc_ir_builder_t* b, ckc_combine_fn combine, void* user, ckc_value_t* const* xs, int n);

/* --------------------------------------------------------- block_lds_reduce */

/* block_lds_reduce(b, val, lds_buf, tid, block_size, combine) analogue.
 *
 * LDS tree reduction across all ``block_size`` lanes. ``val`` is the per-thread
 * f32 partial; ``lds_buf`` is a block_size x f32 LDS allocation owned by the
 * caller. The reduced value is broadcast back to every lane. Returns NULL (and
 * sets sticky error) on a bad combine or non-f32 input. */
ckc_value_t* ckc_block_lds_reduce(ckc_ir_builder_t* b,
                                  ckc_value_t* val,
                                  ckc_value_t* lds_buf,
                                  ckc_value_t* tid,
                                  int block_size,
                                  ckc_reduce_combine_t combine);

/* block_lds_reduce_pair(...) analogue: twin-channel block reduction sharing one
 * barrier schedule. Functionally two back-to-back block_lds_reduce calls, but
 * interleaved inside a single halving loop. Writes the two broadcast results to
 * *out_a / *out_c. Returns 1 on success, 0 (sticky error) on non-f32 input. */
int ckc_block_lds_reduce_pair(ckc_ir_builder_t* b,
                              ckc_value_t* val_a,
                              ckc_value_t* val_c,
                              ckc_value_t* lds_a,
                              ckc_value_t* lds_c,
                              ckc_value_t* tid,
                              int block_size,
                              ckc_reduce_combine_t combine_a,
                              ckc_reduce_combine_t combine_c,
                              ckc_value_t** out_a,
                              ckc_value_t** out_c);

/* block_lds_reduce_with_wave_prologue(...) analogue: wave-XOR butterfly +
 * cross-warp LDS. Six (for wave_size=64) cross-lane shuffle stages with no LDS,
 * then one sync over a num_warps-slot scratch in ``lds_buf``. Returns NULL (and
 * sets sticky error) on non-f32 input. */
ckc_value_t* ckc_block_lds_reduce_with_wave_prologue(ckc_ir_builder_t* b,
                                                     ckc_value_t* val,
                                                     ckc_value_t* lds_buf,
                                                     ckc_value_t* tid,
                                                     int block_size,
                                                     ckc_reduce_combine_t combine,
                                                     int wave_size);

/* welford_block_reduce(...) analogue: numerically-stable mean/variance via the
 * fused (sum, sum_sq) pair fold. ``count_val`` is the compile-time per-thread
 * element count. Writes mean to *out_mean, var to *out_var. Returns 1 on
 * success, 0 (sticky error) on failure. */
int ckc_welford_block_reduce(ckc_ir_builder_t* b,
                             ckc_value_t* sum_val,
                             ckc_value_t* sum_sq_val,
                             int count_val,
                             ckc_value_t* lds_sum,
                             ckc_value_t* lds_sumsq,
                             ckc_value_t* tid,
                             int block_size,
                             ckc_value_t** out_mean,
                             ckc_value_t** out_var);

/* welford_block_reduce_stable(...) analogue: count-weighted (mean, M2, count)
 * parallel merge (CK BlockwiseWelford::Merge). Each thread supplies its own
 * partial Welford triple (all f32, count as an f32 Value). Writes mean to
 * *out_mean, var = M2_total/count_total to *out_var. Returns 1 on success,
 * 0 (sticky error) on non-f32 input. */
int ckc_welford_block_reduce_stable(ckc_ir_builder_t* b,
                                    ckc_value_t* mean_val,
                                    ckc_value_t* m2_val,
                                    ckc_value_t* count_val,
                                    ckc_value_t* lds_mean,
                                    ckc_value_t* lds_m2,
                                    ckc_value_t* lds_count,
                                    ckc_value_t* tid,
                                    int block_size,
                                    ckc_value_t** out_mean,
                                    ckc_value_t** out_var);

/* block_lds_reduce_with_index(...) analogue: LDS tree reduction carrying both
 * value (f32) and index (i32) for argmax / argmin. Uses the CK doubling tree so
 * ties resolve to the LOWEST index. Writes value to *out_val, index to
 * *out_idx. Returns 1 on success, 0 (sticky error) on bad combine, non-f32 val,
 * non-i32 idx, or non-power-of-two block_size. */
int ckc_block_lds_reduce_with_index(ckc_ir_builder_t* b,
                                    ckc_value_t* val,
                                    ckc_value_t* idx,
                                    ckc_value_t* lds_val,
                                    ckc_value_t* lds_idx,
                                    ckc_value_t* tid,
                                    int block_size,
                                    ckc_index_combine_t combine,
                                    ckc_value_t** out_val,
                                    ckc_value_t** out_idx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_REDUCTION_H */
