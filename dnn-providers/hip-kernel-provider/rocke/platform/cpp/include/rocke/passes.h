/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/passes.h -- C99 port of rocke.core.passes.
 *
 * Conservative IR canonicalization passes that run over the FROZEN IR
 * (rocke/ir.h) before LLVM lowering. They are small on purpose: only pure
 * scalar/vector bookkeeping ops are folded, CSE'd, or removed. Loads, stores,
 * barriers, async copies, and MFMA ops are never moved or removed.
 *
 *   Python                       C99 (this header)
 *   --------------------------   --------------------------------------------
 *   @dataclass PassStats         rocke_pass_stats_t
 *   PassStats.__add__            rocke_pass_stats_add()
 *   optimize_kernel(kernel,*)    rocke_optimize_kernel()
 *   canonicalize_region(region)  rocke_canonicalize_region()
 *   eliminate_dead_pure_ops(r)   rocke_eliminate_dead_pure_ops()
 *
 * Lifetime / allocation: the passes mutate the IR in place (rewriting operand
 * arrays, rebuilding region op lists, replacing folded ops). Any fresh array or
 * attr storage required is taken from the builder's arena, so the passes take a
 * rocke_ir_builder_t* (the builder that owns the kernel graph), mirroring the rest
 * of the port. The builder's sticky error is set on OOM.
 */
#ifndef ROCKE_PASSES_H
#define ROCKE_PASSES_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Counters returned by each pass (Python @dataclass(frozen=True) PassStats). */
typedef struct rocke_pass_stats
{
    int constants_folded;
    int common_subexpressions;
    int dead_ops_removed;
} rocke_pass_stats_t;

/* PassStats.__add__: component-wise sum. */
rocke_pass_stats_t rocke_pass_stats_add(rocke_pass_stats_t a, rocke_pass_stats_t b);

/* True iff all counters are zero (Python `stats == PassStats()`). */
bool rocke_pass_stats_is_zero(rocke_pass_stats_t s);

/* Run the default conservative pass pipeline in-place over kernel->body.
 * Iterates canonicalize_region up to max_iter times, stopping early once a
 * round makes no changes. Returns the accumulated stats. `max_iter <= 0` is
 * treated as the Python default of 3. The `b` arena owns the kernel graph. */
rocke_pass_stats_t
    rocke_optimize_kernel(rocke_ir_builder_t* b, rocke_kernel_def_t* kernel, int max_iter);

/* Fold constants, CSE pure ops, and remove dead pure ops in `region`
 * (recursing into nested regions first). Mutates the region in place. */
rocke_pass_stats_t rocke_canonicalize_region(rocke_ir_builder_t* b, rocke_region_t* region);

/* Remove pure ops whose every result is unused (recursing into nested regions).
 * Returns the number of ops removed. Mutates the region in place. */
int rocke_eliminate_dead_pure_ops(rocke_ir_builder_t* b, rocke_region_t* region);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_PASSES_H */
