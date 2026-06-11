/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/passes.h -- C99 port of ck_dsl.core.passes.
 *
 * Conservative IR canonicalization passes that run over the FROZEN IR
 * (ckc/ir.h) before LLVM lowering. They are small on purpose: only pure
 * scalar/vector bookkeeping ops are folded, CSE'd, or removed. Loads, stores,
 * barriers, async copies, and MFMA ops are never moved or removed.
 *
 *   Python                       C99 (this header)
 *   --------------------------   --------------------------------------------
 *   @dataclass PassStats         ckc_pass_stats_t
 *   PassStats.__add__            ckc_pass_stats_add()
 *   optimize_kernel(kernel,*)    ckc_optimize_kernel()
 *   canonicalize_region(region)  ckc_canonicalize_region()
 *   eliminate_dead_pure_ops(r)   ckc_eliminate_dead_pure_ops()
 *
 * Lifetime / allocation: the passes mutate the IR in place (rewriting operand
 * arrays, rebuilding region op lists, replacing folded ops). Any fresh array or
 * attr storage required is taken from the builder's arena, so the passes take a
 * ckc_ir_builder_t* (the builder that owns the kernel graph), mirroring the rest
 * of the port. The builder's sticky error is set on OOM.
 */
#ifndef CKC_PASSES_H
#define CKC_PASSES_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Counters returned by each pass (Python @dataclass(frozen=True) PassStats). */
typedef struct ckc_pass_stats
{
    int constants_folded;
    int common_subexpressions;
    int dead_ops_removed;
} ckc_pass_stats_t;

/* PassStats.__add__: component-wise sum. */
ckc_pass_stats_t ckc_pass_stats_add(ckc_pass_stats_t a, ckc_pass_stats_t b);

/* True iff all counters are zero (Python `stats == PassStats()`). */
bool ckc_pass_stats_is_zero(ckc_pass_stats_t s);

/* Run the default conservative pass pipeline in-place over kernel->body.
 * Iterates canonicalize_region up to max_iter times, stopping early once a
 * round makes no changes. Returns the accumulated stats. `max_iter <= 0` is
 * treated as the Python default of 3. The `b` arena owns the kernel graph. */
ckc_pass_stats_t ckc_optimize_kernel(ckc_ir_builder_t* b, ckc_kernel_def_t* kernel, int max_iter);

/* Fold constants, CSE pure ops, and remove dead pure ops in `region`
 * (recursing into nested regions first). Mutates the region in place. */
ckc_pass_stats_t ckc_canonicalize_region(ckc_ir_builder_t* b, ckc_region_t* region);

/* Remove pure ops whose every result is unused (recursing into nested regions).
 * Returns the number of ops removed. Mutates the region in place. */
int ckc_eliminate_dead_pure_ops(ckc_ir_builder_t* b, ckc_region_t* region);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_PASSES_H */
