/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_permute_nd.h -- C99 port of the N-D tensor permutation kernel
 * instance builder rocke/instances/common/permute_nd.py.
 *
 * Computes Y = np.transpose(X, perm): output axis d indexes input axis perm[d].
 * The permute is a pure gather/scatter (transform-DAG-driven flat offsets, no
 * LDS, no MFMA atom), so the IR is wave-size agnostic and arch-polymorphic;
 * `arch` is threaded through purely so the block-size bound is validated against
 * the target's max_threads_per_block.
 *
 *   Python (permute_nd.py)                C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class PermuteSpec                     rocke_permute_spec_t
 *   PermuteSpec.rank / y_shape /          rocke_permute_rank / _y_shape /
 *     total_elements / vec_width            _total_elements / _vec_width
 *   PermuteSpec.kernel_name()             rocke_permute_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_permute_is_valid_spec(...)
 *   build_permute(spec, arch)             rocke_build_permute(...)
 *   permute_grid(spec)                    rocke_permute_grid(...)
 *   permute_signature(spec)               rocke_permute_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_permute_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has computed
 * @property values (rank, y_shape, total_elements, vec_width); these are NOT
 * stored -- the accessor helpers recompute them exactly as the Python
 * properties do (including the perm[-1]==rank-1 vec-width predicate).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_PERMUTE_ND_H
#define ROCKE_INSTANCE_PERMUTE_ND_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CK Tile GenericPermuteHostArgs::kMaxRanks. */
#define ROCKE_PERMUTE_MAX_RANK 8

/* ------------------------------------------------------------------ PermuteSpec
 *
 * Mirror of Python PermuteSpec. `x_shape` is the input tensor shape (row-major
 * contiguous); `perm` is a length-`rank` permutation of range(rank). `dtype` is
 * "f16" or "bf16". `name` defaults to "rocke_permute" in the Python dataclass;
 * use rocke_permute_spec_default() to get that default. */
typedef struct rocke_permute_spec
{
    int x_shape[ROCKE_PERMUTE_MAX_RANK];
    int perm[ROCKE_PERMUTE_MAX_RANK];
    int rank; /* len(x_shape) == len(perm)              */
    const char* dtype; /* "f16" | "bf16"  (default "f16")        */
    int block_size; /* default 256                            */
    const char* name; /* default "rocke_permute"               */
} rocke_permute_spec_t;

/* Default-constructed spec (dtype="f16", block_size=256, name="rocke_permute",
 * rank=0). The caller must still fill x_shape/perm/rank. */
rocke_permute_spec_t rocke_permute_spec_default(void);

/* PermuteSpec.rank @property. */
int rocke_permute_rank(const rocke_permute_spec_t* spec);

/* PermuteSpec.y_shape @property: y_shape[d] = x_shape[perm[d]]. Writes `rank`
 * extents into out (capacity ROCKE_PERMUTE_MAX_RANK) and returns the rank. */
int rocke_permute_y_shape(const rocke_permute_spec_t* spec, int out[ROCKE_PERMUTE_MAX_RANK]);

/* PermuteSpec.total_elements @property: product of x_shape. */
int rocke_permute_total_elements(const rocke_permute_spec_t* spec);

/* PermuteSpec.vec_width @property. Returns the largest of {8,4,2} that divides
 * both inner length and total_elements when perm[-1]==rank-1, else 1. */
int rocke_permute_vec_width(const rocke_permute_spec_t* spec);

/* PermuteSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t
    rocke_permute_kernel_name(const rocke_permute_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and false is returned. On accept returns true and writes "ok". */
bool rocke_permute_is_valid_spec(const rocke_permute_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap);

/* build_permute(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success or NULL with b's sticky error set.
 * `arch` NULL => "gfx950". This expects the builder to have been created with
 * the spec's kernel_name(); use rocke_build_permute_new() for the convenience. */
rocke_kernel_def_t*
    rocke_build_permute(rocke_ir_builder_t* b, const rocke_permute_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_permute_new(rocke_ir_builder_t* b,
                                            const rocke_permute_spec_t* spec,
                                            const char* arch);

/* permute_grid(spec) -> (x, y, z) launch grid. threads = ceil(total / vec);
 * grid = ceil_div_grid((threads, block_size)). Writes out[0..2]; returns ROCKE_OK
 * or ROCKE_ERR_VALUE on an invalid (non-positive block_size etc.) spec. */
rocke_status_t rocke_permute_grid(const rocke_permute_spec_t* spec, int out[3]);

/* permute_signature(spec) -> two pointer entries (X, Y) of dtype, built via the
 * spec.h SignatureBuilder (.ptr("X",dtype).ptr("Y",dtype).build()). Writes up to
 * `out_cap` entries into out[] and sets *out_count to the number produced (2).
 * `arena` backs the builder's string/array storage. Returns ROCKE_OK, or
 * ROCKE_ERR_VALUE on a too-small out_cap / NULL args. */
rocke_status_t rocke_permute_signature(rocke_arena_t* arena,
                                       const rocke_permute_spec_t* spec,
                                       rocke_sig_entry_t* out,
                                       size_t out_cap,
                                       size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Owns/frees its
 * IRBuilder internally. */
rocke_status_t rocke_permute_lower_to_llvm(const rocke_permute_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_PERMUTE_ND_H */
