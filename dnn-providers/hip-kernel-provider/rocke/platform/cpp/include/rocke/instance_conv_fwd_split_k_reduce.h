/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_split_k_reduce.h
 *
 * C99 port of:
 *   library/kernels/common/conv_fwd_split_k_reduce.py
 *   (FwdSplitKReduceSpec + build_conv_fwd_split_k_reduce)
 *
 * Stage 2 of the split-K forward convolution. Stage 1 (the implicit-GEMM
 * forward with split_k > 1) stores each K slice's fp32 partial output into its
 * own slab of a [ks_count][M][K] workspace -- plain stores, every element
 * written exactly once. This kernel sums the ks_count slabs element-wise in a
 * fixed order and converts the result to dtype_d into D (N[D]HWK, the same
 * packed layout as one slab). The result is deterministic.
 *
 * Kernel signature:
 *   ws_ptr   : f32 global ptr (readonly)  -- workspace [ks_count * elems]
 *   d_ptr    : dtype_d global ptr         -- output [elems]
 *   elems    : i32 -- elements per slab (M * K)
 *   ws_bytes : i32 -- workspace byte size (ABI boundary)
 *   d_bytes  : i32 -- output byte size (ABI boundary)
 *   ks_count : i32 -- number of slabs (the split-K degree)
 *
 * Grid: (ceil(elems / (vec * block_size)), 1, 1); block (block_size, 1, 1).
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_SPLIT_K_REDUCE_H
#define ROCKE_INSTANCE_CONV_FWD_SPLIT_K_REDUCE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h" /* for rocke_llvm_flavor_t */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------- FwdSplitKReduceSpec
 *
 * Mirror of Python FwdSplitKReduceSpec (frozen dataclass):
 *
 *   dtype_d:    str = "bf16"
 *   vec:        int = 4     -- output elements per thread (1, 2 or 4)
 *   block_size: int = 256
 *   degree:     int = 0     -- > 0 unrolls the fold for exactly that many
 *                              slabs (ks_count must then equal it); 0 keeps
 *                              the runtime loop over ks_count
 *   name:       str = "conv_fwd_splitk_reduce"
 */
typedef struct rocke_fwd_split_k_reduce_spec
{
    const char* dtype_d; /* default "bf16" */
    int vec; /* default 4 */
    int block_size; /* default 256 */
    int degree; /* default 0 */
    const char* name; /* default "conv_fwd_splitk_reduce" */
} rocke_fwd_split_k_reduce_spec_t;

/* Default-constructed spec (every field == the Python dataclass default). */
rocke_fwd_split_k_reduce_spec_t rocke_fwd_split_k_reduce_spec_default(void);

/* spec.kernel_name() -> NUL-terminated into out (capacity out_cap). */
rocke_status_t rocke_fwd_split_k_reduce_kernel_name(const rocke_fwd_split_k_reduce_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);

/* spec.validate(). Returns true if valid; false + reason string otherwise. */
bool rocke_fwd_split_k_reduce_is_valid_spec(const rocke_fwd_split_k_reduce_spec_t* spec,
                                            char* reason,
                                            size_t reason_cap);

/* Build the reduction kernel IR into the (initialised) builder b.
 * Returns b->kernel on success, NULL on error (check b's sticky error). */
rocke_kernel_def_t* rocke_build_conv_fwd_split_k_reduce(rocke_ir_builder_t* b,
                                                        const rocke_fwd_split_k_reduce_spec_t* spec,
                                                        const char* arch);

/* Init b with spec.kernel_name(), then build. */
rocke_kernel_def_t* rocke_build_conv_fwd_split_k_reduce_new(
    rocke_ir_builder_t* b, const rocke_fwd_split_k_reduce_spec_t* spec, const char* arch);

/* Launch grid: (max(1, ceil(elems / (vec * block_size))), 1, 1). */
void rocke_fwd_split_k_reduce_grid(const rocke_fwd_split_k_reduce_spec_t* spec,
                                   long long elems,
                                   int* out_gx,
                                   int* out_gy,
                                   int* out_gz);

/* The kernel signature (6 entries: ws_ptr, d_ptr, elems, ws_bytes, d_bytes,
 * ks_count). arena must not be NULL. */
rocke_status_t rocke_fwd_split_k_reduce_signature(rocke_arena_t* arena,
                                                  const rocke_fwd_split_k_reduce_spec_t* spec,
                                                  const rocke_sig_entry_t** out_items,
                                                  size_t* out_count);

/* Convenience: build + lower to LLVM IR in one shot. */
rocke_status_t rocke_fwd_split_k_reduce_lower_to_llvm(const rocke_fwd_split_k_reduce_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_INSTANCE_CONV_FWD_SPLIT_K_REDUCE_H */
