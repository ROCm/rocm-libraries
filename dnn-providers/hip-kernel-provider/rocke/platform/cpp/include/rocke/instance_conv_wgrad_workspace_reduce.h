/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_wgrad_workspace_reduce.h
 *
 * C99 port of:
 *   rocke/instances/common/conv_wgrad_workspace_reduce.py
 *   (WgradReduceSpec + build_conv_wgrad_workspace_reduce)
 *
 * This is Stage 2 of the deterministic two-stage wgrad path.  It reads the
 * f32 partial sums written by Stage 1 (two_stage=True wgrad kernel) from a
 * workspace buffer of shape [split_k, wg_M, wg_N] and reduces them along the
 * split_k axis in a fixed sequential order (k_id = 0, 1, ..., split_k - 1),
 * then stores the result as dtype_d to the dW output tensor.
 *
 * The fixed loop order is what guarantees bit-exact, deterministic output:
 * floating-point summation order is fully determined by the memory layout.
 *
 * Kernel signature:
 *   ws_ptr  : f32 global ptr (readonly)  -- workspace [split_k * wg_M * wg_N]
 *   dw_ptr  : dtype_d global ptr (writeonly) -- weight gradient output
 *   wg_M    : i32
 *   wg_N    : i32
 *   split_k : i32
 *   ws_bytes: i32  -- ABI boundary; not used for bounds checking in the kernel body
 *   dw_bytes: i32  -- ABI boundary; not used for bounds checking in the kernel body
 *
 * Grid: (ceil(wg_N / tile_n), ceil(wg_M / tile_m), 1)
 * Block: (tile_m * tile_n, 1, 1)
 */
#ifndef ROCKE_INSTANCE_CONV_WGRAD_WORKSPACE_REDUCE_H
#define ROCKE_INSTANCE_CONV_WGRAD_WORKSPACE_REDUCE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h" /* for rocke_llvm_flavor_t */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------- WgradReduceSpec
 *
 * Mirror of Python WgradReduceSpec (frozen dataclass):
 *
 *   problem: ConvProblem          -- needed only for wg_M / wg_N / kernel name
 *   dtype_d: str = "fp16"
 *   tile_m:  int = 4
 *   tile_n:  int = 64
 *   name:    str = "conv_wgrad_ws_reduce"
 */
typedef struct rocke_wgrad_reduce_spec
{
    /* Convolution geometry (only wg_M and wg_N are needed at build time). */
    int wg_M; /* _wg_M(problem) = K */
    int wg_N; /* _wg_N(problem) = Y*X*C */
    const char* problem_short; /* short string for kernel name, e.g. "N2H14W14C16_K16Y3X3" */

    const char* dtype_d; /* default "fp16" -- output dtype for dW */
    int tile_m; /* default 4 */
    int tile_n; /* default 64 */
    const char* name; /* default "conv_wgrad_ws_reduce" */
} rocke_wgrad_reduce_spec_t;

/* Default-constructed spec.  Caller must set wg_M, wg_N, problem_short. */
rocke_wgrad_reduce_spec_t rocke_wgrad_reduce_spec_default(void);

/* block_size = tile_m * tile_n */
int rocke_wgrad_reduce_spec_block_size(const rocke_wgrad_reduce_spec_t* spec);

/* Write NUL-terminated kernel name into out (capacity out_cap). */
rocke_status_t
    rocke_wgrad_reduce_kernel_name(const rocke_wgrad_reduce_spec_t* spec, char* out, int out_cap);

/* Validate the spec.  Returns true if valid; false + reason string otherwise. */
bool rocke_wgrad_reduce_is_valid_spec(const rocke_wgrad_reduce_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      int reason_cap);

/* Build the reduction kernel IR into builder b.
 * Returns b->kernel on success, NULL on error (check b's sticky error). */
rocke_kernel_def_t* rocke_build_wgrad_workspace_reduce(rocke_ir_builder_t* b,
                                                       const rocke_wgrad_reduce_spec_t* spec,
                                                       const char* arch);

/* Init builder + build in one call (allocates a fresh builder). */
rocke_kernel_def_t* rocke_build_wgrad_workspace_reduce_new(rocke_ir_builder_t* b,
                                                           const rocke_wgrad_reduce_spec_t* spec,
                                                           const char* arch);

/* Launch grid: (ceil(wg_N / tile_n), ceil(wg_M / tile_m), 1) */
void rocke_wgrad_reduce_grid(const rocke_wgrad_reduce_spec_t* spec,
                             int* out_gx,
                             int* out_gy,
                             int* out_gz);

/* Build the kernel signature (7 entries: ws_ptr, dw_ptr, wg_M, wg_N,
 * split_k, ws_bytes, dw_bytes). arena must not be NULL. */
rocke_status_t rocke_wgrad_reduce_signature(rocke_arena_t* arena,
                                            const rocke_wgrad_reduce_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count);

/* Convenience: build + lower to LLVM IR in one shot. */
rocke_status_t rocke_wgrad_reduce_lower_to_llvm(const rocke_wgrad_reduce_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_INSTANCE_CONV_WGRAD_WORKSPACE_REDUCE_H */
