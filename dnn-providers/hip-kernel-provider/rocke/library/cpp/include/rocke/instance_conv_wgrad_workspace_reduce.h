/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_wgrad_workspace_reduce.h
 *
 * C99 port of:
 *   rocke/instances/common/conv_wgrad_workspace_reduce.py
 *   (WgradReduceSpec + build_conv_wgrad_workspace_reduce)
 *
 * This is Stage 2 of the two-stage wgrad path.  Stage 1 (two_stage=true)
 * f32-atomic-added its partial sums into a scratch of shape
 * [groups * R, wg_M, wg_N], so the reduction over split_k is already complete
 * when this runs.  This kernel folds the R replica slabs and converts
 * f32 -> dtype_d into dW.
 *
 * Kernel signature:
 *   ws_ptr  : f32 global ptr (readonly)  -- scratch [groups * R * wg_M * wg_N]
 *   dw_ptr  : dtype_d global ptr (writeonly) -- weight gradient output [groups * wg_M * wg_N]
 *   wg_M    : i32  -- per-group K dimension
 *   wg_N    : i32  -- per-group Y*X*C dimension
 *   ws_bytes: i32  -- ABI boundary; not used for bounds checking in the kernel body
 *   dw_bytes: i32  -- ABI boundary; not used for bounds checking in the kernel body
 *   groups  : i32  -- number of convolution groups (1 for non-grouped)
 *
 * Grid: (ceil(wg_N / tile_n), ceil(wg_M / tile_m), groups)
 * Block: (tile_m * tile_n, 1, 1)
 * block_id_z encodes the group index; each CTA folds its group's R scratch
 * slabs and writes into the corresponding per-group dW slab.
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
 *   name:    str = "conv_wgrad_ws_cast"
 *   groups:  int = 1             -- number of convolution groups
 *   ws_replicas: int = 8         -- scratch replica slabs to fold
 */
typedef struct rocke_wgrad_reduce_spec
{
    /* Convolution geometry (only wg_M and wg_N are needed at build time). */
    int wg_M; /* per-group _wg_M(problem) = K // groups */
    int wg_N; /* per-group _wg_N(problem) = Y*X * (C // groups) */
    const char* problem_short; /* short string for kernel name, e.g. "N2H14W14C16_K16Y3X3" */

    const char* dtype_d; /* default "fp16" -- output dtype for dW */
    int tile_m; /* default 4 */
    int tile_n; /* default 64 */
    const char* name; /* default "conv_wgrad_ws_cast" */
    int groups; /* default 1 -- number of convolution groups; grid z = groups */
    /* default 8 -- scratch replica slabs to fold.  Must match the Stage 1
     * spec's ws_replicas: folding fewer than Stage 1 wrote silently drops part
     * of the sum, folding more reads past the buffer. */
    int ws_replicas;
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

/* Launch grid: (ceil(wg_N / tile_n), ceil(wg_M / tile_m), groups) */
void rocke_wgrad_reduce_grid(const rocke_wgrad_reduce_spec_t* spec,
                             int* out_gx,
                             int* out_gy,
                             int* out_gz);

/* Build the kernel signature (7 entries: ws_ptr, dw_ptr, wg_M, wg_N,
 * ws_bytes, dw_bytes, groups). arena must not be NULL. */
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
