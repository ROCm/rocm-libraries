/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_dw_wgrad.h -- C++ engine port of the depthwise
 * wgrad kernels and the partial-slot fold in
 * library/kernels/common/conv_direct_dw_wgrad.py.
 *
 * Every block of a depthwise wgrad reduces over the spatial work it owns and
 * stores one partial slot ws[part][tap][c] with a plain store (same-address
 * fp32 atomics on a depthwise dW serialize); the fold sums the partitions.
 *
 *   Python (conv_direct_dw_wgrad.py)           C (this header)
 *   ------------------------------------   -----------------------------------
 *   DwWgradFoldSpec / build_dw_wgrad_fold   rocke_dw_wgrad_fold_spec_t /
 *                                             rocke_build_dw_wgrad_fold(_new)
 *   DwWgradColsSpec / is_valid_.._cols_spec rocke_dw_wgrad_cols_spec_t /
 *     / build_dw_wgrad_cols                   rocke_dw_wgrad_cols_is_valid_spec /
 *                                             rocke_build_dw_wgrad_cols(_new)
 *   DwWgradLdsSpec / is_valid_.._lds_spec   rocke_dw_wgrad_lds_spec_t /
 *     / build_dw_wgrad_lds                    rocke_dw_wgrad_lds_is_valid_spec /
 *                                             rocke_build_dw_wgrad_lds(_new)
 *   .kernel_name() / launch geometry        rocke_dw_wgrad_*_kernel_name /
 *                                             rocke_dw_wgrad_*_launch_geometry
 *
 * The cols and lds kernels declare the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("wgrad"), with D bound to the fp32 partial-slot
 * workspace. The fold declares A (the fp32 slots), D (dW), A_bytes, D_bytes,
 * parts, channels.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_H
#define ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DwWgradFoldSpec:  dW[c, tap] = sum_part ws[part][tap][c]
 *
 *  @dataclass(frozen=True)
 *  class DwWgradFoldSpec:
 *      taps: int
 *      name: str = "dw_wgrad_fold"
 * ===================================================================== */
typedef struct rocke_dw_wgrad_fold_spec
{
    int taps;
    const char* name; /* default "dw_wgrad_fold" */
} rocke_dw_wgrad_fold_spec_t;

rocke_dw_wgrad_fold_spec_t rocke_dw_wgrad_fold_spec_default(void);
/* kernel_name_join(name, t{taps}). */
rocke_status_t rocke_dw_wgrad_fold_kernel_name(const rocke_dw_wgrad_fold_spec_t* spec,
                                               char* out,
                                               size_t out_cap);
void rocke_dw_wgrad_fold_launch_geometry(const rocke_dw_wgrad_fold_spec_t* spec,
                                         int channels,
                                         int grid[3],
                                         int block[3]);
rocke_kernel_def_t* rocke_build_dw_wgrad_fold(rocke_ir_builder_t* b,
                                              const rocke_dw_wgrad_fold_spec_t* spec,
                                              const char* arch);
rocke_kernel_def_t* rocke_build_dw_wgrad_fold_new(rocke_ir_builder_t* b,
                                                  const rocke_dw_wgrad_fold_spec_t* spec,
                                                  const char* arch);
rocke_status_t rocke_dw_wgrad_fold_lower_to_llvm(const rocke_dw_wgrad_fold_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

/* ===================================================================== *
 *  DwWgradColsSpec: small-C depthwise wgrad, lanes along output columns
 *
 *  @dataclass(frozen=True)
 *  class DwWgradColsSpec:
 *      problem: DirectConvProblem
 *      name: str = "dw_wgrad_cols"
 *      strip: int = 8
 *      threads: int = 64
 *      target_blocks: int = 2048
 * ===================================================================== */
typedef struct rocke_dw_wgrad_cols_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "dw_wgrad_cols" */
    int strip; /* default 8 */
    int threads; /* default 64 */
    int target_blocks; /* default 2048 */
} rocke_dw_wgrad_cols_spec_t;

rocke_dw_wgrad_cols_spec_t rocke_dw_wgrad_cols_spec_default(void);
int rocke_dw_wgrad_cols_n_partitions(const rocke_dw_wgrad_cols_spec_t* spec);
long rocke_dw_wgrad_cols_workspace_bytes(const rocke_dw_wgrad_cols_spec_t* spec);
void rocke_dw_wgrad_cols_launch_geometry(const rocke_dw_wgrad_cols_spec_t* spec,
                                         int grid[3],
                                         int block[3]);
/* DwWgradColsSpec.lds_bytes(): the block reduction buffer. */
long rocke_dw_wgrad_cols_lds_bytes(const rocke_dw_wgrad_cols_spec_t* spec);
/* kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}c{groups}, st{strip},
 * t{threads}, tb{n}, flags [bf16]). */
rocke_status_t rocke_dw_wgrad_cols_kernel_name(const rocke_dw_wgrad_cols_spec_t* spec,
                                               char* out,
                                               size_t out_cap);
bool rocke_dw_wgrad_cols_is_valid_spec(const rocke_dw_wgrad_cols_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);
rocke_kernel_def_t* rocke_build_dw_wgrad_cols(rocke_ir_builder_t* b,
                                              const rocke_dw_wgrad_cols_spec_t* spec,
                                              const char* arch);
rocke_kernel_def_t* rocke_build_dw_wgrad_cols_new(rocke_ir_builder_t* b,
                                                  const rocke_dw_wgrad_cols_spec_t* spec,
                                                  const char* arch);
rocke_status_t rocke_dw_wgrad_cols_lower_to_llvm(const rocke_dw_wgrad_cols_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

/* ===================================================================== *
 *  DwWgradLdsSpec: depthwise wgrad, LDS row rings (C % 8 == 0)
 *
 *  @dataclass(frozen=True)
 *  class DwWgradLdsSpec:
 *      problem: DirectConvProblem
 *      name: str = "dw_wgrad_lds"
 *      lanes_c: int = 8
 *      lanes_w: int = 32
 *      cols_per_lane: int = 1
 *      prefetch: int = 2
 *      rows_per_seg: int = 32
 *      target_blocks: int = 1024
 * ===================================================================== */
typedef struct rocke_dw_wgrad_lds_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "dw_wgrad_lds" */
    int lanes_c; /* default 8 */
    int lanes_w; /* default 32 */
    int cols_per_lane; /* default 1 */
    int prefetch; /* default 2 */
    int rows_per_seg; /* default 32 */
    int target_blocks; /* default 1024 */
} rocke_dw_wgrad_lds_spec_t;

rocke_dw_wgrad_lds_spec_t rocke_dw_wgrad_lds_spec_default(void);
int rocke_dw_wgrad_lds_threads(const rocke_dw_wgrad_lds_spec_t* spec);
/* DwWgradLdsSpec.lds_bytes(): the padded X / dY rings plus the reduction. */
long rocke_dw_wgrad_lds_lds_bytes(const rocke_dw_wgrad_lds_spec_t* spec);
int rocke_dw_wgrad_lds_n_partitions(const rocke_dw_wgrad_lds_spec_t* spec);
long rocke_dw_wgrad_lds_workspace_bytes(const rocke_dw_wgrad_lds_spec_t* spec);
void rocke_dw_wgrad_lds_launch_geometry(const rocke_dw_wgrad_lds_spec_t* spec,
                                        int grid[3],
                                        int block[3]);
/* kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}, lc{n}, lw{n}, cpl{n},
 * pf{n}, rps{n}, tb{n}, flags [bf16]). */
rocke_status_t rocke_dw_wgrad_lds_kernel_name(const rocke_dw_wgrad_lds_spec_t* spec,
                                              char* out,
                                              size_t out_cap);
bool rocke_dw_wgrad_lds_is_valid_spec(const rocke_dw_wgrad_lds_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);
rocke_kernel_def_t* rocke_build_dw_wgrad_lds(rocke_ir_builder_t* b,
                                             const rocke_dw_wgrad_lds_spec_t* spec,
                                             const char* arch);
rocke_kernel_def_t* rocke_build_dw_wgrad_lds_new(rocke_ir_builder_t* b,
                                                 const rocke_dw_wgrad_lds_spec_t* spec,
                                                 const char* arch);
rocke_status_t rocke_dw_wgrad_lds_lower_to_llvm(const rocke_dw_wgrad_lds_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_H */
