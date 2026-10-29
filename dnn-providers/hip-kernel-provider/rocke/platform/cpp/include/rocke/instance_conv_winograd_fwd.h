/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd_fwd.h -- C++ engine port of the Winograd
 * F(mh x mw, 3 x 3) forward transform kernels in
 * library/kernels/common/conv_winograd_fwd.py.
 *
 *   Python (conv_winograd_fwd.py)           C (this header)
 *   ------------------------------------    -------------------------------------
 *   @dataclass(frozen) WinogradFwdSpec      rocke_winograd_fwd_spec_t
 *     __post_init__                         rocke_winograd_fwd_is_valid_spec
 *     .kernel_name(stage)                   rocke_winograd_fwd_kernel_name
 *   transform_stages(spec)                  rocke_winograd_fwd_stage_present /
 *                                             rocke_winograd_fwd_stage_grid / _block
 *   build_winograd_filter_transform (K1)    rocke_build_winograd_fwd_stage(K1)
 *   build_winograd_prescale (K2a)           rocke_build_winograd_fwd_stage(K2A)
 *   build_winograd_input_transform (K2)     rocke_build_winograd_fwd_stage(K2)
 *   build_winograd_output_transform (K4)    rocke_build_winograd_fwd_stage(K4)
 *   build_winograd_merged(WinogradMergedSpec)
 *                                           rocke_build_winograd_merged
 *
 * The problem extents are baked into the kernels. Every buffer is addressed
 * with 32-bit offsets through buffer resources (out-of-range offsets read 0 /
 * drop the store).
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_FWD_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_FWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WinogradFwdSpec (field defaults in the comments). */
typedef struct rocke_winograd_fwd_spec
{
    rocke_conv_problem_t problem;
    int m; /* 4: output tile rows (2, 4 or 6) */
    const char* io_dtype; /* "fp16" */
    const char* operand_dtype; /* "fp16" */
    int vec; /* 8 */
    int block_size; /* 256 */
    bool prescale; /* false */
    int tiles_per_thread; /* 1: K2 tiles per thread along W; long register-path
                           * runs of tiles >= 4 spill (see WinogradFwdSpec) */
    const char* prescale_mode; /* "fused" */
    int input_block_tiles[2]; /* (0, 0) */
    int wg_tile_rows; /* 1 */
    bool nt_stores; /* true */
    int lds_lanes; /* 32 */
    int filter_rows; /* 1 */
    int filter_vec; /* 0 */
    int m_w; /* 0: output tile columns (0 = m) */
    const char* name; /* "conv_winograd_fwd" */
} rocke_winograd_fwd_spec_t;

/* The transform stages (TransformStage.name). */
typedef enum rocke_winograd_fwd_stage
{
    ROCKE_WINOGRAD_FWD_K1 = 0, /* "K1": filter transform */
    ROCKE_WINOGRAD_FWD_K2A = 1, /* "K2a": per-pixel max (split prescale) */
    ROCKE_WINOGRAD_FWD_K2 = 2, /* "K2": input transform */
    ROCKE_WINOGRAD_FWD_K4 = 3 /* "K4": output transform */
} rocke_winograd_fwd_stage_t;

/* Spec with the dataclass defaults; `problem` must be filled by the caller. */
rocke_winograd_fwd_spec_t rocke_winograd_fwd_spec_default(void);

/* "K1" / "K2a" / "K2" / "K4" -> stage; false for an unknown name. */
bool rocke_winograd_fwd_stage_from_name(const char* name, rocke_winograd_fwd_stage_t* out);
const char* rocke_winograd_fwd_stage_name(rocke_winograd_fwd_stage_t stage);

/* WinogradFwdSpec.__post_init__: (ok, reason). */
bool rocke_winograd_fwd_is_valid_spec(const rocke_winograd_fwd_spec_t* spec,
                                      char* reason,
                                      size_t reason_cap);

/* Whether `stage` is part of the pipeline (K2a only with the split prescale). */
bool rocke_winograd_fwd_stage_present(const rocke_winograd_fwd_spec_t* spec,
                                      rocke_winograd_fwd_stage_t stage);
/* Launch grid / block of a stage (false + reason when the stage cannot launch). */
bool rocke_winograd_fwd_stage_launch(const rocke_winograd_fwd_spec_t* spec,
                                     rocke_winograd_fwd_stage_t stage,
                                     int grid[3],
                                     int* block,
                                     char* reason,
                                     size_t reason_cap);

/* spec.kernel_name(tag): kernel_name_join(name, tag, short()). */
rocke_status_t rocke_winograd_fwd_kernel_name(const rocke_winograd_fwd_spec_t* spec,
                                              const char* tag,
                                              char* out,
                                              size_t out_cap);

/* One stage into an initialised builder / a fresh one named like Python. */
rocke_kernel_def_t* rocke_build_winograd_fwd_stage(rocke_ir_builder_t* b,
                                                   const rocke_winograd_fwd_spec_t* spec,
                                                   rocke_winograd_fwd_stage_t stage,
                                                   const char* arch);
rocke_kernel_def_t* rocke_build_winograd_fwd_stage_new(rocke_ir_builder_t* b,
                                                       const rocke_winograd_fwd_spec_t* spec,
                                                       rocke_winograd_fwd_stage_t stage,
                                                       const char* arch);

/* build_winograd_merged(WinogradMergedSpec(spec, names)): independent stages
 * as one launch on disjoint ranges of a flat grid. */
rocke_kernel_def_t* rocke_build_winograd_merged(rocke_ir_builder_t* b,
                                                const rocke_winograd_fwd_spec_t* spec,
                                                const rocke_winograd_fwd_stage_t* names,
                                                int n_names,
                                                const char* arch);
rocke_kernel_def_t* rocke_build_winograd_merged_new(rocke_ir_builder_t* b,
                                                    const rocke_winograd_fwd_spec_t* spec,
                                                    const rocke_winograd_fwd_stage_t* names,
                                                    int n_names,
                                                    const char* arch);

/* Build (one stage when n_names == 0, else the merged kernel) + lower to LLVM IR.
 * *out_ll is malloc'd (caller frees). */
rocke_status_t rocke_winograd_fwd_lower_to_llvm(const rocke_winograd_fwd_spec_t* spec,
                                                rocke_winograd_fwd_stage_t stage,
                                                const rocke_winograd_fwd_stage_t* names,
                                                int n_names,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_FWD_H */
