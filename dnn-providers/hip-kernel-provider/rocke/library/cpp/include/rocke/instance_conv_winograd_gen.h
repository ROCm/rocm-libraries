/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd_gen.h -- C++ engine port of the generalized
 * Winograd forward transform kernels in
 * library/kernels/common/conv_winograd_gen.py.
 *
 * Per axis F(m, r) (or, along W, S segments of F(m, m) sharing one input
 * transform) on the default Toom-Cook points; 1-D, r x 1 and 2-D filters,
 * dilation as independent sub-grids, groups, halo tiles of the segmented
 * pipeline. The three stages:
 *
 *   K1 "filter":  w KYXC          -> U [G'][S][Kg][Cg]
 *   K2 "input":   x NHWC          -> V [T + S - 1][bins][C]
 *   K4 "output":  M [T][bins][K]  -> y NHWK
 *
 *   Python (conv_winograd_gen.py)            C (this header)
 *   ------------------------------------     ---------------------------------------
 *   WinoAxis(m, r, seg)                      rocke_wino_axis_t (default points only)
 *   @dataclass(frozen) WinogradGenSpec       rocke_winograd_gen_spec_t
 *     __post_init__                          rocke_winograd_gen_is_valid_spec
 *     .kernel_name(stage)                    rocke_winograd_gen_kernel_name
 *   filter/input/output_transform_grid       rocke_winograd_gen_stage_grid
 *   build_filter/input/output_transform      rocke_build_winograd_gen_stage(_new)
 *   (+ convenience: build -> lower .ll)      rocke_winograd_gen_lower_to_llvm
 *
 * `stage` is "K1", "K2" or "K4". The problem extents are baked into the
 * kernels; the buffer byte sizes are kernargs.
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_GEN_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_GEN_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WinoAxis(m, r, seg=1): F(m, r), or seg > 1 segments of F(m, m). The C++
 * port supports the default interpolation points only (pts=None). */
typedef struct rocke_wino_axis
{
    int m;
    int r;
    int seg; /* default 1 */
} rocke_wino_axis_t;

/* ===================================================================== *
 *  WinogradGenSpec
 *
 *  @dataclass(frozen=True)
 *  class WinogradGenSpec:
 *      problem: ConvProblem
 *      ax_y: WinoAxis
 *      ax_x: WinoAxis
 *      io_dtype: str = "fp16"
 *      operand_dtype: str = "fp16"
 *      vec: int = 4
 *      block_size: int = 256
 *      nt_stores: bool = True
 *      name: str = "conv_winograd_gen"
 * ===================================================================== */
typedef struct rocke_winograd_gen_spec
{
    rocke_conv_problem_t problem;
    rocke_wino_axis_t ax_y;
    rocke_wino_axis_t ax_x;
    const char* io_dtype; /* default "fp16" */
    const char* operand_dtype; /* default "fp16" */
    int vec; /* default 4 */
    int block_size; /* default 256 */
    bool nt_stores; /* default true */
    const char* name; /* default "conv_winograd_gen" */
} rocke_winograd_gen_spec_t;

/* Spec with the dataclass defaults (problem zeroed, both axes F(1, 1)). */
rocke_winograd_gen_spec_t rocke_winograd_gen_spec_default(void);

/* WinogradGenSpec.__post_init__ (and WinoAxis.check): (ok, reason). */
bool rocke_winograd_gen_is_valid_spec(const rocke_winograd_gen_spec_t* spec,
                                      char* reason,
                                      size_t reason_cap);

/* Derived geometry (Python @property). */
int rocke_winograd_gen_bins(const rocke_winograd_gen_spec_t* spec);
int rocke_winograd_gen_tiles_h(const rocke_winograd_gen_spec_t* spec);
int rocke_winograd_gen_tiles_w(const rocke_winograd_gen_spec_t* spec);
int rocke_winograd_gen_tiles_w_in(const rocke_winograd_gen_spec_t* spec);
long long rocke_winograd_gen_num_tiles(const rocke_winograd_gen_spec_t* spec);

/* Launch grid of a stage ("K1" | "K2" | "K4"); ROCKE_ERR_VALUE for an unknown
 * stage or an empty grid. */
rocke_status_t rocke_winograd_gen_stage_grid(const rocke_winograd_gen_spec_t* spec,
                                             const char* stage,
                                             int grid[3]);

/* kernel_name(<filter|input|output>) of a stage. */
rocke_status_t rocke_winograd_gen_kernel_name(const rocke_winograd_gen_spec_t* spec,
                                              const char* stage,
                                              char* out,
                                              size_t out_cap);

/* build_{filter,input,output}_transform(spec) into an initialised builder.
 * The kernels do not depend on `arch` (as in Python). */
rocke_kernel_def_t* rocke_build_winograd_gen_stage(rocke_ir_builder_t* b,
                                                   const rocke_winograd_gen_spec_t* spec,
                                                   const char* stage,
                                                   const char* arch);

/* Initialise `b` with the stage's kernel name, then build. */
rocke_kernel_def_t* rocke_build_winograd_gen_stage_new(rocke_ir_builder_t* b,
                                                       const rocke_winograd_gen_spec_t* spec,
                                                       const char* stage,
                                                       const char* arch);

/* Convenience: build + lower to LLVM IR text (*out_ll malloc'd). */
rocke_status_t rocke_winograd_gen_lower_to_llvm(const rocke_winograd_gen_spec_t* spec,
                                                const char* stage,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_GEN_H */
