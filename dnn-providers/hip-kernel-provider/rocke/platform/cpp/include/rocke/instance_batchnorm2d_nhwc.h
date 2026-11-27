/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batchnorm2d_nhwc.h -- C99 port of the dynamic NHWC
 * (channels-last) BatchNorm2D kernels in rocke/instances/common/batchnorm2d.py
 * (section "NHWC (channels-last) kernels: infer, training forward, backward").
 *
 * Memory is R = N*H*W rows of C contiguous channels. A CTA owns a channel
 * tile of tx * vec channels and a chunk of rows; grid = (row splits, channel
 * tiles, 1). Kinds: infer ("infer"); train / bwd ("fused" single launch,
 * "reduce" + "apply" split pair).
 *
 *   Python (batchnorm2d.py)                    C99 (this header)
 *   ---------------------------------------    -------------------------------------------
 *   class BatchNorm2DNhwcSpec                  rocke_batchnorm2d_nhwc_spec_t
 *   BatchNorm2DNhwcSpec.kernel_name(kind)      rocke_batchnorm2d_nhwc_kernel_name(...)
 *   is_valid_nhwc_spec(spec, arch)             rocke_batchnorm2d_is_valid_nhwc_spec(...)
 *   nhwc_admits(spec, n, c, h, w)              rocke_batchnorm2d_nhwc_admits(...)
 *   batchnorm2d_nhwc_grid(c, plan)             rocke_batchnorm2d_nhwc_grid(...)
 *   batchnorm2d_nhwc_workspace_floats(c, plan) rocke_batchnorm2d_nhwc_workspace_floats(...)
 *   build_batchnorm2d_nhwc_infer(spec)         rocke_build_batchnorm2d_nhwc_infer(b, spec)
 *   build_batchnorm2d_nhwc_train(spec, kind)   rocke_build_batchnorm2d_nhwc_train(b, spec, kind)
 *   build_batchnorm2d_nhwc_bwd(spec, kind)     rocke_build_batchnorm2d_nhwc_bwd(b, spec, kind)
 *   build_batchnorm2d_nhwc(spec, kind)         rocke_build_batchnorm2d_nhwc(b, spec, kind)
 *
 * batchnorm2d_nhwc_plans (rows-per-CTA candidates of one shape) is host-side
 * launch planning with no IR effect; the grid / workspace helpers take the
 * plan's split and tile counts directly.
 */
#ifndef ROCKE_INSTANCE_BATCHNORM2D_NHWC_H
#define ROCKE_INSTANCE_BATCHNORM2D_NHWC_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python BatchNorm2DNhwcSpec (N / C / H / W are runtime args). */
typedef struct rocke_batchnorm2d_nhwc_spec
{
    const char* mode; /* default "train" ("infer"|"train"|"bwd") */
    int block_size; /* default 256 */
    int tx; /* default 8 */
    int vec; /* default 8 */
    int unroll; /* default 4 */
    const char* dtype; /* default "f16" ("f16"|"bf16") */
    int wave_size; /* default 64 */
    bool dynamic; /* default true */
    const char* name; /* default "rocke_batchnorm2d_nhwc" */
} rocke_batchnorm2d_nhwc_spec_t;

rocke_batchnorm2d_nhwc_spec_t rocke_batchnorm2d_nhwc_spec_default(void);

/* BatchNorm2DNhwcSpec.ty / .tile_channels properties. */
int rocke_batchnorm2d_nhwc_ty(const rocke_batchnorm2d_nhwc_spec_t* spec);
int rocke_batchnorm2d_nhwc_tile_channels(const rocke_batchnorm2d_nhwc_spec_t* spec);

/* BatchNorm2DNhwcSpec.kernel_name(kind). */
rocke_status_t rocke_batchnorm2d_nhwc_kernel_name(const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                  const char* kind,
                                                  char* out,
                                                  size_t out_cap);

/* is_valid_nhwc_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". */
bool rocke_batchnorm2d_is_valid_nhwc_spec(const rocke_batchnorm2d_nhwc_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* nhwc_admits(spec, n, c, h, w) -> (ok, reason): runtime-shape admission. */
bool rocke_batchnorm2d_nhwc_admits(const rocke_batchnorm2d_nhwc_spec_t* spec,
                                   int n,
                                   int c,
                                   int h,
                                   int w,
                                   char* reason,
                                   size_t reason_cap);

/* batchnorm2d_nhwc_grid(c, plan) -> (plan.splits, plan.tiles, 1). */
rocke_status_t rocke_batchnorm2d_nhwc_grid(int splits, int tiles, int out[3]);

/* batchnorm2d_nhwc_workspace_floats(c, plan) -> 2 * c * plan.splits. */
long rocke_batchnorm2d_nhwc_workspace_floats(int c, int splits);

/* Builders: build into `b` (already rocke_ir_builder_init'd with
 * spec.kernel_name(kind)); return b->kernel or NULL with b's sticky error set.
 * `kind` is "infer" for mode "infer", else "fused" | "reduce" | "apply". */
rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_infer(rocke_ir_builder_t* b,
                                                       const rocke_batchnorm2d_nhwc_spec_t* spec);
rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_train(rocke_ir_builder_t* b,
                                                       const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                       const char* kind);
rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_bwd(rocke_ir_builder_t* b,
                                                     const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                     const char* kind);

/* build_batchnorm2d_nhwc(spec, kind): dispatch on spec.mode. */
rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc(rocke_ir_builder_t* b,
                                                 const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                 const char* kind);

/* Convenience: init `b` with spec.kernel_name(kind) (kind "infer" for mode
 * "infer"), then build. The caller frees `b` with rocke_ir_builder_free(). */
rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_new(rocke_ir_builder_t* b,
                                                     const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                     const char* kind);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_BATCHNORM2D_NHWC_H */
