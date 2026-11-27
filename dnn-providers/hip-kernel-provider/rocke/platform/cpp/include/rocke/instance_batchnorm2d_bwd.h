/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batchnorm2d_bwd.h -- C99 port of the dynamic NCHW BatchNorm2D
 * backward kernels in rocke/instances/common/batchnorm2d.py (section
 * "Backward (spatial mode, saved mean / inverse std)").
 *
 * Per channel, with xhat = (x - mean) * invstd and M = N*H*W:
 *
 *     dbias[c]  = sum(dy)
 *     dscale[c] = sum(dy * xhat)
 *     dx        = a * dy - k2 * (x - mean) - k3
 *       with a = scale * invstd, k2 = a * invstd * dscale / M, k3 = a * dbias / M
 *
 *   Python (batchnorm2d.py)                    C99 (this header)
 *   ---------------------------------------    -------------------------------------------
 *   class BatchNorm2DBwdSpec                   rocke_batchnorm2d_bwd_spec_t
 *   BatchNorm2DBwdSpec.kernel_name()           rocke_batchnorm2d_bwd_kernel_name(...)
 *   is_valid_bwd_spec(spec, arch)              rocke_batchnorm2d_is_valid_bwd_spec(...)
 *   batchnorm2d_bwd_grid(c, spec)              rocke_batchnorm2d_bwd_grid(...)
 *   build_batchnorm2d_bwd_dynamic(spec)        rocke_build_batchnorm2d_bwd_dynamic(b, spec)
 *   class BatchNorm2DBwdSplitSpec              rocke_batchnorm2d_bwd_split_spec_t
 *   BatchNorm2DBwdSplitSpec.kernel_name(kind)  rocke_batchnorm2d_bwd_split_kernel_name(...)
 *   is_valid_split_dyn_spec(spec, arch)        rocke_batchnorm2d_is_valid_bwd_split_spec(...)
 *   batchnorm2d_bwd_split_workspace_floats     rocke_batchnorm2d_bwd_split_workspace_floats(...)
 *   build_batchnorm2d_bwd_split_reduce(spec)   rocke_build_batchnorm2d_bwd_split_reduce(b, spec)
 *   build_batchnorm2d_bwd_split_apply(spec)    rocke_build_batchnorm2d_bwd_split_apply(b, spec)
 *
 * The split launch geometry (BatchNorm2DSplitPlan, batchnorm2d_split_plan,
 * batchnorm2d_split_dyn_grid) is shared with the forward dynamic split pair
 * and is host-side planning with no IR effect; the workspace size here takes
 * the plan's split count directly.
 */
#ifndef ROCKE_INSTANCE_BATCHNORM2D_BWD_H
#define ROCKE_INSTANCE_BATCHNORM2D_BWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------ single-launch backward (B1)
 *
 * Mirror of Python BatchNorm2DBwdSpec (n/c/h/w are runtime kernel args). */
typedef struct rocke_batchnorm2d_bwd_spec
{
    int block_size; /* default 256 */
    int vec; /* default 4 */
    const char* dtype; /* default "f16" ("f16"|"bf16"|"f32") */
    int channels_per_block; /* default 1 */
    int n_groups; /* default 1 */
    bool fast_reduce; /* default false */
    int wave_size; /* default 64 */
    bool dynamic; /* default true */
    const char* name; /* default "rocke_batchnorm2d_bwd" */
} rocke_batchnorm2d_bwd_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_batchnorm2d_bwd_spec_t rocke_batchnorm2d_bwd_spec_default(void);

/* BatchNorm2DBwdSpec.kernel_name() -> NUL-terminated into out (capacity
 * out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_batchnorm2d_bwd_kernel_name(const rocke_batchnorm2d_bwd_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_bwd_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". */
bool rocke_batchnorm2d_is_valid_bwd_spec(const rocke_batchnorm2d_bwd_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* batchnorm2d_bwd_grid(c, spec) -> (c / channels_per_block, 1, 1). */
rocke_status_t
    rocke_batchnorm2d_bwd_grid(int c, const rocke_batchnorm2d_bwd_spec_t* spec, int out[3]);

/* build_batchnorm2d_bwd_dynamic(spec). Builds into `b` (already
 * rocke_ir_builder_init'd with the kernel name); returns b->kernel or NULL
 * with b's sticky error set. */
rocke_kernel_def_t* rocke_build_batchnorm2d_bwd_dynamic(rocke_ir_builder_t* b,
                                                        const rocke_batchnorm2d_bwd_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller frees
 * `b` with rocke_ir_builder_free(). */
rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_dynamic_new(rocke_ir_builder_t* b,
                                            const rocke_batchnorm2d_bwd_spec_t* spec);

/* ------------------------------------------------- split backward pair (B2)
 *
 * Mirror of Python BatchNorm2DBwdSplitSpec (a BatchNorm2DSplitDynSpec with
 * its own kernel-name prefix). save_mean_invstd / dynamic are carried for
 * field parity; neither changes the backward IR. */
typedef struct rocke_batchnorm2d_bwd_split_spec
{
    int block_size; /* default 256 */
    int vec; /* default 8 */
    const char* dtype; /* default "f16" ("f16"|"bf16"|"f32") */
    bool fast_reduce; /* default false */
    bool save_mean_invstd; /* default true */
    int wave_size; /* default 64 */
    bool dynamic; /* default true */
    const char* name; /* default "rocke_batchnorm2d_bwd_splitdyn" */
} rocke_batchnorm2d_bwd_split_spec_t;

rocke_batchnorm2d_bwd_split_spec_t rocke_batchnorm2d_bwd_split_spec_default(void);

/* BatchNorm2DBwdSplitSpec.kernel_name(kind); kind is "reduce" or "apply". */
rocke_status_t rocke_batchnorm2d_bwd_split_kernel_name(
    const rocke_batchnorm2d_bwd_split_spec_t* spec, const char* kind, char* out, size_t out_cap);

/* is_valid_split_dyn_spec(spec, arch) as the backward builders apply it. */
bool rocke_batchnorm2d_is_valid_bwd_split_spec(const rocke_batchnorm2d_bwd_split_spec_t* spec,
                                               const char* arch,
                                               char* reason,
                                               size_t reason_cap);

/* batchnorm2d_bwd_split_workspace_floats(c, plan) with splits = plan.splits:
 * c * splits * 2 f32 elements. */
int rocke_batchnorm2d_bwd_split_workspace_floats(int c, int splits);

/* build_batchnorm2d_bwd_split_reduce(spec) / build_batchnorm2d_bwd_split_apply(spec).
 * Same builder contract as rocke_build_batchnorm2d_bwd_dynamic. */
rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_reduce(rocke_ir_builder_t* b,
                                             const rocke_batchnorm2d_bwd_split_spec_t* spec);
rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_apply(rocke_ir_builder_t* b,
                                            const rocke_batchnorm2d_bwd_split_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name(kind), then build. */
rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_reduce_new(rocke_ir_builder_t* b,
                                                 const rocke_batchnorm2d_bwd_split_spec_t* spec);
rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_apply_new(rocke_ir_builder_t* b,
                                                const rocke_batchnorm2d_bwd_split_spec_t* spec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_BATCHNORM2D_BWD_H */
