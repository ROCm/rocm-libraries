/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd_fwd_rt.h -- C++ engine port of the
 * shape-generic Winograd F(mh x mw, 3 x 3) forward transform kernels in
 * library/kernels/common/conv_winograd_fwd_rt.py.
 *
 *   Python (conv_winograd_fwd_rt.py)        C (this header)
 *   ------------------------------------    -------------------------------------
 *   RtFilterSpec / RtPrescaleSpec /         rocke_winograd_rt_spec_t (+ .stage)
 *     RtInputSpec / RtOutputSpec
 *     __post_init__                         rocke_winograd_rt_is_valid_spec
 *     .kernel_name()                        rocke_winograd_rt_kernel_name
 *   build_stage(spec)                       rocke_build_winograd_rt(_new)
 *   magic_div(d)                            rocke_winograd_rt_magic_div
 *
 * Every problem extent is a kernel argument (divisions by runtime extents use
 * a host-computed magic number), so one binary serves every problem its knobs
 * fit. The launch geometry and kernarg values are host-side Python
 * (stage_launch); the AOT cache stores the binaries built here.
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_FWD_RT_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_FWD_RT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The stage a spec builds (the Python spec class). */
typedef enum rocke_winograd_rt_stage
{
    ROCKE_WINOGRAD_RT_K1 = 0, /* RtFilterSpec: filter transform */
    ROCKE_WINOGRAD_RT_K2A = 1, /* RtPrescaleSpec: per-pixel max |x| */
    ROCKE_WINOGRAD_RT_K2 = 2, /* RtInputSpec: input transform */
    ROCKE_WINOGRAD_RT_K4 = 3 /* RtOutputSpec: output transform */
} rocke_winograd_rt_stage_t;

/* The union of the four spec classes' fields (defaults in the comments); a
 * stage reads only the fields its Python class has. */
typedef struct rocke_winograd_rt_spec
{
    rocke_winograd_rt_stage_t stage;
    int mh; /* 4 (K1, K2, K4) */
    int mw; /* 4 (K1, K2, K4) */
    const char* io_dtype; /* "fp16" */
    const char* operand_dtype; /* "fp16" (K1, K2, K4) */
    int vec; /* 8 */
    int filter_rows; /* 1 (K1) */
    int block_size; /* 256 (K2, K4) */
    bool prescale; /* false (K2, K4) */
    int tiles_per_thread; /* 1 (K2) */
    int lds[2]; /* (0, 0) (K2) */
    int lds_lanes; /* 16 (K2) */
    bool nt_stores; /* true (K2, K4) */
    bool fused; /* false (K2) */
} rocke_winograd_rt_spec_t;

/* Spec with the dataclass defaults for `stage`. */
rocke_winograd_rt_spec_t rocke_winograd_rt_spec_default(rocke_winograd_rt_stage_t stage);

/* __post_init__ of the stage's spec class: (ok, reason). */
bool rocke_winograd_rt_is_valid_spec(const rocke_winograd_rt_spec_t* spec,
                                     char* reason,
                                     size_t reason_cap);

/* spec.kernel_name(): the AOT identity of the binary. */
rocke_status_t
    rocke_winograd_rt_kernel_name(const rocke_winograd_rt_spec_t* spec, char* out, size_t out_cap);

/* magic_div(d): (magic, shift) with n / d == umulhi(n, magic) >> shift for
 * 0 <= n < 2**31; d == 1 gives (0, 0). False for d < 1. */
bool rocke_winograd_rt_magic_div(int64_t d, uint32_t* magic, int* shift);

/* build_stage(spec) into an initialised builder / a fresh one named like
 * Python. NULL (with the builder's error set) for an invalid spec. */
rocke_kernel_def_t* rocke_build_winograd_rt(rocke_ir_builder_t* b,
                                            const rocke_winograd_rt_spec_t* spec,
                                            const char* arch);
rocke_kernel_def_t* rocke_build_winograd_rt_new(rocke_ir_builder_t* b,
                                                const rocke_winograd_rt_spec_t* spec,
                                                const char* arch);

/* Build + lower to LLVM IR. *out_ll is malloc'd (caller frees). */
rocke_status_t rocke_winograd_rt_lower_to_llvm(const rocke_winograd_rt_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_FWD_RT_H */
