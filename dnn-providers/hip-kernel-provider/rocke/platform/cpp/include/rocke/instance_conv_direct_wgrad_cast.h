/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_wgrad_cast.h -- C++ engine port of
 * library/kernels/common/conv_direct_wgrad_cast.py: a flat elementwise
 * narrowing cast, D[i] = cast_to_dtype_d(A[i]) for i in [0, count).
 *
 * direct_wgrad and direct_wgrad_tiled atomic-add their fp32 partial sums
 * straight into their "D" kernarg with an offset formula that already
 * matches the real dW tensor's final KYXC layout exactly -- no
 * permutation. So once "D" is pointed at an fp32 scratch buffer the same
 * shape as dW instead of the real (fp16/bf16) dW tensor, the only remaining
 * step is a narrowing cast, with no index permutation at all: that is what
 * this kernel does. Unlike rocke_dw_wgrad_fold_spec_t it performs no
 * reduction and no transpose.
 *
 *   Python (conv_direct_wgrad_cast.py)              C (this header)
 *   -------------------------------------------   -----------------------------------
 *   DirectWgradCastSpec / build_direct_wgrad_cast  rocke_direct_wgrad_cast_spec_t /
 *                                                    rocke_build_direct_wgrad_cast(_new)
 *
 * Declares A (the fp32 scratch, final dW order), D (dW, dtype_d), A_bytes,
 * D_bytes, count.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_WGRAD_CAST_H
#define ROCKE_INSTANCE_CONV_DIRECT_WGRAD_CAST_H

#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectWgradCastSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectWgradCastSpec:
 *      dtype_d: str
 *      name: str = "direct_wgrad_cast"
 *      threads: int = 256
 * ===================================================================== */
typedef struct rocke_direct_wgrad_cast_spec
{
    const char* dtype_d; /* "fp16" or "bf16"; required, no default */
    const char* name; /* default "direct_wgrad_cast" */
    int threads; /* default 256 */
} rocke_direct_wgrad_cast_spec_t;

rocke_direct_wgrad_cast_spec_t rocke_direct_wgrad_cast_spec_default(void);
bool rocke_direct_wgrad_cast_is_valid_spec(const rocke_direct_wgrad_cast_spec_t* spec,
                                           char* reason,
                                           size_t reason_cap);
/* kernel_name_join(name, dtype_d, t{threads}). */
rocke_status_t rocke_direct_wgrad_cast_kernel_name(const rocke_direct_wgrad_cast_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);
void rocke_direct_wgrad_cast_launch_geometry(const rocke_direct_wgrad_cast_spec_t* spec,
                                             int count,
                                             int grid[3],
                                             int block[3]);
rocke_kernel_def_t* rocke_build_direct_wgrad_cast(rocke_ir_builder_t* b,
                                                  const rocke_direct_wgrad_cast_spec_t* spec,
                                                  const char* arch);
rocke_kernel_def_t* rocke_build_direct_wgrad_cast_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_wgrad_cast_spec_t* spec,
                                                      const char* arch);
rocke_status_t rocke_direct_wgrad_cast_lower_to_llvm(const rocke_direct_wgrad_cast_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_WGRAD_CAST_H */
