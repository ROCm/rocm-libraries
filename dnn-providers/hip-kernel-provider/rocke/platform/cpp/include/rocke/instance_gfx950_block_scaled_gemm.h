// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_GFX950_BLOCK_SCALED_GEMM_H
#define ROCKE_INSTANCE_GFX950_BLOCK_SCALED_GEMM_H
#include "rocke/ir.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct rocke_gfx950_block_scaled_gemm_spec
{
    const char* name;
    int M, N, K;
    const char* dtype_a;
    const char* dtype_b;
    const char* dtype_c;
    int block_k;
} rocke_gfx950_block_scaled_gemm_spec_t;
rocke_gfx950_block_scaled_gemm_spec_t rocke_gfx950_block_scaled_gemm_spec_default(void);
bool rocke_gfx950_block_scaled_gemm_is_valid_spec(const rocke_gfx950_block_scaled_gemm_spec_t* spec,
                                                  const char* arch);
rocke_kernel_def_t* rocke_build_gfx950_block_scaled_gemm_new(
    rocke_ir_builder_t* b, const rocke_gfx950_block_scaled_gemm_spec_t* spec, const char* arch);
#ifdef __cplusplus
}
#endif
#endif
