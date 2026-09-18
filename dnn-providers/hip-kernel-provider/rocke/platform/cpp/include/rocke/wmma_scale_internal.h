// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Shared gfx1250 operand contracts. Mirrors core/arch/wmma_scale.py:
 * E8M0ScalePacking -> rocke_e8m0_scale_packing_t
 * ScaledWmmaOp     -> rocke_scaled_wmma_op_t
 * gfx1250_scaled_wmma -> rocke_gfx1250_scaled_wmma
 */
#ifndef ROCKE_WMMA_SCALE_INTERNAL_H
#define ROCKE_WMMA_SCALE_INTERNAL_H

#include <string.h>

#include "rocke/ir.h"

typedef struct rocke_e8m0_scale_packing
{
    /* Consecutive K groups, first group in the low byte, independently for A/B. */
    int count;
    int block_k;
} rocke_e8m0_scale_packing_t;

typedef struct rocke_scaled_wmma_op
{
    const char* op_id;
    int matrix_format;
    int matrix_format_b;
    rocke_e8m0_scale_packing_t scales;
} rocke_scaled_wmma_op_t;

static inline const rocke_scaled_wmma_op_t* rocke_gfx1250_scaled_wmma(const char* op_id)
{
    static const rocke_e8m0_scale_packing_t scale = {/*count=*/4, /*block_k=*/32};
    static const rocke_e8m0_scale_packing_t scale16 = {/*count=*/8, /*block_k=*/16};
    static const rocke_scaled_wmma_op_t ops[] = {
        {"wmma_scale_f32_16x16x128_fp8_fp8", 0, 0, scale},
        {"wmma_scale_f32_16x16x128_fp8_bf8", 0, 1, scale},
        {"wmma_scale_f32_16x16x128_fp8_fp6", 0, 2, scale},
        {"wmma_scale_f32_16x16x128_fp8_bf6", 0, 3, scale},
        {"wmma_scale_f32_16x16x128_fp8_fp4", 0, 4, scale},
        {"wmma_scale_f32_16x16x128_bf8_fp8", 1, 0, scale},
        {"wmma_scale_f32_16x16x128_bf8_bf8", 1, 1, scale},
        {"wmma_scale_f32_16x16x128_bf8_fp6", 1, 2, scale},
        {"wmma_scale_f32_16x16x128_bf8_bf6", 1, 3, scale},
        {"wmma_scale_f32_16x16x128_bf8_fp4", 1, 4, scale},
        {"wmma_scale_f32_16x16x128_fp6_fp8", 2, 0, scale},
        {"wmma_scale_f32_16x16x128_fp6_bf8", 2, 1, scale},
        {"wmma_scale_f32_16x16x128_fp6_fp6", 2, 2, scale},
        {"wmma_scale_f32_16x16x128_fp6_bf6", 2, 3, scale},
        {"wmma_scale_f32_16x16x128_fp6_fp4", 2, 4, scale},
        {"wmma_scale_f32_16x16x128_bf6_fp8", 3, 0, scale},
        {"wmma_scale_f32_16x16x128_bf6_bf8", 3, 1, scale},
        {"wmma_scale_f32_16x16x128_bf6_fp6", 3, 2, scale},
        {"wmma_scale_f32_16x16x128_bf6_bf6", 3, 3, scale},
        {"wmma_scale_f32_16x16x128_bf6_fp4", 3, 4, scale},
        {"wmma_scale_f32_16x16x128_fp4_fp8", 4, 0, scale},
        {"wmma_scale_f32_16x16x128_fp4_bf8", 4, 1, scale},
        {"wmma_scale_f32_16x16x128_fp4_fp6", 4, 2, scale},
        {"wmma_scale_f32_16x16x128_fp4_bf6", 4, 3, scale},
        {"wmma_scale_f32_16x16x128_fp4_fp4", 4, 4, scale},
        {"wmma_scale16_f32_16x16x128_fp8_fp8", 0, 0, scale16},
        {"wmma_scale16_f32_16x16x128_fp8_bf8", 0, 1, scale16},
        {"wmma_scale16_f32_16x16x128_fp8_fp6", 0, 2, scale16},
        {"wmma_scale16_f32_16x16x128_fp8_bf6", 0, 3, scale16},
        {"wmma_scale16_f32_16x16x128_fp8_fp4", 0, 4, scale16},
        {"wmma_scale16_f32_16x16x128_bf8_fp8", 1, 0, scale16},
        {"wmma_scale16_f32_16x16x128_bf8_bf8", 1, 1, scale16},
        {"wmma_scale16_f32_16x16x128_bf8_fp6", 1, 2, scale16},
        {"wmma_scale16_f32_16x16x128_bf8_bf6", 1, 3, scale16},
        {"wmma_scale16_f32_16x16x128_bf8_fp4", 1, 4, scale16},
        {"wmma_scale16_f32_16x16x128_fp6_fp8", 2, 0, scale16},
        {"wmma_scale16_f32_16x16x128_fp6_bf8", 2, 1, scale16},
        {"wmma_scale16_f32_16x16x128_fp6_fp6", 2, 2, scale16},
        {"wmma_scale16_f32_16x16x128_fp6_bf6", 2, 3, scale16},
        {"wmma_scale16_f32_16x16x128_fp6_fp4", 2, 4, scale16},
        {"wmma_scale16_f32_16x16x128_bf6_fp8", 3, 0, scale16},
        {"wmma_scale16_f32_16x16x128_bf6_bf8", 3, 1, scale16},
        {"wmma_scale16_f32_16x16x128_bf6_fp6", 3, 2, scale16},
        {"wmma_scale16_f32_16x16x128_bf6_bf6", 3, 3, scale16},
        {"wmma_scale16_f32_16x16x128_bf6_fp4", 3, 4, scale16},
        {"wmma_scale16_f32_16x16x128_fp4_fp8", 4, 0, scale16},
        {"wmma_scale16_f32_16x16x128_fp4_bf8", 4, 1, scale16},
        {"wmma_scale16_f32_16x16x128_fp4_fp6", 4, 2, scale16},
        {"wmma_scale16_f32_16x16x128_fp4_bf6", 4, 3, scale16},
        {"wmma_scale16_f32_16x16x128_fp4_fp4", 4, 4, scale16},
    };
    if(!op_id)
    {
        return NULL;
    }
    if(strncmp(op_id, "tile.", 5) == 0)
    {
        op_id += 5;
    }
    for(size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i)
    {
        if(strcmp(op_id, ops[i].op_id) == 0)
        {
            return &ops[i];
        }
    }
    return NULL;
}

static inline const rocke_scaled_wmma_op_t* rocke_gfx1250_scaled_wmma_from_op(const rocke_op_t* op)
{
    const char* op_id = rocke_attr_get_str(&op->attrs, "op_id");
    return rocke_gfx1250_scaled_wmma(op_id ? op_id : op->name);
}

static inline int rocke_e8m0_scale_word_bits(const rocke_e8m0_scale_packing_t* packing)
{
    return packing->count * 8;
}

static inline bool rocke_wmma_scaled_formats(const char* op_id, bool* scale16, int* a, int* b)
{
    const rocke_scaled_wmma_op_t* spec = rocke_gfx1250_scaled_wmma(op_id);
    if(!spec)
        return false;
    *scale16 = spec->scales.block_k == 16;
    *a = spec->matrix_format;
    *b = spec->matrix_format_b;
    return true;
}

static inline int rocke_wmma_scale_format(const char* dtype)
{
    if(!dtype || strcmp(dtype, "e8m0") == 0 || strcmp(dtype, "i8") == 0)
    {
        return 0;
    }
    if(strcmp(dtype, "e5m3") == 0)
    {
        return 1;
    }
    return strcmp(dtype, "e4m3") == 0 ? 2 : -1;
}

static inline const char* rocke_wmma_scale_error(int a, int b, int sa, int sb)
{
    if(sa < 0 || sb < 0)
    {
        return "scaled WMMA scale types must be e8m0, e5m3, or e4m3";
    }
    if((a != 4 && sa != 0) || (b != 4 && sb != 0))
    {
        return "scaled WMMA e5m3/e4m3 scales require an FP4 operand";
    }
    if(a == 4 && b == 4 && sa != sb)
    {
        return "scaled WMMA FP4 x FP4 requires matching scale formats";
    }
    return NULL;
}

#endif /* ROCKE_WMMA_SCALE_INTERNAL_H */
