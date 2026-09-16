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
    rocke_e8m0_scale_packing_t scales;
} rocke_scaled_wmma_op_t;

static inline const rocke_scaled_wmma_op_t* rocke_gfx1250_scaled_wmma(const char* op_id)
{
    static const rocke_e8m0_scale_packing_t scale = {/*count=*/4, /*block_k=*/32};
    static const rocke_e8m0_scale_packing_t scale16 = {/*count=*/8, /*block_k=*/16};
    static const rocke_scaled_wmma_op_t ops[] = {
        {"wmma_scale_f32_16x16x128_fp8_fp8", 0, scale},
        {"wmma_scale16_f32_16x16x128_fp8_fp8", 0, scale16},
        {"wmma_scale_f32_16x16x128_fp4_fp4", 4, scale},
        {"wmma_scale16_f32_16x16x128_fp4_fp4", 4, scale16},
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

#endif /* ROCKE_WMMA_SCALE_INTERNAL_H */
