// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/tf32.py. Validation sees logical types, never carrier equality. */
#ifndef ROCKE_TF32_INTERNAL_H
#define ROCKE_TF32_INTERNAL_H
#include "rocke/ir.h"
#include <string.h>
static inline int rocke_tf32_mma_count(const char* id)
{
    if(!id)
        return 0;
    if(strcmp(id, "mfma_f32_16x16x8_xf32") == 0)
        return 4;
    if(strcmp(id, "mfma_f32_32x32x4_xf32") == 0)
        return 16;
    return 0;
}
static inline const char* rocke_tf32_mma_error(const char* id, rocke_value_t* const* args, int n)
{
    int count = rocke_tf32_mma_count(id);
    if(!count)
    {
        for(int i = 0; i < n; ++i)
            if(args[i]
               && (strcmp(args[i]->type->name, "tf32") == 0
                   || strncmp(args[i]->type->name, "vec<tf32x", 9) == 0))
                return "TF32 operands require an XF32 MMA";
        return NULL;
    }
    const char* acc = count == 4 ? "vec<f32x4>" : "vec<f32x16>";
    if(n != 3 || !args[0] || !args[1] || !args[2] || strcmp(args[0]->type->name, "vec<tf32x2>") != 0
       || strcmp(args[1]->type->name, "vec<tf32x2>") != 0 || strcmp(args[2]->type->name, acc) != 0)
        return "XF32 MMA requires two vec<tf32x2> operands and its FP32 accumulator";
    return NULL;
}
static inline const char* rocke_tf32_op_error(const rocke_op_t* op)
{
    const char* id = rocke_attr_get_str(&op->attrs, "op_id");
    if(!id && strncmp(op->name, "tile.", 5) == 0)
        id = op->name + 5;
    const char* error = (strcmp(op->name, "tile.mma") == 0 || strncmp(op->name, "tile.mfma", 9) == 0
                         || strncmp(op->name, "tile.wmma", 9) == 0)
                            ? rocke_tf32_mma_error(id, op->operands, op->num_operands)
                            : NULL;
    if(error)
        return error;
    int count = rocke_tf32_mma_count(id);
    if(count
       && (op->num_results != 1
           || strcmp(op->results[0]->type->name, count == 4 ? "vec<f32x4>" : "vec<f32x16>") != 0))
        return "XF32 MMA result must match its FP32 accumulator";
    if((strncmp(op->name, "arith.", 6) == 0 || strncmp(op->name, "math.", 5) == 0)
       && strcmp(op->name, "arith.bitcast") != 0 && strcmp(op->name, "arith.select") != 0)
    {
        for(int i = 0; i < op->num_operands + op->num_results; ++i)
        {
            const rocke_value_t* v
                = i < op->num_operands ? op->operands[i] : op->results[i - op->num_operands];
            if(strcmp(v->type->name, "tf32") == 0 || strncmp(v->type->name, "vec<tf32x", 9) == 0)
                return "TF32 arithmetic requires an explicit conversion to f32";
        }
    }
    return NULL;
}
#endif
