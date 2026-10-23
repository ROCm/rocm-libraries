// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Fixed MFMA lowering ABI checks, mirroring core/mfma.py::validate_mfma.
 * Inspect actual SSA values rather than re-resolving mutable catalog metadata.
 * Returned errors use the caller's buffer; no ownership is transferred. */
#ifndef ROCKE_MFMA_INTERNAL_H
#define ROCKE_MFMA_INTERNAL_H

#include <stdio.h>
#include <string.h>

#include "rocke/ir.h"

static inline int rocke_mfma_carrier_bits(const rocke_type_t* type)
{
    int count = 1;
    if(!type)
        return 0;
    if(type->kind == ROCKE_TYPE_VECTOR)
    {
        count = type->count;
        type = type->elem;
    }
    if(!type || type->kind != ROCKE_TYPE_SCALAR || count <= 0)
        return 0;
    int bits;
    switch(type->scalar)
    {
    case ROCKE_SCALAR_I1:
        bits = 1;
        break;
    case ROCKE_SCALAR_I8:
    case ROCKE_SCALAR_FP8E4M3:
    case ROCKE_SCALAR_BF8E5M2:
        bits = 8;
        break;
    case ROCKE_SCALAR_I16:
    case ROCKE_SCALAR_F16:
    case ROCKE_SCALAR_BF16:
        bits = 16;
        break;
    case ROCKE_SCALAR_I32:
    case ROCKE_SCALAR_F32:
        bits = 32;
        break;
    case ROCKE_SCALAR_F64:
    case ROCKE_SCALAR_I64:
        bits = 64;
        break;
    default:
        return 0;
    }
    // Only fixed MFMA carriers up to 256 bits are supported here.
    return count <= 256 / bits ? count * bits : 0;
}

/* Derive the canonical SSA spelling from the existing fixed LLVM spec table. */
static inline void rocke_mfma_ssa_type(const char* abi, char* name, size_t capacity)
{
    int count = 0;
    char element[32];
    if(sscanf(abi, "<%d x %31[^>]>", &count, element) != 2)
        snprintf(element, sizeof(element), "%s", abi);
    const char* scalar = strcmp(element, "float") == 0    ? "f32"
                         : strcmp(element, "half") == 0   ? "f16"
                         : strcmp(element, "bfloat") == 0 ? "bf16"
                                                          : element;
    if(count)
        snprintf(name, capacity, "vec<%sx%d>", scalar, count);
    else
        snprintf(name, capacity, "%s", scalar);
}

static inline int rocke_mfma_type_matches(const rocke_type_t* type, const char* expected)
{
    if(!type)
        return 0;
    if(strcmp(expected, "vec<i8x8>") == 0)
        return type->kind == ROCKE_TYPE_VECTOR && type->count == 8 && type->elem
               && type->elem->kind == ROCKE_TYPE_SCALAR
               && (type->elem->scalar == ROCKE_SCALAR_I8
                   || type->elem->scalar == ROCKE_SCALAR_FP8E4M3
                   || type->elem->scalar == ROCKE_SCALAR_BF8E5M2);
    int count;
    char element[32];
    if(sscanf(expected, "vec<%31[^x]x%d>", element, &count) == 2)
        return type->kind == ROCKE_TYPE_VECTOR && type->count == count && type->elem
               && type->elem->kind == ROCKE_TYPE_SCALAR
               && rocke_type_eq(type->elem, rocke_scalar_by_name(element));
    if(strcmp(expected, "f32") == 0 && type->kind == ROCKE_TYPE_VECTOR && type->count == 1)
        return rocke_type_eq(type->elem, rocke_f32());
    return type->kind == ROCKE_TYPE_SCALAR && rocke_type_eq(type, rocke_scalar_by_name(expected));
}

static inline const char* rocke_mfma_operand_error(const rocke_op_t* op,
                                                   const char* source_type,
                                                   const char* accumulator_type,
                                                   const char* result_type,
                                                   int packed_bits,
                                                   int scaled,
                                                   char* message,
                                                   size_t capacity)
{
    const int arity = scaled ? 5 : 3;
    if(op->num_operands != arity || op->num_results != 1)
    {
        snprintf(message, capacity, "MFMA expects %d operands and 1 result", arity);
        return message;
    }
    for(int role = 0; role < 2; ++role)
    {
        const rocke_value_t* value = op->operands[role];
        const rocke_type_t* type = value ? value->type : NULL;
        if(packed_bits ? rocke_mfma_carrier_bits(type) != packed_bits
                       : !rocke_mfma_type_matches(type, source_type))
        {
            if(packed_bits)
                snprintf(
                    message, capacity, "MFMA requires src0 and src1 to have %d bits", packed_bits);
            else
                snprintf(message, capacity, "MFMA requires src0 and src1 to be %s", source_type);
            return message;
        }
    }
    const rocke_value_t* values[] = {op->operands[2], op->results[0]};
    const char* expected[] = {accumulator_type, result_type};
    for(int role = 0; role < 2; ++role)
        if(!values[role] || !rocke_mfma_type_matches(values[role]->type, expected[role]))
        {
            if(strcmp(accumulator_type, result_type) == 0)
                snprintf(message, capacity, "MFMA requires src2 and dst to be %s", expected[role]);
            else
                snprintf(message,
                         capacity,
                         "MFMA requires %s to be %s",
                         role ? "dst" : "src2",
                         expected[role]);
            return message;
        }
    if(scaled)
        for(int role = 3; role < 5; ++role)
            if(!op->operands[role] || !rocke_type_eq(op->operands[role]->type, rocke_i32()))
            {
                snprintf(message, capacity, "MFMA requires scale operands to be i32");
                return message;
            }
    return NULL;
}

#endif /* ROCKE_MFMA_INTERNAL_H */
