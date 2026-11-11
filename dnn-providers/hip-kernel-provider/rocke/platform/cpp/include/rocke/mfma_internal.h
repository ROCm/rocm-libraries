// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Fixed MFMA lowering ABI checks, mirroring core/mfma.py::validate_mfma.
 * Inspect actual SSA values rather than re-resolving mutable catalog metadata.
 * Returned errors use the caller's buffer; no ownership is transferred. */
#ifndef ROCKE_MFMA_INTERNAL_H
#define ROCKE_MFMA_INTERNAL_H

#include <stdio.h>
#include <string.h>

#include "rocke/ir_internal.h"

static inline const rocke_type_t* rocke_mfma_carrier_element(const rocke_type_t* type)
{
    if(!type || (type->kind != ROCKE_TYPE_SCALAR && type->kind != ROCKE_TYPE_VECTOR)
       || rocke_i_count_of(type) <= 0)
        return NULL;
    const auto* element = rocke_i_elem_of(type);
    return element && element->kind == ROCKE_TYPE_SCALAR ? element : NULL;
}

static inline int rocke_mfma_carrier_bits(const rocke_type_t* type)
{
    const int count = rocke_i_count_of(type);
    type = rocke_mfma_carrier_element(type);
    if(!type)
        return 0;
    int bits;
    switch(type->scalar)
    {
    case ROCKE_SCALAR_I1:
        bits = 1;
        break;
    case ROCKE_SCALAR_I8:
    case ROCKE_SCALAR_U8:
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

static inline int rocke_mfma_type_matches(const rocke_type_t* type, const char* expected)
{
    if(!type)
        return 0;
    if(strcmp(expected, "vec<fp8e4m3x8>") == 0 || strcmp(expected, "vec<bf8e5m2x8>") == 0)
    {
        /* Byte carriers are untyped; logical floating values must match the instruction. */
        const rocke_scalar_kind_t format
            = strcmp(expected, "vec<fp8e4m3x8>") == 0 ? ROCKE_SCALAR_FP8E4M3 : ROCKE_SCALAR_BF8E5M2;
        return type->kind == ROCKE_TYPE_VECTOR && type->count == 8 && type->elem
               && type->elem->kind == ROCKE_TYPE_SCALAR
               && (type->elem->scalar == ROCKE_SCALAR_I8 || type->elem->scalar == ROCKE_SCALAR_U8
                   || type->elem->scalar == format);
    }
    int count;
    char element[32];
    if(sscanf(expected, "vec<%31[^x]x%d>", element, &count) == 2)
        return type->kind == ROCKE_TYPE_VECTOR && type->count == count && type->elem
               && type->elem->kind == ROCKE_TYPE_SCALAR
               && rocke_type_eq(type->elem, rocke_scalar_by_name(element));
    if(strcmp(expected, "f32") == 0 && type->kind == ROCKE_TYPE_VECTOR && type->count == 1)
        return rocke_type_eq(type->elem, rocke_f32());
    if(strcmp(expected, "f64") == 0 && type->kind == ROCKE_TYPE_VECTOR && type->count == 1)
        return rocke_type_eq(type->elem, rocke_f64());
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
    const int byte_source = !packed_bits
                            && (strcmp(source_type, "vec<fp8e4m3x8>") == 0
                                || strcmp(source_type, "vec<bf8e5m2x8>") == 0);
    for(int role = 0; role < 2; ++role)
    {
        const rocke_value_t* value = op->operands[role];
        const rocke_type_t* type = value ? value->type : NULL;
        if(packed_bits ? rocke_mfma_carrier_bits(type) != packed_bits
                       : !rocke_mfma_type_matches(type, source_type))
        {
            if(byte_source && type && type->kind == ROCKE_TYPE_VECTOR && type->count == 8
               && type->elem && type->elem->kind == ROCKE_TYPE_SCALAR
               && (type->elem->scalar == ROCKE_SCALAR_FP8E4M3
                   || type->elem->scalar == ROCKE_SCALAR_BF8E5M2))
                snprintf(
                    message,
                    capacity,
                    "MFMA source format mismatch - src%d=%s, expected %s or vec<i8x8> or vec<u8x8>",
                    role,
                    type->name,
                    source_type);
            else if(packed_bits)
            {
                const rocke_type_t* element = rocke_mfma_carrier_element(type);
                if(element && !rocke_mfma_carrier_bits(element))
                    snprintf(message,
                             capacity,
                             "MFMA unsupported packed carrier - src%d=%s, expected a supported "
                             "%d-bit register carrier",
                             role,
                             type->name,
                             packed_bits);
                else
                    snprintf(message,
                             capacity,
                             "MFMA requires src0 and src1 to have %d bits",
                             packed_bits);
            }
            else
                snprintf(message,
                         capacity,
                         "MFMA requires src0 and src1 to be %s",
                         byte_source ? "vec<i8x8>" : source_type);
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

/* The A and B type each fixed-signature MFMA instruction takes (Python
 * core/mfma.py _SOURCES), in the spelling rocke_mfma_operand_error reads; NULL
 * for any other op_id. */
static inline const char* rocke_mfma_source_type(const char* op_id)
{
    static const struct
    {
        const char* op_id;
        const char* source_type;
    } k_sources[] = {
        {"mfma_f32_16x16x4_f32", "f32"},
        {"mfma_f32_32x32x2_f32", "f32"},
        {"mfma_f32_16x16x16_f16", "vec<f16x4>"},
        {"mfma_f32_16x16x32_f16", "vec<f16x8>"},
        {"mfma_f32_32x32x8_f16", "vec<f16x4>"},
        {"mfma_f32_32x32x16_f16", "vec<f16x8>"},
        {"mfma_f32_4x4x4_f16", "vec<f16x4>"},
        {"mfma_f32_4x4x4_bf16", "vec<bf16x4>"},
        {"mfma_f64_16x16x4_f64", "f64"},
        {"mfma_f64_4x4x4_f64", "f64"},
        {"mfma_f32_16x16x16_bf16", "vec<bf16x4>"},
        {"mfma_f32_16x16x32_bf16", "vec<bf16x8>"},
        {"mfma_f32_32x32x8_bf16", "vec<bf16x4>"},
        {"mfma_f32_32x32x16_bf16", "vec<bf16x8>"},
        {"mfma_f32_16x16x32_fp8", "vec<fp8e4m3x8>"},
        {"mfma_f32_16x16x32_bf8", "vec<bf8e5m2x8>"},
        {"mfma_f32_32x32x16_fp8", "vec<fp8e4m3x8>"},
        {"mfma_f32_32x32x16_bf8", "vec<bf8e5m2x8>"},
    };
    for(const auto& row : k_sources)
        if(strcmp(row.op_id, op_id) == 0)
            return row.source_type;
    return NULL;
}

/* The accumulator and result type of MFMA `op_id` (Python validate_mfma's
 * rule): vec<f64x1> for the f64 4x4x4 atom, vec<f64x4> for the other f64 one,
 * vec<f32x16> for a 32x32 atom and vec<f32x4> for any other. */
static inline void rocke_mfma_accumulator_type(const char* op_id, char* type, size_t capacity)
{
    const int f64 = strncmp(op_id, "mfma_f64_", 9) == 0;
    const int count = strcmp(op_id, "mfma_f64_4x4x4_f64") == 0    ? 1
                      : strncmp(op_id, "mfma_f32_32x32", 14) == 0 ? 16
                                                                  : 4;
    snprintf(type, capacity, "vec<%sx%d>", f64 ? "f64" : "f32", count);
}

/* Python validate_mfma for a fixed-signature MFMA `op_id`: NULL when `op` fits
 * its instruction (rocke_mfma_source_type, rocke_mfma_accumulator_type), else
 * the Python text in `message`. An op_id with no fixed signature is refused. */
static inline const char* rocke_fixed_mfma_operand_error(const rocke_op_t* op,
                                                         const char* op_id,
                                                         char* message,
                                                         size_t capacity)
{
    const char* source_type = rocke_mfma_source_type(op_id);
    if(!source_type)
    {
        snprintf(message, capacity, "MFMA %s has no fixed operand signature", op_id);
        return message;
    }
    char accumulator_type[32];
    rocke_mfma_accumulator_type(op_id, accumulator_type, sizeof(accumulator_type));
    return rocke_mfma_operand_error(
        op, source_type, accumulator_type, accumulator_type, 0, 0, message, capacity);
}

#endif /* ROCKE_MFMA_INTERNAL_H */
