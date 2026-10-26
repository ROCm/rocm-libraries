// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/mma_operand.py: the rule for passing an MMA operand as the
 * instruction's register type. The lowerers only render its answer. */
#include "rocke/mma_operand_internal.h"

#include <string.h>
#include <utility>

#include "rocke/error.hpp"
#include "rocke/ir_internal.h"

/* The multiplicand roles: A and B. The operand rule applies to no other. */
static bool is_multiplicand(rocke_mma_role_t role)
{
    return role == ROCKE_MMA_ROLE_A || role == ROCKE_MMA_ROLE_B;
}

const rocke_value_t* rocke_mma_operand_value(const rocke_op_t* op, rocke_mma_role_t role)
{
    if(!is_multiplicand(role))
        throw ckc::ValueError("an MMA operand role is A or B");
    /* A is operand 0 and B operand 1; C, the accumulator, follows them. */
    const int index = role == ROCKE_MMA_ROLE_A ? 0 : 1;
    if(op->num_operands <= index)
        throw ckc::ValueError(ckc::format_error("%s has %d operands; an MMA op takes A, B and C",
                                                op->name ? op->name : "(unnamed op)",
                                                op->num_operands));
    return op->operands[index];
}

/* The element types MMA operand `role` may hold in another shape (Python
 * _admitted_elements), sorted by name as Python joins them: the register's own
 * element type (a reshape, such as vec<f32x1> for f32) and the element type the
 * MMA catalog of `gfx` records for that operand (the instruction's packed
 * values, such as vec<i8x16> for vec<i32x4>) unless it is packed-only. Writes
 * one or two types; returns how many. */
static int admitted_elements(const char* op_id,
                             rocke_mma_role_t role,
                             const rocke_type_t* register_type,
                             const char* gfx,
                             const rocke_type_t* elements[2])
{
    int count = 0;
    elements[count++] = rocke_i_elem_of(register_type);
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(gfx);
    if(!arch)
        throw ckc::ValueError(
            ckc::format_error("no MMA catalog for target '%s'", gfx ? gfx : "(null)"));
    const rocke_mma_op_t* catalog_op = rocke_mma_catalog_by_op_id(&arch->mma, op_id);
    if(!catalog_op)
        return count;
    const rocke_type_t* catalog_element = rocke_require_ir_type(
        role == ROCKE_MMA_ROLE_A ? catalog_op->a_dtype : catalog_op->b_dtype);
    /* A packed-only catalog element (i4 for the iu4 instructions) is never a
     * value's element type, so those operands arrive in the register type. */
    if(rocke_i_is_packed_only(catalog_element))
        return count;
    if(!rocke_type_eq(catalog_element, elements[0]))
        elements[count++] = catalog_element;
    if(count == 2 && strcmp(elements[0]->name, elements[1]->name) > 0)
        std::swap(elements[0], elements[1]);
    return count;
}

bool rocke_mma_operand_needs_reinterpret(const char* op_id,
                                         rocke_mma_role_t role,
                                         const rocke_type_t* operand_type,
                                         const rocke_type_t* register_type,
                                         const char* gfx)
{
    if(!is_multiplicand(role))
        throw ckc::ValueError("an MMA operand role is A or B");
    if(rocke_type_eq(operand_type, register_type))
        return false;
    /* Python spells the roles "a" and "b". */
    const char* role_name = role == ROCKE_MMA_ROLE_A ? "a" : "b";
    const rocke_type_t* element = rocke_i_elem_of(operand_type);
    const rocke_type_t* admitted[2];
    const int num_admitted = admitted_elements(op_id, role, register_type, gfx, admitted);
    bool is_admitted = false;
    for(int i = 0; i < num_admitted; ++i)
        is_admitted = is_admitted || rocke_type_eq(element, admitted[i]);
    if(!is_admitted)
        throw ckc::ValueError(ckc::format_error(
            "MMA operand %s of '%s' holds %s values; the instruction takes %s%s%s -- convert "
            "the values first",
            role_name,
            op_id,
            element->name,
            admitted[0]->name,
            num_admitted == 2 ? " or " : "",
            num_admitted == 2 ? admitted[1]->name : ""));
    const int operand_bits = rocke_i_type_bits(operand_type);
    const int register_bits = rocke_i_type_bits(register_type);
    if(operand_bits != register_bits)
        throw ckc::ValueError(ckc::format_error(
            "MMA operand %s of '%s' is %s (%d bits); the instruction takes %s (%d bits) -- "
            "pass a %d-bit value",
            role_name,
            op_id,
            operand_type->name,
            operand_bits,
            register_type->name,
            register_bits,
            register_bits));
    return true;
}
