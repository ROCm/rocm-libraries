// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/mma_operand.py: the rule for passing an MMA operand as the
 * instruction's register type, and for reading its result back from the
 * register type it returns. The lowerers only render its answer. */
#include "rocke/mma_operand_internal.h"

#include <string>
#include <utility>

#include <string.h>

#include "rocke/error.hpp"
#include "rocke/ir_internal.h"

/* Machine operand roles: the multiplicands src0/src1 and accumulator src2.
 * The result has its own admission rule. */
static bool is_operand_role(rocke_mma_role_t role)
{
    return role == ROCKE_MMA_ROLE_SRC0 || role == ROCKE_MMA_ROLE_SRC1
           || role == ROCKE_MMA_ROLE_SRC2;
}

/* Python spells the roles "a", "b" and "c". */
static const char* role_name(rocke_mma_role_t role)
{
    return role == ROCKE_MMA_ROLE_SRC0 ? "a" : role == ROCKE_MMA_ROLE_SRC1 ? "b" : "c";
}

const rocke_value_t* rocke_mma_operand_value(const rocke_op_t* op, rocke_mma_role_t role)
{
    if(!is_operand_role(role))
        throw ckc::ValueError("an MMA operand role is A, B or C");
    /* A is operand 0, B operand 1 and C, the accumulator, operand 2. */
    const int index = role == ROCKE_MMA_ROLE_SRC0 ? 0 : role == ROCKE_MMA_ROLE_SRC1 ? 1 : 2;
    if(op->num_operands <= index)
        throw ckc::ValueError(ckc::format_error("%s has %d operands; an MMA op takes A, B and C",
                                                op->name ? op->name : "(unnamed op)",
                                                op->num_operands));
    return op->operands[index];
}

/* The MMA catalog row of target `gfx` for `op_id`, or NULL when its catalog
 * does not list it. */
static const rocke_mma_op_t* catalog_row(const char* op_id, const char* gfx)
{
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(gfx);
    if(!arch)
        throw ckc::ValueError(
            ckc::format_error("no MMA catalog for target '%s'", gfx ? gfx : "(null)"));
    return rocke_mma_catalog_by_op_id(&arch->mma, op_id);
}

/* The element types a value may hold in another shape than `register_type`
 * (Python _admitted_elements), sorted by name as Python joins them: the
 * register's own element type (a reshape, such as vec<f32x1> for f32) and the
 * element type of `catalog_dtype`, the dtype the MMA catalog row records for
 * the value (the instruction's packed values, such as vec<i8x16> for
 * vec<i32x4>). NULL (an op_id the catalog does not list) admits only the
 * first, and so does a packed-only catalog element. Writes one or two types;
 * returns how many. */
static int admitted_elements(const rocke_type_t* register_type,
                             const char* catalog_dtype,
                             const rocke_type_t* elements[2])
{
    int count = 0;
    elements[count++] = rocke_i_elem_of(register_type);
    if(!catalog_dtype)
        return count;
    const rocke_type_t* catalog_element = rocke_require_ir_type(catalog_dtype);
    /* A packed-only catalog element (i4 for the iu4 instructions) is never a
     * value's element type, so those values arrive in the register type. */
    if(rocke_i_is_packed_only(catalog_element))
        return count;
    if(!rocke_type_eq(catalog_element, elements[0]))
        elements[count++] = catalog_element;
    if(count == 2 && strcmp(elements[0]->name, elements[1]->name) > 0)
        std::swap(elements[0], elements[1]);
    return count;
}

/* Throw unless `value_type` is the bits of `register_type` in another shape:
 * one of the `num_admitted` admitted element types at the register's bit width
 * (Python _require_reshape). `label` names the value, `verb` is what the
 * instruction does with the register ("takes" or "returns"), and each hint
 * ends its refusal. */
static void require_reshape(const std::string& label,
                            const char* verb,
                            const rocke_type_t* value_type,
                            const rocke_type_t* register_type,
                            const rocke_type_t* const admitted[2],
                            int num_admitted,
                            const std::string& format_hint,
                            const std::string& width_hint)
{
    const rocke_type_t* element = rocke_i_elem_of(value_type);
    bool is_admitted = false;
    for(int i = 0; i < num_admitted; ++i)
        is_admitted = is_admitted || rocke_type_eq(element, admitted[i]);
    if(!is_admitted)
        throw ckc::ValueError(ckc::format_error("%s holds %s values; the instruction %s %s%s%s%s",
                                                label.c_str(),
                                                element->name,
                                                verb,
                                                admitted[0]->name,
                                                num_admitted == 2 ? " or " : "",
                                                num_admitted == 2 ? admitted[1]->name : "",
                                                format_hint.c_str()));
    const int value_bits = rocke_i_type_bits(value_type);
    const int register_bits = rocke_i_type_bits(register_type);
    if(value_bits != register_bits)
        throw ckc::ValueError(ckc::format_error("%s is %s (%d bits); the instruction %s %s (%d "
                                                "bits)%s",
                                                label.c_str(),
                                                value_type->name,
                                                value_bits,
                                                verb,
                                                register_type->name,
                                                register_bits,
                                                width_hint.c_str()));
}

bool rocke_mma_operand_needs_reinterpret(const char* op_id,
                                         rocke_mma_role_t role,
                                         const rocke_type_t* operand_type,
                                         const rocke_type_t* register_type,
                                         const char* gfx)
{
    if(!is_operand_role(role))
        throw ckc::ValueError("an MMA operand role is A, B or C");
    if(rocke_type_eq(operand_type, register_type))
        return false;
    const rocke_mma_op_t* catalog_op = catalog_row(op_id, gfx);
    const int index = role == ROCKE_MMA_ROLE_SRC0 ? 0 : role == ROCKE_MMA_ROLE_SRC1 ? 1 : 2;
    const char* catalog_dtype = catalog_op ? catalog_op->srcs[index].dtype : NULL;
    const rocke_type_t* admitted[2];
    const int num_admitted = admitted_elements(register_type, catalog_dtype, admitted);
    const int register_bits = rocke_i_type_bits(register_type);
    require_reshape(std::string("MMA operand ") + role_name(role) + " of '" + op_id + "'",
                    "takes",
                    operand_type,
                    register_type,
                    admitted,
                    num_admitted,
                    " -- convert the values first",
                    " -- pass a " + std::to_string(register_bits) + "-bit value");
    return true;
}

bool rocke_mma_result_needs_reinterpret(const char* op_id,
                                        const rocke_type_t* result_type,
                                        const rocke_type_t* register_type,
                                        const char* gfx)
{
    if(rocke_type_eq(result_type, register_type))
        return false;
    const rocke_mma_op_t* catalog_op = catalog_row(op_id, gfx);
    const rocke_type_t* admitted[2];
    const int num_admitted = admitted_elements(
        register_type, catalog_op ? rocke_mma_op_dst_dtype(catalog_op) : NULL, admitted);
    const int register_bits = rocke_i_type_bits(register_type);
    require_reshape(std::string("MMA result of '") + op_id + "'",
                    "returns",
                    result_type,
                    register_type,
                    admitted,
                    num_admitted,
                    " -- declare the result in an element type it returns",
                    " -- declare a " + std::to_string(register_bits) + "-bit result");
    return true;
}
