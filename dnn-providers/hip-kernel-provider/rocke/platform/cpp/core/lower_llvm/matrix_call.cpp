// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Emit one MMA instruction from its matrix-call description.
 *
 * Mirrors core/isa/matrix_call_lowering.py (emit_matrix_call) and the
 * rendering half of core/isa/matrix_operand_forms.py (AsIs / ZeroPadded /
 * EvenHalves). Every line, fresh name and error text matches the Python engine:
 * fresh names are taken in the same order, so the two emit the same bytes. */
#include "rocke/matrix_calls_internal.h"

#include <string>

#include <string.h>

#include "rocke/error.hpp"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm_internal.h"
#include "rocke/mma_operand_internal.h"

namespace ckc
{

/* The MMA operand role a call argument passes, and its Python spelling. */
static rocke_mma_role_t role_of(rocke_matrix_arg_kind_t kind)
{
    return kind == ROCKE_MATRIX_ARG_A   ? ROCKE_MMA_ROLE_A
           : kind == ROCKE_MATRIX_ARG_B ? ROCKE_MMA_ROLE_B
                                        : ROCKE_MMA_ROLE_ACC;
}

static const char* role_name(rocke_matrix_arg_kind_t kind)
{
    return kind == ROCKE_MATRIX_ARG_A ? "a" : kind == ROCKE_MATRIX_ARG_B ? "b" : "c";
}

static const rocke_matrix_form_t& form_of(const rocke_matrix_call_t* call,
                                          rocke_matrix_arg_kind_t kind)
{
    return kind == ROCKE_MATRIX_ARG_A ? call->a : kind == ROCKE_MATRIX_ARG_B ? call->b : call->c;
}

static bool is_vector(const rocke_type_t* t)
{
    return t && t->kind == ROCKE_TYPE_VECTOR;
}

/* Python AsIs.operand: unchanged, or one bitcast, as the operand rule decides. */
static const char* as_is_operand(rocke_lower_t* L,
                                 const rocke_op_t* op,
                                 const char* op_id,
                                 rocke_matrix_arg_kind_t kind,
                                 const rocke_type_t* register_type)
{
    const rocke_value_t* value = rocke_mma_operand_value(op, role_of(kind));
    if(rocke_mma_operand_needs_reinterpret(
           op_id, role_of(kind), value->type, register_type, L->backend->gfx))
        return rocke_ll_reinterpret(L, value, register_type);
    return rocke_ll_operand(L, value);
}

/* Python ZeroPadded.operand: a packed <n x i32> payload, n <= m, widened to
 * the <m x i32> register with zero high words. */
static const char* zero_padded_operand(rocke_lower_t* L,
                                       const rocke_op_t* op,
                                       const char* op_id,
                                       rocke_matrix_arg_kind_t kind,
                                       const rocke_type_t* register_type)
{
    const rocke_value_t* value = rocke_mma_operand_value(op, role_of(kind));
    const rocke_type_t* payload = value->type;
    if(!(is_vector(payload) && strcmp(payload->elem->name, "i32") == 0
         && payload->count <= register_type->count))
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA operand %s of '%s' is %s; the instruction takes a packed payload of "
                      "whole 32-bit words in %s -- pass the operand's packed words",
                      role_name(kind),
                      op_id,
                      payload->name,
                      rocke_ll_llvm_type(L, register_type));
    if(payload->count == register_type->count)
        return rocke_ll_operand(L, value);
    const int words = payload->count;
    const int width = register_type->count;
    std::string mask;
    for(int i = 0; i < width; ++i)
    {
        const int lane = i < words ? i : words;
        mask += (i ? ", i32 " : "i32 ") + std::to_string(lane);
    }
    const char* padded = rocke_ll_fresh(L, "mma_pad");
    rocke_ll_emitf(
        L,
        "  %s = shufflevector <%d x i32> %s, <%d x i32> zeroinitializer, <%d x i32> <%s>",
        padded,
        words,
        rocke_ll_operand(L, value),
        words,
        width,
        mask.c_str());
    return padded;
}

/* The n-element vector of the register's element that EvenHalves exposes. */
static const rocke_type_t* even_halves_logical(rocke_lower_t* L, const rocke_type_t* register_type)
{
    const rocke_type_t* logical
        = rocke_i_vector_type_in_arena(&L->arena, register_type->elem, register_type->count / 2);
    if(!logical)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "EvenHalves logical type: OOM");
    return logical;
}

/* Python EvenHalves.operand: n logical 16-bit values spread into the even
 * half-slots of the 2n-slot register, odd slots zero. */
static const char* even_halves_operand(rocke_lower_t* L,
                                       const rocke_op_t* op,
                                       const char* op_id,
                                       rocke_matrix_arg_kind_t kind,
                                       const rocke_type_t* register_type)
{
    const rocke_type_t* logical = even_halves_logical(L, register_type);
    const rocke_value_t* value = rocke_mma_operand_value(op, role_of(kind));
    if(!(is_vector(value->type) && value->type->count == logical->count
         && rocke_i_type_bits(value->type->elem) == 16))
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA operand %s of '%s' is %s; the instruction keeps %d results in the "
                      "even halves of %s -- pass the %d logical accumulator values",
                      role_name(kind),
                      op_id,
                      value->type->name,
                      logical->count,
                      rocke_ll_llvm_type(L, register_type),
                      logical->count);
    const char* values = as_is_operand(L, op, op_id, kind, logical);
    std::string mask;
    for(int i = 0; i < logical->count; ++i)
    {
        mask += (i ? ", i32 " : "i32 ") + std::to_string(i);
        mask += ", i32 " + std::to_string(logical->count);
    }
    const char* spread = rocke_ll_fresh(L, "mma_spread");
    const char* logical_text = rocke_ll_llvm_type(L, logical);
    rocke_ll_emitf(L,
                   "  %s = shufflevector %s %s, %s zeroinitializer, <%d x i32> <%s>",
                   spread,
                   logical_text,
                   values,
                   logical_text,
                   register_type->count,
                   mask.c_str());
    return spread;
}

/* Python AsIs.emit_result. */
static void as_is_result(rocke_lower_t* L,
                         const rocke_op_t* op,
                         const char* op_id,
                         const rocke_type_t* register_type,
                         const std::string& call)
{
    const rocke_value_t* result = op->results[0];
    if(rocke_type_eq(result->type, register_type))
    {
        rocke_ll_emitf(L, "  %s = %s", result->name, call.c_str());
        return;
    }
    const int result_bits = rocke_i_type_bits(result->type);
    const int register_bits = rocke_i_type_bits(register_type);
    if(result_bits != register_bits)
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA result of '%s' is %s (%d bits); the instruction returns %s (%d bits)",
                      op_id,
                      result->type->name,
                      result_bits,
                      register_type->name,
                      register_bits);
    const char* returned = rocke_ll_fresh(L, "mma_d");
    rocke_ll_emitf(L, "  %s = %s", returned, call.c_str());
    rocke_ll_emitf(L,
                   "  %s = bitcast %s %s to %s",
                   result->name,
                   rocke_ll_llvm_type(L, register_type),
                   returned,
                   rocke_ll_llvm_type(L, result->type));
}

/* Python EvenHalves.emit_result: gather the even half-slots into the result. */
static void even_halves_result(rocke_lower_t* L,
                               const rocke_op_t* op,
                               const char* op_id,
                               const rocke_type_t* register_type,
                               const std::string& call)
{
    const rocke_type_t* logical = even_halves_logical(L, register_type);
    const rocke_value_t* result = op->results[0];
    if(!(is_vector(result->type) && rocke_i_type_bits(result->type) == rocke_i_type_bits(logical)
         && result->type->count == logical->count))
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA result of '%s' is %s; the instruction returns %d results in the even "
                      "halves of %s",
                      op_id,
                      result->type->name,
                      logical->count,
                      rocke_ll_llvm_type(L, register_type));
    const char* returned = rocke_ll_fresh(L, "mma_d");
    rocke_ll_emitf(L, "  %s = %s", returned, call.c_str());
    const bool same = rocke_type_eq(result->type, logical);
    const char* gathered = same ? result->name : rocke_ll_fresh(L, "mma_gather");
    std::string mask;
    for(int i = 0; i < logical->count; ++i)
        mask += (i ? ", i32 " : "i32 ") + std::to_string(2 * i);
    const char* register_text = rocke_ll_llvm_type(L, register_type);
    rocke_ll_emitf(L,
                   "  %s = shufflevector %s %s, %s poison, <%d x i32> <%s>",
                   gathered,
                   register_text,
                   returned,
                   register_text,
                   logical->count,
                   mask.c_str());
    if(!same)
        rocke_ll_emitf(L,
                       "  %s = bitcast %s %s to %s",
                       result->name,
                       rocke_ll_llvm_type(L, logical),
                       gathered,
                       rocke_ll_llvm_type(L, result->type));
}

/* The declaration text the lowerer resolves for `key`: the flavor's table, then
 * the dynamically registered declarations. NULL when neither holds it. */
static const char* declared_text(rocke_lower_t* L, const char* key)
{
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        if(strcmp(L->dyn_decls.data[i].key, key) == 0)
            return L->dyn_decls.data[i].decl;
    }
    return rocke_ll_decl_text(L->flavor, key);
}

void rocke_ll_emit_matrix_call(rocke_lower_t* L,
                               const rocke_op_t* op,
                               const char* op_id,
                               const rocke_matrix_call_t* call)
{
    if(op->num_operands != 3)
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "tile.%s expects 3 operands, got %d", op_id, op->num_operands);
    if(op->num_results != 1)
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "tile.%s: expected exactly one result, got %d",
                      op_id,
                      op->num_results);
    rocke_matrix_call_check(call);

    /* A call outside the table (one a test builds) registers its derived
     * declaration the way the other dynamically declared intrinsics do; a table
     * entry whose text differs from the derived one is refused. */
    const char* key = rocke_matrix_call_key(call);
    const char* derived = rocke_matrix_call_declaration(call, L->flavor);
    const char* declared = declared_text(L, key);
    if(!declared)
        rocke_ll_need_dynamic(L, key, derived);
    else if(strcmp(declared, derived) != 0)
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "matrix call for '%s' declares '%s' as '%s', but the declaration table "
                      "holds '%s'",
                      op_id,
                      key,
                      derived,
                      declared);
    rocke_ll_need(L, key);

    std::string arguments;
    for(int i = 0; i < call->num_args; ++i)
    {
        const rocke_matrix_arg_t& arg = call->args[i];
        if(i)
            arguments += ", ";
        if(arg.kind == ROCKE_MATRIX_ARG_IMM)
        {
            arguments += std::string(arg.imm_type) + " " + std::to_string(arg.imm_value);
            continue;
        }
        const rocke_matrix_form_t& form = form_of(call, arg.kind);
        const rocke_type_t* register_type = rocke_matrix_form_type_at(&L->arena, &form, L->flavor);
        const char* value = NULL;
        switch(form.kind)
        {
        case ROCKE_MATRIX_FORM_AS_IS:
            value = as_is_operand(L, op, op_id, arg.kind, register_type);
            break;
        case ROCKE_MATRIX_FORM_ZERO_PADDED:
            value = zero_padded_operand(L, op, op_id, arg.kind, register_type);
            break;
        case ROCKE_MATRIX_FORM_EVEN_HALVES:
            value = even_halves_operand(L, op, op_id, arg.kind, register_type);
            break;
        }
        arguments += std::string(rocke_ll_llvm_type(L, register_type)) + " " + value;
    }

    const rocke_type_t* returned = rocke_matrix_form_type_at(&L->arena, &call->result, L->flavor);
    const std::string text = std::string("call ") + rocke_ll_llvm_type(L, returned) + " @"
                             + call->intrinsic + "(" + arguments + ")";
    if(call->result.kind == ROCKE_MATRIX_FORM_EVEN_HALVES)
        even_halves_result(L, op, op_id, returned, text);
    else
        as_is_result(L, op, op_id, returned, text);
}

} /* namespace ckc */
