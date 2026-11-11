// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Emit one MMA instruction from its matrix-call description.
 *
 * Mirrors core/isa/matrix_call_lowering.py (emit_matrix_call), the rendering
 * half of core/isa/matrix_operand_forms.py (AsIs / ZeroPadded / EvenHalves)
 * and core/isa/matrix_calls.py's Signedness. Every line, fresh name and error
 * text matches the Python engine: fresh names are taken in the same order, so
 * the two emit the same bytes. */
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
    return kind == ROCKE_MATRIX_ARG_A   ? ROCKE_MMA_ROLE_SRC0
           : kind == ROCKE_MATRIX_ARG_B ? ROCKE_MMA_ROLE_SRC1
                                        : ROCKE_MMA_ROLE_SRC2;
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

/* Python ZeroPadded.operand: a packed payload of exactly `words` 32-bit words
 * widened to the <m x i32> register with zero high words, or the whole register
 * passed unchanged. */
static void check_zero_padded_operand(rocke_lower_t* L,
                                      const rocke_op_t* op,
                                      const char* op_id,
                                      rocke_matrix_arg_kind_t kind,
                                      const rocke_type_t* register_type,
                                      int words)
{
    const rocke_value_t* value = rocke_mma_operand_value(op, role_of(kind));
    const rocke_type_t* payload = value->type;
    if(!(is_vector(payload) && strcmp(payload->elem->name, "i32") == 0
         && (payload->count == words || payload->count == register_type->count)))
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA operand %s of '%s' is %s; the instruction takes the operand's %d "
                      "packed 32-bit words as vec<i32x%d>, or the whole %s register -- pass "
                      "one of those",
                      role_name(kind),
                      op_id,
                      payload->name,
                      words,
                      words,
                      rocke_ll_llvm_type(L, register_type));
}

static const char* zero_padded_operand(rocke_lower_t* L,
                                       const rocke_op_t* op,
                                       const char* op_id,
                                       rocke_matrix_arg_kind_t kind,
                                       const rocke_type_t* register_type,
                                       int words)
{
    check_zero_padded_operand(L, op, op_id, kind, register_type, words);
    const rocke_value_t* value = rocke_mma_operand_value(op, role_of(kind));
    const rocke_type_t* payload = value->type;
    if(payload->count == register_type->count)
        return rocke_ll_operand(L, value);
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
    /* The public operand renderer has already checked this form. */
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

/* Pure result admission shared by preflight and the two result renderers. */
static bool check_matrix_result(rocke_lower_t* L,
                                const rocke_op_t* op,
                                const char* op_id,
                                const rocke_type_t* register_type,
                                bool even_halves)
{
    const auto* result = op->results[0];
    if(!result || !result->type)
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "MMA result of '%s' must define an IR type", op_id);
    const auto* logical = even_halves ? even_halves_logical(L, register_type) : register_type;
    if(even_halves
       && !(is_vector(result->type) && rocke_i_type_bits(result->type) == rocke_i_type_bits(logical)
            && result->type->count == logical->count))
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA result of '%s' is %s; the instruction returns %d results in the even "
                      "halves of %s",
                      op_id,
                      result->type->name,
                      logical->count,
                      rocke_ll_llvm_type(L, register_type));
    return rocke_mma_result_needs_reinterpret(op_id, result->type, logical, L->backend->gfx);
}

/* Python AsIs.emit_result: the call's value, or one bitcast of it, as the
 * result rule (rocke_mma_result_needs_reinterpret) decides. */
static void as_is_result(rocke_lower_t* L,
                         const rocke_op_t* op,
                         const char* op_id,
                         const rocke_type_t* register_type,
                         const std::string& call)
{
    const rocke_value_t* result = op->results[0];
    if(!check_matrix_result(L, op, op_id, register_type, false))
    {
        rocke_ll_emitf(L, "  %s = %s", result->name, call.c_str());
        return;
    }
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
    const bool reinterpret = check_matrix_result(L, op, op_id, register_type, true);
    const char* returned = rocke_ll_fresh(L, "mma_d");
    rocke_ll_emitf(L, "  %s = %s", returned, call.c_str());
    const char* gathered = reinterpret ? rocke_ll_fresh(L, "mma_gather") : result->name;
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
    if(reinterpret)
        rocke_ll_emitf(L,
                       "  %s = bitcast %s %s to %s",
                       result->name,
                       rocke_ll_llvm_type(L, logical),
                       gathered,
                       rocke_ll_llvm_type(L, result->type));
}

void rocke_ll_check_matrix_operand(rocke_lower_t* L,
                                   const rocke_op_t* op,
                                   const char* op_id,
                                   rocke_matrix_arg_kind_t kind,
                                   const rocke_matrix_form_t* form)
{
    const auto* value = rocke_mma_operand_value(op, role_of(kind));
    if(!value || !value->type)
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "MMA operand %s of '%s' must define an IR type",
                      role_name(kind),
                      op_id);
    const auto* register_type = rocke_matrix_form_type_at(&L->arena, form, L->flavor);
    if(form->kind == ROCKE_MATRIX_FORM_ZERO_PADDED)
    {
        check_zero_padded_operand(L, op, op_id, kind, register_type, form->payload_words);
        return;
    }
    if(form->kind == ROCKE_MATRIX_FORM_EVEN_HALVES)
    {
        const auto* logical = even_halves_logical(L, register_type);
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
        register_type = logical;
    }
    rocke_mma_operand_needs_reinterpret(
        op_id, role_of(kind), value->type, register_type, L->backend->gfx);
}

const char* rocke_ll_matrix_operand(rocke_lower_t* L,
                                    const rocke_op_t* op,
                                    const char* op_id,
                                    rocke_matrix_arg_kind_t kind,
                                    const rocke_matrix_form_t* form)
{
    rocke_ll_check_matrix_operand(L, op, op_id, kind, form);
    const rocke_type_t* register_type = rocke_matrix_form_type_at(&L->arena, form, L->flavor);
    switch(form->kind)
    {
    case ROCKE_MATRIX_FORM_ZERO_PADDED:
        return zero_padded_operand(L, op, op_id, kind, register_type, form->payload_words);
    case ROCKE_MATRIX_FORM_EVEN_HALVES:
        return even_halves_operand(L, op, op_id, kind, register_type);
    case ROCKE_MATRIX_FORM_AS_IS:
    default:
        return as_is_operand(L, op, op_id, kind, register_type);
    }
}

const rocke_mma_op_t* rocke_ll_require_catalog_op(rocke_lower_t* L, const char* op_id)
{
    const char* gfx = (L->backend && L->backend->gfx) ? L->backend->gfx : NULL;
    const rocke_arch_target_t* target = gfx ? rocke_arch_target_from_gfx(gfx) : NULL;
    const rocke_mma_op_t* catalog_op
        = target ? rocke_mma_catalog_by_op_id(&target->mma, op_id) : NULL;
    if(!catalog_op)
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "tile.%s is not available on %s: its MMA catalog does not list it",
                      op_id,
                      gfx ? gfx : "(unknown)");
    return catalog_op;
}

/* Python Signedness.argument: "i1 1" for a signed integer operand of the
 * catalog row, "i1 0" for an unsigned one (rocke_matrix_operand_is_signed). */
static std::string
    signedness_argument(rocke_lower_t* L, const char* op_id, rocke_matrix_arg_kind_t operand)
{
    const rocke_mma_op_t* catalog_op = rocke_ll_require_catalog_op(L, op_id);
    return std::string("i1 ") + (rocke_matrix_operand_is_signed(catalog_op, operand) ? "1" : "0");
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
    if(declared && strcmp(declared, derived) != 0)
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "matrix call for '%s' declares '%s' as '%s', but the declaration table "
                      "holds '%s'",
                      op_id,
                      key,
                      derived,
                      declared);
    /* Every source and result must pass before declarations or transformations
     * can change the lowerer. Rendering reuses the same admission functions. */
    for(int i = 0; i < call->num_args; ++i)
    {
        const auto& arg = call->args[i];
        if(arg.kind == ROCKE_MATRIX_ARG_SIGNEDNESS)
            signedness_argument(L, op_id, arg.operand);
        else if(arg.kind != ROCKE_MATRIX_ARG_IMM)
            rocke_ll_check_matrix_operand(L, op, op_id, arg.kind, &form_of(call, arg.kind));
    }
    const auto* result_type = rocke_matrix_form_type_at(&L->arena, &call->result, L->flavor);
    check_matrix_result(
        L, op, op_id, result_type, call->result.kind == ROCKE_MATRIX_FORM_EVEN_HALVES);
    if(!declared)
        rocke_ll_need_dynamic(L, key, derived);
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
        if(arg.kind == ROCKE_MATRIX_ARG_SIGNEDNESS)
        {
            arguments += signedness_argument(L, op_id, arg.operand);
            continue;
        }
        const rocke_matrix_form_t& form = form_of(call, arg.kind);
        const rocke_type_t* register_type = rocke_matrix_form_type_at(&L->arena, &form, L->flavor);
        const char* value = rocke_ll_matrix_operand(L, op, op_id, arg.kind, &form);
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
