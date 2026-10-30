// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Matrix intrinsic calls described as data: the table, the schema checks and
 * the declaration derivation. Mirrors core/isa/matrix_calls.py and the type
 * half of core/isa/matrix_operand_forms.py; the emission half lives in
 * cpp/core/lower_llvm/matrix_call.cpp. */
#include "rocke/matrix_calls_internal.h"

#include <array>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include <string.h>

#include "rocke/error.hpp"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm_internal.h"

namespace ckc
{

/* op_id -> call, for every MMA instruction lowered from data rather than by a
 * hand-written handler (Python MATRIX_CALLS, same order). */
static const std::array<rocke_matrix_call_entry_t, 0> k_matrix_calls{};

const rocke_matrix_call_entry_t* rocke_matrix_calls(int* out_count)
{
    if(out_count)
        *out_count = (int)k_matrix_calls.size();
    return k_matrix_calls.data();
}

const rocke_matrix_call_t* rocke_matrix_call_for(const char* op_id)
{
    if(!op_id)
        return NULL;
    for(const rocke_matrix_call_entry_t& entry : k_matrix_calls)
    {
        if(strcmp(entry.op_id, op_id) == 0)
            return &entry.call;
    }
    return NULL;
}

const char* rocke_matrix_call_key(const rocke_matrix_call_t* call)
{
    return call->decl_key ? call->decl_key : call->intrinsic;
}

/* ------------------------------------------------------------ register types */

static const rocke_type_t* scalar_type(const char* name)
{
    const rocke_type_t* t = rocke_scalar_by_name(name);
    if(!t)
        throw ValueError(format_error("unknown IR scalar type '%s'", name ? name : "(null)"));
    return t;
}

/* The flavors on the ladder as Python prints the tuple LLVM_FLAVORS, for the
 * errors that list the valid choices: ('llvm20', 'llvm22', 'llvm23'). */
static std::string flavor_choices()
{
    std::string text = "(";
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        text += i ? ", '" : "'";
        text += rocke_llvm_flavor_at(i);
        text += "'";
    }
    return text + ")";
}

/* The type a register takes at `flavor`: its newest override at or below the
 * flavor on the ladder, else its base (Python FlavorType.for_flavor). */
static rocke_matrix_type_t register_at(const rocke_matrix_register_t* reg,
                                       rocke_llvm_flavor_t flavor)
{
    int rung = -1;
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        if(rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(i)) == flavor)
            rung = i;
    }
    if(rung < 0)
        throw ValueError(format_error("unknown LLVM flavor '%s'; expected one of %s",
                                      rocke_llvm_flavor_name(flavor),
                                      flavor_choices().c_str()));
    rocke_matrix_type_t resolved = reg->base;
    for(int i = 0; i <= rung; ++i)
    {
        for(int j = 0; j < reg->num_overrides; ++j)
        {
            if(strcmp(reg->overrides[j].flavor, rocke_llvm_flavor_at(i)) == 0)
                resolved = reg->overrides[j].type;
        }
    }
    return resolved;
}

const rocke_type_t* rocke_matrix_form_type_at(rocke_arena_t* arena,
                                              const rocke_matrix_form_t* form,
                                              rocke_llvm_flavor_t flavor)
{
    const rocke_matrix_type_t t = register_at(&form->register_type, flavor);
    const rocke_type_t* elem = scalar_type(t.elem);
    if(t.count == 0)
        return elem;
    const rocke_type_t* vec = rocke_i_vector_type_in_arena(arena, elem, t.count);
    if(!vec)
        throw std::bad_alloc();
    return vec;
}

/* The IR name of a register type ("vec<f16x8>", "i32"). */
static std::string type_name(const rocke_matrix_type_t& t)
{
    if(t.count == 0)
        return t.elem;
    return format_error("vec<%sx%d>", t.elem, t.count);
}

/* The LLVM text of a register type (Python _llvm_type). */
static std::string llvm_text(const rocke_matrix_type_t& t)
{
    const char* elem = rocke_ll_llvm_type(NULL, scalar_type(t.elem));
    if(t.count == 0)
        return elem;
    return format_error("<%d x %s>", t.count, elem);
}

/* ---------------------------------------------------------------- the schema */

static const char* const k_immediate_types[] = {"i1", "i16", "i32", "i64"};

static void check_immediate(const rocke_matrix_arg_t& arg)
{
    int bits = 0;
    for(const char* name : k_immediate_types)
    {
        if(arg.imm_type && strcmp(arg.imm_type, name) == 0)
            bits = rocke_i_type_bits(scalar_type(name));
    }
    if(bits == 0)
        throw TypeError(format_error("an immediate is one of i1, i16, i32, i64, got %s",
                                     arg.imm_type ? arg.imm_type : "(null)"));
    const long long low = -(1LL << (bits - 1));
    const bool fits
        = arg.imm_value >= low && (bits == 64 || arg.imm_value < (long long)(1ULL << bits));
    if(!fits)
        throw ValueError(
            format_error("%lld does not fit an %s immediate", arg.imm_value, arg.imm_type));
}

/* Every type a register takes at some flavor (Python FlavorType.every_flavor). */
template <class Fn>
static void for_every_type(const rocke_matrix_register_t& reg, Fn fn)
{
    fn(reg.base);
    for(int i = 0; i < reg.num_overrides; ++i)
        fn(reg.overrides[i].type);
}

static void check_form(const rocke_matrix_form_t& form)
{
    for(int i = 0; i < form.register_type.num_overrides; ++i)
    {
        const char* flavor = form.register_type.overrides[i].flavor;
        if(!rocke_llvm_flavor_is_known(rocke_llvm_flavor_from_name(flavor)))
            throw ValueError(format_error("FlavorType names unknown LLVM flavor '%s'; expected "
                                          "one of %s",
                                          flavor ? flavor : "(null)",
                                          flavor_choices().c_str()));
    }
    if(form.kind == ROCKE_MATRIX_FORM_ZERO_PADDED)
    {
        for_every_type(form.register_type, [](const rocke_matrix_type_t& t) {
            if(t.count == 0 || strcmp(t.elem, "i32") != 0)
                throw TypeError(
                    format_error("ZeroPadded register type must be a vector of i32 words, got %s",
                                 type_name(t).c_str()));
        });
    }
    if(form.kind == ROCKE_MATRIX_FORM_EVEN_HALVES)
    {
        for_every_type(form.register_type, [](const rocke_matrix_type_t& t) {
            if(t.count == 0 || t.count % 2 != 0 || rocke_i_type_bits(scalar_type(t.elem)) != 16)
                throw TypeError(format_error("EvenHalves register type must be a vector of an even "
                                             "number of 16-bit elements, got %s",
                                             type_name(t).c_str()));
        });
    }
}

void rocke_matrix_call_check(const rocke_matrix_call_t* call)
{
    static const char* const k_operand_names[] = {"A", "B", "C"};
    int seen[3] = {0, 0, 0};
    std::string got;
    for(int i = 0; i < call->num_args; ++i)
    {
        const rocke_matrix_arg_t& arg = call->args[i];
        if(arg.kind == ROCKE_MATRIX_ARG_IMM)
        {
            check_immediate(arg);
            continue;
        }
        seen[arg.kind] += 1;
        got += got.empty() ? "'" : ", '";
        got += k_operand_names[arg.kind];
        got += "'";
    }
    if(seen[0] != 1 || seen[1] != 1 || seen[2] != 1)
        throw ValueError(
            format_error("MatrixCall '%s' must pass each of A, B and C exactly once, got [%s]",
                         call->intrinsic,
                         got.c_str()));
    if(call->result.kind == ROCKE_MATRIX_FORM_ZERO_PADDED)
        throw TypeError(
            format_error("MatrixCall '%s' result form must be AsIs or EvenHalves, got ZeroPadded",
                         call->intrinsic));
    for(const rocke_matrix_form_t* form : {&call->a, &call->b, &call->c, &call->result})
        check_form(*form);
}

/* ------------------------------------------------------------ declarations */

static const rocke_matrix_form_t& operand_form(const rocke_matrix_call_t* call,
                                               rocke_matrix_arg_kind_t kind)
{
    return kind == ROCKE_MATRIX_ARG_A ? call->a : kind == ROCKE_MATRIX_ARG_B ? call->b : call->c;
}

const char* rocke_matrix_call_declaration(const rocke_matrix_call_t* call,
                                          rocke_llvm_flavor_t flavor)
{
    /* Interned by content: a call's text never changes, and a set node never
     * moves, so the returned pointer stays valid for the process. */
    static std::mutex lock;
    static std::set<std::string> interned;

    std::string text = "declare " + llvm_text(register_at(&call->result.register_type, flavor))
                       + " @" + call->intrinsic + "(";
    for(int i = 0; i < call->num_args; ++i)
    {
        const rocke_matrix_arg_t& arg = call->args[i];
        if(i)
            text += ", ";
        if(arg.kind == ROCKE_MATRIX_ARG_IMM)
        {
            text += arg.imm_type;
            if(arg.immarg)
                text += " immarg";
            continue;
        }
        text += llvm_text(register_at(&operand_form(call, arg.kind).register_type, flavor));
    }
    text += ")";

    std::lock_guard<std::mutex> guard(lock);
    return interned.insert(std::move(text)).first->c_str();
}

} /* namespace ckc */
