// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * range_marker.cpp -- the arith.assume_range marker: its rules and its builder.
 *
 * Mirrors rocke.core.ir: _assume_range_errors (the one home of the marker's
 * attr rules, with the same texts), assume_range_bound_operands (the operand
 * layout) and IRBuilder._assume_range (rocke_b_assume_range). The verifier
 * (cpp/core/verify.cpp) reports every message; the builder raises the first.
 *
 * The marker records ge + ge_plus <= v < lt + lt_plus and v == residue (mod
 * multiple_of). Its operands are the marked value, then the ge operand when
 * ge_plus is present, then the lt operand when one operand is left. It emits
 * no instruction.
 */
#include <stdint.h>
#include <string.h>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"

namespace
{

/* ir._ASSUME_RANGE_INT_ATTRS, in the order their type is checked. */
const char* const kIntAttrs[] = {"multiple_of", "residue", "ge", "ge_plus", "lt", "lt_plus"};

/* ir.ASSUME_RANGE_WRITERS: the writers a marker's `by` attr may name. */
const char* const kWriters[] = {"gpu_checked_value"};

std::string fmt_int(int64_t v)
{
    return std::to_string((long long)v);
}

} // namespace

extern "C" int rocke_assume_range_errors(const rocke_attr_map_t* attrs,
                                         int num_operands,
                                         rocke_param_fact_sink_t sink,
                                         void* ctx)
{
    int count = 0;
    auto report = [&](const std::string& message) {
        ++count;
        if(sink)
            sink(ctx, message.c_str());
    };
    for(const char* key : kIntAttrs)
    {
        const rocke_attr_value_t* value = rocke_attr_get(attrs, key);
        if(value && value->kind != ROCKE_ATTR_INT)
            report(std::string("arith.assume_range: ") + key + " must be an integer");
    }
    if(const rocke_attr_value_t* by = rocke_attr_get(attrs, "by"))
    {
        if(by->kind != ROCKE_ATTR_STR)
        {
            report("arith.assume_range: by must be a string");
        }
        else
        {
            bool known = false;
            std::string names;
            for(const char* writer : kWriters)
            {
                known = known || strcmp(by->u.s, writer) == 0;
                names += (names.empty() ? "" : ", ") + std::string(writer);
            }
            if(!known)
                report(std::string("arith.assume_range: by names no marker writer: '") + by->u.s
                       + "' (one of: " + names + ")");
        }
    }
    if(count)
    {
        /* The value rules below compare these attrs as integers. */
        return count;
    }
    int64_t multiple_of = 1;
    int64_t residue = 0;
    int64_t ge = 0;
    int64_t lt = 0;
    rocke_attr_get_int(attrs, "multiple_of", &multiple_of);
    rocke_attr_get_int(attrs, "residue", &residue);
    if(multiple_of < 1)
        report("arith.assume_range: multiple_of must be >= 1, got " + fmt_int(multiple_of));
    else if(!(residue >= 0 && residue < multiple_of))
        report("arith.assume_range: residue must be in [0, " + fmt_int(multiple_of) + "), got "
               + fmt_int(residue));
    const bool has_ge_operand = rocke_attr_get(attrs, "ge_plus") != NULL;
    const int ends = num_operands - 1;
    const bool has_ge = rocke_attr_get_int(attrs, "ge", &ge);
    const bool has_lt = rocke_attr_get_int(attrs, "lt", &lt);
    if(has_ge_operand && ends < 1)
        report("arith.assume_range: ge_plus is given without a ge operand");
    if(ends > 1 && !has_ge_operand)
        report("arith.assume_range: two Value ends need ge_plus (the operands are the value, "
               "the ge operand, then the lt operand)");
    if(has_ge && has_ge_operand && ends >= 1)
        report("arith.assume_range: ge is given both as an attr and as an operand; "
               "state one lower end");
    const bool has_lt_operand = ends - (has_ge_operand ? 1 : 0) >= 1;
    if(has_lt && has_lt_operand)
        report("arith.assume_range: lt is given both as an attr and as an operand; "
               "state one upper end");
    if(rocke_attr_get(attrs, "lt_plus") != NULL && !has_lt_operand)
        report("arith.assume_range: lt_plus is given without an lt operand");
    if(has_ge && has_lt && ge >= lt)
        report("arith.assume_range: ge (" + fmt_int(ge) + ") must be below lt (" + fmt_int(lt)
               + ")");
    return count;
}

extern "C" void
    rocke_assume_range_bound_operands(const rocke_op_t* op, rocke_value_t** ge, rocke_value_t** lt)
{
    int next = 1;
    *ge = NULL;
    *lt = NULL;
    if(rocke_attr_get(&op->attrs, "ge_plus") != NULL && next < op->num_operands)
        *ge = op->operands[next++];
    if(next < op->num_operands)
        *lt = op->operands[next];
}

static void rocke_i_first_marker_error(void* ctx, const char* message)
{
    std::string* first = (std::string*)ctx;
    if(first->empty())
        *first = message;
}

rocke_value_t* rocke_b_assume_range(rocke_ir_builder_t* b,
                                    rocke_value_t* v,
                                    const rocke_assume_range_opts_t* opts)
{
    if(!rocke_i_live(b))
        return NULL;
    if(v == NULL || opts == NULL)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "_assume_range: NULL argument");
    if(!rocke_i_type_is(v->type, "i32") && !rocke_i_type_is(v->type, "i64"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "_assume_range: v must be i32 or i64, got %s", v->type->name);
    const struct
    {
        const char* word;
        const char* plus;
        const rocke_range_end_t* end;
    } ends[] = {{"ge", "ge_plus", &opts->ge}, {"lt", "lt_plus", &opts->lt}};
    for(const auto& e : ends)
    {
        if(e.end->end != NULL && !rocke_type_eq(e.end->end->type, v->type))
            return (rocke_value_t*)rocke_i_set_err(b,
                                                   ROCKE_ERR_VALUE,
                                                   "_assume_range: %s type %s != v type %s",
                                                   e.word,
                                                   e.end->end->type->name,
                                                   v->type->name);
    }
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "multiple_of", opts->multiple_of_set ? opts->multiple_of : 1);
    rocke_attr_set_int(b, &attrs, "residue", opts->residue);
    rocke_value_t* operands[3] = {v, NULL, NULL};
    int num_operands = 1;
    for(const auto& e : ends)
    {
        if(e.end->end != NULL)
        {
            operands[num_operands++] = e.end->end;
            rocke_attr_set_int(b, &attrs, e.plus, e.end->plus);
        }
        else
        {
            if(e.end->set)
                rocke_attr_set_int(b, &attrs, e.word, e.end->value);
            if(e.end->plus != 0)
                rocke_attr_set_int(b, &attrs, e.plus, e.end->plus);
        }
    }
    if(opts->by != NULL)
        rocke_attr_set_str(b, &attrs, "by", opts->by);
    std::string first;
    rocke_assume_range_errors(&attrs, num_operands, rocke_i_first_marker_error, &first);
    if(!first.empty())
        ckc::raise_status(ROCKE_ERR_VALUE, first.c_str());
    return rocke_i_op1(
        b, ROCKE_OP_ARITH_ASSUME_RANGE, operands, num_operands, v->type, &attrs, "range");
}
