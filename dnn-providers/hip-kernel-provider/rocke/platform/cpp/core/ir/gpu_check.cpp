// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gpu_check.cpp -- rocke_b_gpu_checked, the C++ engine's
 * IRBuilder.gpu_checked_value.
 *
 * Mirrors, with the same texts and the same emitted ops:
 *   - rocke.core.value_sources: who can see a value (sources_of), the load
 *     taint walk (first_unchecked_arithmetic) and how a message names a value
 *     (value_text);
 *   - rocke.core.ir: one_unsigned_compare_checks, _gpu_check_form, the refusal
 *     texts and IRBuilder.gpu_checked_value's emission.
 *
 * Both walks visit values in the same order as the Python ones, so both engines
 * name the same value in the same message.
 */
#include <stdint.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/strbuf.h"

namespace
{

using value_list_t = std::vector<rocke_value_t*>;

/* value_sources._MEMORY_READ_PARTS */
const char* const kMemoryReadParts[] = {"load", "atomic", "ds_read"};

bool is(const rocke_op_t* op, rocke_opcode_t opcode)
{
    return op != NULL && op->opcode == opcode;
}

bool reads_memory(const rocke_op_t* op)
{
    if(rocke_op_is_pure(op))
        return false;
    for(const char* part : kMemoryReadParts)
        if(strstr(op->name, part) != NULL)
            return true;
    return false;
}

bool is_wrapping_arithmetic(const rocke_op_t* op)
{
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
    case ROCKE_OP_ARITH_SUB:
    case ROCKE_OP_ARITH_MUL:
    case ROCKE_OP_ARITH_DIV:
    case ROCKE_OP_ARITH_MOD:
        return true;
    default:
        return false;
    }
}

const char* symbol_of(const rocke_op_t* op)
{
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        return "+";
    case ROCKE_OP_ARITH_SUB:
        return "-";
    case ROCKE_OP_ARITH_MUL:
        return "*";
    case ROCKE_OP_ARITH_DIV:
        return "/";
    case ROCKE_OP_ARITH_MOD:
        return "%";
    default:
        return NULL;
    }
}

std::string fmt_int(int64_t v)
{
    return std::to_string((long long)v);
}

std::string upper_type(const rocke_type_t* t)
{
    std::string s = t->name;
    for(char& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

/* The builder's view of the kernel: which Values are its parameters. */
struct kernel_view_t
{
    const rocke_ir_builder_t* b;

    /* Python Value.param: the declaration of a parameter's Value. */
    const rocke_param_t* param_of(const rocke_value_t* v) const
    {
        for(int i = 0; i < b->num_param_lookup; i++)
        {
            if(b->param_values[i] != v)
                continue;
            const rocke_kernel_def_t* k = b->kernel;
            for(int j = 0; j < k->num_params; j++)
                if(strcmp(k->params[j]->name, b->param_names[i]) == 0)
                    return k->params[j];
        }
        return NULL;
    }

    /* value_sources._atom_text */
    std::string atom_text(const rocke_value_t* v) const
    {
        if(const rocke_param_t* p = param_of(v))
            return p->name;
        const rocke_op_t* op = v->op;
        if(op == NULL)
            return v->name;
        if(is(op, ROCKE_OP_ARITH_CONSTANT))
        {
            const rocke_attr_value_t* c = rocke_attr_get(&op->attrs, "value");
            if(c && c->kind == ROCKE_ATTR_INT)
                return fmt_int(c->u.i);
        }
        if(is(op, ROCKE_OP_GPU_THREAD_ID) || is(op, ROCKE_OP_GPU_BLOCK_ID))
        {
            const char* axis = rocke_attr_get_str(&op->attrs, "axis");
            if(axis)
                return std::string(is(op, ROCKE_OP_GPU_THREAD_ID) ? "thread_id_" : "block_id_")
                       + axis;
        }
        return v->name;
    }

    /* value_sources.value_text */
    std::string value_text(const rocke_value_t* v) const
    {
        const rocke_op_t* op = v->op;
        if(param_of(v) == NULL && op != NULL && symbol_of(op) && op->num_operands == 2)
            return atom_text(op->operands[0]) + " " + symbol_of(op) + " "
                   + atom_text(op->operands[1]);
        return atom_text(v);
    }
};

/* value_sources._carried_index: -1 for the induction variable. */
int carried_index(const rocke_value_t* v, const rocke_op_t* loop)
{
    for(int i = 0; i < loop->num_results; i++)
        if(loop->results[i] == v)
            return i;
    const char* iv = rocke_attr_get_str(&loop->attrs, "iv");
    if(iv && strcmp(v->name, iv) == 0)
        return -1;
    const rocke_attr_value_t* args = rocke_attr_get(&loop->attrs, "iter_args");
    if(args && args->kind == ROCKE_ATTR_LIST)
        for(int i = 0; i < args->u.list.count; i++)
        {
            const char* name = rocke_attr_get_str(args->u.list.items[i], "name");
            if(name && strcmp(v->name, name) == 0)
                return i;
        }
    return -1;
}

/* value_sources._yielded */
void yielded(const rocke_region_t* region, int index, value_list_t* out)
{
    if(region->num_ops == 0)
        return;
    const rocke_op_t* last = region->ops[region->num_ops - 1];
    if(is(last, ROCKE_OP_SCF_YIELD) && index < last->num_operands)
        out->push_back(last->operands[index]);
}

/* value_sources._loop_inputs */
value_list_t loop_inputs(const rocke_value_t* v, const rocke_op_t* loop, bool data)
{
    rocke_value_t* lower = loop->operands[0];
    rocke_value_t* upper = loop->operands[1];
    rocke_value_t* step = loop->operands[2];
    const int index = carried_index(v, loop);
    if(index < 0)
        return data ? value_list_t{lower, step} : value_list_t{lower, upper, step};
    value_list_t carried{loop->operands[3 + index]};
    yielded(loop->regions[0], index, &carried);
    if(data)
        return carried;
    value_list_t all{lower, upper, step};
    all.insert(all.end(), carried.begin(), carried.end());
    return all;
}

/* value_sources._branch_inputs */
value_list_t branch_inputs(const rocke_value_t* v, const rocke_op_t* branch, bool data)
{
    int index = 0;
    while(index < branch->num_results && branch->results[index] != v)
        ++index;
    value_list_t out;
    if(!data)
        out.push_back(branch->operands[0]);
    for(int r = 0; r < branch->num_regions; r++)
        yielded(branch->regions[r], index, &out);
    return out;
}

/* value_sources.Sources */
struct sources_t
{
    bool gpu_only = false;
    value_list_t params;
    value_list_t thread_ids;
};

/* value_sources.sources_of */
sources_t sources_of(const kernel_view_t& view, rocke_value_t* value)
{
    sources_t out;
    std::unordered_set<const rocke_value_t*> seen;
    value_list_t stack{value};
    while(!stack.empty())
    {
        rocke_value_t* v = stack.back();
        stack.pop_back();
        if(!seen.insert(v).second)
            continue;
        const rocke_op_t* op = v->op;
        if(view.param_of(v) != NULL)
        {
            out.params.push_back(v);
            continue;
        }
        if(op == NULL)
        {
            out.gpu_only = true;
            return out;
        }
        if(is(op, ROCKE_OP_ARITH_CONSTANT))
            continue;
        if(is(op, ROCKE_OP_GPU_THREAD_ID))
        {
            out.thread_ids.push_back(v);
            continue;
        }
        value_list_t inputs;
        if(is(op, ROCKE_OP_SCF_FOR))
            inputs = loop_inputs(v, op, false);
        else if(is(op, ROCKE_OP_SCF_IF_ELSE))
            inputs = branch_inputs(v, op, false);
        else if(rocke_op_is_pure(op) && op->num_operands > 0)
            inputs.assign(op->operands, op->operands + op->num_operands);
        else
        {
            /* A load, a block id, or any other value the host cannot compute. */
            out.gpu_only = true;
            return out;
        }
        stack.insert(stack.end(), inputs.rbegin(), inputs.rend());
    }
    return out;
}

/* value_sources._data_inputs */
value_list_t data_inputs(const kernel_view_t& view, const rocke_value_t* v)
{
    const rocke_op_t* op = v->op;
    if(view.param_of(v) != NULL || op == NULL)
        return {};
    if(is(op, ROCKE_OP_ARITH_CONSTANT) || is(op, ROCKE_OP_ARITH_ASSUME_RANGE)
       || is(op, ROCKE_OP_ARITH_CMP) || is(op, ROCKE_OP_ARITH_FCMP))
        return {};
    if(is(op, ROCKE_OP_SCF_FOR))
        return loop_inputs(v, op, true);
    if(is(op, ROCKE_OP_SCF_IF_ELSE))
        return branch_inputs(v, op, true);
    if(is(op, ROCKE_OP_ARITH_SELECT))
        return value_list_t(op->operands + 1, op->operands + op->num_operands);
    return value_list_t(op->operands, op->operands + op->num_operands);
}

/* value_sources._post_order */
value_list_t post_order(const kernel_view_t& view, rocke_value_t* value)
{
    std::unordered_set<const rocke_value_t*> seen;
    value_list_t order;
    std::vector<std::pair<rocke_value_t*, bool>> stack{{value, false}};
    while(!stack.empty())
    {
        auto [v, expanded] = stack.back();
        stack.pop_back();
        if(expanded)
        {
            order.push_back(v);
            continue;
        }
        if(!seen.insert(v).second)
            continue;
        stack.push_back({v, true});
        value_list_t inputs = data_inputs(view, v);
        for(auto it = inputs.rbegin(); it != inputs.rend(); ++it)
            if(!seen.count(*it))
                stack.push_back({*it, false});
    }
    return order;
}

/* value_sources.loaded_directly */
bool loaded_directly(const kernel_view_t& view, const rocke_value_t* v)
{
    while(is(v->op, ROCKE_OP_TILE_READFIRSTLANE))
        v = v->op->operands[0];
    return v->op != NULL && view.param_of(v) == NULL && reads_memory(v->op);
}

/* value_sources.first_unchecked_arithmetic: the op and its tainted operand. */
std::pair<const rocke_op_t*, const rocke_value_t*>
    first_unchecked_arithmetic(const kernel_view_t& view, rocke_value_t* value)
{
    const value_list_t order = post_order(view, value);
    std::unordered_map<const rocke_value_t*, bool> tainted;
    for(const rocke_value_t* v : order)
        tainted[v] = v->op != NULL && view.param_of(v) == NULL && reads_memory(v->op);
    bool changed = true;
    while(changed) /* a loop-carried value can be tainted by what its body yields */
    {
        changed = false;
        for(const rocke_value_t* v : order)
        {
            if(tainted[v])
                continue;
            for(const rocke_value_t* d : data_inputs(view, v))
                if(tainted[d])
                {
                    tainted[v] = true;
                    changed = true;
                    break;
                }
        }
    }
    for(const rocke_value_t* v : order)
    {
        const rocke_op_t* op = v->op;
        if(op == NULL || view.param_of(v) != NULL || !is_wrapping_arithmetic(op))
            continue;
        /* Name the loaded value itself when the op has one: that is what to check. */
        const rocke_value_t* first = NULL;
        for(int i = 0; i < op->num_operands; i++)
        {
            const rocke_value_t* operand = op->operands[i];
            if(!tainted[operand])
                continue;
            if(loaded_directly(view, operand))
                return {op, operand};
            if(first == NULL)
                first = operand;
        }
        if(first != NULL)
            return {op, first};
    }
    return {NULL, NULL};
}

/* ir._marker_states_facts / ir._states_facts */
bool states_facts(const kernel_view_t& view, const rocke_value_t* v)
{
    if(const rocke_param_t* p = view.param_of(v))
    {
        for(const char* word : {"ge", "gt", "le", "lt", "eq", "multiple_of"})
            if(rocke_attr_get(&p->attrs, word))
                return true;
        return false;
    }
    const rocke_op_t* op = v->op;
    if(!is(op, ROCKE_OP_ARITH_ASSUME_RANGE))
        return false;
    rocke_value_t* ge;
    rocke_value_t* lt;
    rocke_assume_range_bound_operands(op, &ge, &lt);
    int64_t multiple_of = 1;
    rocke_attr_get_int(&op->attrs, "multiple_of", &multiple_of);
    return rocke_attr_get(&op->attrs, "ge") || rocke_attr_get(&op->attrs, "lt") || ge || lt
           || multiple_of > 1;
}

/* One stated word of the call: its bound (int, relation or Value). */
struct word_t
{
    const char* word;
    const rocke_int_bound_t* bound;
};

bool stated(const rocke_int_bound_t& bound)
{
    return bound.set || bound.relation != NULL || bound.value_bound != NULL;
}

std::string bound_text(const kernel_view_t& view, const rocke_int_bound_t* bound)
{
    return bound->value_bound ? view.value_text(bound->value_bound) : fmt_int(bound->value);
}

[[noreturn]] void refuse(const std::string& message)
{
    ckc::raise_status(ROCKE_ERR_VALUE, message.c_str());
}

/* ir._gpu_check_form: refuses a call that states no checkable fact; returns the
 * inclusive int ends the int bounds give. */
void gpu_check_form(const kernel_view_t& view,
                    const rocke_value_t* value,
                    const std::vector<word_t>& words,
                    const rocke_int_facts_t* facts,
                    bool* has_lower,
                    int64_t* lower,
                    bool* has_upper,
                    int64_t* upper)
{
    const rocke_type_t* t = value->type;
    const bool is_i64 = rocke_i_type_is(t, "i64");
    if(!rocke_i_type_is(t, "i32") && !is_i64)
        refuse(std::string("gpu_checked_value: the value is ") + t->name
               + " -- a GPU check takes an i32 or i64 value");
    if(words.empty() && !facts->multiple_of_set)
        refuse("gpu_checked_value: state at least one fact: ge, gt, le, lt or multiple_of");
    auto has = [&](const char* w) {
        for(const word_t& entry : words)
            if(strcmp(entry.word, w) == 0)
                return true;
        return false;
    };
    if(has("gt") && has("ge"))
        refuse("gpu_checked_value: states two lower bounds (gt, ge) -- state one");
    if(has("lt") && has("le"))
        refuse("gpu_checked_value: states two upper bounds (lt, le) -- state one");
    const int64_t tmin = is_i64 ? INT64_MIN : INT32_MIN;
    const int64_t tmax = is_i64 ? INT64_MAX : INT32_MAX;
    if(facts->multiple_of_set)
    {
        if(facts->multiple_of < 1)
            refuse("gpu_checked_value: multiple_of must be >= 1, got "
                   + fmt_int(facts->multiple_of));
        if(facts->multiple_of > tmax)
            refuse("gpu_checked_value: multiple_of=" + fmt_int(facts->multiple_of)
                   + " is outside the range of " + t->name);
    }
    for(const word_t& entry : words)
    {
        const rocke_int_bound_t* bound = entry.bound;
        if(bound->value_bound != NULL)
        {
            const std::string shown = view.value_text(bound->value_bound);
            if(!rocke_type_eq(bound->value_bound->type, t))
                refuse(std::string("gpu_checked_value: ") + entry.word + "=" + shown + " is "
                       + bound->value_bound->type->name + " and the value is " + t->name
                       + " -- a bound has the value's type");
            if(!states_facts(view, bound->value_bound))
                refuse(std::string("gpu_checked_value: ") + entry.word + "=" + shown
                       + " has no stated facts -- a Value bound is a kernel parameter whose "
                         "declaration states facts, or a Value gpu_checked_value returned");
        }
        else if(bound->relation != NULL)
        {
            rocke_strbuf_t text;
            rocke_strbuf_init(&text, 64);
            rocke_relation_display(bound->relation, &text);
            const std::string shown = rocke_strbuf_cstr(&text);
            rocke_strbuf_free(&text);
            refuse(std::string("gpu_checked_value: ") + entry.word + "=" + shown
                   + " is a host expression -- on the GPU a bound is an int or one Value; "
                     "compute it with b.add / b.mul, where the instruction can be seen");
        }
        else if(bound->value < tmin || bound->value > tmax)
        {
            refuse(std::string("gpu_checked_value: ") + entry.word + "=" + fmt_int(bound->value)
                   + " is outside the range of " + t->name);
        }
    }
    auto int_bound
        = [&](const rocke_int_bound_t& b) { return b.set && !b.relation && !b.value_bound; };
    /* gt at the type's maximum or lt at its minimum leaves nothing; caught
     * before the +1 / -1 that would leave int64. */
    bool empty = false;
    *has_lower = *has_upper = false;
    if(int_bound(facts->ge))
    {
        *has_lower = true;
        *lower = facts->ge.value;
    }
    else if(int_bound(facts->gt))
    {
        *has_lower = true;
        empty = facts->gt.value == tmax;
        *lower = empty ? tmax : facts->gt.value + 1;
    }
    if(int_bound(facts->le))
    {
        *has_upper = true;
        *upper = facts->le.value;
    }
    else if(int_bound(facts->lt))
    {
        *has_upper = true;
        empty = empty || facts->lt.value == tmin;
        *upper = facts->lt.value == tmin ? tmin : facts->lt.value - 1;
    }
    const int64_t least = *has_lower ? *lower : tmin;
    const int64_t most = *has_upper ? *upper : tmax;
    if(empty || least > most)
    {
        std::string ints;
        int count = 0;
        for(const word_t& entry : words)
            if(int_bound(*entry.bound))
            {
                ints += (ints.empty() ? "" : " and ") + std::string(entry.word) + "="
                        + fmt_int(entry.bound->value);
                ++count;
            }
        refuse("gpu_checked_value: " + ints + (count > 1 ? " leave" : " leaves") + " no " + t->name
               + " value");
    }
}

/* ir._stated_call_text */
std::string stated_call_text(const kernel_view_t& view,
                             const std::vector<word_t>& words,
                             const rocke_int_facts_t* facts)
{
    std::string out;
    for(const word_t& entry : words)
        out += (out.empty() ? "" : ", ") + std::string(entry.word) + "="
               + bound_text(view, entry.bound);
    if(facts->multiple_of_set)
        out += (out.empty() ? "" : ", ") + std::string("multiple_of=")
               + fmt_int(facts->multiple_of);
    return out;
}

/* ir._gpu_check_host_visible_text */
std::string host_visible_text(const kernel_view_t& view,
                              const rocke_value_t* value,
                              const sources_t& sources,
                              const std::vector<word_t>& words,
                              const rocke_int_facts_t* facts,
                              bool has_upper,
                              int64_t upper)
{
    const std::string text = view.value_text(value);
    if(const rocke_param_t* p = view.param_of(value))
    {
        const std::string name = p->name;
        const size_t count = words.size() + (facts->multiple_of_set ? 1 : 0);
        rocke_strbuf_t declaration;
        rocke_strbuf_init(&declaration, 64);
        rocke_param_declaration_text(p, &declaration);
        std::string out
            = "gpu_checked_value: " + name
              + " is a kernel parameter. The host checks every fact of a kernel parameter "
                "before every launch and refuses the launch if one fails, so a GPU check here "
                "would only repeat that check.\nFix: state the "
              + (count > 1 ? "facts" : "fact") + " where " + name + " is declared:\n    b.param(\""
              + name + "\", " + upper_type(value->type) + ", "
              + stated_call_text(view, words, facts) + ")\n" + name
              + " is declared today as: " + rocke_strbuf_cstr(&declaration);
        rocke_strbuf_free(&declaration);
        return out;
    }
    if(!sources.thread_ids.empty())
    {
        const rocke_op_t* op = value->op;
        const char* axis = op ? rocke_attr_get_str(&op->attrs, "axis") : NULL;
        if(is(op, ROCKE_OP_GPU_THREAD_ID) && axis && strcmp(axis, "x") == 0 && has_upper)
            return "gpu_checked_value: " + text
                   + " is bounded by the kernel's workgroup size, which the host checks.\n"
                     "Fix: b.kernel.attrs[\"workgroup_size\"] = "
                   + fmt_int(upper + 1);
        std::string names;
        for(const rocke_value_t* p : sources.params)
            names += (names.empty() ? "" : ", ") + std::string(view.param_of(p)->name);
        const std::string params = names.empty() ? "" : " and kernel parameters (" + names + ")";
        const std::string declarations
            = names.empty() ? "" : ", and the parameters' facts on their declarations";
        return "gpu_checked_value: " + text + " is computed only from thread ids" + params
               + ", which the host bounds before every launch.\n"
                 "Fix: state the workgroup size, b.kernel.attrs[\"workgroup_size\"] = "
                 "<threads per workgroup>"
               + declarations + "; the IR range walk derives the facts of " + text + " from them.";
    }
    if(sources.params.empty())
        return "gpu_checked_value: " + text
               + " is computed only from constants, so it is known when the kernel is built -- "
                 "there is nothing to check on the GPU";
    std::string names;
    std::string declarations;
    for(const rocke_value_t* p : sources.params)
    {
        const std::string name = view.param_of(p)->name;
        names += (names.empty() ? "" : ", ") + name;
        declarations += (declarations.empty() ? "" : ", ") + std::string("b.param(\"") + name
                        + "\", " + upper_type(p->type) + ", ...)";
    }
    const bool one = sources.params.size() == 1;
    const std::string fix = one ? "state the bound on the parameter: " + declarations
                                : "state the bounds on the parameters: " + declarations;
    return "gpu_checked_value: " + text + " is computed only from kernel parameters (" + names
           + "), which the host checks before every launch. Fix: " + fix
           + ". The IR range walk derives the facts of " + text + " from " + (one ? "it" : "them")
           + ".";
}

/* ir._gpu_check_unchecked_text */
std::string
    unchecked_text(const kernel_view_t& view, const rocke_op_t* op, const rocke_value_t* operand)
{
    const char* why = "";
    const char* verb = "";
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        why = "the addition can wrap or become poison first", verb = "add";
        break;
    case ROCKE_OP_ARITH_SUB:
        why = "the subtraction can wrap or become poison first", verb = "subtract";
        break;
    case ROCKE_OP_ARITH_MUL:
        why = "the multiplication can wrap or become poison first", verb = "multiply";
        break;
    case ROCKE_OP_ARITH_DIV:
        why = "the division is undefined for a zero or overflowing divisor", verb = "divide";
        break;
    default:
        why = "the remainder is undefined for a zero or overflowing divisor";
        verb = "take the remainder of";
        break;
    }
    const std::string shown = view.value_text(operand);
    const char* dot = strchr(op->name, '.');
    const std::string short_name = dot ? dot + 1 : op->name;
    return "gpu_checked_value: " + shown + " "
           + (loaded_directly(view, operand) ? "is loaded from memory"
                                             : "is computed from a value loaded from memory")
           + " and unchecked, and " + short_name + " uses it before this check (" + why
           + "). Fix: check " + shown + " with gpu_checked_value first, then " + verb
           + " the Values gpu_checked_value returned.";
}

} // namespace

extern "C" bool
    rocke_i_one_unsigned_compare_checks(int64_t least, int64_t most, const rocke_type_t* t)
{
    const int64_t end = rocke_i_type_is(t, "i64") ? INT64_MAX : (int64_t)INT32_MAX;
    /* most < 2^(bits-1), i.e. most <= the signed maximum. */
    return 0 <= least && most <= end;
}

rocke_value_t*
    rocke_b_gpu_checked(rocke_ir_builder_t* b, rocke_value_t* value, const rocke_int_facts_t* facts)
{
    if(!rocke_i_live(b))
        return NULL;
    if(value == NULL || facts == NULL)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "gpu_checked_value: NULL argument");
    if(stated(facts->eq))
        refuse("gpu_checked_value: eq states a baked length on a parameter's declaration -- "
               "on the GPU state ge and le");
    const kernel_view_t view{b};
    std::vector<word_t> words;
    const word_t candidates[]
        = {{"ge", &facts->ge}, {"gt", &facts->gt}, {"le", &facts->le}, {"lt", &facts->lt}};
    for(const word_t& entry : candidates)
        if(stated(*entry.bound))
            words.push_back(entry);
    bool has_lower;
    bool has_upper;
    int64_t lower = 0;
    int64_t upper = 0;
    gpu_check_form(view, value, words, facts, &has_lower, &lower, &has_upper, &upper);
    const sources_t sources = sources_of(view, value);
    if(!sources.gpu_only)
        refuse(host_visible_text(view, value, sources, words, facts, has_upper, upper));
    const auto unchecked = first_unchecked_arithmetic(view, value);
    if(unchecked.first != NULL)
        refuse(unchecked_text(view, unchecked.first, unchecked.second));

    const bool is_i64 = rocke_i_type_is(value->type, "i64");
    auto constant
        = [&](int64_t v) { return is_i64 ? rocke_b_const_i64(b, v) : rocke_b_const_i32(b, v); };
    const word_t* lower_word = NULL;
    const word_t* upper_word = NULL;
    for(const word_t& entry : words)
    {
        if(strcmp(entry.word, "ge") == 0 || strcmp(entry.word, "gt") == 0)
            lower_word = &entry;
        else
            upper_word = &entry;
    }
    auto end_of = [&](const word_t* entry) {
        return entry->bound->value_bound ? entry->bound->value_bound
                                         : constant(entry->bound->value);
    };
    value_list_t fails;
    const int64_t tmax = is_i64 ? INT64_MAX : (int64_t)INT32_MAX;
    if(has_lower && lower == 0 && has_upper && upper < tmax
       && rocke_i_one_unsigned_compare_checks(upper + 1, upper + 1, value->type))
    {
        fails.push_back(rocke_b_cmp_ult(b, constant(upper), value));
    }
    else
    {
        if(lower_word)
        {
            rocke_value_t* end = end_of(lower_word);
            fails.push_back(strcmp(lower_word->word, "ge") == 0 ? rocke_b_cmp_lt(b, value, end)
                                                                : rocke_b_cmp_le(b, value, end));
        }
        if(upper_word)
        {
            rocke_value_t* end = end_of(upper_word);
            fails.push_back(strcmp(upper_word->word, "le") == 0 ? rocke_b_cmp_gt(b, value, end)
                                                                : rocke_b_cmp_ge(b, value, end));
        }
    }
    if(facts->multiple_of_set && facts->multiple_of > 1)
    {
        rocke_value_t* remainder = rocke_b_mod(b, value, constant(facts->multiple_of));
        fails.push_back(rocke_b_cmp_ne(b, remainder, constant(0)));
    }
    if(!fails.empty())
    {
        rocke_value_t* fail = fails[0];
        for(size_t i = 1; i < fails.size(); i++)
            fail = rocke_b_lor(b, fail, fails[i]);
        rocke_if_t branch = rocke_b_scf_if(b, fail);
        rocke_b_region_enter(b, branch.then_region);
        rocke_b_trap(b);
        rocke_b_region_leave(b);
    }
    rocke_assume_range_opts_t marker;
    memset(&marker, 0, sizeof marker);
    if(lower_word && lower_word->bound->value_bound)
    {
        marker.ge.end = lower_word->bound->value_bound;
        marker.ge.plus = strcmp(lower_word->word, "gt") == 0 ? 1 : 0;
    }
    else if(has_lower)
    {
        marker.ge.set = true;
        marker.ge.value = lower;
    }
    if(upper_word && upper_word->bound->value_bound)
    {
        marker.lt.end = upper_word->bound->value_bound;
        marker.lt.plus = strcmp(upper_word->word, "le") == 0 ? 1 : 0;
    }
    else if(has_upper && upper < tmax)
    {
        /* An upper end at the type's maximum states nothing (and lt would
         * leave int64 for an i64 value). */
        marker.lt.set = true;
        marker.lt.value = upper + 1;
    }
    marker.multiple_of = facts->multiple_of_set ? facts->multiple_of : 1;
    marker.multiple_of_set = true;
    marker.by = "gpu_checked_value";
    return rocke_b_assume_range(b, value, &marker);
}
