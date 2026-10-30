// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * param_facts.cpp -- the rules a kernel parameter's declaration may state, and
 * how its facts print.
 *
 * Mirrors the "parameter facts" section of python/rocke/core/ir.py and the
 * relation readers of python/rocke/core/intexpr.py: the same rules in the same
 * order, with byte-identical texts, because rocke_b_param raises the first
 * message and both verifiers report all of them.
 *
 * A fact on an integer parameter is one of ge / gt / le / lt / eq /
 * multiple_of. A bound is an int attr, or a relation: an intexpr tree over
 * parameters declared earlier, stored as canonical JSON in a string attr (an
 * int, {"param": NAME}, or {"add"|"sub"|"mul"|"div": [a, b]}).
 */
#include <algorithm>
#include <map>
#include <math.h>
#include <optional>
#include <stdio.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

#include "rocke/arena.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/json_dom.h"
#include "rocke/strbuf.h"

namespace
{

const char* const kFactWords[] = {"ge", "gt", "le", "lt", "eq", "multiple_of"};
const char* const kBoundWords[] = {"ge", "gt", "le", "lt", "eq"};
const char* const kPointerAttrs[] = {"noalias",
                                     "readonly",
                                     "writeonly",
                                     "nocapture",
                                     "nonnull",
                                     "align",
                                     "dereferenceable",
                                     "addr_space"};
const int64_t kExactFactInt = 9007199254740991LL; /* 2^53 - 1 */

/* The range of each integer type a fact may be stated on (ir._INT_TYPE_LIMITS). */
bool int_type_limits(const rocke_type_t* t, int64_t* lo, int64_t* hi)
{
    if(!t || t->kind != ROCKE_TYPE_SCALAR || !t->name)
        return false;
    const std::string n = t->name;
    if(n == "i8")
        *lo = -128, *hi = 127;
    else if(n == "i16")
        *lo = -32768, *hi = 32767;
    else if(n == "i32")
        *lo = INT32_MIN, *hi = INT32_MAX;
    else if(n == "i64")
        *lo = INT64_MIN, *hi = INT64_MAX;
    else if(n == "u8")
        *lo = 0, *hi = 255;
    else
        return false;
    return true;
}

bool is_int_type(const rocke_type_t* t)
{
    int64_t lo;
    int64_t hi;
    return int_type_limits(t, &lo, &hi);
}

std::string type_name(const rocke_type_t* t)
{
    return (t && t->name) ? t->name : "";
}

std::string upper(std::string s)
{
    for(char& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

std::string fmt_int(int64_t v)
{
    char buf[32];
    snprintf(buf, sizeof buf, "%lld", (long long)v);
    return buf;
}

/* ------------------------------------------------------------- relations */

/* A JSON number that is an exact integer within +/-(2^53 - 1). */
bool exact_int(double n, int64_t* out)
{
    if(!isfinite(n) || !(n >= -9007199254740991.0 && n <= 9007199254740991.0))
        return false;
    const int64_t value = (int64_t)n;
    if(n != (double)value)
        return false;
    *out = value;
    return true;
}

bool is_relation_op(const char* key)
{
    return strcmp(key, "add") == 0 || strcmp(key, "sub") == 0 || strcmp(key, "mul") == 0
           || strcmp(key, "div") == 0;
}

/* intexpr.relation_error: why `t` is not a relation, or "" when it is one. */
std::string relation_error(const jd_val_t* t)
{
    const std::string generic
        = "a relation is an int, a param leaf, or add / sub / mul / div of two relations";
    if(!t)
        return generic;
    if(t->kind == JD_NUM)
    {
        int64_t unused;
        return exact_int(t->num, &unused) ? "" : "intexpr number is not an integer";
    }
    if(t->kind == JD_OBJ && t->obj_len == 1)
    {
        const char* key = t->obj[0].key;
        const jd_val_t* value = t->obj[0].val;
        if(strcmp(key, "param") == 0)
            return value && value->kind == JD_STR ? "" : "a param leaf names a parameter";
        if(is_relation_op(key))
        {
            if(!value || value->kind != JD_ARR || value->arr_len != 2)
                return std::string("bad intexpr '") + key + "'";
            const std::string left = relation_error(value->arr[0]);
            return left.empty() ? relation_error(value->arr[1]) : left;
        }
    }
    return generic;
}

/* The relation tree in a stored string, or NULL when it holds none. */
const jd_val_t* stored_relation(const char* text, rocke_arena_t* arena)
{
    char err[128];
    const jd_val_t* tree = text ? rocke_json_parse(text, arena, err, sizeof err) : NULL;
    if(!tree || !relation_error(tree).empty())
        return NULL;
    return tree;
}

const char* leaf_name(const jd_val_t* t)
{
    return t->kind == JD_OBJ && strcmp(t->obj[0].key, "param") == 0 ? t->obj[0].val->str : NULL;
}

const char* function_of(const char* op)
{
    if(strcmp(op, "add") == 0)
        return "sum_of";
    if(strcmp(op, "mul") == 0)
        return "product_of";
    if(strcmp(op, "div") == 0)
        return "quotient_of";
    return "difference_of";
}

/* intexpr.relation_text: a left-nested add or mul chain prints as one call. */
std::string relation_text(const jd_val_t* t, const char* leaf_prefix)
{
    if(t->kind == JD_NUM)
        return fmt_int((int64_t)t->num);
    const char* key = t->obj[0].key;
    const jd_val_t* value = t->obj[0].val;
    if(strcmp(key, "param") == 0)
        return std::string(leaf_prefix) + value->str;
    std::vector<const jd_val_t*> terms{value->arr[1]};
    const jd_val_t* head = value->arr[0];
    const bool chains = strcmp(key, "add") == 0 || strcmp(key, "mul") == 0;
    while(chains && head->kind == JD_OBJ && strcmp(head->obj[0].key, key) == 0)
    {
        terms.push_back(head->obj[0].val->arr[1]);
        head = head->obj[0].val->arr[0];
    }
    terms.push_back(head);
    std::string out = std::string(function_of(key)) + "(";
    for(size_t i = terms.size(); i-- > 0;)
    {
        out += relation_text(terms[i], leaf_prefix);
        if(i)
            out += ", ";
    }
    return out + ")";
}

void relation_names(const jd_val_t* t, std::vector<std::string>* out)
{
    if(t->kind != JD_OBJ)
        return;
    if(const char* name = leaf_name(t))
    {
        if(std::find(out->begin(), out->end(), name) == out->end())
            out->push_back(name);
        return;
    }
    relation_names(t->obj[0].val->arr[0], out);
    relation_names(t->obj[0].val->arr[1], out);
}

void append_json_string(std::string* out, const char* s)
{
    out->push_back('"');
    for(; *s; ++s)
    {
        const unsigned char c = (unsigned char)*s;
        if(c == '"' || c == '\\')
        {
            out->push_back('\\');
            out->push_back((char)c);
        }
        else if(c < 0x20)
        {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", c);
            *out += buf;
        }
        else
            out->push_back((char)c);
    }
    out->push_back('"');
}

/* intexpr.relation_json over a relation, renaming each leaf. */
void relation_json(const jd_val_t* t,
                   const char* (*rename)(void* ctx, const char* name),
                   void* ctx,
                   std::string* out)
{
    if(t->kind == JD_NUM)
    {
        *out += fmt_int((int64_t)t->num);
        return;
    }
    const char* key = t->obj[0].key;
    const jd_val_t* value = t->obj[0].val;
    *out += "{";
    append_json_string(out, key);
    *out += ":";
    if(strcmp(key, "param") == 0)
    {
        append_json_string(out, rename ? rename(ctx, value->str) : value->str);
    }
    else
    {
        *out += "[";
        relation_json(value->arr[0], rename, ctx, out);
        *out += ",";
        relation_json(value->arr[1], rename, ctx, out);
        *out += "]";
    }
    *out += "}";
}

/* --------------------------------------------------------------- intervals */

/* An interval end: unknown (std::nullopt) when it leaves int64 or does not
 * resolve. */
using end_t = std::optional<int64_t>;
using interval_t = std::pair<end_t, end_t>;

end_t checked_add(end_t a, end_t b)
{
    if(!a || !b)
        return std::nullopt;
    if((*b > 0 && *a > INT64_MAX - *b) || (*b < 0 && *a < INT64_MIN - *b))
        return std::nullopt;
    return *a + *b;
}

end_t checked_sub(end_t a, end_t b)
{
    if(!a || !b)
        return std::nullopt;
    if((*b < 0 && *a > INT64_MAX + *b) || (*b > 0 && *a < INT64_MIN + *b))
        return std::nullopt;
    return *a - *b;
}

/* a * b, with the sign of an overflowing result in *overflow (0 when exact). */
int64_t mul_or_overflow(int64_t a, int64_t b, int* overflow)
{
    *overflow = 0;
    if(a == 0 || b == 0)
        return 0;
    const bool negative = (a < 0) != (b < 0);
    const bool fits = negative ? (a > 0 ? b >= INT64_MIN / a : a >= INT64_MIN / b)
                               : (a > 0 ? b <= INT64_MAX / a : b >= INT64_MAX / a);
    if(!fits)
    {
        *overflow = negative ? -1 : 1;
        return 0;
    }
    return a * b;
}

/* Python ir._stated_param_interval's ends: unstated, or a resolved end. */
struct stated_end_t
{
    bool stated;
    end_t value;
};

struct resolver_t
{
    const rocke_param_t* const* params; /* the parameters a relation may name */
    int count;
    rocke_arena_t* arena;
    std::map<std::string, interval_t> memo;

    interval_t relation(const jd_val_t* t);
    interval_t param(const std::string& name);
    std::pair<stated_end_t, stated_end_t> stated(const rocke_param_t* p);
};

interval_t resolver_t::relation(const jd_val_t* t)
{
    if(t->kind == JD_NUM)
        return {(int64_t)t->num, (int64_t)t->num};
    if(const char* name = leaf_name(t))
        return param(name);
    const char* op = t->obj[0].key;
    const interval_t a = relation(t->obj[0].val->arr[0]);
    const interval_t b = relation(t->obj[0].val->arr[1]);
    if(strcmp(op, "add") == 0)
        return {checked_add(a.first, b.first), checked_add(a.second, b.second)};
    if(strcmp(op, "sub") == 0)
        return {checked_sub(a.first, b.second), checked_sub(a.second, b.first)};
    if(strcmp(op, "mul") == 0)
    {
        if(!a.first || !a.second || !b.first || !b.second)
            return {std::nullopt, std::nullopt};
        const int64_t xs[2] = {*a.first, *a.second};
        const int64_t ys[2] = {*b.first, *b.second};
        bool low_overflow = false;
        bool high_overflow = false;
        bool any_exact = false;
        int64_t lo = 0;
        int64_t hi = 0;
        int overflow_signs = 0;
        for(int64_t x : xs)
            for(int64_t y : ys)
            {
                int overflow;
                const int64_t p = mul_or_overflow(x, y, &overflow);
                if(overflow < 0)
                    low_overflow = true;
                else if(overflow > 0)
                    high_overflow = true, ++overflow_signs;
                else if(!any_exact)
                    lo = hi = p, any_exact = true;
                else
                    lo = std::min(lo, p), hi = std::max(hi, p);
            }
        /* The exact minimum is below INT64_MIN when any product overflowed
         * negative, and above INT64_MAX when every product overflowed positive;
         * the maximum likewise. */
        const end_t min_end = (low_overflow || !any_exact) ? end_t{} : end_t{lo};
        const end_t max_end = (high_overflow || !any_exact) ? end_t{} : end_t{hi};
        (void)overflow_signs;
        return {min_end, max_end};
    }
    /* div: floor division, defined for a non-negative dividend and a divisor >= 1. */
    if(!a.first || *a.first < 0 || !b.first || *b.first < 1)
        return {std::nullopt, std::nullopt};
    return {b.second ? end_t{*a.first / *b.second} : end_t{},
            a.second ? end_t{*a.second / *b.first} : end_t{}};
}

std::pair<stated_end_t, stated_end_t> resolver_t::stated(const rocke_param_t* p)
{
    auto bound = [&](const char* word) -> interval_t {
        const rocke_attr_value_t* v = rocke_attr_get(&p->attrs, word);
        if(v->kind == ROCKE_ATTR_INT)
            return {v->u.i, v->u.i};
        const jd_val_t* tree = v->kind == ROCKE_ATTR_STR ? stored_relation(v->u.s, arena) : NULL;
        return tree ? relation(tree) : interval_t{};
    };
    const rocke_attr_map_t* attrs = &p->attrs;
    if(rocke_attr_get(attrs, "eq"))
    {
        const interval_t b = bound("eq");
        return {{true, b.first}, {true, b.second}};
    }
    stated_end_t lower{false, std::nullopt};
    stated_end_t upper{false, std::nullopt};
    if(rocke_attr_get(attrs, "ge"))
        lower = {true, bound("ge").first};
    else if(rocke_attr_get(attrs, "gt"))
        lower = {true, checked_add(bound("gt").first, end_t{1})};
    if(rocke_attr_get(attrs, "le"))
        upper = {true, bound("le").second};
    else if(rocke_attr_get(attrs, "lt"))
        upper = {true, checked_sub(bound("lt").second, end_t{1})};
    return {lower, upper};
}

interval_t resolver_t::param(const std::string& name)
{
    auto found = memo.find(name);
    if(found != memo.end())
        return found->second;
    memo[name] = interval_t{}; /* a cycle (refused elsewhere) stays unknown */
    for(int i = 0; i < count; ++i)
    {
        const rocke_param_t* p = params[i];
        int64_t tmin;
        int64_t tmax;
        if(!p->name || name != p->name || !int_type_limits(p->type, &tmin, &tmax))
            continue;
        const auto ends = stated(p);
        const interval_t result{ends.first.stated ? ends.first.value : end_t{tmin},
                                ends.second.stated ? ends.second.value : end_t{tmax}};
        memo[name] = result;
        return result;
    }
    return memo[name];
}

/* ------------------------------------------------------------- the checker */

struct checker_t
{
    const rocke_param_t* const* params;
    int count;
    int index;
    rocke_arena_t arena;
    std::vector<std::string> errors;

    const rocke_param_t* find(const std::string& name, int* position) const
    {
        for(int i = 0; i < count; ++i)
            if(params[i]->name && name == params[i]->name)
            {
                *position = i;
                return params[i];
            }
        return NULL;
    }

    std::string bound_text(const rocke_attr_value_t* v)
    {
        if(v->kind == ROCKE_ATTR_INT)
            return fmt_int(v->u.i);
        if(v->kind != ROCKE_ATTR_STR)
            return "?"; /* neither an int nor a relation: reported on its own */
        const jd_val_t* tree = stored_relation(v->u.s, &arena);
        return tree ? relation_text(tree, "") : std::string(v->u.s);
    }

    void stated_lower_errors(const std::string& name,
                             const std::string& text,
                             const rocke_param_t* operand,
                             int64_t least,
                             resolver_t* resolver)
    {
        const stated_end_t lower = resolver->stated(operand).first;
        if(lower.stated && lower.value && *lower.value >= least)
            return;
        const std::string q = operand->name;
        std::string declared;
        if(!lower.stated)
            declared = "is declared with no lower bound";
        else
        {
            const char* word = rocke_attr_get(&operand->attrs, "eq")   ? "eq"
                               : rocke_attr_get(&operand->attrs, "ge") ? "ge"
                                                                       : "gt";
            declared = std::string("is declared with ") + word + "="
                       + bound_text(rocke_attr_get(&operand->attrs, word));
        }
        errors.push_back("param " + name + ": " + text + " needs " + q + " ge " + fmt_int(least)
                         + " -- " + q + " " + declared + ".\nFix: b.param(\"" + q + "\", "
                         + upper(type_name(operand->type)) + ", ge=" + fmt_int(least) + ", ...)");
    }

    void quotient_errors(const std::string& name, const jd_val_t* t, resolver_t* resolver)
    {
        if(t->kind != JD_OBJ || leaf_name(t))
            return;
        const jd_val_t* left = t->obj[0].val->arr[0];
        const jd_val_t* right = t->obj[0].val->arr[1];
        if(strcmp(t->obj[0].key, "div") == 0)
        {
            const std::string text = relation_text(t, "");
            const struct
            {
                const jd_val_t* operand;
                int64_t least;
                const char* role;
            } sides[2] = {{right, 1, "divisor"}, {left, 0, "dividend"}};
            for(const auto& side : sides)
            {
                if(const char* q = leaf_name(side.operand))
                {
                    int position;
                    stated_lower_errors(name, text, find(q, &position), side.least, resolver);
                }
                else if(side.operand->kind == JD_NUM)
                {
                    if((int64_t)side.operand->num < side.least)
                        errors.push_back("param " + name + ": " + text + " needs a " + side.role
                                         + " of at least " + fmt_int(side.least) + ", got "
                                         + fmt_int((int64_t)side.operand->num));
                }
                else if(strcmp(side.role, "divisor") == 0)
                {
                    errors.push_back("param " + name + ": " + text
                                     + " needs a divisor that is an int or one kernel "
                                       "parameter, so that its lower bound is stated");
                }
            }
        }
        quotient_errors(name, left, resolver);
        quotient_errors(name, right, resolver);
    }

    /* Each leaf of a relation names an integer parameter declared before `name`. */
    bool name_errors(const std::string& name, const jd_val_t* tree)
    {
        std::vector<std::string> leaves;
        relation_names(tree, &leaves);
        const size_t before = errors.size();
        for(const std::string& q : leaves)
        {
            int position = -1;
            const rocke_param_t* named = find(q, &position);
            if(q == name)
                errors.push_back(name + ": the relation names " + name
                                 + " itself -- a parameter cannot bound itself");
            else if(!named)
                errors.push_back(name + ": the relation names " + q
                                 + ", which is not a parameter of this kernel");
            else if(position > index)
            {
                std::string last;
                int last_position = -1;
                for(const std::string& leaf : leaves)
                {
                    int at;
                    if(find(leaf, &at) && at > last_position)
                        last = leaf, last_position = at;
                }
                errors.push_back(name + ": the relation names " + q
                                 + ", which is declared later -- state it on the last-declared "
                                   "term, "
                                 + last
                                 + " (difference_of and quotient_of move a sum or a product "
                                   "to its side)");
            }
            else if(!is_int_type(named->type))
                errors.push_back(name + ": the relation names " + q + ", which is "
                                 + type_name(named->type) + ", not an integer parameter");
        }
        return errors.size() == before;
    }

    void run();
};

void checker_t::run()
{
    const rocke_param_t* p = params[index];
    const std::string name = p->name ? p->name : "";
    const rocke_attr_map_t* attrs = &p->attrs;
    const std::string tname = type_name(p->type);

    std::vector<std::string> keys;
    for(int i = 0; i < attrs->count; ++i)
        keys.push_back(attrs->entries[i].key);
    std::sort(keys.begin(), keys.end());
    for(const std::string& key : keys)
    {
        bool known = false;
        for(const char* k : kPointerAttrs)
            known = known || key == k;
        for(const char* k : kFactWords)
            known = known || key == k;
        if(!known)
            errors.push_back("param '" + name + "' has unknown attr '" + key
                             + "' -- a param attr is one of " ROCKE_PARAM_ATTR_KEYS);
    }
    const bool is_int = is_int_type(p->type);
    const bool is_ptr = p->type && p->type->kind == ROCKE_TYPE_PTR;
    for(const char* word : kFactWords)
        if(rocke_attr_get(attrs, word) && !is_int)
            errors.push_back("param " + name + ": " + word + " states an integer fact, and " + name
                             + " is " + tname + " -- state it on an integer parameter");
    for(const char* key : kPointerAttrs)
        if(rocke_attr_get(attrs, key) && !is_ptr)
            errors.push_back("param " + name + ": " + key + " applies to a pointer parameter, and "
                             + name + " is " + tname);
    if(!errors.empty())
        return;
    for(const char* word : kBoundWords)
    {
        const rocke_attr_value_t* v = rocke_attr_get(attrs, word);
        if(v && v->kind != ROCKE_ATTR_INT && v->kind != ROCKE_ATTR_STR)
            errors.push_back("param " + name + ": " + word + " must be an integer or a relation");
    }
    for(const char* key : {"multiple_of", "align", "dereferenceable"})
    {
        const rocke_attr_value_t* v = rocke_attr_get(attrs, key);
        if(v && v->kind != ROCKE_ATTR_INT)
            errors.push_back("param " + name + ": " + key + " must be an integer");
    }
    if(!errors.empty())
        return;
    if(rocke_attr_get(attrs, "gt") && rocke_attr_get(attrs, "ge"))
        errors.push_back("param " + name + ": states two lower bounds (gt, ge) -- state one");
    if(rocke_attr_get(attrs, "lt") && rocke_attr_get(attrs, "le"))
        errors.push_back("param " + name + ": states two upper bounds (lt, le) -- state one");
    if(const rocke_attr_value_t* eq = rocke_attr_get(attrs, "eq"))
    {
        std::string others;
        for(const char* word : kFactWords)
            if(strcmp(word, "eq") != 0 && rocke_attr_get(attrs, word))
                others += (others.empty() ? "" : ", ") + std::string(word);
        if(!others.empty())
            errors.push_back("param " + name + ": eq excludes every other word -- eq="
                             + bound_text(eq) + " already decides " + others);
    }
    int64_t value;
    if(rocke_attr_get_int(attrs, "multiple_of", &value) && value < 1)
        errors.push_back("param " + name + ": multiple_of must be >= 1, got " + fmt_int(value));
    if(rocke_attr_get_int(attrs, "align", &value) && (value < 1 || (value & (value - 1))))
        errors.push_back("param " + name + ": align must be a power of two >= 1, got "
                         + fmt_int(value));
    if(rocke_attr_get_int(attrs, "dereferenceable", &value) && value < 1)
        errors.push_back("param " + name + ": dereferenceable must be >= 1, got " + fmt_int(value));
    int64_t tmin;
    int64_t tmax;
    if(int_type_limits(p->type, &tmin, &tmax))
        for(const char* word : kFactWords)
        {
            const rocke_attr_value_t* v = rocke_attr_get(attrs, word);
            if(!v || v->kind != ROCKE_ATTR_INT)
                continue;
            const std::string stated = std::string(word) + "=" + fmt_int(v->u.i);
            if(v->u.i < tmin || v->u.i > tmax)
                errors.push_back("param " + name + ": " + stated + " is outside the range of "
                                 + tname);
            else if(v->u.i > kExactFactInt || v->u.i < -kExactFactInt)
                errors.push_back("param " + name + ": " + stated
                                 + " is outside +/-(2^53 - 1), which a stored fact cannot "
                                   "hold exactly");
        }
    resolver_t resolver{params, index, &arena, {}};
    for(const char* word : kBoundWords)
    {
        const rocke_attr_value_t* v = rocke_attr_get(attrs, word);
        if(!v || v->kind != ROCKE_ATTR_STR)
            continue;
        char err[128];
        const jd_val_t* tree = rocke_json_parse(v->u.s, &arena, err, sizeof err);
        const std::string reason = tree ? relation_error(tree) : "it is not JSON";
        if(!reason.empty())
        {
            errors.push_back("param " + name + ": " + word + " is not a relation: " + reason);
            continue;
        }
        if(name_errors(name, tree))
            quotient_errors(name, tree, &resolver);
    }
}

} // namespace

extern "C" int rocke_param_fact_errors(const rocke_param_t* const* params,
                                       int count,
                                       int index,
                                       rocke_param_fact_sink_t sink,
                                       void* ctx)
{
    checker_t checker{params, count, index, {}, {}};
    if(rocke_arena_init(&checker.arena, 0) != 0)
    {
        if(sink)
            sink(ctx, "param facts: out of memory");
        return 1;
    }
    checker.run();
    rocke_arena_destroy(&checker.arena);
    if(sink)
        for(const std::string& message : checker.errors)
            sink(ctx, message.c_str());
    return (int)checker.errors.size();
}

extern "C" void rocke_param_facts_print(const rocke_param_t* p, rocke_strbuf_t* out)
{
    rocke_arena_t arena;
    if(!p || rocke_arena_init(&arena, 0) != 0)
        return;
    std::string inner;
    for(const char* word : kFactWords)
    {
        const rocke_attr_value_t* v = rocke_attr_get(&p->attrs, word);
        if(!v)
            continue;
        std::string text = "?";
        if(v->kind == ROCKE_ATTR_INT)
            text = fmt_int(v->u.i);
        else if(v->kind == ROCKE_ATTR_STR)
        {
            const jd_val_t* tree = stored_relation(v->u.s, &arena);
            text = tree ? relation_text(tree, "%") : v->u.s;
        }
        inner += (inner.empty() ? "" : ", ") + std::string(word) + " = " + text;
    }
    rocke_arena_destroy(&arena);
    if(!inner.empty())
    {
        rocke_strbuf_append(out, " {");
        rocke_strbuf_append(out, inner.c_str());
        rocke_strbuf_append_char(out, '}');
    }
}

extern "C" void rocke_param_declaration_text(const rocke_param_t* p, rocke_strbuf_t* out)
{
    rocke_arena_t arena;
    if(!p || rocke_arena_init(&arena, 0) != 0)
        return;
    std::string text = "b.param(\"" + std::string(p->name) + "\", " + upper(type_name(p->type));
    for(const char* word : kFactWords)
    {
        const rocke_attr_value_t* v = rocke_attr_get(&p->attrs, word);
        if(!v)
            continue;
        std::string bound = "?";
        if(v->kind == ROCKE_ATTR_INT)
            bound = fmt_int(v->u.i);
        else if(v->kind == ROCKE_ATTR_STR)
        {
            const jd_val_t* tree = stored_relation(v->u.s, &arena);
            bound = tree ? relation_text(tree, "") : v->u.s;
        }
        text += ", " + std::string(word) + "=" + bound;
    }
    rocke_arena_destroy(&arena);
    rocke_strbuf_append(out, (text + ")").c_str());
}

extern "C" void rocke_relation_display(const char* text, rocke_strbuf_t* out)
{
    rocke_arena_t arena;
    if(!text || rocke_arena_init(&arena, 0) != 0)
        return;
    const jd_val_t* tree = stored_relation(text, &arena);
    rocke_strbuf_append(out, tree ? relation_text(tree, "").c_str() : text);
    rocke_arena_destroy(&arena);
}

extern "C" char* rocke_relation_canonical(const char* text,
                                          rocke_arena_t* arena,
                                          const char* (*rename)(void* ctx, const char* name),
                                          void* ctx)
{
    const jd_val_t* tree = stored_relation(text, arena);
    if(!tree)
        return NULL;
    std::string out;
    relation_json(tree, rename, ctx, &out);
    return rocke_arena_strdup(arena, out.c_str());
}
