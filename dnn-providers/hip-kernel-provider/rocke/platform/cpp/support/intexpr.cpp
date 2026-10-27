// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * The integer-expression evaluator (see rocke/intexpr.h). Mirrors
 * python/rocke/core/intexpr.py: node kinds are tried in the same order and
 * every error text is byte-identical, because callers compare refusal reasons
 * across the two engines.
 */
#include "rocke/intexpr.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace
{

struct ie_state_t
{
    const rocke_intexpr_env_t* env;
    char* err;
    size_t err_cap;
    bool failed;
};

void ie_fail(ie_state_t* st, const char* fmt, ...)
{
    if(st->failed)
        return;
    st->failed = true;
    if(!st->err || !st->err_cap)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(st->err, st->err_cap, fmt, ap);
    va_end(ap);
}

/* The JSON DOM holds numbers as doubles, so a spelling beyond 2^53 may already
 * have been rounded; refuse it rather than trust it. Checked before the cast,
 * which is undefined for a non-finite or out-of-range double. */
bool ie_number(double n, int64_t* out)
{
    if(!isfinite(n) || !(n >= -9007199254740991.0 && n <= 9007199254740991.0))
        return false;
    int64_t value = (int64_t)n;
    if(n != (double)value)
        return false;
    *out = value;
    return true;
}

/* Overflow is checked before each operation: signed overflow is undefined in
 * C++, and C++20 has no standard checked-arithmetic function. */
bool ie_add_overflows(int64_t a, int64_t b)
{
    return (b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b);
}

bool ie_sub_overflows(int64_t a, int64_t b)
{
    return (b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b);
}

bool ie_mul_overflows(int64_t a, int64_t b)
{
    if(a == 0 || b == 0)
        return false;
    if(a > 0)
        return b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a;
    return b > 0 ? a < INT64_MIN / b : b < INT64_MAX / a;
}

int64_t ie_eval(ie_state_t* st, const jd_val_t* e);

int64_t ie_binary(ie_state_t* st, const char* op, int64_t a, int64_t b)
{
    const long long la = (long long)a;
    const long long lb = (long long)b;
    if(strcmp(op, "div") == 0 || strcmp(op, "mod") == 0)
    {
        if(b == 0)
        {
            ie_fail(st, "intexpr %s by zero", op);
            return 0;
        }
        /* Python floors and C truncates; for non-negative operands they agree.
         * This also rules out INT64_MIN / -1. */
        if(a < 0 || b < 0)
        {
            ie_fail(st, "intexpr %s needs non-negative operands, got (%lld, %lld)", op, la, lb);
            return 0;
        }
        return op[0] == 'd' ? a / b : a % b;
    }
    if(strcmp(op, "add") == 0 || strcmp(op, "sub") == 0 || strcmp(op, "mul") == 0)
    {
        const bool overflows = op[0] == 'a'   ? ie_add_overflows(a, b)
                               : op[0] == 's' ? ie_sub_overflows(a, b)
                                              : ie_mul_overflows(a, b);
        if(overflows)
        {
            ie_fail(st, "intexpr %s overflows int64 (%lld, %lld)", op, la, lb);
            return 0;
        }
        return op[0] == 'a' ? a + b : op[0] == 's' ? a - b : a * b;
    }
    if(strcmp(op, "eq") == 0)
        return a == b;
    if(strcmp(op, "ne") == 0)
        return a != b;
    if(strcmp(op, "lt") == 0)
        return a < b;
    if(strcmp(op, "le") == 0)
        return a <= b;
    if(strcmp(op, "gt") == 0)
        return a > b;
    return a >= b; /* "ge" */
}

/* magic_multiplier / magic_shift: the two operands of a strength-reduced
 * unsigned division, emitted as (umul_hi(n, M) + n) >> s. They are constants a
 * code generator derived from a spec value, not arithmetic on it: the shift is
 * logarithmic in the divisor and the multiplier depends on its odd part, so a
 * recipe has to regenerate them. Mirrors intexpr.py::magic_division_constants
 * and, upstream of both, helpers/magic_division.py::calculate_magic_numbers. */
int64_t ie_magic(ie_state_t* st, bool want_shift, int64_t d)
{
    if(d < 1 || d > 0x7fffffffLL)
    {
        ie_fail(st, "magic division needs 1 <= divisor < 2^31, got %lld", (long long)d);
        return 0;
    }
    int shift = 0;
    while((1LL << shift) < d)
        shift++;
    if(want_shift)
        return shift;
    long long mult = ((((1LL << shift) - d) << 32) / d) + 1;
    if(mult >= (1LL << 31))
        mult -= (1LL << 32);
    return mult;
}

int64_t ie_eval(ie_state_t* st, const jd_val_t* e)
{
    if(st->failed)
        return 0;
    double n;
    if(rocke_jnum(e, &n))
    {
        int64_t value;
        if(!ie_number(n, &value))
        {
            ie_fail(st, "intexpr number is not an integer");
            return 0;
        }
        return value;
    }
    if(e && e->kind == JD_OBJ)
    {
        static const struct
        {
            const char* key;
            rocke_intexpr_leaf_t kind;
            const char* noun;
        } leaves[] = {{"spec", ROCKE_INTEXPR_SPEC, "spec int"},
                      {"var", ROCKE_INTEXPR_VAR, "loop var"},
                      {"param", ROCKE_INTEXPR_PARAM, "param"}};
        for(const auto& leaf : leaves)
        {
            const jd_val_t* name_node = rocke_jget(e, leaf.key);
            if(!name_node)
                continue;
            const char* name = rocke_jstr(name_node);
            int64_t v = 0;
            if(!name || !st->env || !st->env->lookup_int
               || !st->env->lookup_int(st->env->ctx, leaf.kind, name, &v))
            {
                ie_fail(st, "unknown %s '%s'", leaf.noun, name ? name : "?");
                return 0;
            }
            return v;
        }
        /* spec_str_eq: ["specname", "literal"] -> 1 if the spec string matches. */
        const jd_val_t* sse = rocke_jget(e, "spec_str_eq");
        if(sse && sse->kind == JD_ARR && sse->arr_len == 2)
        {
            const char* name = rocke_jstr(sse->arr[0]);
            const char* sv = (name && st->env && st->env->lookup_str)
                                 ? st->env->lookup_str(st->env->ctx, name)
                                 : nullptr;
            const char* lit = rocke_jstr(sse->arr[1]);
            return (sv && lit && strcmp(sv, lit) == 0) ? 1 : 0;
        }
        /* Unary functions take the operand directly, not a 2-element array. */
        const jd_val_t* mm = rocke_jget(e, "magic_multiplier");
        const jd_val_t* ms = rocke_jget(e, "magic_shift");
        if(mm || ms)
        {
            int64_t d = ie_eval(st, mm ? mm : ms);
            return st->failed ? 0 : ie_magic(st, !mm, d);
        }
        static const char* const ops[]
            = {"add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "gt", "ge"};
        for(const char* op : ops)
        {
            const jd_val_t* pair = rocke_jget(e, op);
            if(!pair)
                continue;
            if(pair->kind != JD_ARR || pair->arr_len != 2)
            {
                ie_fail(st, "bad intexpr '%s'", op);
                return 0;
            }
            int64_t a = ie_eval(st, pair->arr[0]);
            int64_t b = ie_eval(st, pair->arr[1]);
            return st->failed ? 0 : ie_binary(st, op, a, b);
        }
    }
    ie_fail(st, "bad intexpr");
    return 0;
}

} // namespace

extern "C" bool rocke_intexpr_eval(
    const jd_val_t* expr, const rocke_intexpr_env_t* env, int64_t* out, char* err, size_t err_cap)
{
    ie_state_t st = {env, err, err_cap, false};
    int64_t value = ie_eval(&st, expr);
    if(st.failed)
        return false;
    if(out)
        *out = value;
    return true;
}
