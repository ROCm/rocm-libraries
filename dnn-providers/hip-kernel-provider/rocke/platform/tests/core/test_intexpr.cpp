// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_intexpr.cpp -- the C++ integer-expression evaluator (rocke/intexpr.h).
 *
 * Run with no argument it checks a few cases through the C API: each leaf kind
 * resolved through the lookup, an unbound leaf, and the refusals (zero divisor,
 * negative operand, int64 overflow, inexact number) with their exact texts.
 *
 * Run with --eval it reads one JSON document from stdin,
 *   {"env": {"spec": {...}, "var": {...}, "param": {...}, "spec_str": {...}},
 *    "cases": [<intexpr>, ...]}
 * and prints one line per case: "value <n>" or "error <text>". The Python test
 * tests/core/test_intexpr.py feeds it the same table it evaluates with
 * rocke.core.intexpr, so the two mirrors are pinned against each other.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/arena.h"
#include "rocke/intexpr.h"
#include "rocke/json_dom.h"

namespace
{

int g_failures = 0;

void fail(const std::string& what, int line)
{
    fprintf(stderr, "FAIL (line %d): %s\n", line, what.c_str());
    ++g_failures;
}

#define EXPECT(cond, what)          \
    do                              \
    {                               \
        if(!(cond))                 \
            fail((what), __LINE__); \
    } while(0)

/* Leaf bindings read from a JSON env object; numbers must be exact integers. */
struct json_env_t
{
    const jd_val_t* spec;
    const jd_val_t* var;
    const jd_val_t* param;
    const jd_val_t* spec_str;
};

bool lookup_int(void* ctx, rocke_intexpr_leaf_t kind, const char* name, int64_t* out)
{
    const json_env_t* env = (const json_env_t*)ctx;
    const jd_val_t* table = kind == ROCKE_INTEXPR_SPEC  ? env->spec
                            : kind == ROCKE_INTEXPR_VAR ? env->var
                                                        : env->param;
    double d = 0.0;
    if(!rocke_jnum(rocke_jget(table, name), &d))
        return false;
    *out = (int64_t)d;
    return true;
}

const char* lookup_str(void* ctx, const char* name)
{
    return rocke_jstr(rocke_jget(((const json_env_t*)ctx)->spec_str, name));
}

/* "value <n>" or "error <text>": the line format both engines are compared on. */
std::string outcome(const jd_val_t* expr, const json_env_t* env)
{
    const rocke_intexpr_env_t lookup = {lookup_int, lookup_str, (void*)env};
    char err[256];
    int64_t value = 0;
    if(!rocke_intexpr_eval(expr, &lookup, &value, err, sizeof err))
        return std::string("error ") + err;
    return "value " + std::to_string((long long)value);
}

std::string outcome_of(const char* expr_json)
{
    rocke_arena_t arena;
    rocke_arena_init(&arena, 0);
    char err[256];
    const char* env_json = "{\"spec\": {\"D\": 6}, \"var\": {\"i\": 2}, \"param\": {\"M\": 64}, "
                           "\"spec_str\": {\"dtype\": \"f16\"}}";
    const jd_val_t* env_dom = rocke_json_parse(env_json, &arena, err, sizeof err);
    const jd_val_t* expr = rocke_json_parse(expr_json, &arena, err, sizeof err);
    json_env_t env = {rocke_jget(env_dom, "spec"),
                      rocke_jget(env_dom, "var"),
                      rocke_jget(env_dom, "param"),
                      rocke_jget(env_dom, "spec_str")};
    std::string out = expr ? outcome(expr, &env) : std::string("unparsed");
    rocke_arena_destroy(&arena);
    return out;
}

void expect_outcome(const char* expr_json, const char* want, int line)
{
    const std::string got = outcome_of(expr_json);
    if(got != want)
        fail(std::string(expr_json) + ": got '" + got + "', want '" + want + "'", line);
}

void case_leaves_and_refusals()
{
    expect_outcome("{\"mul\": [{\"spec\": \"D\"}, {\"var\": \"i\"}]}", "value 12", __LINE__);
    expect_outcome("{\"param\": \"M\"}", "value 64", __LINE__);
    expect_outcome("{\"param\": \"N\"}", "error unknown param 'N'", __LINE__);
    expect_outcome("{\"spec_str_eq\": [\"dtype\", \"f16\"]}", "value 1", __LINE__);
    expect_outcome("{\"div\": [{\"spec\": \"D\"}, 0]}", "error intexpr div by zero", __LINE__);
    expect_outcome("{\"mod\": [-7, 2]}",
                   "error intexpr mod needs non-negative operands, got (-7, 2)",
                   __LINE__);
    expect_outcome("{\"mul\": [{\"mul\": [2147483648, 2147483648]}, 2]}",
                   "error intexpr mul overflows int64 (4611686018427387904, 2)",
                   __LINE__);
    expect_outcome("2.5", "error intexpr number is not an integer", __LINE__);
}

void case_env_may_bind_nothing()
{
    /* A null lookup binds no leaf; the evaluator must report, not crash. */
    rocke_arena_t arena;
    rocke_arena_init(&arena, 0);
    char err[256];
    const jd_val_t* expr = rocke_json_parse("{\"spec\": \"D\"}", &arena, err, sizeof err);
    int64_t value = 7;
    EXPECT(!rocke_intexpr_eval(expr, nullptr, &value, err, sizeof err), "null env evaluates");
    EXPECT(strcmp(err, "unknown spec int 'D'") == 0, std::string("null env text: ") + err);
    EXPECT(value == 7, "a failed evaluation wrote *out");
    rocke_arena_destroy(&arena);
}

int eval_stdin()
{
    std::string text;
    char buf[4096];
    size_t n;
    while((n = fread(buf, 1, sizeof buf, stdin)) > 0)
        text.append(buf, n);
    rocke_arena_t arena;
    rocke_arena_init(&arena, 0);
    char err[256];
    const jd_val_t* doc = rocke_json_parse(text.c_str(), &arena, err, sizeof err);
    const jd_val_t* cases = rocke_jget(doc, "cases");
    if(!cases || cases->kind != JD_ARR)
    {
        fprintf(stderr, "--eval needs {\"env\": ..., \"cases\": [...]} on stdin\n");
        rocke_arena_destroy(&arena);
        return 2;
    }
    const jd_val_t* env_dom = rocke_jget(doc, "env");
    json_env_t env = {rocke_jget(env_dom, "spec"),
                      rocke_jget(env_dom, "var"),
                      rocke_jget(env_dom, "param"),
                      rocke_jget(env_dom, "spec_str")};
    for(int i = 0; i < cases->arr_len; i++)
        printf("%s\n", outcome(cases->arr[i], &env).c_str());
    rocke_arena_destroy(&arena);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc > 1 && strcmp(argv[1], "--eval") == 0)
        return eval_stdin();
    case_leaves_and_refusals();
    case_env_may_bind_nothing();
    if(g_failures)
    {
        fprintf(stderr, "%d intexpr check(s) failed\n", g_failures);
        return 1;
    }
    return 0;
}
