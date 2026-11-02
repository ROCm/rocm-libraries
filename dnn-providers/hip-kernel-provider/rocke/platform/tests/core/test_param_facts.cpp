// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_param_facts.cpp -- facts on a kernel parameter's declaration in the C++
 * engine (rocke_param_opts_t.facts, cpp/core/ir/param_facts.cpp).
 *
 * Run with no argument it checks, through the C API: what rocke_b_param stores
 * (ints, and a relation as canonical JSON whatever its spacing), its refusals
 * with the Python builder's texts (tests/core/test_param_facts.py REFUSALS),
 * and the C++ print_ir of a kernel with facts.
 *
 * Run with --ir it reads one serialized kernel (ck.dsl.ir/v1) from stdin and
 * prints the C++ print_ir text, "-- verify --", one line per verifier message,
 * "-- canonical --" and the canonicalized text. test_param_facts.py prints the
 * same report with the Python engine for every kernel in its IR_CASES and
 * compares the two.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_print.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

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

rocke_param_opts_t no_opts()
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof opts);
    return opts;
}

rocke_int_bound_t int_bound(int64_t value)
{
    return rocke_int_bound_t{true, value, nullptr, nullptr};
}

rocke_int_bound_t relation(const char* json)
{
    return rocke_int_bound_t{true, 0, json, nullptr};
}

/* The parameters every refusal case starts from: M with a lower bound and sb
 * with none (test_param_facts.py _three). */
void declare_m_and_sb(rocke_ir_builder_t* b)
{
    rocke_param_opts_t m = no_opts();
    m.facts.ge = int_bound(1);
    rocke_b_param(b, "M", rocke_i32(), &m);
    rocke_b_param(b, "sb", rocke_i32(), nullptr);
}

std::string attr_text(const rocke_param_t* p, const char* key)
{
    const rocke_attr_value_t* v = rocke_attr_get(&p->attrs, key);
    if(!v)
        return "<absent>";
    if(v->kind == ROCKE_ATTR_INT)
        return std::to_string(v->u.i);
    if(v->kind == ROCKE_ATTR_STR)
        return v->u.s;
    return "<other>";
}

void case_stores_ints_and_canonical_relations()
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "stored");
    rocke_param_opts_t m = no_opts();
    m.facts.ge = int_bound(1);
    m.facts.le = int_bound(1 << 20);
    m.facts.multiple_of = 64;
    m.facts.multiple_of_set = true;
    rocke_b_param(&b, "M", rocke_i32(), &m);
    rocke_param_opts_t ldc = no_opts();
    ldc.facts.ge = relation("{ \"param\": \"M\" }");
    rocke_b_param(&b, "ldc", rocke_i32(), &ldc);
    rocke_param_opts_t sb = no_opts();
    sb.facts.ge = relation("{\"mul\": [{\"param\": \"M\"}, {\"param\": \"ldc\"}]}");
    rocke_b_param(&b, "sb", rocke_i32(), &sb);
    EXPECT(rocke_ir_builder_status(&b) == ROCKE_OK, rocke_ir_builder_error(&b));
    const rocke_kernel_def_t* k = rocke_ir_builder_kernel(&b);
    EXPECT(attr_text(k->params[0], "ge") == "1", "M ge");
    EXPECT(attr_text(k->params[0], "le") == "1048576", "M le");
    EXPECT(attr_text(k->params[0], "multiple_of") == "64", "M multiple_of");
    EXPECT(attr_text(k->params[1], "ge") == "{\"param\":\"M\"}",
           "ldc ge: " + attr_text(k->params[1], "ge"));
    EXPECT(attr_text(k->params[2], "ge") == "{\"mul\":[{\"param\":\"M\"},{\"param\":\"ldc\"}]}",
           "sb ge: " + attr_text(k->params[2], "ge"));
    char* printed = rocke_print_ir_alloc(k);
    const std::string want = "kernel @stored(%M: i32 {ge = 1, le = 1048576, multiple_of = 64}, "
                             "%ldc: i32 {ge = %M}, %sb: i32 {ge = product_of(%M, %ldc)}) {\n}";
    EXPECT(printed && want == printed, std::string("print: ") + (printed ? printed : "NULL"));
    free(printed);
    rocke_ir_builder_free(&b);
}

/* One declaration rocke_b_param must refuse with exactly `want`. */
void expect_refused(const char* name,
                    const rocke_type_t* (*type)(rocke_ir_builder_t*),
                    const std::function<void(rocke_param_opts_t*)>& set,
                    const char* want)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "refused");
    declare_m_and_sb(&b);
    rocke_param_opts_t opts = no_opts();
    set(&opts);
    const rocke_type_t* t = type(&b);
    /* The builder raises (ckc::Error) the way the Python builder does; a public
     * entry point turns that into the sticky status and message. */
    std::string got = "<not refused>";
    try
    {
        if(rocke_b_param(&b, name, t, &opts) == nullptr
           && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE)
            got = rocke_ir_builder_error(&b);
    }
    catch(const ckc::Error& error)
    {
        if(error.code() == ROCKE_ERR_VALUE)
            got = error.what();
    }
    EXPECT(got == want, std::string("want '") + want + "', got '" + got + "'");
    EXPECT(rocke_ir_builder_kernel(&b)->num_params == 2, "a refused param was added");
    rocke_ir_builder_free(&b);
}

const rocke_type_t* i32_type(rocke_ir_builder_t*)
{
    return rocke_i32();
}

const rocke_type_t* i64_type(rocke_ir_builder_t*)
{
    return rocke_i64();
}

const rocke_type_t* f16_ptr_type(rocke_ir_builder_t* b)
{
    return rocke_ptr_type(b, rocke_f16(), "global");
}

void case_refusals_match_the_python_builder()
{
    expect_refused(
        "ldc",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.lt = int_bound(4096);
            o->facts.le = int_bound(4095);
        },
        "param ldc: states two upper bounds (lt, le) -- state one");
    expect_refused(
        "lo",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.gt = int_bound(0);
            o->facts.ge = int_bound(1);
        },
        "param lo: states two lower bounds (gt, ge) -- state one");
    expect_refused(
        "D",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.eq = int_bound(128);
            o->facts.multiple_of = 8;
            o->facts.multiple_of_set = true;
        },
        "param D: eq excludes every other word -- eq=128 already decides multiple_of");
    expect_refused(
        "D",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.eq = relation("{\"param\":\"M\"}");
            o->facts.ge = int_bound(0);
            o->facts.le = int_bound(9);
        },
        "param D: eq excludes every other word -- eq=M already decides ge, le");
    expect_refused(
        "x",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.le = relation("{\"div\":[{\"param\":\"M\"},{\"param\":\"sb\"}]}");
        },
        "param x: quotient_of(M, sb) needs sb ge 1 -- sb is declared with no lower bound.\n"
        "Fix: b.param(\"sb\", I32, ge=1, ...)");
    expect_refused(
        "x",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.le = relation("{\"div\":[{\"param\":\"sb\"},{\"param\":\"M\"}]}");
        },
        "param x: quotient_of(sb, M) needs sb ge 0 -- sb is declared with no lower bound.\n"
        "Fix: b.param(\"sb\", I32, ge=0, ...)");
    expect_refused(
        "x",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.le
                = relation("{\"div\":[4096,{\"add\":[{\"param\":\"M\"},{\"param\":\"M\"}]}]}");
        },
        "param x: quotient_of(4096, sum_of(M, M)) needs a divisor that is an int or one "
        "kernel parameter, so that its lower bound is stated");
    expect_refused(
        "x",
        i32_type,
        [](rocke_param_opts_t* o) { o->facts.le = relation("{\"div\":[{\"param\":\"M\"},0]}"); },
        "param x: quotient_of(M, 0) needs a divisor of at least 1, got 0");
    expect_refused(
        "C",
        f16_ptr_type,
        [](rocke_param_opts_t* o) {
            o->facts.multiple_of = 4;
            o->facts.multiple_of_set = true;
        },
        "param C: multiple_of states an integer fact, and C is ptr<f16,global> -- state it on "
        "an integer parameter");
    expect_refused(
        "n",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->align = 16;
            o->align_set = true;
        },
        "param n: align applies to a pointer parameter, and n is i32");
    expect_refused(
        "C",
        f16_ptr_type,
        [](rocke_param_opts_t* o) {
            o->align = 12;
            o->align_set = true;
        },
        "param C: align must be a power of two >= 1, got 12");
    expect_refused(
        "n",
        i32_type,
        [](rocke_param_opts_t* o) {
            o->facts.multiple_of = 0;
            o->facts.multiple_of_set = true;
        },
        "param n: multiple_of must be >= 1, got 0");
    expect_refused(
        "n",
        i32_type,
        [](rocke_param_opts_t* o) { o->facts.le = int_bound(int64_t(1) << 40); },
        "param n: le=1099511627776 is outside the range of i32");
    expect_refused(
        "n",
        i64_type,
        [](rocke_param_opts_t* o) { o->facts.le = int_bound(int64_t(1) << 60); },
        "param n: le=1152921504606846976 is outside +/-(2^53 - 1), which a stored fact "
        "cannot hold exactly");
    /* The C++ builder names a relation's terms, so it can name one that does not
     * exist yet; the Python builder cannot, because the Value does not. */
    expect_refused(
        "x",
        i32_type,
        [](rocke_param_opts_t* o) { o->facts.ge = relation("{\"param\":\"K\"}"); },
        "x: the relation names K, which is not a parameter of this kernel");
    expect_refused(
        "x",
        i32_type,
        [](rocke_param_opts_t* o) { o->facts.ge = relation("{\"pow\":[2,3]}"); },
        "param x: ge is not a relation: a relation is an int, a param leaf, or add / sub / mul "
        "/ div of two relations");
}

/* --ir: the report test_param_facts.py compares with the Python engine's. */
int report_serialized_ir()
{
    const std::string text{std::istreambuf_iterator<char>(std::cin),
                           std::istreambuf_iterator<char>()};
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "parsed");
    rocke_kernel_def_t* k = nullptr;
    if(rocke_ir_parse(text.c_str(), &b, &k) != ROCKE_OK || !k)
    {
        fprintf(stderr, "parse failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 2;
    }
    char* printed = rocke_print_ir_alloc(k);
    std::string out = printed ? printed : "";
    free(printed);
    out += "\n-- verify --";
    rocke_diag_t* diags = nullptr;
    size_t count = 0;
    rocke_verify(k, &diags, &count);
    for(size_t i = 0; i < count; ++i)
        out += std::string("\n") + diags[i].message;
    rocke_diags_free(diags, count);
    char* canonical = nullptr;
    rocke_ir_canonicalize(text.c_str(), &canonical);
    out += "\n-- canonical --\n";
    out += canonical ? canonical : "";
    free(canonical);
    rocke_ir_builder_free(&b);
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc > 1 && strcmp(argv[1], "--ir") == 0)
        return report_serialized_ir();
    case_stores_ints_and_canonical_relations();
    case_refusals_match_the_python_builder();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("test_param_facts: OK\n");
    return 0;
}
