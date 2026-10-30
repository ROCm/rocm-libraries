// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_gpu_checked_value.cpp -- rocke_b_gpu_checked and rocke_b_assume_range,
 * the C++ engine's IRBuilder.gpu_checked_value and IRBuilder._assume_range.
 *
 * Run with no argument it builds every case below and checks that each
 * accepted kernel verifies clean and each refused one is refused.
 *
 * Run with --cases it prints, for every case in order, "== <name>" and then
 * the C++ print_ir of the kernel, or "refused: <message>".
 * tests/core/test_gpu_checked_value.py builds the same cases with the Python
 * builder and must print the same.
 *
 * Run with --verify it reads one serialized kernel (ck.dsl.ir/v1) from stdin
 * and prints the C++ verifier's messages, one per line; the Python test
 * compares them on markers of every form.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_print.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

namespace
{

/* The parameters every case starts from (test_gpu_checked_value.py
 * _declare). */
struct params_t
{
    rocke_value_t* T;
    rocke_value_t* M;
    rocke_value_t* N64;
    rocke_value_t* K;
    rocke_value_t* cu;
};

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

rocke_int_bound_t value_bound(rocke_value_t* value)
{
    return rocke_int_bound_t{false, 0, nullptr, value};
}

rocke_int_facts_t no_facts()
{
    rocke_int_facts_t facts;
    memset(&facts, 0, sizeof facts);
    return facts;
}

rocke_int_facts_t with_multiple_of(rocke_int_facts_t facts, int64_t n)
{
    facts.multiple_of = n;
    facts.multiple_of_set = true;
    return facts;
}

params_t declare(rocke_ir_builder_t* b)
{
    params_t p;
    rocke_param_opts_t t = no_opts();
    t.facts.ge = int_bound(0);
    t.facts.le = int_bound(4096);
    p.T = rocke_b_param(b, "T", rocke_i32(), &t);
    rocke_param_opts_t m = no_opts();
    m.facts.ge = int_bound(1);
    m.facts.le = int_bound(1 << 20);
    m.facts.multiple_of = 64;
    m.facts.multiple_of_set = true;
    p.M = rocke_b_param(b, "M", rocke_i32(), &m);
    rocke_param_opts_t n = no_opts();
    n.facts.ge = int_bound(1);
    p.N64 = rocke_b_param(b, "N64", rocke_i64(), &n);
    p.K = rocke_b_param(b, "K", rocke_i32(), nullptr);
    rocke_param_opts_t cu = no_opts();
    cu.readonly = true;
    cu.readonly_set = true;
    p.cu = rocke_b_param(b, "cu", rocke_ptr_type(b, rocke_i32(), "global"), &cu);
    return p;
}

rocke_value_t* c32(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

/* readfirstlane(global_load(cu, index)): one wave-uniform loaded i32. */
rocke_value_t* loaded(rocke_ir_builder_t* b, const params_t& p, rocke_value_t* index)
{
    return rocke_b_readfirstlane(b, rocke_b_global_load(b, p.cu, index, rocke_i32(), 1));
}

rocke_value_t* checked(rocke_ir_builder_t* b, rocke_value_t* v, rocke_int_facts_t facts)
{
    return rocke_b_gpu_checked(b, v, &facts);
}

rocke_int_facts_t facts_ge_le(rocke_int_bound_t ge, rocke_int_bound_t le)
{
    rocke_int_facts_t f = no_facts();
    f.ge = ge;
    f.le = le;
    return f;
}

rocke_int_facts_t facts_ge_lt(rocke_int_bound_t ge, rocke_int_bound_t lt)
{
    rocke_int_facts_t f = no_facts();
    f.ge = ge;
    f.lt = lt;
    return f;
}

rocke_int_facts_t facts_word(const char* word, rocke_int_bound_t bound)
{
    rocke_int_facts_t f = no_facts();
    if(!strcmp(word, "ge"))
        f.ge = bound;
    else if(!strcmp(word, "gt"))
        f.gt = bound;
    else if(!strcmp(word, "le"))
        f.le = bound;
    else
        f.lt = bound;
    return f;
}

/* A case builds into a fresh builder that already declares params_t. */
struct case_t
{
    const char* name;
    std::function<void(rocke_ir_builder_t*, const params_t&)> build;
};

/* The loop of the binary search shape: lo += (load(cu, k) < T), so the loads
 * reach the result only through a select's condition. */
rocke_value_t* condition_only_loop(rocke_ir_builder_t* b, const params_t& p, bool through_data)
{
    rocke_value_t* lo = c32(b, 0);
    rocke_value_t* hi = c32(b, 8);
    rocke_value_t* step = c32(b, 1);
    rocke_iter_arg_t arg{"lo", c32(b, 0)};
    rocke_for_t loop = rocke_b_scf_for_iter(b, lo, hi, step, &arg, 1, "k0", false, true);
    rocke_b_region_enter(b, loop.body);
    rocke_value_t* x = rocke_b_global_load(b, p.cu, loop.iv, rocke_i32(), 1);
    rocke_value_t* next;
    if(through_data)
    {
        next = rocke_b_add(b, loop.iter_vars[0], x);
    }
    else
    {
        rocke_value_t* below = rocke_b_cmp_lt(b, x, p.T);
        next = rocke_b_select(
            b, below, rocke_b_add(b, loop.iter_vars[0], c32(b, 1)), loop.iter_vars[0]);
    }
    rocke_b_scf_yield(b, &next, 1);
    rocke_b_region_leave(b);
    return loop.op->results[0];
}

const std::vector<case_t>& cases()
{
    static const std::vector<case_t> all = {
        {"varlen_prologue",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_value_t* s = rocke_b_block_id_x(b);
             rocke_value_t* start
                 = checked(b, loaded(b, p, s), facts_ge_le(int_bound(0), value_bound(p.T)));
             rocke_value_t* end
                 = checked(b,
                           loaded(b, p, rocke_b_add(b, s, c32(b, 1))),
                           with_multiple_of(facts_ge_lt(int_bound(0), int_bound(4096)), 4));
             checked(b, rocke_b_sub(b, end, start), facts_ge_le(int_bound(0), value_bound(p.T)));
         }},
        {"every_end_form",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_value_t* s = rocke_b_block_id_x(b);
             rocke_value_t* x = loaded(b, p, s);
             rocke_value_t* y
                 = checked(b, x, with_multiple_of(facts_word("gt", value_bound(p.T)), 8));
             checked(b, loaded(b, p, y), facts_word("lt", value_bound(y)));
             checked(b, loaded(b, p, s), facts_ge_lt(int_bound(-5), int_bound(100)));
             rocke_int_facts_t gt_le = no_facts();
             gt_le.gt = int_bound(-1);
             gt_le.le = int_bound(99);
             checked(b, loaded(b, p, s), gt_le);
             rocke_value_t* wide = rocke_b_sext(b, s, rocke_i64());
             checked(b, wide, facts_ge_le(int_bound(0), value_bound(p.N64)));
             checked(b, wide, facts_ge_le(int_bound(0), int_bound(INT64_MAX)));
             checked(b, wide, facts_ge_lt(int_bound(0), int_bound(1000)));
         }},
        {"condition_only_load",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(
                 b, condition_only_loop(b, p, false), facts_ge_lt(int_bound(0), value_bound(p.T)));
         }},
        {"branch_yields_a_load",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_value_t* s = rocke_b_block_id_x(b);
             const rocke_type_t* types[1] = {rocke_i32()};
             rocke_if_else_t branch
                 = rocke_b_scf_if_else_results(b, rocke_b_cmp_lt(b, s, p.T), types, 1);
             rocke_b_region_enter(b, branch.then_region);
             rocke_value_t* x = loaded(b, p, s);
             rocke_b_scf_yield(b, &x, 1);
             rocke_b_region_leave(b);
             rocke_b_region_enter(b, branch.else_region);
             rocke_value_t* zero = c32(b, 0);
             rocke_b_scf_yield(b, &zero, 1);
             rocke_b_region_leave(b);
             checked(b, branch.op->results[0], facts_word("ge", int_bound(0)));
         }},
        {"param_value_bound",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_param_opts_t ldc = no_opts();
             ldc.facts.ge = value_bound(p.T);
             rocke_b_param(b, "ldc", rocke_i32(), &ldc);
         }},
        {"refuse_param_bound_computed",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_param_opts_t x = no_opts();
             x.facts.ge = value_bound(rocke_b_add(b, p.T, p.T));
             rocke_b_param(b, "X", rocke_i32(), &x);
         }},
        {"refuse_parameter",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, p.M, facts_word("le", int_bound(4096)));
         }},
        {"refuse_parameter_two_facts",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, p.K, with_multiple_of(facts_ge_le(int_bound(0), value_bound(p.T)), 8));
         }},
        {"refuse_computed_from_one_parameter",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, rocke_b_sub(b, p.T, c32(b, 1)), facts_word("le", int_bound(4095)));
         }},
        {"refuse_computed_from_parameters",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, rocke_b_add(b, p.T, p.M), facts_word("le", int_bound(4095)));
         }},
        {"refuse_thread_id",
         [](rocke_ir_builder_t* b, const params_t&) {
             checked(b, rocke_b_thread_id_x(b), facts_word("lt", int_bound(64)));
         }},
        {"refuse_thread_id_and_parameter",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(
                 b, rocke_b_add(b, rocke_b_thread_id_x(b), p.T), facts_word("lt", int_bound(64)));
         }},
        {"refuse_constants",
         [](rocke_ir_builder_t* b, const params_t&) {
             rocke_value_t* two = c32(b, 2);
             checked(b, rocke_b_add(b, two, c32(b, 3)), facts_word("lt", int_bound(8)));
         }},
        {"refuse_subtraction_on_a_load",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_value_t* raw = loaded(b, p, rocke_b_block_id_x(b));
             rocke_value_t* start = checked(b, raw, facts_ge_le(int_bound(0), value_bound(p.T)));
             checked(b, rocke_b_sub(b, raw, start), facts_word("ge", int_bound(0)));
         }},
        {"refuse_loop_sum_of_loads",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, condition_only_loop(b, p, true), facts_word("ge", int_bound(0)));
         }},
        {"refuse_select_of_a_load",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_value_t* s = rocke_b_block_id_x(b);
             /* One emitting call per statement: C++ leaves argument order open. */
             rocke_value_t* below = rocke_b_cmp_lt(b, s, p.T);
             rocke_value_t* x = loaded(b, p, s);
             rocke_value_t* picked = rocke_b_select(b, below, x, c32(b, 0));
             checked(b, rocke_b_mul(b, picked, c32(b, 4)), facts_word("ge", int_bound(0)));
         }},
        {"refuse_no_fact",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, loaded(b, p, rocke_b_block_id_x(b)), no_facts());
         }},
        {"refuse_two_lower_bounds",
         [](rocke_ir_builder_t* b, const params_t& p) {
             rocke_int_facts_t f = no_facts();
             f.ge = int_bound(0);
             f.gt = int_bound(-1);
             checked(b, loaded(b, p, rocke_b_block_id_x(b)), f);
         }},
        {"refuse_empty_range",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(
                 b, loaded(b, p, rocke_b_block_id_x(b)), facts_ge_lt(int_bound(5), int_bound(5)));
         }},
        {"refuse_gt_at_the_maximum",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(
                 b, loaded(b, p, rocke_b_block_id_x(b)), facts_word("gt", int_bound(INT32_MAX)));
         }},
        {"refuse_bound_without_facts",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, loaded(b, p, rocke_b_block_id_x(b)), facts_word("le", value_bound(p.K)));
         }},
        {"refuse_bound_of_another_type",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, loaded(b, p, rocke_b_block_id_x(b)), facts_word("le", value_bound(p.N64)));
         }},
        {"refuse_multiple_of_zero",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(b, loaded(b, p, rocke_b_block_id_x(b)), with_multiple_of(no_facts(), 0));
         }},
        {"refuse_bound_outside_the_type",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(
                 b, loaded(b, p, rocke_b_block_id_x(b)), facts_word("le", int_bound(1LL << 31)));
         }},
        {"refuse_an_i1_value",
         [](rocke_ir_builder_t* b, const params_t& p) {
             checked(
                 b, rocke_b_cmp_lt(b, rocke_b_block_id_x(b), p.T), facts_word("ge", int_bound(0)));
         }},
    };
    return all;
}

/* One case's report: the kernel's print_ir, or "refused: <message>". */
std::string run_case(const case_t& c, bool* refused, std::string* verify_errors)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, c.name);
    std::string out;
    *refused = false;
    try
    {
        const params_t p = declare(&b);
        c.build(&b, p);
        rocke_b_ret(&b);
        rocke_kernel_def_t* k = rocke_ir_builder_kernel(&b);
        char* printed = rocke_print_ir_alloc(k);
        out = printed ? printed : "";
        free(printed);
        rocke_diag_t* diags = nullptr;
        size_t count = 0;
        rocke_verify(k, &diags, &count);
        for(size_t i = 0; i < count; ++i)
            *verify_errors += std::string(diags[i].message) + "\n";
        rocke_diags_free(diags, count);
    }
    catch(const ckc::Error& error)
    {
        *refused = true;
        out = std::string("refused: ") + error.what();
    }
    rocke_ir_builder_free(&b);
    return out;
}

int print_cases()
{
    std::string out;
    for(const case_t& c : cases())
    {
        bool refused = false;
        std::string errors;
        out += std::string("== ") + c.name + "\n" + run_case(c, &refused, &errors) + "\n";
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}

int print_verifier_messages()
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
    rocke_diag_t* diags = nullptr;
    size_t count = 0;
    rocke_verify(k, &diags, &count);
    std::string out;
    for(size_t i = 0; i < count; ++i)
        out += std::string(diags[i].message) + "\n";
    rocke_diags_free(diags, count);
    rocke_ir_builder_free(&b);
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc > 1 && strcmp(argv[1], "--cases") == 0)
        return print_cases();
    if(argc > 1 && strcmp(argv[1], "--verify") == 0)
        return print_verifier_messages();
    int failures = 0;
    for(const case_t& c : cases())
    {
        bool refused = false;
        std::string errors;
        const std::string report = run_case(c, &refused, &errors);
        const bool want_refused = strncmp(c.name, "refuse_", 7) == 0;
        if(refused != want_refused || !errors.empty())
        {
            fprintf(stderr, "FAIL %s: %s\n%s", c.name, report.c_str(), errors.c_str());
            ++failures;
        }
    }
    if(failures)
    {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test_gpu_checked_value: OK\n");
    return 0;
}
