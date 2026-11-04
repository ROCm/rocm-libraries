// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_if_else_yield_rule.cpp -- the C++ verifier, the if/else rule
 * (rocke_i_if_else_yields) and both lowerings refuse what an scf.if_else
 * cannot hold, with the Python engine's types and texts.
 *
 * Only the scf.yield that ends a branch of an if/else with results has its
 * values taken: a branch of an if/else without results holds no scf.yield, and
 * a branch of one with results holds no other. A result is refused by the
 * packed-only rule (a TypeError) before the LDS rule (a ValueError). The C
 * builder has no point where both branches are done, so it does not check the
 * branches: the verifier and the lowerings do.
 * tests/core/test_scf_if_else_results.py covers the Python engine and both
 * engines' lowering texts; tests/core/test_packed_only.py covers both
 * builders' result-type order.
 */
#include <cstdio>
#include <cstdlib>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"
#include "rocke/verify.h"

namespace
{

int g_failures = 0;

void check(bool ok, const std::string& what)
{
    if(!ok)
    {
        fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

const std::string k_stray_yield_message
    = "scf.if_else has no results, so its then branch cannot contain scf.yield";

/* A refusal: the status and the text. */
struct refusal
{
    rocke_status_t status;
    std::string message;
};

struct Kernel
{
    rocke_ir_builder_t b;
    rocke_value_t* lane = nullptr;

    explicit Kernel(const char* name)
    {
        if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", name);
            std::exit(1);
        }
        lane = rocke_b_thread_id_x(&b);
    }
    ~Kernel()
    {
        rocke_ir_builder_free(&b);
    }

    /* An if/else without results on `lane < bound`; its then branch holds one
     * stray scf.yield when `stray_yield`, and its else branch is empty. */
    rocke_if_else_t build_if_else_without_results(rocke_value_t* bound, bool stray_yield)
    {
        rocke_if_else_t branch = rocke_b_scf_if_else(&b, rocke_b_cmp_lt(&b, lane, bound));
        rocke_b_region_enter(&b, branch.then_region);
        if(stray_yield)
        {
            rocke_value_t* one = rocke_b_const_i32(&b, 1);
            rocke_b_scf_yield(&b, &one, 1);
        }
        rocke_b_region_leave(&b);
        rocke_b_region_enter(&b, branch.else_region);
        rocke_b_region_leave(&b);
        return branch;
    }

    /* An if/else with one i32 result whose branches each yield zero; its
     * then branch first yields one more when `early_yield`. */
    rocke_if_else_t build_if_else_with_a_result(bool early_yield)
    {
        const rocke_type_t* i32 = rocke_i32();
        rocke_if_else_t branch = rocke_b_scf_if_else_results(
            &b, rocke_b_cmp_lt(&b, lane, rocke_b_const_i32(&b, 8)), &i32, 1);
        for(rocke_region_t* region : {branch.then_region, branch.else_region})
        {
            rocke_b_region_enter(&b, region);
            if(early_yield && region == branch.then_region)
            {
                rocke_value_t* one = rocke_b_const_i32(&b, 1);
                rocke_b_scf_yield(&b, &one, 1);
            }
            rocke_value_t* zero = rocke_b_const_i32(&b, 0);
            rocke_b_scf_yield(&b, &zero, 1);
            rocke_b_region_leave(&b);
        }
        return branch;
    }

    /* The verifier's error messages for the kernel built so far. */
    std::string diagnostics()
    {
        rocke_diag_t* diags = nullptr;
        size_t n = 0;
        std::string all;
        if(rocke_verify(rocke_ir_builder_kernel(&b), &diags, &n) != ROCKE_OK)
            return "rocke_verify failed";
        for(size_t i = 0; i < n; ++i)
            all += std::string(diags[i].message) + "\n";
        rocke_diags_free(diags, n);
        return all;
    }

    /* The LLVM lowering's refusal of the kernel. */
    refusal lower_llvm()
    {
        char* text = nullptr;
        char err[ROCKE_ERR_MSG_CAP] = "";
        const rocke_status_t status = rocke_lower_kernel_to_llvm_ex(
            rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_AUTO, "gfx942", &text, err, sizeof err);
        free(text);
        return {status, err};
    }

    /* The HIP lowering's refusal of the kernel. */
    refusal lower_hip()
    {
        rocke_strbuf_t out;
        rocke_lower_hip_opts_t opts{};
        opts.arch = "gfx942";
        rocke_strbuf_init(&out, 0);
        const rocke_status_t status
            = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
        rocke_strbuf_free(&out);
        return {status, rocke_ir_builder_ok(&b) ? "" : rocke_ir_builder_error(&b)};
    }
};

/* The if/else rule refuses `op` with `status` and `message`. */
void check_rule(const rocke_op_t* op,
                rocke_status_t status,
                const std::string& message,
                const std::string& what)
{
    const rocke_op_t* yields[2];
    char err[ROCKE_ERR_MSG_CAP];
    const rocke_status_t got = rocke_i_if_else_yields(op, yields, err, sizeof err);
    check(got == status, what + ": rule status " + std::to_string(got));
    check(err == message, what + ": rule text '" + err + "'");
}

void verifier_accepts_if_elses_with_and_without_results()
{
    Kernel k("if_else_well_formed");
    k.build_if_else_without_results(rocke_b_const_i32(&k.b, 8), false);
    k.build_if_else_with_a_result(false);
    rocke_b_ret(&k.b);
    check(k.diagnostics().empty(), "verifier refused a well-formed if/else: " + k.diagnostics());
}

void verifier_refuses_a_yield_in_an_if_else_without_results()
{
    Kernel k("if_else_stray_yield");
    rocke_if_else_t branch = k.build_if_else_without_results(rocke_b_const_i32(&k.b, 8), true);
    rocke_b_ret(&k.b);
    const std::string diags = k.diagnostics();
    check(diags == k_stray_yield_message + "\n", "verifier, stray yield: " + diags);
    check_rule(branch.op, ROCKE_ERR_VALUE, k_stray_yield_message, "stray yield");
}

/* In a loop body the stray value would count toward the loop's yields; the
 * if/else rule names it instead. */
void verifier_refuses_a_yield_in_an_if_else_without_results_in_a_loop()
{
    Kernel k("if_else_stray_yield_loop");
    rocke_iter_arg_t carry = {"acc", k.lane};
    rocke_for_t loop = rocke_b_scf_for_iter(&k.b,
                                            rocke_b_const_i32(&k.b, 0),
                                            rocke_b_const_i32(&k.b, 4),
                                            rocke_b_const_i32(&k.b, 1),
                                            &carry,
                                            1,
                                            "k",
                                            false,
                                            true);
    rocke_b_region_enter(&k.b, loop.body);
    k.build_if_else_without_results(loop.iv, true);
    rocke_b_scf_yield(&k.b, loop.iter_vars, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_ret(&k.b);
    const std::string diags = k.diagnostics();
    check(diags == k_stray_yield_message + "\n", "verifier, stray yield in a loop: " + diags);
}

void verifier_refuses_a_yield_before_the_end_of_a_branch()
{
    Kernel k("if_else_early_yield");
    rocke_if_else_t branch = k.build_if_else_with_a_result(true);
    rocke_b_ret(&k.b);
    const std::string message = "scf.if_else has 1 results, so its then branch must yield only "
                                "at its end";
    const std::string diags = k.diagnostics();
    check(diags == message + "\n", "verifier, early yield: " + diags);
    check_rule(branch.op, ROCKE_ERR_VALUE, message, "early yield");
}

/* An if/else whose result is an LDS tile of i4, the type swapped in past the
 * builder: the packed-only rule answers first, with a TypeError, in the rule,
 * the verifier and both lowerings. */
void a_packed_only_lds_result_is_refused_as_packed_only()
{
    Kernel k("if_else_packed_only_lds");
    rocke_if_else_t branch = k.build_if_else_with_a_result(false);
    rocke_b_ret(&k.b);
    const int shape[] = {64};
    branch.op->results[0]->type = rocke_smem_type(&k.b, rocke_i4(), shape, 1, 0);

    const std::string message = "scf.if_else: i4 values travel packed in 32-bit words and have "
                                "no scalar or vector form -- keep them in i32 storage and "
                                "unpack explicitly";
    check_rule(branch.op, ROCKE_ERR_TYPE, message, "packed-only LDS result");
    const std::string diags = k.diagnostics();
    check(diags.find(message + "\n") != std::string::npos,
          "verifier, packed-only LDS result: " + diags);
    const refusal llvm = k.lower_llvm();
    check(llvm.status == ROCKE_ERR_TYPE && llvm.message == message,
          "LLVM lowering, packed-only LDS result: " + llvm.message);
    const refusal hip = k.lower_hip();
    check(hip.status == ROCKE_ERR_TYPE && hip.message == message,
          "HIP lowering, packed-only LDS result: " + hip.message);
}

} // namespace

int main()
{
    verifier_accepts_if_elses_with_and_without_results();
    verifier_refuses_a_yield_in_an_if_else_without_results();
    verifier_refuses_a_yield_in_an_if_else_without_results_in_a_loop();
    verifier_refuses_a_yield_before_the_end_of_a_branch();
    a_packed_only_lds_result_is_refused_as_packed_only();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("if/else yield rule: all checks passed\n");
    return 0;
}
