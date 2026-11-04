// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_if_else_yield_rule.cpp -- the C++ verifier, the if/else rule
 * (rocke_i_if_else_yields) and both lowerings refuse what an scf.if_else
 * cannot hold, and an scf.yield where nothing takes it, with the Python
 * engine's types and texts.
 *
 * Only the scf.yield that ends a branch of an if/else with results has its
 * values taken: a branch of an if/else without results holds no scf.yield, and
 * a branch of one with results holds no other, directly or in an scf.if within
 * it. A yield goes directly in an scf.for body or such a branch; one anywhere
 * else (the kernel body, an scf.if) is an orphan. A result is refused by the
 * packed-only rule (a TypeError) before the LDS rule (a ValueError). The C
 * builder has no point where both branches are done, so it does not check the
 * branches: the verifier and the lowerings do, the LLVM lowering also in the
 * body of an unrolled loop of zero trips, which it does not lower.
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
const std::string k_early_yield_message
    = "scf.if_else has 1 results, so its then branch must yield only at its end";
const std::string k_orphan_yield_message = "scf.yield must be directly in an scf.for body or in "
                                           "a branch of an scf.if_else with results";

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

    /* Where an extra scf.yield of one i32 goes: nowhere, directly in the
     * region being built, or in an scf.if on `lane < 8` within it. */
    enum class extra_yield
    {
        none,
        direct,
        in_scf_if
    };

    /* Emit the extra yield `where` into the current region. */
    void emit_extra_yield(extra_yield where)
    {
        if(where == extra_yield::none)
            return;
        if(where == extra_yield::in_scf_if)
        {
            rocke_if_t inner
                = rocke_b_scf_if(&b, rocke_b_cmp_lt(&b, lane, rocke_b_const_i32(&b, 8)));
            rocke_b_region_enter(&b, inner.then_region);
        }
        rocke_value_t* one = rocke_b_const_i32(&b, 1);
        rocke_b_scf_yield(&b, &one, 1);
        if(where == extra_yield::in_scf_if)
            rocke_b_region_leave(&b);
    }

    /* An if/else without results on `lane < bound`; its then branch holds the
     * extra yield `stray`, and its else branch is empty. */
    rocke_if_else_t build_if_else_without_results(rocke_value_t* bound, extra_yield stray)
    {
        rocke_if_else_t branch = rocke_b_scf_if_else(&b, rocke_b_cmp_lt(&b, lane, bound));
        rocke_b_region_enter(&b, branch.then_region);
        emit_extra_yield(stray);
        rocke_b_region_leave(&b);
        rocke_b_region_enter(&b, branch.else_region);
        rocke_b_region_leave(&b);
        return branch;
    }

    /* An if/else with one i32 result whose branches each yield zero; its
     * then branch first holds the extra yield `early`. */
    rocke_if_else_t build_if_else_with_a_result(extra_yield early)
    {
        const rocke_type_t* i32 = rocke_i32();
        rocke_if_else_t branch = rocke_b_scf_if_else_results(
            &b, rocke_b_cmp_lt(&b, lane, rocke_b_const_i32(&b, 8)), &i32, 1);
        for(rocke_region_t* region : {branch.then_region, branch.else_region})
        {
            rocke_b_region_enter(&b, region);
            if(region == branch.then_region)
                emit_extra_yield(early);
            rocke_value_t* zero = rocke_b_const_i32(&b, 0);
            rocke_b_scf_yield(&b, &zero, 1);
            rocke_b_region_leave(&b);
        }
        return branch;
    }

    /* A loop of `trips` trips (unrolled when `unroll`) that carries `lane`;
     * its body holds what `fill` emits, then yields the carried value. */
    template <typename Fill>
    void build_carrying_loop(int trips, bool unroll, Fill fill)
    {
        rocke_iter_arg_t carry = {"acc", lane};
        rocke_for_t loop = rocke_b_scf_for_iter(&b,
                                                rocke_b_const_i32(&b, 0),
                                                rocke_b_const_i32(&b, trips),
                                                rocke_b_const_i32(&b, 1),
                                                &carry,
                                                1,
                                                "k",
                                                unroll,
                                                true);
        rocke_b_region_enter(&b, loop.body);
        fill(loop);
        rocke_b_scf_yield(&b, loop.iter_vars, 1);
        rocke_b_region_leave(&b);
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
        const rocke_status_t status = rocke_lower_kernel_to_llvm_ex(rocke_ir_builder_kernel(&b),
                                                                    ROCKE_LLVM_FLAVOR_LLVM22,
                                                                    "gfx942",
                                                                    &text,
                                                                    err,
                                                                    sizeof err);
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
    char err[ROCKE_ERR_MSG_CAP] = "";
    const rocke_status_t got = rocke_i_if_else_yields(op, yields, err, sizeof err);
    check(got == status, what + ": rule status " + std::to_string(got));
    check(err == message, what + ": rule text '" + err + "'");
}

/* The verifier reports `message` alone, and both lowerings refuse the kernel
 * with `status` and `message`. */
void check_refused(Kernel& k,
                   rocke_status_t status,
                   const std::string& message,
                   const std::string& what)
{
    const std::string diags = k.diagnostics();
    check(diags == message + "\n", what + ", verifier: " + diags);
    const refusal llvm = k.lower_llvm();
    check(llvm.status == status && llvm.message == message,
          what + ", LLVM lowering: " + std::to_string(llvm.status) + " " + llvm.message);
    const refusal hip = k.lower_hip();
    check(hip.status == status && hip.message == message,
          what + ", HIP lowering: " + std::to_string(hip.status) + " " + hip.message);
}

/* The verifier reports nothing, and both lowerings accept the kernel. */
void check_accepted(Kernel& k, const std::string& what)
{
    const std::string diags = k.diagnostics();
    check(diags.empty(), what + ", verifier: " + diags);
    const refusal llvm = k.lower_llvm();
    check(llvm.status == ROCKE_OK, what + ", LLVM lowering: " + llvm.message);
    const refusal hip = k.lower_hip();
    check(hip.status == ROCKE_OK, what + ", HIP lowering: " + hip.message);
}

/* Both kinds of if/else pass the verifier and both lowerings, and so do the
 * yields a nested loop or if/else owns: a loop in a branch, and an if/else with
 * results in an scf.if in a loop body. */
void well_formed_if_elses_are_accepted()
{
    Kernel k("if_else_well_formed");
    k.build_if_else_without_results(rocke_b_const_i32(&k.b, 8), Kernel::extra_yield::none);
    k.build_if_else_with_a_result(Kernel::extra_yield::none);
    rocke_if_else_t outer
        = k.build_if_else_without_results(rocke_b_const_i32(&k.b, 16), Kernel::extra_yield::none);
    rocke_b_region_enter(&k.b, outer.then_region);
    k.build_carrying_loop(4, false, [](const rocke_for_t&) {});
    rocke_b_region_leave(&k.b);
    k.build_carrying_loop(4, false, [&k](const rocke_for_t& loop) {
        rocke_if_t inner = rocke_b_scf_if(&k.b, rocke_b_cmp_lt(&k.b, loop.iv, k.lane));
        rocke_b_region_enter(&k.b, inner.then_region);
        k.build_if_else_with_a_result(Kernel::extra_yield::none);
        rocke_b_region_leave(&k.b);
    });
    rocke_b_ret(&k.b);
    check_accepted(k, "well-formed if/elses");
}

void a_yield_in_an_if_else_without_results_is_refused()
{
    for(Kernel::extra_yield where : {Kernel::extra_yield::direct, Kernel::extra_yield::in_scf_if})
    {
        const std::string what
            = where == Kernel::extra_yield::direct ? "stray yield" : "stray yield in an scf.if";
        Kernel k("if_else_stray_yield");
        rocke_if_else_t branch = k.build_if_else_without_results(rocke_b_const_i32(&k.b, 8), where);
        rocke_b_ret(&k.b);
        check_rule(branch.op, ROCKE_ERR_VALUE, k_stray_yield_message, what);
        check_refused(k, ROCKE_ERR_VALUE, k_stray_yield_message, what);
    }
}

/* In a loop body the stray value would count toward the loop's yields; the
 * if/else rule names it instead. */
void a_yield_in_an_if_else_without_results_in_a_loop_is_refused_as_such()
{
    Kernel k("if_else_stray_yield_loop");
    k.build_carrying_loop(4, false, [&k](const rocke_for_t& loop) {
        k.build_if_else_without_results(loop.iv, Kernel::extra_yield::direct);
    });
    rocke_b_ret(&k.b);
    check_refused(k, ROCKE_ERR_VALUE, k_stray_yield_message, "stray yield in a loop");
}

void a_yield_before_the_end_of_a_branch_is_refused()
{
    for(Kernel::extra_yield where : {Kernel::extra_yield::direct, Kernel::extra_yield::in_scf_if})
    {
        const std::string what
            = where == Kernel::extra_yield::direct ? "early yield" : "early yield in an scf.if";
        Kernel k("if_else_early_yield");
        rocke_if_else_t branch = k.build_if_else_with_a_result(where);
        rocke_b_ret(&k.b);
        check_rule(branch.op, ROCKE_ERR_VALUE, k_early_yield_message, what);
        check_refused(k, ROCKE_ERR_VALUE, k_early_yield_message, what);
    }
}

/* A yield in the kernel body, in an scf.if there, or in an scf.if in a loop
 * body: nothing takes its values. */
void an_orphan_yield_is_refused()
{
    for(Kernel::extra_yield where : {Kernel::extra_yield::direct, Kernel::extra_yield::in_scf_if})
    {
        Kernel k("orphan_yield");
        k.emit_extra_yield(where);
        rocke_b_ret(&k.b);
        check_refused(k,
                      ROCKE_ERR_VALUE,
                      k_orphan_yield_message,
                      where == Kernel::extra_yield::direct ? "orphan yield"
                                                           : "orphan yield in an scf.if");
    }
    Kernel k("orphan_yield_loop");
    k.build_carrying_loop(
        4, false, [&k](const rocke_for_t&) { k.emit_extra_yield(Kernel::extra_yield::in_scf_if); });
    rocke_b_ret(&k.b);
    check_refused(k, ROCKE_ERR_VALUE, k_orphan_yield_message, "yield in an scf.if in a loop");
}

/* An unrolled loop of zero trips lowers no body, yet the LLVM lowering refuses
 * the body's malformed if/else as the verifier and the HIP lowering do. */
void an_unrolled_loop_of_zero_trips_refuses_a_malformed_if_else()
{
    Kernel k("zero_trip_stray_yield");
    k.build_carrying_loop(0, true, [&k](const rocke_for_t& loop) {
        k.build_if_else_without_results(loop.iv, Kernel::extra_yield::direct);
    });
    rocke_b_ret(&k.b);
    check_refused(k, ROCKE_ERR_VALUE, k_stray_yield_message, "zero-trip loop, stray yield");
}

void an_unrolled_loop_of_zero_trips_accepts_a_well_formed_body()
{
    Kernel k("zero_trip_well_formed");
    k.build_carrying_loop(0, true, [&k](const rocke_for_t& loop) {
        k.build_if_else_without_results(loop.iv, Kernel::extra_yield::none);
    });
    rocke_b_ret(&k.b);
    check_accepted(k, "zero-trip loop, well-formed body");
}

/* An if/else made raw with no regions array has two NULL branches; the rule,
 * the verifier and both lowerings count them as missing instead of reading
 * them. */
void an_if_else_with_missing_branches_is_refused()
{
    Kernel k("if_else_missing_branches");
    rocke_value_t* cond = rocke_b_cmp_lt(&k.b, k.lane, rocke_b_const_i32(&k.b, 8));
    const rocke_op_t* op = rocke_i_op(
        &k.b, ROCKE_OP_SCF_IF_ELSE, &cond, 1, nullptr, 0, nullptr, nullptr, 2, "v", nullptr);
    rocke_b_ret(&k.b);
    const std::string message = "scf.if_else: expected 1 condition / 2 branches, got 1 / 0";
    check(op != nullptr && op->regions[0] == nullptr && op->regions[1] == nullptr,
          "raw if/else should leave both branches NULL");
    check_rule(op, ROCKE_ERR_VALUE, message, "missing branches");
    check_refused(k, ROCKE_ERR_VALUE, message, "missing branches");
}

/* An if/else whose result is an LDS tile of i4, the type swapped in past the
 * builder: the packed-only rule answers first, with a TypeError, in the rule,
 * the verifier and both lowerings. */
void a_packed_only_lds_result_is_refused_as_packed_only()
{
    Kernel k("if_else_packed_only_lds");
    rocke_if_else_t branch = k.build_if_else_with_a_result(Kernel::extra_yield::none);
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
    well_formed_if_elses_are_accepted();
    a_yield_in_an_if_else_without_results_is_refused();
    a_yield_in_an_if_else_without_results_in_a_loop_is_refused_as_such();
    a_yield_before_the_end_of_a_branch_is_refused();
    an_orphan_yield_is_refused();
    an_unrolled_loop_of_zero_trips_refuses_a_malformed_if_else();
    an_unrolled_loop_of_zero_trips_accepts_a_well_formed_body();
    an_if_else_with_missing_branches_is_refused();
    a_packed_only_lds_result_is_refused_as_packed_only();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("if/else yield rule: all checks passed\n");
    return 0;
}
