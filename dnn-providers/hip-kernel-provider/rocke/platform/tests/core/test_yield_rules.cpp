// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_yield_rules.cpp -- the C++ verifier, the if/else rule
 * (rocke_i_if_else_yields), the loop rule (rocke_i_scf_for_error) and both
 * lowerings refuse what an scf.if_else or an scf.for body cannot hold, and an
 * scf.yield where nothing takes it, with the Python engine's types and texts.
 *
 * An scf.for's carried-value counts agree, its iter_args metadata names each
 * carried value and its type (which the loop's result shares), and its body
 * yields only at its end, one value per carried value, of its type, ending in
 * that yield whenever the loop carries any.
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
 * tests/core/test_scf_if_else_results.py and tests/core/test_scf_for_yields.py
 * cover the Python engine and both engines' lowering texts;
 * tests/core/test_packed_only.py covers both builders' result-type order.
 */
#include <cstdio>
#include <cstdlib>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
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
const std::string k_loop_early_yield_message
    = "scf.for body yields before its last op; a loop takes its values only from the scf.yield "
      "that ends its body";
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
        init_builder(name);
        lane = rocke_b_thread_id_x(&b);
    }

    /* Selects the constructor that parses a kernel from serialized IR. */
    struct from_text
    {
        const std::string& text;
    };

    /* The kernel `source.text` holds, parsed as serialized IR reaches the
     * engine; it has no `lane` to build on. */
    explicit Kernel(from_text source)
    {
        init_builder("parsed");
        rocke_kernel_def_t* parsed = nullptr;
        if(rocke_ir_parse(source.text.c_str(), &b, &parsed) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: parse: %s\n", rocke_ir_builder_error(&b));
            std::exit(1);
        }
    }
    ~Kernel()
    {
        rocke_ir_builder_free(&b);
    }

    void init_builder(const char* name)
    {
        if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", name);
            std::exit(1);
        }
    }

    /* The kernel built so far, as serialized IR. */
    std::string serialized()
    {
        char* text = nullptr;
        if(rocke_ir_serialize(rocke_ir_builder_kernel(&b), &text) != ROCKE_OK || !text)
        {
            fprintf(stderr, "FAIL: serialize\n");
            std::exit(1);
        }
        std::string out(text);
        free(text);
        return out;
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
     * its body holds what `fill` emits, then, when `close` (the default),
     * yields the carried value. */
    template <typename Fill>
    void build_carrying_loop(int trips, bool unroll, Fill fill, bool close = true)
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
        if(close)
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

/* A loop body yields only at its end: a yield, an op and the closing yield; two
 * yields in a row; and a yield followed by an op and no closing yield are each
 * refused, in a loop that carries a value, in one that carries none, and in an
 * unrolled loop of zero trips. */
void a_loop_yield_before_the_end_of_the_body_is_refused()
{
    const auto yield_carried
        = [](Kernel& k, const rocke_for_t& loop) { rocke_b_scf_yield(&k.b, loop.iter_vars, 1); };
    {
        Kernel k("loop_early_yield");
        k.build_carrying_loop(4, false, [&k, &yield_carried](const rocke_for_t& loop) {
            yield_carried(k, loop);
            rocke_b_const_i32(&k.b, 3);
        });
        rocke_b_ret(&k.b);
        check_refused(
            k, ROCKE_ERR_VALUE, k_loop_early_yield_message, "loop yield before the last op");
    }
    {
        Kernel k("loop_two_yields");
        k.build_carrying_loop(
            4, false, [&k, &yield_carried](const rocke_for_t& loop) { yield_carried(k, loop); });
        rocke_b_ret(&k.b);
        check_refused(k, ROCKE_ERR_VALUE, k_loop_early_yield_message, "loop with two yields");
    }
    {
        Kernel k("loop_yield_not_last");
        k.build_carrying_loop(
            4,
            false,
            [&k, &yield_carried](const rocke_for_t& loop) {
                yield_carried(k, loop);
                rocke_b_const_i32(&k.b, 3);
            },
            false);
        rocke_b_ret(&k.b);
        check_refused(k, ROCKE_ERR_VALUE, k_loop_early_yield_message, "loop yield not the last op");
    }
    {
        Kernel k("zero_trip_two_yields");
        k.build_carrying_loop(
            0, true, [&k, &yield_carried](const rocke_for_t& loop) { yield_carried(k, loop); });
        rocke_b_ret(&k.b);
        check_refused(
            k, ROCKE_ERR_VALUE, k_loop_early_yield_message, "zero-trip loop with two yields");
    }
    {
        Kernel k("plain_loop_two_yields");
        rocke_for_t loop = rocke_b_scf_for(&k.b,
                                           rocke_b_const_i32(&k.b, 0),
                                           rocke_b_const_i32(&k.b, 4),
                                           rocke_b_const_i32(&k.b, 1),
                                           "k");
        rocke_b_region_enter(&k.b, loop.body);
        rocke_b_scf_yield(&k.b, nullptr, 0);
        rocke_b_scf_yield(&k.b, nullptr, 0);
        rocke_b_region_leave(&k.b);
        rocke_b_ret(&k.b);
        check_refused(k,
                      ROCKE_ERR_VALUE,
                      k_loop_early_yield_message,
                      "loop without carried values, two yields");
    }
}

/* A loop that carries a value ends in a yield of one value per carried value,
 * of its type. */
void a_loop_yield_that_does_not_match_the_carried_values_is_refused()
{
    {
        Kernel k("loop_missing_yield");
        k.build_carrying_loop(4, false, [](const rocke_for_t&) {}, false);
        rocke_b_ret(&k.b);
        check_refused(k,
                      ROCKE_ERR_VALUE,
                      "scf.for carries 1 values, so its body must end in scf.yield of 1 values",
                      "loop without its yield");
    }
    {
        Kernel k("loop_yield_count");
        k.build_carrying_loop(
            4,
            false,
            [&k](const rocke_for_t& loop) {
                rocke_value_t* values[2] = {loop.iter_vars[0], loop.iter_vars[0]};
                rocke_b_scf_yield(&k.b, values, 2);
            },
            false);
        rocke_b_ret(&k.b);
        check_refused(k,
                      ROCKE_ERR_VALUE,
                      "scf.for carries 1 values, but its body yields 2 values",
                      "loop yield count");
    }
    {
        Kernel k("loop_yield_type");
        k.build_carrying_loop(
            4,
            false,
            [&k](const rocke_for_t&) {
                rocke_value_t* value = rocke_b_const_f32(&k.b, 1.0f);
                rocke_b_scf_yield(&k.b, &value, 1);
            },
            false);
        rocke_b_ret(&k.b);
        check_refused(k,
                      ROCKE_ERR_VALUE,
                      "scf.for iter_arg 0 is i32, but its body yields f32",
                      "loop yield type");
    }
}

/* A well-formed outer loop does not hide an inner loop that carries two values
 * and yields one: the inner loop's own counts are refused. */
void each_loop_is_held_to_its_own_carried_values()
{
    Kernel k("inner_loop_yield_count");
    k.build_carrying_loop(4, false, [&k](const rocke_for_t& outer) {
        rocke_iter_arg_t carries[2] = {{"x", outer.iv}, {"y", outer.iter_vars[0]}};
        rocke_for_t inner = rocke_b_scf_for_iter(&k.b,
                                                 rocke_b_const_i32(&k.b, 0),
                                                 rocke_b_const_i32(&k.b, 2),
                                                 rocke_b_const_i32(&k.b, 1),
                                                 carries,
                                                 2,
                                                 "inner",
                                                 false,
                                                 true);
        rocke_b_region_enter(&k.b, inner.body);
        rocke_b_scf_yield(&k.b, inner.iter_vars, 1);
        rocke_b_region_leave(&k.b);
    });
    rocke_b_ret(&k.b);
    check_refused(k,
                  ROCKE_ERR_VALUE,
                  "scf.for carries 2 values, but its body yields 1 values",
                  "inner loop yield count");
}

/* Serialized IR can give a loop a num_iter_args or a result count that
 * disagrees with its init operands; the verifier and both lowerings refuse it,
 * the HIP lowering included, instead of reading the disagreeing count. */
void a_loop_whose_carried_counts_disagree_is_refused()
{
    Kernel k("loop_counts");
    rocke_op_t* loop_op = nullptr;
    k.build_carrying_loop(4, false, [&loop_op](const rocke_for_t& loop) { loop_op = loop.op; });
    rocke_b_ret(&k.b);
    rocke_attr_set_int(&k.b, &loop_op->attrs, "num_iter_args", 0);
    check_refused(k,
                  ROCKE_ERR_VALUE,
                  "scf.for declares 0 iter_args but has 1 init operands",
                  "loop with a disagreeing num_iter_args");

    Kernel no_result("loop_counts_results");
    rocke_op_t* result_loop_op = nullptr;
    no_result.build_carrying_loop(
        4, false, [&result_loop_op](const rocke_for_t& loop) { result_loop_op = loop.op; });
    rocke_b_ret(&no_result.b);
    result_loop_op->num_results = 0;
    check_refused(no_result,
                  ROCKE_ERR_VALUE,
                  "scf.for declares 1 iter_args but has 0 results",
                  "loop with a disagreeing result count");
}

/* The loop's iter_args metadata attr, to edit as serialized IR could leave it
 * (the attr API hands out only a read-only view). */
rocke_attr_value_t& iter_args_metadata(rocke_op_t* loop_op)
{
    for(int e = 0; e < loop_op->attrs.count; ++e)
    {
        if(std::string(loop_op->attrs.entries[e].key) == "iter_args")
            return loop_op->attrs.entries[e].value;
    }
    fprintf(stderr, "FAIL: the loop has no iter_args metadata\n");
    std::exit(1);
}

/* Serialized IR can give a loop iter_args metadata that is not a list, has the
 * wrong number of entries, or whose entries do not describe the carried
 * values; the verifier and both lowerings refuse it, the HIP lowering
 * included. Each loop yields its init value, so its body does not depend on
 * the metadata naming the carried value. */
void a_loop_whose_iter_args_metadata_is_malformed_is_refused()
{
    struct metadata_case
    {
        const char* what;
        void (*edit)(rocke_ir_builder_t* b, rocke_op_t* loop_op);
        std::string message;
    };
    const metadata_case cases[] = {
        {"metadata that is not a list",
         [](rocke_ir_builder_t* b, rocke_op_t* loop_op) {
             rocke_attr_set_str(b, &loop_op->attrs, "iter_args", "invalid");
         },
         "scf.for iter_args metadata must be a list"},
        {"metadata with no entries",
         [](rocke_ir_builder_t*, rocke_op_t* loop_op) {
             iter_args_metadata(loop_op).u.list.count = 0;
         },
         "scf.for declares 1 iter_args but has 0 metadata entries"},
        {"a metadata entry that is not a map",
         [](rocke_ir_builder_t*, rocke_op_t* loop_op) {
             iter_args_metadata(loop_op).u.list.items[0] = nullptr;
         },
         "scf.for iter_arg 0 metadata must be a map"},
        {"a metadata entry with an empty name",
         [](rocke_ir_builder_t* b, rocke_op_t* loop_op) {
             rocke_attr_set_str(b, iter_args_metadata(loop_op).u.list.items[0], "name", "");
         },
         "scf.for iter_arg 0 metadata has no name"},
        {"a metadata entry whose name is not a string",
         [](rocke_ir_builder_t* b, rocke_op_t* loop_op) {
             rocke_attr_set_int(b, iter_args_metadata(loop_op).u.list.items[0], "name", 0);
         },
         "scf.for iter_arg 0 metadata has no name"},
        {"a metadata entry whose type is not a string",
         [](rocke_ir_builder_t* b, rocke_op_t* loop_op) {
             rocke_attr_set_int(b, iter_args_metadata(loop_op).u.list.items[0], "type", 0);
         },
         "scf.for: the loop value '%acc' has no type name; its iv_type attr or iter_args entry "
         "must name its type"},
        {"a metadata entry naming another type",
         [](rocke_ir_builder_t* b, rocke_op_t* loop_op) {
             rocke_attr_set_str(b, iter_args_metadata(loop_op).u.list.items[0], "type", "f32");
         },
         "scf.for iter_arg %acc metadata type 'f32' does not match init type 'i32'"},
        {"a result of another type",
         [](rocke_ir_builder_t*, rocke_op_t* loop_op) { loop_op->results[0]->type = rocke_f32(); },
         "scf.for result type 'f32' does not match iter_arg %acc type 'i32'"},
    };
    for(const metadata_case& metadata : cases)
    {
        Kernel k("loop_metadata");
        rocke_op_t* loop_op = nullptr;
        k.build_carrying_loop(
            4,
            false,
            [&k, &loop_op](const rocke_for_t& loop) {
                loop_op = loop.op;
                rocke_b_scf_yield(&k.b, &k.lane, 1);
            },
            false);
        rocke_b_ret(&k.b);
        metadata.edit(&k.b, loop_op);
        check_refused(
            k, ROCKE_ERR_VALUE, metadata.message, std::string("loop with ") + metadata.what);
    }
}

/* Serialized IR can give a loop iter_args metadata that is a bare map rather
 * than a list of maps; the parser reads it (rocke_i_attr_is_bare_map), and the
 * verifier and both lowerings refuse the parsed loop as Python refuses a dict.
 * The loop yields its init value, so its body names no carried value. */
void a_loop_whose_iter_args_metadata_is_a_bare_map_is_refused()
{
    Kernel built("loop_metadata_bare_map");
    built.build_carrying_loop(
        4,
        false,
        [&built](const rocke_for_t&) { rocke_b_scf_yield(&built.b, &built.lane, 1); },
        false);
    rocke_b_ret(&built.b);
    std::string text = built.serialized();
    const std::string list_open = "iter_args = l:[ ";
    const size_t open = text.find(list_open);
    const size_t close = text.find(" ]", open);
    if(open == std::string::npos || close == std::string::npos)
    {
        fprintf(stderr, "FAIL: no iter_args list in the serialized loop:\n%s\n", text.c_str());
        std::exit(1);
    }
    const std::string entry = text.substr(open + list_open.size(), close - open - list_open.size());
    text.replace(open, close + 2 - open, "iter_args = " + entry);
    Kernel parsed(Kernel::from_text{text});
    const rocke_op_t* loop_op = nullptr;
    const rocke_region_t* body = rocke_ir_builder_kernel(&parsed.b)->body;
    for(int i = 0; i < body->num_ops; i++)
    {
        if(body->ops[i]->opcode == ROCKE_OP_SCF_FOR)
            loop_op = body->ops[i];
    }
    check(loop_op && rocke_i_attr_is_bare_map(rocke_attr_get(&loop_op->attrs, "iter_args")),
          "the parser reads the metadata as a bare map");
    check_refused(parsed,
                  ROCKE_ERR_VALUE,
                  "scf.for iter_args metadata must be a list",
                  "loop with metadata that is a bare map");
}

/* The lowerings read only a loop's first region, so serialized IR with no
 * body, or with a second region whose operand-free yield would be dropped
 * unchecked, is refused: the parser reads it, and the verifier and both
 * lowerings refuse the parsed loop. A loop made raw with no regions array has
 * a NULL body, which counts as missing. */
void a_loop_without_exactly_one_body_region_is_refused()
{
    const std::string empty_body = "      region @body {\n      }\n";
    const std::string extra_region
        = "      region @extra {\n        scf.yield (  ) { num = i:0 }\n      }\n";
    const struct
    {
        std::string regions;
        int count;
    } cases[] = {{"", 0}, {empty_body + extra_region, 2}};
    for(const auto& regions_case : cases)
    {
        Kernel built("loop_regions");
        rocke_b_scf_for(&built.b,
                        rocke_b_const_i32(&built.b, 0),
                        rocke_b_const_i32(&built.b, 4),
                        rocke_b_const_i32(&built.b, 1),
                        "k");
        rocke_b_ret(&built.b);
        std::string text = built.serialized();
        const size_t body_at = text.find(empty_body);
        if(body_at == std::string::npos)
        {
            fprintf(stderr, "FAIL: no empty loop body in the serialized loop:\n%s\n", text.c_str());
            std::exit(1);
        }
        text.replace(body_at, empty_body.size(), regions_case.regions);
        Kernel parsed(Kernel::from_text{text});
        const std::string count = std::to_string(regions_case.count);
        const rocke_region_t* body = rocke_ir_builder_kernel(&parsed.b)->body;
        int loop_regions = -1;
        for(int i = 0; i < body->num_ops; i++)
        {
            if(body->ops[i]->opcode == ROCKE_OP_SCF_FOR)
                loop_regions = body->ops[i]->num_regions;
        }
        check(loop_regions == regions_case.count, "the parsed loop holds " + count + " regions");
        check_refused(parsed,
                      ROCKE_ERR_VALUE,
                      "scf.for: expected 1 body region, got " + count,
                      "loop with " + count + " regions");
    }

    Kernel raw("loop_null_body");
    rocke_value_t* bounds[] = {
        rocke_b_const_i32(&raw.b, 0), rocke_b_const_i32(&raw.b, 4), rocke_b_const_i32(&raw.b, 1)};
    const rocke_op_t* loop_op = rocke_i_op(
        &raw.b, ROCKE_OP_SCF_FOR, bounds, 3, nullptr, 0, nullptr, nullptr, 1, "v", nullptr);
    rocke_b_ret(&raw.b);
    check(loop_op != nullptr && loop_op->num_regions == 1 && loop_op->regions[0] == nullptr,
          "raw loop should leave its body NULL");
    check_refused(raw, ROCKE_ERR_VALUE, "scf.for: expected 1 body region, got 0", "NULL body");
}

/* An unrolled loop that carries a pointer, which has no loop-carried LLVM form,
 * and whose body holds a loop with a yield in an scf.if (when `orphan_yield`)
 * or only its closing yield. */
void build_unrolled_pointer_loop(Kernel& k, int trips, bool orphan_yield)
{
    rocke_value_t* pointer
        = rocke_b_param(&k.b, "p", rocke_ptr_type(&k.b, rocke_f32(), "global"), nullptr);
    rocke_iter_arg_t carry = {"acc", pointer};
    rocke_for_t outer = rocke_b_scf_for_iter(&k.b,
                                             rocke_b_const_i32(&k.b, 0),
                                             rocke_b_const_i32(&k.b, trips),
                                             rocke_b_const_i32(&k.b, 1),
                                             &carry,
                                             1,
                                             "k",
                                             true,
                                             true);
    rocke_b_region_enter(&k.b, outer.body);
    if(orphan_yield)
    {
        rocke_iter_arg_t inner_carry = {"x", outer.iv};
        rocke_for_t inner = rocke_b_scf_for_iter(&k.b,
                                                 rocke_b_const_i32(&k.b, 0),
                                                 rocke_b_const_i32(&k.b, 2),
                                                 rocke_b_const_i32(&k.b, 1),
                                                 &inner_carry,
                                                 1,
                                                 "j",
                                                 false,
                                                 true);
        rocke_b_region_enter(&k.b, inner.body);
        rocke_if_t orphan
            = rocke_b_scf_if(&k.b, rocke_b_cmp_lt(&k.b, inner.iv, rocke_b_const_i32(&k.b, 1)));
        rocke_b_region_enter(&k.b, orphan.then_region);
        rocke_b_scf_yield(&k.b, inner.iter_vars, 1);
        rocke_b_region_leave(&k.b);
        rocke_b_scf_yield(&k.b, inner.iter_vars, 1);
        rocke_b_region_leave(&k.b);
    }
    rocke_b_scf_yield(&k.b, outer.iter_vars, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_ret(&k.b);
}

/* The LLVM lowering refuses the pointer carry of build_unrolled_pointer_loop
 * on its own, and, when the body is refused too, refuses the body first,
 * whether it lowers it (two trips) or only checks it (zero trips), as the
 * Python lowering does. */
void an_unrolled_loop_refuses_its_body_before_its_carried_types()
{
    for(int trips : {0, 2})
    {
        const std::string what
            = "unrolled loop of " + std::to_string(trips) + " trips carrying a pointer";
        Kernel sound_body("unrolled_pointer_carry");
        build_unrolled_pointer_loop(sound_body, trips, false);
        const refusal carry = sound_body.lower_llvm();
        check(carry.status == ROCKE_ERR_NOTIMPL
                  && carry.message
                         == "scf.for loop-carried type 'ptr<f32,global>' is unsupported; expected "
                            "a scalar or vector of scalar values",
              what + ", sound body, LLVM lowering: " + std::to_string(carry.status) + " "
                  + carry.message);

        Kernel k("unrolled_body_first");
        build_unrolled_pointer_loop(k, trips, true);
        const refusal llvm = k.lower_llvm();
        check(llvm.status == ROCKE_ERR_VALUE && llvm.message == k_orphan_yield_message,
              what + ", LLVM lowering: " + std::to_string(llvm.status) + " " + llvm.message);
    }
}

/* Well-formed loops lower: one that carries a value, one that carries none
 * with and without an empty closing yield, and one whose body holds a loop and
 * an if/else with results, whose own yields are theirs. */
void well_formed_loops_are_accepted()
{
    Kernel k("loops_well_formed");
    /* A loop that carries nothing, its body ending in an empty yield when
     * `closing_yield`. */
    const auto build_plain_loop = [&k](const char* iv_name, bool closing_yield) {
        rocke_for_t loop = rocke_b_scf_for(&k.b,
                                           rocke_b_const_i32(&k.b, 0),
                                           rocke_b_const_i32(&k.b, 4),
                                           rocke_b_const_i32(&k.b, 1),
                                           iv_name);
        rocke_b_region_enter(&k.b, loop.body);
        rocke_b_const_i32(&k.b, 7);
        if(closing_yield)
            rocke_b_scf_yield(&k.b, nullptr, 0);
        rocke_b_region_leave(&k.b);
    };
    k.build_carrying_loop(4, false, [&k, &build_plain_loop](const rocke_for_t&) {
        build_plain_loop("inner", true);
        k.build_if_else_with_a_result(Kernel::extra_yield::none);
    });
    build_plain_loop("i", false);
    build_plain_loop("j", true);
    rocke_b_ret(&k.b);
    check_accepted(k, "well-formed loops");
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
    a_loop_yield_before_the_end_of_the_body_is_refused();
    a_loop_yield_that_does_not_match_the_carried_values_is_refused();
    each_loop_is_held_to_its_own_carried_values();
    a_loop_whose_carried_counts_disagree_is_refused();
    a_loop_whose_iter_args_metadata_is_malformed_is_refused();
    a_loop_whose_iter_args_metadata_is_a_bare_map_is_refused();
    a_loop_without_exactly_one_body_region_is_refused();
    an_unrolled_loop_refuses_its_body_before_its_carried_types();
    well_formed_loops_are_accepted();
    an_if_else_with_missing_branches_is_refused();
    a_packed_only_lds_result_is_refused_as_packed_only();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("yield rules: all checks passed\n");
    return 0;
}
