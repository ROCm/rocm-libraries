// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_scf_for_name_reuse.cpp -- the C++ builder refuses an scf.for
 * whose induction variable or carried value reuses a name that is taken.
 *
 * A loop names its induction variable and carried values itself, and the
 * serialized IR refers to a value by its name. When a nested loop reuses the
 * name of a value of a loop it sits in, or one loop gives two of its values the
 * same name, a read of that name inside the inner body is ambiguous, and the
 * two engines would read different values. The builder refuses such a name with
 * the Python engine's text; loops side by side may share names.
 * tests/core/test_unrolled_for_nested_regions.py covers the Python builder.
 */
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"

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

std::string message(const std::string& name)
{
    return "scf.for: the name '" + name
           + "' is already taken by an induction variable or carried value of this loop or of a "
             "loop it is nested in; give it a name of its own";
}

/* The refusal `build` meets: a builder refusal throws a ckc::ValueError (and
 * leaves the builder failed). Empty when nothing is refused. */
std::string refusal(rocke_ir_builder_t* b, const std::function<void()>& build)
{
    try
    {
        build();
    }
    catch(const ckc::Error& error)
    {
        check(error.code() == ROCKE_ERR_VALUE, "refusal is not a ValueError");
        return error.what();
    }
    return rocke_ir_builder_ok(b) ? "" : rocke_ir_builder_error(b);
}

struct Kernel
{
    rocke_ir_builder_t b;

    explicit Kernel(const char* name)
    {
        if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", name);
            std::exit(1);
        }
    }
    ~Kernel()
    {
        rocke_ir_builder_free(&b);
    }

    rocke_value_t* c(int v)
    {
        return rocke_b_const_i32(&b, v);
    }

    /* A loop over [0, 2) carrying one i32 named `carried`. */
    rocke_for_t loop(const char* iv_name, const char* carried)
    {
        rocke_iter_arg_t arg = {carried, c(0)};
        return rocke_b_scf_for_iter(&b, c(0), c(2), c(1), &arg, 1, iv_name, false, true);
    }
};

/* Build `inner` inside the body of a loop with induction variable "k0" and
 * carried value "acc"; with `through_if`, inside an scf.if in that body. */
std::string
    nested_refusal(const char* name, bool through_if, const std::function<void(Kernel&)>& inner)
{
    Kernel k(name);
    rocke_for_t outer = k.loop("k0", "acc");
    rocke_b_region_enter(&k.b, outer.body);
    if(through_if)
    {
        rocke_if_t branch = rocke_b_scf_if(&k.b, rocke_b_cmp_lt(&k.b, outer.iv, k.c(1)));
        rocke_b_region_enter(&k.b, branch.then_region);
    }
    return refusal(&k.b, [&] { inner(k); });
}

void refuses_a_nested_loop_reusing_the_induction_variable()
{
    const std::string error = nested_refusal(
        "reuse_iv", false, [](Kernel& k) { rocke_b_scf_for(&k.b, k.c(0), k.c(2), k.c(1), "k0"); });
    check(error == message("%k0"), "nested scf_for reusing k0: '" + error + "'");
}

void refuses_a_nested_loop_reusing_a_carried_value()
{
    const std::string error
        = nested_refusal("reuse_carried", false, [](Kernel& k) { k.loop("j", "acc"); });
    check(error == message("%acc"), "nested scf_for_iter reusing acc: '" + error + "'");
}

void refuses_a_reused_name_inside_an_if_in_the_body()
{
    const std::string error
        = nested_refusal("reuse_through_if", true, [](Kernel& k) { k.loop("k0", "s"); });
    check(error == message("%k0"), "loop in an scf.if reusing k0: '" + error + "'");
}

void refuses_one_loop_naming_two_values_alike()
{
    Kernel k("same_loop");
    const std::string error = refusal(&k.b, [&] { k.loop("k0", "k0"); });
    check(error == message("%k0"), "carried value named like the iv: '" + error + "'");
}

void accepts_new_names_and_loops_side_by_side()
{
    Kernel k("distinct");
    rocke_for_t outer = k.loop("k0", "acc");
    rocke_b_region_enter(&k.b, outer.body);
    rocke_for_t inner = k.loop("j", "s");
    rocke_b_region_enter(&k.b, inner.body);
    rocke_b_scf_yield(&k.b, inner.iter_vars, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_scf_yield(&k.b, outer.iter_vars, 1);
    rocke_b_region_leave(&k.b);
    rocke_for_t after = k.loop("k0", "acc");
    rocke_b_region_enter(&k.b, after.body);
    rocke_b_scf_yield(&k.b, after.iter_vars, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_ret(&k.b);
    check(rocke_ir_builder_ok(&k.b),
          std::string("distinct and side-by-side names refused: ") + rocke_ir_builder_error(&k.b));
}

} // namespace

int main()
{
    refuses_a_nested_loop_reusing_the_induction_variable();
    refuses_a_nested_loop_reusing_a_carried_value();
    refuses_a_reused_name_inside_an_if_in_the_body();
    refuses_one_loop_naming_two_values_alike();
    accepts_new_names_and_loops_side_by_side();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("scf.for name reuse: all checks passed\n");
    return 0;
}
