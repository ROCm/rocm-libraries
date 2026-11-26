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
 * the Python engine's text; loops side by side may share names. Serialized IR
 * can name values that way past the builder, and the parser refuses it with the
 * same text (and a loop value whose type attr is not a type name with a text of
 * its own); IR built past the builder can too, and the verifier reports it with
 * the same text. tests/core/test_unrolled_for_nested_regions.py covers the
 * Python builder, parser and verifier.
 */
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <regex>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
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

void refuses_a_reused_name_when_an_op_follows_the_loop_handle()
{
    /* The outer loop is not the last op of its region: a constant is emitted
     * after its handle is created and before its body is entered. */
    Kernel k("reuse_after_op");
    rocke_for_t outer = k.loop("k0", "acc");
    k.c(7);
    rocke_b_region_enter(&k.b, outer.body);
    const std::string error = refusal(&k.b, [&] { k.loop("k0", "s"); });
    check(error == message("%k0"), "loop reusing k0 after an op: '" + error + "'");
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

/* The inner loop's induction-variable attrs, as kernel_text_naming serializes
 * them. */
const std::string inner_iv_attrs = R"(iv = s:"%j", iv_type = s:"i32")";

/* `text` with its one occurrence of `old` replaced by `replacement`. */
std::string replaced_once(std::string text, const std::string& old, const std::string& replacement)
{
    const size_t at = text.find(old);
    check(at != std::string::npos && text.find(old, at + 1) == std::string::npos,
          "expected one '" + old + "' in the serialized IR");
    if(at != std::string::npos)
    {
        text.replace(at, old.size(), replacement);
    }
    return text;
}

/* The serialized IR of a rolled loop "k0" over "acc" whose body holds a rolled
 * loop "j" over "s", with every occurrence of the name `inner_name` (a value of
 * the inner loop) renamed to `reused`. With `inner_iv_type_not_a_name`, the
 * inner loop's iv_type attr becomes an integer, so the parser never binds its
 * induction variable. */
std::string kernel_text_naming(const std::string& inner_name,
                               const std::string& reused,
                               bool inner_iv_type_not_a_name = false)
{
    Kernel k("parse_reuse");
    rocke_for_t outer = k.loop("k0", "acc");
    rocke_b_region_enter(&k.b, outer.body);
    rocke_for_t inner = k.loop("j", "s");
    rocke_b_region_enter(&k.b, inner.body);
    rocke_value_t* sum = rocke_b_add(&k.b, inner.iter_vars[0], inner.iv);
    rocke_b_scf_yield(&k.b, &sum, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_scf_yield(&k.b, inner.op->results, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_ret(&k.b);
    char* serialized = NULL;
    check(rocke_ir_serialize(rocke_ir_builder_kernel(&k.b), &serialized) == ROCKE_OK,
          "serializing the parse_reuse kernel failed");
    std::string text = serialized ? serialized : "";
    free(serialized);
    if(inner_iv_type_not_a_name)
    {
        text = replaced_once(text, inner_iv_attrs, R"(iv = s:"%j", iv_type = i:0)");
    }
    /* Whole names only: "%s" must not match the start of a longer name. */
    return std::regex_replace(text, std::regex(inner_name + R"((?![-a-zA-Z$._0-9]))"), reused);
}

void parser_refuses_a_reused_loop_name(const std::string& reused,
                                       const std::string& inner_name,
                                       bool inner_iv_type_not_a_name = false)
{
    const std::string text = kernel_text_naming(inner_name, reused, inner_iv_type_not_a_name);
    Kernel parsed("parsed");
    rocke_kernel_def_t* kernel = NULL;
    const rocke_status_t status = rocke_ir_parse(text.c_str(), &parsed.b, &kernel);
    const std::string error = status == ROCKE_OK ? "" : rocke_ir_builder_error(&parsed.b);
    check(status == ROCKE_ERR_VALUE && error == message(reused),
          "parsing " + inner_name + " renamed to " + reused + ": status " + std::to_string(status)
              + ", '" + error + "'");
}

void parser_refuses_reused_loop_names()
{
    parser_refuses_a_reused_loop_name("%k0", "%j"); /* the outer induction variable */
    parser_refuses_a_reused_loop_name("%acc", "%s"); /* the outer carried value */
    parser_refuses_a_reused_loop_name("%j", "%s"); /* a value of the same loop */
    /* An induction variable whose iv_type is not a type name is never bound,
     * but its name is the loop's all the same: unrefused, the inner body's
     * reads of it would read the outer induction variable. */
    parser_refuses_a_reused_loop_name("%k0", "%j", /*inner_iv_type_not_a_name=*/true);
}

/* A parser binds a loop's induction variable or carried value with the type
 * its attr names. When the attr is not a type name the value cannot be bound,
 * and the body's reads of its name would read whatever other value holds that
 * name, so the parser refuses it. */
void parser_refuses_a_loop_value_without_a_type_name(const std::string& typed_attrs,
                                                     const std::string& untyped_attrs,
                                                     const std::string& untyped)
{
    const std::string text
        = replaced_once(kernel_text_naming("%s", "%s"), typed_attrs, untyped_attrs);
    Kernel parsed("parsed");
    rocke_kernel_def_t* kernel = NULL;
    const rocke_status_t status = rocke_ir_parse(text.c_str(), &parsed.b, &kernel);
    const std::string error = status == ROCKE_OK ? "" : rocke_ir_builder_error(&parsed.b);
    check(status == ROCKE_ERR_VALUE
              && error
                     == "scf.for: the loop value '" + untyped
                            + "' has no type name; its iv_type attr or iter_args entry must "
                              "name its type",
          "parsing " + untyped + " with no type name: status " + std::to_string(status) + ", '"
              + error + "'");
}

void parser_refuses_loop_values_without_a_type_name()
{
    parser_refuses_a_loop_value_without_a_type_name(
        inner_iv_attrs, R"(iv = s:"%j", iv_type = i:0)", "%j");
    parser_refuses_a_loop_value_without_a_type_name(
        R"({ name = s:"%s", type = s:"i32" })", R"({ name = s:"%s", type = i:0 })", "%s");
}

void parser_accepts_the_names_the_builder_gave()
{
    const std::string text = kernel_text_naming("%s", "%s");
    Kernel parsed("parsed");
    rocke_kernel_def_t* kernel = NULL;
    check(rocke_ir_parse(text.c_str(), &parsed.b, &kernel) == ROCKE_OK,
          std::string("parsing the parse_reuse kernel failed: ")
              + rocke_ir_builder_error(&parsed.b));
}

/* Every diagnostic the verifier reports for `kernel`, as "<severity>: <text>"
 * lines. */
std::string diagnostics(const rocke_kernel_def_t* kernel)
{
    rocke_diag_t* diags = NULL;
    size_t n = 0;
    if(rocke_verify(kernel, &diags, &n) != ROCKE_OK)
    {
        return "rocke_verify failed";
    }
    std::string all;
    for(size_t i = 0; i < n; ++i)
    {
        all += std::string(diags[i].severity == ROCKE_DIAG_ERROR ? "error: " : "warning: ")
               + diags[i].message + "\n";
    }
    rocke_diags_free(diags, n);
    return all;
}

/* The diagnostics for a rolled loop "k0" over "acc" whose body holds a rolled
 * loop "j" over "s", built by the builder; then, past it, the inner loop's
 * induction variable is renamed to `iv_name` and its carried value to
 * `carried_name`, in the loop's attrs and in the values every read sees. With
 * `iv_type_not_a_name`, the inner loop's iv_type attr becomes an integer. */
std::string diagnostics_renaming(const char* iv_name,
                                 const char* carried_name,
                                 bool iv_type_not_a_name = false)
{
    Kernel k("verify_reuse");
    rocke_for_t outer = k.loop("k0", "acc");
    rocke_b_region_enter(&k.b, outer.body);
    rocke_for_t inner = k.loop("j", "s");
    rocke_b_region_enter(&k.b, inner.body);
    rocke_value_t* sum = rocke_b_add(&k.b, inner.iter_vars[0], inner.iv);
    rocke_b_scf_yield(&k.b, &sum, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_scf_yield(&k.b, inner.op->results, 1);
    rocke_b_region_leave(&k.b);
    rocke_b_ret(&k.b);
    if(iv_type_not_a_name)
    {
        rocke_attr_set_int(&k.b, &inner.op->attrs, "iv_type", 0);
    }
    rocke_attr_set_str(&k.b, &inner.op->attrs, "iv", iv_name);
    inner.iv->name = iv_name;
    rocke_attr_map_t* carried = rocke_attr_get(&inner.op->attrs, "iter_args")->u.list.items[0];
    rocke_attr_set_str(&k.b, carried, "name", carried_name);
    inner.iter_vars[0]->name = carried_name;
    return diagnostics(rocke_ir_builder_kernel(&k.b));
}

/* Each case names the inner loop's values and the one name the verifier must
 * report: the loop's first reused name, its induction variable before its
 * carried value. */
void verifier_reports_reused_loop_names()
{
    struct Case
    {
        const char* iv_name;
        const char* carried_name;
        const char* reported;
    };
    const Case cases[] = {
        {"%k0", "%s", "%k0"}, /* the outer induction variable */
        {"%j", "%acc", "%acc"}, /* the outer carried value */
        {"%j", "%j", "%j"}, /* a value of the same loop */
        {"%k0", "%acc", "%k0"}, /* both outer names: reported once, by the first */
    };
    for(const Case& c : cases)
    {
        const std::string reported = diagnostics_renaming(c.iv_name, c.carried_name);
        check(reported == "error: " + message(c.reported) + "\n",
              std::string("verifying an inner loop over ") + c.iv_name + " carrying "
                  + c.carried_name + ": '" + reported + "'");
    }
}

/* The induction variable is a loop name whenever the loop names it, also when
 * its iv_type attr is not a type name, as in the Python verifier. */
void verifier_counts_an_induction_variable_whose_type_is_not_named()
{
    const std::string reported = diagnostics_renaming("%k0", "%s", /*iv_type_not_a_name=*/true);
    check(reported == "error: " + message("%k0") + "\n",
          "verifying an inner loop over %k0 with an integer iv_type: '" + reported + "'");
}

void verifier_accepts_the_names_the_builder_gave()
{
    const std::string reported = diagnostics_renaming("%j", "%s");
    check(reported.empty(), "verifying the verify_reuse kernel: '" + reported + "'");
}

} // namespace

int main()
{
    refuses_a_nested_loop_reusing_the_induction_variable();
    refuses_a_nested_loop_reusing_a_carried_value();
    refuses_a_reused_name_inside_an_if_in_the_body();
    refuses_a_reused_name_when_an_op_follows_the_loop_handle();
    refuses_one_loop_naming_two_values_alike();
    accepts_new_names_and_loops_side_by_side();
    parser_refuses_reused_loop_names();
    parser_refuses_loop_values_without_a_type_name();
    parser_accepts_the_names_the_builder_gave();
    verifier_reports_reused_loop_names();
    verifier_counts_an_induction_variable_whose_type_is_not_named();
    verifier_accepts_the_names_the_builder_gave();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("scf.for name reuse: all checks passed\n");
    return 0;
}
