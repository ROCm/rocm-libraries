// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_value_names.cpp -- the C++ builder never gives two values
 * the same name.
 *
 * The builder names every value it makes "%<hint>_<n>": the op's hint, '_',
 * then a counter that never repeats inside one builder. Before the '_' was
 * there, the 16-element zero vector made as value 5 (hint "cz16") and the
 * 1-element zero vector made as value 65 (hint "cz1") were both "%cz165". A
 * name the author picks (kernel parameter, loop induction variable,
 * loop-carried value) is refused when it ends in '_' and digits or is not an
 * identifier; a hint that is empty or starts with a digit is refused too. The
 * messages are the Python engine's; tests/core/test_value_names.py covers the
 * Python builder and checks both engines lower the kernel identically.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

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

std::string not_identifier(const std::string& what, const std::string& name)
{
    return what + " name '" + name
           + "' is not an identifier: use letters, digits and '_', not starting with a digit";
}

std::string reserved(const std::string& what, const std::string& name)
{
    return what + " name '" + name
           + "' ends in '_' and digits, the form the builder gives the values it names; choose "
             "a name that does not end in '_<digits>'";
}

std::string bad_hint(const std::string& hint)
{
    return "value name hint '" + hint + "' must be non-empty and must not start with a digit";
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

struct Builder
{
    rocke_ir_builder_t b;

    explicit Builder(const char* name)
    {
        if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", name);
            std::exit(1);
        }
    }
    ~Builder()
    {
        rocke_ir_builder_free(&b);
    }
};

void a_hint_ending_in_digits_does_not_run_into_the_counter()
{
    Builder k("value_names");
    rocke_b_param(&k.b, "Out", rocke_ptr_type(&k.b, rocke_f32(), "global"), NULL);
    for(int i = 0; i < 4; ++i)
        rocke_b_const_i32(&k.b, 0);
    rocke_value_t* z16 = rocke_b_zero_vec_f32(&k.b, 16);
    for(int i = 0; i < 59; ++i)
        rocke_b_const_i32(&k.b, 0);
    rocke_value_t* z1 = rocke_b_zero_vec_f32(&k.b, 1);
    rocke_b_ret(&k.b);
    check(rocke_ir_builder_ok(&k.b), "the kernel did not build");
    check(std::strcmp(z16->name, "%cz16_5") == 0, std::string("16-wide: ") + z16->name);
    check(std::strcmp(z1->name, "%cz1_65") == 0, std::string("1-wide: ") + z1->name);

    char* llvm_text = NULL;
    check(rocke_lower_kernel_to_llvm(k.b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx90a", &llvm_text)
              == ROCKE_OK,
          "lowering failed");
    if(llvm_text != NULL)
    {
        check(std::strstr(llvm_text, "  %cz16_5 = select") != NULL, "%cz16_5 is not defined");
        check(std::strstr(llvm_text, "  %cz1_65 = select") != NULL, "%cz1_65 is not defined");
        std::free(llvm_text);
    }
}

void parameter_names()
{
    const char* refused_reserved[] = {"stride_0", "x_12"};
    for(const char* name : refused_reserved)
    {
        Builder k("param_name");
        const std::string error
            = refusal(&k.b, [&] { rocke_b_param(&k.b, name, rocke_i32(), NULL); });
        check(error == reserved("kernel parameter", name),
              "param '" + std::string(name) + "': '" + error + "'");
    }
    const char* refused_not_identifier[] = {"", "0x", "a.b", "a-b"};
    for(const char* name : refused_not_identifier)
    {
        Builder k("param_name");
        const std::string error
            = refusal(&k.b, [&] { rocke_b_param(&k.b, name, rocke_i32(), NULL); });
        check(error == not_identifier("kernel parameter", name),
              "param '" + std::string(name) + "': '" + error + "'");
    }
    const char* accepted[] = {"stride0", "x_y", "_", "acc_m0_n1"};
    for(const char* name : accepted)
    {
        Builder k("param_name");
        rocke_value_t* v = rocke_b_param(&k.b, name, rocke_i32(), NULL);
        check(v != NULL && v->name == "%" + std::string(name),
              "param '" + std::string(name) + "' was not accepted");
    }
}

void loop_names()
{
    {
        Builder k("iv_name");
        rocke_value_t* lo = rocke_b_const_i32(&k.b, 0);
        rocke_value_t* hi = rocke_b_const_i32(&k.b, 2);
        rocke_value_t* step = rocke_b_const_i32(&k.b, 1);
        const std::string error
            = refusal(&k.b, [&] { rocke_b_scf_for(&k.b, lo, hi, step, "k_0"); });
        check(error == reserved("loop induction variable", "k_0"), "scf_for iv: '" + error + "'");
    }
    {
        Builder k("iv_name");
        rocke_value_t* lo = rocke_b_const_i32(&k.b, 0);
        rocke_value_t* hi = rocke_b_const_i32(&k.b, 2);
        rocke_value_t* step = rocke_b_const_i32(&k.b, 1);
        rocke_iter_arg_t acc = {"acc", lo};
        const std::string error = refusal(
            &k.b, [&] { rocke_b_scf_for_iter(&k.b, lo, hi, step, &acc, 1, "k_0", false, true); });
        check(error == reserved("loop induction variable", "k_0"),
              "scf_for_iter iv: '" + error + "'");
    }
    {
        Builder k("iter_arg_name");
        rocke_value_t* lo = rocke_b_const_i32(&k.b, 0);
        rocke_value_t* hi = rocke_b_const_i32(&k.b, 2);
        rocke_value_t* step = rocke_b_const_i32(&k.b, 1);
        rocke_iter_arg_t args[2] = {{"acc", lo}, {"acc_1", lo}};
        const std::string error = refusal(
            &k.b, [&] { rocke_b_scf_for_iter(&k.b, lo, hi, step, args, 2, "k0", false, true); });
        check(error == reserved("loop-carried value", "acc_1"), "iter arg: '" + error + "'");
    }
    {
        Builder k("iter_arg_name");
        rocke_value_t* lo = rocke_b_const_i32(&k.b, 0);
        rocke_value_t* hi = rocke_b_const_i32(&k.b, 2);
        rocke_value_t* step = rocke_b_const_i32(&k.b, 1);
        rocke_iter_arg_t arg = {"acc.1", lo};
        const std::string error = refusal(
            &k.b, [&] { rocke_b_scf_for_iter(&k.b, lo, hi, step, &arg, 1, "k0", false, true); });
        check(error == not_identifier("loop-carried value", "acc.1"), "iter arg: '" + error + "'");
    }
}

void hints()
{
    const char* refused[] = {"", "16x"};
    for(const char* hint : refused)
    {
        Builder k("hint");
        const rocke_type_t* i32 = rocke_i32();
        const std::string error = refusal(&k.b, [&] {
            rocke_b_op(&k.b, ROCKE_OP_ARITH_CONSTANT, NULL, 0, &i32, 1, NULL, NULL, 0, hint, NULL);
        });
        check(error == bad_hint(hint), "hint '" + std::string(hint) + "': '" + error + "'");
    }
}

} // namespace

int main()
{
    a_hint_ending_in_digits_does_not_run_into_the_counter();
    parameter_names();
    loop_names();
    hints();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("value names: all checks passed\n");
    return 0;
}
