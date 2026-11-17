// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_loop_carried_descriptor.cpp -- the C++ builder and verifier
 * refuse a buffer descriptor as a loop-carried value.
 *
 * A descriptor is <4 x i32> in the IR but an opaque ptr addrspace(8) (HIP
 * rsrc_t) once lowered, so neither the phi nor the HIP variable that holds an
 * scf.for iter_arg can hold one. The builder refuses a descriptor as an
 * iter_arg's initial value and as a yielded value; the verifier refuses the
 * same IR when it was put together without the builder. Both report the
 * Python engine's text. tests/core/test_loop_carried_descriptor.py covers the
 * Python engine and both engines' lowerings.
 */
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
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

std::string message(int index)
{
    return "scf.for iter_arg " + std::to_string(index)
           + " is a buffer descriptor, which a loop cannot carry; build the descriptor before "
             "the loop and advance the offset, or rebuild it inside the loop body";
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
    rocke_value_t* descriptor = nullptr;
    rocke_for_t loop{};

    explicit Kernel(const char* name)
    {
        if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", name);
            std::exit(1);
        }
        const rocke_type_t* ptr = rocke_ptr_type(&b, rocke_f16(), "global");
        rocke_value_t* x = rocke_b_param(&b, "X", ptr, NULL);
        descriptor = rocke_b_buffer_rsrc(&b, x, rocke_b_const_i32(&b, 64));
    }
    ~Kernel()
    {
        rocke_ir_builder_free(&b);
    }

    /* A loop over [0, 2) carrying a plain <4 x i32>, not a descriptor. */
    void build_vector_carry_loop()
    {
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        rocke_iter_arg_t carry = {"carry", rocke_b_vector_splat(&b, zero, 4)};
        loop = rocke_b_scf_for_iter(&b,
                                    rocke_b_const_i32(&b, 0),
                                    rocke_b_const_i32(&b, 2),
                                    rocke_b_const_i32(&b, 1),
                                    &carry,
                                    1,
                                    "k0",
                                    false,
                                    true);
        rocke_b_region_enter(&b, loop.body);
        rocke_b_scf_yield(&b, loop.iter_vars, 1);
        rocke_b_region_leave(&b);
        rocke_b_ret(&b);
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
};

void builder_refuses_a_descriptor_init()
{
    Kernel k("descriptor_init");
    rocke_value_t* lo = rocke_b_const_i32(&k.b, 0);
    rocke_value_t* hi = rocke_b_const_i32(&k.b, 2);
    rocke_value_t* step = rocke_b_const_i32(&k.b, 1);
    rocke_iter_arg_t args[2] = {{"count", lo}, {"r", k.descriptor}};
    const std::string error = refusal(
        &k.b, [&] { rocke_b_scf_for_iter(&k.b, lo, hi, step, args, 2, "k0", false, true); });
    check(error == message(1), "builder, descriptor init: '" + error + "'");
}

void builder_refuses_a_yielded_descriptor()
{
    Kernel k("descriptor_yield");
    rocke_value_t* zero = rocke_b_const_i32(&k.b, 0);
    rocke_iter_arg_t carry = {"carry", rocke_b_vector_splat(&k.b, zero, 4)};
    rocke_for_t loop = rocke_b_scf_for_iter(&k.b,
                                            rocke_b_const_i32(&k.b, 0),
                                            rocke_b_const_i32(&k.b, 2),
                                            rocke_b_const_i32(&k.b, 1),
                                            &carry,
                                            1,
                                            "k0",
                                            false,
                                            true);
    rocke_b_region_enter(&k.b, loop.body);
    const std::string error = refusal(&k.b, [&] { rocke_b_scf_yield(&k.b, &k.descriptor, 1); });
    check(error == message(0), "builder, yielded descriptor: '" + error + "'");
}

void verifier_accepts_a_plain_vector_carry()
{
    Kernel k("vector_carry");
    k.build_vector_carry_loop();
    check(rocke_ir_builder_ok(&k.b), "plain vector carry did not build");
    check(k.diagnostics().empty(), "verifier refused a plain vector carry: " + k.diagnostics());
}

/* Swap the descriptor into the built loop's init operand or its yield, the way
 * IR can reach the verifier without the builder. */
void verifier_refuses_a_carried_descriptor(bool in_yield)
{
    Kernel k("descriptor_carry");
    k.build_vector_carry_loop();
    if(in_yield)
    {
        rocke_region_t* body = k.loop.body;
        body->ops[body->num_ops - 1]->operands[0] = k.descriptor;
    }
    else
    {
        k.loop.op->operands[3] = k.descriptor;
    }
    const std::string diags = k.diagnostics();
    check(diags.find(message(0) + "\n") != std::string::npos,
          std::string("verifier, descriptor in the ") + (in_yield ? "yield" : "init") + ": "
              + diags);
}

} // namespace

int main()
{
    builder_refuses_a_descriptor_init();
    builder_refuses_a_yielded_descriptor();
    verifier_accepts_a_plain_vector_carry();
    verifier_refuses_a_carried_descriptor(false);
    verifier_refuses_a_carried_descriptor(true);
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("loop-carried descriptor: all checks passed\n");
    return 0;
}
