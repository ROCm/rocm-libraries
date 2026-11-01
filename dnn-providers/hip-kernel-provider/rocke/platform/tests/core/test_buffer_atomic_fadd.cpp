// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_buffer_atomic_fadd.cpp -- the C++ side of
 * tests/core/test_buffer_atomic_fadd.py for tile.buffer_atomic_fadd_f32.
 *
 * The byte-identity gate compares the kernels both engines build, so it does
 * not see the builder's refusal of a cache-policy immediate outside 0..31 (no
 * kernel is built). This checks, as the Python test does, that:
 *   - the op lowers to the no-return llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32
 *     call and its declaration, carrying the cache policy as the last operand;
 *   - a cache policy of 32 is refused with a ValueError.
 */
#include <cstdio>
#include <cstdlib>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

namespace
{

int g_failures = 0;

void fail(const std::string& what)
{
    fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

/* The Python test's _kernel(cachepolicy): one f32 atomic add per lane into a
 * buffer bounded by nbytes. */
void build(rocke_ir_builder_t* b, int cachepolicy)
{
    const rocke_type_t* ptr = rocke_ptr_type(b, rocke_f32(), "global");
    rocke_value_t* out = rocke_b_param(b, "out", ptr, NULL);
    rocke_value_t* nbytes = rocke_b_param(b, "nbytes", rocke_i32(), NULL);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, out, nbytes);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* off = rocke_b_mul(b, tid, rocke_b_const_i32(b, 4));
    rocke_value_t* soff = rocke_b_const_i32(b, 0);
    rocke_value_t* one = rocke_b_const_f32(b, 1.0);
    rocke_b_buffer_atomic_fadd_f32(b, rsrc, off, soff, one, cachepolicy);
    rocke_b_ret(b);
}

void check_lowering(int cachepolicy)
{
    const std::string tag = "cachepolicy " + std::to_string(cachepolicy) + ": ";
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "buffer_atomic_fadd") != ROCKE_OK)
    {
        fail(tag + "builder init failed");
        return;
    }
    build(&b, cachepolicy);
    if(!rocke_ir_builder_ok(&b))
    {
        fail(tag + "builder error: " + rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return;
    }
    char* text = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = 0;
    if(rocke_lower_kernel_to_llvm_ex(
           rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_LLVM23, "gfx1250", &text, err, sizeof err)
           != ROCKE_OK
       || text == nullptr)
    {
        fail(tag + "LLVM lowering failed: " + err);
    }
    else
    {
        const std::string ll(text);
        if(ll.find("declare float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(")
           == std::string::npos)
            fail(tag + "no declaration of the buffer atomic intrinsic");
        if(ll.find("call float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float ")
           == std::string::npos)
            fail(tag + "no call of the buffer atomic intrinsic");
        if(ll.find(", i32 " + std::to_string(cachepolicy) + ")") == std::string::npos)
            fail(tag + "the call does not carry the cache policy");
    }
    free(text);
    rocke_ir_builder_free(&b);
}

void check_refuses_out_of_range_cachepolicy()
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "bad_cp") != ROCKE_OK)
    {
        fail("builder init failed");
        return;
    }
    /* A builder refusal throws a ckc::ValueError. */
    bool refused = false;
    try
    {
        build(&b, 32);
    }
    catch(const ckc::Error& error)
    {
        refused = error.code() == ROCKE_ERR_VALUE;
        if(!refused)
            fail("cachepolicy 32 was refused, but not with a ValueError");
    }
    if(!refused && rocke_ir_builder_ok(&b))
        fail("cachepolicy 32 was accepted");
    rocke_ir_builder_free(&b);
}

} // namespace

int main()
{
    check_lowering(0);
    /* CPOL_ATOMIC_CASCADE_DEV: th = TH_ATOMIC_CASCADE (4), scope = SCOPE_DEV (2) << 3. */
    check_lowering(20);
    check_refuses_out_of_range_cachepolicy();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("buffer_atomic_fadd: OK\n");
    return 0;
}
