// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/future_intrinsic_lowering.cpp -- C++ parity for the Python
 * TestHelpers::test_future_intrinsic_lowering gate.
 *
 * Builds a tiny kernel that exercises the new cross-lane / async-marker IR ops
 * and asserts the lowered LLVM IR contains the expected @llvm.amdgcn.* declares.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
        {                                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

static bool ll_contains(const char* haystack, const char* needle)
{
    return haystack && needle && std::strstr(haystack, needle) != nullptr;
}

int main(void)
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "future_intrinsics_smoke") == ROCKE_OK, "init");

    rocke_value_t* out_p
        = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), nullptr);
    rocke_value_t* tid = rocke_b_thread_id_x(&b);
    rocke_value_t* x = rocke_b_const_f32(&b, 1.0);
    rocke_value_t* xi = rocke_b_const_i32(&b, 1);

    rocke_value_t* rmax = rocke_b_wave_reduce(&b, x, "fmax", 0);
    rocke_value_t* radd = rocke_b_wave_reduce(&b, xi, "add", 0);
    rocke_value_t* dpp8 = rocke_b_mov_dpp8(&b, xi, 0x765432);
    rocke_value_t* sw = rocke_b_ds_swizzle_xor(&b, xi, 1);
    rocke_value_t* sw2 = rocke_b_ds_swizzle(&b, xi, 0x041F);
    rocke_value_t* rl = rocke_b_readlane(&b, xi, rocke_b_const_i32(&b, 0));
    rocke_value_t* wl = rocke_b_writelane(&b, xi, rocke_b_const_i32(&b, 0), xi);
    rocke_value_t* pl = rocke_b_permlane64(&b, xi);
    rocke_value_t* ab = rocke_b_alignbyte(&b, xi, xi, rocke_b_const_i32(&b, 8));
    rocke_value_t* wqm = rocke_b_s_wqm(&b, xi);
    rocke_value_t* lo = nullptr;
    rocke_value_t* hi = nullptr;
    rocke_b_permlane32_swap(&b, xi, xi, &lo, &hi);

    rocke_b_asyncmark(&b);
    rocke_b_wait_asyncmark(&b, 0);

    rocke_b_global_store(&b, out_p, tid, rmax, 4);
    (void)radd;
    (void)dpp8;
    (void)sw;
    (void)sw2;
    (void)rl;
    (void)wl;
    (void)pl;
    (void)ab;
    (void)wqm;
    (void)lo;
    (void)hi;

    rocke_b_ret(&b);

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &ll);
    CHECK(st == ROCKE_OK && ll != nullptr, "rocke_lower_kernel_to_llvm");

    static const char* const needles[] = {
        "@llvm.amdgcn.wave.reduce.fmax.f32",
        "@llvm.amdgcn.wave.reduce.add.i32",
        "@llvm.amdgcn.mov.dpp8.i32",
        "@llvm.amdgcn.ds.swizzle",
        "@llvm.amdgcn.readlane.i32",
        "@llvm.amdgcn.writelane.i32",
        "@llvm.amdgcn.permlane64",
        "@llvm.amdgcn.alignbyte",
        "@llvm.amdgcn.s.wqm.i32",
        "@llvm.amdgcn.asyncmark",
        "@llvm.amdgcn.wait.asyncmark",
        "@llvm.amdgcn.permlane32.swap",
    };
    for(const char* needle : needles)
        CHECK(ll_contains(ll, needle), needle);

    std::free(ll);
    rocke_ir_builder_free(&b);
    return g_failures ? 1 : 0;
}
