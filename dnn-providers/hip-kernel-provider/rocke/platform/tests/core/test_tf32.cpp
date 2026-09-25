// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "rocke/error.hpp"
#include "rocke/instance_tf32_mma_probe.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

int main()
{
    CHECK(!rocke_type_eq(rocke_tf32(), rocke_i32()));
    CHECK(!rocke_type_eq(rocke_tf32(), rocke_f32()));
    CHECK(rocke_scalar_by_name("tf32") == rocke_tf32());
    const char* modes[] = {"raw", "carrier", "rne", "prepacked", "fp32"};
    for(int m : {16, 32})
    {
        for(const char* mode : modes)
        {
            rocke_ir_builder_t b;
            auto* kernel = rocke_build_tf32_mma_probe(&b, m, mode);
            CHECK(kernel);
            rocke_diag_t* diagnostics = nullptr;
            size_t count = 0;
            rocke_verify(kernel, &diagnostics, &count);
            CHECK(count == 0);
            rocke_diags_free(diagnostics, count);
            rocke_lower_hip_opts_t opts = {};
            opts.arch = "gfx942";
            rocke_strbuf_t hip;
            CHECK(rocke_strbuf_init(&hip, 0) == 0);
            CHECK(rocke_lower_kernel_to_hip(&b, kernel, &opts, &hip) == ROCKE_OK);
            CHECK(std::strstr(rocke_strbuf_cstr(&hip), "__builtin_amdgcn_mfma_f32_"));
            rocke_strbuf_free(&hip);
            if(std::strcmp(mode, "fp32") != 0)
            {
                char* ll = nullptr;
                CHECK(rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &ll)
                      != ROCKE_OK);
                std::free(ll);
            }
            rocke_ir_builder_free(&b);
        }
    }
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "invalid") == ROCKE_OK);
    auto* integer = rocke_b_const_i32(&b, 1);
    try
    {
        rocke_b_cvt_f32_to_tf32(&b, integer);
        CHECK(false);
    }
    catch(const ckc::Error& error)
    {
        CHECK(error.code() == ROCKE_ERR_VALUE);
    }
    rocke_ir_builder_free(&b);
    return 0;
}
