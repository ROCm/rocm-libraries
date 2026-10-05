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

static int test_invalid_tf32_ops_rejected()
{
    for(int variant = 0; variant < 8; ++variant)
    {
        rocke_ir_builder_t b, parsed;
        CHECK(rocke_ir_builder_init(&b, "invalid_tf32_vector") == ROCKE_OK);
        CHECK(rocke_ir_builder_init(&parsed, "parsed") == ROCKE_OK);
        auto* elem = variant == 5 ? rocke_i32() : rocke_tf32();
        auto* v = rocke_b_param(&b, "v", rocke_vector_type(&b, elem, 2), nullptr);
        switch(variant)
        {
        case 0:
            rocke_b_vector_add(&b, v, v);
            break;
        case 1:
            rocke_b_vector_mul(&b, v, v);
            break;
        case 2:
            rocke_b_vector_sum(&b, v);
            break;
        case 3:
            rocke_b_vector_reduce_max(&b, v);
            break;
        case 4:
            rocke_b_vector_cmp(&b, "lt", v, v);
            break;
        case 5:
            rocke_b_vector_trunc(&b, v, rocke_tf32());
            break;
        }
        if(variant >= 6)
        {
            rocke_ir_builder_free(&b);
            auto* probe = rocke_build_tf32_mma_probe(&b, variant == 6 ? 16 : 32, "fp32");
            CHECK(probe);
            for(int i = 0; i < probe->body->num_ops; ++i)
            {
                auto* op = probe->body->ops[i];
                if(std::strcmp(op->name, "tile.mma") == 0)
                    op->results[0]->type
                        = rocke_vector_type(&b, rocke_tf32(), variant == 6 ? 4 : 16);
            }
        }
        else
            rocke_b_ret(&b);
        const char* message = variant >= 6 ? "MMA results must not use TF32" : "TF32 arithmetic";
        char* text = nullptr;
        CHECK(rocke_ir_serialize(b.kernel, &text) == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        CHECK(rocke_ir_parse(text, &parsed, &kernel) == ROCKE_OK);
        std::free(text);
        rocke_diag_t* diagnostics = nullptr;
        size_t count = 0;
        CHECK(rocke_verify(kernel, &diagnostics, &count) == ROCKE_OK);
        bool rejected = false;
        for(size_t i = 0; i < count; ++i)
            rejected |= std::strstr(diagnostics[i].message, message) != nullptr;
        CHECK(rejected);
        rocke_diags_free(diagnostics, count);
        for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
        {
            char* ll = nullptr;
            auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
            CHECK(rocke_lower_kernel_to_llvm(kernel, flavor, "gfx942", &ll) == ROCKE_ERR_VALUE);
            std::free(ll);
        }
        rocke_strbuf_t hip;
        CHECK(rocke_strbuf_init(&hip, 0) == 0);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx942";
        CHECK(rocke_lower_kernel_to_hip(&parsed, kernel, &opts, &hip) == ROCKE_ERR_VALUE);
        rocke_strbuf_free(&hip);
        rocke_ir_builder_free(&parsed);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_vector_load()
{
    for(int n : {0, 1, 2, 3, 4, 6, 8, 16})
        for(int alignment : {0, 4})
        {
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "tf32_vector_load") == ROCKE_OK);
            auto* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_tf32(), "global"), nullptr);
            auto* index = rocke_b_const_i32(&b, 0);
            if(n != 2 && n != 3 && n != 4 && n != 8)
            {
                try
                {
                    rocke_b_global_load_vN(&b, p, index, rocke_tf32(), n, alignment);
                    CHECK(false);
                }
                catch(const ckc::Error& error)
                {
                    CHECK(error.code() == ROCKE_ERR_VALUE);
                    CHECK(std::strstr(error.what(), "unsupported vector width for tf32"));
                }
                rocke_ir_builder_free(&b);
                continue;
            }
            auto* values = rocke_b_global_load_vN(&b, p, index, rocke_tf32(), n, alignment);
            CHECK(values);
            rocke_b_global_store(&b, p, index, rocke_b_vec_extract(&b, values, n - 1), 4);
            rocke_b_ret(&b);
            int expected_align = alignment ? alignment : (n == 3 ? 4 : n * 4);
            for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
            {
                char* ll = nullptr;
                auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
                CHECK(rocke_lower_kernel_to_llvm(b.kernel, flavor, "gfx942", &ll) == ROCKE_OK);
                char load[64], align[32];
                std::snprintf(load, sizeof(load), "load <%d x i32>", n);
                std::snprintf(align, sizeof(align), "align %d", expected_align);
                CHECK(std::strstr(ll, load));
                CHECK(std::strstr(ll, align));
                std::free(ll);
            }
            rocke_strbuf_t hip;
            CHECK(rocke_strbuf_init(&hip, 0) == 0);
            rocke_lower_hip_opts_t opts = {};
            opts.arch = "gfx942";
            CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
            if(expected_align < n * 4 || n == 3)
            {
                char size[32];
                std::snprintf(size, sizeof(size), "%d);", n * 4);
                CHECK(std::strstr(hip.data, "__builtin_memcpy"));
                CHECK(std::strstr(hip.data, size));
            }
            rocke_strbuf_free(&hip);
            rocke_ir_builder_free(&b);
        }
    return 0;
}

static int test_invalid_store_alignment()
{
    for(int align : {0, -4, 3, 12})
    {
        rocke_ir_builder_t b, parsed;
        CHECK(rocke_ir_builder_init(&b, "invalid_store_alignment") == ROCKE_OK);
        CHECK(rocke_ir_builder_init(&parsed, "parsed") == ROCKE_OK);
        auto* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_tf32(), "global"), nullptr);
        auto* zero = rocke_b_const_i32(&b, 0);
        auto* value = rocke_b_bitcast(&b, zero, rocke_tf32());
        rocke_value_t* lanes[] = {value, value, value, value};
        auto* values = rocke_b_vec_pack(&b, lanes, 4, rocke_tf32());
        rocke_b_global_store_vN(&b, p, zero, values, 4, align);
        auto* body = b.kernel->body;
        rocke_attr_set_int(&b, &body->ops[body->num_ops - 1]->attrs, "align", align);
        rocke_b_ret(&b);
        char* text = nullptr;
        CHECK(rocke_ir_serialize(b.kernel, &text) == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        CHECK(rocke_ir_parse(text, &parsed, &kernel) == ROCKE_OK);
        std::free(text);
        for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
        {
            char* ll = nullptr;
            auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
            CHECK(rocke_lower_kernel_to_llvm(kernel, flavor, "gfx942", &ll) == ROCKE_ERR_VALUE);
            std::free(ll);
        }
        rocke_strbuf_t hip;
        CHECK(rocke_strbuf_init(&hip, 0) == 0);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx942";
        CHECK(rocke_lower_kernel_to_hip(&parsed, kernel, &opts, &hip) == ROCKE_ERR_VALUE);
        rocke_strbuf_free(&hip);
        rocke_ir_builder_free(&parsed);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

int main()
{
    CHECK(test_invalid_store_alignment() == 0);
    CHECK(test_invalid_tf32_ops_rejected() == 0);
    CHECK(test_vector_load() == 0);
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
    // Mirror test_global_vector_store: native authoring must accept the same widths.
    for(int n : {1, 2, 4, 8, 16})
        for(int alignment : {0, 4, 64})
        {
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "tf32_vector_store") == ROCKE_OK);
            auto* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_tf32(), "global"), nullptr);
            auto* index = rocke_b_const_i32(&b, 0);
            auto* value = rocke_b_bitcast(&b, index, rocke_tf32());
            rocke_value_t* components[16];
            for(int i = 0; i < n; ++i)
                components[i] = value;
            auto* values = rocke_b_vec_pack(&b, components, n, rocke_tf32());
            if(n == 16)
            {
                try
                {
                    rocke_b_global_store_vN(&b, p, index, values, n, alignment);
                    CHECK(false);
                }
                catch(const ckc::Error& error)
                {
                    CHECK(error.code() == ROCKE_ERR_VALUE);
                    CHECK(std::strstr(error.what(), "n=16 not supported for tf32"));
                }
            }
            else
            {
                rocke_b_global_store_vN(&b, p, index, values, n, alignment);
                rocke_b_ret(&b);
                for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
                {
                    char* ll = nullptr;
                    auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
                    CHECK(rocke_lower_kernel_to_llvm(
                              rocke_ir_builder_kernel(&b), flavor, "gfx950", &ll)
                          == ROCKE_OK);
                    char store[64], align[32];
                    std::snprintf(store, sizeof(store), "store <%d x i32>", n);
                    std::snprintf(align, sizeof(align), "align %d", alignment ? alignment : n * 4);
                    CHECK(std::strstr(ll, store));
                    CHECK(std::strstr(ll, align));
                    std::free(ll);
                }
                rocke_strbuf_t hip;
                CHECK(rocke_strbuf_init(&hip, 0) == 0);
                rocke_lower_hip_opts_t opts = {};
                opts.arch = "gfx942";
                CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
                if(alignment && alignment < n * 4)
                {
                    char size[32];
                    std::snprintf(size, sizeof(size), ", %d);", n * 4);
                    CHECK(std::strstr(hip.data, "__builtin_memcpy(__builtin_assume_aligned("));
                    CHECK(std::strstr(hip.data, size));
                    CHECK(!std::strstr(hip.data, "*reinterpret_cast<"));
                }
                else
                    CHECK(std::strstr(hip.data, "*reinterpret_cast<"));
                rocke_strbuf_free(&hip);
            }
            rocke_ir_builder_free(&b);
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
