// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native barrier admission, serialization, and LLVM lowering contracts. */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

#define CHECK(x)                                                       \
    do                                                                 \
    {                                                                  \
        if(!(x))                                                       \
        {                                                              \
            fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); \
            return 1;                                                  \
        }                                                              \
    } while(0)

static int test_llvm_roundtrip(rocke_ir_builder_t* b, const char* arch)
{
    char* serialized = nullptr;
    CHECK(rocke_ir_serialize(b->kernel, &serialized) == ROCKE_OK);
    rocke_ir_builder_t copy;
    CHECK(rocke_ir_builder_init(&copy, "barrier_import") == ROCKE_OK);
    rocke_kernel_def_t* kernel = nullptr;
    CHECK(rocke_ir_parse(serialized, &copy, &kernel) == ROCKE_OK);
    CHECK(kernel);
    free(serialized);
    for(int index = 0; index < rocke_llvm_flavor_count(); ++index)
    {
        auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(index));
        char* original = nullptr;
        char* imported = nullptr;
        CHECK(rocke_lower_kernel_to_llvm(b->kernel, flavor, arch, &original) == ROCKE_OK);
        CHECK(rocke_lower_kernel_to_llvm(kernel, flavor, arch, &imported) == ROCKE_OK);
        CHECK(original && imported && strcmp(original, imported) == 0);
        CHECK(strstr(original, "asm \"\", \"=v,0\"") != nullptr);
        CHECK(strstr(original, "asm sideeffect") == nullptr);
        free(original);
        free(imported);
    }
    rocke_ir_builder_free(&copy);
    return 0;
}

static int test_admission()
{
    const rocke_type_t* logical[] = {rocke_fp4e2m1(),
                                     rocke_fp6e2m3(),
                                     rocke_fp6e3m2(),
                                     rocke_e8m0(),
                                     rocke_e5m3(),
                                     rocke_tf32()};
    for(int index = 0; index < 9; ++index)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "barrier_rejection") == ROCKE_OK);
        const auto* type = index < 6    ? logical[index]
                           : index == 6 ? rocke_ptr_type(&b, rocke_f32(), "global")
                                        : rocke_vector_type(&b, rocke_f32(), 2);
        auto* value = index == 8 ? nullptr : rocke_b_param(&b, "value", type, nullptr);
        int before = b.kernel->body->num_ops;
        bool rejected = false;
        try
        {
            rocke_b_optimization_barrier(&b, value);
        }
        catch(const ckc::Error& error)
        {
            rejected = error.code() == ROCKE_ERR_VALUE
                       && strcmp(error.what(),
                                 "optimization_barrier requires a directly lowerable scalar or i1")
                              == 0;
        }
        CHECK(rejected);
        CHECK(b.kernel->body->num_ops == before);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_scalars()
{
    const rocke_type_t* types[] = {rocke_i1(),
                                   rocke_i8(),
                                   rocke_i16(),
                                   rocke_i32(),
                                   rocke_i64(),
                                   rocke_bf16(),
                                   rocke_f16(),
                                   rocke_f32(),
                                   rocke_fp8e4m3(),
                                   rocke_bf8e5m2()};
    const char* arches[] = {"gfx950", "gfx1250"};
    for(const char* arch : arches)
        for(const auto* type : types)
        {
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "barrier_scalar") == ROCKE_OK);
            auto* ptr = rocke_b_param(&b, "p", rocke_ptr_type(&b, type, "global"), nullptr);
            auto* tid = rocke_b_thread_id_x(&b);
            auto* value = rocke_b_global_load(&b, ptr, tid, type, 1);
            auto* result = rocke_b_optimization_barrier(&b, value);
            CHECK(result && rocke_type_eq(result->type, type));
            rocke_b_global_store(&b, ptr, tid, result, 1);
            rocke_b_ret(&b);
            CHECK(test_llvm_roundtrip(&b, arch) == 0);
            rocke_ir_builder_free(&b);
        }
    return 0;
}

static int test_predicate()
{
    const char* arches[] = {"gfx950", "gfx1250"};
    for(const char* arch : arches)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "barrier_predicate") == ROCKE_OK);
        auto* ptr = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_i32(), "global"), nullptr);
        auto* tid = rocke_b_thread_id_x(&b);
        auto* value = rocke_b_global_load(&b, ptr, tid, rocke_i32(), 1);
        auto* predicate = rocke_b_cmp_lt(&b, value, rocke_b_const_i32(&b, 0));
        auto* result = rocke_b_optimization_barrier(&b, predicate);
        CHECK(result && rocke_type_eq(result->type, rocke_i1()));
        rocke_b_global_store(&b, ptr, tid, rocke_b_zext(&b, result, rocke_i32()), 1);
        rocke_b_ret(&b);
        CHECK(test_llvm_roundtrip(&b, arch) == 0);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

int main()
{
    int admission = test_admission();
    int scalars = test_scalars();
    int predicate = test_predicate();
    return admission || scalars || predicate;
}
