// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native barrier admission, serialization, and LLVM/HIP lowering contracts. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
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

static int test_tensor_epilogue_hip()
{
    const char* arches[] = {"gfx950", "gfx1250"};
    const bool clobbers[] = {false, true};
    const rocke_type_t* types[] = {rocke_bf16(), rocke_f16()};
    for(const char* arch : arches)
        for(const auto* type : types)
            for(bool explicit_clobber : clobbers)
            {
                rocke_ir_builder_t b;
                CHECK(rocke_ir_builder_init(&b, "tensor_epilogue") == ROCKE_OK);
                auto* output
                    = rocke_b_param(&b, "output", rocke_ptr_type(&b, type, "global"), nullptr);
                auto* acc = rocke_b_param(&b, "acc", rocke_f32(), nullptr);
                auto* factor_a = rocke_b_param(&b, "factor_a", rocke_f32(), nullptr);
                auto* factor_b = rocke_b_param(&b, "factor_b", rocke_f32(), nullptr);
                auto* factors = rocke_b_fmul(&b, factor_a, factor_b);
                auto* scaled = rocke_b_fmul(&b, acc, factors);
                auto* opaque = rocke_b_optimization_barrier(&b, scaled);
                CHECK(opaque);
                if(explicit_clobber)
                    rocke_attr_set_str(&b, &opaque->op->attrs, "clobber", "");
                auto* tid = rocke_b_thread_id_x(&b);
                auto* converted = rocke_b_cast_f32_to(&b, opaque, type);
                rocke_b_global_store(&b, output, tid, converted, 2);
                rocke_b_ret(&b);
                // The tensor epilogue has the same barrier contract in both routes.
                CHECK(test_llvm_roundtrip(&b, arch) == 0);

                char* serialized = nullptr;
                CHECK(rocke_ir_serialize(b.kernel, &serialized) == ROCKE_OK);
                rocke_ir_builder_t copy;
                CHECK(rocke_ir_builder_init(&copy, "imported_epilogue") == ROCKE_OK);
                rocke_kernel_def_t* kernel = nullptr;
                CHECK(rocke_ir_parse(serialized, &copy, &kernel) == ROCKE_OK);
                CHECK(kernel);
                free(serialized);

                rocke_lower_hip_opts_t opts = {};
                opts.arch = arch;
                rocke_strbuf_t original = {}, imported = {};
                CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &original) == ROCKE_OK);
                CHECK(rocke_lower_kernel_to_hip(&copy, kernel, &opts, &imported) == ROCKE_OK);
                CHECK(original.cstr() && imported.cstr());
                CHECK(strcmp(original.cstr(), imported.cstr()) == 0);
                CHECK(strstr(original.cstr(), "factor_a * factor_b"));
                CHECK(strstr(original.cstr(), "asm (\"\" : \"=v\""));
                CHECK(!strstr(original.cstr(), "asm volatile"));
                CHECK(!strstr(original.cstr(), " : \"memory\""));
                original.free_buffer();
                imported.free_buffer();
                rocke_ir_builder_free(&copy);
                rocke_ir_builder_free(&b);
            }
    return 0;
}

static int test_hip_rejects_malformed_attributes()
{
    for(int index = 0; index < 13; ++index)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "invalid_hip_asm") == ROCKE_OK);
        auto* value = rocke_b_param(&b, "value", rocke_f32(), nullptr);
        auto* result = rocke_b_optimization_barrier(&b, value);
        const char* key = index < 7 ? "clobber" : "sideeffect";
        int kind = index < 7 ? index : index - 7 + 2;
        switch(kind)
        {
        case 0:
            rocke_attr_set_bool(&b, &result->op->attrs, key, false);
            break;
        case 1:
            rocke_attr_set_bool(&b, &result->op->attrs, key, true);
            break;
        case 2:
            rocke_attr_set_int(&b, &result->op->attrs, key, 0);
            break;
        case 3:
            rocke_attr_set_int(&b, &result->op->attrs, key, 1);
            break;
        case 4:
            rocke_attr_set_float(&b, &result->op->attrs, key, 0.0);
            break;
        case 5:
            rocke_attr_set_float(&b, &result->op->attrs, key, 1.0);
            break;
        case 6:
            rocke_attr_set_str(&b, &result->op->attrs, key, "memory");
            break;
        case 7:
            rocke_attr_set_str(&b, &result->op->attrs, key, "");
            break;
        }
        const char* diagnostic = index < 7
                                     ? "HIP optimization barrier clobber must be an empty string"
                                     : "HIP inline asm sideeffect must be a boolean";
        char* serialized = nullptr;
        CHECK(rocke_ir_serialize(b.kernel, &serialized) == ROCKE_OK);
        rocke_ir_builder_t copy;
        CHECK(rocke_ir_builder_init(&copy, "imported_invalid_asm") == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        CHECK(rocke_ir_parse(serialized, &copy, &kernel) == ROCKE_OK);
        free(serialized);
        if(index >= 7)
        {
            const rocke_llvm_flavor_t flavors[] = {ROCKE_LLVM_FLAVOR_LLVM20,
                                                   ROCKE_LLVM_FLAVOR_LLVM22,
                                                   ROCKE_LLVM_FLAVOR_LLVM23,
                                                   ROCKE_LLVM_FLAVOR_LLVM24};
            for(auto flavor : flavors)
            {
                for(auto* candidate : {b.kernel, kernel})
                {
                    char* llvm = nullptr;
                    char err[ROCKE_ERR_MSG_CAP] = {};
                    CHECK(rocke_lower_kernel_to_llvm_ex(
                              candidate, flavor, "gfx1250", &llvm, err, sizeof err)
                          == ROCKE_ERR_VALUE);
                    CHECK(strcmp(err, "LLVM inline asm sideeffect must be a boolean") == 0);
                    free(llvm);
                }
            }
        }
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx1250";
        rocke_strbuf_t original = {}, imported = {};
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &original) == ROCKE_ERR_VALUE);
        CHECK(rocke_lower_kernel_to_hip(&copy, kernel, &opts, &imported) == ROCKE_ERR_VALUE);
        CHECK(strcmp(rocke_ir_builder_error(&b), diagnostic) == 0);
        CHECK(strcmp(rocke_ir_builder_error(&copy), diagnostic) == 0);
        original.free_buffer();
        imported.free_buffer();
        rocke_ir_builder_free(&copy);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_hip_asm_string_roundtrip()
{
    for(bool output : {false, true})
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "asm_string") == ROCKE_OK);
        auto* value = output ? rocke_b_param(&b, "value", rocke_f32(), nullptr) : nullptr;
        rocke_value_t* operands[] = {value};
        const rocke_type_t* types[] = {rocke_f32()};
        const char* prefix = output ? "v_mov_b32 $0, $1" : "s_nop 0";
        char tmpl[128];
        snprintf(tmpl, sizeof tmpl, "%s\n\ts_nop 0 ; \"quoted\" \\path\r\0017\177", prefix);
        CHECK(rocke_b_inline_asm(&b,
                                 tmpl,
                                 output ? "=v,0" : "",
                                 operands,
                                 output ? 1 : 0,
                                 types,
                                 output ? 1 : 0,
                                 nullptr));
        rocke_b_ret(&b);
        char* serialized = nullptr;
        CHECK(rocke_ir_serialize(b.kernel, &serialized) == ROCKE_OK);
        rocke_ir_builder_t copy;
        CHECK(rocke_ir_builder_init(&copy, "imported_asm") == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        CHECK(rocke_ir_parse(serialized, &copy, &kernel) == ROCKE_OK);
        free(serialized);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx950";
        rocke_strbuf_t original = {}, imported = {};
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &original) == ROCKE_OK);
        CHECK(rocke_lower_kernel_to_hip(&copy, kernel, &opts, &imported) == ROCKE_OK);
        CHECK(strcmp(original.cstr(), imported.cstr()) == 0);
        CHECK(strstr(original.cstr(), "\\n\\ts_nop 0 ; \\\"quoted\\\" \\\\path\\r\\0017\\177"));
        CHECK(!strstr(original.cstr(), "s_nop 0 ; \"quoted\""));
        original.free_buffer();
        imported.free_buffer();
        rocke_ir_builder_free(&copy);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_hip_constraint_rejection()
{
    struct rejection_t
    {
        const char* constraints;
        int outputs;
        int inputs;
        const char* diagnostic;
    };
    const rejection_t cases[] = {
        {"=v", 1, 1, "HIP inline asm inputs failed - constraints=0, expected operands=1"},
        {"=v,0,v", 1, 1, "HIP inline asm inputs failed - constraints=2, expected operands=1"},
        {"", 0, 1, "HIP inline asm inputs failed - constraints=0, expected operands=1"},
        {"v", 0, 0, "HIP inline asm inputs failed - constraints=1, expected operands=0"},
        {"=v", 0, 0, "HIP inline asm outputs failed - constraints=1, expected results=0"},
        {"=v,=s", 1, 0, "HIP inline asm outputs failed - constraints=2, expected results=1"},
        {"v", 1, 1, "HIP inline asm requires an output constraint"},
        {"", 1, 0, "HIP inline asm requires an output constraint"},
        {"=v,,0", 1, 1, nullptr},
        {"=v,0,", 1, 1, nullptr},
        {", ", 0, 0, nullptr},
        {"=v,~m0", 1, 0, "~m0"},
        {"~{m0", 0, 0, "~{m0"},
        {"~{}", 0, 0, "~{}"},
        {"~{s0}", 0, 0, "~{s0}"},
        {"~{memory},v", 0, 1, "v"},
        {"v,=v", 1, 1, "=v"},
        {"0", 0, 1, "0"},
        {"=v,1", 1, 1, "1"},
        {"+v", 1, 0, "+v"},
        {"=r", 1, 0, "=r"},
        {"=v,0,0", 1, 2, "0"},
    };
    for(size_t index = 0; index < sizeof(cases) / sizeof(cases[0]) + 4; ++index)
    {
        const bool typed = index >= sizeof(cases) / sizeof(cases[0]);
        const auto* item = typed ? nullptr : &cases[index];
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "asm_constraints") == ROCKE_OK);
        rocke_value_t* operands[2] = {};
        int inputs = typed ? 0 : item->inputs;
        for(int i = 0; i < inputs; ++i)
        {
            char name[32];
            snprintf(name, sizeof name, "value%d", i);
            operands[i] = rocke_b_param(&b, name, rocke_f32(), nullptr);
        }
        const rocke_type_t* types[] = {rocke_f32()};
        auto* op = rocke_b_inline_asm(&b,
                                      "s_nop 0",
                                      typed ? "" : item->constraints,
                                      operands,
                                      inputs,
                                      types,
                                      typed ? 0 : item->outputs,
                                      nullptr);
        CHECK(op);
        if(typed)
        {
            size_t kind = index - sizeof(cases) / sizeof(cases[0]);
            if(kind < 2)
                rocke_attr_set_bool(&b, &op->attrs, "constraints", kind != 0);
            else if(kind == 2)
                rocke_attr_set_int(&b, &op->attrs, "constraints", 0);
            else
                rocke_attr_set_float(&b, &op->attrs, "constraints", 1.0);
        }
        rocke_b_ret(&b);
        char message[256];
        const char* diagnostic
            = typed ? "HIP inline asm constraints must be a string" : item->diagnostic;
        if(!typed && index >= 8)
        {
            snprintf(message,
                     sizeof message,
                     "HIP inline asm constraint failed - constraint=\"%s\", "
                     "expected a supported output, input, tie, or clobber",
                     diagnostic ? diagnostic : "");
            diagnostic = message;
        }
        char* serialized = nullptr;
        CHECK(rocke_ir_serialize(b.kernel, &serialized) == ROCKE_OK);
        rocke_ir_builder_t copy;
        CHECK(rocke_ir_builder_init(&copy, "imported_constraints") == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        CHECK(rocke_ir_parse(serialized, &copy, &kernel) == ROCKE_OK);
        free(serialized);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx950";
        rocke_strbuf_t original = {}, imported = {};
        auto status = typed ? ROCKE_ERR_TYPE : ROCKE_ERR_VALUE;
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &original) == status);
        CHECK(rocke_lower_kernel_to_hip(&copy, kernel, &opts, &imported) == status);
        CHECK(strcmp(rocke_ir_builder_error(&b), diagnostic) == 0);
        CHECK(strcmp(rocke_ir_builder_error(&copy), diagnostic) == 0);
        CHECK(!strstr(original.cstr(), "asm volatile"));
        CHECK(!strstr(imported.cstr(), "asm volatile"));
        original.free_buffer();
        imported.free_buffer();
        rocke_ir_builder_free(&copy);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_hip_clobber_roundtrip()
{
    for(bool output : {false, true})
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "asm_clobbers") == ROCKE_OK);
        auto* value = rocke_b_param(&b, "value", rocke_f32(), nullptr);
        rocke_value_t* operands[] = {value};
        const rocke_type_t* types[] = {rocke_f32()};
        CHECK(rocke_b_inline_asm(&b,
                                 output ? "v_mov_b32 $0, $1" : "s_nop 0",
                                 output ? "=v,0,~{m0},~{memory}" : "v,~{m0},~{memory}",
                                 operands,
                                 1,
                                 types,
                                 output ? 1 : 0,
                                 nullptr));
        rocke_b_ret(&b);
        char* serialized = nullptr;
        CHECK(rocke_ir_serialize(b.kernel, &serialized) == ROCKE_OK);
        rocke_ir_builder_t copy;
        CHECK(rocke_ir_builder_init(&copy, "imported_clobbers") == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        CHECK(rocke_ir_parse(serialized, &copy, &kernel) == ROCKE_OK);
        free(serialized);
        for(const char* arch : {"gfx950", "gfx1250"})
        {
            rocke_lower_hip_opts_t opts = {};
            opts.arch = arch;
            rocke_strbuf_t original = {}, imported = {};
            CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &original) == ROCKE_OK);
            CHECK(rocke_lower_kernel_to_hip(&copy, kernel, &opts, &imported) == ROCKE_OK);
            CHECK(strcmp(original.cstr(), imported.cstr()) == 0);
            CHECK(strstr(original.cstr(), " : \"m0\", \"memory\");"));
            CHECK(strstr(original.cstr(), "(value)"));
            original.free_buffer();
            imported.free_buffer();
        }
        rocke_ir_builder_free(&copy);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

int main()
{
    int admission = test_admission();
    int scalars = test_scalars();
    int predicate = test_predicate();
    int hip = test_tensor_epilogue_hip();
    int attributes = test_hip_rejects_malformed_attributes();
    int strings = test_hip_asm_string_roundtrip();
    int constraints = test_hip_constraint_rejection();
    int clobbers = test_hip_clobber_roundtrip();
    return admission || scalars || predicate || hip || attributes || strings || constraints
           || clobbers;
}
