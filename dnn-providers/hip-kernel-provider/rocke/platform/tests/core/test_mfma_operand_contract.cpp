// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native MFMA lowering admission: the fixed instruction ABI must agree with
 * each actual SSA source/result, independently of catalog result metadata. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/tf32_internal.h"

#define CHECK(expr)                                            \
    do                                                         \
    {                                                          \
        if(!(expr))                                            \
        {                                                      \
            fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                          \
        }                                                      \
    } while(0)

typedef struct mfma_case
{
    const char* id;
    const char* source;
    int source_count;
    int accumulator_count;
    int packed_bits;
    int scaled;
    int hip_supported;
} mfma_case_t;

static const mfma_case_t cases[] = {
    {"mfma_f32_16x16x4_f32", "f32", 1, 4, 0, 0, 1},
    {"mfma_f32_32x32x2_f32", "f32", 1, 16, 0, 0, 1},
    {"mfma_f32_16x16x8_xf32", "tf32", 2, 4, 0, 0, 1},
    {"mfma_f32_32x32x4_xf32", "tf32", 2, 16, 0, 0, 1},
    {"mfma_f32_16x16x16_f16", "f16", 4, 4, 0, 0, 0},
    {"mfma_f32_16x16x32_f16", "f16", 8, 4, 0, 0, 0},
    {"mfma_f32_16x16x16_bf16", "bf16", 4, 4, 0, 0, 0},
    {"mfma_f32_16x16x32_bf16", "bf16", 8, 4, 0, 0, 0},
    {"mfma_f32_32x32x8_f16", "f16", 4, 16, 0, 0, 0},
    {"mfma_f32_32x32x8_bf16", "bf16", 4, 16, 0, 0, 0},
    {"mfma_f32_32x32x16_f16", "f16", 8, 16, 0, 0, 0},
    {"mfma_f32_32x32x16_bf16", "bf16", 8, 16, 0, 0, 0},
    {"mfma_f32_4x4x4_f16", "f16", 4, 4, 0, 0, 0},
    {"mfma_f32_16x16x32_fp8", "fp8e4m3", 8, 4, 0, 0, 0},
    {"mfma_f32_16x16x32_bf8", "bf8e5m2", 8, 4, 0, 0, 0},
    {"mfma_f32_32x32x16_fp8", "fp8e4m3", 8, 16, 0, 0, 0},
    {"mfma_f32_32x32x16_bf8", "bf8e5m2", 8, 16, 0, 0, 0},
    {"mfma_f32_16x16x128_fp4", "i64", 1, 4, 64, 0, 0},
    {"mfma_f32_16x16x96_fp6", "i32", 3, 4, 96, 0, 0},
    {"mfma_f32_16x16x128_fp8", "i32", 8, 4, 256, 0, 0},
    {"mfma_scale_f32_16x16x128_f8f6f4", "i32", 8, 4, 256, 1, 0},
    {"mfma_f64_16x16x4_f64", "f64", 1, 4, 0, 0, 1},
    {"mfma_f64_4x4x4_f64", "f64", 1, 1, 0, 0, 1},
    {"mfma_f32_4x4x4_bf16", "bf16", 4, 4, 0, 0, 0},
};

static int check_case(const mfma_case_t* spec, int variant, int generic)
{
    rocke_ir_builder_t builder;
    CHECK(rocke_ir_builder_init(&builder, "mfma_contract") == ROCKE_OK);
    const auto* element = rocke_scalar_by_name(spec->source);
    const auto* source_type = spec->source_count == 1
                                  ? element
                                  : rocke_vector_type(&builder, element, spec->source_count);
    const auto* accumulator_element
        = strncmp(spec->id, "mfma_f64_", 9) == 0 ? rocke_f64() : rocke_f32();
    const auto* accumulator_type
        = rocke_vector_type(&builder, accumulator_element, spec->accumulator_count);
    auto* a = rocke_b_param(&builder, "a", source_type, NULL);
    auto* b = rocke_b_param(&builder, "b", source_type, NULL);
    auto* c = rocke_b_param(&builder, "c", accumulator_type, NULL);
    auto* sa = rocke_b_param(&builder, "sa", rocke_i32(), NULL);
    auto* sb = rocke_b_param(&builder, "sb", rocke_i32(), NULL);
    rocke_value_t* operands[] = {a, b, c, sa, sb};
    rocke_value_t* extra[] = {sa, sb};
    rocke_op_t* op;
    if(generic)
    {
        rocke_attr_map_t attrs = {};
        rocke_attr_set_str(&builder, &attrs, "op_id", spec->id);
        const rocke_type_t* results[] = {accumulator_type};
        op = rocke_b_op(&builder,
                        ROCKE_OP_TILE_MMA,
                        operands,
                        spec->scaled ? 5 : 3,
                        results,
                        1,
                        &attrs,
                        NULL,
                        0,
                        "acc",
                        NULL);
    }
    else
    {
        auto* result = rocke_b_mma(&builder, spec->id, a, b, c, extra, spec->scaled ? 2 : 0);
        CHECK(result);
        op = builder.kernel->body->ops[builder.kernel->body->num_ops - 1];
    }
    CHECK(op);
    rocke_b_ret(&builder);
    char expected[160] = {};
    const bool xf32 = strcmp(spec->source, "tf32") == 0;
    if(variant == 11 || variant == 12)
    {
        // The native parser accepts neutral tile.mma; exercise its shared
        // pre-dispatch gate directly for Python's concrete MFMA names.
        op->name = "tile.mfma_f32_16x16x4_f32";
        const char* attribute = variant == 11 ? spec->id : "mfma_f32_16x16x16_f16";
        rocke_attr_set_str(&builder, &op->attrs, "op_id", attribute);
        const char* error = rocke_tf32_op_error(op);
        if(variant == 12)
        {
            snprintf(expected,
                     sizeof(expected),
                     "concrete MFMA %s has conflicting op_id %s",
                     op->name,
                     attribute);
            CHECK(error && strcmp(error, expected) == 0);
        }
        else
            CHECK(error == NULL);
        rocke_ir_builder_free(&builder);
        return 0;
    }
    else if(variant == 1 || variant == 2)
    {
        op->operands[variant - 1]->type = rocke_i32();
        if(xf32)
            snprintf(expected,
                     sizeof(expected),
                     "XF32 MMA requires two vec<tf32x2> operands and its FP32 "
                     "accumulator");
        else if(spec->packed_bits)
            snprintf(expected,
                     sizeof(expected),
                     "MFMA requires src0 and src1 to have %d bits",
                     spec->packed_bits);
        else
            snprintf(expected,
                     sizeof(expected),
                     "MFMA requires src0 and src1 to be %s",
                     strcmp(spec->source, "fp8e4m3") == 0 || strcmp(spec->source, "bf8e5m2") == 0
                         ? "vec<i8x8>"
                         : source_type->name);
    }
    else if(variant >= 3 && variant <= 6)
    {
        const auto* bad_type
            = variant <= 4 ? rocke_vector_type(&builder, rocke_i32(), 7)
                           : rocke_vector_type(&builder, rocke_f32(), spec->accumulator_count - 1);
        if(variant == 3 || variant == 5)
            c->type = bad_type;
        else
            op->results[0]->type = bad_type;
        if(xf32)
            snprintf(expected,
                     sizeof(expected),
                     "%s",
                     variant == 3 || variant == 5
                         ? "XF32 MMA requires two vec<tf32x2> operands and its FP32 "
                           "accumulator"
                         : "XF32 MMA result must match its FP32 accumulator");
        else
            snprintf(expected,
                     sizeof(expected),
                     "MFMA requires src2 and dst to be %s",
                     accumulator_type->name);
    }
    else if(variant == 7 || variant == 8)
    {
        if(variant == 7)
            --op->num_operands;
        else
            op->num_results = 0;
        snprintf(expected,
                 sizeof(expected),
                 "MFMA expects %d operands and 1 result",
                 spec->scaled ? 5 : 3);
        if(xf32)
            snprintf(expected,
                     sizeof(expected),
                     "%s",
                     variant == 7 ? "XF32 MMA requires two vec<tf32x2> operands and its FP32 "
                                    "accumulator"
                                  : "XF32 MMA result must match its FP32 accumulator");
    }
    else if(variant == 9 || variant == 10)
    {
        op->operands[variant - 6]->type = rocke_i64();
        snprintf(expected, sizeof(expected), "MFMA requires scale operands to be i32");
    }
    for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
    {
        char* llvm = NULL;
        char error[ROCKE_ERR_MSG_CAP] = {};
        auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
        const auto status = rocke_lower_kernel_to_llvm_ex(
            builder.kernel, flavor, "gfx942", &llvm, error, sizeof(error));
        CHECK(status == (variant ? ROCKE_ERR_VALUE : ROCKE_OK));
        if(variant)
        {
            CHECK(llvm == NULL);
            CHECK(strcmp(error, expected) == 0);
        }
        else
        {
            CHECK(strstr(llvm, "call "));
            CHECK(strstr(llvm, "@llvm.amdgcn.mfma."));
        }
        free(llvm);
    }
    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx942";
    const auto status = rocke_lower_kernel_to_hip(&builder, builder.kernel, &opts, &hip);
    if(spec->hip_supported)
    {
        CHECK(status == (variant ? ROCKE_ERR_VALUE : ROCKE_OK));
        if(variant)
            CHECK(hip.len == 0);
        else
            CHECK(strstr(hip.data, "__builtin_amdgcn_mfma_"));
    }
    else if(strcmp(spec->source, "i64") == 0 && spec->source_count > 1)
        // HIP refuses this LLVM-only packed carrier as an unsupported type.
        CHECK(status == ROCKE_ERR_NOTIMPL);
    else
        CHECK(status == ROCKE_ERR_NOTIMPL);
    rocke_strbuf_free(&hip);
    rocke_ir_builder_free(&builder);
    return 0;
}

static int check_packed_carriers()
{
    const char* carriers[] = {"i8", "u8", "i16", "f16", "bf16", "i32", "f32", "i64", "f64"};
    const int widths[] = {8, 8, 16, 16, 16, 32, 32, 64, 64};
    for(const auto& base : cases)
    {
        if(!base.packed_bits)
            continue;
        for(size_t i = 0; i < sizeof(carriers) / sizeof(carriers[0]); ++i)
        {
            if(base.packed_bits % widths[i])
                continue;
            auto spec = base;
            spec.source = carriers[i];
            spec.source_count = base.packed_bits / widths[i];
            CHECK(check_case(&spec, 0, 1) == 0);
        }
    }
    // Logical byte types and integer byte carriers share the fixed byte ABI.
    const char* byte_carriers[] = {"i8", "u8"};
    for(int index = 13; index <= 16; ++index)
    {
        auto spec = cases[index];
        for(const char* source : byte_carriers)
        {
            spec.source = source;
            CHECK(check_case(&spec, 0, 1) == 0);
        }
    }
    return 0;
}

int main()
{
    for(const auto& spec : cases)
        for(int generic = 0; generic < 2; ++generic)
            for(int variant = 0; variant <= (spec.scaled ? 10 : 8); ++variant)
                CHECK(check_case(&spec, variant, generic) == 0);
    CHECK(check_packed_carriers() == 0);
    CHECK(check_case(&cases[0], 11, 0) == 0);
    CHECK(check_case(&cases[0], 12, 0) == 0);
    return 0;
}
