// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Canonical vector-add variant table — single source of truth for all kernel specs.
// Shared between device (.hip) and host (main.cpp) compilation.
//
// Device code uses vector_add_variant_spec("name") to look up a spec by name at
// compile time. Host code iterates vector_add_variants[] for registry/dispatch.

#pragma once

#include <rocm_ck/elementwise_spec.hpp>

#include <string_view>

namespace rocm_ck {

struct ElementwiseVariant
{
    const char* name;
    ElementwiseSpec spec;
};

inline constexpr ElementwiseVariant vector_add_variants[] = {
    {"vector_add_fp32_b256",
     make_spec(
         Signature{
             .dtype = DataType::FP32,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{256, 1, 256, true})},
    {"vector_add_fp32_b512",
     make_spec(
         Signature{
             .dtype = DataType::FP32,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{512, 1, 512, true})},
    {"vector_add_fp32_b1024",
     make_spec(
         Signature{
             .dtype = DataType::FP32,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_fp16_b512",
     make_spec(
         Signature{
             .dtype = DataType::FP16,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{512, 1, 512, true})},
    {"vector_add_fp16_b1024",
     make_spec(
         Signature{
             .dtype = DataType::FP16,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_bf16_b512",
     make_spec(
         Signature{
             .dtype = DataType::BF16,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{512, 1, 512, true})},
    {"vector_add_bf16_b1024",
     make_spec(
         Signature{
             .dtype = DataType::BF16,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_fp32_b2048_w8",
     make_spec(
         Signature{
             .dtype = DataType::FP32,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{2048, 8, 64, true})},
    {"vector_add_fp16_b1024_w2",
     make_spec(
         Signature{
             .dtype = DataType::FP16,
             .ops   = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 2, 512, true})},
    // Mixed-type variants
    {"vector_add_fp16_fp32_b1024",
     make_spec(
         Signature{
             .dtype   = DataType::FP16,
             .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
             .ops     = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_fp32_fp16_b1024",
     make_spec(
         Signature{
             .dtype   = DataType::FP32,
             .tensors = {Tensor{.name = "C", .dtype = DataType::FP16}},
             .ops     = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_bf16_fp32_b1024",
     make_spec(
         Signature{
             .dtype   = DataType::BF16,
             .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
             .ops     = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}},
         },
         ElementwiseAlgorithm{1024, 1, 1024, true})},
};

inline constexpr int vector_add_variant_count =
    sizeof(vector_add_variants) / sizeof(vector_add_variants[0]);

/// Compile-time variant lookup by name. Typos cause compile errors.
consteval ElementwiseSpec vector_add_variant_spec(const char* name)
{
    for(const auto& v : vector_add_variants)
        if(std::string_view(v.name) == name)
            return v.spec;
    throw "unknown vector-add variant name";
}

/// Find the best variant for the given data types and problem size.
/// Prefers the largest block_tile that divides problem_size cleanly (aligned).
/// Falls back to the largest padded variant if none aligns.
/// Returns nullptr if no variant matches the type pair.
constexpr const ElementwiseVariant*
findVariant(DataType in_dtype, DataType out_dtype, int problem_size)
{
    const ElementwiseVariant* best_aligned = nullptr;
    const ElementwiseVariant* best_padded  = nullptr;

    for(int i = 0; i < vector_add_variant_count; ++i)
    {
        const auto& v = vector_add_variants[i];
        if(v.spec.lhs().dtype != in_dtype || v.spec.output().dtype != out_dtype)
            continue;

        if(isAligned(v.spec, problem_size))
        {
            if(!best_aligned || v.spec.block_tile > best_aligned->spec.block_tile)
                best_aligned = &vector_add_variants[i];
        }

        if(v.spec.pad)
        {
            if(!best_padded || v.spec.block_tile > best_padded->spec.block_tile)
                best_padded = &vector_add_variants[i];
        }
    }

    return best_aligned ? best_aligned : best_padded;
}

} // namespace rocm_ck
