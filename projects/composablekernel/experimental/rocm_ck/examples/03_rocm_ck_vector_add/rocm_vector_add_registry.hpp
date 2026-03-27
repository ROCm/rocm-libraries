// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Variant registry for programmatic kernel selection.
// Host-only header — no CK Tile dependency.

#pragma once

#include "rocm_vector_add_api.hpp"

namespace rocm_ck {

/// Descriptor for a compiled kernel variant in the kpack archive.
struct VariantDescriptor
{
    const char* name;
    ElementwiseSpec kernel;
};

/// Complete table of all compiled variants, matching the kpack archive contents.
/// Each entry corresponds to a .hip file and its make_kernel configuration.
// clang-format off
static constexpr VariantDescriptor ALL_VARIANTS[] = {
    {"vector_add_fp32_b256", make_kernel(
         Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{256, 1, 256, true})},
    {"vector_add_fp32_b512", make_kernel(
         Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{512, 1, 512, true})},
    {"vector_add_fp32_b1024", make_kernel(
         Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_fp16_b512", make_kernel(
         Signature{.dtype = DataType::FP16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{512, 1, 512, true})},
    {"vector_add_fp16_b1024", make_kernel(
         Signature{.dtype = DataType::FP16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_bf16_b512", make_kernel(
         Signature{.dtype = DataType::BF16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{512, 1, 512, true})},
    {"vector_add_bf16_b1024", make_kernel(
         Signature{.dtype = DataType::BF16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_fp32_b2048_w8", make_kernel(
         Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{2048, 8, 64, true})},
    {"vector_add_fp16_b1024_w2", make_kernel(
         Signature{.dtype = DataType::FP16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 2, 512, true})},
    // Mixed-type variants
    {"vector_add_fp16_fp32_b1024", make_kernel(
         Signature{.dtype = DataType::FP16,
                   .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                   .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_fp32_fp16_b1024", make_kernel(
         Signature{.dtype = DataType::FP32,
                   .tensors = {Tensor{.name = "C", .dtype = DataType::FP16}},
                   .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 1, 1024, true})},
    {"vector_add_bf16_fp32_b1024", make_kernel(
         Signature{.dtype = DataType::BF16,
                   .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                   .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
         ElementwiseAlgorithm{1024, 1, 1024, true})},
};
// clang-format on

static constexpr int ALL_VARIANTS_COUNT = sizeof(ALL_VARIANTS) / sizeof(ALL_VARIANTS[0]);

/// Find the best variant for the given data types and problem size.
/// Prefers the largest block_tile that divides problem_size cleanly (aligned).
/// Falls back to the largest padded variant if none aligns.
/// Returns nullptr if no variant matches the type pair.
constexpr const VariantDescriptor*
findVariant(DataType in_dtype, DataType out_dtype, int problem_size)
{
    const VariantDescriptor* best_aligned = nullptr;
    const VariantDescriptor* best_padded  = nullptr;

    for(int i = 0; i < ALL_VARIANTS_COUNT; ++i)
    {
        const auto& v = ALL_VARIANTS[i];
        if(v.kernel.lhs().dtype != in_dtype || v.kernel.output().dtype != out_dtype)
            continue;

        if(isAligned(v.kernel, problem_size))
        {
            if(!best_aligned || v.kernel.block_tile > best_aligned->kernel.block_tile)
                best_aligned = &ALL_VARIANTS[i];
        }

        if(v.kernel.pad)
        {
            if(!best_padded || v.kernel.block_tile > best_padded->kernel.block_tile)
                best_padded = &ALL_VARIANTS[i];
        }
    }

    return best_aligned ? best_aligned : best_padded;
}

} // namespace rocm_ck
