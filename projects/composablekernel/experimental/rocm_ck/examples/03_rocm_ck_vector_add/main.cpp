// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the rocm_ck vector add example. Loads multiple kernel
// variants from a kpack archive and runs each one, verifying correctness.
//
// Supports same-type (fp32, fp16, bf16) and mixed-type (fp16->fp32, fp32->fp16)
// variants. Each variant gets typed device buffers with host-side conversion
// for upload/download/verification. Kernels compute c = alpha * a + beta * b.

#include "vector_add_variants.hpp"

#include <rocm_ck/datatype_convert.hpp>
#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/typed_buffer.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using rocm_ck::vector_add_variant_count;
using rocm_ck::vector_add_variants;

/// Run a single variant: load from archive, launch kernel, verify results.
/// Returns true if the variant passed verification.
static bool runVariant(const rocm_ck::ElementwiseVariant& variant,
                       const rocm_ck::KpackArchive& archive,
                       const std::vector<float>& host_a,
                       const std::vector<float>& host_b,
                       float alpha,
                       float beta)
{
    const int num_elements = static_cast<int>(host_a.size());
    const auto in_dtype    = variant.spec.lhs().dtype;
    const auto out_dtype   = variant.spec.output().dtype;

    // Load kernel
    rocm_ck::KpackKernel kernel;
    if(!kernel.load(archive, variant.name))
        return false;

    // Allocate typed device buffers
    rocm_ck::TypedBuffer buf_a(in_dtype, num_elements);
    rocm_ck::TypedBuffer buf_b(in_dtype, num_elements);
    rocm_ck::TypedBuffer buf_result(out_dtype, num_elements);

    buf_a.upload(host_a.data());
    buf_b.upload(host_b.data());
    buf_result.zero();

    // Launch
    const int grid_size  = (num_elements + variant.spec.block_tile - 1) / variant.spec.block_tile;
    const int block_size = variant.spec.workgroup_size;
    const bool aligned   = rocm_ck::isAligned(variant.spec, num_elements);
    std::printf("  %s: tile=%d, waves=%d, work_items=%d, N=%d %s\n",
                variant.name,
                variant.spec.block_tile,
                variant.spec.block_waves,
                block_size,
                num_elements,
                aligned ? "(aligned)" : "(padded)");

    rocm_ck::Args kernel_args{};
    kernel_args.tensors[0] = {buf_a.ptr(), {num_elements, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}};
    kernel_args.tensors[1] = {buf_b.ptr(), {num_elements, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}};
    kernel_args.tensors[2] = {buf_result.ptr(), {num_elements, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}};
    kernel_args.scalars[0].f32 = alpha;
    kernel_args.scalars[1].f32 = beta;
    size_t kernel_args_size    = sizeof(kernel_args);
    void* launch_config[]      = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                  &kernel_args,
                                  HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                  &kernel_args_size,
                                  HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(
        kernel.function(), grid_size, 1, 1, block_size, 1, 1, 0, nullptr, nullptr, launch_config));
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify
    std::vector<float> result(num_elements);
    buf_result.download(result.data());

    const float tol = rocm_ck::tolerance_for(out_dtype);
    bool passed     = true;
    for(int i = 0; i < num_elements; ++i)
    {
        float expected = alpha * host_a[i] + beta * host_b[i];
        if(std::fabs(result[i] - expected) > tol * std::fabs(expected) + tol)
        {
            std::fprintf(stderr,
                         "  %s: MISMATCH at index %d: got %f, expected %f\n",
                         variant.name,
                         i,
                         result[i],
                         expected);
            passed = false;
            break;
        }
    }

    std::printf("  %s (grid=%d, block=%d): %s\n",
                variant.name,
                grid_size,
                block_size,
                passed ? "PASSED" : "FAILED");

    return passed;
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-kernels.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive ---
    rocm_ck::KpackArchive archive;
    if(!archive.open(argv[1]))
        return 1;

    // --- Test data (small integers exactly representable in fp16 and bf16) ---
    // bf16 has 7 mantissa bits, so only integers <= 128 are exact.
    // Keep sums <= 62 to stay well within range.
    const int NUM_ELEMENTS = 4096;

    std::vector<float> host_a(NUM_ELEMENTS);
    std::vector<float> host_b(NUM_ELEMENTS);
    for(int i = 0; i < NUM_ELEMENTS; ++i)
    {
        host_a[i] = static_cast<float>(i % 32);
        host_b[i] = static_cast<float>((i * 2) % 32);
    }

    // --- Demonstrate findVariant (same-type and mixed-type lookups) ---
    std::printf("\nVariant selection for N=%d:\n", NUM_ELEMENTS);
    for(auto dt : {rocm_ck::DataType::FP32, rocm_ck::DataType::FP16, rocm_ck::DataType::BF16})
    {
        const auto* best = rocm_ck::findVariant(dt, dt, NUM_ELEMENTS);
        if(best)
            std::printf("  %s -> %s (tile=%d, waves=%d)\n",
                        rocm_ck::data_type_name(dt),
                        best->name,
                        best->spec.block_tile,
                        best->spec.block_waves);
    }
    // Mixed-type: widening variants (narrow input -> FP32 output)
    for(auto in_dt : {rocm_ck::DataType::FP16, rocm_ck::DataType::BF16})
    {
        const auto* best = rocm_ck::findVariant(in_dt, rocm_ck::DataType::FP32, NUM_ELEMENTS);
        if(best)
            std::printf("  %s->FP32 -> %s (tile=%d)\n",
                        rocm_ck::data_type_name(in_dt),
                        best->name,
                        best->spec.block_tile);
    }

    // --- Verify all variants with plain add (alpha=1, beta=1) ---
    std::printf("\nRunning all %d variants (alpha=1, beta=1):\n", vector_add_variant_count);
    bool all_passed = true;

    for(const auto& variant : vector_add_variants)
    {
        if(!runVariant(variant, archive, host_a, host_b, 1.0f, 1.0f))
            all_passed = false;
    }

    // --- Scaled-add test (alpha=2, beta=0.5) ---
    std::printf("\nScaled-add test (alpha=2, beta=0.5):\n");
    const auto* scaled_variant =
        rocm_ck::findVariant(rocm_ck::DataType::FP32, rocm_ck::DataType::FP32, NUM_ELEMENTS);
    if(scaled_variant)
    {
        if(!runVariant(*scaled_variant, archive, host_a, host_b, 2.0f, 0.5f))
            all_passed = false;
    }

    return all_passed ? 0 : 1;
}
