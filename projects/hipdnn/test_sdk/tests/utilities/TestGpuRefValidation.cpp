// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>

#include <stdexcept>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

} // namespace

// ============================================================================
// TestGpuConvFwdRefValidation — validateInput throw paths
// ============================================================================

TEST(TestGpuConvFwdRefValidation, ThrowsOnInvalidDimCount)
{
    Tensor<float> x({8, 8});
    Tensor<float> w({8, 8});
    Tensor<float> y({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnWeightDimMismatch)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimMismatch)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnStridesSizeMismatch)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnDilationsSizeMismatch)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPrePaddingSizeMismatch)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPostPaddingSizeMismatch)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnZeroStride)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativeDilation)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePrePadding)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePostPadding)
{
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimValueMismatch)
{
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 → expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// TestGpuRefHipError — throwOnHipError coverage
// ============================================================================

TEST(TestGpuRefHipError, ThrowsOnError)
{
    EXPECT_THROW(detail::throwOnHipError(hipErrorMemoryAllocation, "test"), std::runtime_error);
}

TEST(TestGpuRefHipError, NoThrowOnSuccess)
{
    EXPECT_NO_THROW(detail::throwOnHipError(hipSuccess, "test"));
}

// ============================================================================
// GpuTestKernelCompiler — compiler error path coverage
// ============================================================================

TEST(GpuTestKernelCompiler, ThrowsOnCompilationFailure)
{
    SKIP_IF_NO_DEVICES();

    auto& compiler = detail::GpuRefKernelCompiler::instance();

    // Passing an invalid type define causes a compilation error in the kernel source
    EXPECT_THROW(
        compiler.getOrCompile("GpuRefConvFwd.cpp", {"-DX_TYPE=___invalid___"}, "convFwdRef2d"),
        std::runtime_error);
}

TEST(GpuTestKernelCompiler, ThrowsOnInvalidFunctionName)
{
    SKIP_IF_NO_DEVICES();

    auto& compiler = detail::GpuRefKernelCompiler::instance();

    // Valid defines but non-existent function name
    EXPECT_THROW(
        compiler.getOrCompile(
            "GpuRefConvFwd.cpp",
            {"-DX_TYPE=float", "-DW_TYPE=float", "-DY_TYPE=float", "-DCOMPUTE_TYPE=double"},
            "nonExistentFunction"),
        std::runtime_error);

    // Clear the HIP error state left by the failed hipModuleGetFunction call,
    // so the global HipErrorHandler listener doesn't flag it after this test.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}
