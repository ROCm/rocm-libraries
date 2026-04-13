// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>

#include <stdexcept>

using namespace hipdnn_gpu_ref;

// ============================================================================
// TestGpuRefKernelCompiler — compiler error path coverage
// ============================================================================

TEST(TestGpuRefKernelCompiler, ThrowsOnCompilationFailure)
{
    SKIP_IF_NO_DEVICES();

    auto& compiler = detail::GpuRefKernelCompiler::instance();

    // Passing an invalid type define causes a compilation error in the kernel source
    EXPECT_THROW(
        compiler.getOrCompile("GpuRefConvFwd.cpp", {"-DX_TYPE=___invalid___"}, "convFwdRef2d"),
        std::runtime_error);
}

TEST(TestGpuRefKernelCompiler, ThrowsOnInvalidFunctionName)
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
