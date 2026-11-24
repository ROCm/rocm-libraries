// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>

#include <stdexcept>
#include <string>

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

TEST(TestGpuRefKernelCompiler, CompilesLowPrecisionKernelWithoutEnvironmentPaths)
{
    SKIP_IF_NO_DEVICES();

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter hipPath("HIP_PATH");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter rocmPath("ROCM_PATH");
    hipdnn_data_sdk::utilities::unsetEnv("HIP_PATH");
    hipdnn_data_sdk::utilities::unsetEnv("ROCM_PATH");
    ASSERT_TRUE(hipdnn_data_sdk::utilities::getEnv("HIP_PATH").empty());
    ASSERT_TRUE(hipdnn_data_sdk::utilities::getEnv("ROCM_PATH").empty());

    // Bypass the singleton cache so an earlier block-scale test cannot hide a broken lookup.
    const detail::CompiledKernel kernel(
        "GpuRefBlockScaleQuantize.cpp",
        {"-DGPU_REF_USE_HIP_MX",
         "-DX_TYPE=float",
         std::string("-DY_TYPE=") + detail::HipRtcTypeName<hipdnn_data_sdk::types::fp8_e4m3>::VALUE,
         "-DSCALE_TYPE=float",
         "-DCOMPUTE_TYPE=float"},
        "BlockScaleQuantizeRef");
    EXPECT_NE(kernel.function(), nullptr);
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
