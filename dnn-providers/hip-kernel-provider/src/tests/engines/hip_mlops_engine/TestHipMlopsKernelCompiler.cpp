// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

#include <string>
#include <vector>

namespace hip_kernel_provider
{
namespace
{

TEST(TestHipMlopsKernelCompiler, InvalidParams)
{
    SKIP_IF_NO_DEVICES();

    const HipMlopsKernelCompiler compiler;
    EXPECT_THROW(compiler.compile("invalid.cpp", {"-O3", "-DFLOAT=float"}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipMlopsKernelCompiler, ValidParams)
{
    SKIP_IF_NO_DEVICES();

    const HipMlopsKernelCompiler compiler;
    auto& cache = HipMlopsKernelCompiler::moduleCache();
    EXPECT_EQ(cache.size(), 0);

    auto result = compiler.compile("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    EXPECT_NE(result, nullptr);

    // Loaded on the current device, the one the caller read its properties from.
    int currentDevice = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&currentDevice));
    EXPECT_EQ(cache.size(), 1);
    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}, currentDevice));
}

// The kernels of a compiled program follow the launch device: a second device of the
// build device's architecture gets its own module, through the per-device module cache.
TEST(TestHipMlopsKernelCompiler, KernelsLoadASecondDeviceThroughTheModuleCache)
{
    SKIP_IF_NO_DEVICES();

    int count = 0;
    ASSERT_EQ(hipSuccess, hipGetDeviceCount(&count));
    int current = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&current));
    hipDeviceProp_t currentProps{};
    ASSERT_EQ(hipSuccess, hipGetDeviceProperties(&currentProps, current));
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(currentProps.gcnArchName));

    int second = -1;
    for(int ordinal = 0; ordinal < count && second < 0; ++ordinal)
    {
        hipDeviceProp_t props{};
        if(ordinal != current && hipGetDeviceProperties(&props, ordinal) == hipSuccess
           && hipdnn_plugin_sdk::stripArchFeatures(props.gcnArchName) == arch)
        {
            second = ordinal;
        }
    }
    if(second < 0)
    {
        GTEST_SKIP() << "needs a second device of architecture " << arch;
    }

    // A define no other test uses, so the cache entries checked below are this test's own.
    const std::vector<std::string> options{"-O3", "-DFLOAT=float", "-DSECOND_DEVICE_TEST"};
    const HipMlopsKernelCompiler compiler;
    auto program = compiler.compile("vector_add.cpp", options);
    auto kernel = program->getKernel("vector_add");
    const auto& resolved = dynamic_cast<const compilation::DeviceResolvedKernel&>(*kernel);

    EXPECT_EQ(resolved.resolvedFor(second).program->deviceOrdinal(), second);
    EXPECT_TRUE(HipMlopsKernelCompiler::moduleCache().contains("vector_add.cpp", options, current));
    EXPECT_TRUE(HipMlopsKernelCompiler::moduleCache().contains("vector_add.cpp", options, second));
}

} // namespace
} // namespace hip_kernel_provider
