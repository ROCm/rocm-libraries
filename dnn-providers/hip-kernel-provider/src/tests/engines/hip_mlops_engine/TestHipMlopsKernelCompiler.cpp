// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>

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

    EXPECT_EQ(cache.size(), 1);
    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}));
}

} // namespace
} // namespace hip_kernel_provider
