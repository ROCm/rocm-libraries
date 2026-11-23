// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compilation/HiprtcKernelCompiler.hpp"
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>

namespace hip_kernel_provider::compilation
{
namespace
{

TEST(TestHiprtcKernelCompiler, InvalidParams)
{
    SKIP_IF_NO_DEVICES();

    const HiprtcKernelCompiler compiler;
    EXPECT_THROW(compiler.compile("invalid.cpp", {"-O3", "-DFLOAT=float"}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHiprtcKernelCompiler, ValidParams)
{
    SKIP_IF_NO_DEVICES();

    const HiprtcKernelCompiler compiler;
    auto& cache = HiprtcKernelCompiler::moduleCache();
    EXPECT_EQ(cache.size(), 0);

    auto result = compiler.compile("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    EXPECT_NE(result, nullptr);

    EXPECT_EQ(cache.size(), 1);
    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}));
}

} // namespace
} // namespace hip_kernel_provider::compilation
