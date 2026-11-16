// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compilation/CompileCounters.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>

#include <string>

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

// A program that fails to build is compiled once: the cache's own failure is what the caller
// sees, naming the program and carrying HIPRTC's log.
TEST(TestHipMlopsKernelCompiler, AFailingProgramCompilesOnceAndThrowsItsOwnError)
{
    SKIP_IF_NO_DEVICES();

    const HipMlopsKernelCompiler compiler;
    const auto before = compilation::compileCounters().compileFailures;
    try
    {
        // vector_add.cpp needs FLOAT defined.
        static_cast<void>(compiler.compile("vector_add.cpp", {"-O3", "-DFAILS_ONCE=1"}));
        ADD_FAILURE() << "the compile did not fail";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_NE(std::string(error.what()).find("hiprtcCompileProgram failed for vector_add.cpp"),
                  std::string::npos)
            << error.what();
    }
    EXPECT_EQ(compilation::compileCounters().compileFailures - before, 1u);
    EXPECT_FALSE(HipMlopsKernelCompiler::moduleCache().contains("vector_add.cpp",
                                                                {"-O3", "-DFAILS_ONCE=1"}));
}

} // namespace
} // namespace hip_kernel_provider
