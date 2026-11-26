// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "compilation/Kernel.hpp"
#include "compilation/Program.hpp"
#include "compilation/ProgramCache.hpp"
#include "engines/kernels/MxScaleRelayoutParams.hpp"

using namespace hipblaslt_plugin::compilation;
namespace relayout = hipblaslt_plugin::mx_scale_relayout;

// The provider's only embedded source, the MX scale relayout kernel, stands in for any
// kernel: the tests need no kernel of their own.

namespace
{

std::vector<std::string> optionsForCurrentDevice()
{
    return {"-std=c++17", "--offload-arch=" + hipdnn_test_sdk::utilities::currentDeviceArch()};
}

// Empties `constexpr uint32_t THREADS = 256;` in the kernel's parameter header.
const std::string BREAKS_THE_SOURCE = "-DTHREADS=";

} // namespace

// A moved-from Program would unload its module a second time.
TEST(TestProgram, CannotBeCopiedOrMoved)
{
    EXPECT_FALSE(std::is_copy_constructible_v<Program>);
    EXPECT_FALSE(std::is_move_constructible_v<Program>);
    EXPECT_FALSE(std::is_move_assignable_v<Program>);
}

// The kernel compiles for both architectures whose Tensile MX kernels read swizzled
// scales, whatever device the tests run on (or none).
TEST(TestProgram, CompilesForEveryMxArchitecture)
{
    for(const char* arch : {"gfx950", "gfx1250"})
    {
        EXPECT_FALSE(compileKernelSource(relayout::SOURCE_NAME,
                                         {"-std=c++17", std::string("--offload-arch=") + arch})
                         .empty())
            << arch;
    }
}

TEST(TestProgram, UnknownSourceThrows)
{
    EXPECT_THROW(Program("NoSuchKernel.cpp", {}), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGpuProgram, CompilesTheEmbeddedSource)
{
    SKIP_IF_NO_DEVICES();
    const Program program(relayout::SOURCE_NAME, optionsForCurrentDevice());
    EXPECT_NE(program.getKernel(relayout::GFX950_KERNEL), nullptr);
    EXPECT_NE(program.getKernel(relayout::GFX1250_KERNEL), nullptr);
}

TEST(TestGpuProgram, UnknownKernelThrows)
{
    SKIP_IF_NO_DEVICES();
    const Program program(relayout::SOURCE_NAME, optionsForCurrentDevice());
    EXPECT_THROW(static_cast<void>(program.getKernel("NoSuchKernel")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGpuProgram, CompileFailureReportsTheLog)
{
    SKIP_IF_NO_DEVICES();
    auto options = optionsForCurrentDevice();
    options.push_back(BREAKS_THE_SOURCE);
    try
    {
        const Program program(relayout::SOURCE_NAME, options);
        ADD_FAILURE() << "the broken source compiled";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("hiprtcCompileProgram failed"), std::string::npos) << message;
        EXPECT_NE(message.find("error"), std::string::npos) << "no compile log in: " << message;
    }
}

TEST(TestGpuProgramCache, SameKeySharesOneProgram)
{
    SKIP_IF_NO_DEVICES();
    const auto first = getProgram(relayout::SOURCE_NAME, {"-std=c++17"});
    const auto second = getProgram(relayout::SOURCE_NAME, {"-std=c++17"});
    EXPECT_EQ(first, second);
    EXPECT_NE(getProgram(relayout::SOURCE_NAME, {"-std=c++17", "-DUNUSED_TEST_MACRO=1"}), first);
}

// Concurrent first requests for one key wait for a single compile and share its
// Program. The key is new to the process, so the first request compiles.
TEST(TestGpuProgramCache, ConcurrentRequestsShareOneProgram)
{
    SKIP_IF_NO_DEVICES();
    constexpr size_t THREADS = 8;
    const std::vector<std::string> options{"-std=c++17", "-DCONCURRENT_REQUESTS_TEST=1"};
    std::vector<std::shared_ptr<const Program>> programs(THREADS);
    std::vector<std::string> errors(THREADS);
    std::promise<void> go;
    const std::shared_future<void> start = go.get_future().share();
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for(size_t i = 0; i < THREADS; ++i)
    {
        threads.emplace_back([&, i] {
            start.wait();
            try
            {
                programs[i] = getProgram(relayout::SOURCE_NAME, options);
            }
            catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
            {
                errors[i] = e.what();
            }
        });
    }
    go.set_value();
    for(auto& thread : threads)
    {
        thread.join();
    }
    for(size_t i = 0; i < THREADS; ++i)
    {
        EXPECT_TRUE(errors[i].empty()) << "thread " << i << ": " << errors[i];
        EXPECT_EQ(programs[i], programs[0]) << "thread " << i;
    }
    EXPECT_NE(programs[0], nullptr);
}

// A failed compile is not cached: each request compiles and fails again, and other keys
// are unaffected.
TEST(TestGpuProgramCache, FailedCompileIsNotCached)
{
    SKIP_IF_NO_DEVICES();
    for(int attempt = 0; attempt < 2; ++attempt)
    {
        EXPECT_THROW(
            static_cast<void>(getProgram(relayout::SOURCE_NAME, {"-std=c++17", BREAKS_THE_SOURCE})),
            hipdnn_plugin_sdk::HipdnnPluginException)
            << "attempt " << attempt;
    }
    EXPECT_NE(getProgram(relayout::SOURCE_NAME, {"-std=c++17"}), nullptr);
}

TEST(TestGpuKernel, KeepsItsProgramAlive)
{
    SKIP_IF_NO_DEVICES();
    auto program
        = std::make_shared<const Program>(relayout::SOURCE_NAME, optionsForCurrentDevice());
    const std::weak_ptr<const Program> watch = program;
    auto kernel = std::make_unique<Kernel>(
        program->getKernel(relayout::GFX950_KERNEL), relayout::GFX950_KERNEL, program);
    program.reset();
    EXPECT_FALSE(watch.expired());
    kernel.reset();
    EXPECT_TRUE(watch.expired());
}

TEST(TestGpuKernel, RefusedLaunchThrows)
{
    SKIP_IF_NO_DEVICES();
    const auto program = getProgram(relayout::SOURCE_NAME, {"-std=c++17"});
    Kernel kernel(program->getKernel(relayout::GFX950_KERNEL), relayout::GFX950_KERNEL, program);
    // More lanes than any device allows in one block.
    kernel.setBlockSize(4096);
    const relayout::Batch batch{};
    try
    {
        kernel.launch(nullptr, batch);
        ADD_FAILURE() << "a 4096-lane block was launched";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_NE(std::string(e.what()).find(relayout::GFX950_KERNEL), std::string::npos)
            << e.what();
    }
    // The refused launch is the thread's last error; leave none for later tests.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}
