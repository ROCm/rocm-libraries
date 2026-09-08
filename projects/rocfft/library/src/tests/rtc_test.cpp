// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../../shared/environment.h"
#include "../../../shared/hiprtc_except.h"
#include "rtc_cache.h"
#include "rtc_kernel.h"
#include <gtest/gtest.h>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

#ifndef _WIN32
#include <errno.h> // program_invocation_name
#endif

namespace fs = std::filesystem;

static const char* simple_kernel_jit = R"(
extern "C" __global__ void simple_kernel(int* input)
{
    *input = 1337;
}
)";

// make sure RTC gracefully handles a helper process that crashes
TEST(rocfft_internal, rtc_helper_crash)
{
#ifdef _WIN32
    char filename[MAX_PATH];
    GetModuleFileNameA(NULL, filename, MAX_PATH);
    fs::path test_exe    = filename;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash.exe");
#else
    fs::path test_exe    = program_invocation_name;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash");
#endif

    // don't touch the cache, to force compilation
    EnvironmentSetTemp env_read("ROCFFT_RTC_CACHE_READ_DISABLE", "1");
    EnvironmentSetTemp env_write("ROCFFT_RTC_CACHE_WRITE_DISABLE", "1");
    // force out-of-process compile
    EnvironmentSetTemp env_process("ROCFFT_RTC_PROCESS", "1");

    // build a trivial kernel for some specific arch
    auto generator_func = [](const std::string&) { return std::string(simple_kernel_jit); };
    kernel_src_gen_t generator(generator_func);
    const auto       code = RTCCache::cached_compile(
        "simple_kernel", "gfx1201", generator, std::array<char, 32>{}, crasher_exe.string());

    // we should get compiled code back
    ASSERT_FALSE(code.empty());
}

// Trivial kernel that throws on construction
struct RTCKernelModuleLoadFailure : public RTCKernel
{
    static constexpr auto KERNEL_NAME = "rtc_module_load_failure_kernel";

    RTCKernelModuleLoadFailure(std::shared_future<hipModule_wrapper_t>& module)
        : RTCKernel(KERNEL_NAME, module, {}, {})
    {
    }

    static std::shared_future<std::unique_ptr<RTCKernel>> compile()
    {
        RTCGenerator generator;
        generator.generate_name = []() { return std::string(KERNEL_NAME); };
        generator.generate_src  = [](const std::string& kernel_name) {
            return std::string("extern \"C\" __global__ void ") + kernel_name + "(){}";
        };
        generator.construct_rtckernel = [](const std::string&,
                                           std::shared_future<hipModule_wrapper_t>&,
                                           dim3,
                                           dim3) -> std::unique_ptr<RTCKernel> {
            // Simulate a module load failure by throwing a runtime error on construction.
            throw hip_runtime_error("simulated module load failure", hipErrorNoBinaryForGpu);
        };

        std::string kernel_name;
        return runtime_compile(generator, "gfx90a", kernel_name, std::nullopt, std::nullopt);
    }
};

TEST(rocfft_internal, rtc_module_load_failure)
{
    auto kernel_future = RTCKernelModuleLoadFailure::compile();

    ASSERT_TRUE(kernel_future.valid());
    // Resolve the future, should get a graceful exception back.  If
    // the exception was not handled inside runtime_compile, we'd
    // expect std::terminate to be called and the whole test process
    // would die.
    ASSERT_THROW(kernel_future.get(), hip_runtime_error);
}

struct RTCKernelCompileFailure : public RTCKernel
{
    static constexpr auto KERNEL_NAME = "rtc_compile_failure_kernel";

    RTCKernelCompileFailure(std::shared_future<hipModule_wrapper_t>& module)
        : RTCKernel(KERNEL_NAME, module, {}, {})
    {
    }

    static std::shared_future<std::unique_ptr<RTCKernel>> compile()
    {
        RTCGenerator generator;
        generator.generate_name = []() { return std::string(KERNEL_NAME); };
        generator.generate_src  = [](const std::string&) -> std::string {
            throw hiprtc_runtime_error("simulated compile failure", HIPRTC_ERROR_COMPILATION);
        };
        generator.construct_rtckernel = [](const std::string&,
                                           std::shared_future<hipModule_wrapper_t>& module,
                                           dim3,
                                           dim3) -> std::unique_ptr<RTCKernel> {
            return std::make_unique<RTCKernelCompileFailure>(module);
        };

        std::string kernel_name;
        return runtime_compile(generator, "gfx90a", kernel_name, std::nullopt, std::nullopt);
    }

    RTCKernelArgs get_launch_args(DeviceCallIn& data)
    {
        return {};
    }
};

TEST(rocfft_internal, rtc_compile_failure)
{
    static constexpr unsigned int NUM_THREADS = 4;

    std::vector<std::shared_future<std::unique_ptr<RTCKernel>>> futures(NUM_THREADS);
    std::vector<std::thread>                                    threads(NUM_THREADS);

    for(unsigned int i = 0; i < NUM_THREADS; ++i)
    {
        threads[i]
            = std::thread([&futures, i]() { futures[i] = RTCKernelCompileFailure::compile(); });
    }
    for(auto& t : threads)
        t.join();

    // Every thread should get a graceful exception, whether it was
    // the one that tried to compile or one sharing the module future.
    for(auto& f : futures)
    {
        ASSERT_TRUE(f.valid());
        ASSERT_THROW(f.get(), hiprtc_runtime_error);
    }
}
