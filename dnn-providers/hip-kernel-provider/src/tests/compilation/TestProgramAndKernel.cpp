// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "compilation/Kernel.hpp"
#include "compilation/Program.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <string>
#include <vector>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::compilation;

TEST(TestProgram, CompilesAndGetsKernel)
{
    SKIP_IF_NO_DEVICES();

    const Program program("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    hipFunction_t kernel = program.getKernel("vector_add");
    EXPECT_NE(nullptr, kernel);
}

TEST(TestProgram, InvalidProgramName)
{
    SKIP_IF_NO_DEVICES();
    EXPECT_THROW(Program("bad_filename.cpp", {"-O3", "-DFLOAT=float"}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestProgram, CompileFails)
{
    SKIP_IF_NO_DEVICES();

    // Missing "TYPE" macro definition
    EXPECT_THROW(Program("vector_add.cpp", {"-O3"}), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestKernel, LaunchesVectorAdd)
{
    SKIP_IF_NO_DEVICES();

    constexpr int N = 256;

    // Allocate and initialize
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&devA, N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&devB, N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&devC, N * sizeof(float)));

    std::vector<float> hostA(N, 1.0f);
    std::vector<float> hostB(N, 2.0f);
    std::vector<float> hostC(N);

    ASSERT_EQ(hipSuccess, hipMemcpy(devA, hostA.data(), N * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(devB, hostB.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    const Program program("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    Kernel kernel(program, "vector_add");
    kernel.setBlockSize(256);
    kernel.setGridSize(1);
    kernel.launch(nullptr, devA, devB, devC, N);

    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_EQ(hipSuccess, hipMemcpy(hostC.data(), devC, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify
    EXPECT_FLOAT_EQ(3.0f, hostC[0]);
    EXPECT_FLOAT_EQ(3.0f, hostC[N - 1]);

    ASSERT_EQ(hipSuccess, hipFree(devA));
    ASSERT_EQ(hipSuccess, hipFree(devB));
    ASSERT_EQ(hipSuccess, hipFree(devC));
}

TEST(TestKernelDeviceBinding, RefusesALaunchItCannotBindTheDeviceFor)
{
    SKIP_IF_NO_DEVICES();

    int devices = 0;
    ASSERT_EQ(hipSuccess, hipGetDeviceCount(&devices));

    // An ordinal one past the last device can never be made current, so this stays
    // discriminating on the single-GPU hosts CI runs. The null function handle is never
    // dereferenced: the refusal lands before hipModuleLaunchKernel is reached.
    Kernel kernel(nullptr, "unbindable", devices);
    kernel.setBlockSize(1);
    kernel.setGridSize(1);

    try
    {
        kernel.launch(nullptr);
        FAIL() << "expected a launch onto a device that cannot be made current to be refused";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& failure)
    {
        const std::string message = failure.what();
        EXPECT_NE(message.find("cannot make device " + std::to_string(devices)), std::string::npos)
            << message;
        EXPECT_NE(message.find("unbindable"), std::string::npos)
            << "the message must name the kernel: " << message;
    }

    // The refused hipSetDevice left HIP error state behind on purpose; clear it or the
    // HipErrorHandler listener fails this test for it. Both stores, because the listener
    // reads hipExtGetLastError and clearing only the other one leaves it holding the error.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestKernelDeviceBinding, AKernelWithNoDeviceBindsNothing)
{
    SKIP_IF_NO_DEVICES();

    // Program-sourced kernels carry NO_DEVICE and must reach the launch without binding:
    // one Program is shared across devices, so a bind here would pin every MLOps launch to
    // whichever device compiled first.
    Kernel kernel(nullptr, "unbound", Kernel::NO_DEVICE);
    kernel.setBlockSize(1);
    kernel.setGridSize(1);

    try
    {
        kernel.launch(nullptr);
        FAIL() << "expected a null function handle to be refused by HIP";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& failure)
    {
        const std::string message = failure.what();
        EXPECT_NE(message.find("hipModuleLaunchKernel"), std::string::npos)
            << "a NO_DEVICE kernel must fail at the launch, never at a device bind: " << message;
        EXPECT_EQ(message.find("cannot make device"), std::string::npos)
            << "a NO_DEVICE kernel must not attempt a bind: " << message;
    }

    // The refused launch left HIP error state behind; clear both stores, as above.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}
