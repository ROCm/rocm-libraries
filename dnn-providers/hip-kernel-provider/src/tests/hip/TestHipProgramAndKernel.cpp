// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "hip/HipKernel.hpp"
#include "hip/HipProgram.hpp"

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <vector>

using namespace hip_kernel_provider;

TEST(TestHipProgram, CompilesAndGetsKernel)
{
    SKIP_IF_NO_DEVICES();

    HipProgram program("vector_add.cpp", {"-O3"});
    hipFunction_t kernel = program.getKernel("vector_add");
    EXPECT_NE(nullptr, kernel);
}

TEST(TestHipKernel, LaunchesVectorAdd)
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
    HipProgram program("vector_add.cpp", {"-O3"});
    HipKernel kernel(program, "vector_add");
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
