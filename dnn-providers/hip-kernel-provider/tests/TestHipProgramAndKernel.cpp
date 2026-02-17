// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "hip/HipKernel.hpp"
#include "hip/HipProgram.hpp"

#include <vector>

TEST(TestHipProgram, CompilesAndGetsKernel)
{
    HipProgram program("vector_add.cpp", {"-O3"});
    hipFunction_t kernel = program.GetKernel("vector_add");
    EXPECT_NE(nullptr, kernel);
}

TEST(TestHipKernel, LaunchesVectorAdd)
{
    constexpr int N = 256;

    // Allocate and initialize
    float *d_a, *d_b, *d_c;
    ASSERT_EQ(hipSuccess, hipMalloc(&d_a, N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_b, N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_c, N * sizeof(float)));

    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N);

    ASSERT_EQ(hipSuccess, hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    HipProgram program("vector_add.cpp", {"-O3"});
    HipKernel kernel(program, "vector_add");
    kernel.SetBlockSize(256);
    kernel.SetGridSize(1);
    kernel.Launch(nullptr, d_a, d_b, d_c, N);

    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_EQ(hipSuccess, hipMemcpy(h_c.data(), d_c, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify
    EXPECT_FLOAT_EQ(3.0f, h_c[0]);
    EXPECT_FLOAT_EQ(3.0f, h_c[N - 1]);

    ASSERT_EQ(hipSuccess, hipFree(d_a));
    ASSERT_EQ(hipSuccess, hipFree(d_b));
    ASSERT_EQ(hipSuccess, hipFree(d_c));
}
