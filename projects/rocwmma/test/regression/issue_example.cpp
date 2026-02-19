/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Example regression test demonstrating the infrastructure
//
// This is a working example showing how to write a minimal regression test.
// It can be used to verify the regression test infrastructure is working correctly.
//
// This file can be removed once real regression tests are added.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>
#include "common.hpp"
#include "hip_device.hpp"

// Simple kernel that writes thread indices to output
__global__ void example_kernel(int* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        output[idx] = idx;
    }
}

// Example test that verifies basic device functionality
TEST(RegressionIssue_Example, BasicDeviceOperation)
{
    // Check if a ROCm device is available
    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

    if(deviceCount == 0)
    {
        GTEST_SKIP() << "No ROCm-capable device available";
    }

    // Get device info using HipDevice singleton
    auto& device = rocwmma::HipDevice::instance();
    EXPECT_GT(device->warpSize(), 0);

    // Allocate device memory
    const int size = 256;
    int*      d_output;
    CHECK_HIP_ERROR(hipMalloc(&d_output, size * sizeof(int)));

    // Launch kernel
    dim3 blockDim(256);
    dim3 gridDim(1);
    hipLaunchKernelGGL(example_kernel, gridDim, blockDim, 0, 0, d_output, size);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Copy results back
    std::vector<int> h_output(size);
    CHECK_HIP_ERROR(
        hipMemcpy(h_output.data(), d_output, size * sizeof(int), hipMemcpyDeviceToHost));

    // Verify results
    for(int i = 0; i < size; i++)
    {
        EXPECT_EQ(h_output[i], i) << "Mismatch at index " << i;
    }

    // Clean up
    CHECK_HIP_ERROR(hipFree(d_output));
}

// Example test showing how to skip based on architecture
TEST(RegressionIssue_Example, ArchitectureSpecificTest)
{
    auto& device = rocwmma::HipDevice::instance();

    // This test only runs on specific architectures
    auto arch = device->getGcnArch();
    if(arch != rocwmma::HipDevice::GFX908 && arch != rocwmma::HipDevice::GFX90A
       && arch != rocwmma::HipDevice::GFX942 && arch != rocwmma::HipDevice::GFX1100
       && arch != rocwmma::HipDevice::GFX1200)
    {
        GTEST_SKIP() << "Test skipped for architecture: " << arch;
    }

    // Test would run here for supported architectures
    EXPECT_TRUE(true);
}

// Example test showing minimal test structure
TEST(RegressionIssue_Example, MinimalTest)
{
    // This is the simplest possible test
    // Just verify we can run GTest tests
    EXPECT_TRUE(true);
}
