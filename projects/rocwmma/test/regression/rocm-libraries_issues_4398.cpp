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

// Regression test for issue #4398
// https://github.com/ROCm/rocm-libraries/issues/4398
//
// Bug: Compile failure due to multiple partial template specializations
//      matching for half-precision fragment compute/accumulator type.
// Fix: Explicitly excluded the catch-all partial specialization for the case of
//      half-precision fragments with half-precision compute/accumulator type.
//
// This test verifies that the kernel compiles properly when targeting
// half-precision fragments with half-precision compute/accumulator type, and
// that it can be launched successfully on a ROCm-capable device.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>
#include "common.hpp"
#include "hip_device.hpp"

static __global__ void kernel()
{
    constexpr int frag_m = 16;
    constexpr int frag_n = 16;
    constexpr int D = 128;
    constexpr int VKQ_stride = 128;
    constexpr int ncols = 32;
    typedef rocwmma::fragment<rocwmma::matrix_a,    frag_m, frag_n, 16, half, rocwmma::col_major> frag_a_V;
    typedef rocwmma::fragment<rocwmma::matrix_b,    frag_m, frag_n, 16, half, rocwmma::col_major> frag_b;
    typedef rocwmma::fragment<rocwmma::accumulator, frag_m, frag_n, 16, half> frag_c_VKQ;
    frag_c_VKQ VKQ_c;
    frag_b KQ_b;
    frag_a_V v_a;
    rocwmma::mma_sync(VKQ_c, v_a, KQ_b, VKQ_c);
}

// Example test that verifies basic device functionality
TEST(GitHub_rocm_libraries_issues_4398, HalfHalfHalfTest)
{
    // Check if a ROCm device is available
    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

    if(deviceCount == 0)
    {
        GTEST_SKIP() << "No ROCm-capable device available";
    }

    // Launch kernel
    dim3 blockDim(256);
    dim3 gridDim(1);
    hipLaunchKernelGGL(kernel, gridDim, blockDim, 0, 0);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
}

