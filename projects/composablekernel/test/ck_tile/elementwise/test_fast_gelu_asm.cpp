// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

// FastGeluAsm's float asm writes its scratch register before it reads x and c2. If the scratch
// is not earlyclobber, the register allocator may give it c2's register, and the device functor
// then computes x / (1 + exp((1 + c1) x^3)) instead of FastGelu.

namespace {

using ck_tile::index_t;
using ck_tile::element_wise::FastGeluAsm;

constexpr index_t kElements  = 1024;
constexpr index_t kBlockSize = 256;

constexpr float kLowest  = -6.f;
constexpr float kHighest = 6.f;

// Applies the functor to one element per work-item.
__global__ void fast_gelu_asm(const float* x, float* y, index_t n)
{
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
    {
        float out;
        FastGeluAsm{}(out, x[i]);
        y[i] = out;
    }
}

} // namespace

TEST(FastGeluAsm, FloatMatchesHost)
{
    std::vector<float> x(kElements);
    for(index_t i = 0; i < kElements; ++i)
    {
        x[i] = kLowest + (kHighest - kLowest) * static_cast<float>(i) / (kElements - 1);
    }

    ck_tile::DeviceMem d_x(kElements * sizeof(float));
    ck_tile::DeviceMem d_y(kElements * sizeof(float));
    d_x.ToDevice(x.data());

    const index_t grid_size = (kElements + kBlockSize - 1) / kBlockSize;
    fast_gelu_asm<<<grid_size, kBlockSize>>>(static_cast<const float*>(d_x.GetDeviceBuffer()),
                                             static_cast<float*>(d_y.GetDeviceBuffer()),
                                             kElements);
    ASSERT_EQ(hipGetLastError(), hipSuccess);

    std::vector<float> y(kElements);
    d_y.FromDevice(y.data());

    for(index_t i = 0; i < kElements; ++i)
    {
        // The host overload evaluates the same formula with exp and a true division.
        float expected;
        FastGeluAsm{}(expected, x[i]);

        const float tolerance = 1e-5f + 1e-5f * std::abs(expected);
        EXPECT_NEAR(y[i], expected, tolerance) << "x = " << x[i];
    }
}
