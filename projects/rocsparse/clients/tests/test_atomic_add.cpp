/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// Include the actual rocsparse implementation for atomic_add
#include "rocsparse_common.hpp"
#include "rocsparse_data.hpp"

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// =============================================================================
// Type traits for atomic add tests
// =============================================================================
template <typename T>
struct AtomicAddTraits;

template <>
struct AtomicAddTraits<_Float16>
{
    static constexpr float add_value     = 0.125f;
    static constexpr float min_tolerance = 0.125f;
    static const char*     type_name()
    {
        return "Float16";
    }
};

template <>
struct AtomicAddTraits<rocsparse_bfloat16>
{
    static constexpr float add_value     = 0.125f;
    static constexpr float min_tolerance = 0.125f;
    static const char*     type_name()
    {
        return "BFloat16";
    }
};

// =============================================================================
// Template kernel using rocsparse::atomic_add
// =============================================================================
template <typename T>
__global__ void kernel_atomic_add(T* y, int size, int num_adds, int target_idx)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < num_adds)
    {
        rocsparse::atomic_add(y, target_idx, size, static_cast<T>(AtomicAddTraits<T>::add_value));
    }
}

// =============================================================================
// Template test function to avoid code duplication
// =============================================================================
template <typename T>
void test_atomic_add_accuracy(int size, int num_adds, int target_idx)
{
    using Traits = AtomicAddTraits<T>;

    if(target_idx >= size)
    {
        GTEST_SKIP() << "target_idx >= size, skipping";
    }

    T*         d_y;
    hipError_t hip_status;

    hip_status = hipMalloc(&d_y, size * sizeof(T));
    ASSERT_EQ(hip_status, hipSuccess) << "hipMalloc failed: " << hipGetErrorString(hip_status);

    std::vector<T> h_y(size, static_cast<T>(0.0f));
    hip_status = hipMemcpy(d_y, h_y.data(), size * sizeof(T), hipMemcpyHostToDevice);
    ASSERT_EQ(hip_status, hipSuccess) << "hipMemcpy H2D failed: " << hipGetErrorString(hip_status);

    int threads_per_block = 256;
    int num_blocks        = (num_adds + threads_per_block - 1) / threads_per_block;
    kernel_atomic_add<T><<<num_blocks, threads_per_block>>>(d_y, size, num_adds, target_idx);

    hip_status = hipGetLastError();
    ASSERT_EQ(hip_status, hipSuccess) << "Kernel launch failed: " << hipGetErrorString(hip_status);

    hip_status = hipDeviceSynchronize();
    ASSERT_EQ(hip_status, hipSuccess)
        << "hipDeviceSynchronize failed: " << hipGetErrorString(hip_status);

    hip_status = hipMemcpy(h_y.data(), d_y, size * sizeof(T), hipMemcpyDeviceToHost);
    ASSERT_EQ(hip_status, hipSuccess) << "hipMemcpy D2H failed: " << hipGetErrorString(hip_status);

    float result    = static_cast<float>(h_y[target_idx]);
    float expected  = static_cast<float>(num_adds) * Traits::add_value;
    float tolerance = std::max(Traits::min_tolerance, expected * 0.01f);

    EXPECT_NEAR(result, expected, tolerance)
        << Traits::type_name() << " atomic add failed: size=" << size << ", num_adds=" << num_adds
        << ", target_idx=" << target_idx
        << " (is_odd_last=" << ((size & 1) && target_idx == size - 1) << ")";

    for(int i = 0; i < size; i++)
    {
        if(i != target_idx)
        {
            EXPECT_EQ(static_cast<float>(h_y[i]), 0.0f)
                << Traits::type_name() << ": Element " << i << " should be 0";
        }
    }

    hip_status = hipFree(d_y);
    ASSERT_EQ(hip_status, hipSuccess) << "hipFree failed: " << hipGetErrorString(hip_status);
}

// =============================================================================
// Test classes using the template function
// =============================================================================
class atomic_add_f16_pre_checkin : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
protected:
    void SetUp() override
    {
        if(RocSPARSE_TestData::is_yaml_filter_active())
        {
            GTEST_SKIP() << "Skipping non-yaml test when --yaml filter is active";
        }
    }
};

class atomic_add_bf16_pre_checkin : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
protected:
    void SetUp() override
    {
        if(RocSPARSE_TestData::is_yaml_filter_active())
        {
            GTEST_SKIP() << "Skipping non-yaml test when --yaml filter is active";
        }
    }
};

TEST_P(atomic_add_f16_pre_checkin, Accuracy)
{
    auto [size, num_adds, target_idx] = GetParam();
    test_atomic_add_accuracy<_Float16>(size, num_adds, target_idx);
}

TEST_P(atomic_add_bf16_pre_checkin, Accuracy)
{
    auto [size, num_adds, target_idx] = GetParam();
    test_atomic_add_accuracy<rocsparse_bfloat16>(size, num_adds, target_idx);
}

// =============================================================================
// Parameterized test values
// Using 0.125 per add: max 256 adds = 32.0 (safely within both f16 and bf16 exact range)
// =============================================================================
static const auto atomic_add_test_values = ::testing::Values(
    // Even size arrays - target at index 0 (first element, even index)
    std::make_tuple(100, 1, 0),
    std::make_tuple(100, 64, 0),
    std::make_tuple(100, 256, 0),
    // Even size arrays - target at index 1 (second element, odd index)
    std::make_tuple(100, 1, 1),
    std::make_tuple(100, 64, 1),
    std::make_tuple(100, 256, 1),
    // Even size arrays - target at last element
    std::make_tuple(100, 1, 99),
    std::make_tuple(100, 64, 99),
    std::make_tuple(100, 256, 99),
    // Odd size arrays - target at index 0
    std::make_tuple(99, 1, 0),
    std::make_tuple(99, 64, 0),
    std::make_tuple(99, 256, 0),
    // Odd size arrays - target at index 1
    std::make_tuple(99, 1, 1),
    std::make_tuple(99, 64, 1),
    std::make_tuple(99, 256, 1),
    // Odd size arrays - target at LAST element (requires spinlock path!)
    std::make_tuple(99, 1, 98),
    std::make_tuple(99, 64, 98),
    std::make_tuple(99, 256, 98),
    // Larger odd size - last element (spinlock path)
    std::make_tuple(255, 1, 254),
    std::make_tuple(255, 64, 254),
    std::make_tuple(255, 256, 254),
    // Small arrays - edge cases
    std::make_tuple(1, 1, 0), // Single element (odd size, spinlock)
    std::make_tuple(1, 64, 0),
    std::make_tuple(1, 256, 0),
    std::make_tuple(2, 1, 0), // Two elements (even size)
    std::make_tuple(2, 64, 0),
    std::make_tuple(2, 256, 0),
    std::make_tuple(2, 1, 1),
    std::make_tuple(2, 64, 1),
    std::make_tuple(2, 256, 1));

INSTANTIATE_TEST_SUITE_P(Float16, atomic_add_f16_pre_checkin, atomic_add_test_values);
INSTANTIATE_TEST_SUITE_P(BFloat16, atomic_add_bf16_pre_checkin, atomic_add_test_values);
