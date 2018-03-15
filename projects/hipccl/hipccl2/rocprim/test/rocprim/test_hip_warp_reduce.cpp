// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

namespace rp = rocprim;

template<
    class T,
    unsigned int WarpSize
>
struct params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
};

template<class Params>
class RocprimWarpReduceTests : public ::testing::Test {
public:
    using params = Params;
};


typedef ::testing::Types<
    // shuffle based reduce
    params<int, 2U>,
    params<int, 4U>,
    params<int, 8U>,
    params<int, 16U>,
    params<int, 32U>,
    params<int, 64U>,
    params<float, 2U>,
    params<float, 4U>,
    params<float, 8U>,
    params<float, 16U>,
    params<float, 32U>,
    params<float, 64U>,
    // shared memory reduce
    params<int, 3U>,
    params<int, 7U>,
    params<int, 15U>,
    params<int, 37U>,
    params<int, 61U>,
    params<float, 3U>,
    params<float, 7U>,
    params<float, 15U>,
    params<float, 37U>,
    params<float, 61U>
> Params;

TYPED_TEST_CASE(RocprimWarpReduceTests, Params);

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_reduce_sum_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rp::warp_reduce<T, LogicalWarpSize>;
    tile_static typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, storage[warp_id]);

    if(hipThreadIdx_x%LogicalWarpSize == 0)
    {
        device_output[index/LogicalWarpSize] = value;
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSum)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using T = typename TestFixture::params::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100); // used for input
    std::vector<T> output(input.size() / logical_warp_size, 0);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 1);
    for(size_t i = 0; i < output.size(); i++)
    {
        T value = 0;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        expected[i] = value;
    }

    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size, logical_warp_size>),
        dim3(size/block_size), dim3(block_size), 0, 0,
        device_input, device_output
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < output.size(); i++)
    {
        if (std::is_integral<T>::value)
        {
            EXPECT_EQ(output[i], expected[i]);
        }
        else if(std::is_floating_point<T>::value)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            EXPECT_NEAR(output[i], expected[i], tolerance);
        }
    }

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_allreduce_sum_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rp::warp_reduce<T, LogicalWarpSize, true>;
    tile_static typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, storage[warp_id]);

    device_output[index] = value;
}

TYPED_TEST(RocprimWarpReduceTests, AllReduceSum)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using T = typename TestFixture::params::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100); // used for input
    std::vector<T> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        T value = 0;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        for (size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = value;
        }
    }

    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size, logical_warp_size>),
        dim3(size/block_size), dim3(block_size), 0, 0,
        device_input, device_output
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < output.size(); i++)
    {
        if (std::is_integral<T>::value)
        {
            EXPECT_EQ(output[i], expected[i]);
        }
        else if(std::is_floating_point<T>::value)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            EXPECT_NEAR(output[i], expected[i], tolerance);
        }
    }

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_reduce_sum_kernel(T* device_input, T* device_output, size_t valid)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rp::warp_reduce<T, LogicalWarpSize>;
    tile_static typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, valid, storage[warp_id]);

    if(hipThreadIdx_x%LogicalWarpSize == 0)
    {
        device_output[index/LogicalWarpSize] = value;
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumValid)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using T = typename TestFixture::params::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;
    const size_t valid = logical_warp_size - 1;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100); // used for input
    std::vector<T> output(input.size() / logical_warp_size, 0);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 1);
    for(size_t i = 0; i < output.size(); i++)
    {
        T value = 0;
        for(size_t j = 0; j < valid; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        expected[i] = value;
    }

    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size, logical_warp_size>),
        dim3(size/block_size), dim3(block_size), 0, 0,
        device_input, device_output, valid
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < output.size(); i++)
    {
        if (std::is_integral<T>::value)
        {
            EXPECT_EQ(output[i], expected[i]);
        }
        else if(std::is_floating_point<T>::value)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            EXPECT_NEAR(output[i], expected[i], tolerance);
        }
    }

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}


template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_allreduce_sum_kernel(T* device_input, T* device_output, size_t valid)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rp::warp_reduce<T, LogicalWarpSize, true>;
    tile_static typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, valid, storage[warp_id]);

    device_output[index] = value;
}

TYPED_TEST(RocprimWarpReduceTests, AllReduceSumValid)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using T = typename TestFixture::params::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;
    const size_t valid = logical_warp_size - 1;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100); // used for input
    std::vector<T> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        T value = 0;
        for(size_t j = 0; j < valid; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        for (size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = value;
        }
    }

    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size, logical_warp_size>),
        dim3(size/block_size), dim3(block_size), 0, 0,
        device_input, device_output, valid
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < output.size(); i++)
    {
        if (std::is_integral<T>::value)
        {
            EXPECT_EQ(output[i], expected[i]);
        }
        else if(std::is_floating_point<T>::value)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            EXPECT_NEAR(output[i], expected[i], tolerance);
        }
    }

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumCustomStruct)
{
    using base_type = typename TestFixture::params::type;
    using T = test_utils::custom_test_type<base_type>;

    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input(size);
    {
        auto random_values =
            test_utils::get_random_data<base_type>(2 * input.size(), 0, 100);
        for(size_t i = 0; i < input.size(); i++)
        {
            input[i].x = random_values[i];
            input[i].y = random_values[i + input.size()];
        }
    }
    std::vector<T> output(input.size() / logical_warp_size);

    // Calculate expected results on host
    std::vector<T> expected(output.size());
    for(size_t i = 0; i < output.size(); i++)
    {
        T value(0, 0);
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            value = value + input[idx];
        }
        expected[i] = value;
    }

    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size, logical_warp_size>),
        dim3(size/block_size), dim3(block_size), 0, 0,
        device_input, device_output
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < output.size(); i++)
    {
        auto diffx = std::max<base_type>(std::abs(0.1f * expected[i].x), base_type(0.01f));
        if(std::is_integral<base_type>::value) diffx = 0;
        ASSERT_NEAR(output[i].x, expected[i].x, diffx);

        auto diffy = std::max<base_type>(std::abs(0.1f * expected[i].y), base_type(0.01f));
        if(std::is_integral<base_type>::value) diffy = 0;
        ASSERT_NEAR(output[i].y, expected[i].y, diffy);
    }

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}
