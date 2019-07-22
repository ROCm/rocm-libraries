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

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"
#include "test_utils_types.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

namespace rp = rocprim;

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimWarpScanTests : public ::testing::Test {
public:
    using params = Params;
};

TYPED_TEST_CASE(RocprimWarpScanTests, WarpParams);

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_inclusive_scan_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = rp::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().inclusive_scan(value, value, storage[warp_id]);

    device_output[index] = value;
}

TYPED_TEST(RocprimWarpScanTests, InclusiveScan)
{
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
        ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
        : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output(size);
    std::vector<T> expected(output.size(), 0);

    // Calculate expected results on host
    binary_op_type binary_op;
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = apply(binary_op, input[idx], expected[j > 0 ? idx-1 : idx]);
        }
    }

    // Writing to device memory
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
        HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
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

    // Validating results
    test_utils::assert_near(output, expected, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_inclusive_scan_reduce_kernel(
    T* device_input,
    T* device_output,
    T* device_output_reductions)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + ( hipBlockIdx_x * BlockSize );

    T value = device_input[index];
    T reduction;

    using wscan_t = rp::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().inclusive_scan(value, value, reduction, storage[warp_id]);

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

TYPED_TEST(RocprimWarpScanTests, InclusiveScanReduce)
{
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output(size);
    std::vector<T> output_reductions(size / logical_warp_size);
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);

    // Calculate expected results on host
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = apply(binary_op, input[idx], expected[j > 0 ? idx-1 : idx]);
        }
        expected_reductions[i] = expected[(i+1) * logical_warp_size - 1];
    }

    // Writing to device memory
    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_reductions;
    HIP_CHECK(
        hipMalloc(
            &device_output_reductions,
            output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output, device_output_reductions
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

    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Validating results
    test_utils::assert_near(output, expected, 0.01);
    test_utils::assert_near(output_reductions, expected_reductions, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_exclusive_scan_kernel(T* device_input, T* device_output, T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = rp::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().exclusive_scan(value, value, init, storage[warp_id]);

    device_output[index] = value;
}

TYPED_TEST(RocprimWarpScanTests, ExclusiveScan)
{
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
        ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
        : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output(size);
    std::vector<T> expected(input.size(), 0);
    const T init = test_utils::get_random_value(0, 100);

    // Calculate expected results on host
    binary_op_type binary_op;
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = apply(binary_op, input[idx-1], expected[idx-1]);
        }
    }

    // Writing to device memory
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
        HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output, init
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

    // Validating results
    test_utils::assert_near(output, expected, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_exclusive_scan_reduce_kernel(
    T* device_input,
    T* device_output,
    T* device_output_reductions,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];
    T reduction;

    using wscan_t = rp::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().exclusive_scan(value, value, init, reduction, storage[warp_id]);

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

TYPED_TEST(RocprimWarpScanTests, ExclusiveReduceScan)
{
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
        ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
        : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output(size);
    std::vector<T> output_reductions(size / logical_warp_size);
    std::vector<T> expected(input.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    const T init = test_utils::get_random_value(0, 100);

    // Calculate expected results on host
    binary_op_type binary_op;
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = apply(binary_op, input[idx-1], expected[idx-1]);
        }

        expected_reductions[i] = 0;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected_reductions[i] = apply(binary_op, expected_reductions[i], input[idx]);
        }
    }

    // Writing to device memory
    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_reductions;
    HIP_CHECK(
        hipMalloc(
            &device_output_reductions,
            output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output, device_output_reductions, init
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

    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Validating results
    test_utils::assert_near(output, expected, 0.01);
    test_utils::assert_near(output_reductions, expected_reductions, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_scan_kernel(
    T* device_input,
    T* device_inclusive_output,
    T* device_exclusive_output,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T input = device_input[index];
    T inclusive_output, exclusive_output;

    using wscan_t = rp::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().scan(input, inclusive_output, exclusive_output, init, storage[warp_id]);

    device_inclusive_output[index] = inclusive_output;
    device_exclusive_output[index] = exclusive_output;
}

TYPED_TEST(RocprimWarpScanTests, Scan)
{
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
        ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
        : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output_inclusive(size);
    std::vector<T> output_exclusive(size);
    std::vector<T> expected_inclusive(output_inclusive.size(), 0);
    std::vector<T> expected_exclusive(output_exclusive.size(), 0);
    const T init = test_utils::get_random_value(0, 100);

    // Calculate expected results on host
    binary_op_type binary_op;
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        expected_exclusive[i * logical_warp_size] = init;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected_inclusive[idx] = apply(binary_op, input[idx], expected_inclusive[j > 0 ? idx-1 : idx]);
            if(j > 0)
            {
                expected_exclusive[idx] = apply(binary_op, input[idx-1], expected_exclusive[idx-1]);
            }
        }
    }

    // Writing to device memory
    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_inclusive_output;
    HIP_CHECK(
        hipMalloc(
            &device_inclusive_output,
            output_inclusive.size() * sizeof(typename decltype(output_inclusive)::value_type)
        )
    );
    T* device_exclusive_output;
    HIP_CHECK(
        hipMalloc(
            &device_exclusive_output,
            output_exclusive.size() * sizeof(typename decltype(output_exclusive)::value_type)
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_scan_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_inclusive_output, device_exclusive_output, init
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output_inclusive.data(), device_inclusive_output,
            output_inclusive.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            output_exclusive.data(), device_exclusive_output,
            output_exclusive.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Validating results
    test_utils::assert_near(output_inclusive, expected_inclusive, 0.01);
    test_utils::assert_near(output_exclusive, expected_exclusive, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_inclusive_output));
    HIP_CHECK(hipFree(device_exclusive_output));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
void warp_scan_reduce_kernel(
    T* device_input,
    T* device_inclusive_output,
    T* device_exclusive_output,
    T* device_output_reductions,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rp::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T input = device_input[index];
    T inclusive_output, exclusive_output, reduction;

    using wscan_t = rp::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().scan(input, inclusive_output, exclusive_output, init, reduction, storage[warp_id]);

    device_inclusive_output[index] = inclusive_output;
    device_exclusive_output[index] = exclusive_output;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

TYPED_TEST(RocprimWarpScanTests, ScanReduce)
{
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
        ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
        : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output_inclusive(size);
    std::vector<T> output_exclusive(size);
    std::vector<T> output_reductions(size / logical_warp_size);
    std::vector<T> expected_inclusive(output_inclusive.size(), 0);
    std::vector<T> expected_exclusive(output_exclusive.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    const T init = test_utils::get_random_value(0, 100);

    // Calculate expected results on host
    binary_op_type binary_op;
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        expected_exclusive[i * logical_warp_size] = init;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected_inclusive[idx] = apply(binary_op, input[idx], expected_inclusive[j > 0 ? idx-1 : idx]);
            if(j > 0)
            {
                expected_exclusive[idx] = apply(binary_op, input[idx-1], expected_exclusive[idx-1]);
            }
        }
        expected_reductions[i] = expected_inclusive[(i+1) * logical_warp_size - 1];
    }

    // Writing to device memory
    T* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    T* device_inclusive_output;
    HIP_CHECK(
        hipMalloc(
            &device_inclusive_output,
            output_inclusive.size() * sizeof(typename decltype(output_inclusive)::value_type)
        )
    );
    T* device_exclusive_output;
    HIP_CHECK(
        hipMalloc(
            &device_exclusive_output,
            output_exclusive.size() * sizeof(typename decltype(output_exclusive)::value_type)
        )
    );
    T* device_output_reductions;
    HIP_CHECK(
        hipMalloc(
            &device_output_reductions,
            output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_scan_reduce_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input,
        device_inclusive_output, device_exclusive_output, device_output_reductions, init
    );

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output_inclusive.data(), device_inclusive_output,
            output_inclusive.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            output_exclusive.data(), device_exclusive_output,
            output_exclusive.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Validating results
    test_utils::assert_near(output_inclusive, expected_inclusive, 0.01);
    test_utils::assert_near(output_exclusive, expected_exclusive, 0.01);
    test_utils::assert_near(output_reductions, expected_reductions, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_inclusive_output));
    HIP_CHECK(hipFree(device_exclusive_output));
}

TYPED_TEST(RocprimWarpScanTests, InclusiveScanCustomType)
{
    using base_type = typename TestFixture::params::type;
    using T = test_utils::custom_test_type<base_type>;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
        ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
        : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input(size);
    std::vector<T> output(size);
    std::vector<T> expected(output.size(), T(0));

    // Initializing input data
    {
        auto random_values =
            test_utils::get_random_data<base_type>(2 * input.size(), 0, 100);
        for(size_t i = 0; i < input.size(); i++)
        {
            input[i].x = random_values[i];
            input[i].y = random_values[i + input.size()];
        }
    }

    // Calculate expected results on host
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = input[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    // Writing to device memory
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
        HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size, logical_warp_size>),
        dim3(grid_size), dim3(block_size), 0, 0,
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

    // Validating results
    test_utils::assert_near(output, expected, 0.01);

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}
