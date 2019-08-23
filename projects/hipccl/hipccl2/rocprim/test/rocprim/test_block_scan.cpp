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

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

namespace rp = rocprim;

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimBlockScanSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::input_type;
    static constexpr unsigned int block_size = Params::block_size;
};

TYPED_TEST_CASE(RocprimBlockScanSingleValueTests, BlockParams);

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 0>::type* = nullptr
>
__global__
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)init;
    (void)device_output_b;
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rp::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.inclusive_scan(value, value);
    device_output[index] = value;
}

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 1>::type* = nullptr
>
__global__
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)init;
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    T reduction;
    rp::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.inclusive_scan(value, value, reduction);
    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_b[hipBlockIdx_x] = reduction;
    }
}

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 2>::type* = nullptr
>
__global__
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T prefix_value = init;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value = prefix_value + reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = rp::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().inclusive_scan(value, value, storage, prefix_callback, rp::plus<T>());

    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_b[hipBlockIdx_x] = prefix_value;
    }
}

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 3>::type* = nullptr
>
__global__
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)device_output_b;
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rp::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.exclusive_scan(value, value, init);
    device_output[index] = value;
}

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 4>::type* = nullptr
>
__global__
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    T reduction;
    rp::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.exclusive_scan(value, value, init, reduction);
    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_b[hipBlockIdx_x] = reduction;
    }
}

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 5>::type* = nullptr
>
__global__
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T prefix_value = init;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value = prefix_value + reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = rp::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().exclusive_scan(value, value, storage, prefix_callback, rp::plus<T>());

    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_b[hipBlockIdx_x] = prefix_value;
    }
}

template <
    class T,
    unsigned int BlockSize,
    rp::block_scan_algorithm Algorithm,
    int Method
>
struct static_run_algo
{
    static void run(std::vector<T>& output,
                    std::vector<T>& output_b,
                    std::vector<T>& expected,
                    std::vector<T>& expected_b,
                    T* device_output,
                    T* device_output_b,
                    T init,
                    size_t grid_size)
    {
        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(scan_kernel<Method, BlockSize, Algorithm, T>),
            dim3(grid_size), dim3(BlockSize), 0, 0,
            device_output, device_output_b, init
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        if(device_output_b)
        {
            HIP_CHECK(
                hipMemcpy(
                    output_b.data(), device_output_b,
                    output_b.size() * sizeof(T),
                    hipMemcpyDeviceToHost
                )
            );
        }

        // Verifying results
        test_utils::assert_near(output, expected, 0.01);
        if(device_output_b)
        {
            test_utils::assert_near(output_b, expected_b, 0.01);
        }
    }
};

TYPED_TEST(RocprimBlockScanSingleValueTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output2 = output;

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = apply(binary_op, output[idx], expected[j > 0 ? idx-1 : idx]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    static_run_algo<T, block_size, rp::block_scan_algorithm::using_warp_scan, 0>::run(
        output, output, expected, expected,
        device_output, NULL, T(0), grid_size
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::reduce_then_scan, 0>::run(
        output2, output2, expected, expected,
        device_output, NULL, T(0), grid_size
    );

    HIP_CHECK(hipFree(device_output));
}

TYPED_TEST(RocprimBlockScanSingleValueTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output2 = output;
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = apply(binary_op, output[idx], expected[j > 0 ? idx-1 : idx]);
        }
        expected_reductions[i] = expected[(i+1) * block_size - 1];
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_reductions;
    HIP_CHECK(
        hipMalloc(
              &device_output_reductions,
              output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
        )
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::using_warp_scan, 1>::run(
        output, output_reductions, expected, expected_reductions,
        device_output, device_output_reductions, T(0), grid_size
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::reduce_then_scan, 1>::run(
        output2, output_reductions, expected, expected_reductions,
        device_output, device_output_reductions, T(0), grid_size
    );

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

TYPED_TEST(RocprimBlockScanSingleValueTests, InclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output2 = output;
    std::vector<T> output_block_prefixes(size / block_size);
    T block_prefix = test_utils::get_random_value<T>(0, 5);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = block_prefix;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = apply(binary_op, output[idx], expected[j > 0 ? idx-1 : idx]);
        }
        expected_block_prefixes[i] = expected[(i+1) * block_size - 1];
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_bp;
    HIP_CHECK(
        hipMalloc(
              &device_output_bp,
              output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
        )
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::using_warp_scan, 2>::run(
        output, output_block_prefixes, expected, expected_block_prefixes,
        device_output, device_output_bp, block_prefix, grid_size
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::reduce_then_scan, 2>::run(
        output2, output_block_prefixes, expected, expected_block_prefixes,
        device_output, device_output_bp, block_prefix, grid_size
    );

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_bp));
}

TYPED_TEST(RocprimBlockScanSingleValueTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output2 = output;
    const T init = test_utils::get_random_value<T>(0, 5);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = init;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = apply(binary_op, output[idx-1], expected[idx-1]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    static_run_algo<T, block_size, rp::block_scan_algorithm::using_warp_scan, 3>::run(
        output, output, expected, expected,
        device_output, NULL, init, grid_size
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::reduce_then_scan, 3>::run(
        output2, output2, expected, expected,
        device_output, NULL, init, grid_size
    );

    HIP_CHECK(hipFree(device_output));
}

TYPED_TEST(RocprimBlockScanSingleValueTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output2 = output;
    const T init = test_utils::get_random_value<T>(0, 5);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = init;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = apply(binary_op, output[idx-1], expected[idx-1]);
        }

        expected_reductions[i] = 0;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected_reductions[i] = apply(binary_op, expected_reductions[i], output[idx]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_reductions;
    HIP_CHECK(
        hipMalloc(
              &device_output_reductions,
              output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
        )
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::using_warp_scan, 4>::run(
        output, output_reductions, expected, expected_reductions,
        device_output, device_output_reductions, init, grid_size
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::reduce_then_scan, 4>::run(
        output2, output_reductions, expected, expected_reductions,
        device_output, device_output_reductions, init, grid_size
    );

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

TYPED_TEST(RocprimBlockScanSingleValueTests, ExclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_plus, rp::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output2 = output;
    std::vector<T> output_block_prefixes(size / block_size);
    T block_prefix = test_utils::get_random_value<T>(0, 5);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = block_prefix;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = apply(binary_op, output[idx-1], expected[idx-1]);
        }

        expected_block_prefixes[i] = block_prefix;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected_block_prefixes[i] = apply(binary_op, expected_block_prefixes[i], output[idx]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_bp;
    HIP_CHECK(
        hipMalloc(
              &device_output_bp,
              output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
        )
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::using_warp_scan, 5>::run(
        output, output_block_prefixes, expected, expected_block_prefixes,
        device_output, device_output_bp, block_prefix, grid_size
    );

    static_run_algo<T, block_size, rp::block_scan_algorithm::reduce_then_scan, 5>::run(
        output2, output_block_prefixes, expected, expected_block_prefixes,
        device_output, device_output_bp, block_prefix, grid_size
    );

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_bp));
}

// ---------------------------------------------------------
// Test for scan ops taking array of values as input
// ---------------------------------------------------------

template<class Params>
class RocprimBlockScanInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::input_type;
    static constexpr unsigned int block_size = Params::block_size;
};

TYPED_TEST_CASE(RocprimBlockScanInputArrayTests, BlockParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void inclusive_scan_array_kernel(T* device_output)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize ) + hipThreadIdx_x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rp::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.inclusive_scan(in_out, in_out, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void inclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize ) + hipThreadIdx_x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rp::block_scan<T, BlockSize, Algorithm> bscan;
    T reduction;
    bscan.inclusive_scan(in_out, in_out, reduction, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void inclusive_scan_array_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value = BinaryOp()(prefix_value, reduction);
        return prefix;
    };

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = rp::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().inclusive_scan(in_out, in_out, storage, prefix_callback, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_bp[hipBlockIdx_x] = prefix_value;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void exclusive_scan_array_kernel(T* device_output, T init)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rp::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.exclusive_scan(in_out, in_out, init, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void exclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions, T init)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rp::block_scan<T, BlockSize, Algorithm> bscan;
    T reduction;
    bscan.exclusive_scan(in_out, in_out, init, reduction, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void exclusive_scan_prefix_callback_array_kernel(
    T* device_output,
    T* device_output_bp,
    T block_prefix
)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value = BinaryOp()(prefix_value, reduction);
        return prefix;
    };

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index+ j];
    }

    using bscan_t = rp::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().exclusive_scan(in_out, in_out, storage, prefix_callback, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_bp[hipBlockIdx_x] = prefix_value;
    }
}

// Test for scan
template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_scan_algorithm Algorithm = rp::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 0>::type
{
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_maximum, rp::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 100);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = apply(binary_op, output[idx], expected[j > 0 ? idx-1 : idx]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(inclusive_scan_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output
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
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_scan_algorithm Algorithm = rp::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 1>::type
{
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_maximum, rp::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 100);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size, 0);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = apply(binary_op, output[idx], expected[j > 0 ? idx-1 : idx]);
        }
        expected_reductions[i] = expected[(i+1) * items_per_block - 1];
    }

    // Writing to device memory
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
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_output_reductions, output_reductions.data(),
            output_reductions.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(inclusive_scan_reduce_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_reductions
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
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_reductions, expected_reductions));

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_scan_algorithm Algorithm = rp::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 2>::type
{
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_maximum, rp::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 100);
    std::vector<T> output_block_prefixes(size / items_per_block, 0);
    T block_prefix = test_utils::get_random_value<T>(0, 100);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        expected[i * items_per_block] = block_prefix;
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = apply(binary_op, output[idx], expected[j > 0 ? idx-1 : idx]);
        }
        expected_block_prefixes[i] = expected[(i+1) * items_per_block - 1];
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_bp;
    HIP_CHECK(
        hipMalloc(
              &device_output_bp,
              output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_output_bp, output_block_prefixes.data(),
            output_block_prefixes.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            inclusive_scan_array_prefix_callback_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>
        ),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_bp, block_prefix
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
            output_block_prefixes.data(), device_output_bp,
            output_block_prefixes.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Validating results
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_block_prefixes, expected_block_prefixes));

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_bp));
}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_scan_algorithm Algorithm = rp::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 3>::type
{
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_maximum, rp::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 100);
    const T init = test_utils::get_random_value<T>(0, 100);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        expected[i * items_per_block] = init;
        for(size_t j = 1; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = apply(binary_op, output[idx-1], expected[idx-1]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(exclusive_scan_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, init
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
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_scan_algorithm Algorithm = rp::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 4>::type
{
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_maximum, rp::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 100);

    // Output reduce results
    std::vector<T> output_reductions(size / items_per_block);
    const T init = test_utils::get_random_value<T>(0, 100);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        expected[i * items_per_block] = init;
        for(size_t j = 1; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = apply(binary_op, output[idx-1], expected[idx-1]);
        }
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected_reductions[i] = apply(binary_op, expected_reductions[i], output[idx]);
        }
    }

    // Writing to device memory
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
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            exclusive_scan_reduce_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>
        ),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_reductions, init
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
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_reductions, expected_reductions));
}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_scan_algorithm Algorithm = rp::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 5>::type
{
    using binary_op_type = typename std::conditional<std::is_same<T, rp::half>::value, test_utils::half_maximum, rp::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 100);
    std::vector<T> output_block_prefixes(size / items_per_block);
    T block_prefix = test_utils::get_random_value<T>(0, 100);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        expected[i * items_per_block] = block_prefix;
        for(size_t j = 1; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = apply(binary_op, output[idx-1], expected[idx-1]);
        }
        expected_block_prefixes[i] = block_prefix;
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected_block_prefixes[i] = apply(binary_op, expected_block_prefixes[i], output[idx]);
        }
    }

    // Writing to device memory
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    T* device_output_bp;
    HIP_CHECK(
        hipMalloc(
              &device_output_bp,
              output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            exclusive_scan_prefix_callback_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>
        ),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_bp, block_prefix
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
            output_block_prefixes.data(), device_output_bp,
            output_block_prefixes.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Validating results
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_block_prefixes, expected_block_prefixes));

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_bp));
}

// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class T,
    int Method,
    unsigned int BlockSize = 256U
>
struct static_for_input_array
{
    static void run()
    {
        test_block_scan_input_arrays<T, Method, BlockSize, items[First], rp::block_scan_algorithm::using_warp_scan>();
        test_block_scan_input_arrays<T, Method, BlockSize, items[First], rp::block_scan_algorithm::reduce_then_scan>();
        static_for_input_array<First + 1, Last, T, Method, BlockSize>::run();
    }
};

template <
    unsigned int N,
    class T,
    int Method,
    unsigned int BlockSize
>
struct static_for_input_array<N, N, T, Method, BlockSize>
{
    static void run()
    {
    }
};

TYPED_TEST(RocprimBlockScanInputArrayTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 0, block_size>::run();
}

TYPED_TEST(RocprimBlockScanInputArrayTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 1, block_size>::run();
}

TYPED_TEST(RocprimBlockScanInputArrayTests, InclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 2, block_size>::run();
}

TYPED_TEST(RocprimBlockScanInputArrayTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 3, block_size>::run();
}

TYPED_TEST(RocprimBlockScanInputArrayTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 4, block_size>::run();
}

TYPED_TEST(RocprimBlockScanInputArrayTests, ExclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 5, block_size>::run();
}
