// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BLOCK_SCAN_KERNELS_HPP_
#define TEST_BLOCK_SCAN_KERNELS_HPP_

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include <rocprim/block/block_scan.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

template<
    int Method,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    typename std::enable_if<Method == 0>::type* = nullptr
>
__global__
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)init;
    (void)device_output_b;
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
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
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)init;
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    T reduction;
    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.inclusive_scan(value, value, reduction);
    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_b[blockIdx.x] = reduction;
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
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T prefix_value = init;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value = prefix_value + reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = rocprim::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().inclusive_scan(value, value, storage, prefix_callback, rocprim::plus<T>());

    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_b[blockIdx.x] = prefix_value;
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
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)device_output_b;
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
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
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    T reduction;
    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.exclusive_scan(value, value, init, reduction);
    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_b[blockIdx.x] = reduction;
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
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T prefix_value = init;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value = prefix_value + reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = rocprim::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().exclusive_scan(value, value, storage, prefix_callback, rocprim::plus<T>());

    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_b[blockIdx.x] = prefix_value;
    }
}

template<int                           Method,
         unsigned int                  BlockSize,
         rocprim::block_scan_algorithm Algorithm,
         class T,
         typename std::enable_if<Method == 6>::type* = nullptr>
__global__
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    (void)device_output_b;
    const unsigned int                           index    = (blockIdx.x * BlockSize) + threadIdx.x;
    T                                            input[1] = {device_output[index]};
    T                                            output[1];
    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.inclusive_scan(input, init, output);
    device_output[index] = output[0];
}

template<int                           Method,
         unsigned int                  BlockSize,
         rocprim::block_scan_algorithm Algorithm,
         class T,
         typename std::enable_if<Method == 7>::type* = nullptr>
__global__
__launch_bounds__(BlockSize)
void scan_kernel(T* device_output, T* device_output_b, T init)
{
    const unsigned int                           index    = (blockIdx.x * BlockSize) + threadIdx.x;
    T                                            input[1] = {device_output[index]};
    T                                            output[1];
    T                                            reduction;
    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
    bscan.inclusive_scan(input, init, output, reduction);
    device_output[index] = output[0];
    if(threadIdx.x == 0)
    {
        device_output_b[blockIdx.x] = reduction;
    }
}

template <
    class T,
    unsigned int BlockSize,
    rocprim::block_scan_algorithm Algorithm,
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
        HIP_CHECK(hipGetLastError());

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
        test_utils::assert_near(output, expected, test_utils::precision<T> * BlockSize);
        if(device_output_b)
        {
            test_utils::assert_near(output_b, expected_b, test_utils::precision<T> * BlockSize);
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_scan_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_array_kernel(T* device_output)
{
    const unsigned int index = ((blockIdx.x * BlockSize ) + threadIdx.x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
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
__launch_bounds__(BlockSize)
void inclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((blockIdx.x * BlockSize ) + threadIdx.x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
    T reduction;
    bscan.inclusive_scan(in_out, in_out, reduction, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
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
__launch_bounds__(BlockSize)
void inclusive_scan_array_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
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

    using bscan_t = rocprim::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().inclusive_scan(in_out, in_out, storage, prefix_callback, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_bp[blockIdx.x] = prefix_value;
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
__launch_bounds__(BlockSize)
void exclusive_scan_array_kernel(T* device_output, T init)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
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
__launch_bounds__(BlockSize)
void exclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions, T init)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_scan<T, BlockSize, Algorithm> bscan;
    T reduction;
    bscan.exclusive_scan(in_out, in_out, init, reduction, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
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
__launch_bounds__(BlockSize)
void exclusive_scan_prefix_callback_array_kernel(
    T* device_output,
    T* device_output_bp,
    T block_prefix
)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
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

    using bscan_t = rocprim::block_scan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::storage_type storage;
    bscan_t().exclusive_scan(in_out, in_out, storage, prefix_callback, BinaryOp());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_bp[blockIdx.x] = prefix_value;
    }
}

// Test for scan
template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_scan_algorithm Algorithm = rocprim::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 0>::type
{
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data_wrapped<T>(size, 2, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = binary_op(output[idx], expected[j > 0 ? idx-1 : idx]);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_output(output);

        // Launching kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inclusive_scan_array_kernel<block_size,
                                                                       items_per_thread,
                                                                       algorithm,
                                                                       T,
                                                                       binary_op_type>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_output.get());

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }

}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_scan_algorithm Algorithm = rocprim::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 1>::type
{
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data_wrapped<T>(size, 2, 100, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, T(0));

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = binary_op(output[idx], expected[j > 0 ? idx-1 : idx]);
            }
            expected_reductions[i] = expected[(i+1) * items_per_block - 1];
        }

        // Writing to device memory
        common::device_ptr<T> device_output(output);
        common::device_ptr<T> device_output_reductions(output_reductions);

        // Launching kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inclusive_scan_reduce_array_kernel<block_size,
                                                                              items_per_thread,
                                                                              algorithm,
                                                                              T,
                                                                              binary_op_type>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_output.get(),
                           device_output_reductions.get());

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output            = device_output.load();
        output_reductions = device_output_reductions.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_reductions, expected_reductions));
    }

}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_scan_algorithm Algorithm = rocprim::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 2>::type
{
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data_wrapped<T>(size, 2, 100, seed_value);
        std::vector<T> output_block_prefixes(size / items_per_block, T(0));
        T block_prefix = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = block_prefix;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = binary_op(output[idx], expected[j > 0 ? idx-1 : idx]);
            }
            expected_block_prefixes[i] = expected[(i+1) * items_per_block - 1];
        }

        // Writing to device memory
        common::device_ptr<T> device_output(output);
        common::device_ptr<T> device_output_bp(output_block_prefixes);

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_array_prefix_callback_kernel<block_size,
                                                                        items_per_thread,
                                                                        algorithm,
                                                                        T,
                                                                        binary_op_type>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_output.get(),
            device_output_bp.get(),
            block_prefix);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output                = device_output.load();
        output_block_prefixes = device_output_bp.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output_block_prefixes, expected_block_prefixes));
    }

}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_scan_algorithm Algorithm = rocprim::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 3>::type
{
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size            = items_per_block * 19;
    const size_t grid_size       = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data_wrapped<T>(size, 2, 100, seed_value);
        const T init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = init;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = binary_op(output[idx-1], expected[idx-1]);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_output(output);

        // Launching kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(exclusive_scan_array_kernel<block_size,
                                                                       items_per_thread,
                                                                       algorithm,
                                                                       T,
                                                                       binary_op_type>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_output.get(),
                           init);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }

}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_scan_algorithm Algorithm = rocprim::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 4>::type
{
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data_wrapped<T>(size, 2, 100, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / items_per_block);
        const T init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = init;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = binary_op(output[idx-1], expected[idx-1]);
            }
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected_reductions[i] = binary_op(expected_reductions[i], output[idx]);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_output(output);
        common::device_ptr<T> device_output_reductions(output_reductions.size());

        // Launching kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(exclusive_scan_reduce_array_kernel<block_size,
                                                                              items_per_thread,
                                                                              algorithm,
                                                                              T,
                                                                              binary_op_type>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_output.get(),
                           device_output_reductions.get(),
                           init);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output            = device_output.load();
        output_reductions = device_output_reductions.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_reductions, expected_reductions));
    }

}

template<
    class T,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_scan_algorithm Algorithm = rocprim::block_scan_algorithm::using_warp_scan
>
auto test_block_scan_input_arrays()
-> typename std::enable_if<Method == 5>::type
{
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data_wrapped<T>(size, 2, 100, seed_value);
        std::vector<T> output_block_prefixes(size / items_per_block);
        T block_prefix = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = block_prefix;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = binary_op(output[idx-1], expected[idx-1]);
            }
            expected_block_prefixes[i] = block_prefix;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected_block_prefixes[i] = binary_op(expected_block_prefixes[i], output[idx]);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_output(output);
        common::device_ptr<T> device_output_bp(output_block_prefixes.size());

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(exclusive_scan_prefix_callback_array_kernel<block_size,
                                                                        items_per_thread,
                                                                        algorithm,
                                                                        T,
                                                                        binary_op_type>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_output.get(),
            device_output_bp.get(),
            block_prefix);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output                = device_output.load();
        output_block_prefixes = device_output_bp.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output_block_prefixes, expected_block_prefixes));
    }

}

// Static for-loop
template<unsigned int First, unsigned int Last, class T, int Method, unsigned int BlockSize = 256U>
struct static_for_input_array
{
    static void run()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            int device_id = test_common_utils::obtain_device_from_ctest();
            SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
            HIP_CHECK(hipSetDevice(device_id));

            test_block_scan_input_arrays<T,
                                         Method,
                                         BlockSize,
                                         items[First],
                                         rocprim::block_scan_algorithm::using_warp_scan>();
            test_block_scan_input_arrays<T,
                                         Method,
                                         BlockSize,
                                         items[First],
                                         rocprim::block_scan_algorithm::reduce_then_scan>();
        }
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

#endif // TEST_BLOCK_SCAN_KERNELS_HPP_
