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

#ifndef TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_
#define TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_

#include "test_utils.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"

#include <rocprim/block/block_adjacent_difference.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store.hpp>

// Host (CPU) implementaions of the wrapping function that allows to pass 3 args
template<class T, class FlagType, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index)
    -> decltype(flag_op(b, a, b_index))
{
    return flag_op(b, a, b_index);
}

template<class T, class FlagType, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int) -> decltype(flag_op(b, a))
{
    return flag_op(b, a);
}

template<typename T>
struct test_op
{
    __host__ __device__
    T operator()(const T& a, const T& b) const
    {
        return (b + b) - a;
    }
};

template<typename T,
         typename Output,
         typename StorageType,
         typename BinaryFunction,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) void subtract_left_kernel(
    const T* input, StorageType* output)
{
    const unsigned int lid             = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset    = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage;

    Output thread_output[ItemsPerThread];

    if(blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.subtract_left(thread_items,
                                          thread_output,
                                          BinaryFunction{},
                                          tile_predecessor_item,
                                          storage);
    }
    else
    {
        adjacent_difference.subtract_left(thread_items, thread_output, BinaryFunction{}, storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<typename T,
         typename Output,
         typename StorageType,
         typename BinaryFunction,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(
    BlockSize,
    ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) void subtract_left_partial_kernel(const T*      input,
                                                                        unsigned int* tile_sizes,
                                                                        StorageType*  output)
{
    const unsigned int lid             = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset    = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage;

    Output thread_output[ItemsPerThread];

    const unsigned int tile_size = tile_sizes[blockIdx.x];
    if(blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.subtract_left_partial(thread_items,
                                                  thread_output,
                                                  BinaryFunction{},
                                                  tile_predecessor_item,
                                                  tile_size,
                                                  storage);
    }
    else
    {
        adjacent_difference.subtract_left_partial(thread_items,
                                                  thread_output,
                                                  BinaryFunction{},
                                                  tile_size,
                                                  storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<typename T,
         typename Output,
         typename StorageType,
         typename BinaryFunction,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(
    BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) void subtract_right_kernel(const T*     input,
                                                                            StorageType* output)
{
    const unsigned int lid             = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset    = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage;

    Output thread_output[ItemsPerThread];

    if(blockIdx.x % 2 == 0)
    {
        const T tile_successor_item = input[block_offset + items_per_block];
        adjacent_difference.subtract_right(thread_items,
                                           thread_output,
                                           BinaryFunction{},
                                           tile_successor_item,
                                           storage);
    }
    else
    {
        adjacent_difference.subtract_right(thread_items, thread_output, BinaryFunction{}, storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<typename T,
         typename Output,
         typename StorageType,
         typename BinaryFunction,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(
    BlockSize,
    ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) void subtract_right_partial_kernel(const T*      input,
                                                                         unsigned int* tile_sizes,
                                                                         StorageType*  output)
{
    const unsigned int lid             = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset    = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage;

    Output thread_output[ItemsPerThread];

    const unsigned int tile_size = tile_sizes[blockIdx.x];
    adjacent_difference.subtract_right_partial(thread_items,
                                               thread_output,
                                               BinaryFunction{},
                                               tile_size,
                                               storage);

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<typename T,
         typename Output,
         typename BinaryFunction,
         unsigned int Method,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 0>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    SCOPED_TRACE(testing::Message() << "items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "size = " << size);
    SCOPED_TRACE(testing::Message() << "grid_size = " << grid_size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        const std::vector<T> input
            = test_utils::get_random_data_wrapped<T>(size, 0, 10, seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        BinaryFunction           op;
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == 0)
                {
                    expected[i] = block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i];
                }
                else
                {
                    expected[i] = op(input[i], input[i - 1]);
                }
            }
        }

        // Preparing Device
        common::device_ptr<T>           d_input(input);
        common::device_ptr<stored_type> d_output(size);

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(subtract_left_kernel<T,
                                                                Output,
                                                                stored_type,
                                                                BinaryFunction,
                                                                block_size,
                                                                items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           d_input.get(),
                           d_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results
        const auto output = d_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output,
            expected,
            std::max(test_utils::precision<T>, test_utils::precision<stored_type>)));
    }
}

template<typename T,
         typename Output,
         typename BinaryFunction,
         unsigned int Method,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 1>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    SCOPED_TRACE(testing::Message() << "items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "size = " << size);
    SCOPED_TRACE(testing::Message() << "grid_size = " << grid_size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        const std::vector<T> input
            = test_utils::get_random_data_wrapped<T>(size, 0, 10, seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        BinaryFunction           op;
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == items_per_block - 1)
                {
                    expected[i] = block_index % 2 == 0 ? op(input[i], input[i + 1]) : input[i];
                }
                else
                {
                    expected[i] = op(input[i], input[i + 1]);
                }
            }
        }

        // Preparing Device
        common::device_ptr<T>           d_input(input);
        common::device_ptr<stored_type> d_output(size);

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(subtract_right_kernel<T,
                                                                 Output,
                                                                 stored_type,
                                                                 BinaryFunction,
                                                                 block_size,
                                                                 items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           d_input.get(),
                           d_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results
        const auto output = d_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output,
            expected,
            std::max(test_utils::precision<T>, test_utils::precision<stored_type>)));
    }
}

template<typename T,
         typename Output,
         typename BinaryFunction,
         unsigned int Method,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 2>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    SCOPED_TRACE(testing::Message() << "items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "size = " << size);
    SCOPED_TRACE(testing::Message() << "grid_size = " << grid_size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        const std::vector<T> input
            = test_utils::get_random_data_wrapped<T>(size, 0, 10, seed_value);

        const std::vector<unsigned int> tile_sizes
            = test_utils::get_random_data_wrapped<unsigned int>(grid_size,
                                                                0,
                                                                items_per_block,
                                                                seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        BinaryFunction           op;
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item < tile_sizes[block_index])
                {
                    if(item == 0)
                    {
                        expected[i] = block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i];
                    }
                    else
                    {
                        expected[i] = op(input[i], input[i - 1]);
                    }
                }
                else
                {
                    expected[i] = input[i];
                }
            }
        }

        // Preparing Device
        common::device_ptr<T>            d_input(input);
        common::device_ptr<unsigned int> d_tile_sizes(tile_sizes);
        common::device_ptr<stored_type>  d_output(size);

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(subtract_left_partial_kernel<T,
                                                                        Output,
                                                                        stored_type,
                                                                        BinaryFunction,
                                                                        block_size,
                                                                        items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           d_input.get(),
                           d_tile_sizes.get(),
                           d_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results
        const auto output = d_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output,
            expected,
            std::max(test_utils::precision<T>, test_utils::precision<stored_type>)));
    }
}

template<typename T,
         typename Output,
         typename BinaryFunction,
         unsigned int Method,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 3>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    SCOPED_TRACE(testing::Message() << "items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "size = " << size);
    SCOPED_TRACE(testing::Message() << "grid_size = " << grid_size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        const std::vector<T> input
            = test_utils::get_random_data_wrapped<T>(size, 0, 10, seed_value);

        const std::vector<unsigned int> tile_sizes
            = test_utils::get_random_data_wrapped<unsigned int>(grid_size,
                                                                0,
                                                                items_per_block,
                                                                seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        BinaryFunction           op;
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item < tile_sizes[block_index])
                {
                    if(item == tile_sizes[block_index] - 1 || item == items_per_block - 1)
                    {
                        expected[i] = input[i];
                    }
                    else
                    {
                        expected[i] = op(input[i], input[i + 1]);
                    }
                }
                else
                {
                    expected[i] = input[i];
                }
            }
        }

        // Preparing Device
        common::device_ptr<T>            d_input(input);
        common::device_ptr<unsigned int> d_tile_sizes(tile_sizes);
        common::device_ptr<stored_type>  d_output(size);

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(subtract_right_partial_kernel<T,
                                                                         Output,
                                                                         stored_type,
                                                                         BinaryFunction,
                                                                         block_size,
                                                                         items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           d_input.get(),
                           d_tile_sizes.get(),
                           d_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results
        const auto output = d_output.load();

        using is_add_op = test_utils::is_add_operator<BinaryFunction>;
        // clang-format off
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected,
            is_add_op::value
                ? std::max(test_utils::precision<T>, test_utils::precision<stored_type>)
                : std::is_same<T, stored_type>::value 
                    ? 0 
                    : test_utils::precision<stored_type>));
        // clang-format on
    }
}

// Static for-loop
template<unsigned int First,
         unsigned int Last,
         class Type,
         class FlagType,
         class FlagOpType,
         unsigned int Method,
         unsigned int BlockSize = 256U>
struct static_for
{
    static void run()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            int device_id = test_common_utils::obtain_device_from_ctest();
            SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
            HIP_CHECK(hipSetDevice(device_id));

            test_block_adjacent_difference<Type,
                                           FlagType,
                                           FlagOpType,
                                           Method,
                                           BlockSize,
                                           items[First]>();
        }
        static_for<First + 1, Last, Type, FlagType, FlagOpType, Method, BlockSize>::run();
    }
};

template<unsigned int N,
         class Type,
         class FlagType,
         class FlagOpType,
         unsigned int Method,
         unsigned int BlockSize>
struct static_for<N, N, Type, FlagType, FlagOpType, Method, BlockSize>
{
    static void run() {}
};

#endif // TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_
