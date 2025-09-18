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

#ifndef TEST_BLOCK_DISCONTINUITY_KERNELS_HPP_
#define TEST_BLOCK_DISCONTINUITY_KERNELS_HPP_

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"
#include "test_utils.hpp"
#include "test_utils_data_generation.hpp"

#include <rocprim/block/block_discontinuity.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

template<class T>
struct custom_flag_op1
{
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b, unsigned int b_index) const
    {
        return (a == b) || (b_index % 10 == 0);
    }
};

struct custom_flag_op2
{
    template<class T>
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return (a - b > 5);
    }
};

// Host (CPU) implementaions of the wrapping function that allows to pass 3 args
template <class T, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index)
    -> decltype(flag_op(a, b, b_index))
{
    return flag_op(a, b, b_index);
}

template<class T, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int) -> decltype(flag_op(a, b))
{
    return flag_op(a, b);
}

template<class Type,
         class FlagType,
         class FlagOpType,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void flag_heads_kernel(Type* device_input, FlagType* device_heads)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_discontinuity<Type, BlockSize>              bdiscontinuity;
    __shared__ typename decltype(bdiscontinuity)::storage_type storage;

    FlagType head_flags[ItemsPerThread];
    if(blockIdx.x % 2 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads(head_flags, tile_predecessor_item, input, FlagOpType(), storage);
    }
    else
    {
        bdiscontinuity.flag_heads(head_flags, input, FlagOpType(), storage);
    }

    rocprim::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
}

template<class Type,
         class FlagType,
         class FlagOpType,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void flag_tails_kernel(Type* device_input, FlagType* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_discontinuity<Type, BlockSize>              bdiscontinuity;
    __shared__ typename decltype(bdiscontinuity)::storage_type storage;

    FlagType tail_flags[ItemsPerThread];
    if(blockIdx.x % 2 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_tails(tail_flags, tile_successor_item, input, FlagOpType(), storage);
    }
    else
    {
        bdiscontinuity.flag_tails(tail_flags, input, FlagOpType(), storage);
    }

    rocprim::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
}

template<class Type,
         class FlagType,
         class FlagOpType,
         unsigned int BlockSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void flag_heads_and_tails_kernel(Type* device_input, FlagType* device_heads, FlagType* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_discontinuity<Type, BlockSize>              bdiscontinuity;
    __shared__ typename decltype(bdiscontinuity)::storage_type storage;

    FlagType head_flags[ItemsPerThread];
    FlagType tail_flags[ItemsPerThread];
    if(blockIdx.x % 4 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags,
                                            tail_flags,
                                            tile_successor_item,
                                            input,
                                            FlagOpType(),
                                            storage);
    }
    else if(blockIdx.x % 4 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        const Type tile_successor_item   = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags,
                                            tile_predecessor_item,
                                            tail_flags,
                                            tile_successor_item,
                                            input,
                                            FlagOpType(),
                                            storage);
    }
    else if(blockIdx.x % 4 == 2)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads_and_tails(head_flags,
                                            tile_predecessor_item,
                                            tail_flags,
                                            input,
                                            FlagOpType(),
                                            storage);
    }
    else if(blockIdx.x % 4 == 3)
    {
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, input, FlagOpType(), storage);
    }

    rocprim::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
    rocprim::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
auto test_block_discontinuity()
-> typename std::enable_if<Method == 0>::type
{
    using type                               = Type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

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
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<type> input
            = test_utils::get_random_data_wrapped<type>(size, 0, 10, seed_value);

        // Calculate expected results on host
        std::vector<flag_type> expected_heads(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = bi % 2 == 1
                                            ? flag_type(apply(flag_op, input[i - 1], input[i], ii))
                                            : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
                }
            }
        }

        // Preparing Device
        common::device_ptr<type>      device_input(input);
        common::device_ptr<flag_type> device_heads(size);

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_heads_kernel<type, flag_type, flag_op_type, block_size, items_per_thread>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_input.get(),
            device_heads.get());
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        const auto heads = device_heads.load_to_unique_ptr();
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(heads.get(),
                                                      heads.get() + size,
                                                      expected_heads.begin(),
                                                      expected_heads.end()));
    }
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
auto test_block_discontinuity()
-> typename std::enable_if<Method == 1>::type
{
    using type                               = Type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

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
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<type> input
            = test_utils::get_random_data_wrapped<type>(size, 0, 10, seed_value);

        // Calculate expected results on host
        std::vector<flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == items_per_block - 1)
                {
                    expected_tails[i]
                        = bi % 2 == 0 ? flag_type(apply(flag_op, input[i], input[i + 1], ii + 1))
                                      : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        common::device_ptr<type>      device_input(input);
        common::device_ptr<flag_type> device_tails(size);

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_tails_kernel<type, flag_type, flag_op_type, block_size, items_per_thread>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_input.get(),
            device_tails.get());
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        const auto tails = device_tails.load_to_unique_ptr();
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(tails.get(),
                                                      tails.get() + size,
                                                      expected_tails.begin(),
                                                      expected_tails.end()));
    }
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
auto test_block_discontinuity()
-> typename std::enable_if<Method == 2>::type
{
    using type                               = Type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

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
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<type> input
            = test_utils::get_random_data_wrapped<type>(size, 0, 10, seed_value);

        // Calculate expected results on host
        std::vector<flag_type> expected_heads(size);
        std::vector<flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = (bi % 4 == 1 || bi % 4 == 2)
                                            ? flag_type(apply(flag_op, input[i - 1], input[i], ii))
                                            : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
                }
                if(ii == items_per_block - 1)
                {
                    expected_tails[i]
                        = (bi % 4 == 0 || bi % 4 == 1)
                              ? flag_type(apply(flag_op, input[i], input[i + 1], ii + 1))
                              : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        common::device_ptr<type>      device_input(input);
        common::device_ptr<flag_type> device_heads(size);
        common::device_ptr<flag_type> device_tails(size);

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(flag_heads_and_tails_kernel<type,
                                                                       flag_type,
                                                                       flag_op_type,
                                                                       block_size,
                                                                       items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_input.get(),
                           device_heads.get(),
                           device_tails.get());
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        const auto heads = device_heads.load_to_unique_ptr();
        const auto tails = device_tails.load_to_unique_ptr();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(heads.get(),
                                                      heads.get() + size,
                                                      expected_heads.begin(),
                                                      expected_heads.end()));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(tails.get(),
                                                      tails.get() + size,
                                                      expected_tails.begin(),
                                                      expected_tails.end()));
    }
}

// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize = 256U
>
struct static_for
{
    static void run()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            int device_id = test_common_utils::obtain_device_from_ctest();
            SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
            HIP_CHECK(hipSetDevice(device_id));

            test_block_discontinuity<Type, FlagType, FlagOpType, Method, BlockSize, items[First]>();
        }
        static_for<First + 1, Last, Type, FlagType, FlagOpType, Method, BlockSize>::run();
    }
};

template <
    unsigned int N,
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize
>
struct static_for<N, N, Type, FlagType, FlagOpType, Method, BlockSize>
{
    static void run()
    {
    }
};

#endif // TEST_BLOCK_DISCONTINUITY_KERNELS_HPP_
