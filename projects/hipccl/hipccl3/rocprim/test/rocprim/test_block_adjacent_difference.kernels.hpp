// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

// Host (CPU) implementaions of the wrapping function that allows to pass 3 args
template <class T, class FlagType, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index)
    -> decltype(flag_op(b, a, b_index))
{
    return flag_op(b, a, b_index);
}

template <class T, class FlagType, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int) -> decltype(flag_op(b, a))
{
    return flag_op(b, a);
}

template<typename T>
struct test_op
{
    __host__ __device__ T operator()(const T& a, const T& b) const
    {
        return (b + b) - a;
    }
};

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void flag_heads_kernel(Type* device_input, long long* device_heads)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_adjacent_difference<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];

    // Still need to test it even tough its deprecated
    ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
    if(blockIdx.x % 2 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads(head_flags, tile_predecessor_item, input, FlagOpType());
    }
    else
    {
        bdiscontinuity.flag_heads(head_flags, input, FlagOpType());
    }
    ROCPRIM_CLANG_SUPPRESS_WARNING_POP

    rocprim::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void subtract_left_kernel(const T* input, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage; 

    Output thread_output[ItemsPerThread];

    if(blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.subtract_left(
            thread_items, thread_output, BinaryFunction{}, tile_predecessor_item, storage);
    }
    else
    {
        adjacent_difference.subtract_left(thread_items, thread_output, BinaryFunction{}, storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void subtract_left_partial_kernel(const T* input, unsigned int* tile_sizes, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

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
                                                  BinaryFunction {},
                                                  tile_predecessor_item,
                                                  tile_size,
                                                  storage);
    }
    else
    {
        adjacent_difference.subtract_left_partial(
            thread_items, thread_output, BinaryFunction {}, tile_size, storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void subtract_right_kernel(const T* input, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage; 

    Output thread_output[ItemsPerThread];

    if(blockIdx.x % 2 == 0)
    {
        const T tile_successor_item = input[block_offset + items_per_block];
        adjacent_difference.subtract_right(
            thread_items, thread_output, BinaryFunction{}, tile_successor_item, storage);
    }
    else
    {
        adjacent_difference.subtract_right(thread_items, thread_output, BinaryFunction{}, storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void subtract_right_partial_kernel(const T* input, unsigned int* tile_sizes, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage; 

    Output thread_output[ItemsPerThread];

    const unsigned int tile_size = tile_sizes[blockIdx.x];
    adjacent_difference.subtract_right_partial(
        thread_items, thread_output, BinaryFunction {}, tile_size, storage);

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void flag_tails_kernel(Type* device_input, long long* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_adjacent_difference<Type, BlockSize> bdiscontinuity;

    FlagType tail_flags[ItemsPerThread];

    // Still need to test it even tough its deprecated
    ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
    if(blockIdx.x % 2 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_tails(tail_flags, tile_successor_item, input, FlagOpType());
    }
    else
    {
        bdiscontinuity.flag_tails(tail_flags, input, FlagOpType());
    }
    ROCPRIM_CLANG_SUPPRESS_WARNING_POP

    rocprim::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void flag_heads_and_tails_kernel(Type* device_input, long long* device_heads, long long* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_adjacent_difference<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    FlagType tail_flags[ItemsPerThread];

    // Still need to test it even tough its deprecated
    ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
    if(blockIdx.x % 4 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 2)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 3)
    {
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, input, FlagOpType());
    }
    ROCPRIM_CLANG_SUPPRESS_WARNING_POP

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
auto test_block_adjacent_difference()
-> typename std::enable_if<Method == 0>::type
{
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    static constexpr size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<long long> heads(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_heads(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = bi % 2 == 1
                        ? apply<Type, FlagType, FlagOpType>(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply<Type, FlagType, FlagOpType>(flag_op, input[i - 1], input[i], ii);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_heads), heads.size() * sizeof(typename decltype(heads)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_heads_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_heads
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                heads.data(), device_heads,
                heads.size() * sizeof(typename decltype(heads)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(heads[i], expected_heads[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_heads));
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
auto test_block_adjacent_difference()
-> typename std::enable_if<Method == 1>::type
{
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    static constexpr size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<long long> tails(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = bi % 2 == 0
                        ? apply<Type, FlagType, FlagOpType>(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply<Type, FlagType, FlagOpType>(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_tails;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_tails), tails.size() * sizeof(typename decltype(tails)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_tails_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_tails
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                tails.data(), device_tails,
                tails.size() * sizeof(typename decltype(tails)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(tails[i], expected_tails[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_tails));
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
auto test_block_adjacent_difference()
-> typename std::enable_if<Method == 2>::type
{
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    static constexpr size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<long long> heads(size);
        std::vector<long long> tails(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_heads(size);
        std::vector<stored_flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = (bi % 4 == 1 || bi % 4 == 2)
                        ? apply<Type, FlagType, FlagOpType>(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply<Type, FlagType, FlagOpType>(flag_op, input[i - 1], input[i], ii);
                }
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = (bi % 4 == 0 || bi % 4 == 1)
                        ? apply<Type, FlagType, FlagOpType>(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply<Type, FlagType, FlagOpType>(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_heads), tails.size() * sizeof(typename decltype(heads)::value_type)));
        long long* device_tails;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_tails), tails.size() * sizeof(typename decltype(tails)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_heads_and_tails_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_heads, device_tails
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                heads.data(), device_heads,
                heads.size() * sizeof(typename decltype(heads)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                tails.data(), device_tails,
                tails.size() * sizeof(typename decltype(tails)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(heads[i], expected_heads[i]);
            ASSERT_EQ(tails[i], expected_tails[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_heads));
        HIP_CHECK(hipFree(device_tails));
    }


}

template <typename T,
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

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<T>     input = test_utils::get_random_data<T>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

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
        T*           d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(hipMemcpy(
            d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

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
                           d_input,
                           d_output);
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(hipMemcpy(
            output.data(), d_output, output.size() * sizeof(output[0]), hipMemcpyDeviceToHost));

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output, expected, test_utils::precision_threshold<T>::percentage));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}

template <typename T,
          typename Output,
          typename BinaryFunction,
          unsigned int Method,
          unsigned int BlockSize,
          unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 4>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<T>     input = test_utils::get_random_data<T>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

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
        T*           d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(hipMemcpy(
            d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

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
                           d_input,
                           d_output);
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(hipMemcpy(
            output.data(), d_output, output.size() * sizeof(output[0]), hipMemcpyDeviceToHost));

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output, expected, test_utils::precision_threshold<T>::percentage));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}

template <typename T,
          typename Output,
          typename BinaryFunction,
          unsigned int Method,
          unsigned int BlockSize,
          unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 5>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<T>     input = test_utils::get_random_data<T>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        const std::vector<unsigned int> tile_sizes
            = test_utils::get_random_data<unsigned int>(grid_size, 0, items_per_block, seed_value);

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
        T*            d_input;
        unsigned int* d_tile_sizes;
        stored_type*  d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(hipMemcpy(
            d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_tile_sizes,
                            tile_sizes.data(),
                            tile_sizes.size() * sizeof(tile_sizes[0]),
                            hipMemcpyHostToDevice));

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
                           d_input,
                           d_tile_sizes,
                           d_output);
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(hipMemcpy(
            output.data(), d_output, output.size() * sizeof(output[0]), hipMemcpyDeviceToHost));

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output, expected, test_utils::precision_threshold<T>::percentage));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
    }
}

template <typename T,
          typename Output,
          typename BinaryFunction,
          unsigned int Method,
          unsigned int BlockSize,
          unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 6>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<T>     input = test_utils::get_random_data<T>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        const std::vector<unsigned int> tile_sizes
            = test_utils::get_random_data<unsigned int>(grid_size, 0, items_per_block, seed_value);

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
        T*            d_input;
        unsigned int* d_tile_sizes;
        stored_type*  d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(hipMemcpy(
            d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_tile_sizes,
                            tile_sizes.data(),
                            tile_sizes.size() * sizeof(tile_sizes[0]),
                            hipMemcpyHostToDevice));

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
                           d_input,
                           d_tile_sizes,
                           d_output);
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(hipMemcpy(
            output.data(), d_output, output.size() * sizeof(output[0]), hipMemcpyDeviceToHost));

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output, expected, test_utils::precision_threshold<T>::percentage));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
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
        int device_id = test_common_utils::obtain_device_from_ctest();
        SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
        HIP_CHECK(hipSetDevice(device_id));

        test_block_adjacent_difference<Type, FlagType, FlagOpType, Method, BlockSize, items[First]>();
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

#endif // TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_
