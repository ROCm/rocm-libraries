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

#include "common_test_header.hpp"

// required rocprim headers
#include <hipcub/config.hpp>
#include <hipcub/block/block_adjacent_difference.hpp>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/thread/thread_operators.hpp>

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class FlagOp
>
struct params
{
    using type = T;
    using flag_type = Flag;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    using flag_op_type = FlagOp;
};

template<class Params>
class HipcubBlockAdjacentDifference : public ::testing::Test {
public:
    using params = Params;
};

template<class T>
struct custom_flag_op1
{
    HIPCUB_HOST_DEVICE
    bool operator()(const T& a, const T& b, int b_index) const
    {
        return (a == b) || (b_index % 10 == 0);
    }
};

template<class T>
struct custom_flag_op2
{
    HIPCUB_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return (a - b > 5);
    }
};

// Host (CPU) implementations of the wrapping function that allows to pass 3 args
template<class T, class FlagType, class FlagOp>
typename std::enable_if<hipcub::detail::WithBIndexArg<T, FlagOp>::value, FlagType>::type
apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index)
{
    return flag_op(b, a, b_index);
}

template<class T, class FlagType, class FlagOp>
typename std::enable_if<!hipcub::detail::WithBIndexArg<T, FlagOp>::value, FlagType>::type
apply(FlagOp flag_op, const T& a, const T& b, unsigned int)
{
    return flag_op(b, a);
}

using Params = ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1, hipcub::Equality>,
    params<int, bool, 128U, 1, hipcub::Inequality>,
    params<float, int, 256U, 1, test_utils::less>,
#ifndef __HIP_PLATFORM_NVIDIA__
    // Nvidia does not handle half and bfloat16 properly
    params<test_utils::half, int, 256U, 1, test_utils::less>,
    params<test_utils::bfloat16, int, 256U, 1, test_utils::less>,
#endif
    params<char, char, 1024U, 1, test_utils::less_equal>,
    params<int, bool, 256U, 1, custom_flag_op1<int>>,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U, 1, test_utils::greater>,
    params<float, int, 37U, 1, custom_flag_op1<float>>,

#ifndef __HIP_PLATFORM_NVIDIA__
    params<test_utils::half, int, 37U, 1, test_utils::greater>,
    params<test_utils::bfloat16, int, 37U, 1, test_utils::greater>,
#endif

    params<long long, char, 510U, 1, test_utils::greater_equal>,
    params<unsigned int, long long, 162U, 1, hipcub::Inequality>,
    params<unsigned char, bool, 255U, 1, hipcub::Equality>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<int, char, 64U, 2, custom_flag_op2<int>>,
    params<int, short, 128U, 4, test_utils::less>,
    params<unsigned short, unsigned char, 256U, 7, custom_flag_op2<unsigned short>>,
    params<short, short, 512U, 8, hipcub::Equality>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, int, 33U, 5, custom_flag_op2<double>>,
    params<double, unsigned int, 464U, 2, hipcub::Equality>,
#ifndef __HIP_PLATFORM_NVIDIA__
    params<test_utils::half, int, 37U, 5, test_utils::greater>,
    params<test_utils::bfloat16, int, 37U, 3, test_utils::greater>,
#endif
    params<unsigned short, int, 100U, 3, test_utils::greater>,
    params<short, bool, 234U, 9, custom_flag_op1<short>>>;

TYPED_TEST_SUITE(HipcubBlockAdjacentDifference, Params);

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_heads_kernel(Type* device_input, long long* device_heads)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    Type input[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_input + block_offset, input);

    hipcub::BlockAdjacentDifference<Type, BlockSize> bAdjacentDiff;

    FlagType head_flags[ItemsPerThread];

    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    if(hipBlockIdx_x % 2 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bAdjacentDiff.FlagHeads(head_flags, input, FlagOpType(), tile_predecessor_item);
    }
    else
    {
        bAdjacentDiff.FlagHeads(head_flags, input, FlagOpType());
    }
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

    hipcub::StoreDirectBlocked(lid, device_heads + block_offset, head_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_tails_kernel(Type* device_input, long long* device_tails)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    Type input[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_input + block_offset, input);

    hipcub::BlockAdjacentDifference<Type, BlockSize> bAdjacentDiff;

    FlagType tail_flags[ItemsPerThread];

    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    if(hipBlockIdx_x % 2 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bAdjacentDiff.FlagTails(tail_flags, input, FlagOpType(), tile_successor_item);
    }
    else
    {
        bAdjacentDiff.FlagTails(tail_flags, input, FlagOpType());
    }
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

    hipcub::StoreDirectBlocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_heads_and_tails_kernel(Type* device_input, long long* device_heads, long long* device_tails)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    Type input[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_input + block_offset, input);

    hipcub::BlockAdjacentDifference<Type, BlockSize> bAdjacentDiff;

    FlagType head_flags[ItemsPerThread];
    FlagType tail_flags[ItemsPerThread];

    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    if(hipBlockIdx_x % 4 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bAdjacentDiff.FlagHeadsAndTails(head_flags, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(hipBlockIdx_x % 4 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bAdjacentDiff.FlagHeadsAndTails(head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(hipBlockIdx_x % 4 == 2)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bAdjacentDiff.FlagHeadsAndTails(head_flags, tile_predecessor_item, tail_flags, input, FlagOpType());
    }
    else if(hipBlockIdx_x % 4 == 3)
    {
        bAdjacentDiff.FlagHeadsAndTails(head_flags, tail_flags, input, FlagOpType());
    }
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

    hipcub::StoreDirectBlocked(lid, device_heads + block_offset, head_flags);
    hipcub::StoreDirectBlocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class T,
    class Output,
    class BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct params_subtract
{
    using type = T;
    using output = Output;
    using binary_function = BinaryFunction;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

template<class ParamsSubtract>
class HipcubBlockAdjacentDifferenceSubtract : public ::testing::Test {
public:
    using params_subtract = ParamsSubtract;
};

struct custom_op1
{
    template<class T>
    HIPCUB_HOST_DEVICE
    T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

struct custom_op2
{
    template<class T>
    HIPCUB_HOST_DEVICE
    T operator()(const T& a, const T& b) const
    {
        return (b + b) - a;
    }
};

using ParamsSubtract
    = ::testing::Types<params_subtract<unsigned int, int, hipcub::Sum, 64U, 1>,
                       params_subtract<int, bool, custom_op1, 128U, 1>,
                       params_subtract<float, int, custom_op2, 256U, 1>,
                       params_subtract<test_utils::half, int, custom_op1, 256U, 1>,
                       params_subtract<test_utils::bfloat16, int, custom_op2, 256U, 1>,
                       params_subtract<int, bool, custom_op1, 256U, 1>,

                       params_subtract<float, int, hipcub::Sum, 37U, 1>,
                       params_subtract<long long, char, custom_op1, 510U, 1>,
                       params_subtract<unsigned int, long long, custom_op2, 162U, 1>,
                       params_subtract<unsigned char, bool, hipcub::Sum, 255U, 1>,

                       params_subtract<int, char, custom_op1, 64U, 2>,
                       params_subtract<int, short, custom_op2, 128U, 4>,
                       params_subtract<unsigned short, unsigned char, hipcub::Sum, 256U, 7>,
                       params_subtract<short, short, custom_op1, 512U, 8>,

                       params_subtract<double, int, custom_op2, 33U, 5>,
                       params_subtract<double, unsigned int, hipcub::Sum, 464U, 2>,
                       params_subtract<unsigned short, int, custom_op1, 100U, 3>,
                       params_subtract<short, bool, custom_op2, 234U, 9>>;

TYPED_TEST_SUITE(HipcubBlockAdjacentDifferenceSubtract, ParamsSubtract);

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void subtract_left_kernel(const T* input, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    if (blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.SubtractLeft(thread_items, thread_output, BinaryFunction{}, tile_predecessor_item);
    }
    else
    {
        adjacent_difference.SubtractLeft(thread_items, thread_output, BinaryFunction{});
    }

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
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
__launch_bounds__(BlockSize)
void subtract_left_partial_tile_kernel(const T* input, int* tile_sizes, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    int tile_size = tile_sizes[blockIdx.x];

    if(blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.SubtractLeftPartialTile(thread_items,
                                                    thread_output,
                                                    BinaryFunction{},
                                                    tile_size,
                                                    tile_predecessor_item);
    }
    else
    {
        adjacent_difference.SubtractLeftPartialTile(thread_items,
                                                    thread_output,
                                                    BinaryFunction{},
                                                    tile_size);
    }

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
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
__launch_bounds__(BlockSize)
void subtract_right_kernel(const T* input, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    if (blockIdx.x % 2 == 0)
    {
        const T tile_successor_item = input[block_offset + items_per_block];
        adjacent_difference.SubtractRight(thread_items, thread_output, BinaryFunction{}, tile_successor_item);
    }
    else
    {
        adjacent_difference.SubtractRight(thread_items, thread_output, BinaryFunction{});
    }

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
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
__launch_bounds__(BlockSize)
void subtract_right_partial_tile_kernel(const T* input, int* tile_sizes, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    int tile_size = tile_sizes[blockIdx.x];
    
    adjacent_difference.SubtractRightPartialTile(thread_items, thread_output, BinaryFunction{}, tile_size);

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
}

// Don't test FlagHeads, FlagTails nor FlagHeadsAndTails for CCCL >= 2.6.0, as they were removed
#if defined(__HIP_PLATFORM_AMD__) || CUB_VERSION < 200600
TYPED_TEST(HipcubBlockAdjacentDifference, FlagHeads)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params::type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
        std::is_same<bool, typename TestFixture::params::flag_type>::value,
        int,
        typename TestFixture::params::flag_type>::type;
    using flag_type                   = typename TestFixture::params::flag_type;
    using flag_op_type                = typename TestFixture::params::flag_op_type;
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block  = block_size * items_per_thread;
    const size_t     size             = items_per_block * 2048;
    constexpr size_t grid_size        = size / items_per_block;

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
                        ? apply<type, flag_type, flag_op_type>(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply<type, flag_type, flag_op_type>(flag_op, input[i - 1], input[i], ii);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        HIP_CHECK(hipMalloc(&device_heads, heads.size() * sizeof(typename decltype(heads)::value_type)));

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
        HIP_CHECK(hipPeekAtLastError());
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

TYPED_TEST(HipcubBlockAdjacentDifference, FlagTails)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params::type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
        std::is_same<bool, typename TestFixture::params::flag_type>::value,
        int,
        typename TestFixture::params::flag_type>::type;
    using flag_type                   = typename TestFixture::params::flag_type;
    using flag_op_type                = typename TestFixture::params::flag_op_type;
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block  = block_size * items_per_thread;
    const size_t     size             = items_per_block * 2048;
    constexpr size_t grid_size        = size / items_per_block;

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
                        ? apply<type, flag_type, flag_op_type>(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply<type, flag_type, flag_op_type>(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_tails;
        HIP_CHECK(hipMalloc(&device_tails, tails.size() * sizeof(typename decltype(tails)::value_type)));

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
        HIP_CHECK(hipPeekAtLastError());
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

TYPED_TEST(HipcubBlockAdjacentDifference, FlagHeadsAndTails)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params::type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
        std::is_same<bool, typename TestFixture::params::flag_type>::value,
        int,
        typename TestFixture::params::flag_type>::type;
    using flag_type                   = typename TestFixture::params::flag_type;
    using flag_op_type                = typename TestFixture::params::flag_op_type;
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block  = block_size * items_per_thread;
    const size_t     size             = items_per_block * 2048;
    constexpr size_t grid_size        = size / items_per_block;

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
                        ? apply<type, flag_type, flag_op_type>(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply<type, flag_type, flag_op_type>(flag_op, input[i - 1], input[i], ii);
                }
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = (bi % 4 == 0 || bi % 4 == 1)
                        ? apply<type, flag_type, flag_op_type>(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply<type, flag_type, flag_op_type>(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        HIP_CHECK(hipMalloc(&device_heads, tails.size() * sizeof(typename decltype(heads)::value_type)));
        long long* device_tails;
        HIP_CHECK(hipMalloc(&device_tails, tails.size() * sizeof(typename decltype(tails)::value_type)));

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
        HIP_CHECK(hipPeekAtLastError());
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
#endif // defined(__HIP_PLATFORM_AMD__) || CUB_VERSION < 200600

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractLeft)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == 0) 
                {
                    expected[i]
                        = static_cast<output_type>(block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i]);
                } 
                else 
                {
                    expected[i] = static_cast<output_type>(op(input[i], input[i - 1]));
                }
            }
        }

        // Preparing Device
        type* d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_left_kernel<type, output_type, stored_type, 
                                     binary_function, block_size,
                                     items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output,
                                    expected,
                                    std::max(test_utils::precision<type>::value,
                                             test_utils::precision<stored_type>::value)));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractLeftPartialTile)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        const std::vector<int> tile_sizes 
            = test_utils::get_random_data<int>(grid_size, 0, items_per_block, seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if (item < tile_sizes[block_index]) 
                {
                    if(item == 0) 
                    {
                        expected[i] = static_cast<output_type>(
                            block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i]);
                    } 
                    else 
                    {
                        expected[i] = static_cast<output_type>(op(input[i], input[i - 1]));
                    }
                }
                else
                {
                    expected[i] = static_cast<output_type>(input[i]);
                }
            }
        }

        // Preparing Device
        type* d_input;
        int* d_tile_sizes;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_tile_sizes, tile_sizes.data(),
                tile_sizes.size() * sizeof(tile_sizes[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_left_partial_tile_kernel<type, output_type, stored_type, 
                                                  binary_function, block_size,
                                                  items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_tile_sizes, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output,
                                    expected,
                                    std::max(test_utils::precision<type>::value,
                                             test_utils::precision<stored_type>::value)));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractRight)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type>     input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == items_per_block - 1) 
                {
                    expected[i]
                        = static_cast<output_type>(block_index % 2 == 0 ? op(input[i], input[i + 1]) : input[i]);
                } 
                else 
                {
                    expected[i] = static_cast<output_type>(op(input[i], input[i + 1]));
                }
            }
        }

        // Preparing Device
        type* d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_right_kernel<type, output_type, stored_type, 
                                     binary_function, block_size,
                                     items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output,
                                    expected,
                                    std::max(test_utils::precision<type>::value,
                                             test_utils::precision<stored_type>::value)));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractRightPartialTile)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type>     input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        const std::vector<int> tile_sizes 
            = test_utils::get_random_data<int>(grid_size, 0, items_per_block, seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if (item < tile_sizes[block_index]) 
                {
                    if(item == tile_sizes[block_index] - 1 || item == items_per_block - 1) 
                    {
                        expected[i] = static_cast<output_type>(input[i]);
                    } 
                    else 
                    {
                        expected[i] = static_cast<output_type>(op(input[i], input[i + 1]));
                    }
                }
                else
                {
                    expected[i] = static_cast<output_type>(input[i]);
                }
            }
        }

        // Preparing Device
        type* d_input;
        int* d_tile_sizes;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_tile_sizes, tile_sizes.data(),
                tile_sizes.size() * sizeof(tile_sizes[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_right_partial_tile_kernel<type, output_type, stored_type, 
                                                   binary_function, block_size,
                                                   items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_tile_sizes, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        using is_add_op = test_utils::is_add_operator<binary_function>;
        // clang-format off
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected,
            is_add_op::value
                ? std::max(test_utils::precision<type>::value, test_utils::precision<stored_type>::value)
                : std::is_same<type, stored_type>::value 
                    ? 0 
                    : test_utils::precision<stored_type>::value));
        // clang-format on

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
    }
}
