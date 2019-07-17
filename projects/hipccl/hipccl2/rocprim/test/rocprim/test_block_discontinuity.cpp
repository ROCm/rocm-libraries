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

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"
#include "test_utils_types.hpp"

#define HIP_CHECK(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

namespace rp = rocprim;

template<class Params>
class RocprimBlockDiscontinuity : public ::testing::Test {
public:
    using params = Params;
};

template<class T>
struct custom_flag_op1
{
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b, unsigned int b_index)
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
template<class T, class FlagOp>
typename std::enable_if<rp::detail::with_b_index_arg<T, FlagOp>::value, bool>::type
apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index)
{
    return flag_op(a, b, b_index);
}

template<class T, class FlagOp>
typename std::enable_if<!rp::detail::with_b_index_arg<T, FlagOp>::value, bool>::type
apply(FlagOp flag_op, const T& a, const T& b, unsigned int)
{
    return flag_op(a, b);
}

TEST(RocprimBlockDiscontinuity, Traits)
{
    ASSERT_FALSE((rp::detail::with_b_index_arg<int, rocprim::less<int>>::value));
    ASSERT_FALSE((rp::detail::with_b_index_arg<int, custom_flag_op2>::value));
    ASSERT_TRUE((rp::detail::with_b_index_arg<int, custom_flag_op1<int>>::value));

    auto f1 = [](const int& a, const int& b, unsigned int b_index) { return (a == b) || (b_index % 10 == 0); };
    auto f2 = [](const int& a, const int& b) { return (a == b); };
    ASSERT_TRUE((rp::detail::with_b_index_arg<int, decltype(f1)>::value));
    ASSERT_FALSE((rp::detail::with_b_index_arg<int, decltype(f2)>::value));

    auto f3 = [](int a, int b, int b_index) { return (a == b) || (b_index % 10 == 0); };
    auto f4 = [](const int a, const int b) { return (a == b); };
    ASSERT_TRUE((rp::detail::with_b_index_arg<int, decltype(f3)>::value));
    ASSERT_FALSE((rp::detail::with_b_index_arg<int, decltype(f4)>::value));
}

typedef ::testing::Types<
    block_param_type(test_utils::custom_test_type<int>, int),
    block_param_type(float, char),
    block_param_type(double, unsigned int),
    block_param_type(uint8_t, bool),
    block_param_type(int8_t, bool),
    block_param_type(rocprim::half, int)
> BlockDiscParams;

TYPED_TEST_CASE(RocprimBlockDiscontinuity, BlockDiscParams);

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
void flag_heads_kernel(Type* device_input, long long* device_heads)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    Type input[ItemsPerThread];
    rp::block_load_direct_blocked(lid, device_input + block_offset, input);

    rp::block_discontinuity<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    if(hipBlockIdx_x % 2 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads(head_flags, tile_predecessor_item, input, FlagOpType());
    }
    else
    {
        bdiscontinuity.flag_heads(head_flags, input, FlagOpType());
    }

    rp::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
void flag_tails_kernel(Type* device_input, long long* device_tails)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    Type input[ItemsPerThread];
    rp::block_load_direct_blocked(lid, device_input + block_offset, input);

    rp::block_discontinuity<Type, BlockSize> bdiscontinuity;

    FlagType tail_flags[ItemsPerThread];
    if(hipBlockIdx_x % 2 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_tails(tail_flags, tile_successor_item, input, FlagOpType());
    }
    else
    {
        bdiscontinuity.flag_tails(tail_flags, input, FlagOpType());
    }

    rp::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
void flag_heads_and_tails_kernel(Type* device_input, long long* device_heads, long long* device_tails)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    Type input[ItemsPerThread];
    rp::block_load_direct_blocked(lid, device_input + block_offset, input);

    rp::block_discontinuity<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    FlagType tail_flags[ItemsPerThread];
    if(hipBlockIdx_x % 4 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(hipBlockIdx_x % 4 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(hipBlockIdx_x % 4 == 2)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, FlagOpType());
    }
    else if(hipBlockIdx_x % 4 == 3)
    {
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, input, FlagOpType());
    }

    rp::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
    rp::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
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
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    // http://en.cppreference.com/w/cpp/container/vector_bool
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    // Generate data
    std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10);
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
                    ? apply(flag_op, input[i - 1], input[i], ii)
                    : flag_type(true);
            }
            else
            {
                expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
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
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    // http://en.cppreference.com/w/cpp/container/vector_bool
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    // Generate data
    std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10);
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
                    ? apply(flag_op, input[i], input[i + 1], ii + 1)
                    : flag_type(true);
            }
            else
            {
                expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
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
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    // http://en.cppreference.com/w/cpp/container/vector_bool
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    // Generate data
    std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10);
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
                    ? apply(flag_op, input[i - 1], input[i], ii)
                    : flag_type(true);
            }
            else
            {
                expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
            }
            if(ii == items_per_block - 1)
            {
                expected_tails[i] = (bi % 4 == 0 || bi % 4 == 1)
                    ? apply(flag_op, input[i], input[i + 1], ii + 1)
                    : flag_type(true);
            }
            else
            {
                expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
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
        test_block_discontinuity<Type, FlagType, FlagOpType, Method, BlockSize, items[First]>();
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

TYPED_TEST(RocprimBlockDiscontinuity, FlagHeads)
{
    using type = typename TestFixture::params::input_type;
    using flag_type = typename TestFixture::params::output_type;
    using flag_op_type_1 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_less, rp::less<type>>::type;
    using flag_op_type_2 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_equal_to, rp::equal_to<type>>::type;
    using flag_op_type_3 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_greater, rp::greater<type>>::type;
    using flag_op_type_4 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_not_equal_to, rp::not_equal_to<type>>::type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 2, type, flag_type, flag_op_type_1, 0, block_size>::run();
    static_for<2, 4, type, flag_type, flag_op_type_2, 0, block_size>::run();
    static_for<4, 6, type, flag_type, flag_op_type_3, 0, block_size>::run();
    static_for<6, n_items, type, flag_type, flag_op_type_4, 0, block_size>::run();
}

TYPED_TEST(RocprimBlockDiscontinuity, FlagTails)
{
    using type = typename TestFixture::params::input_type;
    using flag_type = typename TestFixture::params::output_type;
    using flag_op_type_1 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_less, rp::less<type>>::type;
    using flag_op_type_2 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_equal_to, rp::equal_to<type>>::type;
    using flag_op_type_3 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_greater, rp::greater<type>>::type;
    using flag_op_type_4 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_not_equal_to, rp::not_equal_to<type>>::type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 2, type, flag_type, flag_op_type_1, 1, block_size>::run();
    static_for<2, 4, type, flag_type, flag_op_type_2, 1, block_size>::run();
    static_for<4, 6, type, flag_type, flag_op_type_3, 1, block_size>::run();
    static_for<6, n_items, type, flag_type, flag_op_type_4, 1, block_size>::run();
}

TYPED_TEST(RocprimBlockDiscontinuity, FlagHeadsAndTails)
{
    using type = typename TestFixture::params::input_type;
    using flag_type = typename TestFixture::params::output_type;
    using flag_op_type_1 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_less, rp::less<type>>::type;
    using flag_op_type_2 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_equal_to, rp::equal_to<type>>::type;
    using flag_op_type_3 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_greater, rp::greater<type>>::type;
    using flag_op_type_4 = typename std::conditional<std::is_same<type, rp::half>::value, test_utils::half_not_equal_to, rp::not_equal_to<type>>::type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 2, type, flag_type, flag_op_type_1, 2, block_size>::run();
    static_for<2, 4, type, flag_type, flag_op_type_2, 2, block_size>::run();
    static_for<4, 6, type, flag_type, flag_op_type_3, 2, block_size>::run();
    static_for<6, n_items, type, flag_type, flag_op_type_4, 2, block_size>::run();
}
