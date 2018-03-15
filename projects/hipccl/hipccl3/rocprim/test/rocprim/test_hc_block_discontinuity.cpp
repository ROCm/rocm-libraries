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
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

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
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1, rocprim::equal_to<unsigned int> >,
    params<int, bool, 128U, 1, rocprim::not_equal_to<int> >,
    params<float, int, 256U, 1, rocprim::less<float> >,
    params<char, char, 1024U, 1, rocprim::less_equal<char> >,
    params<int, bool, 256U, 1, custom_flag_op1<int> >,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U, 1, rocprim::greater<double> >,
    params<float, int, 37U, 1, custom_flag_op1<float> >,
    params<long long, char, 510U, 1, rocprim::greater_equal<long long> >,
    params<unsigned int, long long, 162U, 1, rocprim::not_equal_to<unsigned int> >,
    params<unsigned char, bool, 255U, 1, rocprim::equal_to<unsigned char> >,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<int, char, 64U, 2, custom_flag_op2>,
    params<int, short, 128U, 4, rocprim::less<int> >,
    params<unsigned short, unsigned char, 256U, 7, custom_flag_op2>,
    params<short, short, 512U, 8, rocprim::equal_to<short> >,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, int, 33U, 5, custom_flag_op2>,
    params<double, unsigned int, 464U, 2, rocprim::equal_to<double> >,
    params<unsigned short, int, 100U, 3, rocprim::greater<unsigned short> >,
    params<short, bool, 234U, 9, custom_flag_op1<short> >
> Params;

TYPED_TEST_CASE(RocprimBlockDiscontinuity, Params);

TYPED_TEST(RocprimBlockDiscontinuity, FlagHeads)
{
    hc::accelerator acc;

    using type = typename TestFixture::params::type;
    using flag_type = typename TestFixture::params::flag_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using flag_op_type = typename TestFixture::params::flag_op_type;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = items_per_block * 2048;
    // Generate data
    std::vector<type> input = test_utils::get_random_data<type>(size, type(0), type(10));
    std::vector<long long> heads(size);

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
                    ? apply(flag_op, input[i - 1], input[i], ii)
                    : flag_type(true);
            }
            else
            {
                expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
            }
        }
    }

    hc::array_view<type, 1> d_input(size, input.data());
    hc::array_view<long long, 1> d_heads(size, heads.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(size / items_per_thread).tile(block_size),
        [=](hc::tiled_index<1> idx) [[hc]]
        {
            const unsigned int lid = idx.local[0];
            const unsigned int block_offset = idx.tile[0] * items_per_block;

            type input[items_per_thread];
            rp::block_load_direct_blocked(lid, d_input.data() + block_offset, input);

            rp::block_discontinuity<type, block_size> bdiscontinuity;

            flag_type head_flags[items_per_thread];
            if(idx.tile[0] % 2 == 1)
            {
                const type tile_predecessor_item = d_input[block_offset - 1];
                bdiscontinuity.flag_heads(head_flags, tile_predecessor_item, input, flag_op_type());
            }
            else
            {
                bdiscontinuity.flag_heads(head_flags, input, flag_op_type());
            }

            rp::block_store_direct_blocked(lid, d_heads.data() + block_offset, head_flags);
        }
    );

    d_heads.synchronize();
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(heads[i], expected_heads[i]);
    }
}

TYPED_TEST(RocprimBlockDiscontinuity, FlagTails)
{
    hc::accelerator acc;

    using type = typename TestFixture::params::type;
    using flag_type = typename TestFixture::params::flag_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using flag_op_type = typename TestFixture::params::flag_op_type;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = items_per_block * 2048;
    // Generate data
    std::vector<type> input = test_utils::get_random_data<type>(size, type(0), type(10));
    std::vector<long long> tails(size);

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

    hc::array_view<type, 1> d_input(size, input.data());
    hc::array_view<long long, 1> d_tails(size, tails.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(size / items_per_thread).tile(block_size),
        [=](hc::tiled_index<1> idx) [[hc]]
        {
            const unsigned int lid = idx.local[0];
            const unsigned int block_offset = idx.tile[0] * items_per_block;

            type input[items_per_thread];
            rp::block_load_direct_blocked(lid, d_input.data() + block_offset, input);

            rp::block_discontinuity<type, block_size> bdiscontinuity;

            flag_type tail_flags[items_per_thread];
            if(idx.tile[0] % 2 == 0)
            {
                const type tile_successor_item = d_input[block_offset + items_per_block];
                bdiscontinuity.flag_tails(tail_flags, tile_successor_item, input, flag_op_type());
            }
            else
            {
                bdiscontinuity.flag_tails(tail_flags, input, flag_op_type());
            }

            rp::block_store_direct_blocked(lid, d_tails.data() + block_offset, tail_flags);
        }
    );

    d_tails.synchronize();
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(tails[i], expected_tails[i]);
    }
}

TYPED_TEST(RocprimBlockDiscontinuity, FlagHeadsAndTails)
{
    hc::accelerator acc;

    using type = typename TestFixture::params::type;
    using flag_type = typename TestFixture::params::flag_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using flag_op_type = typename TestFixture::params::flag_op_type;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = items_per_block * 2048;
    // Generate data
    std::vector<type> input = test_utils::get_random_data<type>(size, type(0), type(10));
    std::vector<long long> heads(size);
    std::vector<long long> tails(size);

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

    hc::array_view<type, 1> d_input(size, input.data());
    hc::array_view<long long, 1> d_heads(size, heads.data());
    hc::array_view<long long, 1> d_tails(size, tails.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(size / items_per_thread).tile(block_size),
        [=](hc::tiled_index<1> idx) [[hc]]
        {
            const unsigned int lid = idx.local[0];
            const unsigned int block_offset = idx.tile[0] * items_per_block;

            type input[items_per_thread];
            rp::block_load_direct_blocked(lid, d_input.data() + block_offset, input);

            rp::block_discontinuity<type, block_size> bdiscontinuity;

            flag_type head_flags[items_per_thread];
            flag_type tail_flags[items_per_thread];
            if(idx.tile[0] % 4 == 0)
            {
                const type tile_successor_item = d_input[block_offset + items_per_block];
                bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, flag_op_type());
            }
            else if(idx.tile[0] % 4 == 1)
            {
                const type tile_predecessor_item = d_input[block_offset - 1];
                const type tile_successor_item = d_input[block_offset + items_per_block];
                bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input, flag_op_type());
            }
            else if(idx.tile[0] % 4 == 2)
            {
                const type tile_predecessor_item = d_input[block_offset - 1];
                bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, flag_op_type());
            }
            else if(idx.tile[0] % 4 == 3)
            {
                bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, input, flag_op_type());
            }

            rp::block_store_direct_blocked(lid, d_heads.data() + block_offset, head_flags);
            rp::block_store_direct_blocked(lid, d_tails.data() + block_offset, tail_flags);
        }
    );

    d_heads.synchronize();
    d_tails.synchronize();
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(heads[i], expected_heads[i]);
        ASSERT_EQ(tails[i], expected_tails[i]);
    }
}
