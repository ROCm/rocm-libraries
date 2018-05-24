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
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

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

TYPED_TEST(RocprimWarpReduceTests, ReduceSum)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using type = typename TestFixture::params::type;
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
    std::vector<type> input = test_utils::get_random_data<type>(size, -100, 100); // used for input
    std::vector<type> output(input.size() / logical_warp_size, 0);

    // Calculate expected results on host
    std::vector<type> expected(output.size(), 1);
    for(size_t i = 0; i < output.size(); i++)
    {
        type value = 0;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        expected[i] = value;
    }

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().reduce(value, value, storage[warp_id]);

            if(i.local[0] % logical_warp_size == 0)
            {
                d_output[i.global[0] / logical_warp_size] = value;
            }
        }
    );
    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        auto diff = std::max<type>(std::abs(0.1f * expected[i]), type(0.01f));
        if(std::is_integral<type>::value) diff = 0;
        ASSERT_NEAR(output[i], expected[i], diff);
    }
}

TYPED_TEST(RocprimWarpReduceTests, AllReduceSum)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using type = typename TestFixture::params::type;
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
    std::vector<type> input = test_utils::get_random_data<type>(size, -100, 100); // used for input
    std::vector<type> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<type> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        type value = 0;
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

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size, true>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().reduce(value, value, storage[warp_id]);

            d_output[i] = value;
        }
    );
    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        auto diff = std::max<type>(std::abs(0.1f * expected[i]), type(0.01f));
        if(std::is_integral<type>::value) diff = 0;
        ASSERT_NEAR(output[i], expected[i], diff);
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumValid)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using type = typename TestFixture::params::type;
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
    std::vector<type> input = test_utils::get_random_data<type>(size, -100, 100); // used for input
    std::vector<type> output(input.size() / logical_warp_size, 0);

    // Calculate expected results on host
    std::vector<type> expected(output.size(), 1);
    for(size_t i = 0; i < output.size(); i++)
    {
        type value = 0;
        for(size_t j = 0; j < valid; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        expected[i] = value;
    }

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().reduce(value, value, valid, storage[warp_id]);

            if(i.local[0] % logical_warp_size == 0)
            {
                d_output[i.global[0] / logical_warp_size] = value;
            }
        }
    );
    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        auto diff = std::max<type>(std::abs(0.1f * expected[i]), type(0.01f));
        if(std::is_integral<type>::value) diff = 0;
        ASSERT_NEAR(output[i], expected[i], diff);
    }
}

TYPED_TEST(RocprimWarpReduceTests, AllReduceSumValid)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using type = typename TestFixture::params::type;
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
    std::vector<type> input = test_utils::get_random_data<type>(size, -100, 100); // used for input
    std::vector<type> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<type> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        type value = 0;
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

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size, true>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().reduce(value, value, valid, storage[warp_id]);

            d_output[i] = value;
        }
    );
    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        auto diff = std::max<type>(std::abs(0.1f * expected[i]), type(0.01f));
        if(std::is_integral<type>::value) diff = 0;
        ASSERT_NEAR(output[i], expected[i], diff);
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumCustomStruct)
{
    using base_type = typename TestFixture::params::type;
    using type = test_utils::custom_test_type<base_type>;

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
    std::vector<type> input(size);
    {
        auto random_values =
            test_utils::get_random_data<base_type>(2 * input.size(), 0, 100);
        for(size_t i = 0; i < input.size(); i++)
        {
            input[i].x = random_values[i];
            input[i].y = random_values[i + input.size()];
        }
    }
    std::vector<type> output(input.size() / logical_warp_size);

    // Calculate expected results on host
    std::vector<type> expected(output.size());
    for(size_t i = 0; i < output.size(); i++)
    {
        type value(0, 0);
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            value = value + input[idx];
        }
        expected[i] = value;
    }

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().reduce(value, value, storage[warp_id]);

            if(i.local[0] % logical_warp_size == 0)
            {
                d_output[i.global[0] / logical_warp_size] = value;
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        auto diffx = std::max<base_type>(std::abs(0.1f * expected[i].x), base_type(0.01f));
        if(std::is_integral<base_type>::value) diffx = 0;
        ASSERT_NEAR(output[i].x, expected[i].x, diffx);

        auto diffy = std::max<base_type>(std::abs(0.1f * expected[i].y), base_type(0.01f));
        if(std::is_integral<base_type>::value) diffy = 0;
        ASSERT_NEAR(output[i].y, expected[i].y, diffy);
    }
}

TYPED_TEST(RocprimWarpReduceTests, HeadSegmentedReduceSum)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using type = typename TestFixture::params::type;
    using flag_type = unsigned char;
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
    std::vector<type> input = test_utils::get_random_data<type>(size, 1, 10); // used for input
    std::vector<flag_type> flags = test_utils::get_random_data01<flag_type>(size, 0.25f);
    for(size_t i = 0; i < flags.size(); i+= logical_warp_size)
    {
        flags[i] = 1;
    }
    std::vector<type> output(input.size());

    // Calculate expected results on host
    std::vector<type> expected(output.size());
    size_t segment_head_index = 0;
    type reduction;
    for(size_t i = 0; i < output.size(); i++)
    {
        if(i%logical_warp_size == 0 || flags[i])
        {
            expected[segment_head_index] = reduction;
            segment_head_index = i;
            reduction = input[i];
        }
        else
        {
            reduction = reduction + input[i];
        }
    }
    expected[segment_head_index] = reduction;

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<flag_type, 1> d_flag(flags.size(), flags.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];
            flag_type flag = d_flag[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().head_segmented_reduce(value, value, flag, storage[warp_id]);

            d_output[i] = value;
        }
    );
    d_output.synchronize();

    for(size_t i = 0; i < output.size(); i++)
    {
        if(flags[i])
        {
            auto diff = std::max<type>(std::abs(0.1f * expected[i]), type(0.01f));
            if(std::is_integral<type>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff) << " with index: " << index;
        }
    }
}

TYPED_TEST(RocprimWarpReduceTests, TailSegmentedReduceSum)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    using type = typename TestFixture::params::type;
    using flag_type = unsigned char;
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
    std::vector<type> input = test_utils::get_random_data<type>(size, 1, 10); // used for input
    std::vector<flag_type> flags = test_utils::get_random_data01<flag_type>(size, 0.25f);
    for(size_t i = logical_warp_size - 1; i < flags.size(); i+= logical_warp_size)
    {
        flags[i] = 1;
    }
    std::vector<type> output(input.size());

    // Calculate expected results on host
    std::vector<type> expected(output.size());
    std::vector<size_t> segment_indexes;
    size_t segment_index = 0;
    type reduction;
    for(size_t i = 0; i < output.size(); i++)
    {
        // single value segments
        if(flags[i])
        {
            expected[i] = input[i];
            segment_indexes.push_back(i);
        }
        else
        {
            segment_index = i;
            reduction = input[i];
            auto next = i + 1;
            while(next < output.size() && !flags[next])
            {
                reduction = reduction + input[next];
                i++;
                next++;
            }
            i++;
            expected[segment_index] = reduction + input[i];
            segment_indexes.push_back(segment_index);
        }
    }

    hc::array_view<type, 1> d_input(input.size(), input.data());
    hc::array_view<flag_type, 1> d_flag(flags.size(), flags.data());
    hc::array_view<type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            type value = d_input[i];
            flag_type flag = d_flag[i];

            using wreduce_t = rp::warp_reduce<type, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().tail_segmented_reduce(value, value, flag, storage[warp_id]);

            d_output[i] = value;
        }
    );
    d_output.synchronize();

    for(size_t i = 0; i < segment_indexes.size(); i++)
    {
        auto index = segment_indexes[i];
        auto diff = std::max<type>(std::abs(0.1f * expected[i]), type(0.01f));
        if(std::is_integral<type>::value) diff = 0;
        ASSERT_NEAR(output[index], expected[index], diff) << " with index: " << index;
    }
}
