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
// rocPRIM
#include <warp/warp_reduce.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<typename WarpSizeWrapper>
class RocprimWarpReduceTests : public ::testing::Test {
public:
    static constexpr unsigned int warp_size = WarpSizeWrapper::value;
};

typedef ::testing::Types<
    // shuffle based scan
    uint_wrapper<2U>,
    uint_wrapper<4U>,
    uint_wrapper<8U>,
    uint_wrapper<16U>,
    uint_wrapper<32U>,
    uint_wrapper<64U>//,
    // shared memory scan
    /*uint_wrapper<3U>,
    uint_wrapper<7U>,
    uint_wrapper<15U>,
    uint_wrapper<37U>,
    uint_wrapper<61U>*/
> WarpSizes;

TYPED_TEST_CASE(RocprimWarpReduceTests, WarpSizes);

TYPED_TEST(RocprimWarpReduceTests, ReduceSumInt)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
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
    std::vector<int> input = get_random_data<int>(size, -100, 100); // used for input
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output(input.size() / logical_warp_size, 0);
    
    // Calculate expected results on host
    std::vector<int> expected(output.size(), 1);
    for(size_t i = 0; i < output.size(); i++)
    {
        int value = 0;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        expected[i] = value;
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            int value = d_input[i];

            using wreduce_t = rp::warp_reduce<int, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().sum(value, value, storage[warp_id]);
            
            if (i.local[0] % logical_warp_size == 0)
            {
                d_output[i.global[0] / logical_warp_size] = value;
            }
        }
    );
    d_input.synchronize();
    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumValidInt)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
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
    std::vector<int> input = get_random_data<int>(size, -100, 100); // used for input
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output(input.size() / logical_warp_size, 0);
    
    // Calculate expected results on host
    std::vector<int> expected(output.size(), 1);
    for(size_t i = 0; i < output.size(); i++)
    {
        int value = 0;
        for(size_t j = 0; j < valid; j++)
        {
            auto idx = i * logical_warp_size + j;
            value += input[idx];
        }
        expected[i] = value;
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            int value = d_input[i];

            using wreduce_t = rp::warp_reduce<int, logical_warp_size>;
            tile_static typename wreduce_t::storage_type storage[warps_no];
            wreduce_t().sum(value, value, valid, storage[warp_id]);
            
            if (i.local[0] % logical_warp_size == 0)
            {
                d_output[i.global[0] / logical_warp_size] = value;
            }
        }
    );
    d_input.synchronize();
    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}
