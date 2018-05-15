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
#include <iostream>
#include <random>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<unsigned int WarpSize>
struct params
{
    static constexpr unsigned int warp_size = WarpSize;
};

template<typename Params>
class RocprimWarpSortShuffleBasedTests : public ::testing::Test {
public:
    static constexpr unsigned int warp_size = Params::warp_size;
};

template<class T>
bool test(const T& a, const T& b) [[hc]]
{
    return a < b;
}

typedef ::testing::Types<
    params<2U>,
    params<4U>,
    params<8U>,
    params<16U>,
    params<32U>,
    params<64U>
> WarpSizes;

TYPED_TEST_CASE(RocprimWarpSortShuffleBasedTests, WarpSizes);

TYPED_TEST(RocprimWarpSortShuffleBasedTests, SortInt)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    const size_t block_size = std::max<size_t>(rp::warp_size(), 4 * logical_warp_size);
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size() || !rp::detail::is_power_of_two(logical_warp_size))
    {
        return;
    }

    // Generate data
    std::vector<int> output = test_utils::get_random_data<int>(size, -100, 100);

    // Calculate expected results on host
    std::vector<int> expected(output);

    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        std::sort(expected.begin() + (i * logical_warp_size), expected.begin() + ((i + 1) * logical_warp_size));
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            rp::warp_sort<int, logical_warp_size> wsort;
            wsort.sort(value);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimWarpSortShuffleBasedTests, SortKeyInt)
{
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    const size_t block_size = std::max<size_t>(rp::warp_size(), 4 * logical_warp_size);
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size() || !rp::detail::is_power_of_two(logical_warp_size))
    {
        return;
    }

    // Generate data
    std::vector<int> output_key(size);
    std::iota(output_key.begin(), output_key.end(), 0);
    std::shuffle(output_key.begin(), output_key.end(), std::mt19937{std::random_device{}()});
    std::vector<int> output_value = test_utils::get_random_data<int>(size, -100, 100);

    // Combine vectors to form pairs with key and value
    std::vector<std::pair<int, int>> target(size);
    for (unsigned i = 0; i < target.size(); i++)
        target[i] = std::make_pair(output_key[i], output_value[i]);

    // Calculate expected results on host
    std::vector<std::pair<int, int>> expected(target);

    for(size_t i = 0; i < expected.size() / logical_warp_size; i++)
    {
        std::sort(expected.begin() + (i * logical_warp_size), expected.begin() + ((i + 1) * logical_warp_size));
    }

    hc::array_view<int, 1> d_output_key(output_key.size(), output_key.data());
    hc::array_view<int, 1> d_output_value(output_value.size(), output_value.data());
    hc::parallel_for_each(
        hc::extent<1>(output_key.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int key = d_output_key[i];
            int value = d_output_value[i];
            rp::warp_sort<int, logical_warp_size, int> wsort;
            wsort.sort(key, value);
            d_output_key[i] = key;
            d_output_value[i] = value;
        }
    );

    d_output_key.synchronize();
    d_output_value.synchronize();
    for(size_t i = 0; i < expected.size(); i++)
    {
        ASSERT_EQ(d_output_key[i], expected[i].first);
        ASSERT_EQ(d_output_value[i], expected[i].second);
    }
}
