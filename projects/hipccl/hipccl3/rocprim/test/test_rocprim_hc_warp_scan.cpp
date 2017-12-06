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

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM
#include <warp/warp_scan.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<typename WarpSizeWrapper>
class RocprimWarpScanShuffleBasedTests : public ::testing::Test {
public:
    static constexpr unsigned int warp_size = WarpSizeWrapper::value;
};

TYPED_TEST_CASE(RocprimWarpScanShuffleBasedTests, WarpSizes);

TYPED_TEST(RocprimWarpScanShuffleBasedTests, InclusiveScanInt)
{
    constexpr size_t warp_size = TestFixture::warp_size;
    // Given warp size not supported
    if(warp_size > rp::warp_size() || !rp::detail::is_power_of_two(warp_size))
    {
        return;
    }

    const size_t size = warp_size * 4;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / warp_size; i++)
    {
        for(size_t j = 0; j < warp_size; j++)
        {
            auto idx = i * warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            rp::warp_scan<int, warp_size> wscan;
            value = wscan.inclusive_scan(value);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimWarpScanShuffleBasedTests, InclusiveScanReduceInt)
{
    constexpr size_t warp_size = TestFixture::warp_size;
    // Given warp size not supported
    if(warp_size > rp::warp_size() || !rp::detail::is_power_of_two(warp_size))
    {
        return;
    }

    const size_t size = warp_size * 4;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);
    std::vector<int> output_reductions(size / warp_size);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    std::vector<int> expected_reductions(output.size(), 0);
    for(size_t i = 0; i < output.size() / warp_size; i++)
    {
        for(size_t j = 0; j < warp_size; j++)
        {
            auto idx = i * warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = expected[(i+1) * warp_size - 1];
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::array_view<int, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            rp::warp_scan<int, warp_size> wscan;
            auto result = wscan.inclusive_scan_reduce(value);
            d_output[i] = result.scan;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = result.reduction;
            }
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }

    d_output_r.synchronize();
    for(int i = 0; i < output_reductions.size(); i++)
    {
        EXPECT_EQ(output_reductions[i], expected_reductions[i]);
    }
}