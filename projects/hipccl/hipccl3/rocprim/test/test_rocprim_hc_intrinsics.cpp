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
#include <intrinsics.hpp>

#include "test_utils.hpp"

namespace rocprim = roc::prim;

TEST(RocprimIntrinsicsTests, WarpShuffleUp)
{
    const size_t warp_size = rocprim::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<int, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            value += rocprim::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpDouble)
{
    const size_t warp_size = rocprim::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<double> output = get_random_data<double>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<double> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<double, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            double value = d_output[i];
            value += rocprim::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}
