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

// Custom big structure
struct custom_struct
{
    short i;
    double d;
    float f;
    unsigned int u;

    ~custom_struct() [[cpu]] [[hc]]
    {
    }

    custom_struct& operator+=(const custom_struct& rhs) [[cpu]] [[hc]]
    {
        this->i += rhs.i;
        this->d += rhs.d;
        this->f += rhs.f;
        this->u += rhs.u;
        return *this;
    }
};

inline custom_struct operator+(custom_struct lhs,
                             const custom_struct& rhs) [[cpu]] [[hc]]
{
    lhs += rhs;
    return lhs;
}

inline bool operator==(const custom_struct& lhs, const custom_struct& rhs)
{
    return lhs.i == rhs.i && lhs.d == rhs.d
        && lhs.f == rhs.f && lhs.u == rhs.u;
}

namespace rp = rocprim;

TEST(RocprimWarpScanShuffleBasedTests, InclusiveScanInt)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i] + expected[i > 0 ? i-1 : 0];
    }

    hc::array_view<int, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
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
