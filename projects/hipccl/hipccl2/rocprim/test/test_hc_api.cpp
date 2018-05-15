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

template<class T>
T ax(const T a, const T x) [[hc]]
{
    return x * a;
}

TEST(HCTests, Saxpy)
{
    const size_t N = 100;

    const float a = 100.0f;
    std::vector<float> x(N, 2.0f);
    std::vector<float> y(N, 1.0f);

    hc::array_view<float, 1> av_x(N, x.data());
    hc::array_view<float, 1> av_y(N, y.data());

    hc::parallel_for_each(
        hc::extent<1>(N),
        [=](hc::index<1> i) [[hc]] {
            av_y[i] = ax(a, av_x[i]) + av_y[i];
        }
    );

    av_y.synchronize();
    for(size_t i = 0; i < N; i++)
    {
        ASSERT_NEAR(y[i], 201.0f, 0.1f);
    }
}
