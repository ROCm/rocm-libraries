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
#include <cmath>

// Google Test
#include <gtest/gtest.h>

// HC API
#include <hcc/hc.hpp>

// rocPRIM
#include <intrinsics.hpp>

#include "test_utils.hpp"

// Custom big structure
struct custom_struct
{
    short i;
    double d;
    float f;
    unsigned int u;

    custom_struct() [[cpu]] [[hc]] = default;
    ~custom_struct() [[cpu]] [[hc]] = default;

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

TEST(RocprimIntrinsicsTests, WarpShuffleUpInt)
{
    const size_t warp_size = rp::warp_size();
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
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpChar)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<char> output = get_random_data<char>(size, -2, 2);

    // Calulcate expected results on host
    std::vector<char> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<char, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            char value = d_output[i];
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(char i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpFloat)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<float> output = get_random_data<float>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<float> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<float, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            float value = d_output[i];
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        EXPECT_NEAR(output[i], expected[i], std::abs(0.01f * expected[i]));
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpDouble)
{
    const size_t warp_size = rp::warp_size();
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
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        EXPECT_NEAR(output[i], expected[i], std::abs(0.01 * expected[i]));
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpCustomStruct)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<double> random_data = get_random_data<double>(4 * size, -100, 100);
    std::vector<custom_struct> output(size);
    for(size_t i = 0; i < 4 * output.size(); i+=4)
    {
        output[i/4].i = random_data[i];
        output[i/4].d = random_data[i+1];
        output[i/4].f = random_data[i+2];
        output[i/4].u = random_data[i+3];
    }

    // Calulcate expected results on host
    std::vector<custom_struct> expected(size);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<custom_struct, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            custom_struct value = d_output[i];
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleDown)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i] + output[i+1 < output.size() ? i+1 : i];
    }

    hc::array_view<int, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            value += rp::warp_shuffle_down(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffle)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(size, 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[(i + 32)%output.size()];
    }

    hc::array_view<int, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            int index = i.global[0];
            value = rp::warp_shuffle(value, (index+32), warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleXor)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(size, 0);
    for(size_t i = 0; i < output.size(); i+=2)
    {
        expected[i]   = output[i+1];
        expected[i+1] = output[i];
    }

    hc::array_view<int, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            value = rp::warp_shuffle_xor(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpId)
{
    const size_t warp_size = rp::warp_size();
    const size_t block_size = 4 * warp_size;
    const size_t size = 16 * block_size;

    std::vector<int> output(size);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = (i%block_size)/warp_size;
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            d_output[i] = rp::warp_id();
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}
