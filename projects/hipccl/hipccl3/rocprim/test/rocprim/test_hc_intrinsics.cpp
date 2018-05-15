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
#include <tuple>

// Google Test
#include <gtest/gtest.h>

// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

// Custom structure
struct custom
{
    short i;
    double d;
    float f;
    unsigned int u;

    ROCPRIM_HOST_DEVICE
    custom() {};
    ROCPRIM_HOST_DEVICE
    ~custom() {};

    ROCPRIM_HOST_DEVICE
    custom& operator+=(const custom& rhs)
    {
        this->i += rhs.i;
        this->d += rhs.d;
        this->f += rhs.f;
        this->u += rhs.u;
        return *this;
    }
};

ROCPRIM_HOST_DEVICE
inline custom operator+(custom lhs, const custom& rhs)
{
    lhs += rhs;
    return lhs;
}

ROCPRIM_HOST_DEVICE
inline bool operator==(const custom& lhs, const custom& rhs)
{
    return std::tie(lhs.i, lhs.d, lhs.f, lhs.u) ==
        std::tie(rhs.i, rhs.d, rhs.f, rhs.u);
}

// Custom structure aligned to 16 bytes
struct custom_16aligned
{
    int i;
    unsigned int u;
    float f;

    ROCPRIM_HOST_DEVICE
    custom_16aligned() {};
    ROCPRIM_HOST_DEVICE
    ~custom_16aligned() {};

    ROCPRIM_HOST_DEVICE
    custom_16aligned& operator+=(const custom_16aligned& rhs)
    {
        this->i += rhs.i;
        this->u += rhs.u;
        this->f += rhs.f;
        return *this;
    }
} __attribute__((aligned(16)));;

inline ROCPRIM_HOST_DEVICE
custom_16aligned operator+(custom_16aligned lhs, const custom_16aligned& rhs)
{
    lhs += rhs;
    return lhs;
}

inline ROCPRIM_HOST_DEVICE
bool operator==(const custom_16aligned& lhs, const custom_16aligned& rhs)
{
    return lhs.i == rhs.i && lhs.f == rhs.f && lhs.u == rhs.u;
}

namespace rp = rocprim;

TEST(RocprimIntrinsicsTests, WarpShuffleUpInt)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = test_utils::get_random_data<int>(size, -100, 100);

    // Calculate expected results on host
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
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpChar)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<char> output = test_utils::get_random_data<char>(size, -2, 2);

    // Calculate expected results on host
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
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpFloat)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<float> output = test_utils::get_random_data<float>(size, -100, 100);

    // Calculate expected results on host
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
        ASSERT_NEAR(output[i], expected[i], std::abs(0.01f * expected[i]));
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpDouble)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<double> output = test_utils::get_random_data<double>(size, -100, 100);

    // Calculate expected results on host
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
        ASSERT_NEAR(output[i], expected[i], std::abs(0.01 * expected[i]));
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpCustomStruct)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<double> random_data = test_utils::get_random_data<double>(4 * size, -100, 100);
    std::vector<custom> output(size);
    for(size_t i = 0; i < 4 * output.size(); i+=4)
    {
        output[i/4].i = random_data[i];
        output[i/4].d = random_data[i+1];
        output[i/4].f = random_data[i+2];
        output[i/4].u = random_data[i+3];
    }

    // Calculate expected results on host
    std::vector<custom> expected(size);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<custom, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            custom value = d_output[i];
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleUpCustomAlignedStruct)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<double> random_data = test_utils::get_random_data<double>(3 * size, -100, 100);
    std::vector<custom_16aligned> output(size);
    for(size_t i = 0; i < 3 * output.size(); i+=3)
    {
        output[i/3].i = random_data[i];
        output[i/3].u = random_data[i+1];
        output[i/3].f = random_data[i+2];
    }

    // Calculate expected results on host
    std::vector<custom_16aligned> expected(size);
    for(size_t i = 0; i < output.size(); i++)
    {
        expected[i] = output[i > 0 ? i-1 : 0] + output[i];
    }

    hc::array_view<custom_16aligned, 1> d_output(size, output.data());
    hc::parallel_for_each(
        hc::extent<1>(size).tile(warp_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            custom_16aligned value = d_output[i];
            value += rp::warp_shuffle_up(value, 1, warp_size);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleDown)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = test_utils::get_random_data<int>(size, -100, 100);

    // Calculate expected results on host
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
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffle)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = test_utils::get_random_data<int>(size, -100, 100);

    // Calculate expected results on host
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
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpShuffleXor)
{
    const size_t warp_size = rp::warp_size();
    const size_t size = warp_size;

    // Generate data
    std::vector<int> output = test_utils::get_random_data<int>(size, -100, 100);

    // Calculate expected results on host
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
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TEST(RocprimIntrinsicsTests, WarpId)
{
    const size_t warp_size = rp::warp_size();
    const size_t block_size = 4 * warp_size;
    const size_t size = 16 * block_size;

    std::vector<int> output(size);

    // Calculate expected results on host
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
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}
