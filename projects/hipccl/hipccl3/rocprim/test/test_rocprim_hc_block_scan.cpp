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
#include <block/block_scan.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<typename BlockSizeWrapper>
class RocprimBlockScanShuffleBasedTests : public ::testing::Test {
public:
    static constexpr unsigned int block_size = BlockSizeWrapper::value;
};

typedef ::testing::Types<
    uint_wrapper<64U>,
    uint_wrapper<128U>,
    uint_wrapper<256U>,
    uint_wrapper<512U>,
    uint_wrapper<1024U>,
    uint_wrapper<65U>,
    uint_wrapper<37U>,
    uint_wrapper<162U>,
    uint_wrapper<255U>
> BlockSizes;

TYPED_TEST_CASE(RocprimBlockScanShuffleBasedTests, BlockSizes);

TYPED_TEST(RocprimBlockScanShuffleBasedTests, InclusiveScanInt)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::block_size;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            rp::block_scan<int, block_size> bscan;
            bscan.inclusive_scan(value, value);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockScanShuffleBasedTests, InclusiveScanReduceInt)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::block_size;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);
    std::vector<int> output_reductions(size / block_size);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    std::vector<int> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = expected[(i+1) * block_size - 1];
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::array_view<int, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            int reduction;
            rp::block_scan<int, block_size> bscan;
            bscan.inclusive_scan(value, value, reduction);
            d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
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

TYPED_TEST(RocprimBlockScanShuffleBasedTests, ExclusiveScanInt)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::block_size;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);
    const int init = get_random_value(0, 100);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = init;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx-1] + expected[idx-1];
        }
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            rp::block_scan<int, block_size> bscan;
            bscan.exclusive_scan(value, value, init);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockScanShuffleBasedTests, ExclusiveScanReduceInt)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::block_size;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);
    const int init = get_random_value(0, 100);

    // Output reduce results
    std::vector<int> output_reductions(size / block_size);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    std::vector<int> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = init;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx-1] + expected[idx-1];
        }

        expected_reductions[i] = 0;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected_reductions[i] += output[idx];
        }
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::array_view<int, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            int reduction;
            rp::block_scan<int, block_size> bscan;
            bscan.exclusive_scan(value, value, init, reduction);
            d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
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
