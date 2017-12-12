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
#include <block/block_load.hpp>
#include <block/block_store.hpp>
#include <iterator/input_iterator.hpp>
#include <iterator/output_iterator.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<typename BlockSizeWrapper>
class RocprimBlockLoadStoreTests : public ::testing::Test {
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

TYPED_TEST_CASE(RocprimBlockLoadStoreTests, BlockSizes);

TYPED_TEST(RocprimBlockLoadStoreTests, LoadStoreDirectBlocked)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::block_size;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t items_per_thread = 16;
    // Generate data
    std::vector<int> output = get_random_data<int>(size, -100, 100);
    std::vector<int> output2(output.size(), 0);

    // Calculate expected results on host
    std::vector<int> expected(output);

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::array_view<int, 1> d_output2(output2.size(), output2.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int t[items_per_thread];
            int idx = i.global[0];
            rp::block_load_direct_blocked<int, items_per_thread>(
                idx, 
                rp::input_iterator<int>(d_output.data()), 
                t, size);
            rp::block_store_direct_blocked<int, items_per_thread>(
                idx, 
                rp::output_iterator<int>(d_output2.data()), 
                t, size);
        }
    );

    d_output.synchronize();
    d_output2.synchronize();
    for(int i = 0; i < output2.size(); i++)
    {
        EXPECT_EQ(output2[i], expected[i]);
    }
}
