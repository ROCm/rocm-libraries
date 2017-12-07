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
    std::vector<int> output = get_random_data<int>(size, -100, 100); // used for input/output

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            rp::warp_scan<int, logical_warp_size> wscan;
            wscan.inclusive_scan(value, value);
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
    std::vector<int> output = get_random_data<int>(size, -100, 100);  // used for input/output
    std::vector<int> output_reductions(size / logical_warp_size);

    // Calulcate expected results on host
    std::vector<int> expected(output.size(), 0);
    std::vector<int> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = expected[(i+1) * logical_warp_size - 1];
    }

    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::array_view<int, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_output[i];
            int reduction;
            rp::warp_scan<int, logical_warp_size> wscan;
            wscan.inclusive_scan(value, value, reduction);
            d_output[i] = value;
            if(i.local[0]%logical_warp_size == 0)
            {
                d_output_r[i.global[0]/logical_warp_size] = reduction;
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

TYPED_TEST(RocprimWarpScanShuffleBasedTests, ExclusiveScanInt)
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
    std::vector<int> in_out_array = get_random_data<int>(size, -100, 100);
    const int init = get_random_value(0, 100);

    // Calulcate expected results on host
    std::vector<int> expected(in_out_array.size(), 0);
    for(size_t i = 0; i < in_out_array.size() / logical_warp_size; i++)
    {
        expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = in_out_array[idx-1] + expected[idx-1];
        }
    }

    hc::array_view<int, 1> d_in_out_array(in_out_array.size(), in_out_array.data());
    hc::parallel_for_each(
        hc::extent<1>(in_out_array.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_in_out_array[i];
            rp::warp_scan<int, logical_warp_size> wscan;
            wscan.exclusive_scan(value, value, init);
            d_in_out_array[i] = value;
        }
    );

    d_in_out_array.synchronize();
    for(int i = 0; i < in_out_array.size(); i++)
    {
        EXPECT_EQ(in_out_array[i], expected[i]);
    }
}

TYPED_TEST(RocprimWarpScanShuffleBasedTests, ExclusiveScanReduceInt)
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
    std::vector<int> in_out_array = get_random_data<int>(size, -100, 100);
    const int init = get_random_value(0, 100);

    std::vector<int> output_reductions(size / logical_warp_size);

    // Calulcate expected results on host
    std::vector<int> expected(in_out_array.size(), 0);
    std::vector<int> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < in_out_array.size() / logical_warp_size; i++)
    {
        expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = in_out_array[idx-1] + expected[idx-1];
        }

        expected_reductions[i] = 0;
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected_reductions[i] += in_out_array[idx];
        }
    }

    hc::array_view<int, 1> d_in_out_array(in_out_array.size(), in_out_array.data());
    hc::array_view<int, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(in_out_array.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int value = d_in_out_array[i];
            int reduction;
            rp::warp_scan<int, logical_warp_size> wscan;
            wscan.exclusive_scan(value, value, init, reduction);
            d_in_out_array[i] = value;
            if(i.local[0]%logical_warp_size == 0)
            {
                d_output_r[i.global[0]/logical_warp_size] = reduction;
            }
        }
    );

    d_in_out_array.synchronize();
    for(int i = 0; i < in_out_array.size(); i++)
    {
        EXPECT_EQ(in_out_array[i], expected[i]);
    }

    d_output_r.synchronize();
    for(int i = 0; i < output_reductions.size(); i++)
    {
        EXPECT_EQ(output_reductions[i], expected_reductions[i]);
    }
}

TYPED_TEST(RocprimWarpScanShuffleBasedTests, ScanInt)
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
    std::vector<int> input = get_random_data<int>(size, -100, 100);
    const int init = get_random_value(0, 100);

    std::vector<int> i_output(input.size());
    std::vector<int> e_output(input.size());

    // Calulcate expected results on host
    std::vector<int> e_expected(input.size(), 0);
    std::vector<int> i_expected(input.size(), 0);
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            i_expected[idx] = input[idx] + i_expected[j > 0 ? idx-1 : idx];
        }

        e_expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            e_expected[idx] = input[idx-1] + e_expected[idx-1];
        }
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_i_output(i_output.size(), i_output.data());
    hc::array_view<int, 1> d_e_output(e_output.size(), e_output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            rp::warp_scan<int, logical_warp_size> wscan;

            int input = d_input[i];
            int i_output, e_output;
            wscan.scan(input, i_output, e_output, init);
            d_i_output[i] = i_output;
            d_e_output[i] = e_output;
        }
    );

    d_i_output.synchronize();
    for(int i = 0; i < i_output.size(); i++)
    {
        EXPECT_EQ(i_output[i], i_expected[i]);
    }

    d_e_output.synchronize();
    for(int i = 0; i < e_output.size(); i++)
    {
        EXPECT_EQ(e_output[i], e_expected[i]);
    }
}

TYPED_TEST(RocprimWarpScanShuffleBasedTests, ScanReduceInt)
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
    std::vector<int> input = get_random_data<int>(size, -100, 100);
    const int init = get_random_value(0, 100);

    std::vector<int> i_output(input.size());
    std::vector<int> e_output(input.size());
    std::vector<int> output_reductions(input.size() / logical_warp_size);

    // Calulcate expected results on host
    std::vector<int> e_expected(input.size(), 0);
    std::vector<int> i_expected(input.size(), 0);
    std::vector<int> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < input.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            i_expected[idx] = input[idx] + i_expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = i_expected[(i+1) * logical_warp_size - 1];

        e_expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            e_expected[idx] = input[idx-1] + e_expected[idx-1];
        }
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_i_output(i_output.size(), i_output.data());
    hc::array_view<int, 1> d_e_output(e_output.size(), e_output.data());
    hc::array_view<int, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            rp::warp_scan<int, logical_warp_size> wscan;

            int input = d_input[i];
            int i_output, e_output, reduction;
            wscan.scan(input, i_output, e_output, init, reduction);
            d_i_output[i] = i_output;
            d_e_output[i] = e_output;
            if(i.local[0]%logical_warp_size == 0)
            {
                d_output_r[i.global[0]/logical_warp_size] = reduction;
            }
        }
    );

    d_i_output.synchronize();
    for(int i = 0; i < i_output.size(); i++)
    {
        EXPECT_EQ(i_output[i], i_expected[i]);
    }

    d_e_output.synchronize();
    for(int i = 0; i < e_output.size(); i++)
    {
        EXPECT_EQ(e_output[i], e_expected[i]);
    }

    d_output_r.synchronize();
    for(int i = 0; i < output_reductions.size(); i++)
    {
        EXPECT_EQ(output_reductions[i], expected_reductions[i]);
    }
}
