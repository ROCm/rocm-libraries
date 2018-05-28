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
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<
    class T,
    unsigned int WarpSize
>
struct params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
};

template<typename Params>
class RocprimWarpScanTests : public ::testing::Test {
public:
    using type = typename Params::type;
    static constexpr unsigned int warp_size = Params::warp_size;
};

typedef ::testing::Types<

    // Integer
    // shuffle based scan
    params<int, 2U>,
    params<int, 4U>,
    params<int, 8U>,
    params<int, 16U>,
    params<int, 32U>,
    params<int, 64U>,
    // shared memory scan
    params<int, 3U>,
    params<int, 7U>,
    params<int, 15U>,
    params<int, 37U>,
    params<int, 61U>,

    // Float
    // shuffle based scan
    params<float, 2U>,
    params<float, 4U>,
    params<float, 8U>,
    params<float, 16U>,
    params<float, 32U>,
    params<float, 64U>,
    // shared memory scan
    params<float, 3U>,
    params<float, 7U>,
    params<float, 15U>,
    params<float, 37U>,
    params<float, 61U>

> WarpScanTestParams;

TYPED_TEST_CASE(RocprimWarpScanTests, WarpScanTestParams);

TYPED_TEST(RocprimWarpScanTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, -100, 100); // used for input/output

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T value = d_output[i];

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().inclusive_scan(value, value, storage[warp_id]);

            d_output[i] = value;
        }
    );

    d_output.synchronize();

     // Validating results
    if (std::is_integral<T>::value)
    {
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }
    }
    else if (std::is_floating_point<T>::value)
    {
        for(size_t i = 0; i < output.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            ASSERT_NEAR(output[i], expected[i], tolerance);
        }
    }
}

TYPED_TEST(RocprimWarpScanTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, -100, 100);  // used for input/output
    std::vector<T> output_reductions(size / logical_warp_size);

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = expected[(i+1) * logical_warp_size - 1];
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T value = d_output[i];
            T reduction;

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().inclusive_scan(value, value, reduction, storage[warp_id]);

            d_output[i] = value;
            if(i.local[0]%logical_warp_size == 0)
            {
                d_output_r[i.global[0]/logical_warp_size] = reduction;
            }
        }
    );

    d_output.synchronize();
    d_output_r.synchronize();

    // Validating results
    if (std::is_integral<T>::value)
    {
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }
    }
    else if (std::is_floating_point<T>::value)
    {
        for(size_t i = 0; i < output.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            ASSERT_NEAR(output[i], expected[i], tolerance);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected_reductions[i]), T(0.01f));
            ASSERT_NEAR(output_reductions[i], expected_reductions[i], tolerance);
        }
    }
}

TYPED_TEST(RocprimWarpScanTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> in_out_array = test_utils::get_random_data<T>(size, -100, 100);
    const T init = test_utils::get_random_value(0, 100);

    // Calculate expected results on host
    std::vector<T> expected(in_out_array.size(), 0);
    for(size_t i = 0; i < in_out_array.size() / logical_warp_size; i++)
    {
        expected[i * logical_warp_size] = init;
        for(size_t j = 1; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = in_out_array[idx-1] + expected[idx-1];
        }
    }

    hc::array_view<T, 1> d_in_out_array(in_out_array.size(), in_out_array.data());
    hc::parallel_for_each(
        hc::extent<1>(in_out_array.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T value = d_in_out_array[i];

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().exclusive_scan(value, value, init, storage[warp_id]);

            d_in_out_array[i] = value;
        }
    );

    d_in_out_array.synchronize();

    // Validating results
    if (std::is_integral<T>::value)
    {
        for(size_t i = 0; i < in_out_array.size(); i++)
        {
            ASSERT_EQ(in_out_array[i], expected[i]);
        }
    }
    else if (std::is_floating_point<T>::value)
    {
        for(size_t i = 0; i < in_out_array.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            ASSERT_NEAR(in_out_array[i], expected[i], tolerance);
        }
    }
}

TYPED_TEST(RocprimWarpScanTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> in_out_array = test_utils::get_random_data<T>(size, -100, 100);
    const int init = test_utils::get_random_value(0, 100);

    std::vector<T> output_reductions(size / logical_warp_size);

    // Calculate expected results on host
    std::vector<T> expected(in_out_array.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
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

    hc::array_view<T, 1> d_in_out_array(in_out_array.size(), in_out_array.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(in_out_array.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T value = d_in_out_array[i];
            T reduction;

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().exclusive_scan(value, value, init, reduction, storage[warp_id]);

            d_in_out_array[i] = value;
            if(i.local[0]%logical_warp_size == 0)
            {
                d_output_r[i.global[0]/logical_warp_size] = reduction;
            }
        }
    );

    d_in_out_array.synchronize();
    d_output_r.synchronize();

    // Validating results
    if (std::is_integral<T>::value)
    {
        for(size_t i = 0; i < in_out_array.size(); i++)
        {
            ASSERT_EQ(in_out_array[i], expected[i]);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }
    }
    else if (std::is_floating_point<T>::value)
    {
        for(size_t i = 0; i < in_out_array.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            ASSERT_NEAR(in_out_array[i], expected[i], tolerance);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected_reductions[i]), T(0.01f));
            ASSERT_NEAR(output_reductions[i], expected_reductions[i], tolerance);
        }
    }
}

TYPED_TEST(RocprimWarpScanTests, Scan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100);
    const T init = test_utils::get_random_value(0, 100);

    std::vector<T> i_output(input.size());
    std::vector<T> e_output(input.size());

    // Calculate expected results on host
    std::vector<T> e_expected(input.size(), 0);
    std::vector<T> i_expected(input.size(), 0);
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

    hc::array_view<T, 1> d_input(input.size(), input.data());
    hc::array_view<T, 1> d_i_output(i_output.size(), i_output.data());
    hc::array_view<T, 1> d_e_output(e_output.size(), e_output.data());
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T input = d_input[i];
            T i_output, e_output;

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().scan(input, i_output, e_output, init, storage[warp_id]);

            d_i_output[i] = i_output;
            d_e_output[i] = e_output;
        }
    );

    d_i_output.synchronize();
    d_e_output.synchronize();

    // Validating results
    if (std::is_integral<T>::value)
    {
        for(size_t i = 0; i < i_output.size(); i++)
        {
            ASSERT_EQ(i_output[i], i_expected[i]);
            ASSERT_EQ(e_output[i], e_expected[i]);
        }
    }
    else if (std::is_floating_point<T>::value)
    {
        for(size_t i = 0; i < i_output.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * i_expected[i]), T(0.01f));
            ASSERT_NEAR(i_output[i], i_expected[i], tolerance);

            tolerance = std::max<T>(std::abs(0.1f * e_expected[i]), T(0.01f));
            ASSERT_NEAR(e_output[i], e_expected[i], tolerance);
        }
    }
}

TYPED_TEST(RocprimWarpScanTests, ScanReduce)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100);
    const T init = test_utils::get_random_value(0, 100);

    std::vector<T> i_output(input.size());
    std::vector<T> e_output(input.size());
    std::vector<T> output_reductions(input.size() / logical_warp_size);

    // Calculate expected results on host
    std::vector<T> e_expected(input.size(), 0);
    std::vector<T> i_expected(input.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
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

    hc::array_view<T, 1> d_input(input.size(), input.data());
    hc::array_view<T, 1> d_i_output(i_output.size(), i_output.data());
    hc::array_view<T, 1> d_e_output(e_output.size(), e_output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        hc::extent<1>(input.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T input = d_input[i];
            T i_output, e_output, reduction;

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().scan(input, i_output, e_output, init, reduction, storage[warp_id]);

            d_i_output[i] = i_output;
            d_e_output[i] = e_output;
            if(i.local[0]%logical_warp_size == 0)
            {
                d_output_r[i.global[0]/logical_warp_size] = reduction;
            }
        }
    );

    d_i_output.synchronize();
    d_e_output.synchronize();
    d_output_r.synchronize();

    // Validating results
    if (std::is_integral<T>::value)
    {
        for(size_t i = 0; i < i_output.size(); i++)
        {
            ASSERT_EQ(i_output[i], i_expected[i]);
            ASSERT_EQ(e_output[i], e_expected[i]);
        }
        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }
    }
    else if (std::is_floating_point<T>::value)
    {
        for(size_t i = 0; i < i_output.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * i_expected[i]), T(0.01f));
            ASSERT_NEAR(i_output[i], i_expected[i], tolerance);

            tolerance = std::max<T>(std::abs(0.1f * e_expected[i]), T(0.01f));
            ASSERT_NEAR(e_output[i], e_expected[i], tolerance);
        }
        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected_reductions[i]), T(0.01f));
            ASSERT_NEAR(output_reductions[i], expected_reductions[i], tolerance);
        }
    }
}

TYPED_TEST(RocprimWarpScanTests, InclusiveScanCustomStruct)
{
    using base_type = typename TestFixture::type;
    using T = test_utils::custom_test_type<base_type>;

    // logical warp side for warp primitive, execution warp size is always rp::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        rp::detail::is_power_of_two(logical_warp_size)
            ? rp::max<size_t>(rp::warp_size(), logical_warp_size * 4)
            : (rp::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rp::warp_size())
    {
        return;
    }

    // Generate data
    std::vector<T> output(size); // used for input/output
    {
        auto random_values =
            test_utils::get_random_data<base_type>(2 * output.size(), 0, 100);
        for(size_t i = 0; i < output.size(); i++)
        {
            output[i].x = random_values[i];
            output[i].y = random_values[i + output.size()];
        }
    }

    // Calculate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / logical_warp_size; i++)
    {
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            auto idx = i * logical_warp_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            constexpr unsigned int warps_no = block_size/logical_warp_size;
            const unsigned int warp_id = rp::detail::logical_warp_id<logical_warp_size>();

            T value = d_output[i];

            using wscan_t = rp::warp_scan<T, logical_warp_size>;
            tile_static typename wscan_t::storage_type storage[warps_no];
            wscan_t().inclusive_scan(value, value, storage[warp_id]);

            d_output[i] = value;
        }
    );

    d_output.synchronize();

    // Validating results
    if (std::is_integral<base_type>::value)
    {
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }
    }
    else if (std::is_floating_point<base_type>::value)
    {
        for(size_t i = 0; i < output.size(); i++)
        {
            auto tolerance_x = std::max<base_type>(std::abs(0.1f * expected[i].x), base_type(0.01f));
            auto tolerance_y = std::max<base_type>(std::abs(0.1f * expected[i].y), base_type(0.01f));
            ASSERT_NEAR(output[i].x, expected[i].x, tolerance_x);
            ASSERT_NEAR(output[i].y, expected[i].y, tolerance_y);
        }
    }
}
