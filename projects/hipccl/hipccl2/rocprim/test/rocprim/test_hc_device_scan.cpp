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
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

// Params for tests
template<
    class InputType,
    class OutputType = InputType
>
struct DeviceScanParams
{
    using input_type = InputType;
    using output_type = OutputType;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceScanTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    //
    // -----------------------------------------------------------------------
    DeviceScanParams<int, long>,
    DeviceScanParams<unsigned char, float>
> RocprimDeviceScanTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        2, 32, 65, 378,
        1512, 3048, 4096,
        27845, (1 << 18) + 1111
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceScanTests, RocprimDeviceScanTestsParams);

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanSum)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);

        hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        acc_view.wait();

        // scan function
        ::rocprim::plus<U> plus_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        test_utils::host_inclusive_scan(
            input.begin(), input.end(), expected.begin(), plus_op
        );

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::inclusive_scan(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            input.size(),
            plus_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
        acc_view.wait();

        // Run
        rocprim::inclusive_scan(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            input.size(),
            plus_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < output.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "where index = " << i);
            auto diff = std::max<U>(std::abs(0.1f * expected[i]), U(0.01f));
            if(std::is_integral<U>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff);
        }
    }
}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScanSum)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);

        hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        acc_view.wait();

        // scan function
        ::rocprim::plus<T> plus_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size(), 0);
        T initial_value = test_utils::get_random_value<T>(1, 100);
        test_utils::host_exclusive_scan(
            input.begin(), input.end(),
            initial_value, expected.begin(), plus_op
        );

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::exclusive_scan(
            nullptr, temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            initial_value,
            input.size(),
            plus_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
        acc_view.wait();

        // Run
        rocprim::exclusive_scan(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            initial_value,
            input.size(),
            plus_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < output.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "where index = " << i);
            auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
            if(std::is_integral<U>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff);
        }
    }
}
