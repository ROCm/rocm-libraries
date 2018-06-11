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

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class FlagType = unsigned int
>
struct DevicePartitionParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using flag_type = FlagType;
};

template<class Params>
class RocprimDevicePartitionTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using flag_type = typename Params::flag_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DevicePartitionParams<int, int, unsigned char>,
    DevicePartitionParams<unsigned int, unsigned long>,
    DevicePartitionParams<unsigned char, float>,
    DevicePartitionParams<test_utils::custom_test_type<long long>>
> RocprimDevicePartitionTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        2, 32, 64, 256,
        1024, 2048,
        3072, 4096,
        27845, (1 << 18) + 1111
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDevicePartitionTests, RocprimDevicePartitionTestsParams);

TYPED_TEST(RocprimDevicePartitionTests, Flagged)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);
        std::vector<F> flags = test_utils::get_random_data<F>(size, 0, 1);

        hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<F> d_flags(hc::extent<1>(size), flags.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        hc::array<unsigned int> d_selected_count_output(1, acc_view);
        acc_view.wait();

        // Calculate expected_selected and expected_rejected results on host
        std::vector<U> expected_selected;
        std::vector<U> expected_rejected;
        expected_selected.reserve(input.size()/2);
        expected_rejected.reserve(input.size()/2);
        for(size_t i = 0; i < input.size(); i++)
        {
            if(flags[i] != 0)
            {
                expected_selected.push_back(input[i]);
            }
            else
            {
                expected_rejected.push_back(input[i]);
            }
        }
        std::reverse(expected_rejected.begin(), expected_rejected.end());

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::partition(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_flags.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_selected_count_output.accelerator_pointer(),
            input.size(),
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
        rocprim::partition(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_flags.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_selected_count_output.accelerator_pointer(),
            input.size(),
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if number of selected value is as expected
        std::vector<unsigned int> selected_count_output = d_selected_count_output;
        ASSERT_EQ(selected_count_output[0], expected_selected.size());

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < expected_selected.size(); i++)
        {
            ASSERT_EQ(output[i], expected_selected[i]) << "where index = " << i;
        }
        for(size_t i = 0; i < expected_rejected.size(); i++)
        {
            auto j = i + expected_selected.size();
            ASSERT_EQ(output[j], expected_rejected[i]) << "where index = " << j;
        }
    }
}

TYPED_TEST(RocprimDevicePartitionTests, Predicate)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    auto select_op = [](const T& value) [[hc,cpu]] -> bool
        {
            if(value == T(50)) return true;
            return false;
        };

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100);

        hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        hc::array<unsigned int> d_selected_count_output(1, acc_view);
        acc_view.wait();

        // Calculate expected_selected and expected_rejected results on host
        std::vector<U> expected_selected;
        std::vector<U> expected_rejected;
        expected_selected.reserve(input.size()/2);
        expected_rejected.reserve(input.size()/2);
        for(size_t i = 0; i < input.size(); i++)
        {
            if(select_op(input[i]))
            {
                expected_selected.push_back(input[i]);
            }
            else
            {
                expected_rejected.push_back(input[i]);
            }
        }
        std::reverse(expected_rejected.begin(), expected_rejected.end());

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::partition(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_selected_count_output.accelerator_pointer(),
            input.size(),
            select_op,
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
        rocprim::partition(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_selected_count_output.accelerator_pointer(),
            input.size(),
            select_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if number of selected value is as expected
        std::vector<unsigned int> selected_count_output = d_selected_count_output;
        ASSERT_EQ(selected_count_output[0], expected_selected.size());

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < expected_selected.size(); i++)
        {
            ASSERT_EQ(output[i], expected_selected[i]) << "where index = " << i;
        }
        for(size_t i = 0; i < expected_rejected.size(); i++)
        {
            auto j = i + expected_selected.size();
            ASSERT_EQ(output[j], expected_rejected[i]) << "where index = " << j;
        }
    }
}
