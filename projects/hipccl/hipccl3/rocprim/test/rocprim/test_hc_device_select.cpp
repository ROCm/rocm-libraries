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
    class OutputType = InputType,
    class FlagType = unsigned int
>
struct DeviceSelectParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using flag_type = FlagType;
};

template<class Params>
class RocprimDeviceSelectTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using flag_type = typename Params::flag_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceSelectParams<int, long>,
    DeviceSelectParams<unsigned char, float>
> RocprimDeviceSelectTestsParams;

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

TYPED_TEST_CASE(RocprimDeviceSelectTests, RocprimDeviceSelectTestsParams);

TYPED_TEST(RocprimDeviceSelectTests, Flagged)
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

        // Calculate expected results on host
        std::vector<U> expected;
        expected.reserve(input.size());
        for(size_t i = 0; i < input.size(); i++)
        {
            if(flags[i] != 0)
            {
                expected.push_back(input[i]);
            }
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::select(
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
        rocprim::select(
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
        ASSERT_EQ(selected_count_output[0], expected.size());

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < expected.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "where index = " << i);
            ASSERT_EQ(output[i], expected[i]);
        }
    }
}

TYPED_TEST(RocprimDeviceSelectTests, SelectOp)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    auto select_op = [](const T& value) [[hc,cpu]] -> bool
        {
            if(value > 50) return true;
            return false;
        };

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 0, 1);

        hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        hc::array<unsigned int> d_selected_count_output(1, acc_view);
        acc_view.wait();

        // Calculate expected results on host
        std::vector<U> expected;
        expected.reserve(input.size());
        for(size_t i = 0; i < input.size(); i++)
        {
            if(select_op(input[i]))
            {
                expected.push_back(input[i]);
            }
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::select(
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
        rocprim::select(
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
        ASSERT_EQ(selected_count_output[0], expected.size());

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < expected.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "where index = " << i);
            ASSERT_EQ(output[i], expected[i]);
        }
    }
}

std::vector<float> get_discontinuity_probabilities()
{
    std::vector<float> probabilities = {
        0.5, 0.25, 0.5, 0.75, 0.95
    };
    return probabilities;
}

TYPED_TEST(RocprimDeviceSelectTests, Unique)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const auto sizes = get_sizes();
    const auto probabilities = get_discontinuity_probabilities();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto p : probabilities)
        {
            SCOPED_TRACE(testing::Message() << "with p = " << p);

            // Generate data
            std::vector<T> input(size);
            {
                std::vector<T> input01 = test_utils::get_random_data01<T>(size, p);
                test_utils::host_inclusive_scan(
                    input01.begin(), input01.end(), input.begin(), rocprim::plus<T>()
                );
            }

            // Allocate and copy to device
            hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
            hc::array<U> d_output(size, acc_view);
            hc::array<unsigned int> d_selected_count_output(1, acc_view);
            acc_view.wait();

            // Calculate expected results on host
            std::vector<U> expected;
            expected.reserve(input.size());
            expected.push_back(input[0]);
            for(size_t i = 1; i < input.size(); i++)
            {
                if(!(input[i-1] == input[i]))
                {
                    expected.push_back(input[i]);
                }
            }

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            rocprim::unique(
                nullptr,
                temp_storage_size_bytes,
                d_input.accelerator_pointer(),
                d_output.accelerator_pointer(),
                d_selected_count_output.accelerator_pointer(),
                input.size(),
                ::rocprim::equal_to<T>(),
                acc_view,
                debug_synchronous
            );
            acc_view.wait();

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            hc::array<unsigned char> d_temp_storage(temp_storage_size_bytes, acc_view);
            acc_view.wait();

            // Run
            rocprim::unique(
                d_temp_storage.accelerator_pointer(),
                temp_storage_size_bytes,
                d_input.accelerator_pointer(),
                d_output.accelerator_pointer(),
                d_selected_count_output.accelerator_pointer(),
                input.size(),
                ::rocprim::equal_to<T>(),
                acc_view,
                debug_synchronous
            );
            acc_view.wait();

            // Check if number of selected value is as expected
            std::vector<unsigned int> selected_count_output = d_selected_count_output;
            ASSERT_EQ(selected_count_output[0], expected.size());

            // Check if output values are as expected
            std::vector<U> output = d_output;
            for(size_t i = 0; i < expected.size(); i++)
            {
                SCOPED_TRACE(testing::Message() << "where index = " << i);
                ASSERT_EQ(output[i], expected[i]);
            }
        }
    }
}
