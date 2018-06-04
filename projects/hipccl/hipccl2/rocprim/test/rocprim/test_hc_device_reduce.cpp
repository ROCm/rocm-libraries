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

// rocPRIM HC API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

// Params for tests
template<
    class InputType,
    class OutputType = InputType
>
struct DeviceReduceParams
{
    using input_type = InputType;
    using output_type = OutputType;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceReduceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceReduceParams<int, long>,
    DeviceReduceParams<unsigned char, float>,
    DeviceReduceParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>,
    DeviceReduceParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>
> RocprimDeviceReduceTestsParams;

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

TYPED_TEST_CASE(RocprimDeviceReduceTests, RocprimDeviceReduceTestsParams);

TYPED_TEST(RocprimDeviceReduceTests, Reduce)
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
        hc::array<U> d_output(1, acc_view);
        acc_view.wait();

        // reduce function
        ::rocprim::plus<U> plus_op;

        // Calculate expected results on host
        U expected = U(0);
        for(unsigned int i = 0; i < input.size(); i++)
        {
            expected = plus_op(expected, input[i]);
        }


        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::reduce(
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
        rocprim::reduce(
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, 0.01f));
    }
}

TYPED_TEST(RocprimDeviceReduceTests, ReduceMinimum)
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
        hc::array<U> d_output(1, acc_view);
        acc_view.wait();

        // reduce function
        ::rocprim::minimum<U> min_op;

        // Calculate expected results on host
        U expected = U(test_utils::numeric_limits<U>::max());
        for(unsigned int i = 0; i < input.size(); i++)
        {
            expected = min_op(expected, input[i]);
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::reduce(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            test_utils::numeric_limits<U>::max(),
            input.size(),
            min_op,
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
        rocprim::reduce(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            test_utils::numeric_limits<U>::max(),
            input.size(),
            min_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<U> output = d_output;
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near<U>(output[0], expected, 0.01f));
    }
}

template<
    class Key,
    class Value
>
struct arg_min
{
    ROCPRIM_HOST_DEVICE inline
    constexpr rocprim::key_value_pair<Key, Value>
    operator()(const rocprim::key_value_pair<Key, Value>& a,
               const rocprim::key_value_pair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

TYPED_TEST(RocprimDeviceReduceTests, ReduceArgMinimum)
{
    using T = typename TestFixture::input_type;
    using key_value = rocprim::key_value_pair<int, T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_value> input(size);
        for (size_t i = 0; i < size; i++)
        {
            input[i].key = i;
            input[i].value = test_utils::get_random_value<T>(1, 100);
        }

        hc::array<key_value> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<key_value> d_output(1, acc_view);
        acc_view.wait();

        arg_min<int, T> reduce_op;
        const key_value max(std::numeric_limits<int>::max(), test_utils::numeric_limits<T>::max());

        // Calculate expected results on host
        key_value expected = max;
        for(unsigned int i = 0; i < input.size(); i++)
        {
            expected = reduce_op(expected, input[i]);
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::reduce(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            max,
            input.size(),
            reduce_op,
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
        rocprim::reduce(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            max,
            input.size(),
            reduce_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<key_value> output = d_output;
        ASSERT_EQ(output[0].key, expected.key);
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0].value, expected.value, 0.01f));
    }
}
