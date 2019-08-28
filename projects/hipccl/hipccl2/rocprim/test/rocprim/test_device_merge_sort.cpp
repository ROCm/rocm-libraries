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

// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

namespace rp = rocprim;

// Params for tests
template<
    class KeyType,
    class ValueType = KeyType,
    class CompareFunction = ::rocprim::less<KeyType>
>
struct DeviceSortParams
{
    using key_type = KeyType;
    using value_type = ValueType;
    using compare_function = CompareFunction;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceSortTests : public ::testing::Test
{
public:
    using key_type = typename Params::key_type;
    using value_type = typename Params::value_type;
    using compare_function = typename Params::compare_function;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceSortParams<unsigned short, int>,
    DeviceSortParams<signed char, test_utils::custom_test_type<float>>,
    DeviceSortParams<int>,
    DeviceSortParams<test_utils::custom_test_type<int>>,
    DeviceSortParams<unsigned long>,
    DeviceSortParams<float, double>,
    DeviceSortParams<int8_t, int8_t>,
    DeviceSortParams<uint8_t, uint8_t>,
    DeviceSortParams<rocprim::half, rocprim::half, test_utils::half_less>,
    DeviceSortParams<int, float, ::rocprim::greater<int>>,
    DeviceSortParams<short, test_utils::custom_test_type<int>>,
    DeviceSortParams<double, test_utils::custom_test_type<double>>,
    DeviceSortParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<double>>
> RocprimDeviceSortTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        128, 256, 512,
        1024, 2048, 5000,
        34567, (1 << 17) - 1220, (1 << 20) - 123
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 100000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceSortTests, RocprimDeviceSortTestsParams);

TYPED_TEST(RocprimDeviceSortTests, SortKey)
{
    using key_type = typename TestFixture::key_type;
    using compare_function = typename TestFixture::compare_function;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    bool in_place = false;

    for(size_t size : get_sizes())
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        in_place = !in_place;

        // Generate data
        std::vector<key_type> input = test_utils::get_random_data<key_type>(size, -100, 100); // float16 can't exceed 65504
        std::vector<key_type> output(size);

        key_type * d_input;
        key_type * d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(key_type)));
        if(in_place)
        {
            d_output = d_input;
        }
        else
        {
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(key_type)));
        }
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // compare function
        compare_function compare_op;

        // Calculate expected results on host
        std::vector<key_type> expected(input);
        std::stable_sort(
            expected.begin(),
            expected.end(),
            compare_op
        );

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::merge_sort(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, input.size(),
                compare_op, stream, debug_synchronous
            )
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::merge_sort(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, input.size(),
                compare_op, stream, debug_synchronous
            )
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

        hipFree(d_input);
        if(!in_place)
        {
            hipFree(d_output);
        }
        hipFree(d_temp_storage);
    }
}

TYPED_TEST(RocprimDeviceSortTests, SortKeyValue)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using compare_function = typename TestFixture::compare_function;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    bool in_place = false;

    for(size_t size : get_sizes())
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        in_place = !in_place;

        // Generate data
        std::vector<key_type> keys_input = test_utils::get_random_data<key_type>(size, -100, 100); // float16 can't exceed 65504

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        std::vector<key_type> keys_output(size);
        std::vector<value_type> values_output(size);

        key_type * d_keys_input;
        key_type * d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, keys_input.size() * sizeof(key_type)));
        if(in_place)
        {
            d_keys_output = d_keys_input;
        }
        else
        {
            HIP_CHECK(hipMalloc(&d_keys_output, keys_output.size() * sizeof(key_type)));
        }
        HIP_CHECK(
            hipMemcpy(
                d_keys_input, keys_input.data(),
                keys_input.size() * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        value_type * d_values_input;
        value_type * d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, values_input.size() * sizeof(value_type)));
        if(in_place)
        {
            d_values_output = d_values_input;
        }
        else
        {
            HIP_CHECK(hipMalloc(&d_values_output, values_output.size() * sizeof(value_type)));
        }
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                values_input.size() * sizeof(value_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // compare function
        compare_function compare_op;

        // Calculate expected results on host
        using key_value = std::pair<key_type, value_type>;
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_input[i], values_input[i]);
        }
        std::stable_sort(
            expected.begin(),
            expected.end(),
            [compare_op](const key_value& a, const key_value& b) { return compare_op(a.first, b.first); }
        );

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::merge_sort(
                d_temp_storage, temp_storage_size_bytes,
                d_keys_input, d_keys_output,
                d_values_input, d_values_output, keys_input.size(),
                compare_op, stream, debug_synchronous
            )
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::merge_sort(
                d_temp_storage, temp_storage_size_bytes,
                d_keys_input, d_keys_output,
                d_values_input, d_values_output, keys_input.size(),
                compare_op, stream, debug_synchronous
            )
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), d_keys_output,
                keys_output.size() * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                values_output.data(), d_values_output,
                values_output.size() * sizeof(value_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        std::vector<key_type> expected_key(expected.size());
        std::vector<value_type> expected_value(expected.size());
        for(size_t i = 0; i < expected.size(); i++)
        {
            expected_key[i] = expected[i].first;
            expected_value[i] = expected[i].second;
        }
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected_key));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, expected_value));

        hipFree(d_keys_input);
        hipFree(d_values_input);
        if(!in_place)
        {
            hipFree(d_keys_output);
            hipFree(d_values_output);
        }
        hipFree(d_temp_storage);
    }
}
