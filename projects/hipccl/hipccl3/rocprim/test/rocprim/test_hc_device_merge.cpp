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
    class KeyType,
    class ValueType,
    class CompareOp = ::rocprim::less<KeyType>
>
struct DeviceMergeParams
{
    using key_type = KeyType;
    using value_type = ValueType;
    using compare_op_type = CompareOp;
};

template<class Params>
class RocprimDeviceMergeTests : public ::testing::Test
{
public:
    using key_type = typename Params::key_type;
    using value_type = typename Params::value_type;
    using compare_op_type = typename Params::compare_op_type;
    const bool debug_synchronous = false;
};

using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    DeviceMergeParams<int, double>,
    DeviceMergeParams<unsigned long, unsigned int, ::rocprim::greater<unsigned long> >,
    DeviceMergeParams<float, custom_double2>,
    DeviceMergeParams<int, float>,
    DeviceMergeParams<custom_double2, custom_int2, ::rocprim::greater<custom_double2> >,
    DeviceMergeParams<custom_int2, char>
> RocprimDeviceMergeTestsParams;

// size1, size2
std::vector<std::tuple<size_t, size_t>> get_sizes()
{
    std::vector<std::tuple<size_t, size_t>> sizes = {
        std::make_tuple(2, 1),
        std::make_tuple(10, 10),
        std::make_tuple(111, 111),
        std::make_tuple(128, 1289),
        std::make_tuple(12, 1000),
        std::make_tuple(123, 3000),
        std::make_tuple(1024, 512),
        std::make_tuple(2345, 49),
        std::make_tuple(17867, 41),
        std::make_tuple(17867, 34567),
        std::make_tuple(34567, (1 << 17) - 1220),
        std::make_tuple(924353, 1723454),
    };
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceMergeTests, RocprimDeviceMergeTestsParams);

TYPED_TEST(RocprimDeviceMergeTests, MergeKey)
{
    using key_type = typename TestFixture::key_type;
    using compare_op_type = typename TestFixture::compare_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    for(auto sizes : get_sizes())
    {
        SCOPED_TRACE(
            testing::Message() << "with sizes = {" <<
            std::get<0>(sizes) << ", " << std::get<1>(sizes) << "}"
        );

        const size_t size1 = std::get<0>(sizes);
        const size_t size2 = std::get<1>(sizes);

        // compare function
        compare_op_type compare_op;

        // Generate data
        std::vector<key_type> keys_input1 = test_utils::get_random_data<key_type>(size1, 0, size1);
        std::vector<key_type> keys_input2 = test_utils::get_random_data<key_type>(size2, 0, size2);
        std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_op);

        // Calculate expected results on host
        std::vector<key_type> expected(size1 + size2);
        std::merge(
            keys_input1.begin(),
            keys_input1.end(),
            keys_input2.begin(),
            keys_input2.end(),
            expected.begin(),
            compare_op
        );

        test_utils::out_of_bounds_flag out_of_bounds(acc_view);

        hc::array<key_type> d_keys_input1(hc::extent<1>(size1), keys_input1.begin(), acc_view);
        hc::array<key_type> d_keys_input2(hc::extent<1>(size2), keys_input2.begin(), acc_view);
        hc::array<key_type> d_keys_output(size1 + size2, acc_view);


        test_utils::bounds_checking_iterator<key_type> d_keys_checking_output(
            d_keys_output.accelerator_pointer(),
            out_of_bounds.device_pointer(),
            size1 + size2
        );

        // temp storage
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        rocprim::merge(
            nullptr, temp_storage_size_bytes,
            d_keys_input1.accelerator_pointer(),
            d_keys_input2.accelerator_pointer(),
            d_keys_checking_output,
            keys_input1.size(), keys_input2.size(),
            compare_op, acc_view, debug_synchronous
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);

        // Run
        rocprim::merge(
            d_temp_storage.accelerator_pointer(), temp_storage_size_bytes,
            d_keys_input1.accelerator_pointer(),
            d_keys_input2.accelerator_pointer(),
            d_keys_checking_output,
            keys_input1.size(), keys_input2.size(),
            compare_op, acc_view, debug_synchronous
        );
        acc_view.wait();

        ASSERT_FALSE(out_of_bounds.get());

        // Check if keys_output values are as expected
        std::vector<key_type> keys_output = d_keys_output;
        for(size_t i = 0; i < keys_output.size(); i++)
        {
            ASSERT_EQ(keys_output[i], expected[i]);
        }
    }
}

TYPED_TEST(RocprimDeviceMergeTests, MergeKeyValue)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using compare_op_type = typename TestFixture::compare_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    using key_value = std::pair<key_type, value_type>;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    for(auto sizes : get_sizes())
    {
        SCOPED_TRACE(
            testing::Message() << "with sizes = {" <<
            std::get<0>(sizes) << ", " << std::get<1>(sizes) << "}"
        );

        const size_t size1 = std::get<0>(sizes);
        const size_t size2 = std::get<1>(sizes);

        // compare function
        compare_op_type compare_op;

        // Generate data
        std::vector<key_type> keys_input1 = test_utils::get_random_data<key_type>(size1, 0, size1);
        std::vector<key_type> keys_input2 = test_utils::get_random_data<key_type>(size2, 0, size2);
        std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_op);
        std::vector<value_type> values_input1(size1);
        std::vector<value_type> values_input2(size2);
        std::iota(values_input1.begin(), values_input1.end(), 0);
        std::iota(values_input2.begin(), values_input2.end(), size1);

        // Calculate expected results on host
        std::vector<key_value> vector1(size1);
        std::vector<key_value> vector2(size2);

        for(size_t i = 0; i < size1; i++)
        {
            vector1[i] = key_value(keys_input1[i], values_input1[i]);
        }
        for(size_t i = 0; i < size2; i++)
        {
            vector2[i] = key_value(keys_input2[i], values_input2[i]);
        }

        std::vector<key_value> expected(size1 + size2);
        std::merge(
            vector1.begin(),
            vector1.end(),
            vector2.begin(),
            vector2.end(),
            expected.begin(),
            [compare_op](const key_value& a, const key_value& b) { return compare_op(a.first, b.first); }
        );

        test_utils::out_of_bounds_flag out_of_bounds(acc_view);

        hc::array<key_type> d_keys_input1(hc::extent<1>(size1), keys_input1.begin(), acc_view);
        hc::array<key_type> d_keys_input2(hc::extent<1>(size2), keys_input2.begin(), acc_view);
        hc::array<key_type> d_keys_output(size1 + size2, acc_view);
        hc::array<value_type> d_values_input1(hc::extent<1>(size1), values_input1.begin(), acc_view);
        hc::array<value_type> d_values_input2(hc::extent<1>(size2), values_input2.begin(), acc_view);
        hc::array<value_type> d_values_output(size1 + size2, acc_view);


        test_utils::bounds_checking_iterator<key_type> d_keys_checking_output(
            d_keys_output.accelerator_pointer(),
            out_of_bounds.device_pointer(),
            size1 + size2
        );
        test_utils::bounds_checking_iterator<value_type> d_values_checking_output(
            d_values_output.accelerator_pointer(),
            out_of_bounds.device_pointer(),
            size1 + size2
        );

        // temp storage
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        rocprim::merge(
            nullptr, temp_storage_size_bytes,
            d_keys_input1.accelerator_pointer(),
            d_keys_input2.accelerator_pointer(),
            d_keys_checking_output,
            d_values_input1.accelerator_pointer(),
			d_values_input2.accelerator_pointer(),
            d_values_checking_output,
            keys_input1.size(), keys_input2.size(),
            compare_op, acc_view, debug_synchronous
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);

        // Run
        rocprim::merge(
            d_temp_storage.accelerator_pointer(), temp_storage_size_bytes,
            d_keys_input1.accelerator_pointer(),
            d_keys_input2.accelerator_pointer(),
            d_keys_checking_output,
            d_values_input1.accelerator_pointer(),
			d_values_input2.accelerator_pointer(),
            d_values_checking_output,
            keys_input1.size(), keys_input2.size(),
            compare_op, acc_view, debug_synchronous
        );
        acc_view.wait();

        ASSERT_FALSE(out_of_bounds.get());

        // Check if keys_output values are as expected
        std::vector<key_type> keys_output = d_keys_output;
        std::vector<value_type> values_output = d_values_output;
        for(size_t i = 0; i < keys_output.size(); i++)
        {
            ASSERT_EQ(keys_output[i], expected[i].first);
            ASSERT_EQ(values_output[i], expected[i].second);
        }
    }
}
