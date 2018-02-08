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

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>

// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<
    class Key,
    class Value,
    class ReduceOp,
    unsigned int MinSegmentLength = 1,
    unsigned int MaxSegmentLength = 30
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    using reduce_op_type = ReduceOp;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;

};

template<class Params>
class RocprimDeviceReduceByKey : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<unsigned int, unsigned int, rp::plus<unsigned int>, 1, 1>,
    params<float, int, rp::plus<int>, 1, 10>,
    params<unsigned long long, float, rp::minimum<float>, 1, 30>,
    params<int, unsigned int, rp::maximum<unsigned int>, 20, 100>,
    params<int, unsigned long long, rp::maximum<unsigned long long>, 100, 400>,
    params<unsigned int, unsigned int, rp::plus<unsigned int>, 200, 600>,
    params<int, unsigned int, rp::plus<unsigned int>, 1000, 5000>,
    params<unsigned int, int, rp::plus<int>, 2048, 2048>,
    params<unsigned int, double, rp::minimum<double>, 1000, 50000>,
    params<unsigned long long, unsigned long long, rp::plus<unsigned long long>, 100000, 100000>
> Params;

TYPED_TEST_CASE(RocprimDeviceReduceByKey, Params);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        100000,
        (1 << 16) - 1220, (1 << 23) - 76543
    };
    const std::vector<size_t> random_sizes = get_random_data<size_t>(100, 1, 1000000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST(RocprimDeviceReduceByKey, ReduceByKey)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;
    using key_distribution_type = typename std::conditional<
        std::is_floating_point<key_type>::value,
        std::uniform_real_distribution<key_type>,
        std::uniform_int_distribution<key_type>
    >::type;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const bool debug_synchronous = false;

    reduce_op_type reduce_op;

    const std::vector<size_t> sizes = get_sizes();

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data and calculate expected results
        std::vector<key_type> unique_expected;
        std::vector<value_type> aggregates_expected;
        size_t unique_count_expected = 0;

        std::vector<key_type> keys_input(size);
        key_distribution_type key_delta_dis(1, 5);
        std::uniform_int_distribution<size_t> key_count_dis(
            TestFixture::params::min_segment_length,
            TestFixture::params::max_segment_length
        );
        std::vector<value_type> values_input = get_random_data<value_type>(size, 0, 100);

        size_t offset = 0;
        key_type current_key = key_distribution_type(0, 100)(gen);
        while(offset < size)
        {
            const size_t key_count = key_count_dis(gen);
            current_key += key_delta_dis(gen);

            const size_t end = std::min(size, offset + key_count);
            for(size_t i = offset; i < end; i++)
            {
                keys_input[i] = current_key;
            }
            unique_expected.push_back(current_key);
            unique_count_expected++;
            value_type aggregate = values_input[offset];
            for(size_t i = offset + 1; i < end; i++)
            {
                aggregate = reduce_op(aggregate, values_input[i]);
            }
            aggregates_expected.push_back(aggregate);

            offset += key_count;
        }

        hc::array<key_type> d_keys_input(hc::extent<1>(size), keys_input.begin(), acc_view);
        hc::array<value_type> d_values_input(hc::extent<1>(size), values_input.begin(), acc_view);

        hc::array<key_type> d_unique_output(size, acc_view);
        hc::array<value_type> d_aggregates_output(size, acc_view);
        hc::array<unsigned int> d_unique_count_output(1, acc_view);

        size_t temporary_storage_bytes;

        rp::device_reduce_by_key(
            nullptr, temporary_storage_bytes,
            d_keys_input.accelerator_pointer(), d_values_input.accelerator_pointer(), size,
            d_unique_output.accelerator_pointer(), d_aggregates_output.accelerator_pointer(),
            d_unique_count_output.accelerator_pointer(),
            reduce_op,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0);

        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rp::device_reduce_by_key(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_keys_input.accelerator_pointer(), d_values_input.accelerator_pointer(), size,
            d_unique_output.accelerator_pointer(), d_aggregates_output.accelerator_pointer(),
            d_unique_count_output.accelerator_pointer(),
            reduce_op,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<key_type> unique_output = d_unique_output;
        std::vector<value_type> aggregates_output = d_aggregates_output;
        std::vector<unsigned int> unique_count_output = d_unique_count_output;

        ASSERT_EQ(unique_count_output[0], unique_count_expected);

        for(size_t i = 0; i < unique_count_expected; i++)
        {
            ASSERT_EQ(unique_output[i], unique_expected[i]);
        }

        for(size_t i = 0; i < unique_count_expected; i++)
        {
            ASSERT_EQ(aggregates_output[i], aggregates_expected[i]);
        }
    }
}
