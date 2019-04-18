// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
#include <limits>
#include <random>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// hipCUB API
#include <hipcub/hipcub.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        100000,
        (1 << 16) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 1000000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

template<
    class Input,
    class Output,
    class ReduceOp = hipcub::Sum,
    int Init = 0, // as only integral types supported, int is used here even for floating point inputs
    unsigned int MinSegmentLength = 0,
    unsigned int MaxSegmentLength = 1000
>
struct params1
{
    using input_type = Input;
    using output_type = Output;
    using reduce_op_type = ReduceOp;
    static constexpr input_type init = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedReduceOp : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params1<unsigned int, unsigned int, hipcub::Sum>,
    params1<int, int, hipcub::Sum, -100, 0, 10000>,
    params1<double, double, hipcub::Min, 1000, 0, 10000>,
    params1<int, short, hipcub::Max, 10, 1000, 10000>,
    params1<short, double, hipcub::Sum, 5, 1, 1000>,
    params1<float, double, hipcub::Max, 50, 2, 10>,
    params1<float, float, hipcub::Sum, 123, 100, 200>
> Params1;

TYPED_TEST_CASE(HipcubDeviceSegmentedReduceOp, Params1);

TYPED_TEST(HipcubDeviceSegmentedReduceOp, Reduce)
{
    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;

    using result_type = output_type;
    using offset_type = unsigned int;

    constexpr input_type init = TestFixture::params::init;
    const bool debug_synchronous = false;
    reduce_op_type reduce_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipStream_t stream = 0; // default

        // Generate data and calculate expected results
        std::vector<output_type> aggregates_expected;

        std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100);

        std::vector<offset_type> offsets;
        unsigned int segments_count = 0;
        size_t offset = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const size_t end = std::min(size, offset + segment_length);
            result_type aggregate = init;
            for(size_t i = offset; i < end; i++)
            {
                aggregate = reduce_op(aggregate, static_cast<result_type>(values_input[i]));
            }
            aggregates_expected.push_back(aggregate);

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        input_type * d_values_input;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(input_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );

        offset_type * d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(
            hipMemcpy(
                d_offsets, offsets.data(),
                (segments_count + 1) * sizeof(offset_type),
                hipMemcpyHostToDevice
            )
        );

        output_type * d_aggregates_output;
        HIP_CHECK(hipMalloc(&d_aggregates_output, segments_count * sizeof(output_type)));

        size_t temporary_storage_bytes;

        HIP_CHECK(
            hipcub::DeviceSegmentedReduce::Reduce(
                nullptr, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                reduce_op, init,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            hipcub::DeviceSegmentedReduce::Reduce(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                reduce_op, init,
                stream, debug_synchronous
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<output_type> aggregates_output(segments_count);
        HIP_CHECK(
            hipMemcpy(
                aggregates_output.data(), d_aggregates_output,
                segments_count * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_aggregates_output));

        for(size_t i = 0; i < segments_count; i++)
        {
            if(std::is_integral<output_type>::value)
            {
                ASSERT_EQ(aggregates_output[i], aggregates_expected[i]);
            }
            else
            {
                auto diff = std::max<output_type>(
                    std::abs(0.01 * aggregates_expected[i]), output_type(0.01)
                );
                ASSERT_NEAR(aggregates_output[i], aggregates_expected[i], diff);
            }
        }
    }
}

template<
    class Input,
    class Output,
    unsigned int MinSegmentLength = 0,
    unsigned int MaxSegmentLength = 1000
>
struct params2
{
    using input_type = Input;
    using output_type = Output;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedReduce : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params2<unsigned int, unsigned int>,
    params2<int, int, 0, 10000>,
    params2<double, double, 0, 10000>,
    params2<int, long long, 1000, 10000>,
    params2<float, double, 2, 10>,
    params2<float, float, 100, 200>
> Params2;

TYPED_TEST_CASE(HipcubDeviceSegmentedReduce, Params2);

TYPED_TEST(HipcubDeviceSegmentedReduce, Sum)
{
    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using reduce_op_type = typename hipcub::Sum;
    using result_type = output_type;
    using offset_type = unsigned int;

    constexpr input_type init = input_type(0);
    const bool debug_synchronous = false;
    reduce_op_type reduce_op;


    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipStream_t stream = 0; // default

        // Generate data and calculate expected results
        std::vector<output_type> aggregates_expected;

        std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100);

        std::vector<offset_type> offsets;
        unsigned int segments_count = 0;
        size_t offset = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const size_t end = std::min(size, offset + segment_length);
            result_type aggregate = init;
            for(size_t i = offset; i < end; i++)
            {
                aggregate = reduce_op(aggregate, static_cast<result_type>(values_input[i]));
            }
            aggregates_expected.push_back(aggregate);

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        input_type * d_values_input;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(input_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );

        offset_type * d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(
            hipMemcpy(
                d_offsets, offsets.data(),
                (segments_count + 1) * sizeof(offset_type),
                hipMemcpyHostToDevice
            )
        );

        output_type * d_aggregates_output;
        HIP_CHECK(hipMalloc(&d_aggregates_output, segments_count * sizeof(output_type)));

        size_t temporary_storage_bytes;

        HIP_CHECK(
            hipcub::DeviceSegmentedReduce::Sum(
                nullptr, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            hipcub::DeviceSegmentedReduce::Sum(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                stream, debug_synchronous
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<output_type> aggregates_output(segments_count);
        HIP_CHECK(
            hipMemcpy(
                aggregates_output.data(), d_aggregates_output,
                segments_count * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_aggregates_output));

        for(size_t i = 0; i < segments_count; i++)
        {
            if(std::is_integral<output_type>::value)
            {
                ASSERT_EQ(aggregates_output[i], aggregates_expected[i]);
            }
            else
            {
                auto diff = std::max<output_type>(
                    std::abs(0.01 * aggregates_expected[i]), output_type(0.01)
                );
                ASSERT_NEAR(aggregates_output[i], aggregates_expected[i], diff);
            }
        }
    }
}

TYPED_TEST(HipcubDeviceSegmentedReduce, Min)
{
    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using reduce_op_type = typename hipcub::Min;
    using result_type = output_type;
    using offset_type = unsigned int;

    constexpr input_type init = std::numeric_limits<input_type>::max();
    const bool debug_synchronous = false;
    reduce_op_type reduce_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipStream_t stream = 0; // default

        // Generate data and calculate expected results
        std::vector<output_type> aggregates_expected;

        std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100);

        std::vector<offset_type> offsets;
        unsigned int segments_count = 0;
        size_t offset = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const size_t end = std::min(size, offset + segment_length);
            result_type aggregate = init;
            for(size_t i = offset; i < end; i++)
            {
                aggregate = reduce_op(aggregate, static_cast<result_type>(values_input[i]));
            }
            aggregates_expected.push_back(aggregate);

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        input_type * d_values_input;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(input_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );

        offset_type * d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(
            hipMemcpy(
                d_offsets, offsets.data(),
                (segments_count + 1) * sizeof(offset_type),
                hipMemcpyHostToDevice
            )
        );

        output_type * d_aggregates_output;
        HIP_CHECK(hipMalloc(&d_aggregates_output, segments_count * sizeof(output_type)));

        size_t temporary_storage_bytes;

        HIP_CHECK(
            hipcub::DeviceSegmentedReduce::Min(
                nullptr, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            hipcub::DeviceSegmentedReduce::Min(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                stream, debug_synchronous
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<output_type> aggregates_output(segments_count);
        HIP_CHECK(
            hipMemcpy(
                aggregates_output.data(), d_aggregates_output,
                segments_count * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_aggregates_output));

        for(size_t i = 0; i < segments_count; i++)
        {
            ASSERT_EQ(aggregates_output[i], aggregates_expected[i]);
        }
    }
}
