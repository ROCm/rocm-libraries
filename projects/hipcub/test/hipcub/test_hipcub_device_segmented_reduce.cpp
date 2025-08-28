// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// Thread operators fixes for extended float types
#include "test_utils_data_generation.hpp"
#include "test_utils_thread_operators.hpp"

// hipcub API
#include <hipcub/device/device_segmented_reduce.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>

template<class Input,
         class Output,
         class ReduceOp = hipcub::Sum,
         int Init
         = 0, // as only integral types supported, int is used here even for floating point inputs
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000,
         bool         UseGraphs        = false>
struct params1
{
    using input_type                                 = Input;
    using output_type                                = Output;
    using reduce_op_type                             = ReduceOp;
    static constexpr int          init               = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool         use_graphs         = UseGraphs;
};

template<class Params>
class HipcubDeviceSegmentedReduceOp : public ::testing::Test
{
public:
    using params = Params;
};

using Params1
    = ::testing::Types<params1<unsigned int, unsigned int, hipcub::Sum>,
                       params1<int, int, hipcub::Sum, -100, 0, 10000>,
                       params1<double, double, hipcub::Min, 1000, 0, 10000>,
                       params1<int, short, hipcub::Max, 10, 1000, 10000>,
                       params1<short, double, hipcub::Sum, 5, 1, 1000>,
                       params1<float, double, hipcub::Max, 50, 2, 10>,
                       params1<float, float, hipcub::Sum, 123, 100, 200>,
                       params1<test_utils::half, test_utils::half, hipcub::Max, 50, 2, 10>,
                       params1<test_utils::bfloat16, test_utils::bfloat16, hipcub::Max, 50, 2, 10>,
                       params1<unsigned int, unsigned int, hipcub::Sum, 0, 1000, true>>;

TYPED_TEST_SUITE(HipcubDeviceSegmentedReduceOp, Params1);

TYPED_TEST(HipcubDeviceSegmentedReduceOp, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::params::input_type;
    using output_type    = typename TestFixture::params::output_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;

    using result_type = output_type;
    using offset_type = unsigned int;

    const input_type init = test_utils::convert_to_device<input_type>(TestFixture::params::init);
    reduce_op_type   reduce_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                result_type aggregate = init;
                for(size_t i = offset; i < end; i++)
                {
                    aggregate = reduce_op(aggregate, values_input[i]);
                }
                aggregates_expected.push_back(aggregate);

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            const float precision = test_utils::precision<result_type>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            output_type* d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                         segments_count * sizeof(output_type)));

            size_t temporary_storage_bytes;

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Reduce(nullptr,
                                                            temporary_storage_bytes,
                                                            d_values_input,
                                                            d_aggregates_output,
                                                            segments_count,
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            reduce_op,
                                                            init,
                                                            stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Reduce(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_values_input,
                                                            d_aggregates_output,
                                                            segments_count,
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            reduce_op,
                                                            init,
                                                            stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<output_type> aggregates_output(segments_count);
            HIP_CHECK(hipMemcpy(aggregates_output.data(),
                                d_aggregates_output,
                                segments_count * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_aggregates_output));

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(aggregates_output, aggregates_expected, precision));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<class Input,
         class Output,
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000,
         bool         UseGraphs        = false>
struct params2
{
    using input_type                                 = Input;
    using output_type                                = Output;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool         use_graphs         = UseGraphs;
};

template<class Params>
class HipcubDeviceSegmentedReduce : public ::testing::Test
{
public:
    using params = Params;
};

using Params2
    = ::testing::Types<params2<unsigned int, unsigned int>,
                       params2<int, int, 0, 10000>,
                       params2<double, double, 0, 10000>,
                       params2<int, long long, 1000, 10000>,
                       params2<float, double, 2, 10>,
                       params2<float, float, 100, 200>,
                       params2<unsigned int, unsigned int, 0, 1000, true>
#ifdef __HIP_PLATFORM_AMD__
                       ,
                       params2<test_utils::half, float>, // Doesn't work on NVIDIA / CUB
                       params2<test_utils::bfloat16, float> // Doesn't work on NVIDIA / CUB
#endif
                       >;

TYPED_TEST_SUITE(HipcubDeviceSegmentedReduce, Params2);

TYPED_TEST(HipcubDeviceSegmentedReduce, Sum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::params::input_type;
    using output_type    = typename TestFixture::params::output_type;
    using reduce_op_type = typename hipcub::Sum;
    using result_type    = output_type;
    using offset_type    = unsigned int;

    const input_type init = input_type(0);
    reduce_op_type   reduce_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

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

            const float precision = test_utils::precision<result_type>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            output_type* d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                         segments_count * sizeof(output_type)));

            size_t temporary_storage_bytes;

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Sum(nullptr,
                                                         temporary_storage_bytes,
                                                         d_values_input,
                                                         d_aggregates_output,
                                                         segments_count,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Sum(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_values_input,
                                                         d_aggregates_output,
                                                         segments_count,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<output_type> aggregates_output(segments_count);
            HIP_CHECK(hipMemcpy(aggregates_output.data(),
                                d_aggregates_output,
                                segments_count * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_aggregates_output));

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(aggregates_output, aggregates_expected, precision));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceSegmentedReduce, Min)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::params::input_type;
    using output_type    = typename TestFixture::params::output_type;
    using reduce_op_type = typename hipcub::Min;
    using result_type    = output_type;
    using offset_type    = unsigned int;

    constexpr input_type init = std::numeric_limits<input_type>::max();
    reduce_op_type       reduce_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

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

            const float precision = test_utils::precision<result_type>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            output_type* d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                         segments_count * sizeof(output_type)));

            size_t temporary_storage_bytes;

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Min(nullptr,
                                                         temporary_storage_bytes,
                                                         d_values_input,
                                                         d_aggregates_output,
                                                         segments_count,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Min(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_values_input,
                                                         d_aggregates_output,
                                                         segments_count,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<output_type> aggregates_output(segments_count);
            HIP_CHECK(hipMemcpy(aggregates_output.data(),
                                d_aggregates_output,
                                segments_count * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_aggregates_output));

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(aggregates_output, aggregates_expected, precision));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceSegmentedReduce, Max)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::params::input_type;
    using output_type    = typename TestFixture::params::output_type;
    using reduce_op_type = typename hipcub::Max;
    using result_type    = output_type;
    using offset_type    = unsigned int;

    constexpr input_type init = std::numeric_limits<input_type>::lowest();
    reduce_op_type       reduce_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

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

            const float precision = test_utils::precision<result_type>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            output_type* d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                         segments_count * sizeof(output_type)));

            size_t temporary_storage_bytes;

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Max(nullptr,
                                                         temporary_storage_bytes,
                                                         d_values_input,
                                                         d_aggregates_output,
                                                         segments_count,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(hipcub::DeviceSegmentedReduce::Max(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_values_input,
                                                         d_aggregates_output,
                                                         segments_count,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<output_type> aggregates_output(segments_count);
            HIP_CHECK(hipMemcpy(aggregates_output.data(),
                                d_aggregates_output,
                                segments_count * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_aggregates_output));

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(aggregates_output, aggregates_expected, precision));

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

struct ArgMinDispatch
{
    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    auto operator()(void*           d_temp_storage,
                    size_t&         temp_storage_bytes,
                    InputIteratorT  d_in,
                    OutputIteratorT d_out,
                    int             num_segments,
                    OffsetIteratorT d_begin_offsets,
                    OffsetIteratorT d_end_offsets,
                    hipStream_t     stream) const
    {
        return hipcub::DeviceSegmentedReduce::ArgMin(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_in,
                                                     d_out,
                                                     num_segments,
                                                     d_begin_offsets,
                                                     d_end_offsets,
                                                     stream);
    }
};

struct ArgMaxDispatch
{
    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    auto operator()(void*           d_temp_storage,
                    size_t&         temp_storage_bytes,
                    InputIteratorT  d_in,
                    OutputIteratorT d_out,
                    int             num_segments,
                    OffsetIteratorT d_begin_offsets,
                    OffsetIteratorT d_end_offsets,
                    hipStream_t     stream) const
    {
        return hipcub::DeviceSegmentedReduce::ArgMax(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_in,
                                                     d_out,
                                                     num_segments,
                                                     d_begin_offsets,
                                                     d_end_offsets,
                                                     stream);
    }
};

template<typename TestFixture, typename DispatchFunction, typename HostOp>
void test_argminmax(typename TestFixture::params::input_type empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::params::input_type;
    using Iterator    = typename hipcub::ArgIndexInputIterator<input_type*, int>;
    using key_value   = typename Iterator::value_type;
    using offset_type = unsigned int;

    DispatchFunction                      function;
    std::random_device                    rd;
    std::default_random_engine            gen(rd());
    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<key_value> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            HostOp                   host_op{};
            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);
                Iterator x(&values_input[offset]);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);
                if(offset < end)
                {
                    key_value aggregate(0, values_input[offset]);
                    for(size_t i = 0; i < end - offset; i++)
                    {
                        aggregate = host_op(aggregate, x[i]);
                    }
                    aggregates_expected.push_back(aggregate);
                }
                else
                {
                    // empty segments produce a special value
                    key_value aggregate(1, empty_value);
                    aggregates_expected.push_back(aggregate);
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            const float precision = test_utils::precision<key_value>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            key_value* d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                         segments_count * sizeof(key_value)));

            size_t temporary_storage_bytes{};
            void*  d_temporary_storage{};
            HIP_CHECK(function(nullptr,
                               temporary_storage_bytes,
                               d_values_input,
                               d_aggregates_output,
                               segments_count,
                               d_offsets,
                               d_offsets + 1,
                               stream));

            // temp_storage_size_bytes must be > 0
            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(function(d_temporary_storage,
                               temporary_storage_bytes,
                               d_values_input,
                               d_aggregates_output,
                               segments_count,
                               d_offsets,
                               d_offsets + 1,
                               stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<key_value> aggregates_output(segments_count);
            HIP_CHECK(hipMemcpy(aggregates_output.data(),
                                d_aggregates_output,
                                segments_count * sizeof(key_value),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_aggregates_output));
            HIP_CHECK(hipFree(d_temporary_storage));

            for(size_t i = 0; i < segments_count; i++)
            {
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(aggregates_output[i].key, aggregates_expected[i].key));
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(aggregates_output[i].value,
                                                                aggregates_expected[i].value,
                                                                precision));
            }

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceSegmentedReduce, ArgMin)
{
    using T = typename TestFixture::params::input_type;
    // Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMinSelector<T>::type;
    test_argminmax<TestFixture, ArgMinDispatch, HostOp>(test_utils::numeric_limits<T>::max());
}

TYPED_TEST(HipcubDeviceSegmentedReduce, ArgMax)
{
    using T = typename TestFixture::params::input_type;
    // Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMaxSelector<T>::type;
    test_argminmax<TestFixture, ArgMaxDispatch, HostOp>(test_utils::numeric_limits<T>::lowest());
}

template<class T>
class HipcubDeviceSegmentedReduceArgMinMaxSpecialTests : public testing::Test
{};

using HipcubDeviceSegmentedReduceArgMinMaxSpecialTestsParams
    = ::testing::Types<float, test_utils::half, test_utils::bfloat16>;
TYPED_TEST_SUITE(HipcubDeviceSegmentedReduceArgMinMaxSpecialTests,
                 HipcubDeviceSegmentedReduceArgMinMaxSpecialTestsParams);

template<typename TypeParam, typename DispatchFunction>
void test_argminmax_allinf(TypeParam value, TypeParam empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = TypeParam;
    using Iterator    = typename hipcub::ArgIndexInputIterator<input_type*, int>;
    using key_value   = typename Iterator::value_type;
    using offset_type = unsigned int;

    hipStream_t                stream = 0; // default
    DispatchFunction           function;
    std::random_device         rd;
    std::default_random_engine gen(rd());
    // include empty segments
    std::uniform_int_distribution<size_t> segment_length_dis(0, 1000);
    const size_t                          size = 100'000;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data and calculate expected results
        std::vector<input_type> values_input(size, value);

        std::vector<offset_type> offsets;
        unsigned int             segments_count = 0;
        size_t                   offset         = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);
            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        input_type* d_values_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(input_type),
                            hipMemcpyHostToDevice));

        offset_type* d_offsets;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets,
                                                     (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

        key_value* d_aggregates_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                     segments_count * sizeof(key_value)));

        size_t temporary_storage_bytes{};
        void*  d_temporary_storage{};

        HIP_CHECK(function(nullptr,
                           temporary_storage_bytes,
                           d_values_input,
                           d_aggregates_output,
                           segments_count,
                           d_offsets,
                           d_offsets + 1,
                           stream));

        // temp_storage_size_bytes must be > 0
        ASSERT_GT(temporary_storage_bytes, 0U);

        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(function(d_temporary_storage,
                           temporary_storage_bytes,
                           d_values_input,
                           d_aggregates_output,
                           segments_count,
                           d_offsets,
                           d_offsets + 1,
                           stream));

        std::vector<key_value> aggregates_output(segments_count);
        HIP_CHECK(hipMemcpy(aggregates_output.data(),
                            d_aggregates_output,
                            segments_count * sizeof(key_value),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_aggregates_output));
        HIP_CHECK(hipFree(d_temporary_storage));

        for(size_t i = 0; i < segments_count; i++)
        {
            if(offsets[i] < offsets[i + 1])
            {
                // all +/- infinity should produce +/- infinity
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output[i].key, 0));
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output[i].value, value));
            }
            else
            {
                // empty input should produce a special value
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output[i].key, 1));
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(aggregates_output[i].value, empty_value));
            }
        }
    }
}

// TODO: enable for NVIDIA platform once CUB backend incorporates fix
// Tests failed with CUB_VERSION = 200500 CUB::DeviceSegmentedReduce::Max(Min)
#ifdef __HIP_PLATFORM_AMD__
// ArgMin with all +Inf should result in +Inf.
TYPED_TEST(HipcubDeviceSegmentedReduceArgMinMaxSpecialTests, ReduceArgMinInf)
{
    test_argminmax_allinf<TypeParam, ArgMinDispatch>(
        test_utils::numeric_limits<TypeParam>::infinity(),
        test_utils::numeric_limits<TypeParam>::max());
}

// ArgMax with all -Inf should result in -Inf.
TYPED_TEST(HipcubDeviceSegmentedReduceArgMinMaxSpecialTests, ReduceArgMaxInf)
{
    test_argminmax_allinf<TypeParam, ArgMaxDispatch>(
        test_utils::numeric_limits<TypeParam>::infinity_neg(),
        test_utils::numeric_limits<TypeParam>::lowest());
}
#endif // __HIP_PLATFORM_AMD__

// ---------------------------------------------------------
// Test for large indices
// ---------------------------------------------------------

TEST(HipcubDeviceSegmentedReduceLargeIndicesTests, LargeIndices)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T              = size_t;
    using input_type     = T;
    using output_type    = T;
    using IteratorType   = rocprim::counting_iterator<input_type>;
    using reduce_op_type = typename hipcub::Sum;
    using offset_type    = T;

    const input_type init = input_type(0);
    reduce_op_type   reduce_op;

    static constexpr hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = test_utils::get_large_sizes(seed_value);

        for(const auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            const size_t min_segment_length = size_t{1} << 31;
            const size_t max_segment_length = std::max(min_segment_length, size);

            std::default_random_engine            gen(seed_value);
            std::uniform_int_distribution<size_t> segment_length_dis(min_segment_length,
                                                                     max_segment_length);

            const auto gauss_sum = [&](output_type n)
            { return (n % 2 == 0) ? (n / 2) * (n - 1) : n * ((n - 1) / 2); };

            std::vector<output_type> aggregates_expected;
            std::vector<offset_type> offsets;

            unsigned int num_segments = 0;
            offset_type  offset       = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const offset_type end       = std::min(size, offset + segment_length);
                output_type       aggregate = init;
                aggregate = reduce_op(aggregate, gauss_sum(end) - gauss_sum(offset));
                aggregates_expected.push_back(aggregate);

                num_segments++;
                offset += segment_length;
            }
            offsets.push_back(size);

            // Device inputs
            IteratorType d_input(input_type{0});
            offset_type* d_offsets = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets,
                                                         sizeof(offset_type) * (num_segments + 1)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                sizeof(offset_type) * (num_segments + 1),
                                hipMemcpyHostToDevice));

            // Device outputs
            output_type* d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output,
                                                         sizeof(output_type) * num_segments));
            HIP_CHECK(hipDeviceSynchronize());

            // Temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;

            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DeviceSegmentedReduce::Sum(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_input,
                                                         d_aggregates_output,
                                                         num_segments,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // Allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            // Run
            HIP_CHECK(hipcub::DeviceSegmentedReduce::Sum(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_input,
                                                         d_aggregates_output,
                                                         num_segments,
                                                         d_offsets,
                                                         d_offsets + 1,
                                                         stream));

            // Copy output to host
            std::vector<output_type> aggregates_output(num_segments);
            HIP_CHECK(hipMemcpy(aggregates_output.data(),
                                d_aggregates_output,
                                sizeof(output_type) * num_segments,
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_aggregates_output));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output, aggregates_expected));
        }
    }
}
