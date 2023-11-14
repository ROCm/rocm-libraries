// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_utils_thread_operators.hpp"

// hipcub API
#include "hipcub/device/device_segmented_reduce.hpp"
#include "test_utils_data_generation.hpp"

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        100000,
        (1 << 16) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 1000000, rand());
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
    using input_type                                 = Input;
    using output_type                                = Output;
    using reduce_op_type                             = ReduceOp;
    static constexpr int          init               = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedReduceOp : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<params1<unsigned int, unsigned int, hipcub::Sum>,
                         params1<int, int, hipcub::Sum, -100, 0, 10000>,
                         params1<double, double, hipcub::Min, 1000, 0, 10000>,
                         params1<int, short, hipcub::Max, 10, 1000, 10000>,
                         params1<short, double, hipcub::Sum, 5, 1, 1000>,
                         params1<float, double, hipcub::Max, 50, 2, 10>,
                         params1<float, float, hipcub::Sum, 123, 100, 200>
#ifdef __HIP_PLATFORM_AMD__
                         ,
                         params1<test_utils::half, test_utils::half, hipcub::Max, 50, 2, 10>,
                         params1<test_utils::bfloat16, test_utils::bfloat16, hipcub::Max, 50, 2, 10>
#endif
                         >
    Params1;

TYPED_TEST_SUITE(HipcubDeviceSegmentedReduceOp, Params1);

TYPED_TEST(HipcubDeviceSegmentedReduceOp, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;

    using result_type = output_type;
    using offset_type = unsigned int;

    const input_type init = test_utils::convert_to_device<input_type>(TestFixture::params::init);
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
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input = test_utils::get_random_data<input_type>(
                size,
                0,
                100,
                seed_value
            );

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
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_values_input, values_input.data(),
                    size * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );

            offset_type * d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );

            output_type * d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output, segments_count * sizeof(output_type)));

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
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

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
                    ASSERT_EQ(test_utils::convert_to_native(aggregates_output[i]),
                              test_utils::convert_to_native(aggregates_expected[i]));
                }
                else
                {
                    auto diff = std::max<test_utils::convert_to_fundamental_t<output_type>>(
                        std::abs(0.01 * test_utils::convert_to_fundamental(aggregates_expected[i])),
                        output_type(0.01));
                    ASSERT_NEAR(test_utils::convert_to_native(aggregates_output[i]),
                                test_utils::convert_to_native(aggregates_expected[i]),
                                diff);
                }
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

typedef ::testing::Types<params2<unsigned int, unsigned int>,
                         params2<int, int, 0, 10000>,
                         params2<double, double, 0, 10000>,
                         params2<int, long long, 1000, 10000>,
                         params2<float, double, 2, 10>,
                         params2<float, float, 100, 200>
#ifdef __HIP_PLATFORM_AMD__
                         ,
                         params2<test_utils::half, float>, // Doesn't work on NVIDIA / CUB
                         params2<test_utils::bfloat16, float> // Doesn't work on NVIDIA / CUB
#endif
                         >
    Params2;

TYPED_TEST_SUITE(HipcubDeviceSegmentedReduce, Params2);

TYPED_TEST(HipcubDeviceSegmentedReduce, Sum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using reduce_op_type = typename hipcub::Sum;
    using result_type = output_type;
    using offset_type = unsigned int;

    const input_type init              = input_type(0);
    const bool       debug_synchronous = false;
    reduce_op_type   reduce_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input = test_utils::get_random_data<input_type>(
                size,
                0,
                100,
                seed_value
            );

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
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_values_input, values_input.data(),
                    size * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );

            offset_type * d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );

            output_type * d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output, segments_count * sizeof(output_type)));

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
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

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
                    ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(aggregates_output[i],
                                                                    aggregates_expected[i],
                                                                    diff));
                }
            }
        }
    }
}

TYPED_TEST(HipcubDeviceSegmentedReduce, Min)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

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
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input = test_utils::get_random_data<input_type>(
                size,
                0,
                100,
                seed_value
            );

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
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_values_input, values_input.data(),
                    size * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );

            offset_type * d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );

            output_type * d_aggregates_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_aggregates_output, segments_count * sizeof(output_type)));

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
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

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
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_near(aggregates_output[i], aggregates_expected[i], .01f));
            }
        }
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
                    hipStream_t     stream,
                    bool            debug_synchronous) const
    {
        return hipcub::DeviceSegmentedReduce::ArgMin(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_in,
                                                     d_out,
                                                     num_segments,
                                                     d_begin_offsets,
                                                     d_end_offsets,
                                                     stream,
                                                     debug_synchronous);
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
                    hipStream_t     stream,
                    bool            debug_synchronous) const
    {
        return hipcub::DeviceSegmentedReduce::ArgMax(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_in,
                                                     d_out,
                                                     num_segments,
                                                     d_begin_offsets,
                                                     d_end_offsets,
                                                     stream,
                                                     debug_synchronous);
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
    const bool                            debug_synchronous = false;
    std::random_device                    rd;
    std::default_random_engine            gen(rd());
    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data and calculate expected results
            std::vector<key_value> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            HostOp                   host_op{};
            std::vector<offset_type> offsets;
            unsigned int             segments_count = 0;
            size_t                   offset         = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);
                Iterator x(&values_input[offset]);

                const size_t end = std::min(size, offset + segment_length);
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
                               stream,
                               debug_synchronous));

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
                               stream,
                               debug_synchronous));
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
                                                                .01f));
            }
        }
    }
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
class HipcubDeviceReduceArgMinMaxSpecialTests : public testing::Test
{};

using HipcubDeviceReduceArgMinMaxSpecialTestsParams
    = ::testing::Types<float, test_utils::half, test_utils::bfloat16>;
TYPED_TEST_SUITE(HipcubDeviceReduceArgMinMaxSpecialTests,
                 HipcubDeviceReduceArgMinMaxSpecialTestsParams);

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

    constexpr bool             debug_synchronous = false;
    hipStream_t                stream            = 0; // default
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
                           stream,
                           debug_synchronous));

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
                           stream,
                           debug_synchronous));

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
#ifdef __HIP_PLATFORM_AMD__
/// ArgMin with all +Inf should result in +Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMinInf)
{
    test_argminmax_allinf<TypeParam, ArgMinDispatch>(
        test_utils::numeric_limits<TypeParam>::infinity(),
        test_utils::numeric_limits<TypeParam>::max());
}

/// ArgMax with all -Inf should result in -Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMaxInf)
{
    test_argminmax_allinf<TypeParam, ArgMaxDispatch>(
        test_utils::numeric_limits<TypeParam>::infinity_neg(),
        test_utils::numeric_limits<TypeParam>::lowest());
}
#endif // __HIP_PLATFORM_AMD__
