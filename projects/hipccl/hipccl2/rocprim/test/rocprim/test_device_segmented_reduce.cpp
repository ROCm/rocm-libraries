// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/device_segmented_reduce.hpp>

// required test headers
#include "test_utils_types.hpp"

template<
    class Input,
    class Output,
    class ReduceOp = ::rocprim::plus<Input>,
    int Init = 0, // as only integral types supported, int is used here even for floating point inputs
    unsigned int MinSegmentLength = 0,
    unsigned int MaxSegmentLength = 1000,
    // Tests output iterator with void value_type (OutputIterator concept)
    bool UseIdentityIterator = false
>
struct params
{
    using input_type = Input;
    using output_type = Output;
    using reduce_op_type = ReduceOp;
    static constexpr int init = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceSegmentedReduce : public ::testing::Test {
public:
    using params = Params;
};

using custom_short2 = test_utils::custom_test_type<short>;
using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;
using half           = rocprim::half;
using bfloat16       = rocprim::bfloat16;

typedef ::testing::Types<
    params<unsigned char, unsigned int, rocprim::plus<unsigned int>>,
    params<int, int, rocprim::plus<int>, -100, 0, 10000>,
    params<double, double, rocprim::minimum<double>, 1000, 0, 10000>,
    params<int8_t, int8_t, rocprim::maximum<int8_t>, 0, 0, 2000>,
    params<uint8_t, uint8_t, rocprim::plus<uint8_t>, 10, 1000, 10000>,
    params<uint8_t, uint8_t, rocprim::maximum<uint8_t>, 50, 2, 10>,
    params<half, half, rocprim::maximum<half>, 0, 1000, 2000>,
    params<half, half, rocprim::plus<half>, 50, 2, 10>,
    params<bfloat16, bfloat16, rocprim::maximum<bfloat16>, 0, 1000, 2000>,
    params<bfloat16, bfloat16, rocprim::plus<bfloat16>, 50, 2, 10>,
    params<custom_short2, custom_int2, rocprim::plus<custom_int2>, 10, 1000, 10000>,
    params<custom_double2, custom_double2, rocprim::maximum<custom_double2>, 50, 2, 10>,
    params<float, float, rocprim::plus<float>, 123, 100, 200>,
    params<unsigned char, long long, rocprim::plus<int>, 10, 3000, 4000>,
    params<half, float, rocprim::plus<float>, 0, 10, 300>,
    params<bfloat16, float, rocprim::plus<float>, 0, 10, 300>,
    params<half, half, rocprim::minimum<half>, 0, 1000, 30000>,
    params<bfloat16, bfloat16, rocprim::minimum<bfloat16>, 0, 1000, 30000>>
    Params;

TYPED_TEST_SUITE(RocprimDeviceSegmentedReduce, Params);

TYPED_TEST(RocprimDeviceSegmentedReduce, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::params::input_type;
    using output_type    = typename TestFixture::params::output_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;
    using offset_type    = unsigned int;

    reduce_op_type reduce_op;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;

    const input_type init              = input_type{TestFixture::params::init};
    const bool       debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int segments_count = 0;
            size_t offset = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                output_type aggregate = init;
                for(size_t i = offset; i < end; i++)
                {
                    aggregate = reduce_op(aggregate, values_input[i]);
                }
                aggregates_expected.push_back(aggregate);

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            // intermediate results for segmented reduce are stored as output_type,
            // but reduced by the reduce_op_type operation,
            // however that opeartion uses the same output_type for all tests
            const float precision = test_utils::is_plus_operator<reduce_op_type>::value
                                        ? test_utils::precision<output_type> * max_segment_length
                                        : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

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
                rocprim::segmented_reduce(
                    nullptr, temporary_storage_bytes,
                    d_values_input, d_aggregates_output,
                    segments_count,
                    d_offsets, d_offsets + 1,
                    reduce_op, init,
                    stream, debug_synchronous
                )
            );

            ASSERT_GT(temporary_storage_bytes, 0);

            void * d_temporary_storage;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            HIP_CHECK(
                rocprim::segmented_reduce(
                    d_temporary_storage, temporary_storage_bytes,
                    d_values_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_aggregates_output),
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

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(aggregates_output, aggregates_expected, precision));
        }
    }

}
