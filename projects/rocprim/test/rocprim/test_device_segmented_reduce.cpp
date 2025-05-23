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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "identity_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/block/block_reduce.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_segmented_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdint.h>
#include <vector>

using bra = ::rocprim::block_reduce_algorithm;

template<class Input,
         class Output,
         class ReduceOp                = ::rocprim::plus<Input>,
         int          Init             = 0,
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000,
         // Tests output iterator with void value_type (OutputIterator concept)
         bool UseIdentityIterator = false,
         bra  Algo                = bra::default_algorithm,
         bool UseDefaultConfig    = false,
         bool UseGraphs           = false>
struct SegmentedReduceParams
{
    using input_type                                    = Input;
    using output_type                                   = Output;
    using reduce_op_type                                = ReduceOp;
    static constexpr int          init                  = Init;
    static constexpr unsigned int min_segment_length    = MinSegmentLength;
    static constexpr unsigned int max_segment_length    = MaxSegmentLength;
    static constexpr bool         use_identity_iterator = UseIdentityIterator;
    static constexpr bra          algo                  = Algo;
    static constexpr bool         use_default_config    = UseDefaultConfig;
    static constexpr bool         use_graphs            = UseGraphs;
};

// clang-format off
#define SegmentedReduceParamsList(...)                                       \
    SegmentedReduceParams<__VA_ARGS__, bra::using_warp_reduce>,              \
    SegmentedReduceParams<__VA_ARGS__, bra::raking_reduce>,                  \
    SegmentedReduceParams<__VA_ARGS__, bra::raking_reduce_commutative_only>, \
    SegmentedReduceParams<__VA_ARGS__, bra::default_algorithm, true>
// clang-format on

template<bra Algo, bool UseDefaultConfig = false>
struct algo_config
{
    using type = rocprim::reduce_config<128, 8, Algo>;
};

template<>
struct algo_config<bra::default_algorithm, true>
{
    using type = rocprim::default_config;
};

template<bra Algo, bool UseDefaultConfig>
using algo_config_t = typename algo_config<Algo, UseDefaultConfig>::type;

template<class Params>
class RocprimDeviceSegmentedReduce : public ::testing::Test
{
public:
    using params = Params;
};

using custom_short2  = common::custom_type<short, short, true>;
using custom_int2    = common::custom_type<int, int, true>;
using custom_double2 = common::custom_type<double, double, true>;
using half           = rocprim::half;
using bfloat16       = rocprim::bfloat16;

#define plus rocprim::plus
#define maximum rocprim::maximum
#define minimum rocprim::minimum

using Params = ::testing::Types<
    // Integer types
    SegmentedReduceParamsList(int, int, plus<int>, -100, 0, 10000, false),
    SegmentedReduceParamsList(int8_t, int8_t, maximum<int8_t>, 0, 0, 2000, false),
    SegmentedReduceParamsList(uint8_t, uint8_t, plus<uint8_t>, 10, 1000, 10000, true),
    SegmentedReduceParamsList(uint8_t, uint8_t, maximum<uint8_t>, 50, 2, 10, false),
    SegmentedReduceParamsList(short, short, minimum<short>, -15, 1, 100, true),
    // Floating point types
    SegmentedReduceParamsList(double, double, minimum<double>, 1000, 0, 10000, false),
    SegmentedReduceParamsList(float, float, plus<float>, 123, 100, 200, false),
    SegmentedReduceParamsList(half, half, plus<half>, 50, 2, 10, false),
    SegmentedReduceParamsList(half, half, maximum<half>, 0, 1000, 2000, true),
    SegmentedReduceParamsList(half, half, minimum<half>, 0, 1000, 30000, false),
    SegmentedReduceParamsList(bfloat16, bfloat16, plus<bfloat16>, 50, 2, 10, true),
    SegmentedReduceParamsList(bfloat16, bfloat16, maximum<bfloat16>, 0, 1000, 2000, false),
    SegmentedReduceParamsList(bfloat16, bfloat16, minimum<bfloat16>, 0, 1000, 30000, false),
    // Custom types
    SegmentedReduceParamsList(
        custom_short2, custom_int2, plus<custom_int2>, 10, 1000, 10000, false),
    SegmentedReduceParamsList(
        custom_double2, custom_double2, maximum<custom_double2>, 50, 2, 10, false),
    // Types conversion
    SegmentedReduceParamsList(unsigned char, unsigned int, plus<unsigned int>, 0, 0, 1000, false),
    SegmentedReduceParamsList(unsigned char, long long, plus<int>, 10, 3000, 4000, true),
    SegmentedReduceParamsList(half, float, plus<float>, 0, 10, 300, false),
    SegmentedReduceParamsList(bfloat16, float, plus<double>, 0, 10, 300, false),
    // Test with graphs
    SegmentedReduceParams<int,
                          int,
                          plus<int>,
                          0,
                          0,
                          1000,
                          false,
                          bra::default_algorithm,
                          false,
                          true>>;

#undef plus
#undef maximum
#undef minimum

TYPED_TEST_SUITE(RocprimDeviceSegmentedReduce, Params);

TYPED_TEST(RocprimDeviceSegmentedReduce, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Config
        = algo_config_t<TestFixture::params::algo, TestFixture::params::use_default_config>;

    using input_type     = typename TestFixture::params::input_type;
    using output_type    = typename TestFixture::params::output_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;
    using offset_type    = unsigned int;

    reduce_op_type reduce_op;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;

    const input_type init              = input_type{TestFixture::params::init};
    const bool       debug_synchronous = false;

    std::random_device                       rd;
    const size_t                             seed = rd();
    std::default_random_engine               gen(seed);
    common::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default
            if(TestFixture::params::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            // Generate data and calculate expected results
            std::vector<output_type> aggregates_expected;

            std::vector<input_type> values_input
                = test_utils::get_random_data_wrapped<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            std::vector<size_t>      sizes;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                sizes.push_back(segment_length);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
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

            common::device_ptr<input_type>  d_values_input(values_input);
            common::device_ptr<offset_type> d_offsets(offsets);
            common::device_ptr<output_type> d_aggregates_output(segments_count);

            size_t temp_storage_bytes;

            HIP_CHECK(rocprim::segmented_reduce<Config>(nullptr,
                                                        temp_storage_bytes,
                                                        d_values_input.get(),
                                                        d_aggregates_output.get(),
                                                        segments_count,
                                                        d_offsets.get(),
                                                        d_offsets.get() + 1,
                                                        reduce_op,
                                                        init,
                                                        stream,
                                                        debug_synchronous));

            ASSERT_GT(temp_storage_bytes, 0);

            common::device_ptr<void> d_temp_storage(temp_storage_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::segmented_reduce<Config>(
                d_temp_storage.get(),
                temp_storage_bytes,
                d_values_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_aggregates_output.get()),
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                reduce_op,
                init,
                stream,
                debug_synchronous));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            const auto aggregates_output = d_aggregates_output.load();

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
            SCOPED_TRACE(testing::Message() << "with seed = " << seed);

            if(size > 0)
            {
                const float single_op_precision = precision / max_segment_length;

                for(size_t i = 0; i < aggregates_output.size(); ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(aggregates_output[i],
                                                aggregates_expected[i],
                                                single_op_precision * (sizes[i] - 1)));
                }
            }
        }
    }
}

template<bool use_graphs = false>
void testLargeIndices()
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T              = std::size_t;
    using Iterator       = rocprim::counting_iterator<T>;
    using reduce_op_type = rocprim::plus<T>;

    const reduce_op_type reduce_op{};
    const T              init{0};
    const bool           debug_synchronous = false;

    hipStream_t stream = 0; // default
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(auto size : test_utils::get_large_sizes(42))
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data and calculate expected results
        const T large_segment_size = size_t{1} << 31;
        const T min_segment_length
            = size < large_segment_size
                  ? (size_t{1} << 30) - 1 /*smallest size in get_large_sizes()*/
                  : large_segment_size;
        const T max_segment_length = size;

        std::random_device                       rd;
        const size_t                             seed = rd();
        std::default_random_engine               gen(seed);
        common::uniform_int_distribution<size_t> segment_length_dis(min_segment_length,
                                                                    max_segment_length);

        const auto gauss_sum
            = [&](T n) { return (n % 2 == 0) ? (n / 2) * (n - 1) : n * ((n - 1) / 2); };

        std::vector<T> aggregates_expected;
        std::vector<T> offsets;

        int    segments_count = 0;
        size_t offset         = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const T end       = std::min(size, offset + segment_length);
            T       aggregate = reduce_op(init, gauss_sum(end) - gauss_sum(offset));
            aggregates_expected.push_back(aggregate);

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        // Device inputs
        const Iterator            values_input{0};
        common::device_ptr<T>     d_offsets(offsets);

        // Device outputs
        common::device_ptr<T> d_aggregates_output(segments_count);

        // temp storage
        size_t temp_storage_size_bytes = 0;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::segmented_reduce(nullptr,
                                            temp_storage_size_bytes,
                                            values_input,
                                            d_aggregates_output.get(),
                                            segments_count,
                                            d_offsets.get(),
                                            d_offsets.get() + 1,
                                            reduce_op,
                                            init,
                                            stream,
                                            debug_synchronous));

        // Allocate temporary storage
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        test_utils::GraphHelper gHelper;
        if(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK(rocprim::segmented_reduce(d_temp_storage.get(),
                                            temp_storage_size_bytes,
                                            values_input,
                                            d_aggregates_output.get(),
                                            segments_count,
                                            d_offsets.get(),
                                            d_offsets.get() + 1,
                                            reduce_op,
                                            init,
                                            stream,
                                            debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream, true, false);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        const auto aggregates_output = d_aggregates_output.load();
        HIP_CHECK(hipDeviceSynchronize());

        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output, aggregates_expected));

        if(use_graphs)
        {
            gHelper.cleanupGraphHelper();
        }
    }

    if(use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceSegmentedReduce, LargeIndices)
{
    testLargeIndices<>();
}

TEST(RocprimDeviceSegmentedReduce, LargeIndicesWithGraphs)
{
    testLargeIndices<true>();
}
