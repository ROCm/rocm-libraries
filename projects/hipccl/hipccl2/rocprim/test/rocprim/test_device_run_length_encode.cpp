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
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_run_length_encode.hpp>
#include <rocprim/device/device_run_length_encode_config.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdint.h>
#include <vector>

template<class Key,
         class Count,
         unsigned int MinSegmentLength,
         unsigned int MaxSegmentLength,
         // Tests output iterator with void value_type (OutputIterator concept)
         bool UseIdentityIterator = false,
         class Config             = rocprim::default_config,
         class NonTrivialConfig   = rocprim::default_config>
struct params
{
    using key_type = Key;
    using count_type = Count;
    using config                                        = Config;
    using non_trivial_config                            = NonTrivialConfig;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceRunLengthEncode : public ::testing::Test {
public:
    using params = Params;
};

using custom_int2    = common::custom_type<int, int, true>;
using custom_double2 = common::custom_type<double, double, true>;

using Params = ::testing::Types<
    // Tests with default configuration
    params<int8_t, int8_t, 100, 2000>,
    params<uint8_t, uint8_t, 100, 2000>,
    params<int8_t, int8_t, 1000, 5000>,
    params<uint8_t, uint8_t, 1000, 5000>,
    params<int, unsigned int, 1000, 5000>,
    params<unsigned int, size_t, 2048, 2048>,
    params<unsigned int, unsigned int, 1000, 50000>,
    params<unsigned long long, size_t, 1, 30>,
    params<float, unsigned long long, 100, 400>,
    params<float, int, 1, 10>,
    params<double, int, 3, 5>,
    params<double, int, 100, 2000>,
    params<int, rocprim::half, 100, 2000>,
    // half should be supported, but is missing some key operators.
    // we should uncomment these, as soon as these are implemented and the tests compile and work as intended.
    //params<rocprim::half, int, 100, 2000>,
    params<int, rocprim::bfloat16, 1000, 5000>,
    params<rocprim::bfloat16, int, 1000, 5000>,
    // Tests for custom types
    params<custom_int2, unsigned int, 20, 100>,
    params<custom_double2, custom_int2, 10, 30000, true>,
    params<unsigned long long, custom_double2, 100000, 100000>,
    // Tests for supported config structs
    params<unsigned int,
           unsigned int,
           200,
           600,
           false,
           // RLE config
           rocprim::run_length_encode_config<rocprim::reduce_by_key_config<128, 5>,
                                             rocprim::select_config<64, 3>>,
           // RLE non-trivial config
           rocprim::run_length_encode_config<rocprim::reduce_by_key_config<256, 15>,
                                             rocprim::select_config<256, 13>>>,
    // Tests for when output's value_type is void
    params<int, int, 1, 1, true>>;

TYPED_TEST_SUITE(RocprimDeviceRunLengthEncode, Params);

template <class T>
T get_random_value_no_duplicate(const T duplicate, const std::vector<T> &source, const size_t start_index)
{
    T val;
    size_t i = 0;
    do
    {
        val = source[(start_index+i) % source.size()];
        i++;
    } while (val == duplicate && i < source.size());
    return val;
}

TYPED_TEST(RocprimDeviceRunLengthEncode, Encode)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    using count_type = typename TestFixture::params::count_type;
    using config     = typename TestFixture::params::config;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    const bool debug_synchronous = false;

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);
    std::vector<key_type>      random_keys
        = test_utils::get_random_data_wrapped<key_type>(64, -100, 100, seed);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Default stream does not support hipGraph stream capture, so create one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

            // Generate data and calculate expected results
            std::vector<key_type> unique_expected;
            std::vector<count_type> counts_expected;
            size_t runs_count_expected = 0;

            std::vector<key_type>                    input(size);
            common::uniform_int_distribution<size_t> key_count_dis(
                TestFixture::params::min_segment_length,
                TestFixture::params::max_segment_length);
            std::vector<count_type> values_input
                = test_utils::get_random_data_wrapped<count_type>(size, 0, 100, seed_value);

            size_t offset = 0;
            key_type current_key = get_random_value_no_duplicate(key_type(0), random_keys, size);
            while(offset < size)
            {
                size_t key_count = key_count_dis(gen);
                const size_t end = std::min(size, offset + key_count);

                current_key = get_random_value_no_duplicate(current_key, random_keys, end);

                key_count = end - offset;
                for(size_t i = offset; i < end; i++)
                {
                    input[i] = current_key;
                }

                unique_expected.push_back(current_key);
                runs_count_expected++;
                counts_expected.push_back(static_cast<count_type>(key_count));

                offset += key_count;
            }

            common::device_ptr<key_type> d_input(input);

            common::device_ptr<key_type>   d_unique_output(runs_count_expected);
            common::device_ptr<count_type> d_counts_output(runs_count_expected);
            common::device_ptr<count_type> d_runs_count_output(1);

            size_t temporary_storage_bytes = 0;

            HIP_CHECK(rocprim::run_length_encode<config>(
                nullptr,
                temporary_storage_bytes,
                d_input.get(),
                size,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unique_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_counts_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_runs_count_output.get()),
                stream,
                debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            gHelper.startStreamCapture(stream);

            HIP_CHECK(rocprim::run_length_encode<config>(
                d_temporary_storage.get(),
                temporary_storage_bytes,
                d_input.get(),
                size,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unique_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_counts_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_runs_count_output.get()),
                stream,
                debug_synchronous));

            gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            const auto unique_output     = d_unique_output.load();
            const auto counts_output     = d_counts_output.load();
            const auto runs_count_output = d_runs_count_output.load();

            gHelper.cleanupGraphHelper();
            HIP_CHECK(hipStreamDestroy(stream));

            // Validating results

            std::vector<count_type> runs_count_expected_2;
            runs_count_expected_2.push_back(static_cast<count_type>(runs_count_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(runs_count_output, runs_count_expected_2, 1));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(unique_output, unique_expected, runs_count_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(counts_output, counts_expected, runs_count_expected));
        }
    }

}

TYPED_TEST(RocprimDeviceRunLengthEncode, NonTrivialRuns)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    using count_type = typename TestFixture::params::count_type;
    using offset_type = typename TestFixture::params::count_type;
    using config      = typename TestFixture::params::non_trivial_config;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;

    const bool debug_synchronous = false;

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);
    std::vector<key_type>      random_keys
        = test_utils::get_random_data_wrapped<key_type>(64, -100, 100, seed);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Default stream does not support hipGraph stream capture, so create one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

            // Generate data and calculate expected results
            std::vector<offset_type> offsets_expected;
            std::vector<count_type> counts_expected;
            size_t runs_count_expected = 0;

            std::vector<key_type>                    input(size);
            common::uniform_int_distribution<size_t> key_count_dis(
                TestFixture::params::min_segment_length,
                TestFixture::params::max_segment_length);
            std::bernoulli_distribution is_trivial_dis(0.1);

            size_t offset = 0;
            key_type current_key = get_random_value_no_duplicate(key_type(0), random_keys, size);
            while(offset < size)
            {
                size_t key_count;
                if(TestFixture::params::min_segment_length == 1 && is_trivial_dis(gen))
                {
                    // Increased probability of trivial runs for long segments
                    key_count = 1;
                }
                else
                {
                    key_count = key_count_dis(gen);
                }
                const size_t end = std::min(size, offset + key_count);

                current_key = get_random_value_no_duplicate(current_key, random_keys, end);

                key_count = end - offset;
                for(size_t i = offset; i < end; i++)
                {
                    input[i] = current_key;
                }

                if(key_count > 1)
                {
                    offsets_expected.push_back(static_cast<offset_type>(offset));
                    runs_count_expected++;
                    counts_expected.push_back(static_cast<count_type>(key_count));
                }

                offset += key_count;
            }

            common::device_ptr<key_type> d_input(input);

            common::device_ptr<offset_type> d_offsets_output(
                std::max<size_t>(1, runs_count_expected));
            common::device_ptr<count_type> d_counts_output(
                std::max<size_t>(1, runs_count_expected));
            common::device_ptr<count_type> d_runs_count_output(1);

            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::run_length_encode_non_trivial_runs<config>(
                nullptr,
                temporary_storage_bytes,
                d_input.get(),
                size,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_offsets_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_counts_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_runs_count_output.get()),
                stream,
                debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            gHelper.startStreamCapture(stream);

            HIP_CHECK(rocprim::run_length_encode_non_trivial_runs<config>(
                d_temporary_storage.get(),
                temporary_storage_bytes,
                d_input.get(),
                size,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_offsets_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_counts_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_runs_count_output.get()),
                stream,
                debug_synchronous));

            gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            std::vector<offset_type> offsets_output;
            std::vector<count_type>  counts_output;
            const auto               runs_count_output = d_runs_count_output.load();

            if(runs_count_expected > 0)
            {
                offsets_output = d_offsets_output.load();
                counts_output  = d_counts_output.load();
            }

            gHelper.cleanupGraphHelper();
            HIP_CHECK(hipStreamDestroy(stream));

            // Validating results

            std::vector<count_type> runs_count_expected_2;
            runs_count_expected_2.push_back(static_cast<count_type>(runs_count_expected));
            SCOPED_TRACE(testing::Message() << "runs_count_output");
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_eq(runs_count_output, runs_count_expected_2, 1));
            SCOPED_TRACE(testing::Message() << "offsets_output");
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_eq(offsets_output, offsets_expected, runs_count_expected));
            SCOPED_TRACE(testing::Message() << "counts_output");
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_eq(counts_output, counts_expected, runs_count_expected));
        }
    }
}
