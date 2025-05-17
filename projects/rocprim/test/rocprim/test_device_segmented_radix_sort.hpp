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

#ifndef TEST_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define TEST_DEVICE_SEGMENTED_RADIX_SORT_HPP_

#include "../common_test_header.hpp"

#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

// required rocprim headers
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types/double_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

template<class Key,
         class Value,
         bool         Descending,
         unsigned int StartBit,
         unsigned int EndBit,
         unsigned int MinSegmentLength,
         unsigned int MaxSegmentLength,
         class Config = rocprim::default_config>
struct params
{
    using key_type                                   = Key;
    using value_type                                 = Value;
    static constexpr bool         descending         = Descending;
    static constexpr unsigned int start_bit          = StartBit;
    static constexpr unsigned int end_bit            = EndBit;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    using config                                     = Config;
};

using config_default
    = rocprim::segmented_radix_sort_config<4, //< radix bits
                                           rocprim::kernel_config<256, //< sort block size,
                                                                  4>>; //< items per thread

using config_semi_custom
    = rocprim::segmented_radix_sort_config<3, //< radix bits
                                           rocprim::kernel_config<128, //< sort block size
                                                                  4>, //< items per thread
                                           rocprim::WarpSortConfig<16, //< logical warp size small
                                                                   8>, //< items per thread small
                                           false>; //< enable unpartitioned sort

using config_semi_custom_warp_config
    = rocprim::segmented_radix_sort_config<3, //< radix bits
                                           rocprim::kernel_config<128, //< sort block size
                                                                  4>, //< items per thread
                                           rocprim::WarpSortConfig<16, //< logical warp size small
                                                                   2, //< items per thread small
                                                                   512, //< block size small
                                                                   0>, //< partitioning threshold
                                           true>; //< enable unpartitioned sort

using config_custom
    = rocprim::segmented_radix_sort_config<3, //< radix bits
                                           rocprim::kernel_config<128, //< sort block size
                                                                  4>, //< items per thread
                                           rocprim::WarpSortConfig<16, //< logical warp size small
                                                                   2, //< items per thread small
                                                                   512, //< block size small
                                                                   0, //< partitioning threshold
                                                                   32, //< logical warp size medium
                                                                   4, //< items per thread medium
                                                                   256>, //< block size medium
                                           true>; //< enable unpartitioned sort

template<class Params>
class RocprimDeviceSegmentedRadixSort : public ::testing::Test
{
public:
    using params = Params;
};

TYPED_TEST_SUITE_P(RocprimDeviceSegmentedRadixSort);

template<typename TestFixture>
inline void sort_keys()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    using config                             = typename TestFixture::params::config;
    static constexpr bool         descending = TestFixture::params::descending;
    static constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    static constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device         rd;
    std::default_random_engine gen(rd());

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

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

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

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output(size);

            common::device_ptr<offset_type> d_offsets(offsets);

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(
                    expected.begin() + offsets[i],
                    expected.begin() + offsets[i + 1],
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>());
            }

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys_input.get(),
                                                                 d_keys_output.get(),
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets.get(),
                                                                 d_offsets.get() + 1,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(rocprim::segmented_radix_sort_keys_desc<config>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_keys_input.get(),
                                                                          d_keys_output.get(),
                                                                          size,
                                                                          segments_count,
                                                                          d_offsets.get(),
                                                                          d_offsets.get() + 1,
                                                                          start_bit,
                                                                          end_bit,
                                                                          stream,
                                                                          debug_synchronous));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     d_keys_output.get(),
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets.get(),
                                                                     d_offsets.get() + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream,
                                                                     debug_synchronous));
            }

            const auto keys_output = d_keys_output.load();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
        }
    }
}

template<typename TestFixture>
inline void sort_keys_empty_data()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    using config                             = typename TestFixture::params::config;
    static constexpr bool         descending = TestFixture::params::descending;
    static constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    static constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const std::vector<size_t> sizes = {0, 1024};
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        const std::vector<size_t> segments_counts = {0, 1};
        for(size_t segments_count : segments_counts)
        {
            unsigned int seed_value = seeds[0];
            SCOPED_TRACE(testing::Message() << "with segments_count = " << segments_count);

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

            std::vector<offset_type> offsets(2);
            offsets[0] = 0;
            offsets[1] = 0;

            common::device_ptr<key_type> d_keys(keys_input);

            common::device_ptr<offset_type> d_offsets(offsets);

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys.get(),
                                                                 d_keys.get(),
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets.get(),
                                                                 d_offsets.get() + 1,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_pairs_desc<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys.get(),
                                                                     d_keys.get(),
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets.get(),
                                                                     d_offsets.get() + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys.get(),
                                                                      d_keys.get(),
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets.get(),
                                                                      d_offsets.get() + 1,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream));
            }

            const auto keys_output = d_keys.load();

            // Output should not have changed
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_input));
        }
    }
}

template<typename TestFixture>
inline void sort_keys_large_segments()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using config                      = typename TestFixture::params::config;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    size_t size           = 1 << 20;
    size_t segments_count = 2;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
            size,
            common::generate_limits<key_type>::min(),
            common::generate_limits<key_type>::max(),
            seed_value);

        std::vector<offset_type> offsets(3);
        offsets[0] = 0;
        offsets[1] = static_cast<offset_type>(size / 2);
        offsets[2] = static_cast<offset_type>(size);

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(size);

        common::device_ptr<offset_type> d_offsets(offsets);

        // Calculate expected results on host
        std::vector<key_type> expected(keys_input);
        for(size_t i = 0; i < segments_count; i++)
        {
            std::stable_sort(
                expected.begin() + offsets[i],
                expected.begin() + offsets[i + 1],
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>());
        }

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(nullptr,
                                                             temporary_storage_bytes,
                                                             d_keys_input.get(),
                                                             d_keys_output.get(),
                                                             size,
                                                             segments_count,
                                                             d_offsets.get(),
                                                             d_offsets.get() + 1,
                                                             start_bit,
                                                             end_bit));

        ASSERT_GT(temporary_storage_bytes, 0U);

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        if(descending)
        {
            HIP_CHECK(rocprim::segmented_radix_sort_keys_desc<config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys_input.get(),
                                                                      d_keys_output.get(),
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets.get(),
                                                                      d_offsets.get() + 1,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream));
        }
        else
        {
            HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(d_temporary_storage.get(),
                                                                 temporary_storage_bytes,
                                                                 d_keys_input.get(),
                                                                 d_keys_output.get(),
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets.get(),
                                                                 d_offsets.get() + 1,
                                                                 start_bit,
                                                                 end_bit,
                                                                 stream));
        }

        const auto keys_output = d_keys_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
    }
}

template<typename TestFixture>
inline void sort_keys_unspecified_ranges()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using config                      = typename TestFixture::params::config;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

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

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

            std::vector<offset_type> begin_offsets;
            unsigned int             segments_count = 0;
            size_t                   offset         = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                begin_offsets.push_back(offset);
                segments_count++;
                offset += segment_length;
            }
            begin_offsets.push_back(size);
            std::vector<offset_type> end_offsets(begin_offsets.cbegin() + 1, begin_offsets.cend());
            begin_offsets.pop_back();

            size_t            empty_segments = rocprim::max(segments_count / 16, 1u);
            std::vector<bool> is_empty_segment(segments_count, false);
            std::fill(is_empty_segment.begin(), is_empty_segment.begin() + empty_segments, true);
            std::shuffle(is_empty_segment.begin(), is_empty_segment.end(), gen);

            for(size_t i = 0; i < segments_count; i++)
            {
                if(is_empty_segment[i])
                {
                    begin_offsets[i] = 0;
                    end_offsets[i]   = 0;
                }
            }

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output(keys_input);

            common::device_ptr<offset_type> d_offsets_begin(begin_offsets);
            common::device_ptr<offset_type> d_offsets_end(end_offsets);

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(
                    expected.begin() + begin_offsets[i],
                    expected.begin() + end_offsets[i],
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>());
            }

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys_input.get(),
                                                                 d_keys_output.get(),
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets_begin.get(),
                                                                 d_offsets_end.get(),
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(rocprim::segmented_radix_sort_keys_desc<config>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_keys_input.get(),
                                                                          d_keys_output.get(),
                                                                          size,
                                                                          segments_count,
                                                                          d_offsets_begin.get(),
                                                                          d_offsets_end.get(),
                                                                          start_bit,
                                                                          end_bit,
                                                                          stream));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     d_keys_output.get(),
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets_begin.get(),
                                                                     d_offsets_end.get(),
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }

            const auto keys_output = d_keys_output.load();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
        }
    }
}

template<typename TestFixture>
inline void sort_pairs()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    using config                      = typename TestFixture::params::config;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device         rd;
    std::default_random_engine gen(rd());

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

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

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

            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output(size);

            common::device_ptr<value_type> d_values_input(values_input);
            common::device_ptr<value_type> d_values_output(size);

            common::device_ptr<offset_type> d_offsets(offsets);

            using key_value = std::pair<key_type, value_type>;

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(expected.begin() + offsets[i],
                                 expected.begin() + offsets[i + 1],
                                 test_utils::key_value_comparator<key_type,
                                                                  value_type,
                                                                  descending,
                                                                  start_bit,
                                                                  end_bit>());
            }
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input.get(),
                                                                  d_keys_output.get(),
                                                                  d_values_input.get(),
                                                                  d_values_output.get(),
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets.get(),
                                                                  d_offsets.get() + 1,
                                                                  start_bit,
                                                                  end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_pairs_desc<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     d_keys_output.get(),
                                                                     d_values_input.get(),
                                                                     d_values_output.get(),
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets.get(),
                                                                     d_offsets.get() + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream,
                                                                     debug_synchronous));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys_input.get(),
                                                                      d_keys_output.get(),
                                                                      d_values_input.get(),
                                                                      d_values_output.get(),
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets.get(),
                                                                      d_offsets.get() + 1,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream,
                                                                      debug_synchronous));
            }

            const auto keys_output   = d_keys_output.load();
            const auto values_output = d_values_output.load();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));
        }
    }
}

template<typename TestFixture>
inline void sort_pairs_unspecified_ranges()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    using config                      = typename TestFixture::params::config;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

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

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            std::vector<offset_type> begin_offsets;
            unsigned int             segments_count = 0;
            size_t                   offset         = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                begin_offsets.push_back(offset);
                segments_count++;
                offset += segment_length;
            }
            begin_offsets.push_back(size);
            std::vector<offset_type> end_offsets(begin_offsets.cbegin() + 1, begin_offsets.cend());
            begin_offsets.pop_back();

            size_t            empty_segments = rocprim::max(segments_count / 16, 1u);
            std::vector<bool> is_empty_segment(segments_count, false);
            std::fill(is_empty_segment.begin(), is_empty_segment.begin() + empty_segments, true);
            std::shuffle(is_empty_segment.begin(), is_empty_segment.end(), gen);

            for(size_t i = 0; i < segments_count; i++)
            {
                if(is_empty_segment[i])
                {
                    begin_offsets[i] = 0;
                    end_offsets[i]   = 0;
                }
            }

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output(keys_input);

            common::device_ptr<value_type> d_values_input(values_input);
            common::device_ptr<value_type> d_values_output(values_input);

            common::device_ptr<offset_type> d_offsets_begin(begin_offsets);
            common::device_ptr<offset_type> d_offsets_end(end_offsets);

            using key_value = std::pair<key_type, value_type>;

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(expected.begin() + begin_offsets[i],
                                 expected.begin() + end_offsets[i],
                                 test_utils::key_value_comparator<key_type,
                                                                  value_type,
                                                                  descending,
                                                                  start_bit,
                                                                  end_bit>());
            }

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input.get(),
                                                                  d_keys_output.get(),
                                                                  d_values_input.get(),
                                                                  d_values_output.get(),
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets_begin.get(),
                                                                  d_offsets_end.get(),
                                                                  start_bit,
                                                                  end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_pairs_desc<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     d_keys_output.get(),
                                                                     d_values_input.get(),
                                                                     d_values_output.get(),
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets_begin.get(),
                                                                     d_offsets_end.get(),
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys_input.get(),
                                                                      d_keys_output.get(),
                                                                      d_values_input.get(),
                                                                      d_values_output.get(),
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets_begin.get(),
                                                                      d_offsets_end.get(),
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream));
            }

            const auto keys_output   = d_keys_output.load();
            const auto values_output = d_values_output.load();

            for(size_t i = 0; i < size; i++)
            {
                ASSERT_EQ(keys_output[i], expected[i].first);
                ASSERT_EQ(values_output[i], expected[i].second);
            }
        }
    }
}

template<typename TestFixture>
inline void sort_keys_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using config                      = typename TestFixture::params::config;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device         rd;
    std::default_random_engine gen(rd());

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

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

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

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output(size);

            common::device_ptr<offset_type> d_offsets(offsets);

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(
                    expected.begin() + offsets[i],
                    expected.begin() + offsets[i + 1],
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>());
            }

            rocprim::double_buffer<key_type> d_keys(d_keys_input.get(), d_keys_output.get());

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys,
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets.get(),
                                                                 d_offsets.get() + 1,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(rocprim::segmented_radix_sort_keys_desc<config>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_keys,
                                                                          size,
                                                                          segments_count,
                                                                          d_offsets.get(),
                                                                          d_offsets.get() + 1,
                                                                          start_bit,
                                                                          end_bit,
                                                                          stream,
                                                                          debug_synchronous));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_keys<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys,
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets.get(),
                                                                     d_offsets.get() + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream,
                                                                     debug_synchronous));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys.current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
        }
    }
}

template<typename TestFixture>
inline void sort_pairs_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    using config                      = typename TestFixture::params::config;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device         rd;
    std::default_random_engine gen(rd());

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

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                common::generate_limits<key_type>::min(),
                common::generate_limits<key_type>::max(),
                seed_value);

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

            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output(size);

            common::device_ptr<value_type> d_values_input(values_input);
            common::device_ptr<value_type> d_values_output(size);

            common::device_ptr<offset_type> d_offsets(offsets);

            using key_value = std::pair<key_type, value_type>;

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(expected.begin() + offsets[i],
                                 expected.begin() + offsets[i + 1],
                                 test_utils::key_value_comparator<key_type,
                                                                  value_type,
                                                                  descending,
                                                                  start_bit,
                                                                  end_bit>());
            }
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            rocprim::double_buffer<key_type>   d_keys(d_keys_input.get(), d_keys_output.get());
            rocprim::double_buffer<value_type> d_values(d_values_input.get(),
                                                        d_values_output.get());

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys,
                                                                  d_values,
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets.get(),
                                                                  d_offsets.get() + 1,
                                                                  start_bit,
                                                                  end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(descending)
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_pairs_desc<config>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys,
                                                                     d_values,
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets.get(),
                                                                     d_offsets.get() + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream,
                                                                     debug_synchronous));
            }
            else
            {
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys,
                                                                      d_values,
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets.get(),
                                                                      d_offsets.get() + 1,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream,
                                                                      debug_synchronous));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys.current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values.current(),
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));
        }
    }
}

#endif // TEST_DEVICE_SEGMENTED_RADIX_SORT_HPP_
