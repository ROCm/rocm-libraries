// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_RADIX_SORT_HPP_

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/device/device_segmented_radix_sort.hpp>

#include "test_utils_data_generation.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

template<class Key,
         class Value,
         bool         Descending,
         unsigned int StartBit,
         unsigned int EndBit,
         unsigned int MinSegmentLength,
         unsigned int MaxSegmentLength>
struct params
{
    using key_type                                   = Key;
    using value_type                                 = Value;
    static constexpr bool         descending         = Descending;
    static constexpr unsigned int start_bit          = StartBit;
    static constexpr unsigned int end_bit            = EndBit;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedRadixSort : public ::testing::Test
{
public:
    using params = Params;
};

TYPED_TEST_SUITE_P(HipcubDeviceSegmentedRadixSort);

template<typename TestFixture>
inline void sort_keys()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   (key_type)-1000,
                                                                   (key_type) + 1000,
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

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

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

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
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys_input,
                                                                 d_keys_output,
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets,
                                                                 d_offsets + 1,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortKeysDescending(d_temporary_storage,
                                                                         temporary_storage_bytes,
                                                                         d_keys_input,
                                                                         d_keys_output,
                                                                         size,
                                                                         segments_count,
                                                                         d_offsets,
                                                                         d_offsets + 1,
                                                                         start_bit,
                                                                         end_bit,
                                                                         stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(d_temporary_storage,
                                                                     temporary_storage_bytes,
                                                                     d_keys_input,
                                                                     d_keys_output,
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets,
                                                                     d_offsets + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            for(size_t i = 0; i < size; i++)
            {
                ASSERT_EQ(keys_output[i], expected[i]);
            }
        }
    }
}

template<typename TestFixture>
inline void sort_keys_empty_data()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

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
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   static_cast<key_type>(-1000),
                                                                   static_cast<key_type>(1000),
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

            std::vector<offset_type> offsets(2);
            offsets[0] = 0;
            offsets[1] = 0;

            key_type* d_keys;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys,
                                                                 d_keys,
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets,
                                                                 d_offsets + 1,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortKeysDescending(d_temporary_storage,
                                                                         temporary_storage_bytes,
                                                                         d_keys,
                                                                         d_keys,
                                                                         size,
                                                                         segments_count,
                                                                         d_offsets,
                                                                         d_offsets + 1,
                                                                         start_bit,
                                                                         end_bit,
                                                                         stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(d_temporary_storage,
                                                                     temporary_storage_bytes,
                                                                     d_keys,
                                                                     d_keys,
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets,
                                                                     d_offsets + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_offsets));

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
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    size_t size           = 1 << 20;
    size_t segments_count = 2;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = test_utils::get_random_data<key_type>(size,
                                                               static_cast<key_type>(-1000),
                                                               static_cast<key_type>(1000),
                                                               seed_value);
        }
        else
        {
            keys_input = test_utils::get_random_data<key_type>(size,
                                                               std::numeric_limits<key_type>::min(),
                                                               std::numeric_limits<key_type>::max(),
                                                               seed_value + seed_value_addition);
        }

        std::vector<offset_type> offsets(3);
        offsets[0] = 0;
        offsets[1] = static_cast<offset_type>(size / 2);
        offsets[2] = static_cast<offset_type>(size);

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        offset_type* d_offsets;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets,
                                                     (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

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
        HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                             temporary_storage_bytes,
                                                             d_keys_input,
                                                             d_keys_output,
                                                             size,
                                                             segments_count,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             start_bit,
                                                             end_bit));

        ASSERT_GT(temporary_storage_bytes, 0U);

        void* d_temporary_storage;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

        if(descending)
        {
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeysDescending(d_temporary_storage,
                                                                           temporary_storage_bytes,
                                                                           d_keys_input,
                                                                           d_keys_output,
                                                                           size,
                                                                           segments_count,
                                                                           d_offsets,
                                                                           d_offsets + 1,
                                                                           start_bit,
                                                                           end_bit,
                                                                           stream));
        }
        else
        {
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(d_temporary_storage,
                                                                 temporary_storage_bytes,
                                                                 d_keys_input,
                                                                 d_keys_output,
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets,
                                                                 d_offsets + 1,
                                                                 start_bit,
                                                                 end_bit,
                                                                 stream));
        }

        std::vector<key_type> keys_output(size);
        HIP_CHECK(hipMemcpy(keys_output.data(),
                            d_keys_output,
                            size * sizeof(key_type),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_offsets));

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
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   static_cast<key_type>(-1000),
                                                                   static_cast<key_type>(1000),
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

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

            size_t            empty_segments = test_utils::max(segments_count / 16, 1);
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

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_keys_output,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets_begin;
            offset_type* d_offsets_end;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets_begin,
                                                         segments_count * sizeof(offset_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets_end,
                                                         segments_count * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets_begin,
                                begin_offsets.data(),
                                segments_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            HIP_CHECK(hipMemcpy(d_offsets_end,
                                end_offsets.data(),
                                segments_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));

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
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys_input,
                                                                 d_keys_output,
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets_begin,
                                                                 d_offsets_end,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortKeysDescending(d_temporary_storage,
                                                                         temporary_storage_bytes,
                                                                         d_keys_input,
                                                                         d_keys_output,
                                                                         size,
                                                                         segments_count,
                                                                         d_offsets_begin,
                                                                         d_offsets_end,
                                                                         start_bit,
                                                                         end_bit,
                                                                         stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(d_temporary_storage,
                                                                     temporary_storage_bytes,
                                                                     d_keys_input,
                                                                     d_keys_output,
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets_begin,
                                                                     d_offsets_end,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets_begin));
            HIP_CHECK(hipFree(d_offsets_end));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
        }
    }
}

template<typename TestFixture>
inline void sort_pairs()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   (key_type)-1000,
                                                                   (key_type) + 1000,
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

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
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

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

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(d_temporary_storage,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input,
                                                                  d_keys_output,
                                                                  d_values_input,
                                                                  d_values_output,
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets,
                                                                  d_offsets + 1,
                                                                  start_bit,
                                                                  end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortPairsDescending(d_temporary_storage,
                                                                          temporary_storage_bytes,
                                                                          d_keys_input,
                                                                          d_keys_output,
                                                                          d_values_input,
                                                                          d_values_output,
                                                                          size,
                                                                          segments_count,
                                                                          d_offsets,
                                                                          d_offsets + 1,
                                                                          start_bit,
                                                                          end_bit,
                                                                          stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(d_temporary_storage,
                                                                      temporary_storage_bytes,
                                                                      d_keys_input,
                                                                      d_keys_output,
                                                                      d_values_input,
                                                                      d_values_output,
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets,
                                                                      d_offsets + 1,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));

            for(size_t i = 0; i < size; i++)
            {
                ASSERT_EQ(keys_output[i], expected[i].first);
                ASSERT_EQ(values_output[i], expected[i].second);
            }
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
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   static_cast<key_type>(-1000),
                                                                   static_cast<key_type>(1000),
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

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

            size_t            empty_segments = test_utils::max(segments_count / 16, 1);
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

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_keys_output,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_values_output,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets_begin;
            offset_type* d_offsets_end;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets_begin,
                                                         segments_count * sizeof(offset_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets_end,
                                                         segments_count * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets_begin,
                                begin_offsets.data(),
                                segments_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            HIP_CHECK(hipMemcpy(d_offsets_end,
                                end_offsets.data(),
                                segments_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));
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
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input,
                                                                  d_keys_output,
                                                                  d_values_input,
                                                                  d_values_output,
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets_begin,
                                                                  d_offsets_end,
                                                                  start_bit,
                                                                  end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortPairsDescending(d_temporary_storage,
                                                                          temporary_storage_bytes,
                                                                          d_keys_input,
                                                                          d_keys_output,
                                                                          d_values_input,
                                                                          d_values_output,
                                                                          size,
                                                                          segments_count,
                                                                          d_offsets_begin,
                                                                          d_offsets_end,
                                                                          start_bit,
                                                                          end_bit,
                                                                          stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(d_temporary_storage,
                                                                      temporary_storage_bytes,
                                                                      d_keys_input,
                                                                      d_keys_output,
                                                                      d_values_input,
                                                                      d_values_output,
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets_begin,
                                                                      d_offsets_end,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets_begin));
            HIP_CHECK(hipFree(d_offsets_end));

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
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   (key_type)-1000,
                                                                   (key_type) + 1000,
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

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

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(
                    expected.begin() + offsets[i],
                    expected.begin() + offsets[i + 1],
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>());
            }

            hipcub::DoubleBuffer<key_type> d_keys(d_keys_input, d_keys_output);

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                                 temporary_storage_bytes,
                                                                 d_keys,
                                                                 size,
                                                                 segments_count,
                                                                 d_offsets,
                                                                 d_offsets + 1,
                                                                 start_bit,
                                                                 end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortKeysDescending(d_temporary_storage,
                                                                         temporary_storage_bytes,
                                                                         d_keys,
                                                                         size,
                                                                         segments_count,
                                                                         d_offsets,
                                                                         d_offsets + 1,
                                                                         start_bit,
                                                                         end_bit,
                                                                         stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(d_temporary_storage,
                                                                     temporary_storage_bytes,
                                                                     d_keys,
                                                                     size,
                                                                     segments_count,
                                                                     d_offsets,
                                                                     d_offsets + 1,
                                                                     start_bit,
                                                                     end_bit,
                                                                     stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys.Current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            for(size_t i = 0; i < size; i++)
            {
                ASSERT_EQ(keys_output[i], expected[i]);
            }
        }
    }
}

template<typename TestFixture>
inline void sort_pairs_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    constexpr bool         descending = TestFixture::params::descending;
    constexpr unsigned int start_bit  = TestFixture::params::start_bit;
    constexpr unsigned int end_bit    = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            std::vector<key_type> keys_input;
            if(std::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size,
                                                                   (key_type)-1000,
                                                                   (key_type) + 1000,
                                                                   seed_value);
            }
            else
            {
                keys_input
                    = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value + seed_value_addition);
            }

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
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_offsets,
                                                   (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                (segments_count + 1) * sizeof(offset_type),
                                hipMemcpyHostToDevice));

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

            hipcub::DoubleBuffer<key_type>   d_keys(d_keys_input, d_keys_output);
            hipcub::DoubleBuffer<value_type> d_values(d_values_input, d_values_output);

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(d_temporary_storage,
                                                                  temporary_storage_bytes,
                                                                  d_keys,
                                                                  d_values,
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets,
                                                                  d_offsets + 1,
                                                                  start_bit,
                                                                  end_bit));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    hipcub::DeviceSegmentedRadixSort::SortPairsDescending(d_temporary_storage,
                                                                          temporary_storage_bytes,
                                                                          d_keys,
                                                                          d_values,
                                                                          size,
                                                                          segments_count,
                                                                          d_offsets,
                                                                          d_offsets + 1,
                                                                          start_bit,
                                                                          end_bit,
                                                                          stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(d_temporary_storage,
                                                                      temporary_storage_bytes,
                                                                      d_keys,
                                                                      d_values,
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets,
                                                                      d_offsets + 1,
                                                                      start_bit,
                                                                      end_bit,
                                                                      stream));
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys.Current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values.Current(),
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));

            for(size_t i = 0; i < size; i++)
            {
                ASSERT_EQ(keys_output[i], expected[i].first);
                ASSERT_EQ(values_output[i], expected[i].second);
            }
        }
    }
}

#endif // HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_RADIX_SORT_HPP_
