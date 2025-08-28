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

#ifndef HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_SORT_HPP_
#define HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_SORT_HPP_

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/device/device_segmented_sort.hpp>

#include "test_utils_data_generation.hpp"

enum class SortMethod
{
    SortAscending,
    StableSortAscending,
    SortDescending,
    StableSortDescending
};

constexpr bool is_descending(const SortMethod method)
{
    return method == SortMethod::SortDescending || method == SortMethod::StableSortDescending;
}

template<class Key,
         class Value,
         SortMethod   Method,
         unsigned int MinSegmentLength,
         unsigned int MaxSegmentLength>
struct params
{
    using key_type                                   = Key;
    using value_type                                 = Value;
    static constexpr SortMethod   method             = Method;
    static constexpr bool         descending         = is_descending(method);
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedSort : public ::testing::Test
{
public:
    using params = Params;
};

template<typename... Args>
inline void dispatch_sort_keys(const SortMethod method, Args&&... args)
{
    switch(method)
    {
        case SortMethod::SortAscending:
            HIP_CHECK(hipcub::DeviceSegmentedSort::SortKeys(std::forward<Args>(args)...););
            break;
        case SortMethod::SortDescending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::SortKeysDescending(std::forward<Args>(args)...););
            break;
        case SortMethod::StableSortAscending:
            HIP_CHECK(hipcub::DeviceSegmentedSort::StableSortKeys(std::forward<Args>(args)...););
            break;
        case SortMethod::StableSortDescending:
            HIP_CHECK(hipcub::DeviceSegmentedSort::StableSortKeysDescending(
                          std::forward<Args>(args)...););
            break;
        default: FAIL();
    }
}

template<typename key_type, typename offset_type>
inline void generate_input_data(std::vector<key_type>&    keys_input,
                                std::vector<offset_type>& offsets,
                                const size_t              size,
                                const int                 seed_value,
                                const unsigned            min_segment_length,
                                const unsigned            max_segment_length)
{
    std::default_random_engine            gen(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(min_segment_length,
                                                                      max_segment_length);

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

    offsets.clear();
    size_t offset = 0;
    while(offset < size)
    {
        const size_t segment_length = segment_length_distribution(gen);
        offsets.push_back(offset);
        offset += segment_length;
    }
    offsets.push_back(size);
}

template<typename T>
inline T* hipMallocAndCopy(const std::vector<T>& data)
{
    T* d_ptr{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_ptr, data.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_ptr, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
    return d_ptr;
}

template<typename key_type, typename offset_type>
inline std::vector<key_type> generate_expected_data(const std::vector<key_type>&    keys_input,
                                                    const std::vector<offset_type>& offsets,
                                                    const bool                      descending)
{
    const size_t          segments_count = offsets.size() - 1;
    std::vector<key_type> expected(keys_input);
    for(size_t i = 0; i < segments_count; ++i)
    {
        std::stable_sort(expected.begin() + offsets[i], expected.begin() + offsets[i + 1]);
        if(descending)
        {
            std::reverse(expected.begin() + offsets[i], expected.begin() + offsets[i + 1]);
        }
    }
    return expected;
}

template<bool descending, typename key_type, typename value_type, typename offset_type>
std::vector<std::pair<key_type, value_type>> inline generate_expected_data(
    const std::vector<key_type>&    keys_input,
    const std::vector<value_type>&  values_input,
    const std::vector<offset_type>& offsets)
{
    const size_t                                 size           = keys_input.size();
    const size_t                                 segments_count = offsets.size() - 1;
    std::vector<std::pair<key_type, value_type>> expected(size);
    for(size_t i = 0; i < size; ++i)
    {
        expected[i] = std::make_pair(keys_input[i], values_input[i]);
    }
    for(size_t i = 0; i < segments_count; ++i)
    {
        std::stable_sort(
            expected.begin() + offsets[i],
            expected.begin() + offsets[i + 1],
            test_utils::
                key_value_comparator<key_type, value_type, descending, 0, sizeof(key_type) * 8>());
    }
    return expected;
}

template<typename T>
inline std::vector<T> download(const T* const d_ptr, const size_t size)
{
    std::vector<T> data(size);
    HIP_CHECK(hipMemcpy(data.data(), d_ptr, size * sizeof(T), hipMemcpyDeviceToHost));
    return data;
}

TYPED_TEST_SUITE_P(HipcubDeviceSegmentedSort);

template<typename TestFixture>
inline void sort_keys()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                            = typename TestFixture::params::key_type;
    using offset_type                         = unsigned int;
    constexpr SortMethod   method             = TestFixture::params::method;
    constexpr bool         descending         = TestFixture::params::descending;
    constexpr unsigned int min_segment_length = TestFixture::params::min_segment_length;
    constexpr unsigned int max_segment_length = TestFixture::params::max_segment_length;
    constexpr hipStream_t  stream             = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(const size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            std::vector<key_type>    keys_input;
            std::vector<offset_type> offsets;
            generate_input_data(keys_input,
                                offsets,
                                size,
                                seed_value,
                                min_segment_length,
                                max_segment_length);
            const size_t segments_count = offsets.size() - 1;

            key_type*    d_keys_input = hipMallocAndCopy(keys_input);
            offset_type* d_offsets    = hipMallocAndCopy(offsets);
            key_type*    d_keys_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));

            const std::vector<key_type> expected
                = generate_expected_data(keys_input, offsets, descending);

            size_t temporary_storage_bytes{};
            dispatch_sort_keys(method,
                               static_cast<void*>(nullptr),
                               temporary_storage_bytes,
                               d_keys_input,
                               d_keys_output,
                               size,
                               segments_count,
                               d_offsets,
                               d_offsets + 1);

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage{};
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            dispatch_sort_keys(method,
                               d_temporary_storage,
                               temporary_storage_bytes,
                               d_keys_input,
                               d_keys_output,
                               size,
                               segments_count,
                               d_offsets,
                               d_offsets + 1,
                               stream);

            const std::vector<key_type> keys_output = download(d_keys_output, size);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            ASSERT_EQ(keys_output, expected);
        }
    }
}

template<typename TestFixture>
inline void sort_keys_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                            = typename TestFixture::params::key_type;
    using offset_type                         = unsigned int;
    constexpr SortMethod   method             = TestFixture::params::method;
    constexpr bool         descending         = TestFixture::params::descending;
    constexpr unsigned int min_segment_length = TestFixture::params::min_segment_length;
    constexpr unsigned int max_segment_length = TestFixture::params::max_segment_length;
    constexpr hipStream_t  stream             = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(const size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            std::vector<key_type>    keys_input;
            std::vector<offset_type> offsets;
            generate_input_data(keys_input,
                                offsets,
                                size,
                                seed_value,
                                min_segment_length,
                                max_segment_length);
            const size_t segments_count = offsets.size() - 1;

            key_type*    d_keys_input = hipMallocAndCopy(keys_input);
            offset_type* d_offsets    = hipMallocAndCopy(offsets);
            key_type*    d_keys_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));

            std::vector<key_type> expected
                = generate_expected_data(keys_input, offsets, descending);

            hipcub::DoubleBuffer<key_type> d_keys(d_keys_input, d_keys_output);

            size_t temporary_storage_bytes{};
            dispatch_sort_keys(method,
                               static_cast<void*>(nullptr),
                               temporary_storage_bytes,
                               d_keys,
                               size,
                               segments_count,
                               d_offsets,
                               d_offsets + 1);

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage{};
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            dispatch_sort_keys(method,
                               d_temporary_storage,
                               temporary_storage_bytes,
                               d_keys,
                               size,
                               segments_count,
                               d_offsets,
                               d_offsets + 1,
                               stream);

            const std::vector<key_type> keys_output = download(d_keys.Current(), size);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            ASSERT_EQ(keys_output, expected);
        }
    }
}

template<typename... Args>
void dispatch_sort_pairs(const SortMethod method, Args&&... args)
{
    switch(method)
    {
        case SortMethod::SortAscending:
            HIP_CHECK(hipcub::DeviceSegmentedSort::SortPairs(std::forward<Args>(args)...));
            break;
        case SortMethod::SortDescending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::SortPairsDescending(std::forward<Args>(args)...));
            break;
        case SortMethod::StableSortAscending:
            HIP_CHECK(hipcub::DeviceSegmentedSort::StableSortPairs(std::forward<Args>(args)...));
            break;
        case SortMethod::StableSortDescending:
            HIP_CHECK(hipcub::DeviceSegmentedSort::StableSortPairsDescending(
                std::forward<Args>(args)...));
            break;
        default: FAIL();
    }
}

template<typename TestFixture>
inline void sort_pairs()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                            = typename TestFixture::params::key_type;
    using value_type                          = typename TestFixture::params::value_type;
    using offset_type                         = unsigned int;
    constexpr SortMethod   method             = TestFixture::params::method;
    constexpr bool         descending         = TestFixture::params::descending;
    constexpr unsigned int min_segment_length = TestFixture::params::min_segment_length;
    constexpr unsigned int max_segment_length = TestFixture::params::max_segment_length;
    constexpr hipStream_t  stream             = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(const size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            std::vector<key_type>    keys_input;
            std::vector<offset_type> offsets;
            generate_input_data(keys_input,
                                offsets,
                                size,
                                seed_value,
                                min_segment_length,
                                max_segment_length);
            const size_t segments_count = offsets.size() - 1;

            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type*    d_keys_input   = hipMallocAndCopy(keys_input);
            value_type*  d_values_input = hipMallocAndCopy(values_input);
            offset_type* d_offsets      = hipMallocAndCopy(offsets);
            key_type*    d_keys_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            value_type* d_values_output{};
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));

            const std::vector<std::pair<key_type, value_type>> expected
                = generate_expected_data<descending>(keys_input, values_input, offsets);

            size_t temporary_storage_bytes{};
            dispatch_sort_pairs(method,
                                static_cast<void*>(nullptr),
                                temporary_storage_bytes,
                                d_keys_input,
                                d_keys_output,
                                d_values_input,
                                d_values_output,
                                size,
                                segments_count,
                                d_offsets,
                                d_offsets + 1);
            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage{};
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            dispatch_sort_pairs(method,
                                d_temporary_storage,
                                temporary_storage_bytes,
                                d_keys_input,
                                d_keys_output,
                                d_values_input,
                                d_values_output,
                                size,
                                segments_count,
                                d_offsets,
                                d_offsets + 1,
                                stream);

            const std::vector<key_type>   keys_output   = download(d_keys_output, size);
            const std::vector<value_type> values_output = download(d_values_output, size);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));

            for(size_t i = 0; i < size; ++i)
            {
                ASSERT_EQ(keys_output[i], expected[i].first);
                ASSERT_EQ(values_output[i], expected[i].second);
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

    using key_type                            = typename TestFixture::params::key_type;
    using value_type                          = typename TestFixture::params::value_type;
    using offset_type                         = unsigned int;
    constexpr SortMethod   method             = TestFixture::params::method;
    constexpr bool         descending         = TestFixture::params::descending;
    constexpr unsigned int min_segment_length = TestFixture::params::min_segment_length;
    constexpr unsigned int max_segment_length = TestFixture::params::max_segment_length;
    constexpr hipStream_t  stream             = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(const size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            std::vector<key_type>    keys_input;
            std::vector<offset_type> offsets;
            generate_input_data(keys_input,
                                offsets,
                                size,
                                seed_value,
                                min_segment_length,
                                max_segment_length);
            const size_t segments_count = offsets.size() - 1;

            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type*    d_keys_input   = hipMallocAndCopy(keys_input);
            value_type*  d_values_input = hipMallocAndCopy(values_input);
            offset_type* d_offsets      = hipMallocAndCopy(offsets);
            key_type*    d_keys_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            value_type* d_values_output{};
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));

            const std::vector<std::pair<key_type, value_type>> expected
                = generate_expected_data<descending>(keys_input, values_input, offsets);

            hipcub::DoubleBuffer<key_type>   d_keys(d_keys_input, d_keys_output);
            hipcub::DoubleBuffer<value_type> d_values(d_values_input, d_values_output);

            size_t temporary_storage_bytes{};
            dispatch_sort_pairs(method,
                                static_cast<void*>(nullptr),
                                temporary_storage_bytes,
                                d_keys,
                                d_values,
                                size,
                                segments_count,
                                d_offsets,
                                d_offsets + 1);
            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage{};
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            dispatch_sort_pairs(method,
                                d_temporary_storage,
                                temporary_storage_bytes,
                                d_keys,
                                d_values,
                                size,
                                segments_count,
                                d_offsets,
                                d_offsets + 1,
                                stream);

            const std::vector<key_type>   keys_output   = download(d_keys.Current(), size);
            const std::vector<value_type> values_output = download(d_values.Current(), size);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));

            for(size_t i = 0; i < size; ++i)
            {
                ASSERT_EQ(keys_output[i], expected[i].first);
                ASSERT_EQ(values_output[i], expected[i].second);
            }
        }
    }
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HipcubDeviceSegmentedSort);

#endif // HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_SORT_HPP_
