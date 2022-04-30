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

#ifndef HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_SORT_HPP_
#define HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_SORT_HPP_

#include "common_test_header.hpp"

// hipcub API
#include "hipcub/device/device_segmented_sort.hpp"

enum class SortMethod
{
    SortAscending,
    StableSortAscending,
    SortDescending,
    StableSortDescending
};

constexpr bool is_descending(const SortMethod method)
{
    return method == SortMethod::SortDescending
        || method == SortMethod::StableSortDescending;
}

template<
    class Key,
    class Value,
    SortMethod Method,
    unsigned int MinSegmentLength,
    unsigned int MaxSegmentLength
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr SortMethod method = Method;
    static constexpr bool descending = is_descending(method);
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedSort : public ::testing::Test {
public:
    using params = Params;
};

using Params = ::testing::Types<
    params<int, int, SortMethod::SortAscending, 0, 100>,
    params<int, int, SortMethod::SortDescending, 0, 100>,
    params<int, int, SortMethod::StableSortAscending, 0, 100>,
    params<int, int, SortMethod::StableSortDescending, 0, 100>,

    params<unsigned, int, SortMethod::SortAscending, 10, 312>,
    params<unsigned, int, SortMethod::SortDescending, 10, 312>,
    params<unsigned, int, SortMethod::StableSortAscending, 10, 312>,
    params<unsigned, int, SortMethod::StableSortDescending, 10, 312>,

    params<char, int, SortMethod::SortAscending, 1, 1239>,
    params<char, int, SortMethod::SortDescending, 1, 1239>,
    params<char, int, SortMethod::StableSortAscending, 1, 1239>,
    params<char, int, SortMethod::StableSortDescending, 1, 1239>,

    params<float, int, SortMethod::SortAscending, 0, 322>,
    params<float, int, SortMethod::SortDescending, 0, 322>,
    params<float, int, SortMethod::StableSortAscending, 0, 322>,
    params<float, int, SortMethod::StableSortDescending, 0, 322>,

    params<double, int, SortMethod::SortAscending, 321, 555>,
    params<double, int, SortMethod::SortDescending, 321, 555>,
    params<double, int, SortMethod::StableSortAscending, 321, 555>,
    params<double, int, SortMethod::StableSortDescending, 321, 555>
>;

template<typename ... Args>
inline void dispatch_sort_keys(const SortMethod method, Args&& ... args)
{
    switch (method)
    {
        case SortMethod::SortAscending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::SortKeys(std::forward<Args>(args) ...);
            );
            break;
        case SortMethod::SortDescending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::SortKeysDescending(std::forward<Args>(args) ...);
            );
            break;
        case SortMethod::StableSortAscending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::StableSortKeys(std::forward<Args>(args) ...);
            );
            break;
        case SortMethod::StableSortDescending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::StableSortKeysDescending(std::forward<Args>(args) ...);
            );
            break;
        default:
            FAIL();
    }
}

inline std::vector<size_t> get_sizes(const int seed_value)
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        1000000,
        (1 << 16) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 100000, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

template<typename key_type, typename offset_type>
void generate_input_data(std::vector<key_type> &keys_input,
                         std::vector<offset_type> &offsets,
                         const size_t size,
                         const int seed_value,
                         const unsigned min_segment_length,
                         const unsigned max_segment_length)
{
    std::default_random_engine gen(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(
        min_segment_length,
        max_segment_length
    );

    if (std::is_floating_point<key_type>::value)
    {
        keys_input = test_utils::get_random_data<key_type>(
            size,
            static_cast<key_type>(-1000),
            static_cast<key_type>(1000),
            seed_value
        );
    }
    else
    {
        keys_input = test_utils::get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max(),
            seed_value + seed_value_addition
        );
    }
    
    offsets.clear();
    unsigned segments_count = 0;
    size_t offset = 0;
    while(offset < size)
    {
        const size_t segment_length = segment_length_distribution(gen);
        offsets.push_back(offset);
        ++segments_count;
        offset += segment_length;
    }
    offsets.push_back(size);
}

template<typename T>
T * hipMallocAndCopy(const std::vector<T> &data)
{
    T * d_ptr{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_ptr, data.size() * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_ptr, data.data(),
            data.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );    
    return d_ptr;
}

template<typename key_type, typename offset_type>
std::vector<key_type> generate_expected_data(const std::vector<key_type> &keys_input,
                                             const std::vector<offset_type> &offsets,
                                             const bool descending)
{
    const size_t segments_count = offsets.size() - 1;
    std::vector<key_type> expected(keys_input);
    for (size_t i = 0; i < segments_count; ++i)
    {
        std::stable_sort(
            expected.begin() + offsets[i],
            expected.begin() + offsets[i + 1]
        );
        if (descending)
        {
            std::reverse(
                expected.begin() + offsets[i],
                expected.begin() + offsets[i + 1]
            );
        }
    }
    return expected;
}

template<bool descending, typename key_type, typename value_type, typename offset_type>
std::vector<std::pair<key_type, value_type>>
generate_expected_data(const std::vector<key_type> &keys_input,
                       const std::vector<value_type> &values_input,
                       const std::vector<offset_type> &offsets)
{
    const size_t size = keys_input.size();
    const size_t segments_count = offsets.size() - 1;
    std::vector<std::pair<key_type, value_type>> expected(size);
    for (size_t i = 0; i < size; ++i)
    {
        expected[i] = std::make_pair(keys_input[i], values_input[i]);
    }
    for (size_t i = 0; i < segments_count; ++i)
    {
        std::stable_sort(
            expected.begin() + offsets[i],
            expected.begin() + offsets[i + 1],
            test_utils::key_value_comparator<key_type, value_type, descending, 0, sizeof(key_type) * 8>()
        );
    }
    return expected;
}

template<typename T>
std::vector<T> download(const T * const d_ptr, const size_t size)
{
    std::vector<T> data(size);
    HIP_CHECK(
        hipMemcpy(
            data.data(), d_ptr,
            size * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
    return data;
}

TYPED_TEST_SUITE(HipcubDeviceSegmentedSort, Params);

template<typename ... Args>
void dispatch_sort_pairs(const SortMethod method, Args&& ... args)
{
    switch (method)
    {
        case SortMethod::SortAscending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::SortPairs(std::forward<Args>(args) ...)
            );
            break;
        case SortMethod::SortDescending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::SortPairsDescending(std::forward<Args>(args) ...)
            );
            break;
        case SortMethod::StableSortAscending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::StableSortPairs(std::forward<Args>(args) ...)
            );
            break;
        case SortMethod::StableSortDescending:
            HIP_CHECK(
                hipcub::DeviceSegmentedSort::StableSortPairsDescending(std::forward<Args>(args) ...)
            );
            break;
        default:
            FAIL();
    }
}

#endif // HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_SORT_HPP_
