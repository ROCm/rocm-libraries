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
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// hipCUB API
#include <hipcub/hipcub.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

template<
    class Key,
    class Value,
    bool Descending = false,
    unsigned int StartBit = 0,
    unsigned int EndBit = sizeof(Key) * 8,
    bool CheckHugeSizes = false
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr bool descending = Descending;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
    static constexpr bool check_huge_sizes = CheckHugeSizes;
};

template<class Params>
class HipcubDeviceRadixSort : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<signed char, double, true>,
    params<int, short>,
    params<short, int, true>,
    params<long long, char>,
    params<double, unsigned int>,
    params<double, int, true>,
    params<float, int>,
    params<int, test_utils::custom_test_type<float>>,

    // start_bit and end_bit
    params<unsigned char, int, true, 0, 7>,
    params<unsigned short, int, true, 4, 10>,
    params<unsigned int, short, false, 3, 22>,
    params<unsigned int, double, true, 4, 21>,
    params<unsigned int, short, true, 0, 15>,
    params<unsigned long long, char, false, 8, 20>,
    params<unsigned short, double, false, 8, 11>,

    // huge sizes to check correctness of more than 1 block per batch
    params<float, char, true, 0, 32, true>
> Params;

TYPED_TEST_CASE(HipcubDeviceRadixSort, Params);

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator
{
private:
    template<unsigned int CStartBit, unsigned int CEndBit>
    constexpr static bool all_bits()
    {
        return (CStartBit == 0 && CEndBit == sizeof(Key) * 8);
    }

    template<unsigned int CStartBit, unsigned int CEndBit>
    auto compare(const Key& lhs, const Key& rhs) const
        -> typename std::enable_if<all_bits<CStartBit, CEndBit>(), bool>::type
    {
        return Descending ? (rhs < lhs) : (lhs < rhs);
    }

    template<unsigned int CStartBit, unsigned int CEndBit>
    auto compare(const Key& lhs, const Key& rhs) const
        -> typename std::enable_if<!all_bits<CStartBit, CEndBit>(), bool>::type
    {
        auto mask = (1ull << (EndBit - StartBit)) - 1;
        auto l = (static_cast<unsigned long long>(lhs) >> StartBit) & mask;
        auto r = (static_cast<unsigned long long>(rhs) >> StartBit) & mask;
        return Descending ? (r < l) : (l < r);
    }

public:
    static_assert(
        key_comparator::all_bits<StartBit, EndBit>() || std::is_unsigned<Key>::value,
        "Test supports start and end bits only for unsigned integers"
    );

    bool operator()(const Key& lhs, const Key& rhs)
    {
        return this->compare<StartBit, EndBit>(lhs, rhs);
    }
};

template<class Key, class Value, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_value_comparator
{
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return key_comparator<Key, Descending, StartBit, EndBit>()(lhs.first, rhs.first);
    }
};

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = { 1, 10, 53, 211, 1024, 2345, 4096, 34567, (1 << 16) - 1220, (1 << 23) - 76543 };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(10, 1, 100000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST(HipcubDeviceRadixSort, SortKeys)
{
    using key_type = typename TestFixture::params::key_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        if(size > (1 << 20) && !check_huge_sizes) continue;

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
        }
        else
        {
            keys_input = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max()
            );
        }

        key_type * d_keys_input;
        key_type * d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(
                d_keys_input, keys_input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        // Calculate expected results on host
        std::vector<key_type> expected(keys_input);
        std::stable_sort(expected.begin(), expected.end(), key_comparator<key_type, descending, start_bit, end_bit>());

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(
            hipcub::DeviceRadixSort::SortKeys(
                nullptr, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                start_bit, end_bit
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(descending)
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortKeysDescending(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortKeys(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));

        std::vector<key_type> keys_output(size);
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), d_keys_output,
                size * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_keys_output));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(keys_output[i], expected[i]);
        }
    }
}

TYPED_TEST(HipcubDeviceRadixSort, SortPairs)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        if(size > (1 << 20) && !check_huge_sizes) continue;

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
        }
        else
        {
            keys_input = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max()
            );
        }

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        key_type * d_keys_input;
        key_type * d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(
                d_keys_input, keys_input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        value_type * d_values_input;
        value_type * d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(value_type),
                hipMemcpyHostToDevice
            )
        );

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_input[i], values_input[i]);
        }
        std::stable_sort(
            expected.begin(), expected.end(),
            key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
        );

        void * d_temporary_storage = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(
            hipcub::DeviceRadixSort::SortPairs(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                start_bit, end_bit
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(descending)
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortPairsDescending(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortPairs(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_values_input));

        std::vector<key_type> keys_output(size);
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), d_keys_output,
                size * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );

        std::vector<value_type> values_output(size);
        HIP_CHECK(
            hipMemcpy(
                values_output.data(), d_values_output,
                size * sizeof(value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_output));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(keys_output[i], expected[i].first);
            ASSERT_EQ(values_output[i], expected[i].second);
        }
    }
}

TYPED_TEST(HipcubDeviceRadixSort, SortKeysDoubleBuffer)
{
    using key_type = typename TestFixture::params::key_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        if(size > (1 << 20) && !check_huge_sizes) continue;

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
        }
        else
        {
            keys_input = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max()
            );
        }

        key_type * d_keys_input;
        key_type * d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(
                d_keys_input, keys_input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        // Calculate expected results on host
        std::vector<key_type> expected(keys_input);
        std::stable_sort(expected.begin(), expected.end(), key_comparator<key_type, descending, start_bit, end_bit>());

        hipcub::DoubleBuffer<key_type> d_keys(d_keys_input, d_keys_output);

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(
            hipcub::DeviceRadixSort::SortKeys(
                nullptr, temporary_storage_bytes,
                d_keys, size,
                start_bit, end_bit
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(descending)
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortKeysDescending(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortKeys(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<key_type> keys_output(size);
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), d_keys.Current(),
                size * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(keys_output[i], expected[i]);
        }
    }
}

TYPED_TEST(HipcubDeviceRadixSort, SortPairsDoubleBuffer)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        if(size > (1 << 20) && !check_huge_sizes) continue;

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
        }
        else
        {
            keys_input = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max()
            );
        }

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        key_type * d_keys_input;
        key_type * d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(
                d_keys_input, keys_input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        value_type * d_values_input;
        value_type * d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(value_type),
                hipMemcpyHostToDevice
            )
        );

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_input[i], values_input[i]);
        }
        std::stable_sort(
            expected.begin(), expected.end(),
            key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
        );

        hipcub::DoubleBuffer<key_type> d_keys(d_keys_input, d_keys_output);
        hipcub::DoubleBuffer<value_type> d_values(d_values_input, d_values_output);

        void * d_temporary_storage = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(
            hipcub::DeviceRadixSort::SortPairs(
                d_temporary_storage, temporary_storage_bytes,
                d_keys, d_values, size,
                start_bit, end_bit
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(descending)
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortPairsDescending(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys, d_values, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceRadixSort::SortPairs(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys, d_values, size,
                    start_bit, end_bit,
                    stream, debug_synchronous
                )
            );
        }

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<key_type> keys_output(size);
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), d_keys.Current(),
                size * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );

        std::vector<value_type> values_output(size);
        HIP_CHECK(
            hipMemcpy(
                values_output.data(), d_values.Current(),
                size * sizeof(value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(keys_output[i], expected[i].first);
            ASSERT_EQ(values_output[i], expected[i].second);
        }
    }
}
