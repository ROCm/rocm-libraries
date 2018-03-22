// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 18) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 100000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

// Generate values ouside the desired histogram range (+-10%)
// (correctly handling test cases like uchar [0, 256), ushort [0, 65536))
template<class T, class U>
inline auto get_random_samples(size_t size, U min, U max)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    const long long min1 = static_cast<long long>(min);
    const long long max1 = static_cast<long long>(max);
    const long long d = max1 - min1;
    return test_utils::get_random_data<T>(
        size,
        static_cast<T>(std::max(min1 - d / 10, static_cast<long long>(std::numeric_limits<T>::lowest()))),
        static_cast<T>(std::min(max1 + d / 10, static_cast<long long>(std::numeric_limits<T>::max())))
    );
}

template<class T, class U>
inline auto get_random_samples(size_t size, U min, U max)
    -> typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    const double min1 = static_cast<double>(min);
    const double max1 = static_cast<double>(max);
    const double d = max1 - min1;
    return test_utils::get_random_data<T>(
        size,
        static_cast<T>(std::max(min1 - d / 10, static_cast<double>(std::numeric_limits<T>::lowest()))),
        static_cast<T>(std::min(max1 + d / 10, static_cast<double>(std::numeric_limits<T>::max())))
    );
}

template<
    class SampleType,
    unsigned int Bins,
    int LowerLevel,
    int UpperLevel,
    class LevelType = SampleType,
    class CounterType = int
>
struct params1
{
    using sample_type = SampleType;
    static constexpr unsigned int bins = Bins;
    static constexpr int lower_level = LowerLevel;
    static constexpr int upper_level = UpperLevel;
    using level_type = LevelType;
    using counter_type = CounterType;
};

template<class Params>
class RocprimDeviceHistogramEven : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params1<int, 10, 0, 10>,
    params1<int, 128, 0, 256>,
    params1<unsigned int, 12345, 10, 12355, short>,
    params1<unsigned short, 65536, 0, 65536, int>,
    params1<unsigned char, 10, 20, 240, unsigned char, unsigned int>,
    params1<unsigned char, 256, 0, 256, short>,

    params1<double, 10, 0, 1000, double, int>,
    params1<int, 123, 123, 5678, int>,
    params1<double, 55, -123, +123, double>
> Params1;

TYPED_TEST_CASE(RocprimDeviceHistogramEven, Params1);

TEST(RocprimDeviceHistogramEven, IncorrectInput)
{
    size_t temporary_storage_bytes = 0;
    int * d_input = nullptr;
    int * d_histogram = nullptr;
    ASSERT_THROW(
        rp::histogram_even(
            nullptr, temporary_storage_bytes,
            d_input, 123,
            d_histogram,
            1, 1, 2
        ),
        hc::runtime_exception
    );
}

TYPED_TEST(RocprimDeviceHistogramEven, Even)
{
    using sample_type = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type = typename TestFixture::params::level_type;
    constexpr unsigned int bins = TestFixture::params::bins;
    constexpr level_type lower_level = TestFixture::params::lower_level;
    constexpr level_type upper_level = TestFixture::params::upper_level;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<sample_type> input = get_random_samples<sample_type>(size, lower_level, upper_level);

        hc::array<sample_type> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<counter_type> d_histogram(bins, acc_view);

        // Calculate expected results on host
        std::vector<counter_type> histogram_expected(bins, 0);
        const level_type scale = (upper_level - lower_level) / bins;
        for(sample_type sample : input)
        {
            const level_type s = static_cast<level_type>(sample);
            if(s >= lower_level && s < upper_level)
            {
                const int bin = (s - lower_level) / scale;
                histogram_expected[bin]++;
            }
        }

        size_t temporary_storage_bytes = 0;
        rp::histogram_even(
            nullptr, temporary_storage_bytes,
            d_input.accelerator_pointer(), size,
            d_histogram.accelerator_pointer(),
            bins + 1, lower_level, upper_level,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rp::histogram_even(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_input.accelerator_pointer(), size,
            d_histogram.accelerator_pointer(),
            bins + 1, lower_level, upper_level,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<counter_type> histogram = d_histogram;

        for(size_t i = 0; i < bins; i++)
        {
            ASSERT_EQ(histogram[i], histogram_expected[i]);
        }
    }
}

template<
    class SampleType,
    unsigned int Bins,
    int StartLevel = 0,
    unsigned int MinBinWidth = 1,
    unsigned int MaxBinWidth = 10,
    class LevelType = SampleType,
    class CounterType = int
>
struct params2
{
    using sample_type = SampleType;
    static constexpr unsigned int bins = Bins;
    static constexpr int start_level = StartLevel;
    static constexpr unsigned int min_bin_length = MinBinWidth;
    static constexpr unsigned int max_bin_length = MaxBinWidth;
    using level_type = LevelType;
    using counter_type = CounterType;
};

template<class Params>
class RocprimDeviceHistogramRange : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params2<int, 10, 0, 1, 10>,
    params2<unsigned char, 5, 10, 10, 20>,
    params2<unsigned int, 10000, 0, 1, 100>,
    params2<unsigned short, 65536, 0, 1, 1, int>,
    params2<unsigned char, 256, 0, 1, 1, unsigned short>,

    params2<float, 456, -100, 1, 123>,
    params2<double, 3, 10000, 1000, 1000, double, unsigned int>
> Params2;

TYPED_TEST_CASE(RocprimDeviceHistogramRange, Params2);

TEST(RocprimDeviceHistogramRange, IncorrectInput)
{
    size_t temporary_storage_bytes = 0;
    int * d_input = nullptr;
    int * d_histogram = nullptr;
    int * d_levels = nullptr;
    ASSERT_THROW(
        rp::histogram_range(
            nullptr, temporary_storage_bytes,
            d_input, 123,
            d_histogram,
            1, d_levels
        ),
        hc::runtime_exception
    );
}

TYPED_TEST(RocprimDeviceHistogramRange, Range)
{
    using sample_type = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type = typename TestFixture::params::level_type;
    constexpr unsigned int bins = TestFixture::params::bins;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<unsigned int> bin_length_dis(
        TestFixture::params::min_bin_length,
        TestFixture::params::max_bin_length
    );

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<level_type> levels;
        level_type level = TestFixture::params::start_level;
        for(unsigned int bin = 0 ; bin < bins; bin++)
        {
            levels.push_back(level);
            level += bin_length_dis(gen);
        }
        levels.push_back(level);

        std::vector<sample_type> input = get_random_samples<sample_type>(size, levels[0], levels[bins]);

        hc::array<sample_type> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<level_type> d_levels(hc::extent<1>(bins + 1), levels.begin(), acc_view);
        hc::array<counter_type> d_histogram(bins, acc_view);

        // Calculate expected results on host
        std::vector<counter_type> histogram_expected(bins, 0);
        for(sample_type sample : input)
        {
            const level_type s = static_cast<level_type>(sample);
            if(s >= levels[0] && s < levels[bins])
            {
                const auto bin_iter = std::upper_bound(levels.begin(), levels.end(), s);
                histogram_expected[bin_iter - levels.begin() - 1]++;
            }
        }

        size_t temporary_storage_bytes = 0;
        rp::histogram_range(
            nullptr, temporary_storage_bytes,
            d_input.accelerator_pointer(), size,
            d_histogram.accelerator_pointer(),
            bins + 1, d_levels.accelerator_pointer(),
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rp::histogram_range(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_input.accelerator_pointer(), size,
            d_histogram.accelerator_pointer(),
            bins + 1, d_levels.accelerator_pointer(),
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<counter_type> histogram = d_histogram;

        for(size_t i = 0; i < bins; i++)
        {
            ASSERT_EQ(histogram[i], histogram_expected[i]);
        }
    }
}
