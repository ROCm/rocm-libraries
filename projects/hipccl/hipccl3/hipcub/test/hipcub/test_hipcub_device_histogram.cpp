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

// CUB's implementation of DeviceHistogram has unused parameters,
// disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVCC__
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// hipCUB API
#include <hipcub/hipcub.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

// rows, columns, (row_stride - columns * Channels)
std::vector<std::tuple<size_t, size_t, size_t>> get_dims()
{
    std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        // Empty
        std::make_tuple(0, 0, 0),
        std::make_tuple(1, 0, 0),
        std::make_tuple(0, 1, 0),
        // Linear
        std::make_tuple(1, 1, 0),
        std::make_tuple(1, 53, 0),
        std::make_tuple(1, 5096, 0),
        std::make_tuple(1, 34567, 0),
        std::make_tuple(1, (1 << 18) - 1220, 0),
        // Strided
        std::make_tuple(2, 1, 0),
        std::make_tuple(10, 10, 11),
        std::make_tuple(111, 111, 111),
        std::make_tuple(128, 1289, 0),
        std::make_tuple(12, 1000, 24),
        std::make_tuple(123, 3000, 121),
        std::make_tuple(1024, 512, 0),
        std::make_tuple(2345, 49, 2),
        std::make_tuple(17867, 41, 13),
    };
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

// Does nothing, used for testing iterators (not raw pointers) as samples input
template<class T>
struct transform_op
{
    __host__ __device__ inline
    T operator()(T x) const
    {
        return x * 1;
    }
};

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
class HipcubDeviceHistogramEven : public ::testing::Test {
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
    params1<int, 123, 100, 5635, int>,
    params1<double, 55, -123, +123, double>
> Params1;

TYPED_TEST_CASE(HipcubDeviceHistogramEven, Params1);

TYPED_TEST(HipcubDeviceHistogramEven, Even)
{
    using sample_type = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type = typename TestFixture::params::level_type;
    constexpr unsigned int bins = TestFixture::params::bins;
    constexpr level_type lower_level = TestFixture::params::lower_level;
    constexpr level_type upper_level = TestFixture::params::upper_level;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    for(auto dim : get_dims())
    {
        SCOPED_TRACE(
            testing::Message() << "with dim = {" <<
            std::get<0>(dim) << ", " << std::get<1>(dim) << ", " << std::get<2>(dim) << "}"
        );

        const size_t rows = std::get<0>(dim);
        const size_t columns = std::get<1>(dim);
        const size_t row_stride = columns + std::get<2>(dim);

        const size_t row_stride_bytes = row_stride * sizeof(sample_type);
        const size_t size = std::max<size_t>(1, rows * row_stride);

        // Generate data
        std::vector<sample_type> input = get_random_samples<sample_type>(size, lower_level, upper_level);

        sample_type * d_input;
        counter_type * d_histogram;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(sample_type)));
        HIP_CHECK(hipMalloc(&d_histogram, bins * sizeof(counter_type)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(sample_type),
                hipMemcpyHostToDevice
            )
        );

        // Calculate expected results on host
        std::vector<counter_type> histogram_expected(bins, 0);
        const level_type scale = (upper_level - lower_level) / bins;
        for(size_t row = 0; row < rows; row++)
        {
            for(size_t column = 0; column < columns; column++)
            {
                const sample_type sample = input[row * row_stride + column];
                const level_type s = static_cast<level_type>(sample);
                if(s >= lower_level && s < upper_level)
                {
                    const int bin = (s - lower_level) / scale;
                    histogram_expected[bin]++;
                }
            }
        }

        size_t temporary_storage_bytes = 0;
        if(rows == 1)
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramEven(
                    nullptr, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, lower_level, upper_level,
                    int(columns),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramEven(
                    nullptr, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, lower_level, upper_level,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            );
        }

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(rows == 1)
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramEven(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, lower_level, upper_level,
                    int(columns),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramEven(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, lower_level, upper_level,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            );
        }

        std::vector<counter_type> histogram(bins);
        HIP_CHECK(
            hipMemcpy(
                histogram.data(), d_histogram,
                bins * sizeof(counter_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_histogram));

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
class HipcubDeviceHistogramRange : public ::testing::Test {
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

TYPED_TEST_CASE(HipcubDeviceHistogramRange, Params2);

TYPED_TEST(HipcubDeviceHistogramRange, Range)
{
    using sample_type = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type = typename TestFixture::params::level_type;
    constexpr unsigned int bins = TestFixture::params::bins;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<unsigned int> bin_length_dis(
        TestFixture::params::min_bin_length,
        TestFixture::params::max_bin_length
    );

    for(auto dim : get_dims())
    {
        SCOPED_TRACE(
            testing::Message() << "with dim = {" <<
            std::get<0>(dim) << ", " << std::get<1>(dim) << ", " << std::get<2>(dim) << "}"
        );

        const size_t rows = std::get<0>(dim);
        const size_t columns = std::get<1>(dim);
        const size_t row_stride = columns + std::get<2>(dim);

        const size_t row_stride_bytes = row_stride * sizeof(sample_type);
        const size_t size = std::max<size_t>(1, rows * row_stride);

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

        sample_type * d_input;
        level_type * d_levels;
        counter_type * d_histogram;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(sample_type)));
        HIP_CHECK(hipMalloc(&d_levels, (bins + 1) * sizeof(level_type)));
        HIP_CHECK(hipMalloc(&d_histogram, bins * sizeof(counter_type)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(sample_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_levels, levels.data(),
                (bins + 1) * sizeof(level_type),
                hipMemcpyHostToDevice
            )
        );

        // Calculate expected results on host
        std::vector<counter_type> histogram_expected(bins, 0);
        for(size_t row = 0; row < rows; row++)
        {
            for(size_t column = 0; column < columns; column++)
            {
                const sample_type sample = input[row * row_stride + column];
                const level_type s = static_cast<level_type>(sample);
                if(s >= levels[0] && s < levels[bins])
                {
                    const auto bin_iter = std::upper_bound(levels.begin(), levels.end(), s);
                    histogram_expected[bin_iter - levels.begin() - 1]++;
                }
            }
        }

        size_t temporary_storage_bytes = 0;
        if(rows == 1)
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramRange(
                    nullptr, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, d_levels,
                    int(columns),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramRange(
                    nullptr, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, d_levels,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            );
        }

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(rows == 1)
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramRange(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, d_levels,
                    int(columns),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceHistogram::HistogramRange(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input, d_histogram,
                    bins + 1, d_levels,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            );
        }

        std::vector<counter_type> histogram(bins);
        HIP_CHECK(
            hipMemcpy(
                histogram.data(), d_histogram,
                bins * sizeof(counter_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_levels));
        HIP_CHECK(hipFree(d_histogram));

        for(size_t i = 0; i < bins; i++)
        {
            ASSERT_EQ(histogram[i], histogram_expected[i]);
        }
    }
}

template<
    class SampleType,
    unsigned int Channels,
    unsigned int ActiveChannels,
    unsigned int Bins,
    int LowerLevel,
    int UpperLevel,
    class LevelType = SampleType,
    class CounterType = int
>
struct params3
{
    using sample_type = SampleType;
    static constexpr unsigned int channels = Channels;
    static constexpr unsigned int active_channels = ActiveChannels;
    static constexpr unsigned int bins = Bins;
    static constexpr int lower_level = LowerLevel;
    static constexpr int upper_level = UpperLevel;
    using level_type = LevelType;
    using counter_type = CounterType;
};

template<class Params>
class HipcubDeviceHistogramMultiEven : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params3<int, 4, 3, 2000, 0, 2000>,
    params3<int, 2, 1, 10, 0, 10>,
    params3<int, 3, 3, 128, 0, 256>,
    params3<unsigned int, 1, 1, 12345, 10, 12355, short>,
    params3<unsigned short, 4, 4, 65536, 0, 65536, int>,
    params3<unsigned char, 3, 1, 10, 20, 240, unsigned char, unsigned int>,
    params3<unsigned char, 2, 2, 256, 0, 256, short>,

    params3<double, 4, 2, 10, 0, 1000, double, int>,
    params3<int, 3, 2, 123, 100, 5635, int>,
    params3<double, 4, 3, 55, -123, +123, double>
> Params3;

TYPED_TEST_CASE(HipcubDeviceHistogramMultiEven, Params3);

TYPED_TEST(HipcubDeviceHistogramMultiEven, MultiEven)
{
    using sample_type = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type = typename TestFixture::params::level_type;
    constexpr unsigned int channels = TestFixture::params::channels;
    constexpr unsigned int active_channels = TestFixture::params::active_channels;

    unsigned int bins[active_channels];
    int num_levels[active_channels];
    level_type lower_level[active_channels];
    level_type upper_level[active_channels];
    for(unsigned int channel = 0; channel < active_channels; channel++)
    {
        // Use different ranges for different channels
        constexpr level_type d = TestFixture::params::upper_level - TestFixture::params::lower_level;
        const level_type scale = d / TestFixture::params::bins;
        lower_level[channel] = TestFixture::params::lower_level + channel * d / 9;
        upper_level[channel] = TestFixture::params::upper_level - channel * d / 7;
        bins[channel] = (upper_level[channel] - lower_level[channel]) / scale;
        upper_level[channel] = lower_level[channel] + bins[channel] * scale;
        num_levels[channel] = bins[channel] + 1;
    }

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    for(auto dim : get_dims())
    {
        SCOPED_TRACE(
            testing::Message() << "with dim = {" <<
            std::get<0>(dim) << ", " << std::get<1>(dim) << ", " << std::get<2>(dim) << "}"
        );

        const size_t rows = std::get<0>(dim);
        const size_t columns = std::get<1>(dim);
        const size_t row_stride = columns * channels + std::get<2>(dim);

        const size_t row_stride_bytes = row_stride * sizeof(sample_type);
        const size_t size = std::max<size_t>(1, rows * row_stride);

        // Generate data
        std::vector<sample_type> input(size);
        for(unsigned int channel = 0; channel < channels; channel++)
        {
            const size_t gen_columns = (row_stride + channels - 1) / channels;
            const size_t gen_size = rows * gen_columns;

            std::vector<sample_type> channel_input;
            if(channel < active_channels)
            {
                channel_input = get_random_samples<sample_type>(gen_size, lower_level[channel], upper_level[channel]);
            }
            else
            {
                channel_input = get_random_samples<sample_type>(gen_size, lower_level[0], upper_level[0]);
            }
            // Interleave values
            for(size_t row = 0; row < rows; row++)
            {
                for(size_t column = 0; column < gen_columns; column++)
                {
                    const size_t index = column * channels + channel;
                    if(index < row_stride)
                    {
                        input[row * row_stride + index] = channel_input[row * gen_columns + column];
                    }
                }
            }
        }

        sample_type * d_input;
        counter_type * d_histogram[active_channels];
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(sample_type)));
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            HIP_CHECK(hipMalloc(&d_histogram[channel], bins[channel] * sizeof(counter_type)));
        }
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(sample_type),
                hipMemcpyHostToDevice
            )
        );

        // Calculate expected results on host
        std::vector<counter_type> histogram_expected[active_channels];
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            histogram_expected[channel] = std::vector<counter_type>(bins[channel], 0);
            const level_type scale = (upper_level[channel] - lower_level[channel]) / bins[channel];

            for(size_t row = 0; row < rows; row++)
            {
                for(size_t column = 0; column < columns; column++)
                {
                    const sample_type sample = input[row * row_stride + column * channels + channel];
                    const level_type s = static_cast<level_type>(sample);
                    if(s >= lower_level[channel] && s < upper_level[channel])
                    {
                        const int bin = (s - lower_level[channel]) / scale;
                        histogram_expected[channel][bin]++;
                    }
                }
            }
        }

        hipcub::TransformInputIterator<sample_type, transform_op<sample_type>, sample_type *> d_input2(
            d_input,
            transform_op<sample_type>()
        );

        size_t temporary_storage_bytes = 0;
        if(rows == 1)
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramEven<channels, active_channels>(
                    nullptr, temporary_storage_bytes,
                    d_input2,
                    d_histogram,
                    num_levels, lower_level, upper_level,
                    int(columns),
                    stream, debug_synchronous
                )
            ));
        }
        else
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramEven<channels, active_channels>(
                    nullptr, temporary_storage_bytes,
                    d_input2,
                    d_histogram,
                    num_levels, lower_level, upper_level,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            ));
        }

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(rows == 1)
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramEven<channels, active_channels>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input2,
                    d_histogram,
                    num_levels, lower_level, upper_level,
                    int(columns),
                    stream, debug_synchronous
                )
            ));
        }
        else
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramEven<channels, active_channels>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input2,
                    d_histogram,
                    num_levels, lower_level, upper_level,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            ));
        }

        std::vector<counter_type> histogram[active_channels];
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            histogram[channel] = std::vector<counter_type>(bins[channel]);
            HIP_CHECK(
                hipMemcpy(
                    histogram[channel].data(), d_histogram[channel],
                    bins[channel] * sizeof(counter_type),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipFree(d_histogram[channel]));
        }

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_input));

        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            SCOPED_TRACE(testing::Message() << "with channel = " << channel);

            for(size_t i = 0; i < bins[channel]; i++)
            {
                ASSERT_EQ(histogram[channel][i], histogram_expected[channel][i]);
            }
        }
    }
}

template<
    class SampleType,
    unsigned int Channels,
    unsigned int ActiveChannels,
    unsigned int Bins,
    int StartLevel = 0,
    unsigned int MinBinWidth = 1,
    unsigned int MaxBinWidth = 10,
    class LevelType = SampleType,
    class CounterType = int
>
struct params4
{
    using sample_type = SampleType;
    static constexpr unsigned int channels = Channels;
    static constexpr unsigned int active_channels = ActiveChannels;
    static constexpr unsigned int bins = Bins;
    static constexpr int start_level = StartLevel;
    static constexpr unsigned int min_bin_length = MinBinWidth;
    static constexpr unsigned int max_bin_length = MaxBinWidth;
    using level_type = LevelType;
    using counter_type = CounterType;
};

template<class Params>
class HipcubDeviceHistogramMultiRange : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params4<int, 4, 3, 10, 0, 1, 10>,
    params4<unsigned char, 2, 2, 5, 10, 10, 20>,
    params4<unsigned int, 1, 1, 10000, 0, 1, 100>,
    params4<unsigned short, 4, 4, 65536, 0, 1, 1, int>,
    params4<unsigned char, 3, 2, 256, 0, 1, 1, unsigned short>,

    params4<float, 4, 2, 456, -100, 1, 123>,
    params4<double, 3, 1, 3, 10000, 1000, 1000, double, unsigned int>
> Params4;

TYPED_TEST_CASE(HipcubDeviceHistogramMultiRange, Params4);

TYPED_TEST(HipcubDeviceHistogramMultiRange, MultiRange)
{
    using sample_type = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type = typename TestFixture::params::level_type;
    constexpr unsigned int channels = TestFixture::params::channels;
    constexpr unsigned int active_channels = TestFixture::params::active_channels;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    unsigned int bins[active_channels];
    int num_levels[active_channels];
    std::uniform_int_distribution<unsigned int> bin_length_dis[active_channels];
    for(unsigned int channel = 0; channel < active_channels; channel++)
    {
        // Use different ranges for different channels
        bins[channel] = TestFixture::params::bins + channel;
        num_levels[channel] = bins[channel] + 1;
        bin_length_dis[channel] = std::uniform_int_distribution<unsigned int>(
            TestFixture::params::min_bin_length,
            TestFixture::params::max_bin_length
        );
    }

    for(auto dim : get_dims())
    {
        SCOPED_TRACE(
            testing::Message() << "with dim = {" <<
            std::get<0>(dim) << ", " << std::get<1>(dim) << ", " << std::get<2>(dim) << "}"
        );

        const size_t rows = std::get<0>(dim);
        const size_t columns = std::get<1>(dim);
        const size_t row_stride = columns * channels + std::get<2>(dim);

        const size_t row_stride_bytes = row_stride * sizeof(sample_type);
        const size_t size = std::max<size_t>(1, rows * row_stride);

        // Generate data
        std::vector<level_type> levels[active_channels];
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            level_type level = TestFixture::params::start_level;
            for(unsigned int bin = 0 ; bin < bins[channel]; bin++)
            {
                levels[channel].push_back(level);
                level += bin_length_dis[channel](gen);
            }
            levels[channel].push_back(level);
        }

        std::vector<sample_type> input(size);
        for(unsigned int channel = 0; channel < channels; channel++)
        {
            const size_t gen_columns = (row_stride + channels - 1) / channels;
            const size_t gen_size = rows * gen_columns;

            std::vector<sample_type> channel_input;
            if(channel < active_channels)
            {
                channel_input = get_random_samples<sample_type>(
                    gen_size, levels[channel][0], levels[channel][bins[channel]]
                );
            }
            else
            {
                channel_input = get_random_samples<sample_type>(gen_size, levels[0][0], levels[0][bins[0]]);
            }
            // Interleave values
            for(size_t row = 0; row < rows; row++)
            {
                for(size_t column = 0; column < gen_columns; column++)
                {
                    const size_t index = column * channels + channel;
                    if(index < row_stride)
                    {
                        input[row * row_stride + index] = channel_input[row * gen_columns + column];
                    }
                }
            }
        }

        sample_type * d_input;
        level_type * d_levels[active_channels];
        counter_type * d_histogram[active_channels];
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(sample_type)));
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            HIP_CHECK(hipMalloc(&d_levels[channel], num_levels[channel] * sizeof(level_type)));
            HIP_CHECK(hipMalloc(&d_histogram[channel], bins[channel] * sizeof(counter_type)));
        }
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(sample_type),
                hipMemcpyHostToDevice
            )
        );
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            HIP_CHECK(
                hipMemcpy(
                    d_levels[channel], levels[channel].data(),
                    num_levels[channel] * sizeof(level_type),
                    hipMemcpyHostToDevice
                )
            );
        }

        // Calculate expected results on host
        std::vector<counter_type> histogram_expected[active_channels];
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            histogram_expected[channel] = std::vector<counter_type>(bins[channel], 0);

            for(size_t row = 0; row < rows; row++)
            {
                for(size_t column = 0; column < columns; column++)
                {
                    const sample_type sample = input[row * row_stride + column * channels + channel];
                    const level_type s = static_cast<level_type>(sample);
                    if(s >= levels[channel][0] && s < levels[channel][bins[channel]])
                    {
                        const auto bin_iter = std::upper_bound(levels[channel].begin(), levels[channel].end(), s);
                        const int bin = bin_iter - levels[channel].begin() - 1;
                        histogram_expected[channel][bin]++;
                    }
                }
            }
        }

        hipcub::TransformInputIterator<sample_type, transform_op<sample_type>, sample_type *> d_input2(
            d_input,
            transform_op<sample_type>()
        );

        size_t temporary_storage_bytes = 0;
        if(rows == 1)
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramRange<channels, active_channels>(
                    nullptr, temporary_storage_bytes,
                    d_input2, d_histogram,
                    num_levels, d_levels,
                    int(columns),
                    stream, debug_synchronous
                )
            ));
        }
        else
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramRange<channels, active_channels>(
                    nullptr, temporary_storage_bytes,
                    d_input2, d_histogram,
                    num_levels, d_levels,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            ));
        }

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        if(rows == 1)
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramRange<channels, active_channels>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input2, d_histogram,
                    num_levels, d_levels,
                    int(columns),
                    stream, debug_synchronous
                )
            ));
        }
        else
        {
            HIP_CHECK((
                hipcub::DeviceHistogram::MultiHistogramRange<channels, active_channels>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_input2, d_histogram,
                    num_levels, d_levels,
                    int(columns), int(rows), row_stride_bytes,
                    stream, debug_synchronous
                )
            ));
        }

        std::vector<counter_type> histogram[active_channels];
        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            histogram[channel] = std::vector<counter_type>(bins[channel]);
            HIP_CHECK(
                hipMemcpy(
                    histogram[channel].data(), d_histogram[channel],
                    bins[channel] * sizeof(counter_type),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipFree(d_levels[channel]));
            HIP_CHECK(hipFree(d_histogram[channel]));
        }

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_input));

        for(unsigned int channel = 0; channel < active_channels; channel++)
        {
            SCOPED_TRACE(testing::Message() << "with channel = " << channel);

            for(size_t i = 0; i < bins[channel]; i++)
            {
                ASSERT_EQ(histogram[channel][i], histogram_expected[channel][i]);
            }
        }
    }
}
