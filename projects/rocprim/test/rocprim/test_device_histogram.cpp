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

#include "../../common/utils.hpp"
#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_histogram.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <vector>

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
inline auto get_random_samples(size_t size, U min, U max, int seed_value) ->
    typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
{
    const long long min1 = static_cast<long long>(min);
    const long long max1 = static_cast<long long>(max);
    const long long d    = max1 - min1;
    return test_utils::get_random_data_wrapped<T>(
        size,
        static_cast<T>(
            std::max(min1 - d / 10, static_cast<long long>(rocprim::numeric_limits<T>::lowest()))),
        static_cast<T>(
            std::min(max1 + d / 10, static_cast<long long>(rocprim::numeric_limits<T>::max()))),
        seed_value);
}

template<class T, class U>
inline auto get_random_samples(size_t size, U min, U max, int seed_value) ->
    typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    const double min1 = static_cast<double>(min);
    const double max1 = static_cast<double>(max);
    const double d    = max1 - min1;
    return test_utils::get_random_data_wrapped<T>(
        size,
        static_cast<T>(
            std::max(min1 - d / 10, static_cast<double>(rocprim::numeric_limits<T>::lowest()))),
        static_cast<T>(
            std::min(max1 + d / 10, static_cast<double>(rocprim::numeric_limits<T>::max()))),
        seed_value);
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

template<class SampleType,
         unsigned int Bins,
         int          LowerLevel,
         int          UpperLevel,
         class LevelType          = SampleType,
         class CounterType        = int,
         class Config             = rocprim::default_config,
         bool UseIndirectIterator = false,
         bool UseGraphs           = false>
struct params1
{
    using sample_type                           = SampleType;
    static constexpr unsigned int bins          = Bins;
    static constexpr int          lower_level   = LowerLevel;
    static constexpr int          upper_level   = UpperLevel;
    using level_type                            = LevelType;
    using counter_type                          = CounterType;
    using config                                = Config;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
    static constexpr bool use_graphs            = UseGraphs;
};

template<class Params>
class RocprimDeviceHistogramEven : public ::testing::Test {
public:
    using params = Params;
};

using custom_config1 = rocprim::histogram_config<rocprim::kernel_config<128, 5>>;

using Params1
    = ::testing::Types<params1<int, 10, 0, 10>,
                       params1<float, 10, 0, 10>,
                       params1<float, 10, 0, 10, float, float>,
                       params1<int, 10, 0, 10, unsigned long, unsigned long>,
                       params1<rocprim::half, 10, 0, 10>,
                       params1<rocprim::bfloat16, 10, 0, 10>,
                       params1<int8_t, 10, 0, 10>,
                       params1<int, 128, 0, 256, int, int, custom_config1>,
                       params1<unsigned int, 12345, 10, 12355, short>,
                       params1<unsigned short, 65536, 0, 65536, int>,
                       params1<unsigned char, 10, 20, 240, unsigned char, unsigned int>,
                       params1<unsigned char, 256, 0, 256, short>,
                       params1<double, 10, 0, 1000, double, int>,
                       params1<double, 10, 0, 1000, double, double>,
                       params1<int, 123, 100, 5635, int>,
                       params1<double, 55, -123, +123, double, unsigned int, custom_config1>,
                       params1<int, 10, 0, 10, int, int, rocprim::default_config, true>,
                       params1<int, 10, 0, 10, int, int, rocprim::default_config, false, true>>;

TYPED_TEST_SUITE(RocprimDeviceHistogramEven, Params1);

void testHistogramEvenIncorrectInput()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0;

    size_t temporary_storage_bytes = 0;
    int * d_input = nullptr;
    int * d_histogram = nullptr;

    // This check happens on host so there is nothing to capture for hipGraph.
    hipError_t result = rocprim::histogram_even(nullptr,
                                                temporary_storage_bytes,
                                                d_input,
                                                123,
                                                d_histogram,
                                                1,
                                                1,
                                                2,
                                                stream);

    ASSERT_EQ(result, hipErrorInvalidValue);
}

TEST(RocprimDeviceHistogramEven, IncorrectInput)
{
    testHistogramEvenIncorrectInput();
}

template<class T>
using is_half = std::is_same<rocprim::half, typename std::remove_cv<T>::type>;

template<class T>
using is_bfloat16 = std::is_same<rocprim::bfloat16, typename std::remove_cv<T>::type>;

template<class T>
using convert_to_fundamental_t
    = std::conditional_t<is_half<T>::value || is_bfloat16<T>::value, float, T>;

template<class T>
inline auto convert_to_fundamental(T value)
{
    return static_cast<convert_to_fundamental_t<T>>(value);
}

TYPED_TEST(RocprimDeviceHistogramEven, Even)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using sample_type  = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type   = typename TestFixture::params::level_type;
    using config       = typename TestFixture::params::config;

    constexpr unsigned int bins        = TestFixture::params::bins;
    const level_type       lower_level = static_cast<level_type>(TestFixture::params::lower_level);
    const level_type       upper_level = static_cast<level_type>(TestFixture::params::upper_level);

    hipStream_t stream = 0;
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

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

        for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<sample_type> input = get_random_samples<sample_type>(size, lower_level, upper_level, seed_value);

            common::device_ptr<sample_type>  d_input(input);
            common::device_ptr<counter_type> d_histogram(bins);

            // Calculate expected results on host
            // native host types - used for low-precision floating-point types
            // to provide correct arithmetics (rocprim::half have no host-side operators on 5.7.1)
            using native_level_type               = convert_to_fundamental_t<level_type>;
            const native_level_type n_lower_level = static_cast<native_level_type>(lower_level);
            const native_level_type n_upper_level = static_cast<native_level_type>(upper_level);

            std::vector<counter_type> histogram_expected(bins, 0);
            const native_level_type   scale = (n_upper_level - n_lower_level) / bins;
            for(size_t row = 0; row < rows; row++)
            {
                for(size_t column = 0; column < columns; column++)
                {
                    const sample_type       sample   = input[row * row_stride + column];
                    const native_level_type n_sample = static_cast<native_level_type>(sample);
                    if(n_sample >= n_lower_level && n_sample < n_upper_level)
                    {
                        const unsigned int bin = (n_sample - n_lower_level) / scale;
                        histogram_expected[bin]++;
                    }
                }
            }

            auto input_it
                = test_utils::wrap_in_indirect_iterator<TestFixture::params::use_indirect_iterator>(
                    d_input.get());

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    if(rows == 1)
                    {
                        return rocprim::histogram_even<config>(temp_storage,
                                                               storage_bytes,
                                                               input_it,
                                                               static_cast<unsigned int>(columns),
                                                               d_histogram.get(),
                                                               bins + 1,
                                                               lower_level,
                                                               upper_level,
                                                               stream,
                                                               debug_synchronous);
                    }
                    else
                    {
                        return rocprim::histogram_even<config>(temp_storage,
                                                               storage_bytes,
                                                               input_it,
                                                               columns,
                                                               rows,
                                                               row_stride_bytes,
                                                               d_histogram.get(),
                                                               bins + 1,
                                                               lower_level,
                                                               upper_level,
                                                               stream,
                                                               debug_synchronous);
                    }
                },
                stream,
                TestFixture::params::use_graphs);

            const auto histogram = d_histogram.load();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(histogram, histogram_expected, bins));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class SampleType,
         unsigned int Bins,
         int          StartLevel  = 0,
         unsigned int MinBinWidth = 1,
         unsigned int MaxBinWidth = 10,
         class LevelType          = SampleType,
         class CounterType        = int,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false>
struct params2
{
    using sample_type                            = SampleType;
    static constexpr unsigned int bins           = Bins;
    static constexpr int          start_level    = StartLevel;
    static constexpr unsigned int min_bin_length = MinBinWidth;
    static constexpr unsigned int max_bin_length = MaxBinWidth;
    using level_type                             = LevelType;
    using counter_type                           = CounterType;
    using config                                 = Config;
    static constexpr bool         use_graphs     = UseGraphs;
};

template<class Params>
class RocprimDeviceHistogramRange : public ::testing::Test {
public:
    using params = Params;
};

using custom_config2 = rocprim::histogram_config<rocprim::kernel_config<256, 2>>;

using Params2 = ::testing::Types<
    params2<int, 10, 0, 1, 10>,
    params2<unsigned char, 5, 10, 10, 20>,
    params2<unsigned int, 10000, 0, 1, 100, unsigned int, unsigned long long, custom_config2>,
    params2<unsigned short, 65536, 0, 1, 1, int>,
    params2<unsigned char, 256, 0, 1, 1, unsigned short, int, custom_config2>,

    params2<float, 456, -100, 1, 123>,
    params2<double, 3, 10000, 1000, 1000, double, unsigned int>,
    params2<int, 10, 0, 1, 10, int, int, rocprim::default_config, true>>;

TYPED_TEST_SUITE(RocprimDeviceHistogramRange, Params2);

void testHistogramRangeIncorrectInput()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0;

    size_t temporary_storage_bytes = 0;
    int * d_input = nullptr;
    int * d_histogram = nullptr;
    int * d_levels = nullptr;

    // This check happens on host so there is nothing to capture for hipGraph.
    hipError_t result = rocprim::histogram_range(
                                                 nullptr, temporary_storage_bytes,
                                                 d_input, 123,
                                                 d_histogram,
                                                 1, d_levels, stream
                                                 );

    ASSERT_EQ(result, hipErrorInvalidValue);
}

TEST(RocprimDeviceHistogramRange, RangeIncorrectInput)
{
    testHistogramRangeIncorrectInput();
}

TYPED_TEST(RocprimDeviceHistogramRange, Range)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using sample_type  = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type   = typename TestFixture::params::level_type;
    using config       = typename TestFixture::params::config;

    constexpr unsigned int bins = TestFixture::params::bins;

    hipStream_t stream = 0; // default
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    common::uniform_int_distribution<unsigned int> bin_length_dis(
        TestFixture::params::min_bin_length,
        TestFixture::params::max_bin_length);

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

        for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<sample_type> input = get_random_samples<sample_type>(size, levels[0], levels[bins], seed_value);

            common::device_ptr<sample_type>  d_input(input);
            common::device_ptr<level_type>   d_levels(levels);
            common::device_ptr<counter_type> d_histogram(bins);

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

            rocprim::transform_iterator<sample_type*, transform_op<sample_type>, sample_type>
                d_input2(d_input.get(), transform_op<sample_type>());

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    if(rows == 1)
                    {
                        return rocprim::histogram_range<config>(temp_storage,
                                                                storage_bytes,
                                                                d_input2,
                                                                columns,
                                                                d_histogram.get(),
                                                                bins + 1,
                                                                d_levels.get(),
                                                                stream,
                                                                debug_synchronous);
                    }
                    else
                    {
                        return rocprim::histogram_range<config>(temp_storage,
                                                                storage_bytes,
                                                                d_input2,
                                                                columns,
                                                                rows,
                                                                row_stride_bytes,
                                                                d_histogram.get(),
                                                                bins + 1,
                                                                d_levels.get(),
                                                                stream,
                                                                debug_synchronous);
                    }
                },
                stream,
                TestFixture::params::use_graphs);

            const auto histogram = d_histogram.load();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(histogram, histogram_expected, bins));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class SampleType,
         unsigned int Channels,
         unsigned int ActiveChannels,
         unsigned int Bins,
         int          LowerLevel,
         int          UpperLevel,
         class LevelType   = SampleType,
         class CounterType = int,
         class Config      = rocprim::default_config,
         bool  UseGraphs   = false>
struct params3
{
    using sample_type                             = SampleType;
    static constexpr unsigned int channels        = Channels;
    static constexpr unsigned int active_channels = ActiveChannels;
    static constexpr unsigned int bins            = Bins;
    static constexpr int          lower_level     = LowerLevel;
    static constexpr int          upper_level     = UpperLevel;
    using level_type                              = LevelType;
    using counter_type                            = CounterType;
    using config                                  = Config;
    static constexpr bool         use_graphs      = UseGraphs;
};

template<class Params>
class RocprimDeviceHistogramMultiEven : public ::testing::Test {
public:
    using params = Params;
};

using custom_config3 = rocprim::histogram_config<rocprim::kernel_config<192, 1>>;

using Params3 = ::testing::Types<
    params3<int, 4, 3, 2000, 0, 2000>,
    params3<int, 2, 1, 10, 0, 10, int, int, custom_config3>,
    params3<int, 3, 3, 128, 0, 256>,
    params3<unsigned int, 1, 1, 12345, 10, 12355, short>,
    params3<unsigned short, 4, 4, 65536, 0, 65536, int>,
    params3<unsigned char, 3, 1, 10, 20, 240, unsigned char, unsigned int>,
    params3<unsigned char, 2, 2, 256, 0, 256, short>,

    params3<double, 4, 2, 10, 0, 1000, double, int>,
    params3<int, 3, 2, 123, 100, 5635, int>,
    params3<double, 4, 3, 55, -123, +123, double, unsigned long long, custom_config3>,
    params3<int, 4, 3, 2000, 0, 2000, int, int, rocprim::default_config, true>>;

TYPED_TEST_SUITE(RocprimDeviceHistogramMultiEven, Params3);

TYPED_TEST(RocprimDeviceHistogramMultiEven, MultiEven)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using sample_type  = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type   = typename TestFixture::params::level_type;
    using config       = typename TestFixture::params::config;

    constexpr unsigned int channels        = TestFixture::params::channels;
    constexpr unsigned int active_channels = TestFixture::params::active_channels;

    unsigned int bins[active_channels];
    unsigned int num_levels[active_channels];
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
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

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

        for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<sample_type> input(size);
            for(unsigned int channel = 0; channel < channels; channel++)
            {
                const size_t gen_columns = (row_stride + channels - 1) / channels;
                const size_t gen_size = rows * gen_columns;

                std::vector<sample_type> channel_input;
                if(channel < active_channels)
                {
                    channel_input = get_random_samples<sample_type>(gen_size, lower_level[channel], upper_level[channel], seed_value);
                }
                else
                {
                    channel_input = get_random_samples<sample_type>(gen_size, lower_level[0], upper_level[0], seed_value);
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

            common::device_ptr<sample_type>     d_input(input);
            counter_type*                       d_histogram[active_channels];
            for(unsigned int channel = 0; channel < active_channels; channel++)
            {
                HIP_CHECK(common::hipMallocHelper(&d_histogram[channel],
                                                  bins[channel] * sizeof(counter_type)));
            }

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

            rocprim::transform_iterator<sample_type*, transform_op<sample_type>, sample_type>
                d_input2(d_input.get(), transform_op<sample_type>());

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    if(rows == 1)
                    {
                        return rocprim::multi_histogram_even<channels, active_channels, config>(
                            temp_storage,
                            storage_bytes,
                            d_input2,
                            columns,
                            d_histogram,
                            num_levels,
                            lower_level,
                            upper_level,
                            stream,
                            debug_synchronous);
                    }
                    else
                    {
                        return rocprim::multi_histogram_even<channels, active_channels, config>(
                            temp_storage,
                            storage_bytes,
                            d_input2,
                            columns,
                            rows,
                            row_stride_bytes,
                            d_histogram,
                            num_levels,
                            lower_level,
                            upper_level,
                            stream,
                            debug_synchronous);
                    }
                },
                stream,
                TestFixture::params::use_graphs);

            std::vector<counter_type> histogram[active_channels];
            for(unsigned int channel = 0; channel < active_channels; channel++)
            {
                histogram[channel] = std::vector<counter_type>(bins[channel]);
                HIP_CHECK(hipMemcpy(histogram[channel].data(),
                                    d_histogram[channel],
                                    bins[channel] * sizeof(counter_type),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipFree(d_histogram[channel]));
            }

            for(unsigned int channel = 0; channel < active_channels; channel++)
            {
                SCOPED_TRACE(testing::Message() << "with channel = " << channel);
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(histogram[channel],
                                                              histogram_expected[channel],
                                                              bins[channel]));
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class SampleType,
         unsigned int Channels,
         unsigned int ActiveChannels,
         unsigned int Bins,
         int          StartLevel  = 0,
         unsigned int MinBinWidth = 1,
         unsigned int MaxBinWidth = 10,
         class LevelType          = SampleType,
         class CounterType        = int,
         class Config             = rocprim::default_config,
         bool  UseGraphs          = false>
struct params4
{
    using sample_type                             = SampleType;
    static constexpr unsigned int channels        = Channels;
    static constexpr unsigned int active_channels = ActiveChannels;
    static constexpr unsigned int bins            = Bins;
    static constexpr int          start_level     = StartLevel;
    static constexpr unsigned int min_bin_length  = MinBinWidth;
    static constexpr unsigned int max_bin_length  = MaxBinWidth;
    using level_type                              = LevelType;
    using counter_type                            = CounterType;
    using config                                  = Config;
    static constexpr bool         use_graphs      = UseGraphs;
};

template<class Params>
class RocprimDeviceHistogramMultiRange : public ::testing::Test {
public:
    using params = Params;
};

using custom_config4 = rocprim::histogram_config<rocprim::kernel_config<256, 7>>;

using Params4 = ::testing::Types<
    params4<int, 4, 3, 10, 0, 1, 10>,
    params4<unsigned char, 2, 2, 5, 10, 10, 20>,
    params4<unsigned int, 1, 1, 10000, 0, 1, 100>,
    params4<unsigned short, 4, 4, 65536, 0, 1, 1, int>,
    params4<unsigned char, 3, 2, 256, 0, 1, 1, unsigned short, int, custom_config4>,

    params4<float, 4, 2, 456, -100, 1, 123, float, int, custom_config4>,
    params4<double, 3, 1, 3, 10000, 1000, 1000, double, unsigned int>,
    params4<int, 3, 2, 10, 0, 1, 10, int, int, rocprim::default_config, true>>;

TYPED_TEST_SUITE(RocprimDeviceHistogramMultiRange, Params4);

TYPED_TEST(RocprimDeviceHistogramMultiRange, MultiRange)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using sample_type  = typename TestFixture::params::sample_type;
    using counter_type = typename TestFixture::params::counter_type;
    using level_type   = typename TestFixture::params::level_type;
    using config       = typename TestFixture::params::config;

    constexpr unsigned int channels        = TestFixture::params::channels;
    constexpr unsigned int active_channels = TestFixture::params::active_channels;

    hipStream_t stream = 0;
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    unsigned int                                   bins[active_channels];
    unsigned int                                   num_levels[active_channels];
    common::uniform_int_distribution<unsigned int> bin_length_dis[active_channels];
    for(unsigned int channel = 0; channel < active_channels; channel++)
    {
        // Use different ranges for different channels
        bins[channel]       = TestFixture::params::bins + channel;
        num_levels[channel] = bins[channel] + 1;
        bin_length_dis[channel]
            = common::uniform_int_distribution<unsigned int>(TestFixture::params::min_bin_length,
                                                             TestFixture::params::max_bin_length);
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

        for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

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
                        gen_size, levels[channel][0], levels[channel][bins[channel]], seed_value
                    );
                }
                else
                {
                    channel_input = get_random_samples<sample_type>(gen_size, levels[0][0], levels[0][bins[0]], seed_value);
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

            common::device_ptr<sample_type>     d_input(input);
            level_type * d_levels[active_channels];
            counter_type*                       d_histogram[active_channels];
            for(unsigned int channel = 0; channel < active_channels; channel++)
            {
                HIP_CHECK(common::hipMallocHelper(&d_levels[channel],
                                                  num_levels[channel] * sizeof(level_type)));
                HIP_CHECK(common::hipMallocHelper(&d_histogram[channel],
                                                  bins[channel] * sizeof(counter_type)));
            }
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

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    if(rows == 1)
                    {
                        return rocprim::multi_histogram_range<channels, active_channels, config>(
                            temp_storage,
                            storage_bytes,
                            d_input.get(),
                            columns,
                            d_histogram,
                            num_levels,
                            d_levels,
                            stream,
                            debug_synchronous);
                    }
                    else
                    {
                        return rocprim::multi_histogram_range<channels, active_channels, config>(
                            temp_storage,
                            storage_bytes,
                            d_input.get(),
                            columns,
                            rows,
                            row_stride_bytes,
                            d_histogram,
                            num_levels,
                            d_levels,
                            stream,
                            debug_synchronous);
                    }
                },
                stream,
                TestFixture::params::use_graphs);

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

            for(unsigned int channel = 0; channel < active_channels; channel++)
            {
                SCOPED_TRACE(testing::Message() << "with channel = " << channel);
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(histogram[channel],
                                                              histogram_expected[channel],
                                                              bins[channel]));
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}
