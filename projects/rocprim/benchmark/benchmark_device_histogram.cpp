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

#include "benchmark_device_histogram.parallel.hpp"
#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_histogram.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

int get_entropy_percents(int entropy_reduction)
{
    switch(entropy_reduction)
    {
        case 0: return 100;
        case 1: return 81;
        case 2: return 54;
        case 3: return 33;
        case 4: return 20;
        default: return 0;
    }
}

template<typename T>
void run_even_benchmark(benchmark_utils::state&& state,
                        size_t                   bins,
                        size_t                   scale,
                        int                      entropy_reduction)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;

    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    const level_type lower_level = 0;
    const level_type upper_level = bins * scale;

    // Generate data
    std::vector<T> input = generate<T>(size, entropy_reduction, lower_level, upper_level);

    common::device_ptr<T>            d_input(input);
    common::device_ptr<counter_type> d_histogram(bins);

    size_t temporary_storage_bytes = 0;
    HIP_CHECK(rocprim::histogram_even(nullptr,
                                      temporary_storage_bytes,
                                      d_input.get(),
                                      size,
                                      d_histogram.get(),
                                      bins + 1,
                                      lower_level,
                                      upper_level,
                                      stream,
                                      false));

    common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            HIP_CHECK(rocprim::histogram_even(d_temporary_storage.get(),
                                              temporary_storage_bytes,
                                              d_input.get(),
                                              size,
                                              d_histogram.get(),
                                              bins + 1,
                                              lower_level,
                                              upper_level,
                                              stream,
                                              false));
        });

    state.set_throughput(size, sizeof(T));
}

template<typename T, unsigned int Channels, unsigned int ActiveChannels>
void run_multi_even_benchmark(benchmark_utils::state&& state,
                              size_t                   bins,
                              size_t                   scale,
                              int                      entropy_reduction)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;

    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    unsigned int num_levels[ActiveChannels];
    level_type   lower_level[ActiveChannels];
    level_type   upper_level[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        lower_level[channel] = 0;
        upper_level[channel] = bins * scale;
        num_levels[channel]  = bins + 1;
    }

    // Generate data
    std::vector<T> input
        = generate<T>(size * Channels, entropy_reduction, lower_level[0], upper_level[0]);

    common::device_ptr<T> d_input(input);
    counter_type*         d_histogram[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        HIP_CHECK(hipMalloc(&d_histogram[channel], bins * sizeof(counter_type)));
    }

    size_t temporary_storage_bytes = 0;
    HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels>(nullptr,
                                                                       temporary_storage_bytes,
                                                                       d_input.get(),
                                                                       size,
                                                                       d_histogram,
                                                                       num_levels,
                                                                       lower_level,
                                                                       upper_level,
                                                                       stream,
                                                                       false)));

    common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            HIP_CHECK(
                (rocprim::multi_histogram_even<Channels, ActiveChannels>(d_temporary_storage.get(),
                                                                         temporary_storage_bytes,
                                                                         d_input.get(),
                                                                         size,
                                                                         d_histogram,
                                                                         num_levels,
                                                                         lower_level,
                                                                         upper_level,
                                                                         stream,
                                                                         false)));
        });

    state.set_throughput(size * Channels, sizeof(T));

    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        HIP_CHECK(hipFree(d_histogram[channel]));
    }
}

template<typename T>
void run_range_benchmark(benchmark_utils::state&& state, size_t bins)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;

    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    // Generate data
    const auto     random_range = limit_random_range<T>(0, bins);
    std::vector<T> input
        = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());

    std::vector<level_type> levels(bins + 1);
    for(size_t i = 0; i < levels.size(); ++i)
    {
        levels[i] = static_cast<level_type>(i);
    }

    common::device_ptr<T>            d_input(input);
    common::device_ptr<level_type>   d_levels(levels);
    common::device_ptr<counter_type> d_histogram(bins);

    size_t temporary_storage_bytes = 0;
    HIP_CHECK(rocprim::histogram_range(nullptr,
                                       temporary_storage_bytes,
                                       d_input.get(),
                                       size,
                                       d_histogram.get(),
                                       bins + 1,
                                       d_levels.get(),
                                       stream,
                                       false));

    common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            HIP_CHECK(rocprim::histogram_range(d_temporary_storage.get(),
                                               temporary_storage_bytes,
                                               d_input.get(),
                                               size,
                                               d_histogram.get(),
                                               bins + 1,
                                               d_levels.get(),
                                               stream,
                                               false));
        });

    state.set_throughput(size, sizeof(T));
}

template<typename T, unsigned int Channels, unsigned int ActiveChannels>
void run_multi_range_benchmark(benchmark_utils::state&& state, size_t bins)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;

    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    const int               num_levels_channel = bins + 1;
    unsigned int            num_levels[ActiveChannels];
    std::vector<level_type> levels[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        levels[channel].resize(num_levels_channel);
        for(size_t i = 0; i < levels[channel].size(); ++i)
        {
            levels[channel][i] = static_cast<level_type>(i);
        }
        num_levels[channel] = num_levels_channel;
    }

    // Generate data
    const auto     random_range = limit_random_range<T>(0, bins);
    std::vector<T> input        = get_random_data<T>(size * Channels,
                                              random_range.first,
                                              random_range.second,
                                              seed.get_0());

    common::device_ptr<T> d_input(input);
    level_type*   d_levels[ActiveChannels];
    counter_type*         d_histogram[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        HIP_CHECK(hipMalloc(&d_levels[channel], num_levels_channel * sizeof(level_type)));
        HIP_CHECK(hipMalloc(&d_histogram[channel], bins * sizeof(counter_type)));
    }

    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        HIP_CHECK(hipMemcpy(d_levels[channel],
                            levels[channel].data(),
                            num_levels_channel * sizeof(level_type),
                            hipMemcpyHostToDevice));
    }

    size_t temporary_storage_bytes = 0;
    HIP_CHECK((rocprim::multi_histogram_range<Channels, ActiveChannels>(nullptr,
                                                                        temporary_storage_bytes,
                                                                        d_input.get(),
                                                                        size,
                                                                        d_histogram,
                                                                        num_levels,
                                                                        d_levels,
                                                                        stream,
                                                                        false)));

    common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            HIP_CHECK(
                (rocprim::multi_histogram_range<Channels, ActiveChannels>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_input.get(),
                                                                          size,
                                                                          d_histogram,
                                                                          num_levels,
                                                                          d_levels,
                                                                          stream,
                                                                          false)));
        });

    state.set_throughput(size * Channels, sizeof(T));

    for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
    {
        HIP_CHECK(hipFree(d_levels[channel]));
        HIP_CHECK(hipFree(d_histogram[channel]));
    }
}

#define CREATE_EVEN_BENCHMARK(T, BINS, SCALE)                                                  \
    executor.queue_fn(                                                                         \
        bench_naming::format_name("{lvl:device,algo:histogram_even,value_type:" #T ",entropy:" \
                                  + std::to_string(get_entropy_percents(entropy_reduction))    \
                                  + ",bins:" + std::to_string(BINS) + ",cfg:default_config}")  \
            .c_str(),                                                                          \
        [=](benchmark_utils::state&& state)                                                    \
        {                                                                                      \
            run_even_benchmark<T>(std::forward<benchmark_utils::state>(state),                 \
                                  BINS,                                                        \
                                  SCALE,                                                       \
                                  entropy_reduction);                                          \
        });

#define BENCHMARK_EVEN_TYPE(T, S)     \
    CREATE_EVEN_BENCHMARK(T, 10, S)   \
    CREATE_EVEN_BENCHMARK(T, 100, S)  \
    CREATE_EVEN_BENCHMARK(T, 1000, S) \
    CREATE_EVEN_BENCHMARK(T, 10000, S)

#define CREATE_MULTI_EVEN_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS, SCALE)                    \
    executor.queue_fn(bench_naming::format_name(                                                  \
                          "{lvl:device,algo:multi_histogram_even,value_type:" #T                  \
                          ",channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS ",entropy:" \
                          + std::to_string(get_entropy_percents(entropy_reduction))               \
                          + ",bins:" + std::to_string(BINS) + ",cfg:default_config}")             \
                          .c_str(),                                                               \
                      [=](benchmark_utils::state&& state)                                         \
                      {                                                                           \
                          run_multi_even_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(                 \
                              std::forward<benchmark_utils::state>(state),                        \
                              BINS,                                                               \
                              SCALE,                                                              \
                              entropy_reduction);                                                 \
                      });

#define BENCHMARK_MULTI_EVEN_TYPE(C, A, T, S)     \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 10, S)   \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 100, S)  \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 1000, S) \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 10000, S)

#define CREATE_RANGE_BENCHMARK(T, BINS)                                                      \
    executor.queue_fn(                                                                       \
        bench_naming::format_name("{lvl:device,algo:histogram_range,value_type:" #T ",bins:" \
                                  + std::to_string(BINS) + ",cfg:default_config}")           \
            .c_str(),                                                                        \
        [=](benchmark_utils::state&& state)                                                  \
        { run_range_benchmark<T>(std::forward<benchmark_utils::state>(state), BINS); });

#define BENCHMARK_RANGE_TYPE(T)     \
    CREATE_RANGE_BENCHMARK(T, 10)   \
    CREATE_RANGE_BENCHMARK(T, 100)  \
    CREATE_RANGE_BENCHMARK(T, 1000) \
    CREATE_RANGE_BENCHMARK(T, 10000)

#define CREATE_MULTI_RANGE_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS)                       \
    executor.queue_fn(bench_naming::format_name(                                               \
                          "{lvl:device,algo:multi_histogram_range,value_type:" #T              \
                          ",channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS ",bins:" \
                          + std::to_string(BINS) + ",cfg:default_config}")                     \
                          .c_str(),                                                            \
                      [=](benchmark_utils::state&& state)                                      \
                      {                                                                        \
                          run_multi_range_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(             \
                              std::forward<benchmark_utils::state>(state),                     \
                              BINS);                                                           \
                      });

#define BENCHMARK_MULTI_RANGE_TYPE(C, A, T)     \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 10)   \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 100)  \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 1000) \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 10000)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    const int entropy_reductions[] = {0, 2, 4, 6};

    // Even benchmarks
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_EVEN_TYPE(long long, 12345)
        BENCHMARK_EVEN_TYPE(int, 1234)
        BENCHMARK_EVEN_TYPE(short, 5)
        CREATE_EVEN_BENCHMARK(unsigned char, 16, 16)
        CREATE_EVEN_BENCHMARK(unsigned char, 256, 1)
        BENCHMARK_EVEN_TYPE(double, 1234)
        BENCHMARK_EVEN_TYPE(float, 1234)
        BENCHMARK_EVEN_TYPE(rocprim::half, 5)
        CREATE_EVEN_BENCHMARK(rocprim::int128_t, 16, 16)
        CREATE_EVEN_BENCHMARK(rocprim::int128_t, 256, 1)
        CREATE_EVEN_BENCHMARK(rocprim::uint128_t, 16, 16)
        CREATE_EVEN_BENCHMARK(rocprim::uint128_t, 256, 1)
    }

    // Multi-even benchmarks
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_MULTI_EVEN_TYPE(4, 4, int, 1234)
        BENCHMARK_MULTI_EVEN_TYPE(4, 3, short, 5)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 16, 16)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 256, 1)
        BENCHMARK_MULTI_EVEN_TYPE(3, 3, float, 1234)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::int128_t, 16, 16)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::int128_t, 256, 1)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::uint128_t, 16, 16)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::uint128_t, 256, 1)
    }

    // Range benchmarks
    BENCHMARK_RANGE_TYPE(long long)
    BENCHMARK_RANGE_TYPE(int)
    BENCHMARK_RANGE_TYPE(short)
    CREATE_RANGE_BENCHMARK(unsigned char, 16)
    CREATE_RANGE_BENCHMARK(unsigned char, 256)
    BENCHMARK_RANGE_TYPE(double)
    BENCHMARK_RANGE_TYPE(float)
    BENCHMARK_RANGE_TYPE(rocprim::half)
    CREATE_RANGE_BENCHMARK(rocprim::int128_t, 16)
    CREATE_RANGE_BENCHMARK(rocprim::int128_t, 256)
    CREATE_RANGE_BENCHMARK(rocprim::uint128_t, 16)
    CREATE_RANGE_BENCHMARK(rocprim::uint128_t, 256)

    // Multi-range benchmarks
    BENCHMARK_MULTI_RANGE_TYPE(4, 4, int)
    BENCHMARK_MULTI_RANGE_TYPE(4, 3, short)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, unsigned char, 16)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, unsigned char, 256)
    BENCHMARK_MULTI_RANGE_TYPE(3, 3, float)
    BENCHMARK_MULTI_RANGE_TYPE(2, 2, double)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::int128_t, 16)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::int128_t, 256)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::uint128_t, 16)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::uint128_t, 256)
#endif

    executor.run();
}
