// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_device_histogram.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

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

const int entropy_reductions[] = {0, 2, 4, 6};

template<class T>
void run_even_benchmark(benchmark::State& state,
                        size_t            bins,
                        size_t            scale,
                        int               entropy_reduction,
                        hipStream_t       stream,
                        size_t            size)
{
    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    const level_type lower_level = 0;
    const level_type upper_level = bins * scale;

    // Generate data
    std::vector<T> input = generate<T>(size, entropy_reduction, lower_level, upper_level);

    T*            d_input;
    counter_type* d_histogram;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_histogram, size * sizeof(counter_type)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(rp::histogram_even(d_temporary_storage,
                                 temporary_storage_bytes,
                                 d_input,
                                 size,
                                 d_histogram,
                                 bins + 1,
                                 lower_level,
                                 upper_level,
                                 stream,
                                 false));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(rp::histogram_even(d_temporary_storage,
                                     temporary_storage_bytes,
                                     d_input,
                                     size,
                                     d_histogram,
                                     bins + 1,
                                     lower_level,
                                     upper_level,
                                     stream,
                                     false));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(rp::histogram_even(d_temporary_storage,
                                         temporary_storage_bytes,
                                         d_input,
                                         size,
                                         d_histogram,
                                         bins + 1,
                                         lower_level,
                                         upper_level,
                                         stream,
                                         false));
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_histogram));
}

template<class T, unsigned int Channels, unsigned int ActiveChannels>
void run_multi_even_benchmark(benchmark::State& state,
                              size_t            bins,
                              size_t            scale,
                              int               entropy_reduction,
                              hipStream_t       stream,
                              size_t            size)
{
    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    unsigned int num_levels[ActiveChannels];
    level_type   lower_level[ActiveChannels];
    level_type   upper_level[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        lower_level[channel] = 0;
        upper_level[channel] = bins * scale;
        num_levels[channel]  = bins + 1;
    }

    // Generate data
    std::vector<T> input
        = generate<T>(size * Channels, entropy_reduction, lower_level[0], upper_level[0]);

    T*            d_input;
    counter_type* d_histogram[ActiveChannels];
    HIP_CHECK(hipMalloc(&d_input, size * Channels * sizeof(T)));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipMalloc(&d_histogram[channel], bins * sizeof(counter_type)));
    }
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * Channels * sizeof(T), hipMemcpyHostToDevice));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK((rp::multi_histogram_even<Channels, ActiveChannels>(d_temporary_storage,
                                                                  temporary_storage_bytes,
                                                                  d_input,
                                                                  size,
                                                                  d_histogram,
                                                                  num_levels,
                                                                  lower_level,
                                                                  upper_level,
                                                                  stream,
                                                                  false)));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK((rp::multi_histogram_even<Channels, ActiveChannels>(d_temporary_storage,
                                                                      temporary_storage_bytes,
                                                                      d_input,
                                                                      size,
                                                                      d_histogram,
                                                                      num_levels,
                                                                      lower_level,
                                                                      upper_level,
                                                                      stream,
                                                                      false)));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK((rp::multi_histogram_even<Channels, ActiveChannels>(d_temporary_storage,
                                                                          temporary_storage_bytes,
                                                                          d_input,
                                                                          size,
                                                                          d_histogram,
                                                                          num_levels,
                                                                          lower_level,
                                                                          upper_level,
                                                                          stream,
                                                                          false)));
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * Channels * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size * Channels);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_input));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipFree(d_histogram[channel]));
    }
}

template<class T>
void run_range_benchmark(benchmark::State& state, size_t bins, hipStream_t stream, size_t size)
{
    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    // Generate data
    std::vector<T> input = get_random_data<T>(size, 0, bins);

    std::vector<level_type> levels(bins + 1);
    for(size_t i = 0; i < levels.size(); i++)
    {
        levels[i] = static_cast<level_type>(i);
    }

    T*            d_input;
    level_type*   d_levels;
    counter_type* d_histogram;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_levels, (bins + 1) * sizeof(level_type)));
    HIP_CHECK(hipMalloc(&d_histogram, size * sizeof(counter_type)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_levels, levels.data(), (bins + 1) * sizeof(level_type), hipMemcpyHostToDevice));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(rp::histogram_range(d_temporary_storage,
                                  temporary_storage_bytes,
                                  d_input,
                                  size,
                                  d_histogram,
                                  bins + 1,
                                  d_levels,
                                  stream,
                                  false));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(rp::histogram_range(d_temporary_storage,
                                      temporary_storage_bytes,
                                      d_input,
                                      size,
                                      d_histogram,
                                      bins + 1,
                                      d_levels,
                                      stream,
                                      false));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(rp::histogram_range(d_temporary_storage,
                                          temporary_storage_bytes,
                                          d_input,
                                          size,
                                          d_histogram,
                                          bins + 1,
                                          d_levels,
                                          stream,
                                          false));
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_levels));
    HIP_CHECK(hipFree(d_histogram));
}

template<class T, unsigned int Channels, unsigned int ActiveChannels>
void run_multi_range_benchmark(benchmark::State& state,
                               size_t            bins,
                               hipStream_t       stream,
                               size_t            size)
{
    using counter_type = unsigned int;
    using level_type =
        typename std::conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

    const int               num_levels_channel = bins + 1;
    unsigned int            num_levels[ActiveChannels];
    std::vector<level_type> levels[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        levels[channel].resize(num_levels_channel);
        for(size_t i = 0; i < levels[channel].size(); i++)
        {
            levels[channel][i] = static_cast<level_type>(i);
        }
        num_levels[channel] = num_levels_channel;
    }

    // Generate data
    std::vector<T> input = get_random_data<T>(size * Channels, 0, bins);

    T*            d_input;
    level_type*   d_levels[ActiveChannels];
    counter_type* d_histogram[ActiveChannels];
    HIP_CHECK(hipMalloc(&d_input, size * Channels * sizeof(T)));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipMalloc(&d_levels[channel], num_levels_channel * sizeof(level_type)));
        HIP_CHECK(hipMalloc(&d_histogram[channel], size * sizeof(counter_type)));
    }

    HIP_CHECK(hipMemcpy(d_input, input.data(), size * Channels * sizeof(T), hipMemcpyHostToDevice));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipMemcpy(d_levels[channel],
                            levels[channel].data(),
                            num_levels_channel * sizeof(level_type),
                            hipMemcpyHostToDevice));
    }

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK((rp::multi_histogram_range<Channels, ActiveChannels>(d_temporary_storage,
                                                                   temporary_storage_bytes,
                                                                   d_input,
                                                                   size,
                                                                   d_histogram,
                                                                   num_levels,
                                                                   d_levels,
                                                                   stream,
                                                                   false)));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK((rp::multi_histogram_range<Channels, ActiveChannels>(d_temporary_storage,
                                                                       temporary_storage_bytes,
                                                                       d_input,
                                                                       size,
                                                                       d_histogram,
                                                                       num_levels,
                                                                       d_levels,
                                                                       stream,
                                                                       false)));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK((rp::multi_histogram_range<Channels, ActiveChannels>(d_temporary_storage,
                                                                           temporary_storage_bytes,
                                                                           d_input,
                                                                           size,
                                                                           d_histogram,
                                                                           num_levels,
                                                                           d_levels,
                                                                           stream,
                                                                           false)));
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * Channels * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size * Channels);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_input));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipFree(d_levels[channel]));
        HIP_CHECK(hipFree(d_histogram[channel]));
    }
}

#define CREATE_EVEN_BENCHMARK(VECTOR, T, BINS, SCALE)                                          \
    VECTOR.push_back(benchmark::RegisterBenchmark(                                             \
        bench_naming::format_name("{lvl:device,algo:histogram_even,value_type:" #T ",entropy:" \
                                  + std::to_string(get_entropy_percents(entropy_reduction))    \
                                  + ",bins:" + std::to_string(BINS) + ",cfg:default_config}")  \
            .c_str(),                                                                          \
        [=](benchmark::State& state)                                                           \
        { run_even_benchmark<T>(state, BINS, SCALE, entropy_reduction, stream, size); }));

#define BENCHMARK_EVEN_TYPE(VECTOR, T, S)      \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 10, S);   \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 100, S);  \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 1000, S); \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 10000, S);

void add_even_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                         hipStream_t                                   stream,
                         size_t                                        size)
{
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_EVEN_TYPE(benchmarks, long long, 12345);
        BENCHMARK_EVEN_TYPE(benchmarks, int, 1234);
        BENCHMARK_EVEN_TYPE(benchmarks, short, 5);
        CREATE_EVEN_BENCHMARK(benchmarks, unsigned char, 16, 16);
        CREATE_EVEN_BENCHMARK(benchmarks, unsigned char, 256, 1);
        BENCHMARK_EVEN_TYPE(benchmarks, double, 1234);
        BENCHMARK_EVEN_TYPE(benchmarks, float, 1234);
        BENCHMARK_EVEN_TYPE(benchmarks, rocprim::half, 5);
    };
}

#define CREATE_MULTI_EVEN_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS, SCALE)                \
    benchmark::RegisterBenchmark(                                                             \
        bench_naming::format_name("{lvl:device,algo:multi_histogram_even,value_type:" #T      \
                                  ",channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS \
                                  ",entropy:"                                                 \
                                  + std::to_string(get_entropy_percents(entropy_reduction))   \
                                  + ",bins:" + std::to_string(BINS) + ",cfg:default_config}") \
            .c_str(),                                                                         \
        [=](benchmark::State& state)                                                          \
        {                                                                                     \
            run_multi_even_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(state,                     \
                                                                   BINS,                      \
                                                                   SCALE,                     \
                                                                   entropy_reduction,         \
                                                                   stream,                    \
                                                                   size);                     \
        })

#define BENCHMARK_MULTI_EVEN_TYPE(C, A, T, S)                                                  \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 10, S), CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 100, S), \
        CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 1000, S),                                         \
        CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 10000, S)

void add_multi_even_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                               hipStream_t                                   stream,
                               size_t                                        size)
{
    for(int entropy_reduction : entropy_reductions)
    {
        std::vector<benchmark::internal::Benchmark*> bs = {
            BENCHMARK_MULTI_EVEN_TYPE(4, 4, int, 1234),
            BENCHMARK_MULTI_EVEN_TYPE(4, 3, short, 5),
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 16, 16),
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 256, 1),
            BENCHMARK_MULTI_EVEN_TYPE(3, 3, float, 1234),
        };
        benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
    };
}

#define CREATE_RANGE_BENCHMARK(T, BINS)                                                      \
    benchmark::RegisterBenchmark(                                                            \
        bench_naming::format_name("{lvl:device,algo:histogram_range,value_type:" #T ",bins:" \
                                  + std::to_string(BINS) + ",cfg:default_config}")           \
            .c_str(),                                                                        \
        [=](benchmark::State& state) { run_range_benchmark<T>(state, BINS, stream, size); })

#define BENCHMARK_RANGE_TYPE(T)                                    \
    CREATE_RANGE_BENCHMARK(T, 10), CREATE_RANGE_BENCHMARK(T, 100), \
        CREATE_RANGE_BENCHMARK(T, 1000), CREATE_RANGE_BENCHMARK(T, 10000)

void add_range_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                          hipStream_t                                   stream,
                          size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        BENCHMARK_RANGE_TYPE(long long),
        BENCHMARK_RANGE_TYPE(int),
        BENCHMARK_RANGE_TYPE(short),
        CREATE_RANGE_BENCHMARK(unsigned char, 16),
        CREATE_RANGE_BENCHMARK(unsigned char, 256),
        BENCHMARK_RANGE_TYPE(double),
        BENCHMARK_RANGE_TYPE(float),
        BENCHMARK_RANGE_TYPE(rocprim::half),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#define CREATE_MULTI_RANGE_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS)                      \
    benchmark::RegisterBenchmark(                                                             \
        bench_naming::format_name("{lvl:device,algo:multi_histogram_range,value_type:" #T     \
                                  ",channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS \
                                  ",bins:"                                                    \
                                  + std::to_string(BINS) + ",cfg:default_config}")            \
            .c_str(),                                                                         \
        [=](benchmark::State& state)                                                          \
        { run_multi_range_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(state, BINS, stream, size); })

#define BENCHMARK_MULTI_RANGE_TYPE(C, A, T)                                                \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 10), CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 100), \
        CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 1000), CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 10000)

void add_multi_range_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                hipStream_t                                   stream,
                                size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        BENCHMARK_MULTI_RANGE_TYPE(4, 4, int),
        BENCHMARK_MULTI_RANGE_TYPE(4, 3, short),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, unsigned char, 16),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, unsigned char, 256),
        BENCHMARK_MULTI_RANGE_TYPE(3, 3, float),
        BENCHMARK_MULTI_RANGE_TYPE(2, 2, double),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
#ifdef BENCHMARK_CONFIG_TUNING
    // optionally run an evenly split subset of benchmarks, when making multiple program invocations
    parser.set_optional<int>("parallel_instance",
                             "parallel_instance",
                             0,
                             "parallel instance index");
    parser.set_optional<int>("parallel_instances",
                             "parallel_instances",
                             1,
                             "total parallel instances");
#endif
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        size,
                                                        stream);
#else // BENCHMARK_CONFIG_TUNING
    add_even_benchmarks(benchmarks, stream, size);
    add_multi_even_benchmarks(benchmarks, stream, size);
    add_range_benchmarks(benchmarks, stream, size);
    add_multi_range_benchmarks(benchmarks, stream, size);
#endif // BENCHMARK_CONFIG_TUNING

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
