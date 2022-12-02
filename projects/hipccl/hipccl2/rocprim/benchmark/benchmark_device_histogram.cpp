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

#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class T>
std::vector<T> generate(size_t size, int entropy_reduction, int lower_level, int upper_level)
{
    if(entropy_reduction >= 5)
    {
        return std::vector<T>(size, (T)((lower_level + upper_level) / 2));
    }

    const size_t max_random_size = 1024 * 1024;

    std::random_device         rd;
    std::default_random_engine gen(rd());
    std::vector<T>             data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]()
                  {
                      // Reduce enthropy by applying bitwise AND to random bits
                      // "An Improved Supercomputer Sorting Benchmark", 1992
                      // Kurt Thearling & Stephen Smith
                      auto v = gen();
                      for(int e = 0; e < entropy_reduction; e++)
                      {
                          v &= gen();
                      }
                      return T(lower_level + v % (upper_level - lower_level));
                  });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

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

    const T lower_level = 0;
    const T upper_level = bins * scale;

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

    unsigned int num_levels[ActiveChannels];
    int          lower_level[ActiveChannels];
    int          upper_level[ActiveChannels];
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

    // Generate data
    std::vector<T> input = get_random_data<T>(size, 0, bins);

    std::vector<T> levels(bins + 1);
    std::iota(levels.begin(), levels.end(), 0);

    T*            d_input;
    T*            d_levels;
    counter_type* d_histogram;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_levels, (bins + 1) * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_histogram, size * sizeof(counter_type)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_levels, levels.data(), (bins + 1) * sizeof(T), hipMemcpyHostToDevice));

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

    const int      num_levels_channel = bins + 1;
    unsigned int   num_levels[ActiveChannels];
    std::vector<T> levels[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        levels[channel].resize(num_levels_channel);
        std::iota(levels[channel].begin(), levels[channel].end(), static_cast<T>(0));
        num_levels[channel] = num_levels_channel;
    }

    // Generate data
    std::vector<T> input = get_random_data<T>(size * Channels, 0, bins);

    T*            d_input;
    T*            d_levels[ActiveChannels];
    counter_type* d_histogram[ActiveChannels];
    HIP_CHECK(hipMalloc(&d_input, size * Channels * sizeof(T)));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipMalloc(&d_levels[channel], num_levels_channel * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_histogram[channel], size * sizeof(counter_type)));
    }

    HIP_CHECK(hipMemcpy(d_input, input.data(), size * Channels * sizeof(T), hipMemcpyHostToDevice));
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        HIP_CHECK(hipMemcpy(d_levels[channel],
                            levels[channel].data(),
                            num_levels_channel * sizeof(T),
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

template<class T>
struct num_limits
{
    static constexpr T max()
    {
        return std::numeric_limits<T>::max();
    };
};

template<>
struct num_limits<rocprim::half>
{
    static constexpr double max()
    {
        return 65504.0;
    };
};

#define CREATE_EVEN_BENCHMARK(VECTOR, T, BINS, SCALE)                                             \
    if(num_limits<T>::max() > BINS * SCALE)                                                       \
    {                                                                                             \
        VECTOR.push_back(benchmark::RegisterBenchmark(                                            \
            bench_naming::format_name("{lvl:device,algo:histogram_even,key_type:" #T ",entropy:"  \
                                      + std::to_string(get_entropy_percents(entropy_reduction))   \
                                      + ",bins:" + std::to_string(BINS) + ",cfg:default_config}") \
                .c_str(),                                                                         \
            [=](benchmark::State& state)                                                          \
            { run_even_benchmark<T>(state, BINS, SCALE, entropy_reduction, stream, size); }));    \
    }

#define BENCHMARK_TYPE(VECTOR, T)                 \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 10, 1234);   \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 100, 1234);  \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 1000, 1234); \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 16, 10);     \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 256, 10);    \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 65536, 1)

void add_even_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                         hipStream_t                                   stream,
                         size_t                                        size)
{
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_TYPE(benchmarks, long long);
        BENCHMARK_TYPE(benchmarks, int);
        BENCHMARK_TYPE(benchmarks, unsigned short);
        BENCHMARK_TYPE(benchmarks, uint8_t);
        BENCHMARK_TYPE(benchmarks, double);
        BENCHMARK_TYPE(benchmarks, float);
        BENCHMARK_TYPE(benchmarks, rocprim::half);
    };
}

#define CREATE_MULTI_EVEN_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS, SCALE)                \
    benchmark::RegisterBenchmark(                                                             \
        bench_naming::format_name("{lvl:device,algo:histogram_even_multi,key_type:" #T        \
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

void add_multi_even_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                               hipStream_t                                   stream,
                               size_t                                        size)
{
    for(int entropy_reduction : entropy_reductions)
    {
        std::vector<benchmark::internal::Benchmark*> bs = {
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, int, 10, 1234),
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, int, 100, 1234),

            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 16, 10),
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 256, 1),

            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned short, 16, 10),
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned short, 256, 10),
            CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned short, 65536, 1),
        };
        benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
    };
}

#define CREATE_RANGE_BENCHMARK(T, BINS)                                                    \
    benchmark::RegisterBenchmark(                                                          \
        bench_naming::format_name("{lvl:device,algo:histogram_range,key_type:" #T ",bins:" \
                                  + std::to_string(BINS) + ",cfg:default_config}")         \
            .c_str(),                                                                      \
        [=](benchmark::State& state) { run_range_benchmark<T>(state, BINS, stream, size); })

#define BENCHMARK_RANGE_TYPE(T)                                            \
    CREATE_RANGE_BENCHMARK(T, 10), CREATE_RANGE_BENCHMARK(T, 100),         \
        CREATE_RANGE_BENCHMARK(T, 1000), CREATE_RANGE_BENCHMARK(T, 10000), \
        CREATE_RANGE_BENCHMARK(T, 100000), CREATE_RANGE_BENCHMARK(T, 1000000)

void add_range_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                          hipStream_t                                   stream,
                          size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs
        = {BENCHMARK_RANGE_TYPE(float), BENCHMARK_RANGE_TYPE(double)};
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#define CREATE_MULTI_RANGE_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS)                          \
    benchmark::RegisterBenchmark(                                                                 \
        bench_naming::format_name(                                                                \
            "{lvl:device,algo:histogram_range,key_type:" #T ",bins:" + std::to_string(BINS)       \
            + ",channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS ",cfg:default_config}") \
            .c_str(),                                                                             \
        [=](benchmark::State& state)                                                              \
        { run_multi_range_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(state, BINS, stream, size); })

void add_multi_range_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                hipStream_t                                   stream,
                                size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs
        = {CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 10),
           CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 100),
           CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 1000),
           CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 10000),
           CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 100000),
           CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 1000000)};
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
    add_even_benchmarks(benchmarks, stream, size);
    add_multi_even_benchmarks(benchmarks, stream, size);
    add_range_benchmarks(benchmarks, stream, size);
    add_multi_range_benchmarks(benchmarks, stream, size);

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
