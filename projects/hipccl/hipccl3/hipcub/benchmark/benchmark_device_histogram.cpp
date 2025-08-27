// MIT License
//
// Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// CUB's implementation of DeviceRunLengthEncode has unused parameters,
// disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVIDIA__
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "common_benchmark_header.hpp"

// HIP API
#include <hipcub/device/device_histogram.hpp>
#include <hipcub/iterator/transform_input_iterator.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class T>
std::vector<T>
    generate(size_t size, int entropy_reduction, long long lower_level, long long upper_level)
{
    if(entropy_reduction >= 5)
    {
        return std::vector<T>(size, (lower_level + upper_level) / 2);
    }

    const size_t max_random_size = 1024 * 1024;

    std::random_device         rd;
    std::default_random_engine gen(rd());
    std::vector<T>             data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]()
                  {
                      // Reduce entropy by applying bitwise AND to random bits
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
    // casting for compilation with CUB backend because
    // there is no casting from size_t (aka unsigned long) to __half
    const T upper_level = static_cast<unsigned long long>(bins * scale);

    // Generate data
    std::vector<T> input = generate<T>(size, entropy_reduction, lower_level, upper_level);

    T*            d_input;
    counter_type* d_histogram;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_histogram, size * sizeof(counter_type)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(hipcub::DeviceHistogram::HistogramEven(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_input,
                                                     d_histogram,
                                                     bins + 1,
                                                     lower_level,
                                                     upper_level,
                                                     int(size),
                                                     stream));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(hipcub::DeviceHistogram::HistogramEven(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_input,
                                                         d_histogram,
                                                         bins + 1,
                                                         lower_level,
                                                         upper_level,
                                                         int(size),
                                                         stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceHistogram::HistogramEven(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_input,
                                                             d_histogram,
                                                             bins + 1,
                                                             lower_level,
                                                             upper_level,
                                                             int(size),
                                                             stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
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

    int num_levels[ActiveChannels];
    int lower_level[ActiveChannels];
    int upper_level[ActiveChannels];
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
    HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramEven<Channels, ActiveChannels>(
        d_temporary_storage,
        temporary_storage_bytes,
        d_input,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        int(size),
        stream)));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramEven<Channels, ActiveChannels>(
            d_temporary_storage,
            temporary_storage_bytes,
            d_input,
            d_histogram,
            num_levels,
            lower_level,
            upper_level,
            int(size),
            stream)));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramEven<Channels, ActiveChannels>(
                d_temporary_storage,
                temporary_storage_bytes,
                d_input,
                d_histogram,
                num_levels,
                lower_level,
                upper_level,
                int(size),
                stream)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
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
    std::vector<T> input = benchmark_utils::get_random_data<T>(size, 0, bins);

    std::vector<T> levels(bins + 1);
    std::iota(levels.begin(), levels.end(), static_cast<T>(0));

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
    HIP_CHECK(hipcub::DeviceHistogram::HistogramRange(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_input,
                                                      d_histogram,
                                                      bins + 1,
                                                      d_levels,
                                                      int(size),
                                                      stream));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(hipcub::DeviceHistogram::HistogramRange(d_temporary_storage,
                                                          temporary_storage_bytes,
                                                          d_input,
                                                          d_histogram,
                                                          bins + 1,
                                                          d_levels,
                                                          int(size),
                                                          stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceHistogram::HistogramRange(d_temporary_storage,
                                                              temporary_storage_bytes,
                                                              d_input,
                                                              d_histogram,
                                                              bins + 1,
                                                              d_levels,
                                                              int(size),
                                                              stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
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

    // Number of levels for a single channel
    const int      num_levels_channel = bins + 1;
    int            num_levels[ActiveChannels];
    std::vector<T> levels[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        levels[channel].resize(num_levels_channel);
        std::iota(levels[channel].begin(), levels[channel].end(), static_cast<T>(0));
        num_levels[channel] = num_levels_channel;
    }

    // Generate data
    std::vector<T> input = benchmark_utils::get_random_data<T>(size * Channels, 0, bins);

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
    HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramRange<Channels, ActiveChannels>(
        d_temporary_storage,
        temporary_storage_bytes,
        d_input,
        d_histogram,
        num_levels,
        d_levels,
        int(size),
        stream)));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramRange<Channels, ActiveChannels>(
            d_temporary_storage,
            temporary_storage_bytes,
            d_input,
            d_histogram,
            num_levels,
            d_levels,
            int(size),
            stream)));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramRange<Channels, ActiveChannels>(
                d_temporary_storage,
                temporary_storage_bytes,
                d_input,
                d_histogram,
                num_levels,
                d_levels,
                int(size),
                stream)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
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
struct num_limits<__half>
{
    static constexpr double max()
    {
        return 65504.0;
    };
};

#define CREATE_EVEN_BENCHMARK(VECTOR, T, BINS, SCALE)                                          \
    if(num_limits<T>::max() > BINS * SCALE)                                                    \
    {                                                                                          \
        VECTOR.push_back(benchmark::RegisterBenchmark(                                         \
            std::string("device_histogram_even"                                                \
                        "<data_type:" #T ">."                                                  \
                        "(entropy_percent:"                                                    \
                        + std::to_string(get_entropy_percents(entropy_reduction))              \
                        + "%,bin_count:" + std::to_string(BINS) + " bins)")                    \
                .c_str(),                                                                      \
            [=](benchmark::State& state)                                                       \
            { run_even_benchmark<T>(state, BINS, SCALE, entropy_reduction, stream, size); })); \
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
        // this limitation can be removed once
        // https://github.com/NVIDIA/cub/issues/484 is fixed
#ifdef __HIP_PLATFORM_AMD__
        BENCHMARK_TYPE(benchmarks, __half);
#endif
    };
}

#define CREATE_MULTI_EVEN_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS, SCALE)                   \
    benchmark::RegisterBenchmark(                                                                \
        std::string("device_multi_histogram_even"                                                \
                    "<channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS ",data_type:" #T \
                    ">."                                                                         \
                    "(entropy_percent:"                                                          \
                    + std::to_string(get_entropy_percents(entropy_reduction))                    \
                    + "%,bin_count:" + std::to_string(BINS) + " bins)")                          \
            .c_str(),                                                                            \
        [=](benchmark::State& state)                                                             \
        {                                                                                        \
            run_multi_even_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(state,                        \
                                                                   BINS,                         \
                                                                   SCALE,                        \
                                                                   entropy_reduction,            \
                                                                   stream,                       \
                                                                   size);                        \
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

#define CREATE_RANGE_BENCHMARK(T, BINS)                                         \
    benchmark::RegisterBenchmark(std::string("device_histogram_range"           \
                                             "<data_type:" #T ">."              \
                                             "(bin_count:"                      \
                                             + std::to_string(BINS) + " bins)") \
                                     .c_str(),                                  \
                                 [=](benchmark::State& state)                   \
                                 { run_range_benchmark<T>(state, BINS, stream, size); })

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

#define CREATE_MULTI_RANGE_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS)                         \
    benchmark::RegisterBenchmark(                                                                \
        std::string("device_multi_histogram_range"                                               \
                    "<channels:" #CHANNELS ",active_channels:" #ACTIVE_CHANNELS ",data_type:" #T \
                    ">.(bin_count:"                                                              \
                    + std::to_string(BINS) + " bins)")                                           \
            .c_str(),                                                                            \
        [=](benchmark::State& state)                                                             \
        { run_multi_range_benchmark<T, CHANNELS, ACTIVE_CHANNELS>(state, BINS, stream, size); })

void add_multi_range_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                hipStream_t                                   stream,
                                size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 10),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 100),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 1000),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 10000),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 100000),
        CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 1000000),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    std::cout << "benchmark_device_histogram" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

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
