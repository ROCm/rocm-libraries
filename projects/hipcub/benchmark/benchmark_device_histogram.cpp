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

#include "benchmark_utils.hpp"

#include <hipcub/device/device_histogram.hpp>
#include <hipcub/iterator/transform_input_iterator.hpp>

template<class T>
std::vector<T>
    generate(size_t items, int entropy_reduction, int64_t lower_level, int64_t upper_level)
{
    if(entropy_reduction >= 5)
    {
        return std::vector<T>(items, (lower_level + upper_level) / 2);
    }

    const size_t max_random_size = 1024 * 1024;

    std::random_device         rd;
    std::default_random_engine gen(rd());
    std::vector<T>             data(items);
    std::generate(data.begin(),
                  data.begin() + std::min(items, max_random_size),
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
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(items - i, max_random_size), data.begin() + i);
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

template<class T, size_t Bins, size_t Scale>
class even_benchmark : public primbench::benchmark_interface
{
public:
    even_benchmark(int entropy_reduction) : m_entropy_reduction(entropy_reduction) {}

private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_histogram")
            .add("subalgo", "even")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("bin_count", Bins)
            .add("entropy_percent", std::to_string(get_entropy_percents(m_entropy_reduction)));
    }

    void run(primbench::state& state) override
    {
        using counter_type = unsigned int;

        const T lower_level = 0;
        // casting for compilation with CUB backend because
        // there is no casting from size_t (aka unsigned long) to __half
        const T upper_level = static_cast<T>(Bins * Scale);

        const size_t items  = state.size;
        const auto&  stream = state.stream;

        // Generate data
        std::vector<T> input = generate<T>(items, m_entropy_reduction, lower_level, upper_level);

        T*            d_input;
        counter_type* d_histogram;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_histogram, items * sizeof(counter_type)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_input,
                                                             d_histogram,
                                                             Bins + 1,
                                                             lower_level,
                                                             upper_level,
                                                             int(items),
                                                             stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_histogram));
    }

    int m_entropy_reduction;
};

template<class T, unsigned int Channels, unsigned int ActiveChannels, size_t Bins, size_t Scale>
class multi_even_benchmark : public primbench::benchmark_interface
{
public:
    multi_even_benchmark(int entropy_reduction) : m_entropy_reduction(entropy_reduction) {}

private:
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_histogram")
            .add("subalgo", "multi_even")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("bin_count", Bins)
            .add("channels", Channels)
            .add("active_channels", ActiveChannels)
            .add("entropy_percent", std::to_string(get_entropy_percents(m_entropy_reduction)));
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using counter_type = unsigned int;

        int num_levels[ActiveChannels];
        int lower_level[ActiveChannels];
        int upper_level[ActiveChannels];
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            lower_level[channel] = 0;
            upper_level[channel] = Bins * Scale;
            num_levels[channel]  = Bins + 1;
        }

        // Generate data
        std::vector<T> input
            = generate<T>(items * Channels, m_entropy_reduction, lower_level[0], upper_level[0]);

        T*            d_input;
        counter_type* d_histogram[ActiveChannels];
        HIP_CHECK(hipMalloc(&d_input, items * Channels * sizeof(T)));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipMalloc(&d_histogram[channel], Bins * sizeof(counter_type)));
        }
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), items * Channels * sizeof(T), hipMemcpyHostToDevice));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramEven<Channels, ActiveChannels>(
                d_temp_storage,
                temp_storage_bytes,
                d_input,
                d_histogram,
                num_levels,
                lower_level,
                upper_level,
                int(items),
                stream)));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items * Channels);
        state.add_writes<T>(items * Channels);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipFree(d_histogram[channel]));
        }
    }

    int m_entropy_reduction;
};

template<class T, size_t Bins>
class range_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_histogram")
            .add("subalgo", "range")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("bin_count", Bins);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using counter_type = unsigned int;

        // Generate data
        std::vector<T> input = benchmark_utils::get_random_data<T>(items, 0, Bins);

        std::vector<T> levels(Bins + 1);
        std::iota(levels.begin(), levels.end(), static_cast<T>(0));

        T*            d_input;
        T*            d_levels;
        counter_type* d_histogram;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_levels, (Bins + 1) * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_histogram, items * sizeof(counter_type)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(d_levels, levels.data(), (Bins + 1) * sizeof(T), hipMemcpyHostToDevice));

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK(hipcub::DeviceHistogram::HistogramRange(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_input,
                                                              d_histogram,
                                                              Bins + 1,
                                                              d_levels,
                                                              int(items),
                                                              stream));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_levels));
        HIP_CHECK(hipFree(d_histogram));
    }
};

template<class T, unsigned int Channels, unsigned int ActiveChannels, size_t Bins>
class multi_range_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_histogram")
            .add("subalgo", "multi_range")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("channels", Channels)
            .add("active_channels", ActiveChannels)
            .add("bin_count", Bins);
    }

    void run(primbench::state& state) override
    {
        const size_t items  = state.size;
        const auto&  stream = state.stream;

        using counter_type = unsigned int;

        // Number of levels for a single channel
        const int      num_levels_channel = Bins + 1;
        int            num_levels[ActiveChannels];
        std::vector<T> levels[ActiveChannels];
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            levels[channel].resize(num_levels_channel);
            std::iota(levels[channel].begin(), levels[channel].end(), static_cast<T>(0));
            num_levels[channel] = num_levels_channel;
        }

        // Generate data
        std::vector<T> input = benchmark_utils::get_random_data<T>(items * Channels, 0, Bins);

        T*            d_input;
        T*            d_levels[ActiveChannels];
        counter_type* d_histogram[ActiveChannels];
        HIP_CHECK(hipMalloc(&d_input, items * Channels * sizeof(T)));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipMalloc(&d_levels[channel], num_levels_channel * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_histogram[channel], items * sizeof(counter_type)));
        }

        HIP_CHECK(
            hipMemcpy(d_input, input.data(), items * Channels * sizeof(T), hipMemcpyHostToDevice));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipMemcpy(d_levels[channel],
                                levels[channel].data(),
                                num_levels_channel * sizeof(T),
                                hipMemcpyHostToDevice));
        }

        void*  d_temp_storage = nullptr;
        size_t temp_storage_bytes;

        const auto launch = [&]
        {
            HIP_CHECK((hipcub::DeviceHistogram::MultiHistogramRange<Channels, ActiveChannels>(
                d_temp_storage,
                temp_storage_bytes,
                d_input,
                d_histogram,
                num_levels,
                d_levels,
                int(items),
                stream)));
        };

        launch();

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(items * Channels);
        state.add_writes<T>(items * Channels);

        state.run(launch);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipFree(d_levels[channel]));
            HIP_CHECK(hipFree(d_histogram[channel]));
        }
    }
};

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

#define CREATE_EVEN_BENCHMARK(VECTOR, T, BINS, SCALE)                      \
    if(num_limits<T>::max() > BINS * SCALE)                                \
    {                                                                      \
        executor.queue<even_benchmark<T, BINS, SCALE>>(entropy_reduction); \
    }

#define BENCHMARK_TYPE(VECTOR, T)                 \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 10, 1234);   \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 100, 1234);  \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 1000, 1234); \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 16, 10);     \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 256, 10);    \
    CREATE_EVEN_BENCHMARK(VECTOR, T, 65536, 1)

void add_even_benchmarks(primbench::executor& executor)
{
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_TYPE(benchmarks, int64_t);
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

#define CREATE_MULTI_EVEN_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS, SCALE)       \
    executor.queue<multi_even_benchmark<T, CHANNELS, ACTIVE_CHANNELS, BINS, SCALE>>( \
        entropy_reduction);

void add_multi_even_benchmarks(primbench::executor& executor)
{
    for(int entropy_reduction : entropy_reductions)
    {
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, int, 10, 1234);
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, int, 100, 1234);

        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 16, 10);
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned char, 256, 1);

        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned short, 16, 10);
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned short, 256, 10);
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, unsigned short, 65536, 1);
    };
}

#define CREATE_RANGE_BENCHMARK(T, BINS) executor.queue<range_benchmark<T, BINS>>();

#define BENCHMARK_RANGE_TYPE(T)        \
    CREATE_RANGE_BENCHMARK(T, 10);     \
    CREATE_RANGE_BENCHMARK(T, 100);    \
    CREATE_RANGE_BENCHMARK(T, 1000);   \
    CREATE_RANGE_BENCHMARK(T, 10000);  \
    CREATE_RANGE_BENCHMARK(T, 100000); \
    CREATE_RANGE_BENCHMARK(T, 1000000)

void add_range_benchmarks(primbench::executor& executor)
{
    BENCHMARK_RANGE_TYPE(float);
    BENCHMARK_RANGE_TYPE(double);
}

#define CREATE_MULTI_RANGE_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS) \
    executor.queue<multi_range_benchmark<T, CHANNELS, ACTIVE_CHANNELS, BINS>>()

void add_multi_range_benchmarks(primbench::executor& executor)
{
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 10);
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 100);
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 1000);
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 10000);
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 100000);
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, float, 1000000);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    // Add benchmarks
    add_even_benchmarks(executor);
    add_multi_even_benchmarks(executor);
    add_range_benchmarks(executor);
    add_multi_range_benchmarks(executor);

    executor.run();
}
