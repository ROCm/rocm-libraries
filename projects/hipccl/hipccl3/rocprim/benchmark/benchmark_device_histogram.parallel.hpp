// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_HISTOGRAM_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_HISTOGRAM_PARALLEL_HPP_

#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_histogram.hpp>

#include "benchmark_utils.hpp"

template<class T>
std::vector<T> generate(size_t size, int entropy_reduction, int lower_level, int upper_level)
{
    if(entropy_reduction >= 5)
    {
        return std::vector<T>(size, static_cast<T>((lower_level + upper_level) / 2));
    }

    const size_t max_random_size = 1024 * 1024 + 4321;

    const unsigned int         seed = 123;
    std::default_random_engine gen(seed);
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

// Cache for input data when multiple cases must be benchmarked with various configurations and
// same inputs can be used for consecutive benchmarks.
// It must be used as a singleton.
class input_cache
{
public:
    ~input_cache()
    {
        clear();
    }

    void clear()
    {
        for(auto& i : cache)
        {
            HIP_CHECK(hipFree(i.second));
        }
        cache.clear();
    }

    // The function returns an exisitng buffer if main_key matches and there is additional_key
    // in the cache or generates a new buffer using gen().
    // If main_key does not match, it frees all device buffers and resets the cache.
    template<typename T, typename F>
    T* get_or_generate(const std::string& main_key,
                       const std::string& additional_key,
                       size_t             size,
                       F                  gen)
    {
        if(this->main_key != main_key)
        {
            // The main key (for example, data type) has been changed, clear the cache
            clear();
            this->main_key = main_key;
        }

        auto result = cache.find(additional_key);
        if(result != cache.end())
        {
            return reinterpret_cast<T*>(result->second);
        }

        // Generate a new buffer
        std::vector<T> data = gen();
        T*             d_buffer;
        HIP_CHECK(hipMalloc(&d_buffer, size * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_buffer, data.data(), size * sizeof(T), hipMemcpyHostToDevice));
        cache[additional_key] = d_buffer;
        return d_buffer;
    }

    static input_cache& instance()
    {
        static input_cache instance;
        return instance;
    }

private:
    std::string                  main_key;
    std::map<std::string, void*> cache;
};

template<typename Config>
std::string config_name()
{
    const rocprim::detail::histogram_config_params config = Config();
    return "{bs:" + std::to_string(config.histogram_config.block_size)
           + ",ipt:" + std::to_string(config.histogram_config.items_per_thread)
           + ",max_grid_size:" + std::to_string(config.max_grid_size)
           + ",shared_impl_max_bins:" + std::to_string(config.shared_impl_max_bins)
           + ",shared_impl_histograms:" + std::to_string(config.shared_impl_histograms) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T, unsigned int Channels, unsigned int ActiveChannels, typename Config>
struct device_histogram_benchmark : public config_autotune_interface
{
    std::vector<unsigned int> cases;

    device_histogram_benchmark(const std::vector<unsigned int>& cases) : cases(cases) {}

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:histogram,value_type:" + std::string(Traits<T>::name()) + ",channels:"
            + std::to_string(Channels) + ",active_channels:" + std::to_string(ActiveChannels)
            + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 3;
    static constexpr unsigned int warmup_size = 5;

    void run(benchmark::State& state,
             const std::size_t full_size,
             const hipStream_t stream) const override
    {
        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

        struct case_data
        {
            level_type   lower_level[ActiveChannels];
            level_type   upper_level[ActiveChannels];
            unsigned int num_levels[ActiveChannels];
            T*           d_input;
        };

        const std::size_t size = full_size / Channels;

        size_t        temporary_storage_bytes = 0;
        void*         d_temporary_storage     = nullptr;
        counter_type* d_histogram[ActiveChannels];
        unsigned int  max_bins = 0;

        std::vector<case_data> cases_data;
        for(auto& bins : cases)
        {
            for(int entropy_reduction : {0, 2, 4, 6})
            {
                case_data data;

                // Reuse inputs for the same sample type. This autotune uses multipe inputs for all
                // combinations of bins and entropy, but the inputs do not depend on autotuned
                // params (bs, ipt, shared_impl_max_bins) and can be reused saving time needed for
                // generating and copying to device.
                data.d_input = input_cache::instance().get_or_generate<T>(
                    std::string(Traits<T>::name()),
                    std::to_string(bins) + "_" + std::to_string(entropy_reduction),
                    full_size,
                    [&]() { return generate<T>(full_size, entropy_reduction, 0, bins); });

                for(unsigned int channel = 0; channel < ActiveChannels; channel++)
                {
                    data.lower_level[channel] = 0;
                    data.upper_level[channel] = bins;
                    data.num_levels[channel]  = bins + 1;
                }
                cases_data.push_back(data);

                size_t current_temporary_storage_bytes = 0;
                HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels, Config>(
                    d_temporary_storage,
                    current_temporary_storage_bytes,
                    data.d_input,
                    size,
                    d_histogram,
                    data.num_levels,
                    data.lower_level,
                    data.upper_level,
                    stream,
                    false)));

                temporary_storage_bytes
                    = std::max(temporary_storage_bytes, current_temporary_storage_bytes);
                max_bins = std::max(max_bins, bins);
            }
        }

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipMalloc(&d_histogram[channel], max_bins * sizeof(counter_type)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            for(auto& data : cases_data)
            {
                HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels, Config>(
                    d_temporary_storage,
                    temporary_storage_bytes,
                    data.d_input,
                    size,
                    d_histogram,
                    data.num_levels,
                    data.lower_level,
                    data.upper_level,
                    stream,
                    false)));
            }
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

            for(auto& data : cases_data)
            {
                for(size_t i = 0; i < batch_size; i++)
                {
                    HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels, Config>(
                        d_temporary_storage,
                        temporary_storage_bytes,
                        data.d_input,
                        size,
                        d_histogram,
                        data.num_levels,
                        data.lower_level,
                        data.upper_level,
                        stream,
                        false)));
                }
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

        state.SetBytesProcessed(state.iterations() * cases_data.size() * batch_size * size
                                * Channels * sizeof(T));
        state.SetItemsProcessed(state.iterations() * cases_data.size() * batch_size * size
                                * Channels);

        HIP_CHECK(hipFree(d_temporary_storage));
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            HIP_CHECK(hipFree(d_histogram[channel]));
        }
    }
};

template<typename T, unsigned int BlockSize>
struct device_histogram_benchmark_generator
{
    static constexpr unsigned int min_items_per_thread       = 1;
    static constexpr unsigned int max_items_per_thread       = 16;
    static constexpr unsigned int min_shared_impl_histograms = 2;
    static constexpr unsigned int max_shared_impl_histograms = 4;

    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        template<unsigned int SharedImplHistograms>
        struct create_shared_impl_histograms
        {
            using generated_config
                = rocprim::histogram_config<rocprim::kernel_config<BlockSize, ItemsPerThread>,
                                            2048,
                                            2048,
                                            SharedImplHistograms>;

            template<unsigned int Channels,
                     unsigned int ActiveChannels,
                     unsigned int items_per_thread = ItemsPerThread>
            auto create(std::vector<std::unique_ptr<config_autotune_interface>>& storage,
                        const std::vector<unsigned int>&                         cases) ->
                typename std::enable_if<(items_per_thread * Channels <= max_items_per_thread),
                                        void>::type
            {
                storage.emplace_back(
                    std::make_unique<
                        device_histogram_benchmark<T, Channels, ActiveChannels, generated_config>>(
                        cases));
            }

            template<unsigned int Channels,
                     unsigned int ActiveChannels,
                     unsigned int items_per_thread = ItemsPerThread>
            auto create(std::vector<std::unique_ptr<config_autotune_interface>>& storage,
                        const std::vector<unsigned int>&                         cases) ->
                typename std::enable_if<!(items_per_thread * Channels <= max_items_per_thread),
                                        void>::type
            {}

            void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage,
                            const std::vector<unsigned int>&                         cases)
            {
                // Tune histograms for single-channel data (histogram_even)
                create<1, 1>(storage, cases);
                // and some multi-channel configurations (multi_histogram_even)
                create<2, 2>(storage, cases);
                create<3, 3>(storage, cases);
                create<4, 4>(storage, cases);
                create<4, 3>(storage, cases);
            }
        };

        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage,
                        const std::vector<unsigned int>&                         cases)
        {
            static_for_each<make_index_range<unsigned int,
                                             min_shared_impl_histograms,
                                             max_shared_impl_histograms>,
                            create_shared_impl_histograms>(storage, cases);
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        // Benchmark multiple cases (with various sample distributions) and use sum of all cases
        // as a measurement for autotuning
        std::vector<unsigned int> cases;
        if(std::is_same<T, int8_t>::value)
        {
            cases = {16, 127};
        }
        else
        {
            cases = {
                10,
                100,
                1000,
                10000 // Multiple bins to trigger a global memory implementation
            };
        }
        static_for_each<make_index_range<unsigned int, min_items_per_thread, max_items_per_thread>,
                        create_ipt>(storage, cases);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_HISTOGRAM_PARALLEL_HPP_
