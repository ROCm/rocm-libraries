// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_histogram.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template<typename T>
std::vector<T> generate(size_t size, int entropy_reduction, int lower_level, int upper_level)
{
    if(entropy_reduction >= 5)
    {
        return std::vector<T>(size, static_cast<T>((lower_level + upper_level) / 2));
    }

    const size_t max_random_size = 1024 * 1024 + 4321;

    const unsigned int seed = 123;
    engine_type        gen(seed);
    std::vector<T>     data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]()
                  {
                      // Reduce enthropy by applying bitwise AND to random bits
                      // "An Improved Supercomputer Sorting Benchmark", 1992
                      // Kurt Thearling & Stephen Smith
                      auto v = gen();
                      for(int e = 0; e < entropy_reduction; ++e)
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
template<typename T>
class input_cache
{
public:
    ~input_cache()
    {
        clear();
    }

    void clear()
    {
        total_cache_size = 0;
        cache.clear();
    }

    // The function returns an existing buffer if main_key matches and there is additional_key
    // in the cache, or generates a new buffer using gen().
    // If main_key does not match, it frees all device buffers and resets the cache.
    template<typename F>
    T* get_or_generate(const std::string& main_key, const std::string& additional_key, F gen)
    {
        // Experimentally determined maximum size, before the GPU runs out of memory.
        static constexpr short max_default_bytes_count = 88;
        if(this->main_key != main_key)
        {
            // The main key (for example, data type) has been changed, clear the cache
            clear();
            this->main_key = main_key;
        }

        auto result = cache.find(additional_key);
        if(result != cache.end())
        {
            return reinterpret_cast<T*>(result->second.get());
        }

        // Generate a new buffer
        std::vector<T> data = gen();
        common::device_ptr<T> d_buffer;
        if(total_cache_size >= max_default_bytes_count)
        {
            // the memory space of the value of last key-value pair is held by d_buffer
            // and the pair is erased from the cache map
            auto iter = cache.end();
            --iter;
            d_buffer = std::move(iter->second);
            cache.erase(iter);
        }
        else
        {
            // it will generate a new memory space to store in cache
            // so records the new size in advance
            total_cache_size += sizeof(T);
        }
        d_buffer.store(data);
        cache[additional_key] = std::move(d_buffer);
        return cache[additional_key].get();
    }

    static input_cache& instance()
    {
        static input_cache instance;
        return instance;
    }

private:
    std::string                  main_key;
    std::map<std::string, common::device_ptr<T>> cache;
    short                        total_cache_size = 0;
};

template<typename Config>
std::string config_name()
{
    const rocprim::detail::histogram_config_params config = Config();
    return "{bs:" + std::to_string(config.histogram_config.block_size)
           + ",ipt:" + std::to_string(config.histogram_config.items_per_thread)
           + ",max_grid_size:" + std::to_string(config.max_grid_size)
           + ",shared_impl_max_bins:" + std::to_string(config.shared_impl_max_bins)
           + ",shared_impl_histograms:" + std::to_string(config.shared_impl_histograms)
           + ",global_hist_bs:" + std::to_string(config.histogram_global_config.block_size)
           + ",global_hist_ipt:" + std::to_string(config.histogram_global_config.items_per_thread)
           + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T,
         unsigned int Channels,
         unsigned int ActiveChannels,
         typename Config = rocprim::default_config>
struct device_histogram_benchmark : public benchmark_utils::autotune_interface
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

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;

        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<std::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

        struct case_data
        {
            unsigned int bins;
            int          entropy_reduction;
            level_type   lower_level[ActiveChannels]{};
            level_type   upper_level[ActiveChannels]{};
            unsigned int num_levels[ActiveChannels]{};
            T*           get_d_input(size_t bytes)
            {
                return input_cache<T>::instance().get_or_generate(
                    std::string(Traits<T>::name()),
                    std::to_string(bins) + "_" + std::to_string(entropy_reduction),
                    [&]() { return generate<T>(bytes, entropy_reduction, 0, bins); });
            };
        };

        const std::size_t size = bytes / Channels;

        size_t        temporary_storage_bytes = 0;
        counter_type* d_histogram[ActiveChannels];
        unsigned int  max_bins = 0;

        std::vector<case_data> cases_data;
        for(const auto& bins : cases)
        {
            for(int entropy_reduction : {0, 2, 4, 6})
            {
                case_data data = {bins, entropy_reduction};

                // Reuse inputs for the same sample type. This autotune uses multipe inputs for all
                // combinations of bins and entropy, but the inputs do not depend on autotuned
                // params (bs, ipt, shared_impl_max_bins) and can be reused saving time needed for
                // generating and copying to device.

                for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
                {
                    data.lower_level[channel] = 0;
                    data.upper_level[channel] = bins;
                    data.num_levels[channel]  = bins + 1;
                }
                cases_data.push_back(data);

                size_t current_temporary_storage_bytes = 0;
                HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels, Config>(
                    nullptr,
                    current_temporary_storage_bytes,
                    data.get_d_input(bytes),
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

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            HIP_CHECK(hipMalloc(&d_histogram[channel], max_bins * sizeof(counter_type)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        size_t total_size = 0;

        for(auto& data : cases_data)
        {
            T* d_input = data.get_d_input(bytes);

            state.run(
                [&]
                {
                    HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels, Config>(
                        d_temporary_storage.get(),
                        temporary_storage_bytes,
                        d_input,
                        size,
                        d_histogram,
                        data.num_levels,
                        data.lower_level,
                        data.upper_level,
                        stream,
                        false)));
                });

            total_size += size * Channels;
        }

        state.set_throughput(total_size, sizeof(T));

        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
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
                                            SharedImplHistograms,
                                            rocprim::kernel_config<1024, 4>>;

            template<unsigned int Channels,
                     unsigned int ActiveChannels,
                     unsigned int items_per_thread = ItemsPerThread>
            auto create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage,
                        const std::vector<unsigned int>&                                   cases) ->
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
            auto create(
                std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& /*storage*/,
                const std::vector<unsigned int>& /*cases*/) ->
                typename std::enable_if<!(items_per_thread * Channels <= max_items_per_thread),
                                        void>::type
            {}

            void operator()(
                std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage,
                const std::vector<unsigned int>&                                   cases)
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

        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage,
                        const std::vector<unsigned int>&                                   cases)
        {
            static_for_each<make_index_range<unsigned int,
                                             min_shared_impl_histograms,
                                             max_shared_impl_histograms>,
                            create_shared_impl_histograms>(storage, cases);
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
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
