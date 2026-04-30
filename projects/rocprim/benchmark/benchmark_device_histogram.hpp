// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/device_histogram.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

template<typename T>
std::vector<T> generate(size_t items, int entropy_reduction, int lower_level, int upper_level)
{
    if(entropy_reduction >= 5)
    {
        return std::vector<T>(items, static_cast<T>((lower_level + upper_level) / 2));
    }

    const size_t max_random_size = 1024 * 1024 + 4321;

    const unsigned int seed = 123;
    engine_type        gen(seed);
    std::vector<T>     data(items);
    std::generate(data.begin(),
                  data.begin() + std::min(items, max_random_size),
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
    for(size_t i = max_random_size; i < items; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(items - i, max_random_size), data.begin() + i);
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
    T* get_or_generate(std::string_view main_key, std::string_view additional_key, F gen)
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
        std::vector<T>        data = gen();
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
        auto it = cache.emplace(std::string(additional_key), std::move(d_buffer)).first;
        return it->second.get();
    }

    static input_cache& instance()
    {
        static input_cache instance;
        return instance;
    }

private:
    std::string                                               main_key;
    std::map<std::string, common::device_ptr<T>, std::less<>> cache;
    short                                                     total_cache_size = 0;
};

template<typename Config>
constexpr auto config_name()
{
    if constexpr(std::is_same_v<Config, rocprim::default_config>)
    {
        return std::string("default");
    }
    else
    {
        constexpr rocprim::detail::histogram_config_params config = Config();

        return primbench::json{}
            .add("bs", config.histogram_config.block_size)
            .add("ipt", config.histogram_config.items_per_thread)
            .add("max_grid_size", config.max_grid_size)
            .add("shared_impl_max_bins", config.shared_impl_max_bins)
            .add("shared_impl_histograms", config.shared_impl_histograms)
            .add("global_hist_bs", config.histogram_global_config.block_size)
            .add("global_hist_ipt", config.histogram_global_config.items_per_thread);
    }
}

inline int get_entropy_percents(int entropy_reduction)
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

template<typename T, typename Config = rocprim::default_config>
struct device_histogram_even_benchmark : public primbench::benchmark_interface
{
    device_histogram_even_benchmark(size_t bins, size_t scale, int entropy_reduction)
        : m_bins(bins), m_scale(scale), m_entropy_reduction(entropy_reduction)
    {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_histogram")
            .add("subalgo", "even")
            .add("value_type", primbench::name<T>())
            .add("bins", m_bins)
            .add("scale", m_scale)
            .add("entropy", get_entropy_percents(m_entropy_reduction))
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        size_t items = bytes / sizeof(T);

        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<rocprim::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

        const level_type lower_level = 0;
        const level_type upper_level = m_bins * m_scale;

        // Generate data
        std::vector<T> input = generate<T>(items, m_entropy_reduction, lower_level, upper_level);

        common::device_ptr<T>            d_input(input);
        common::device_ptr<counter_type> d_histogram(m_bins);

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::histogram_even(nullptr,
                                          temporary_storage_bytes,
                                          d_input.get(),
                                          items,
                                          d_histogram.get(),
                                          m_bins + 1,
                                          lower_level,
                                          upper_level,
                                          stream,
                                          false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<T>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::histogram_even(d_temporary_storage.get(),
                                                  temporary_storage_bytes,
                                                  d_input.get(),
                                                  items,
                                                  d_histogram.get(),
                                                  m_bins + 1,
                                                  lower_level,
                                                  upper_level,
                                                  stream,
                                                  false));
            });
    }

private:
    size_t m_bins;
    size_t m_scale;
    int    m_entropy_reduction;
};

template<typename T,
         unsigned int Channels,
         unsigned int ActiveChannels,
         bool         IsAutotuning = false,
         typename Config           = rocprim::default_config>
struct device_multi_histogram_even_benchmark : public primbench::benchmark_interface
{
    device_multi_histogram_even_benchmark(size_t bins, size_t scale, int entropy_reduction)
        : m_bins(bins), m_scale(scale), m_entropy_reduction(entropy_reduction)
    {}

    device_multi_histogram_even_benchmark(const std::vector<unsigned int>& cases) : m_cases(cases)
    {}

    primbench::json meta() const override
    {
        auto j = primbench::json{}
                     .add("lvl", "device")
                     .add("algo", "device_histogram")
                     .add("subalgo", "multi_even")
                     .add("value_type", primbench::name<T>())
                     .add("channels", Channels)
                     .add("active_channels", ActiveChannels)
                     .add("cfg", config_name<Config>());

        if constexpr(!IsAutotuning)
        {
            j.add("bins", m_bins);
            j.add("scale", m_scale);
            j.add("entropy", get_entropy_percents(m_entropy_reduction));
        }

        return j;
    }

    void run(primbench::state& state) override
    {
        if constexpr(IsAutotuning)
        {
            run_autotune(state);
        }
        else
        {
            run_benchmark(state);
        }
    }

private:
    void run_benchmark(primbench::state& state)
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        size_t items = bytes / sizeof(T);

        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<rocprim::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

        unsigned int num_levels[ActiveChannels];
        level_type   lower_level[ActiveChannels];
        level_type   upper_level[ActiveChannels];
        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            lower_level[channel] = 0;
            upper_level[channel] = m_bins * m_scale;
            num_levels[channel]  = m_bins + 1;
        }

        // Generate data
        std::vector<T> input
            = generate<T>(items * Channels, m_entropy_reduction, lower_level[0], upper_level[0]);

        common::device_ptr<T> d_input(input);
        counter_type*         d_histogram[ActiveChannels];
        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            HIP_CHECK(hipMalloc(&d_histogram[channel], m_bins * sizeof(counter_type)));
        }

        size_t temporary_storage_bytes = 0;
        HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels>(nullptr,
                                                                           temporary_storage_bytes,
                                                                           d_input.get(),
                                                                           items,
                                                                           d_histogram,
                                                                           num_levels,
                                                                           lower_level,
                                                                           upper_level,
                                                                           stream,
                                                                           false)));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<T>(items * Channels);

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::multi_histogram_even<Channels, ActiveChannels>(
                    d_temporary_storage.get(),
                    temporary_storage_bytes,
                    d_input.get(),
                    items,
                    d_histogram,
                    num_levels,
                    lower_level,
                    upper_level,
                    stream,
                    false)));
            });

        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            HIP_CHECK(hipFree(d_histogram[channel]));
        }
    }

    void run_autotune(primbench::state& state)
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<rocprim::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

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
                    primbench::name<T>(),
                    std::to_string(bins) + "_" + std::to_string(entropy_reduction),
                    [&]() { return generate<T>(bytes, entropy_reduction, 0, bins); });
            };
        };

        // Clear caches for other types that are either empty or already done.
        clear_other_caches<rocprim::int128_t,
                           int64_t,
                           int,
                           short,
                           int8_t,
                           double,
                           float,
                           rocprim::half>();

        const std::size_t items = bytes / Channels;

        size_t        temporary_storage_bytes = 0;
        counter_type* d_histogram[ActiveChannels];
        unsigned int  max_bins = 0;

        std::vector<case_data> cases_data;
        for(const auto& bins : m_cases)
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
                    items,
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

        state.set_items(items);

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
                        items,
                        d_histogram,
                        data.num_levels,
                        data.lower_level,
                        data.upper_level,
                        stream,
                        false)));
                });

            state.add_reads<T>(items * Channels);
        }

        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            HIP_CHECK(hipFree(d_histogram[channel]));
        }
    }

    template<class... Args>
    void clear_other_caches()
    {
        (
            [](auto u)
            {
                using U = decltype(u);
                if(!std::is_same_v<T, U>)
                {
                    input_cache<U>::instance().clear();
                }
            }(Args{}),
            ...);
    }

    size_t                    m_bins;
    size_t                    m_scale;
    int                       m_entropy_reduction;
    std::vector<unsigned int> m_cases;
};

template<typename T, typename Config = rocprim::default_config>
struct device_histogram_range_benchmark : public primbench::benchmark_interface
{
    device_histogram_range_benchmark(size_t bins) : m_bins(bins) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_histogram")
            .add("subalgo", "range")
            .add("value_type", primbench::name<T>())
            .add("bins", m_bins)
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        size_t items = bytes / sizeof(T);

        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<rocprim::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

        // Generate data
        const auto     random_range = limit_random_range<T>(0, m_bins);
        std::vector<T> input
            = get_random_data<T>(items, random_range.first, random_range.second, seed);

        std::vector<level_type> levels(m_bins + 1);
        for(size_t i = 0; i < levels.size(); ++i)
        {
            levels[i] = static_cast<level_type>(i);
        }

        common::device_ptr<T>            d_input(input);
        common::device_ptr<level_type>   d_levels(levels);
        common::device_ptr<counter_type> d_histogram(m_bins);

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::histogram_range(nullptr,
                                           temporary_storage_bytes,
                                           d_input.get(),
                                           items,
                                           d_histogram.get(),
                                           m_bins + 1,
                                           d_levels.get(),
                                           stream,
                                           false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<T>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::histogram_range(d_temporary_storage.get(),
                                                   temporary_storage_bytes,
                                                   d_input.get(),
                                                   items,
                                                   d_histogram.get(),
                                                   m_bins + 1,
                                                   d_levels.get(),
                                                   stream,
                                                   false));
            });
    }

private:
    size_t m_bins;
};

template<typename T,
         unsigned int Channels,
         unsigned int ActiveChannels,
         typename Config = rocprim::default_config>
struct device_multi_histogram_range_benchmark : public primbench::benchmark_interface
{
    device_multi_histogram_range_benchmark(size_t bins) : m_bins(bins) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_histogram")
            .add("subalgo", "multi_range")
            .add("value_type", primbench::name<T>())
            .add("channels", Channels)
            .add("active_channels", ActiveChannels)
            .add("bins", m_bins)
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        size_t items = bytes / sizeof(T);

        using counter_type = unsigned int;
        using level_type   = typename std::
            conditional_t<rocprim::is_integral<T>::value && sizeof(T) < sizeof(int), int, T>;

        const int               num_levels_channel = m_bins + 1;
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
        const auto     random_range = limit_random_range<T>(0, m_bins);
        std::vector<T> input
            = get_random_data<T>(items * Channels, random_range.first, random_range.second, seed);

        common::device_ptr<T> d_input(input);
        level_type*           d_levels[ActiveChannels];
        counter_type*         d_histogram[ActiveChannels];
        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            HIP_CHECK(hipMalloc(&d_levels[channel], num_levels_channel * sizeof(level_type)));
            HIP_CHECK(hipMalloc(&d_histogram[channel], m_bins * sizeof(counter_type)));
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
                                                                            items,
                                                                            d_histogram,
                                                                            num_levels,
                                                                            d_levels,
                                                                            stream,
                                                                            false)));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<T>(items * Channels);

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::multi_histogram_range<Channels, ActiveChannels>(
                    d_temporary_storage.get(),
                    temporary_storage_bytes,
                    d_input.get(),
                    items,
                    d_histogram,
                    num_levels,
                    d_levels,
                    stream,
                    false)));
            });

        for(unsigned int channel = 0; channel < ActiveChannels; ++channel)
        {
            HIP_CHECK(hipFree(d_levels[channel]));
            HIP_CHECK(hipFree(d_histogram[channel]));
        }
    }

private:
    size_t m_bins;
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
            auto create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage,
                        const std::vector<unsigned int>&                              cases) ->
                typename std::enable_if<(items_per_thread * Channels <= max_items_per_thread),
                                        void>::type
            {
                storage.emplace_back(
                    std::make_unique<device_multi_histogram_even_benchmark<T,
                                                                           Channels,
                                                                           ActiveChannels,
                                                                           true,
                                                                           generated_config>>(
                        cases));
            }

            template<unsigned int Channels,
                     unsigned int ActiveChannels,
                     unsigned int items_per_thread = ItemsPerThread>
            auto create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& /*storage*/,
                        const std::vector<unsigned int>& /*cases*/) ->
                typename std::enable_if<!(items_per_thread * Channels <= max_items_per_thread),
                                        void>::type
            {}

            void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage,
                            const std::vector<unsigned int>&                              cases)
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

        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage,
                        const std::vector<unsigned int>&                              cases)
        {
            static_for_each<make_index_range<unsigned int,
                                             min_shared_impl_histograms,
                                             max_shared_impl_histograms>,
                            create_shared_impl_histograms>(storage, cases);
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
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
