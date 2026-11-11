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
         typename Config = rocprim::default_config>
struct device_multi_histogram_even_benchmark : public primbench::benchmark_interface
{
    device_multi_histogram_even_benchmark(size_t bins, size_t scale, int entropy_reduction)
        : m_bins(bins), m_scale(scale), m_entropy_reduction(entropy_reduction)
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
                     .add("cfg", config_name<Config>())
                     .add("bins", m_bins)
                     .add("scale", m_scale)
                     .add("entropy", get_entropy_percents(m_entropy_reduction));

        return j;
    }

    void run(primbench::state& state) override
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

private:
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
