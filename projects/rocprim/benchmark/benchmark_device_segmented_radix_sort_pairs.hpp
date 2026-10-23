// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

template<typename T>
primbench::json warp_sort_config_name(T const& warp_sort_config)
{
    return primbench::json{}
        .add("pa", warp_sort_config.partitioning_allowed)
        .add("lwss", warp_sort_config.logical_warp_size_small)
        .add("ipts", warp_sort_config.items_per_thread_small)
        .add("bss", warp_sort_config.block_size_small)
        .add("pt", warp_sort_config.partitioning_threshold)
        .add("lwsm", warp_sort_config.logical_warp_size_medium)
        .add("iptm", warp_sort_config.items_per_thread_medium)
        .add("bsm", warp_sort_config.block_size_medium);
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
        constexpr rocprim::detail::segmented_radix_sort_config_params config = Config();

        return primbench::json{}
            .add("bs", config.kernel_config.block_size)
            .add("ipt", config.kernel_config.items_per_thread)
            .add("rb", config.radix_bits)
            .add("eupws", config.enable_unpartitioned_warp_sort)
            .add("wsc", warp_sort_config_name(config.warp_sort_config));
    }
}

template<typename Key, typename Value, typename Config = rocprim::default_config>
struct device_segmented_radix_sort_pairs_benchmark : public primbench::benchmark_interface
{
    device_segmented_radix_sort_pairs_benchmark(size_t segment_count, size_t segment_length)
    {
        m_segment_counts.push_back(segment_count);
        m_segment_lengths.push_back(segment_length);
    }

    device_segmented_radix_sort_pairs_benchmark(const std::vector<size_t>& segment_counts,
                                                const std::vector<size_t>& segment_lengths)
        : m_segment_counts(segment_counts), m_segment_lengths(segment_lengths)
    {}

    primbench::json meta() const override
    {
        auto j = primbench::json{}
                     .add("lvl", "device")
                     .add("algo", "device_segmented_radix_sort_pairs")
                     .add("key_type", primbench::name<Key>())
                     .add("value_type", primbench::name<Value>())
                     .add("cfg", config_name<Config>());

        if(m_segment_counts.size() == 1)
        {
            j.add("segment_count", m_segment_counts[0]);
        }
        if(m_segment_lengths.size() == 1)
        {
            j.add("segment_length", m_segment_lengths[0]);
        }

        return j;
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& seed   = state.seed;

        using offset_type = int;
        using key_type    = Key;
        using value_type  = Value;

        // Collect the (num_segments, mean_segment_length) pairs to actually run
        std::vector<std::pair<size_t, size_t>> combos;

        if(m_segment_counts.size() == 1)
        {
            combos.emplace_back(m_segment_counts[0], m_segment_lengths[0]);
        }
        else
        {
            constexpr size_t min_size = 300000;
            constexpr size_t max_size = 33554432;

            for(const auto segment_count : m_segment_counts)
            {
                for(const auto segment_length : m_segment_lengths)
                {
                    const auto number_of_elements = segment_count * segment_length;
                    if(number_of_elements < min_size || number_of_elements > max_size)
                    {
                        continue;
                    }
                    combos.emplace_back(segment_count, segment_length);
                }
            }
        }

        const int num_input_arrays = combos.size();

        static constexpr int iseed = 716;
        engine_type          gen(iseed);

        // Build offsets + keys + values for every combo up front
        std::vector<std::vector<offset_type>> offsets_arrays(num_input_arrays);
        std::vector<std::vector<key_type>>    keys_input_arrays(num_input_arrays);
        std::vector<std::vector<value_type>>  values_input_arrays(num_input_arrays);
        std::vector<size_t>                   items_per_run(num_input_arrays);
        std::vector<size_t>                   segments_counts(num_input_arrays);
        size_t                                max_items = 0;

        for(int i = 0; i < num_input_arrays; ++i)
        {
            const size_t num_segments        = combos[i].first;
            const size_t mean_segment_length = combos[i].second;

            primbench::log("Creating offsets");
            std::vector<offset_type>& offsets = offsets_arrays[i];
            offsets.push_back(0);

            primbench::log("Generating segment_length_dis");
            std::normal_distribution<double> segment_length_dis(
                static_cast<double>(mean_segment_length),
                0.1 * mean_segment_length);

            primbench::log("Calculating offsets");
            size_t offset = 0;
            for(size_t segment_index = 0; segment_index < num_segments;)
            {
                const double segment_length_candidate = std::round(segment_length_dis(gen));
                if(segment_length_candidate < 0)
                {
                    continue;
                }
                const offset_type segment_length
                    = static_cast<offset_type>(segment_length_candidate);
                offset += segment_length;
                offsets.push_back(offset);
                ++segment_index;
            }
            const size_t items = offset;

            items_per_run[i]   = items;
            segments_counts[i] = offsets.size() - 1;
            max_items          = std::max(max_items, items);

            primbench::log("Generating keys_input");
            keys_input_arrays[i]
                = get_random_data<key_type>(items,
                                            common::generate_limits<key_type>::min(),
                                            common::generate_limits<key_type>::max(),
                                            seed);

            primbench::log("Generating values_input");
            values_input_arrays[i]
                = get_random_data<value_type>(items,
                                              common::generate_limits<value_type>::min(),
                                              common::generate_limits<value_type>::max(),
                                              seed);
        }

        primbench::log("Creating d_offsets_arrays");
        std::vector<common::device_ptr<offset_type>> d_offsets_arrays(num_input_arrays);
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_offsets_arrays[i].store(offsets_arrays[i]);
        }

        primbench::log("Creating d_keys_input_arrays");
        std::vector<common::device_ptr<key_type>> d_keys_input_arrays(num_input_arrays);
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_keys_input_arrays[i].store(keys_input_arrays[i]);
        }

        primbench::log("Creating d_values_input_arrays");
        std::vector<common::device_ptr<value_type>> d_values_input_arrays(num_input_arrays);
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_values_input_arrays[i].store(values_input_arrays[i]);
        }

        // Shared, reused across all runs (sized to the largest)
        primbench::log("Creating d_keys_output");
        common::device_ptr<key_type> d_keys_output(max_items);
        primbench::log("Creating d_values_output");
        common::device_ptr<value_type> d_values_output(max_items);

        // Single call to segmented_radix_sort_pairs for one combo, with its own storage size.
        const auto dispatch_input = [&](void*        d_temp_storage,
                                        size_t&      temp_storage_size_bytes,
                                        key_type*    d_keys_input,
                                        value_type*  d_values_input,
                                        size_t       items,
                                        size_t       segments_count,
                                        offset_type* d_offsets)
        {
            HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(d_temp_storage,
                                                                  temp_storage_size_bytes,
                                                                  d_keys_input,
                                                                  d_keys_output.get(),
                                                                  d_values_input,
                                                                  d_values_output.get(),
                                                                  items,
                                                                  segments_count,
                                                                  d_offsets,
                                                                  d_offsets + 1,
                                                                  0,
                                                                  sizeof(key_type) * 8,
                                                                  stream,
                                                                  false));
        };

        // Size each combo independently and track the max required storage.
        primbench::log("Calculating d_temporary_storage size");
        std::vector<size_t> temp_storage_bytes_per_run(num_input_arrays);
        size_t              max_temporary_storage_bytes = 0;

        for(int i = 0; i < num_input_arrays; ++i)
        {
            size_t temp_storage_size_bytes = 0;
            dispatch_input(nullptr,
                           temp_storage_size_bytes,
                           d_keys_input_arrays[i].get(),
                           d_values_input_arrays[i].get(),
                           items_per_run[i],
                           segments_counts[i],
                           d_offsets_arrays[i].get());

            temp_storage_bytes_per_run[i] = temp_storage_size_bytes;
            max_temporary_storage_bytes
                = std::max(max_temporary_storage_bytes, temp_storage_size_bytes);
        }

        primbench::log("Resizing d_temporary_storage");
        common::device_ptr<void> d_temporary_storage(max_temporary_storage_bytes);

        const size_t total_items
            = std::accumulate(items_per_run.begin(), items_per_run.end(), size_t{0});
        state.set_items(total_items);
        state.add_reads<key_type>(total_items);
        state.add_reads<value_type>(total_items);

        state.run(
            [&]
            {
                for(int i = 0; i < num_input_arrays; ++i)
                {
                    dispatch_input(d_temporary_storage.get(),
                                   temp_storage_bytes_per_run[i],
                                   d_keys_input_arrays[i].get(),
                                   d_values_input_arrays[i].get(),
                                   items_per_run[i],
                                   segments_counts[i],
                                   d_offsets_arrays[i].get());
                }
            });
    }

private:
    std::vector<size_t> m_segment_counts;
    std::vector<size_t> m_segment_lengths;
};

template<unsigned int RadixBits,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int WarpSmallLWS,
         unsigned int WarpSmallIPT,
         unsigned int WarpSmallBS,
         unsigned int WarpPartition,
         unsigned int WarpMediumLWS,
         unsigned int WarpMediumIPT,
         unsigned int WarpMediumBS,
         typename Key,
         typename Value,
         bool UnpartitionWarpAllowed = true>
struct device_segmented_radix_sort_pairs_benchmark_generator
{
    template<size_t key_size = sizeof(Key), size_t value_type = sizeof(Value)>
    static auto _create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        if constexpr((key_size + value_type) * BlockSize * ItemsPerThread
                     <= TUNING_SHARED_MEMORY_MAX)
        {
            const std::vector<size_t>
                segment_counts{10, 100, 1000, 2500, 5000, 7500, 10000, 100000};
            const std::vector<size_t> segment_lengths{30, 256, 3000, 300000};

            storage.emplace_back(
                std::make_unique<device_segmented_radix_sort_pairs_benchmark<
                    Key,
                    Value,
                    rocprim::segmented_radix_sort_config<
                        RadixBits,
                        rocprim::kernel_config<BlockSize, ItemsPerThread>,
                        rocprim::WarpSortConfig<WarpSmallLWS,
                                                WarpSmallIPT,
                                                WarpSmallBS,
                                                WarpPartition,
                                                WarpMediumLWS,
                                                WarpMediumIPT,
                                                WarpMediumBS>,
                        UnpartitionWarpAllowed>>>(segment_counts, segment_lengths));
        }
    }

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        _create(storage);
    }
};
