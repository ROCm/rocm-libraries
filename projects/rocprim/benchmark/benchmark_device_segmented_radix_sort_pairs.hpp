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

    void run_benchmark(primbench::state&& state, size_t num_segments, size_t mean_segment_length)
    {
        const auto& stream = state.stream;
        const auto& seed   = state.seed;

        using offset_type = int;
        using key_type    = Key;
        using value_type  = Value;

        primbench::log("Creating offsets");
        std::vector<offset_type> offsets;
        offsets.push_back(0);

        primbench::log("Creating gen");
        static constexpr int iseed = 716;
        engine_type          gen(iseed);

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
            const offset_type segment_length = static_cast<offset_type>(segment_length_candidate);
            offset += segment_length;
            offsets.push_back(offset);
            ++segment_index;
        }
        const size_t items          = offset;
        const size_t segments_count = offsets.size() - 1;

        primbench::log("Generating keys_input");
        std::vector<key_type> keys_input
            = get_random_data<key_type>(items,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        primbench::log("Generating values_input");
        std::vector<value_type> values_input
            = get_random_data<value_type>(items,
                                          common::generate_limits<value_type>::min(),
                                          common::generate_limits<value_type>::max(),
                                          seed);

        primbench::log("Creating d_offsets");
        common::device_ptr<offset_type> d_offsets(offsets);

        primbench::log("Creating d_keys_input");
        common::device_ptr<key_type> d_keys_input(keys_input);
        primbench::log("Creating d_keys_output");
        common::device_ptr<key_type> d_keys_output(items);

        primbench::log("Creating d_values_input");
        common::device_ptr<value_type> d_values_input(values_input);
        primbench::log("Creating d_values_output");
        common::device_ptr<value_type> d_values_output(items);

        primbench::log("Calculating d_temporary_storage size");
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(nullptr,
                                                              temporary_storage_bytes,
                                                              d_keys_input.get(),
                                                              d_keys_output.get(),
                                                              d_values_input.get(),
                                                              d_values_output.get(),
                                                              items,
                                                              segments_count,
                                                              d_offsets.get(),
                                                              d_offsets.get() + 1,
                                                              0,
                                                              sizeof(key_type) * 8,
                                                              stream,
                                                              false));

        primbench::log("Resizing d_temporary_storage");
        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);
        state.add_reads<value_type>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys_input.get(),
                                                                      d_keys_output.get(),
                                                                      d_values_input.get(),
                                                                      d_values_output.get(),
                                                                      items,
                                                                      segments_count,
                                                                      d_offsets.get(),
                                                                      d_offsets.get() + 1,
                                                                      0,
                                                                      sizeof(key_type) * 8,
                                                                      stream,
                                                                      false));
            });
    }

    void run(primbench::state& state) override
    {
        if(m_segment_counts.size() == 1)
        {
            run_benchmark(std::forward<primbench::state>(state),
                          m_segment_counts[0],
                          m_segment_lengths[0]);
            return;
        }

        constexpr size_t min_size = 300000;
        constexpr size_t max_size = 33554432;

        // TODO: Replace with KernelTuner-based autotuning that generates one benchmark per segment_count+length combo.
        for(const auto segment_count : m_segment_counts)
        {
            for(const auto segment_length : m_segment_lengths)
            {
                const auto number_of_elements = segment_count * segment_length;
                if(number_of_elements < min_size || number_of_elements > max_size)
                {
                    continue;
                }

                run_benchmark(std::forward<primbench::state>(state), segment_count, segment_length);
            }
        }
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
