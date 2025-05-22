// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_RADIX_SORT_PAIRS_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_RADIX_SORT_PAIRS_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
std::string warp_sort_config_name(T const& warp_sort_config)
{
    return "{pa:" + std::to_string(warp_sort_config.partitioning_allowed)
           + ",lwss:" + std::to_string(warp_sort_config.logical_warp_size_small)
           + ",ipts:" + std::to_string(warp_sort_config.items_per_thread_small)
           + ",bss:" + std::to_string(warp_sort_config.block_size_small)
           + ",pt:" + std::to_string(warp_sort_config.partitioning_threshold)
           + ",lwsm:" + std::to_string(warp_sort_config.logical_warp_size_medium)
           + ",iptm:" + std::to_string(warp_sort_config.items_per_thread_medium)
           + ",bsm:" + std::to_string(warp_sort_config.block_size_medium) + "}";
}

template<typename Config>
std::string config_name()
{
    const rocprim::detail::segmented_radix_sort_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread)
           + ",rb:" + std::to_string(config.radix_bits)
           + ",eupws:" + std::to_string(config.enable_unpartitioned_warp_sort)
           + ",wsc:" + warp_sort_config_name(config.warp_sort_config) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename Key, typename Value, typename Config = rocprim::default_config>
struct device_segmented_radix_sort_pairs_benchmark : public benchmark_utils::autotune_interface
{
private:
    std::vector<size_t> segment_counts;
    std::vector<size_t> segment_lengths;
    size_t              total_size;

public:
    device_segmented_radix_sort_pairs_benchmark(size_t segment_count, size_t segment_length)
    {
        segment_counts.push_back(segment_count);
        segment_lengths.push_back(segment_length);
    }

    device_segmented_radix_sort_pairs_benchmark(const std::vector<size_t>& segment_counts,
                                                const std::vector<size_t>& segment_lengths)
    {
        this->segment_counts  = segment_counts;
        this->segment_lengths = segment_lengths;
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:segmented_radix_sort,key_type:" + std::string(Traits<Key>::name())
            + ",value_type:" + std::string(Traits<Value>::name())
            + (segment_counts.size() == 1 ? ",segment_count:"s + std::to_string(segment_counts[0])
                                          : ""s)
            + (segment_lengths.size() == 1
                   ? ",segment_length:"s + std::to_string(segment_lengths[0])
                   : ""s)
            + ",cfg:" + config_name<Config>() + "}");
    }

    void run_benchmark(benchmark_utils::state&& state,
                       size_t                   num_segments,
                       size_t                   mean_segment_length)
    {
        const auto& stream = state.stream;
        const auto& seed   = state.seed;

        using offset_type = int;
        using key_type    = Key;
        using value_type  = Value;

        std::vector<offset_type> offsets;
        offsets.push_back(0);

        static constexpr int iseed = 716;
        engine_type          gen(iseed);

        std::normal_distribution<double> segment_length_dis(
            static_cast<double>(mean_segment_length),
            0.1 * mean_segment_length);

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
        const size_t size           = offset;
        const size_t segments_count = offsets.size() - 1;

        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        std::vector<value_type> values_input
            = get_random_data<value_type>(size,
                                          common::generate_limits<value_type>::min(),
                                          common::generate_limits<value_type>::max(),
                                          seed.get_0());

        common::device_ptr<offset_type> d_offsets(offsets);

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(size);

        common::device_ptr<value_type> d_values_input(values_input);
        common::device_ptr<value_type> d_values_output(size);

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(nullptr,
                                                              temporary_storage_bytes,
                                                              d_keys_input.get(),
                                                              d_keys_output.get(),
                                                              d_values_input.get(),
                                                              d_values_output.get(),
                                                              size,
                                                              segments_count,
                                                              d_offsets.get(),
                                                              d_offsets.get() + 1,
                                                              0,
                                                              sizeof(key_type) * 8,
                                                              stream,
                                                              false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(d_temporary_storage.get(),
                                                                      temporary_storage_bytes,
                                                                      d_keys_input.get(),
                                                                      d_keys_output.get(),
                                                                      d_values_input.get(),
                                                                      d_values_output.get(),
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets.get(),
                                                                      d_offsets.get() + 1,
                                                                      0,
                                                                      sizeof(key_type) * 8,
                                                                      stream,
                                                                      false));
            });

        total_size += size;
    }

    void run(benchmark_utils::state&& state) override
    {
        total_size = 0;

        if(segment_counts.size() == 1)
        {
            run_benchmark(std::forward<benchmark_utils::state>(state),
                          segment_counts[0],
                          segment_lengths[0]);
        }
        else
        {
            state.accumulate_total_gbench_iterations_every_run();

            constexpr size_t min_size = 300000;
            constexpr size_t max_size = 33554432;

            for(const auto segment_count : segment_counts)
            {
                for(const auto segment_length : segment_lengths)
                {
                    const auto number_of_elements = segment_count * segment_length;
                    if(number_of_elements < min_size || number_of_elements > max_size)
                    {
                        continue;
                    }

                    run_benchmark(std::forward<benchmark_utils::state>(state),
                                  segment_count,
                                  segment_length);
                }
            }
        }

        state.set_throughput(total_size, sizeof(Key) + sizeof(Value));
    }
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
    static auto _create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        -> std::enable_if_t<((key_size + value_type) * BlockSize * ItemsPerThread
                             <= TUNING_SHARED_MEMORY_MAX)>
    {
        const std::vector<size_t> segment_counts{10, 100, 1000, 2500, 5000, 7500, 10000, 100000};
        const std::vector<size_t> segment_lengths{30, 256, 3000, 300000};

        storage.emplace_back(std::make_unique<device_segmented_radix_sort_pairs_benchmark<
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

    template<size_t key_size = sizeof(Key), size_t value_type = sizeof(Value)>
    static auto _create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>&)
        -> std::enable_if_t<!((key_size + value_type) * BlockSize * ItemsPerThread
                              <= TUNING_SHARED_MEMORY_MAX)>
    {}

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        _create(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_RADIX_SORT_PAIRS_PARALLEL_HPP_
