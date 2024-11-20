// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>

#include <string>
#include <vector>

#include <cstddef>

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
           + ",lrb:" + std::to_string(config.long_radix_bits)
           + ",srb:" + std::to_string(config.short_radix_bits)
           + ",eupws:" + std::to_string(config.enable_unpartitioned_warp_sort)
           + ",wsc:" + warp_sort_config_name(config.warp_sort_config) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename Key, typename Value, typename Config>
struct device_segmented_radix_sort_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        const rocprim::detail::segmented_radix_sort_config_params config = Config();
        return bench_naming::format_name("{lvl:device,algo:segmented_radix_sort,key_type:"
                                         + std::string(Traits<Key>::name())
                                         + ",value_type:" + std::string(Traits<Value>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    void run_benchmark(benchmark::State&   state,
                       size_t              num_segments,
                       size_t              mean_segment_length,
                       size_t              target_size,
                       const managed_seed& seed,
                       hipStream_t         stream) const
    {
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
                                        generate_limits<key_type>::min(),
                                        generate_limits<key_type>::max(),
                                        seed.get_0());

        std::vector<value_type> values_input
            = get_random_data<value_type>(size,
                                          generate_limits<value_type>::min(),
                                          generate_limits<value_type>::max(),
                                          seed.get_0());

        size_t batch_size = 1;
        if(size < target_size)
        {
            batch_size = (target_size + size - 1) / size;
        }

        offset_type* d_offsets;
        HIP_CHECK(hipMalloc(&d_offsets, offsets.size() * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            offsets.size() * sizeof(offset_type),
                            hipMemcpyHostToDevice));

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        value_type* d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(d_temporary_storage,
                                                              temporary_storage_bytes,
                                                              d_keys_input,
                                                              d_keys_output,
                                                              d_values_input,
                                                              d_values_output,
                                                              size,
                                                              segments_count,
                                                              d_offsets,
                                                              d_offsets + 1,
                                                              0,
                                                              sizeof(key_type) * 8,
                                                              stream,
                                                              false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(d_temporary_storage,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input,
                                                                  d_keys_output,
                                                                  d_values_input,
                                                                  d_values_output,
                                                                  size,
                                                                  segments_count,
                                                                  d_offsets,
                                                                  d_offsets + 1,
                                                                  0,
                                                                  sizeof(key_type) * 8,
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
                HIP_CHECK(rocprim::segmented_radix_sort_pairs<Config>(d_temporary_storage,
                                                                      temporary_storage_bytes,
                                                                      d_keys_input,
                                                                      d_keys_output,
                                                                      d_values_input,
                                                                      d_values_output,
                                                                      size,
                                                                      segments_count,
                                                                      d_offsets,
                                                                      d_offsets + 1,
                                                                      0,
                                                                      sizeof(key_type) * 8,
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

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        // Calculate the number of elements 
        size_t size = bytes / sizeof(Key);
        
        constexpr std::array<size_t, 8>
            segment_counts{10, 100, 1000, 2500, 5000, 7500, 10000, 100000};
        constexpr std::array<size_t, 4> segment_lengths{30, 256, 3000, 300000};

        for(const auto segment_count : segment_counts)
        {
            for(const auto segment_length : segment_lengths)
            {
                const auto number_of_elements = segment_count * segment_length;
                if(number_of_elements > 33554432 || number_of_elements < 300000)
                {
                    continue;
                }

                run_benchmark(state, segment_count, segment_length, size, seed, stream);
            }
        }
    }
};

template<typename Tp, template<Tp> class T, bool enable, Tp... Idx>
struct decider;

template<unsigned int LongBits,
         unsigned int ShortBits,
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
struct device_segmented_radix_sort_benchmark_generator
{
    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        storage.emplace_back(std::make_unique<device_segmented_radix_sort_benchmark<
                                 Key,
                                 Value,
                                 rocprim::segmented_radix_sort_config<
                                     LongBits,
                                     ShortBits,
                                     rocprim::kernel_config<BlockSize, ItemsPerThread>,
                                     rocprim::WarpSortConfig<WarpSmallLWS,
                                                             WarpSmallIPT,
                                                             WarpSmallBS,
                                                             WarpPartition,
                                                             WarpMediumLWS,
                                                             WarpMediumIPT,
                                                             WarpMediumBS>,
                                     UnpartitionWarpAllowed>>>());
    }
};

template<typename Tp, template<Tp> class T, Tp... Idx>
struct decider<Tp, T, true, Idx...>
{
    inline static void
        do_the_thing(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static_for_each<std::integer_sequence<Tp, Idx...>, T>(storage);
    }
};

template<typename Tp, template<Tp> class T, Tp... Idx>
struct decider<Tp, T, false, Idx...>
{
    inline static void
        do_the_thing(std::vector<std::unique_ptr<config_autotune_interface>>& /*storage*/)
    {}
};

#endif // ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_RADIX_SORT_PAIRS_PARALLEL_HPP_
