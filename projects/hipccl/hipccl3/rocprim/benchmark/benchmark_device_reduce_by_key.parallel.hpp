// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_REDUCE_BY_KEY_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_REDUCE_BY_KEY_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM HIP API
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>

#include <cstddef>
#include <string>
#include <vector>

template<typename Config>
std::string config_name()
{
    const rocprim::detail::reduce_by_key_config_params params = Config();
    return "{bs:" + std::to_string(params.kernel_config.block_size)
           + ",ipt:" + std::to_string(params.kernel_config.items_per_thread)
           + ",tpb:" + std::to_string(params.tiles_per_block) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename KeyType,
         typename ValueType,
         int  MaxSegmentLength,
         bool Deterministic = false,
         typename Config    = rocprim::default_config>
struct device_reduce_by_key_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name(
            "{lvl:device,algo:reduce_by_key,key_type:" + std::string(Traits<KeyType>::name())
            + ",value_type:" + std::string(Traits<ValueType>::name()) + ",max_segment_length:"
            + std::to_string(MaxSegmentLength) + ",cfg:" + config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        constexpr int                batch_size                 = 10;
        constexpr int                warmup_size                = 5;
        constexpr std::array<int, 2> tuning_max_segment_lengths = {10, 1000};
        constexpr int    num_input_arrays = is_tuning ? tuning_max_segment_lengths.size() : 1;
        constexpr size_t item_size        = sizeof(KeyType) + sizeof(ValueType);

        const size_t size = bytes / item_size;

        std::vector<KeyType> key_inputs[num_input_arrays];
        if(is_tuning)
        {
            for(size_t i = 0; i < tuning_max_segment_lengths.size(); ++i)
            {
                key_inputs[i] = get_random_segments_iota<KeyType>(size,
                                                                  tuning_max_segment_lengths[i],
                                                                  seed.get_0());
            }
        }
        else
        {
            key_inputs[0] = get_random_segments_iota<KeyType>(size, MaxSegmentLength, seed.get_0());
        }

        std::vector<ValueType> value_input(size);
        std::iota(value_input.begin(), value_input.end(), 0);

        KeyType* d_key_inputs[num_input_arrays];
        for(int i = 0; i < num_input_arrays; ++i)
        {
            HIP_CHECK(hipMalloc(&d_key_inputs[i], size * sizeof(*d_key_inputs[i])));
            HIP_CHECK(hipMemcpy(d_key_inputs[i],
                                key_inputs[i].data(),
                                size * sizeof(*d_key_inputs[i]),
                                hipMemcpyHostToDevice));
        }

        ValueType* d_value_input;
        HIP_CHECK(hipMalloc(&d_value_input, size * sizeof(*d_value_input)));
        HIP_CHECK(hipMemcpy(d_value_input,
                            value_input.data(),
                            size * sizeof(*d_value_input),
                            hipMemcpyHostToDevice));

        KeyType*      d_unique_output;
        ValueType*    d_aggregates_output;
        unsigned int* d_unique_count_output;
        HIP_CHECK(hipMalloc(&d_unique_output, size * sizeof(*d_unique_output)));
        HIP_CHECK(hipMalloc(&d_aggregates_output, size * sizeof(*d_aggregates_output)));
        HIP_CHECK(hipMalloc(&d_unique_count_output, sizeof(*d_unique_count_output)));

        rocprim::plus<ValueType>   reduce_op;
        rocprim::equal_to<KeyType> key_compare_op;

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_input = [&](KeyType* d_key_input)
            {
                if ROCPRIM_IF_CONSTEXPR(!Deterministic)
                {
                    HIP_CHECK(rocprim::reduce_by_key<Config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             d_key_input,
                                                             d_value_input,
                                                             size,
                                                             d_unique_output,
                                                             d_aggregates_output,
                                                             d_unique_count_output,
                                                             reduce_op,
                                                             key_compare_op,
                                                             stream));
                }
                else
                {
                    HIP_CHECK(rocprim::deterministic_reduce_by_key<Config>(d_temp_storage,
                                                                           temp_storage_size_bytes,
                                                                           d_key_input,
                                                                           d_value_input,
                                                                           size,
                                                                           d_unique_output,
                                                                           d_aggregates_output,
                                                                           d_unique_count_output,
                                                                           reduce_op,
                                                                           key_compare_op,
                                                                           stream));
                }
            };

            // One tuning iteration runs multiple inputs with different distributions,
            //   preventing overfitting the config to a specific data distrubution.
            //   Note that this does not weigh the inputs/distributions equally as
            //   generally larger segments perform better.
            for(int i = 0; i < num_input_arrays; ++i)
            {
                dispatch_input(d_key_inputs[i]);
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_size; ++i)
        {
            dispatch(d_temp_storage, temp_storage_size_bytes);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                dispatch(d_temp_storage, temp_storage_size_bytes);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * item_size);
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temp_storage));
        for(int i = 0; i < num_input_arrays; ++i)
        {
            HIP_CHECK(hipFree(d_key_inputs[i]));
        }
        HIP_CHECK(hipFree(d_value_input));
        HIP_CHECK(hipFree(d_unique_output));
        HIP_CHECK(hipFree(d_aggregates_output));
        HIP_CHECK(hipFree(d_unique_count_output));
    }

    static constexpr bool is_tuning = !std::is_same<Config, rocprim::default_config>::value;
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename KeyType, typename ValueType, int BlockSize, int TilesPerBlock>
struct device_reduce_by_key_benchmark_generator
{
    template<int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            using config
                = rocprim::reduce_by_key_config<BlockSize,
                                                ItemsPerThread,
                                                rocprim::block_load_method::block_load_transpose,
                                                rocprim::block_load_method::block_load_transpose,
                                                rocprim::block_scan_algorithm::using_warp_scan,
                                                TilesPerBlock>;
            // max segment length argument is irrelevant, tuning overrides segment length
            storage.emplace_back(
                std::make_unique<
                    device_reduce_by_key_benchmark<KeyType, ValueType, 0, false, config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static_for_each<make_index_range<int, 4, 15>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_REDUCE_BY_KEY_PARALLEL_HPP_
