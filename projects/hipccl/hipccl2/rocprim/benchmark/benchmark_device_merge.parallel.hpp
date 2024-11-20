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

#ifndef ROCPRIM_BENCHMARK_DEVICE_MERGE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_MERGE_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <cstdint>
#include <hip/hip_runtime.h>

// rocPRIM HIP API
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_merge.hpp>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>

namespace rp = rocprim;

template<typename Config>
std::string config_name()
{
    const rocprim::detail::merge_config_params params = Config();
    return "{bs:" + std::to_string(params.kernel_config.block_size)
           + ",ipt:" + std::to_string(params.kernel_config.items_per_thread) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename KeyType,
         typename ValueType = rocprim::empty_type,
         typename Config    = rocprim::default_config>
struct device_merge_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:merge,key_type:"
                                         + std::string(Traits<KeyType>::name())
                                         + ",value_type:" + std::string(Traits<ValueType>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    // keys benchmark
    template<typename val = ValueType>
    auto do_run(benchmark::State&   state,
                size_t              bytes,
                const managed_seed& seed,
                hipStream_t         stream) const ->
        typename std::enable_if<std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        using key_type = KeyType;
        using compare_op_type =
            typename std::conditional<std::is_same<key_type, rocprim::half>::value,
                                      half_less,
                                      rocprim::less<key_type>>::type;

        size_t size = bytes / sizeof(key_type);

        const size_t size1 = size / 2;
        const size_t size2 = size - size1;

        compare_op_type compare_op;

        // Generate data
        const auto random_range = limit_random_range<key_type>(0, size);

        std::vector<key_type> keys_input1 = get_random_data<key_type>(size1,
                                                                      random_range.first,
                                                                      random_range.second,
                                                                      seed.get_0());
        std::vector<key_type> keys_input2 = get_random_data<key_type>(size2,
                                                                      random_range.first,
                                                                      random_range.second,
                                                                      seed.get_1());
        std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_op);

        key_type* d_keys_input1;
        key_type* d_keys_input2;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input1), size1 * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input2), size2 * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input1,
                            keys_input1.data(),
                            size1 * sizeof(key_type),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_keys_input2,
                            keys_input2.data(),
                            size2 * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::merge<Config>(d_temporary_storage,
                                    temporary_storage_bytes,
                                    d_keys_input1,
                                    d_keys_input2,
                                    d_keys_output,
                                    size1,
                                    size2,
                                    compare_op,
                                    stream,
                                    false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::merge<Config>(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_input1,
                                        d_keys_input2,
                                        d_keys_output,
                                        size1,
                                        size2,
                                        compare_op,
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
                HIP_CHECK(rp::merge<Config>(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys_input1,
                                            d_keys_input2,
                                            d_keys_output,
                                            size1,
                                            size2,
                                            compare_op,
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
        HIP_CHECK(hipFree(d_keys_input1));
        HIP_CHECK(hipFree(d_keys_input2));
        HIP_CHECK(hipFree(d_keys_output));
    }

    // pairs benchmark
    template<typename val = ValueType>
    auto do_run(benchmark::State&   state,
                size_t              bytes,
                const managed_seed& seed,
                hipStream_t         stream) const ->
        typename std::enable_if<!std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        using key_type   = KeyType;
        using value_type = ValueType;
        using compare_op_type =
            typename std::conditional<std::is_same<key_type, rocprim::half>::value,
                                      half_less,
                                      rocprim::less<key_type>>::type;

        size_t size = bytes / sizeof(key_type);

        const size_t size1 = size / 2;
        const size_t size2 = size - size1;

        compare_op_type compare_op;

        // Generate data
        const auto            random_range = limit_random_range<key_type>(0, size);
        std::vector<key_type> keys_input1  = get_random_data<key_type>(size1,
                                                                      random_range.first,
                                                                      random_range.second,
                                                                      seed.get_0());
        std::vector<key_type> keys_input2  = get_random_data<key_type>(size2,
                                                                      random_range.first,
                                                                      random_range.second,
                                                                      seed.get_1());
        std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
        std::sort(keys_input2.begin(), keys_input2.end(), compare_op);
        std::vector<value_type> values_input1(size1);
        std::vector<value_type> values_input2(size2);
        std::iota(values_input1.begin(), values_input1.end(), 0);
        std::iota(values_input2.begin(), values_input2.end(), size1);

        key_type*   d_keys_input1;
        key_type*   d_keys_input2;
        key_type*   d_keys_output;
        value_type* d_values_input1;
        value_type* d_values_input2;
        value_type* d_values_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input1), size1 * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input2), size2 * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_values_input1), size1 * sizeof(value_type)));
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_values_input2), size2 * sizeof(value_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_output), size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_keys_input1,
                            keys_input1.data(),
                            size1 * sizeof(key_type),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_keys_input2,
                            keys_input2.data(),
                            size2 * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::merge<Config>(d_temporary_storage,
                                    temporary_storage_bytes,
                                    d_keys_input1,
                                    d_keys_input2,
                                    d_keys_output,
                                    d_values_input1,
                                    d_values_input2,
                                    d_values_output,
                                    size1,
                                    size2,
                                    compare_op,
                                    stream,
                                    false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::merge<Config>(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_input1,
                                        d_keys_input2,
                                        d_keys_output,
                                        d_values_input1,
                                        d_values_input2,
                                        d_values_output,
                                        size1,
                                        size2,
                                        compare_op,
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
                HIP_CHECK(rp::merge<Config>(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys_input1,
                                            d_keys_input2,
                                            d_keys_output,
                                            d_values_input1,
                                            d_values_input2,
                                            d_values_output,
                                            size1,
                                            size2,
                                            compare_op,
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

        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(key_type) + sizeof(value_type)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input1));
        HIP_CHECK(hipFree(d_keys_input2));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input1));
        HIP_CHECK(hipFree(d_values_input2));
        HIP_CHECK(hipFree(d_values_output));
    }

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        do_run(state, bytes, seed, stream);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING
template<typename KeyType, typename ValueType, int BlockSize>
struct device_merge_benchmark_generator
{

    template<unsigned int ItemsPerThreadExponent>
    struct create_ipt
    {
        static constexpr unsigned int items_per_thread = 1u << ItemsPerThreadExponent;
        using generated_config = rocprim::merge_config<BlockSize, items_per_thread>;
        using benchmark_struct = device_merge_benchmark<KeyType, ValueType, generated_config>;

        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            storage.emplace_back(std::make_unique<benchmark_struct>());
        }
    };

    struct create_default_config
    {
        using default_config =
            typename rocprim::detail::default_merge_config_base<KeyType, ValueType>::type;
        using benchmark_struct = device_merge_benchmark<KeyType, ValueType, default_config>;

        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            storage.emplace_back(std::make_unique<benchmark_struct>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr unsigned int min_items_per_thread_exponent = 0u;

        // Very large block sizes don't work with large items_per_thread since
        // shared memory is limited
        static constexpr unsigned int max_shared_memory    = TUNING_SHARED_MEMORY_MAX;
        static constexpr unsigned int max_size_per_element = sizeof(KeyType) + sizeof(ValueType);
        static constexpr unsigned int max_items_per_thread
            = max_shared_memory / (BlockSize * max_size_per_element);
        static constexpr unsigned int max_items_per_thread_exponent
            = rocprim::Log2<max_items_per_thread>::VALUE - 1;

        create_default_config()(storage);

        static_for_each<make_index_range<unsigned int,
                                         min_items_per_thread_exponent,
                                         max_items_per_thread_exponent>,
                        create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_MERGE_PARALLEL_HPP_