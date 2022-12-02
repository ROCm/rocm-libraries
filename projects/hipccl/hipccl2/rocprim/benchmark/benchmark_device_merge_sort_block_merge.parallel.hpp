// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DETAIL_BENCHMARK_DEVICE_MERGE_SORT_BLOCK_MERGE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DETAIL_BENCHMARK_DEVICE_MERGE_SORT_BLOCK_MERGE_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

namespace rp = rocprim;

template<typename Config>
std::string config_name()
{
    const rocprim::detail::merge_sort_block_merge_config_params config = Config();
    return "{oddeven_bs:" + std::to_string(config.merge_oddeven_config.block_size) + ",oddeven_ipt:"
           + std::to_string(config.merge_oddeven_config.items_per_thread) + ",oddeven_size_limit:"
           + std::to_string(config.merge_oddeven_config.size_limit) + ",mergepath_partition_bs:"
           + std::to_string(config.merge_mergepath_partition_config.block_size) + ",mergepath_bs:"
           + std::to_string(config.merge_mergepath_config.block_size) + ",mergepath_ipt:"
           + std::to_string(config.merge_mergepath_config.items_per_thread) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_merge_sort_block_merge_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:merge_sort_block_merge,key_type:"
                                         + std::string(Traits<Key>::name())
                                         + ",value_type:" + std::string(Traits<Value>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;
    // Because merge_sort_block_merge expects partially sorted input:
    using block_sort_config = rocprim::default_config;

    // keys benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        using key_type = Key;

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = get_random_data<key_type>(size,
                                                   static_cast<key_type>(-1000),
                                                   static_cast<key_type>(1000));
        }
        else
        {
            keys_input = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max());
        }

        key_type* d_keys_input;
        key_type* d_keys;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));
        hipDeviceSynchronize();

        ::rocprim::less<key_type> lesser_op;
        rocprim::empty_type*      values_ptr = nullptr;

        // Merge_sort_block_merge algorithm expects partially sorted input:
        unsigned int sorted_block_size;
        HIP_CHECK(rp::detail::merge_sort_block_sort<block_sort_config>(d_keys_input,
                                                                       d_keys_input,
                                                                       values_ptr,
                                                                       values_ptr,
                                                                       size,
                                                                       sorted_block_size,
                                                                       lesser_op,
                                                                       stream,
                                                                       false));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::detail::merge_sort_block_merge<Config>(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_keys,
                                                             values_ptr,
                                                             size,
                                                             sorted_block_size,
                                                             lesser_op,
                                                             stream,
                                                             false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        hipError_t err;
        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            err = rp::detail::merge_sort_block_merge<Config>(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_keys,
                                                             values_ptr,
                                                             size,
                                                             sorted_block_size,
                                                             lesser_op,
                                                             stream,
                                                             false);
        }
        if(err == hipError_t::hipErrorAssert)
        {
            state.SkipWithError("SKIPPING: block_sort_items_per_block <= "
                                "block_merge_items_per_block does not hold");
            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys));
            return;
        }
        else if(err != hipSuccess)
        {
            std::cout << "HIP error: " << err << " line: " << __LINE__ << std::endl;
            exit(err);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            hipMemcpyAsync(d_keys,
                           d_keys_input,
                           size * sizeof(key_type),
                           hipMemcpyDeviceToDevice,
                           stream);
            HIP_CHECK(hipEventRecord(start, stream));
            HIP_CHECK(rp::detail::merge_sort_block_merge<Config>(d_temporary_storage,
                                                                 temporary_storage_bytes,
                                                                 d_keys,
                                                                 values_ptr,
                                                                 size,
                                                                 sorted_block_size,
                                                                 lesser_op,
                                                                 stream,
                                                                 false));

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
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys));
    }

    // pairs benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<!std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        using key_type   = Key;
        using value_type = Value;

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = get_random_data<key_type>(size,
                                                   static_cast<key_type>(-1000),
                                                   static_cast<key_type>(1000));
        }
        else
        {
            keys_input = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max());
        }
        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        key_type* d_keys_input;
        key_type* d_keys;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        value_type* d_values;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_input), size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values), size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        hipDeviceSynchronize();

        ::rocprim::less<key_type> lesser_op;

        // Merge_sort_block_merge algorithm expects partially sorted input:
        unsigned int sorted_block_size;
        HIP_CHECK(rp::detail::merge_sort_block_sort<block_sort_config>(d_keys_input,
                                                                       d_keys_input,
                                                                       d_values_input,
                                                                       d_values_input,
                                                                       size,
                                                                       sorted_block_size,
                                                                       lesser_op,
                                                                       stream,
                                                                       false));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::detail::merge_sort_block_merge<Config>(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_keys,
                                                             d_values,
                                                             size,
                                                             sorted_block_size,
                                                             lesser_op,
                                                             stream,
                                                             false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        hipError_t err;
        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            err = rp::detail::merge_sort_block_merge<Config>(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_keys,
                                                             d_values,
                                                             size,
                                                             sorted_block_size,
                                                             lesser_op,
                                                             stream,
                                                             false);
        }
        if(err == hipError_t::hipErrorAssert)
        {
            state.SkipWithError("SKIPPING: block_sort_items_per_block <= "
                                "block_merge_items_per_block does not hold");
            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values));
            return;
        }
        else if(err != hipSuccess)
        {
            std::cout << "HIP error: " << err << " line: " << __LINE__ << std::endl;
            exit(err);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            hipMemcpyAsync(d_keys,
                           d_keys_input,
                           size * sizeof(key_type),
                           hipMemcpyDeviceToDevice,
                           stream);
            hipMemcpyAsync(d_values,
                           d_values_input,
                           size * sizeof(key_type),
                           hipMemcpyDeviceToDevice,
                           stream);
            HIP_CHECK(hipEventRecord(start, stream));
            HIP_CHECK(rp::detail::merge_sort_block_merge<Config>(d_temporary_storage,
                                                                 temporary_storage_bytes,
                                                                 d_keys,
                                                                 d_values,
                                                                 size,
                                                                 sorted_block_size,
                                                                 lesser_op,
                                                                 stream,
                                                                 false));

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
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values));
    }

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        do_run(state, size, stream);
    }
};

template<unsigned int BlockSize,
         bool         use_mergepath,
         typename Key,
         typename Value = rocprim::empty_type>
struct device_merge_sort_block_merge_benchmark_generator
{
    static constexpr unsigned int get_limit()
    {
        return use_mergepath ? 0 : UINT32_MAX;
    }

    template<unsigned int ItemsPerThreadExponent>
    struct create_ipt
    {
        static constexpr unsigned int items_per_thread = 1u << ItemsPerThreadExponent;
        using generated_config = rocprim::detail::merge_sort_block_merge_config<BlockSize,
                                                                                items_per_thread,
                                                                                get_limit(),
                                                                                128,
                                                                                BlockSize,
                                                                                items_per_thread>;
        using benchmark_struct
            = device_merge_sort_block_merge_benchmark<Key, Value, generated_config>;

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
        static constexpr unsigned int max_size_per_element = sizeof(Key) + sizeof(Value);
        static constexpr unsigned int max_items_per_thread
            = max_shared_memory / (BlockSize * max_size_per_element);
        static constexpr unsigned int max_items_per_thread_exponent
            = rocprim::Log2<max_items_per_thread>::VALUE - 1;

        static_for_each<make_index_range<unsigned int,
                                         min_items_per_thread_exponent,
                                         max_items_per_thread_exponent>,
                        create_ipt>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DETAIL_BENCHMARK_DEVICE_MERGE_SORT_BLOCK_MERGE_PARALLEL_HPP_
