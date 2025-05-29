// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/device/device_merge_sort.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

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
struct device_merge_sort_block_merge_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:merge_sort_block_merge,key_type:"
                                         + std::string(Traits<Key>::name())
                                         + ",value_type:" + std::string(Traits<Value>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    // Because merge_sort_block_merge expects partially sorted input:
    using block_sort_config = rocprim::default_config;

    // keys benchmark
    template<typename val = Value>
    auto do_run(benchmark_utils::state&& state) const ->
        typename std::enable_if<std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using key_type = Key;

        // Calculate the number of elements
        size_t size = bytes / sizeof(key_type);

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys(size);
        HIP_CHECK(hipDeviceSynchronize());

        ::rocprim::less<key_type> lesser_op;
        rocprim::empty_type*      values_ptr = nullptr;

        // Merge_sort_block_merge algorithm expects partially sorted input:
        unsigned int sorted_block_size;
        HIP_CHECK(rocprim::detail::merge_sort_block_sort<block_sort_config>(d_keys_input.get(),
                                                                            d_keys_input.get(),
                                                                            values_ptr,
                                                                            values_ptr,
                                                                            size,
                                                                            sorted_block_size,
                                                                            lesser_op,
                                                                            stream,
                                                                            false));

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::detail::merge_sort_block_merge<Config>(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys.get(),
                                                                  values_ptr,
                                                                  size,
                                                                  sorted_block_size,
                                                                  lesser_op,
                                                                  stream,
                                                                  false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        hipError_t err = rocprim::detail::merge_sort_block_merge<Config>(d_temporary_storage.get(),
                                                                         temporary_storage_bytes,
                                                                         d_keys.get(),
                                                                         values_ptr,
                                                                         size,
                                                                         sorted_block_size,
                                                                         lesser_op,
                                                                         stream,
                                                                         false);
        if(err == hipError_t::hipErrorAssert)
        {
            state.gbench_state.SkipWithError("SKIPPING: block_sort_items_per_block >= "
                                             "block_merge_items_per_block does not hold");
            return;
        }
        else if(err != hipSuccess)
        {
            std::cout << "HIP error: " << err << " line: " << __LINE__ << std::endl;
            exit(err);
        }
        HIP_CHECK(hipDeviceSynchronize());

        state.run_before_every_iteration(
            [&]
            {
                HIP_CHECK(hipMemcpyAsync(d_keys.get(),
                                         d_keys_input.get(),
                                         size * sizeof(key_type),
                                         hipMemcpyDeviceToDevice,
                                         stream));
            });

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::detail::merge_sort_block_merge<Config>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_keys.get(),
                                                                          values_ptr,
                                                                          size,
                                                                          sorted_block_size,
                                                                          lesser_op,
                                                                          stream,
                                                                          false));
            });

        state.set_throughput(size, sizeof(key_type));
    }

    // pairs benchmark
    template<typename val = Value>
    auto do_run(benchmark_utils::state&& state) const ->
        typename std::enable_if<!std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using key_type   = Key;
        using value_type = Value;

        // Calculate the number of elements
        size_t size = bytes / sizeof(key_type);

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys(size);

        common::device_ptr<value_type> d_values_input(values_input);
        common::device_ptr<value_type> d_values(size);

        HIP_CHECK(hipDeviceSynchronize());

        ::rocprim::less<key_type> lesser_op;

        // Merge_sort_block_merge algorithm expects partially sorted input:
        unsigned int sorted_block_size;
        HIP_CHECK(rocprim::detail::merge_sort_block_sort<block_sort_config>(d_keys_input.get(),
                                                                            d_keys_input.get(),
                                                                            d_values_input.get(),
                                                                            d_values_input.get(),
                                                                            size,
                                                                            sorted_block_size,
                                                                            lesser_op,
                                                                            stream,
                                                                            false));

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::detail::merge_sort_block_merge<Config>(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys.get(),
                                                                  d_values.get(),
                                                                  size,
                                                                  sorted_block_size,
                                                                  lesser_op,
                                                                  stream,
                                                                  false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        hipError_t err = rocprim::detail::merge_sort_block_merge<Config>(d_temporary_storage.get(),
                                                                         temporary_storage_bytes,
                                                                         d_keys.get(),
                                                                         d_values.get(),
                                                                         size,
                                                                         sorted_block_size,
                                                                         lesser_op,
                                                                         stream,
                                                                         false);
        if(err == hipError_t::hipErrorAssert)
        {
            state.gbench_state.SkipWithError("SKIPPING: block_sort_items_per_block >= "
                                             "block_merge_items_per_block does not hold");
            return;
        }
        else if(err != hipSuccess)
        {
            std::cout << "HIP error: " << err << " line: " << __LINE__ << std::endl;
            exit(err);
        }
        HIP_CHECK(hipDeviceSynchronize());

        state.run_before_every_iteration(
            [&]
            {
                HIP_CHECK(hipMemcpyAsync(d_keys.get(),
                                         d_keys_input.get(),
                                         size * sizeof(key_type),
                                         hipMemcpyDeviceToDevice,
                                         stream));
                HIP_CHECK(hipMemcpyAsync(d_values.get(),
                                         d_values_input.get(),
                                         size * sizeof(value_type),
                                         hipMemcpyDeviceToDevice,
                                         stream));
            });

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::detail::merge_sort_block_merge<Config>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_keys.get(),
                                                                          d_values.get(),
                                                                          size,
                                                                          sorted_block_size,
                                                                          lesser_op,
                                                                          stream,
                                                                          false));
            });

        state.set_throughput(size, sizeof(key_type));
    }

    void run(benchmark_utils::state&& state) override
    {
        do_run(std::forward<benchmark_utils::state>(state));
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

        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            storage.emplace_back(std::make_unique<benchmark_struct>());
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
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
