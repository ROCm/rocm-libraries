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

#ifndef ROCPRIM_BENCHMARK_DETAIL_BENCHMARK_DEVICE_RADIX_SORT_BLOCK_SORT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DETAIL_BENCHMARK_DEVICE_RADIX_SORT_BLOCK_SORT_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

template<typename Config>
std::string config_name()
{
    const rocprim::detail::kernel_config_params config = Config();
    return "{bs:" + std::to_string(config.block_size)
           + ",ipt:" + std::to_string(config.items_per_thread) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_radix_sort_block_sort_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:radix_sort_block_sort,key_type:"
                                         + std::string(Traits<Key>::name())
                                         + ",value_type:" + std::string(Traits<Value>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

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
        common::device_ptr<key_type> d_keys_output(size);

        rocprim::empty_type* values_ptr = nullptr;
        unsigned int         items_per_block;

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::detail::radix_sort_block_sort<Config, false>(
                    d_keys_input.get(),
                    d_keys_output.get(),
                    values_ptr,
                    values_ptr,
                    size,
                    items_per_block,
                    rocprim::identity_decomposer{},
                    0,
                    sizeof(key_type) * 8,
                    stream,
                    false)));
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
        for(size_t i = 0; i < size; ++i)
        {
            values_input[i] = value_type(i);
        }

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(size);

        common::device_ptr<value_type> d_values_input(values_input);
        common::device_ptr<value_type> d_values_output(size);

        unsigned int items_per_block;

        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::detail::radix_sort_block_sort<Config, false>(
                    d_keys_input.get(),
                    d_keys_output.get(),
                    d_values_input.get(),
                    d_values_output.get(),
                    size,
                    items_per_block,
                    rocprim::identity_decomposer{},
                    0,
                    sizeof(key_type) * 8,
                    stream,
                    false)));
            });

        state.set_throughput(size, sizeof(key_type) + sizeof(value_type));
    }

    void run(benchmark_utils::state&& state) override
    {
        do_run(std::forward<benchmark_utils::state>(state));
    }
};

template<unsigned int BlockSize, typename Key, typename Value = rocprim::empty_type>
struct device_radix_sort_block_sort_benchmark_generator
{
    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        using generated_config = rocprim::kernel_config<BlockSize, ItemsPerThread>;

        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<
                    device_radix_sort_block_sort_benchmark<Key, Value, generated_config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        // Sort_items_per_block must be equal or larger than radix_items_per_block, so make
        // the items_per_thread at least as large so the sort_items_per_block
        // would be atleast 1024.
        static constexpr unsigned int min_items_per_thread = 1024 / BlockSize;

        // Very large block sizes don't work with large items_per_blocks since
        // shared memory is limited
        static constexpr unsigned int max_shared_memory    = TUNING_SHARED_MEMORY_MAX - 2000;
        static constexpr unsigned int max_size_per_element = std::max(sizeof(Key), sizeof(Value));
        static constexpr unsigned int max_items_per_thread
            = std::min(32u, max_shared_memory / (BlockSize * max_size_per_element));

        static_for_each<make_index_range<unsigned int, min_items_per_thread, max_items_per_thread>,
                        create_ipt>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DETAIL_BENCHMARK_DEVICE_RADIX_SORT_BLOCK_SORT_PARALLEL_HPP_
