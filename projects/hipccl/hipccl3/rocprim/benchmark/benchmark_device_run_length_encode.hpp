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

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_run_length_encode.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/block/block_load.hpp>
    #include <rocprim/block/block_scan.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <memory>
#endif

template<typename Config>
constexpr auto config_name()
{
    if constexpr(std::is_same_v<Config, rocprim::default_config>)
    {
        return std::string("default");
    }
    else
    {
        using ReduceByKeyConfig =
            typename rocprim::detail::select_reduce_by_key_config<Config>::type;
        auto config = ReduceByKeyConfig();
        return primbench::json{}
            .add("bs", config.kernel_config.block_size)
            .add("ipt", config.kernel_config.items_per_thread);
    }
}

template<typename T, size_t MaxLength, typename Config = rocprim::default_config>
struct device_run_length_encode_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_run_length_encode")
            .add("key_type", primbench::name<T>())
            .add("keys_max_length", MaxLength)
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type   = T;
        using count_type = unsigned int;

        const size_t items = bytes / sizeof(T);

        // Generate data
        std::vector<key_type> input(items);

        unsigned int        runs_count   = 0;
        const auto          random_range = limit_random_range<size_t>(1, MaxLength);
        std::vector<size_t> key_counts
            = get_random_data<size_t>(100000, random_range.first, random_range.second, seed);
        size_t offset = 0;
        while(offset < items)
        {
            const size_t key_count = key_counts[runs_count % key_counts.size()];
            const size_t end       = std::min(items, offset + key_count);
            for(size_t i = offset; i < end; ++i)
            {
                input[i] = runs_count;
            }

            ++runs_count;
            offset += key_count;
        }

        common::device_ptr<key_type> d_input(input);

        common::device_ptr<key_type>   d_unique_output(runs_count);
        common::device_ptr<count_type> d_counts_output(runs_count);
        common::device_ptr<count_type> d_runs_count_output(1);

        size_t temporary_storage_bytes = 0;

        HIP_CHECK(rocprim::run_length_encode<Config>(nullptr,
                                                     temporary_storage_bytes,
                                                     d_input.get(),
                                                     items,
                                                     d_unique_output.get(),
                                                     d_counts_output.get(),
                                                     d_runs_count_output.get(),
                                                     stream,
                                                     false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::run_length_encode<Config>(d_temporary_storage.get(),
                                                             temporary_storage_bytes,
                                                             d_input.get(),
                                                             items,
                                                             d_unique_output.get(),
                                                             d_counts_output.get(),
                                                             d_runs_count_output.get(),
                                                             stream,
                                                             false));
            });
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename T, unsigned int BlockSize>
struct device_run_length_encode_benchmark_generator
{
    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            using config = rocprim::run_length_encode_config<
                rocprim::reduce_by_key_config<BlockSize,
                                              ItemsPerThread,
                                              rocprim::block_load_method::block_load_transpose,
                                              rocprim::block_load_method::block_load_transpose,
                                              rocprim::block_scan_algorithm::using_warp_scan>,
                rocprim::default_config>;

            storage.emplace_back(
                std::make_unique<device_run_length_encode_benchmark<T, 10, config>>());
            storage.emplace_back(
                std::make_unique<device_run_length_encode_benchmark<T, 1000, config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        static constexpr unsigned int max_items_per_thread
            = std::min(TUNING_SHARED_MEMORY_MAX / sizeof(T) / BlockSize - 1, size_t{15});
        static_for_each<make_index_range<unsigned int, 4u, max_items_per_thread>, create_ipt>(
            storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING
