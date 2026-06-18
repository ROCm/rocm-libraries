// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
constexpr auto config_name()
{
    if constexpr(std::is_same_v<Config, rocprim::default_config>)
    {
        return std::string("default");
    }
    else
    {
        auto config = Config();
        return primbench::json{}
            .add("oddeven_bs", config.merge_oddeven_config.block_size)
            .add("oddeven_ipt", config.merge_oddeven_config.items_per_thread)
            .add("oddeven_size_limit", config.merge_oddeven_config.size_limit)
            .add("mergepath_partition_bs", config.merge_mergepath_partition_config.block_size)
            .add("mergepath_bs", config.merge_mergepath_config.block_size)
            .add("mergepath_ipt", config.merge_mergepath_config.items_per_thread);
    }
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_merge_sort_block_merge_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_merge_sort_block_merge")
            .add("key_type", primbench::name<Key>())
            .add("value_type", primbench::name<Value>())
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        do_run(std::forward<primbench::state>(state));
    }

private:
    // Because merge_sort_block_merge expects partially sorted input:
    using block_sort_config = rocprim::default_config;

    // keys benchmark
    template<typename val = Value>
    auto do_run(primbench::state&& state) const
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type = Key;

        size_t items = bytes / sizeof(key_type);

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(items,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys(items);

        using value_ptr_type = std::
            conditional_t<std::is_same_v<val, rocprim::empty_type>, rocprim::empty_type*, Value*>;

        common::device_ptr<Value> d_values_input;

        value_ptr_type values_ptr;
        if constexpr(!std::is_same_v<val, rocprim::empty_type>)
        {
            using value_type = Value;

            std::vector<value_type> values_input(items);
            std::iota(values_input.begin(), values_input.end(), 0);

            d_values_input = common::device_ptr<Value>(values_input);
            common::device_ptr<value_type> d_values(items);
            values_ptr = d_values.get();
        }
        else
        {
            values_ptr = nullptr;
        }

        ::rocprim::less<key_type> lesser_op;

        // Merge_sort_block_merge algorithm expects partially sorted input:
        unsigned int sorted_block_size;
        if constexpr(std::is_same_v<val, rocprim::empty_type>)
        {
            HIP_CHECK(rocprim::detail::merge_sort_block_sort<block_sort_config>(d_keys_input.get(),
                                                                                d_keys_input.get(),
                                                                                values_ptr,
                                                                                values_ptr,
                                                                                items,
                                                                                sorted_block_size,
                                                                                lesser_op,
                                                                                stream,
                                                                                false));
        }
        {
            HIP_CHECK(
                rocprim::detail::merge_sort_block_sort<block_sort_config>(d_keys_input.get(),
                                                                          d_keys_input.get(),
                                                                          d_values_input.get(),
                                                                          d_values_input.get(),
                                                                          items,
                                                                          sorted_block_size,
                                                                          lesser_op,
                                                                          stream,
                                                                          false));
        }

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::detail::merge_sort_block_merge<Config>(nullptr,
                                                                  temporary_storage_bytes,
                                                                  d_keys.get(),
                                                                  values_ptr,
                                                                  items,
                                                                  sorted_block_size,
                                                                  lesser_op,
                                                                  stream,
                                                                  false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        hipError_t err = rocprim::detail::merge_sort_block_merge<Config>(d_temporary_storage.get(),
                                                                         temporary_storage_bytes,
                                                                         d_keys.get(),
                                                                         values_ptr,
                                                                         items,
                                                                         sorted_block_size,
                                                                         lesser_op,
                                                                         stream,
                                                                         false);
        if(err == hipError_t::hipErrorAssert)
        {
            // state.gbench_state.SkipWithError("SKIPPING: block_sort_items_per_block >= "
            //                                  "block_merge_items_per_block does not hold");
            return;
        }
        else if(err != hipSuccess)
        {
            std::cout << "HIP error: " << err << " line: " << __LINE__ << std::endl;
            exit(err);
        }

        state.run_before_every_iteration(
            [&]
            {
                HIP_CHECK(hipMemcpyAsync(d_keys.get(),
                                         d_keys_input.get(),
                                         items * sizeof(key_type),
                                         hipMemcpyDeviceToDevice,
                                         stream));
                if constexpr(!std::is_same_v<val, rocprim::empty_type>)
                {
                    HIP_CHECK(hipMemcpyAsync(values_ptr,
                                             d_values_input.get(),
                                             items * sizeof(Value),
                                             hipMemcpyDeviceToDevice,
                                             stream));
                }
            });

        state.set_items(items);
        state.add_reads<key_type>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::detail::merge_sort_block_merge<Config>(d_temporary_storage.get(),
                                                                          temporary_storage_bytes,
                                                                          d_keys.get(),
                                                                          values_ptr,
                                                                          items,
                                                                          sorted_block_size,
                                                                          lesser_op,
                                                                          stream,
                                                                          false));
            });
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

        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            storage.emplace_back(std::make_unique<benchmark_struct>());
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
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
