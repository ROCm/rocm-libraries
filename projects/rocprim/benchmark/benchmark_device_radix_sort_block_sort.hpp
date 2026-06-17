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
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
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
        return primbench::json{}.add("bs", config.block_size).add("ipt", config.items_per_thread);
    }
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_radix_sort_block_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_radix_sort_block_sort")
            .add("key_type", primbench::name<Key>())
            .add("value_type", primbench::name<Value>())
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        do_run(std::forward<primbench::state>(state));
    }

private:
    template<typename val = Value>
    auto do_run(primbench::state&& state) const
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type   = Key;
        using value_type = Value;

        size_t items = bytes / sizeof(key_type);

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(items,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        std::vector<value_type> values_input;

        if constexpr(!std::is_same<val, rocprim::empty_type>::value)
        {
            values_input = std::vector<value_type>(items);
            for(size_t i = 0; i < items; ++i)
            {
                values_input[i] = value_type(i);
            }
        }

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(items);

        using value_pointer_type
            = std::conditional_t<!std::is_same<val, rocprim::empty_type>::value,
                                 value_type*,
                                 rocprim::empty_type*>;

        value_pointer_type d_values_input  = nullptr;
        value_pointer_type d_values_output = nullptr;

        if constexpr(!std::is_same<val, rocprim::empty_type>::value)
        {
            common::device_ptr<value_type> d_values_input_tmp(values_input);
            common::device_ptr<value_type> d_values_output_tmp(items);
            d_values_input  = d_values_input_tmp.get();
            d_values_output = d_values_output_tmp.get();
        }

        unsigned int items_per_block;

        state.set_items(items);
        state.add_reads<key_type>(items);

        if constexpr(!std::is_same<val, rocprim::empty_type>::value)
        {
            state.add_reads<value_type>(items);
        }

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::detail::radix_sort_block_sort<Config, false>(
                    d_keys_input.get(),
                    d_keys_output.get(),
                    d_values_input,
                    d_values_output,
                    items,
                    items_per_block,
                    rocprim::identity_decomposer{},
                    0,
                    sizeof(key_type) * 8,
                    stream,
                    false)));
            });
    }
};

template<unsigned int BlockSize, typename Key, typename Value = rocprim::empty_type>
struct device_radix_sort_block_sort_benchmark_generator
{
    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        using generated_config = rocprim::kernel_config<BlockSize, ItemsPerThread>;

        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<
                    device_radix_sort_block_sort_benchmark<Key, Value, generated_config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
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
