// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/device/device_merge.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <type_traits>
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
        auto config = Config();
        return primbench::json{}
            .add("bs", config.kernel_config.block_size)
            .add("ipt", config.kernel_config.items_per_thread);
    }
}

template<typename KeyType,
         typename ValueType = rocprim::empty_type,
         typename Config    = rocprim::default_config>
struct device_merge_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_merge")
            .add("key_type", primbench::name<KeyType>())
            .add("value_type", primbench::name<ValueType>())
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        do_run(std::forward<primbench::state>(state));
    }

private:
    template<typename val = ValueType>
    auto do_run(primbench::state&& state) const
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        auto seeds = primbench::seeds<2>(seed);

        using key_type   = KeyType;
        using value_type = ValueType;
        using compare_op_type =
            typename std::conditional<std::is_same<key_type, rocprim::half>::value,
                                      half_less,
                                      rocprim::less<key_type>>::type;

        size_t items = bytes / sizeof(key_type);

        const size_t size1 = items / 2;
        const size_t size2 = items - size1;

        compare_op_type compare_op;

        // Generate data
        primbench::log("Generating random_range");
        const auto random_range = limit_random_range<key_type>(0, items);

        primbench::log("Generating keys_input1");
        std::vector<key_type> keys_input1
            = get_random_data<key_type>(size1, random_range.first, random_range.second, seeds[0]);
        primbench::log("Generating keys_input2");
        std::vector<key_type> keys_input2
            = get_random_data<key_type>(size2, random_range.first, random_range.second, seeds[1]);

        primbench::log("Sorting keys_input1");
        std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
        primbench::log("Sorting keys_input2");
        std::sort(keys_input2.begin(), keys_input2.end(), compare_op);


        std::vector<value_type> values_input1;
        std::vector<value_type> values_input2;
        if constexpr(!std::is_same<val, ::rocprim::empty_type>::value)
        {
            primbench::log("Allocating values_input1");
            values_input1 = std::vector<value_type>(size1);
            primbench::log("Allocating values_input2");
            values_input2 = std::vector<value_type>(size2);

            primbench::log("Filling values_input1");
            std::iota(values_input1.begin(), values_input1.end(), 0);
            primbench::log("Filling values_input2");
            std::iota(values_input2.begin(), values_input2.end(), size1);
        }

        primbench::log("Creating device pointers");
        common::device_ptr<key_type>   d_keys_input1(keys_input1);
        common::device_ptr<key_type>   d_keys_input2(keys_input2);
        common::device_ptr<key_type>   d_keys_output(items);
        common::device_ptr<value_type> d_values_output(items);
        common::device_ptr<value_type> d_values_input1;
        common::device_ptr<value_type> d_values_input2;

        if constexpr(!std::is_same<val, ::rocprim::empty_type>::value)
        {
            d_values_input1 = common::device_ptr<value_type>{values_input1};
            d_values_input2 = common::device_ptr<value_type>{values_input2};
        }

        primbench::log("Calculating d_temporary_storage size");
        common::device_ptr<void> d_temporary_storage;
        size_t                   temporary_storage_bytes = 0;

        if constexpr(!std::is_same<val, ::rocprim::empty_type>::value)
        {
            HIP_CHECK(rocprim::merge<Config>(d_temporary_storage.get(),
                                             temporary_storage_bytes,
                                             d_keys_input1.get(),
                                             d_keys_input2.get(),
                                             d_keys_output.get(),
                                             d_values_input1.get(),
                                             d_values_input2.get(),
                                             d_values_output.get(),
                                             size1,
                                             size2,
                                             compare_op,
                                             stream,
                                             false));
        }
        else
        {
            HIP_CHECK(rocprim::merge<Config>(d_temporary_storage.get(),
                                             temporary_storage_bytes,
                                             d_keys_input1.get(),
                                             d_keys_input2.get(),
                                             d_keys_output.get(),
                                             size1,
                                             size2,
                                             compare_op,
                                             stream,
                                             false));
        }

        primbench::log("Resizing d_temporary_storage");
        d_temporary_storage.resize(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);

        if constexpr(!std::is_same<val, ::rocprim::empty_type>::value)
        {
            state.add_reads<value_type>(items);

            state.run(
                [&]
                {
                    HIP_CHECK(rocprim::merge<Config>(d_temporary_storage.get(),
                                                     temporary_storage_bytes,
                                                     d_keys_input1.get(),
                                                     d_keys_input2.get(),
                                                     d_keys_output.get(),
                                                     d_values_input1.get(),
                                                     d_values_input2.get(),
                                                     d_values_output.get(),
                                                     size1,
                                                     size2,
                                                     compare_op,
                                                     stream,
                                                     false));
                });
        }
        else
        {
            state.run(
                [&]
                {
                    HIP_CHECK(rocprim::merge<Config>(d_temporary_storage.get(),
                                                     temporary_storage_bytes,
                                                     d_keys_input1.get(),
                                                     d_keys_input2.get(),
                                                     d_keys_output.get(),
                                                     size1,
                                                     size2,
                                                     compare_op,
                                                     stream,
                                                     false));
                });
        }
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

        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            storage.emplace_back(std::make_unique<benchmark_struct>());
        }
    };

    struct create_default_config
    {
        using benchmark_struct = device_merge_benchmark<KeyType, ValueType>;

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
