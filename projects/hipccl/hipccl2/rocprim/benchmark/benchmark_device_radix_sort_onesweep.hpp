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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/block/block_radix_rank.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/types.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/device/detail/device_radix_sort.hpp>
#endif

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <memory>
#else
    #include <stdint.h>
#endif

constexpr const char* radix_rank_algorithm_name(rocprim::block_radix_rank_algorithm algorithm)
{
    switch(algorithm)
    {
        case rocprim::block_radix_rank_algorithm::basic: return "block_radix_rank_algorithm::basic";
        case rocprim::block_radix_rank_algorithm::basic_memoize:
            return "block_radix_rank_algorithm::basic_memoize";
        case rocprim::block_radix_rank_algorithm::match: return "block_radix_rank_algorithm::match";
    }

    return ""; // unknown algorithm
}

template<typename Config>
constexpr auto config_name()
{
    if constexpr(std::is_same_v<Config, rocprim::default_config>)
    {
        return std::string("default");
    }
    else
    {
        constexpr rocprim::detail::radix_sort_onesweep_config_params config = Config();

        return primbench::json{}
            .add("histogram",
                 primbench::json{}
                     .add("bs", config.histogram.block_size)
                     .add("ipt", config.histogram.items_per_thread))
            .add("sort",
                 primbench::json{}
                     .add("bs", config.sort.block_size)
                     .add("ipt", config.sort.items_per_thread))
            .add("bits_per_place", config.radix_bits_per_place)
            .add("algorithm", radix_rank_algorithm_name(config.radix_rank_algorithm));
    }
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_radix_sort_onesweep_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_radix_sort_onesweep")
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

        std::vector<key_type> keys_input
            = get_random_data<key_type>(items,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        std::vector<value_type> values_input(items);

        using value_pointer_type
            = std::conditional_t<!std::is_same<val, rocprim::empty_type>::value,
                                 value_type*,
                                 rocprim::empty_type*>;

        value_pointer_type d_values_input_ptr  = nullptr;
        value_pointer_type d_values_output_ptr = nullptr;

        if constexpr(!std::is_same<value_type, rocprim::empty_type>::value)
        {
            for(size_t i = 0; i < items; ++i)
            {
                values_input[i] = value_type(i);
            }

            common::device_ptr<value_type> d_values_input_tmp(values_input);
            common::device_ptr<value_type> d_values_output_tmp(items);
            d_values_input_ptr  = d_values_input_tmp.get();
            d_values_output_ptr = d_values_output_tmp.get();
        }

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(items);

        common::device_ptr<void> d_temporary_storage;
        size_t                   temporary_storage_bytes = 0;

        bool is_result_in_output = true;
        HIP_CHECK((
            rocprim::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     nullptr,
                                                                     d_keys_output.get(),
                                                                     d_values_input_ptr,
                                                                     nullptr,
                                                                     d_values_output_ptr,
                                                                     items,
                                                                     is_result_in_output,
                                                                     rocprim::identity_decomposer{},
                                                                     0,
                                                                     sizeof(key_type) * 8,
                                                                     stream,
                                                                     false,
                                                                     false)));

        d_temporary_storage.resize(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);
        if constexpr(!std::is_same<value_type, rocprim::empty_type>::value)
        {
            state.add_reads<value_type>(items);
        }

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::detail::radix_sort_onesweep_impl<Config, false>(
                    d_temporary_storage.get(),
                    temporary_storage_bytes,
                    d_keys_input.get(),
                    nullptr,
                    d_keys_output.get(),
                    d_values_input_ptr,
                    nullptr,
                    d_values_output_ptr,
                    items,
                    is_result_in_output,
                    rocprim::identity_decomposer{},
                    0,
                    sizeof(key_type) * 8,
                    stream,
                    false,
                    false)));
            });
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<unsigned int BlockSize,
         unsigned int RadixBits,
         typename Key,
         typename Value = rocprim::empty_type>
struct device_radix_sort_onesweep_benchmark_generator
{
    template<unsigned int ItemsPerThread, rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    static constexpr bool is_buildable()
    {
        // Calculation uses `rocprim::arch::wavefront::min_size()`, which is 32 on host side unless overridden.
        //   However, this does not affect the total size of shared memory for the current configuration space.
        //   Were the implementation to change, causing retuning, this needs to be re-evaluated and possibly taken into account.
        using sharedmem_storage = typename rocprim::detail::onesweep_iteration_helper<
            Key,
            Value,
            size_t,
            BlockSize,
            ItemsPerThread,
            RadixBits,
            false,
            RadixRankAlgorithm,
            rocprim::identity_decomposer,
            rocprim::detail::block_id_wrapper<>>::storage_type;
        return sizeof(sharedmem_storage) < TUNING_SHARED_MEMORY_MAX;
    }

    template<unsigned int                        ItemsPerThread,
             rocprim::block_radix_rank_algorithm RadixRankAlgorithm,
             typename Enable = void>
    struct create_ipt;

    template<unsigned int ItemsPerThread, rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    struct create_ipt<ItemsPerThread,
                      RadixRankAlgorithm,
                      std::enable_if_t<(is_buildable<ItemsPerThread, RadixRankAlgorithm>())>>
    {
        using generated_config
            = rocprim::radix_sort_onesweep_config<rocprim::kernel_config<BlockSize, ItemsPerThread>,
                                                  rocprim::kernel_config<BlockSize, ItemsPerThread>,
                                                  RadixBits,
                                                  RadixRankAlgorithm>;
        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<
                    device_radix_sort_onesweep_benchmark<Key, Value, generated_config>>());
        }
    };

    template<unsigned int ItemsPerThread, rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    struct create_ipt<ItemsPerThread,
                      RadixRankAlgorithm,
                      std::enable_if_t<(!is_buildable<ItemsPerThread, RadixRankAlgorithm>())>>
    {
        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>&) const {}
    };

    template<rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    static void create_algo(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        create_ipt<1u, RadixRankAlgorithm>()(storage);
        create_ipt<4u, RadixRankAlgorithm>()(storage);
        create_ipt<6u, RadixRankAlgorithm>()(storage);
        create_ipt<8u, RadixRankAlgorithm>()(storage);
        create_ipt<12u, RadixRankAlgorithm>()(storage);
        create_ipt<16u, RadixRankAlgorithm>()(storage);
        create_ipt<18u, RadixRankAlgorithm>()(storage);
        create_ipt<22u, RadixRankAlgorithm>()(storage);
    }

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        create_algo<rocprim::block_radix_rank_algorithm::basic>(storage);
        create_algo<rocprim::block_radix_rank_algorithm::match>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING
