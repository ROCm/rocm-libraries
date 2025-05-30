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

#ifndef ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_ONESWEEP_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_ONESWEEP_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
std::string config_name()
{
    constexpr rocprim::detail::radix_sort_onesweep_config_params params = Config();
    return "{histogram:{bs:" + std::to_string(params.histogram.block_size)
           + ",ipt:" + std::to_string(params.histogram.items_per_thread) + "},sort:{"
           + "bs:" + std::to_string(params.sort.block_size)
           + ",ipt:" + std::to_string(params.sort.items_per_thread)
           + "},bits_per_place:" + std::to_string(params.radix_bits_per_place)
           + ",algorithm:" + radix_rank_algorithm_name(params.radix_rank_algorithm) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_radix_sort_onesweep_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:radix_sort_onesweep,key_type:"
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

        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(size);

        common::device_ptr<void> d_temporary_storage;
        size_t temporary_storage_bytes = 0;

        bool                 is_result_in_output = true;
        rocprim::empty_type* d_values_ptr        = nullptr;
        HIP_CHECK((
            rocprim::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     nullptr,
                                                                     d_keys_output.get(),
                                                                     d_values_ptr,
                                                                     nullptr,
                                                                     d_values_ptr,
                                                                     size,
                                                                     is_result_in_output,
                                                                     rocprim::identity_decomposer{},
                                                                     0,
                                                                     sizeof(key_type) * 8,
                                                                     stream,
                                                                     false,
                                                                     false)));

        d_temporary_storage.resize(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::detail::radix_sort_onesweep_impl<Config, false>(
                    d_temporary_storage.get(),
                    temporary_storage_bytes,
                    d_keys_input.get(),
                    nullptr,
                    d_keys_output.get(),
                    d_values_ptr,
                    nullptr,
                    d_values_ptr,
                    size,
                    is_result_in_output,
                    rocprim::identity_decomposer{},
                    0,
                    sizeof(key_type) * 8,
                    stream,
                    false,
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

        common::device_ptr<void> d_temporary_storage;
        size_t temporary_storage_bytes = 0;

        bool is_result_in_output = true;
        HIP_CHECK((
            rocprim::detail::radix_sort_onesweep_impl<Config, false>(d_temporary_storage.get(),
                                                                     temporary_storage_bytes,
                                                                     d_keys_input.get(),
                                                                     nullptr,
                                                                     d_keys_output.get(),
                                                                     d_values_input.get(),
                                                                     nullptr,
                                                                     d_values_output.get(),
                                                                     size,
                                                                     is_result_in_output,
                                                                     rocprim::identity_decomposer{},
                                                                     0,
                                                                     sizeof(key_type) * 8,
                                                                     stream,
                                                                     false,
                                                                     false)));

        d_temporary_storage.resize(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK((rocprim::detail::radix_sort_onesweep_impl<Config, false>(
                    d_temporary_storage.get(),
                    temporary_storage_bytes,
                    d_keys_input.get(),
                    nullptr,
                    d_keys_output.get(),
                    d_values_input.get(),
                    nullptr,
                    d_values_output.get(),
                    size,
                    is_result_in_output,
                    rocprim::identity_decomposer{},
                    0,
                    sizeof(key_type) * 8,
                    stream,
                    false,
                    false)));
            });

        state.set_throughput(size, sizeof(key_type) + sizeof(value_type));
    }

    void run(benchmark_utils::state&& state) override
    {
        do_run(std::forward<benchmark_utils::state>(state));
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
            rocprim::identity_decomposer>::storage_type;
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
        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
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
        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>&) const {}
    };

    template<rocprim::block_radix_rank_algorithm RadixRankAlgorithm>
    static void
        create_algo(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
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

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        create_algo<rocprim::block_radix_rank_algorithm::basic>(storage);
        create_algo<rocprim::block_radix_rank_algorithm::match>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_ONESWEEP_PARALLEL_HPP_
