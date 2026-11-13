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

#ifndef ROCPRIM_BENCHMARK_DEVICE_TOPK_AIR_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_TOPK_AIR_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_topk_air.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template<typename Config>
auto config_name()
{
    if constexpr(std::is_same_v<Config, rocprim::default_config>)
    {
        return std::string("default");
    }
    else
    {
        constexpr auto config = Config{};
        return primbench::json{}
            .add("bs", config.kernel_config.block_size)
            .add("ipt", config.kernel_config.items_per_thread)
            .add("rb", config.radix_bits)
            .add("adapt_coeff", config.candidate_buffer_coefficient)
            .add("limit", config.thread_counter_limit);
    }
}

template<typename Key,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_topk_air_benchmark : public primbench::benchmark_interface
{
    bool small_k                  = false;
    bool adversarial_distribution = false;

    device_topk_air_benchmark(bool SmallK, bool AdversarialDistribution)
        : small_k(SmallK), adversarial_distribution(AdversarialDistribution)
    {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "topk")
            .add("subalgo", "air")
            .add("is_small", small_k)
            .add("is_adversarial", adversarial_distribution)
            .add("key_type", primbench::name<Key>())
            .add("value_type", primbench::name<Value>())
            .add("cfg", config_name<Config>());
    }

    template<class SizeT, class SeedT>
    inline auto generate_input(SizeT const& size, SeedT const& seed) const
    {
        using key_type = Key;
        using matched_int_t =
            typename rocprim::detail::device_topk_air_helper::matched_int<key_type>::type;
        static_assert(!std::is_same<matched_int_t, void>::value, "Input type not supported");
        static_assert(sizeof(key_type) == sizeof(matched_int_t),
                      "Size of mathed_int_t is not the same as input key_type");

        // TODO: get this value from Config
        constexpr unsigned int radix_bits = 8;
        matched_int_t          max_int    = key_type{0};
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < radix_bits; ++i)
        {
            max_int |= matched_int_t{1} << i;
        }
        key_type max_val;
        memcpy(&max_val, &max_int, sizeof(key_type));
        // Generate uniformly distributed data
        return adversarial_distribution
                   ? get_random_data<key_type>(size, key_type{0}, max_val, seed)
                   : get_random_data<key_type>(size,
                                               common::generate_limits<key_type>::min(),
                                               common::generate_limits<key_type>::max(),
                                               seed);
    }

private:
    // Keys benchmark
    template<typename val = Value>
    auto do_run(primbench::state& state) const
        -> std::enable_if_t<std::is_same<val, ::rocprim::empty_type>::value, void>
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type = Key;

        // Calculate the number of elements
        size_t items = bytes / sizeof(key_type);
        size_t k     = 10;

        if(!small_k)
        {
            k = items / 2;
        }

        // Generate uniformly distributed data
        const auto keys_input = generate_input(items, seed);

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(items);

        // Get size of d_temporary_storage
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_topk_air(nullptr,
                                  temporary_storage_bytes,
                                  d_keys_input.get(),
                                  d_keys_output.get(),
                                  static_cast<Value*>(nullptr),
                                  static_cast<Value*>(nullptr),
                                  items,
                                  k,
                                  stream,
                                  false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);

        // Run
        state.run(
            [&]
            {
                HIP_CHECK(invoke_topk_air(d_temporary_storage.get(),
                                          temporary_storage_bytes,
                                          d_keys_input.get(),
                                          d_keys_output.get(),
                                          static_cast<Value*>(nullptr),
                                          static_cast<Value*>(nullptr),
                                          items,
                                          k,
                                          stream,
                                          false));
            });
    }

    // Pairs benchmark
    template<typename val = Value>
    auto do_run(primbench::state& state) const
        -> std::enable_if_t<!std::is_same<val, ::rocprim::empty_type>::value, void>
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type   = Key;
        using value_type = Value;

        // Calculate the number of elements
        size_t items = bytes / (sizeof(key_type) + sizeof(value_type));
        size_t k     = 10;

        if(!small_k)
        {
            k = items / 2;
        }

        // Generate uniformly distributed data
        const auto keys_input = generate_input(items, seed);

        std::vector<value_type> values_input(items);
        for(size_t i = 0; i < items; ++i)
        {
            values_input[i] = value_type(i);
        }

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(items);

        common::device_ptr<value_type> d_values_input(values_input);
        common::device_ptr<value_type> d_values_output(items);

        // Get size of d_temporary_storage
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_topk_air(nullptr,
                                  temporary_storage_bytes,
                                  d_keys_input.get(),
                                  d_keys_output.get(),
                                  d_values_input.get(),
                                  d_values_output.get(),
                                  items,
                                  k,
                                  stream,
                                  false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);
        state.add_reads<value_type>(items);
        // Run
        state.run(
            [&]
            {
                HIP_CHECK(invoke_topk_air(d_temporary_storage.get(),
                                          temporary_storage_bytes,
                                          d_keys_input.get(),
                                          d_keys_output.get(),
                                          d_values_input.get(),
                                          d_values_output.get(),
                                          items,
                                          k,
                                          stream,
                                          false));
            });
    }

public:
    void run(primbench::state& state) override
    {
        do_run(state);
    }

private:
    static hipError_t invoke_topk_air(void*       d_temporary_storage,
                                      size_t&     temp_storage_bytes,
                                      Key*        keys_input,
                                      Key*        keys_output,
                                      Value*      values_input,
                                      Value*      values_output,
                                      size_t      size,
                                      size_t      k,
                                      hipStream_t stream,
                                      bool        debug_synchronous)
    {
        using decomposer = std::conditional_t<common::is_custom_type<Key>::value,
                                              custom_type_decomposer<Key>,
                                              ::rocprim::identity_decomposer>;
        if constexpr(std::is_same<Value, rocprim::empty_type>::value)
        {
            (void)values_input;
            (void)values_output;

            return rocprim::detail::device_topk_air<Config>(d_temporary_storage,
                                                            temp_storage_bytes,
                                                            keys_input,
                                                            keys_output,
                                                            nullptr,
                                                            nullptr,
                                                            size,
                                                            k,
                                                            decomposer{},
                                                            stream,
                                                            debug_synchronous);
        }
        else
        {
            return rocprim::detail::device_topk_air<Config>(d_temporary_storage,
                                                            temp_storage_bytes,
                                                            keys_input,
                                                            keys_output,
                                                            values_input,
                                                            values_output,
                                                            size,
                                                            k,
                                                            decomposer{},
                                                            stream,
                                                            debug_synchronous);
        }
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename Key,
         typename Value,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         unsigned int AdaptCoeff,
         unsigned int Limit>
struct device_topk_air_benchmark_generator
{
    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        using config
            = rocprim::topk_air_config<BlockSize, ItemsPerThread, RadixBits, AdaptCoeff, Limit>;
        storage.emplace_back(
            std::make_unique<device_topk_air_benchmark<Key, Value, config>>(true, true));
        storage.emplace_back(
            std::make_unique<device_topk_air_benchmark<Key, Value, config>>(true, false));
        storage.emplace_back(
            std::make_unique<device_topk_air_benchmark<Key, Value, config>>(false, true));
        storage.emplace_back(
            std::make_unique<device_topk_air_benchmark<Key, Value, config>>(false, false));
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_TOPK_AIR_PARALLEL_HPP_
