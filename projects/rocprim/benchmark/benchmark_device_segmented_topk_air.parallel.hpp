// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_TOPK_AIR_TOPK_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_TOPK_AIR_TOPK_PARALLEL_HPP_

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_segmented_topk_air.hpp>
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
         typename Value                = rocprim::empty_type,
         typename Config               = rocprim::default_config,
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000>
struct device_segmented_topk_air_benchmark : public primbench::benchmark_interface
{
    using size_out_type = size_t;
    using size_in_type  = size_t;
    using offsets_type  = size_in_type;
    using invoker       = rocprim::detail::device_segmented_topk_air_impl_invoker<
              Config,
              true,
              false, // TODO: need to change this to true, after adaptive is available
              Key*,
              Key*,
              Value*,
              Value*,
              size_in_type,
              size_out_type,
              offsets_type*,
              rocprim::identity_decomposer>;

    bool small_k                  = false;
    bool adversarial_distribution = false;

    device_segmented_topk_air_benchmark(bool SmallK, bool AdversarialDistribution)
        : small_k(SmallK), adversarial_distribution(AdversarialDistribution)
    {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "segmented_topk")
            .add("subalgo", "air")
            .add("is_small", small_k)
            .add("is_adversarial", adversarial_distribution)
            .add("key_type", primbench::name<Key>())
            .add("value_type", primbench::name<Value>())
            .add("cfg", config_name<Config>());
    }

    template<class SeedsT>
    inline auto
        generate_input(size_in_type const& size, SeedsT const& seeds, hipStream_t stream) const
    {
        using key_type = Key;
        using matched_int_t =
            typename rocprim::detail::device_topk_air_helper::matched_int<key_type>::type;
        static_assert(!std::is_same<matched_int_t, void>::value, "Input type not supported");
        static_assert(sizeof(key_type) == sizeof(matched_int_t),
                      "Size of mathed_int_t is not the same as input key_type");
        rocprim::detail::segmented_topk_air_config_params params;
        HIP_CHECK(invoker::get_params(params, stream));
        const unsigned int radix_bits = params.radix_bits;
        matched_int_t      max_int    = key_type{0};
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < radix_bits; ++i)
        {
            max_int |= matched_int_t{1} << i;
        }
        key_type max_val;
        memcpy(&max_val, &max_int, sizeof(key_type));

        // Generate uniformly distributed data
        return adversarial_distribution
                   ? get_random_data<key_type>(size, key_type{0}, max_val, seeds[0])
                   : get_random_data<key_type>(size,
                                               common::generate_limits<key_type>::min(),
                                               common::generate_limits<key_type>::max(),
                                               seeds[0]);
    }

    template<class SeedsT>
    inline auto generate_offsets(size_in_type const& size, SeedsT const& seeds) const
    {
        std::default_random_engine               gen(seeds[1]);
        common::uniform_int_distribution<size_t> segment_length_dis(MinSegmentLength,
                                                                    MaxSegmentLength);

        std::vector<offsets_type> offsets;
        offsets_type              offset             = 0;
        auto                      min_segment_length = static_cast<offsets_type>(size);
        while(offset < size)
        {
            const offsets_type segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const auto end     = std::min<offsets_type>(size, offset + segment_length);
            min_segment_length = std::min<offsets_type>(min_segment_length, end - offset);

            offset += segment_length;
        }
        offsets.push_back(size);
        return std::make_tuple(offsets, min_segment_length);
    }

    // Keys benchmark
    template<typename val = Value>
    auto do_run(primbench::state& state) const
        -> std::enable_if_t<std::is_same<val, ::rocprim::empty_type>::value, void>
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        const auto seeds = primbench::seeds<2>(seed);

        using key_type = Key;

        // Calculate the number of elements
        size_in_type items = bytes / sizeof(key_type);

        // Generate uniformly distributed data
        const auto keys_input                    = generate_input(items, seeds, stream);
        const auto [offsets, min_segment_length] = generate_offsets(items, seeds);

        size_out_type k;
        if(!small_k)
        {
            k = min_segment_length / 2;
        }
        else
        {
            k = std::min<decltype(min_segment_length)>(min_segment_length, 10);
        }

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(items);

        common::device_ptr<offsets_type> d_offsets(offsets);
        const auto                       num_segments = offsets.size() - 1;

        // Get size of d_temporary_storage
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_segmented_topk_air(nullptr,
                                            temporary_storage_bytes,
                                            d_keys_input.get(),
                                            d_keys_output.get(),
                                            static_cast<Value*>(nullptr),
                                            static_cast<Value*>(nullptr),
                                            items,
                                            k,
                                            num_segments,
                                            d_offsets.get(),
                                            d_offsets.get() + 1,
                                            stream,
                                            false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        assert(temporary_storage_bytes > 0);

        state.set_items(items);
        state.add_reads<key_type>(items);

        // Run
        state.run(
            [&]
            {
                HIP_CHECK(invoke_segmented_topk_air(d_temporary_storage.get(),
                                                    temporary_storage_bytes,
                                                    d_keys_input.get(),
                                                    d_keys_output.get(),
                                                    static_cast<Value*>(nullptr),
                                                    static_cast<Value*>(nullptr),
                                                    items,
                                                    k,
                                                    num_segments,
                                                    d_offsets.get(),
                                                    d_offsets.get() + 1,
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

        const auto seeds = primbench::seeds<2>(seed);

        using key_type   = Key;
        using value_type = Value;

        // Calculate the number of elements
        size_in_type items = bytes / (sizeof(key_type) + sizeof(value_type));

        // Generate uniformly distributed data
        const auto keys_input                    = generate_input(items, seeds, stream);
        const auto [offsets, min_segment_length] = generate_offsets(items, seeds);

        size_out_type k;
        if(!small_k)
        {
            k = min_segment_length / 2;
        }
        else
        {
            k = std::min<decltype(min_segment_length)>(min_segment_length, 10);
        }

        std::vector<value_type> values_input(items);
        for(size_t i = 0; i < items; ++i)
        {
            values_input[i] = value_type(i);
        }

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(items);

        common::device_ptr<value_type> d_values_input(values_input);
        common::device_ptr<value_type> d_values_output(items);

        common::device_ptr<offsets_type> d_offsets(offsets);
        const auto                       num_segments = offsets.size() - 1;

        // Get size of d_temporary_storage
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_segmented_topk_air(nullptr,
                                            temporary_storage_bytes,
                                            d_keys_input.get(),
                                            d_keys_output.get(),
                                            d_values_input.get(),
                                            d_values_output.get(),
                                            items,
                                            k,
                                            num_segments,
                                            d_offsets.get(),
                                            d_offsets.get() + 1,
                                            stream,
                                            false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        assert(temporary_storage_bytes > 0);

        state.set_items(items);
        state.add_reads<key_type>(items);
        state.add_reads<value_type>(items);

        // Run
        state.run(
            [&]
            {
                HIP_CHECK(invoke_segmented_topk_air(d_temporary_storage.get(),
                                                    temporary_storage_bytes,
                                                    d_keys_input.get(),
                                                    d_keys_output.get(),
                                                    d_values_input.get(),
                                                    d_values_output.get(),
                                                    items,
                                                    k,
                                                    num_segments,
                                                    d_offsets.get(),
                                                    d_offsets.get() + 1,
                                                    stream,
                                                    false));
            });
    }

    void run(primbench::state& state) override
    {
        do_run(state);
    }

private:
    static hipError_t invoke_segmented_topk_air(void*         d_temporary_storage,
                                                size_t&       temp_storage_bytes,
                                                Key*          keys_input,
                                                Key*          keys_output,
                                                Value*        values_input,
                                                Value*        values_output,
                                                size_in_type  size,
                                                size_out_type k,
                                                unsigned int  segments,
                                                offsets_type* begin_offsets,
                                                offsets_type* end_offsets,
                                                hipStream_t   stream,
                                                bool          debug_synchronous)
    {
        using decomposer = std::conditional_t<common::is_custom_type<Key>::value,
                                              custom_type_decomposer<Key>,
                                              ::rocprim::identity_decomposer>;
        if constexpr(std::is_same<Value, rocprim::empty_type>::value)
        {
            (void)values_input;
            (void)values_output;

            return rocprim::detail::device_segmented_topk_air<Config>(d_temporary_storage,
                                                                      temp_storage_bytes,
                                                                      keys_input,
                                                                      keys_output,
                                                                      nullptr,
                                                                      nullptr,
                                                                      size,
                                                                      k,
                                                                      segments,
                                                                      begin_offsets,
                                                                      end_offsets,
                                                                      decomposer{},
                                                                      stream,
                                                                      debug_synchronous);
        }
        else
        {
            return rocprim::detail::device_segmented_topk_air<Config>(d_temporary_storage,
                                                                      temp_storage_bytes,
                                                                      keys_input,
                                                                      keys_output,
                                                                      values_input,
                                                                      values_output,
                                                                      size,
                                                                      k,
                                                                      segments,
                                                                      begin_offsets,
                                                                      end_offsets,
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
struct device_segmented_topk_air_benchmark_generator
{
    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        using config = rocprim::
            segmented_topk_air_config<BlockSize, ItemsPerThread, RadixBits, AdaptCoeff, Limit>;
        storage.emplace_back(
            std::make_unique<device_segmented_topk_air_benchmark<Key, Value, config>>(true, true));
        storage.emplace_back(
            std::make_unique<device_segmented_topk_air_benchmark<Key, Value, config>>(true, false));
        storage.emplace_back(
            std::make_unique<device_segmented_topk_air_benchmark<Key, Value, config>>(false, true));
        storage.emplace_back(
            std::make_unique<device_segmented_topk_air_benchmark<Key, Value, config>>(false,
                                                                                      false));
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_TOPK_AIR_TOPK_PARALLEL_HPP_
