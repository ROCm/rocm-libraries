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

#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>
#include <rocprim/functional.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/block/block_load.hpp>
    #include <rocprim/block/block_scan.hpp>
#endif

#include <array>
#include <cstddef>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <algorithm>
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
         typename ValueType,
         int  MaxSegmentLength,
         bool Deterministic,
         typename Config = rocprim::default_config>
struct device_reduce_by_key_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        auto algo = []
        {
            if constexpr(Deterministic)
            {
                return "device_reduce_by_key_deterministic";
            }
            else
            {
                return "device_reduce_by_key";
            }
        };

        return primbench::json{}
            .add("lvl", "device")
            .add("algo", algo())
            .add("key_type", primbench::name<KeyType>())
            .add("value_type", primbench::name<ValueType>())
            .add("max_segment_length", MaxSegmentLength)
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        constexpr std::array<int, 2> tuning_max_segment_lengths = {10, 1000};
        constexpr int    num_input_arrays = is_tuning ? tuning_max_segment_lengths.size() : 1;
        constexpr size_t item_size        = sizeof(KeyType) + sizeof(ValueType);

        const size_t items = bytes / item_size;

        std::vector<KeyType> key_inputs[num_input_arrays];
        if(is_tuning)
        {
            for(size_t i = 0; i < tuning_max_segment_lengths.size(); ++i)
            {
                key_inputs[i]
                    = get_random_segments_iota<KeyType>(items, tuning_max_segment_lengths[i], seed);
            }
        }
        else
        {
            key_inputs[0] = get_random_segments_iota<KeyType>(items, MaxSegmentLength, seed);
        }

        std::vector<ValueType> value_input(items);
        std::iota(value_input.begin(), value_input.end(), 0);

        common::device_ptr<KeyType> d_key_inputs[num_input_arrays];
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_key_inputs[i].store(key_inputs[i]);
        }

        common::device_ptr<ValueType> d_value_input(value_input);

        common::device_ptr<KeyType>      d_unique_output(items);
        common::device_ptr<ValueType>    d_aggregates_output(items);
        common::device_ptr<unsigned int> d_unique_count_output(1);

        rocprim::plus<ValueType>   reduce_op;
        rocprim::equal_to<KeyType> key_compare_op;

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_input = [&](KeyType* d_key_input)
            {
                if constexpr(!Deterministic)
                {
                    HIP_CHECK(rocprim::reduce_by_key<Config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             d_key_input,
                                                             d_value_input.get(),
                                                             items,
                                                             d_unique_output.get(),
                                                             d_aggregates_output.get(),
                                                             d_unique_count_output.get(),
                                                             reduce_op,
                                                             key_compare_op,
                                                             stream));
                }
                else
                {
                    HIP_CHECK(
                        rocprim::deterministic_reduce_by_key<Config>(d_temp_storage,
                                                                     temp_storage_size_bytes,
                                                                     d_key_input,
                                                                     d_value_input.get(),
                                                                     items,
                                                                     d_unique_output.get(),
                                                                     d_aggregates_output.get(),
                                                                     d_unique_count_output.get(),
                                                                     reduce_op,
                                                                     key_compare_op,
                                                                     stream));
                }
            };

            // One tuning iteration runs multiple inputs with different distributions,
            //   preventing overfitting the config to a specific data distrubution.
            //   Note that this does not weigh the inputs/distributions equally as
            //   generally larger segments perform better.
            for(int i = 0; i < num_input_arrays; ++i)
            {
                dispatch_input(d_key_inputs[i].get());
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.set_items(items);
        state.add_reads<KeyType>(items);
        state.add_reads<ValueType>(items);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });
    }

    static constexpr bool is_tuning = !std::is_same<Config, rocprim::default_config>::value;
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename KeyType, typename ValueType, unsigned int BlockSize>
struct device_reduce_by_key_benchmark_generator
{
    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            using config
                = rocprim::reduce_by_key_config<BlockSize,
                                                ItemsPerThread,
                                                rocprim::block_load_method::block_load_transpose,
                                                rocprim::block_load_method::block_load_transpose,
                                                rocprim::block_scan_algorithm::using_warp_scan>;
            // max segment length argument is irrelevant, tuning overrides segment length
            storage.emplace_back(
                std::make_unique<
                    device_reduce_by_key_benchmark<KeyType, ValueType, 0, false, config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        static constexpr unsigned int max_items_per_thread = std::min(
            TUNING_SHARED_MEMORY_MAX / std::max(sizeof(KeyType), sizeof(ValueType)) / BlockSize - 1,
            size_t{15});
        static_for_each<make_index_range<unsigned int, 4u, max_items_per_thread>, create_ipt>(
            storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#define CREATE_BENCHMARK(KEY, VALUE, MAX_SEGMENT_LENGTH) \
    executor.queue<device_reduce_by_key_benchmark<KEY, VALUE, MAX_SEGMENT_LENGTH, Deterministic>>();

#define CREATE_BENCHMARK_TYPE(KEY, VALUE) \
    CREATE_BENCHMARK(KEY, VALUE, 10)      \
    CREATE_BENCHMARK(KEY, VALUE, 1000)

// Some of the tuned types
#define CREATE_BENCHMARK_TYPES(KEY)                \
    CREATE_BENCHMARK_TYPE(KEY, int8_t)             \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::half)      \
    CREATE_BENCHMARK_TYPE(KEY, int32_t)            \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::int128_t)  \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::uint128_t) \
    CREATE_BENCHMARK_TYPE(KEY, float)              \
    CREATE_BENCHMARK_TYPE(KEY, double)

// All of the tuned types
#define CREATE_BENCHMARK_TYPE_TUNING(KEY)         \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::int128_t) \
    CREATE_BENCHMARK_TYPE(KEY, int64_t)           \
    CREATE_BENCHMARK_TYPE(KEY, int)               \
    CREATE_BENCHMARK_TYPE(KEY, short)             \
    CREATE_BENCHMARK_TYPE(KEY, int8_t)            \
    CREATE_BENCHMARK_TYPE(KEY, double)            \
    CREATE_BENCHMARK_TYPE(KEY, float)             \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::half)

template<bool Deterministic>
void add_benchmarks(primbench::executor& executor)
{
    // Tuned types
    CREATE_BENCHMARK_TYPE_TUNING(rocprim::int128_t)
    CREATE_BENCHMARK_TYPE_TUNING(int64_t)
    CREATE_BENCHMARK_TYPE_TUNING(int)
    CREATE_BENCHMARK_TYPE_TUNING(short)
    CREATE_BENCHMARK_TYPE_TUNING(int8_t)

#ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_BENCHMARK_TYPE(rocprim::int128_t, rocprim::uint128_t)
    CREATE_BENCHMARK_TYPE(int, rocprim::uint128_t)
    CREATE_BENCHMARK_TYPE(short, rocprim::uint128_t)
    CREATE_BENCHMARK_TYPE(int8_t, rocprim::uint128_t)
    CREATE_BENCHMARK_TYPES(double)
    CREATE_BENCHMARK_TYPES(float)
    CREATE_BENCHMARK_TYPES(rocprim::half)
    CREATE_BENCHMARK_TYPES(rocprim::uint128_t)

    // Not tuned custom types
    using custom_f32_f32 = common::custom_type<float, float>;
    using custom_f64_f64 = common::custom_type<double, double>;

    CREATE_BENCHMARK_TYPE(int, custom_f32_f32)
    CREATE_BENCHMARK_TYPE(int, custom_f64_f64)

    CREATE_BENCHMARK_TYPE(int64_t, custom_f32_f32)
    CREATE_BENCHMARK_TYPE(int64_t, custom_f64_f64)
#endif
}
