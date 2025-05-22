// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_select.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/device/detail/device_config_helper.hpp>
#endif

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <algorithm>
    #include <memory>
#endif

enum class select_probability
{
    p005,
    p025,
    p050,
    p075,
    tuning
};

inline float get_probability(select_probability probability)
{
    switch(probability)
    {
        case select_probability::p005: return 0.05f;
        case select_probability::p025: return 0.25f;
        case select_probability::p050: return 0.50f;
        case select_probability::p075: return 0.75f;
        case select_probability::tuning: return 0.0f; // not used
    }
    return 0.0f;
}

inline const char* get_probability_name(select_probability probability)
{
    switch(probability)
    {
        case select_probability::p005: return "0.05";
        case select_probability::p025: return "0.25";
        case select_probability::p050: return "0.50";
        case select_probability::p075: return "0.75";
        case select_probability::tuning: return "tuning";
    }
    return "invalid";
}

template<typename DataType,
         typename Config                = rocprim::default_config,
         typename FlagType              = char,
         select_probability Probability = select_probability::tuning>
struct device_select_flag_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:flag,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",flag_type:" + std::string(Traits<FlagType>::name())
                                         + ",probability:" + get_probability_name(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(DataType);

        std::vector<DataType> input
            = get_random_data<DataType>(size,
                                        common::generate_limits<DataType>::min(),
                                        common::generate_limits<DataType>::max(),
                                        seed.get_0());

        std::vector<FlagType> flags_0;
        std::vector<FlagType> flags_1;
        std::vector<FlagType> flags_2;

        if(is_tuning)
        {
            flags_0 = get_random_data01<FlagType>(size, 0.0f, seed.get_1());
            flags_1 = get_random_data01<FlagType>(size, 0.5f, seed.get_1());
            flags_2 = get_random_data01<FlagType>(size, 1.0f, seed.get_1());
        }
        else
        {
            flags_0 = get_random_data01<FlagType>(size, get_probability(Probability), seed.get_1());
        }

        common::device_ptr<DataType> d_input(input);

        common::device_ptr<FlagType> d_flags_0(flags_0);
        common::device_ptr<FlagType> d_flags_1;
        common::device_ptr<FlagType> d_flags_2;
        if(is_tuning)
        {
            d_flags_1.store(flags_1);
            d_flags_2.store(flags_2);
        }

        common::device_ptr<DataType> d_output(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](FlagType* d_flags)
            {
                HIP_CHECK(rocprim::select<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input.get(),
                                                  d_flags,
                                                  d_output.get(),
                                                  d_selected_count_output.get(),
                                                  size,
                                                  stream));
            };

            dispatch_flags(d_flags_0.get());
            if(is_tuning)
            {
                dispatch_flags(d_flags_1.get());
                dispatch_flags(d_flags_2.get());
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<typename DataType,
         typename Config                = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_predicate_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:predicate,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + get_probability_name(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(DataType);

        // all data types can represent [0, 127], -1 so a predicate can select all
        std::vector<DataType> input = get_random_data<DataType>(size,
                                                                static_cast<DataType>(0),
                                                                static_cast<DataType>(126),
                                                                seed.get_0());

        common::device_ptr<DataType> d_input(input);

        common::device_ptr<DataType> d_output(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_predicate = [&](float probability)
            {
                auto predicate = [probability](const DataType& value) -> bool
                { return value < static_cast<DataType>(127 * probability); };
                HIP_CHECK(rocprim::select<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input.get(),
                                                  d_output.get(),
                                                  d_selected_count_output.get(),
                                                  size,
                                                  predicate,
                                                  stream));
            };

            if(is_tuning)
            {
                dispatch_predicate(0.0f);
                dispatch_predicate(0.5f);
                dispatch_predicate(1.0f);
            }
            else
            {
                dispatch_predicate(get_probability(Probability));
            }
        };

        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<typename DataType,
         typename FlagType              = int,
         typename Config                = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_predicated_flag_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:select,subalgo:predicated_flag,data_type:"
            + std::string(Traits<DataType>::name())
            + ",flag_type:" + std::string(Traits<FlagType>::name()) + ",probability:"
            + get_probability_name(Probability) + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(DataType);

        std::vector<DataType> input
            = get_random_data<DataType>(size,
                                        common::generate_limits<DataType>::min(),
                                        common::generate_limits<DataType>::max(),
                                        seed.get_0());

        std::vector<FlagType> flags_0;
        std::vector<FlagType> flags_1;
        std::vector<FlagType> flags_2;

        if(is_tuning)
        {
            flags_0 = get_random_data01<FlagType>(size, 0.0f, seed.get_1());
            flags_1 = get_random_data01<FlagType>(size, 0.5f, seed.get_1());
            flags_2 = get_random_data01<FlagType>(size, 1.0f, seed.get_1());
        }
        else
        {
            flags_0 = get_random_data01<FlagType>(size, get_probability(Probability), seed.get_1());
        }

        common::device_ptr<DataType> d_input(input);

        common::device_ptr<FlagType> d_flags_0(flags_0);
        common::device_ptr<FlagType> d_flags_1;
        common::device_ptr<FlagType> d_flags_2;
        if(is_tuning)
        {
            d_flags_1.store(flags_1);
            d_flags_2.store(flags_2);
        }

        common::device_ptr<DataType> d_output(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_predicated_flags = [&](FlagType* d_flags)
            {
                auto predicate = [](const FlagType& value) -> bool { return value; };
                HIP_CHECK(rocprim::select<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input.get(),
                                                  d_flags,
                                                  d_output.get(),
                                                  d_selected_count_output.get(),
                                                  size,
                                                  predicate,
                                                  stream));
            };

            dispatch_predicated_flags(d_flags_0.get());
            if(is_tuning)
            {
                dispatch_predicated_flags(d_flags_1.get());
                dispatch_predicated_flags(d_flags_2.get());
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<typename DataType>
inline std::vector<DataType> get_unique_input(size_t size, float probability, unsigned int seed)
{
    using op_type = typename std::conditional<std::is_same<DataType, rocprim::half>::value,
                                              half_plus,
                                              rocprim::plus<DataType>>::type;
    op_type               op;
    std::vector<DataType> input(size);
    auto                  input01 = get_random_data01<DataType>(size, probability, seed);
    auto                  acc     = input01[0];
    input[0]                      = acc;
    for(size_t i = 1; i < input01.size(); ++i)
    {
        input[i] = op(acc, input01[i]);
    }

    return input;
}

template<typename DataType,
         typename Config                = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_unique_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:unique,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + get_probability_name(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(DataType);

        std::vector<DataType> input_0;
        std::vector<DataType> input_1;
        std::vector<DataType> input_2;

        if(is_tuning)
        {
            input_0 = get_unique_input<DataType>(size, 0.0f, seed.get_0());
            input_1 = get_unique_input<DataType>(size, 0.5f, seed.get_0());
            input_2 = get_unique_input<DataType>(size, 1.0f, seed.get_0());
        }
        else
        {
            input_0 = get_unique_input<DataType>(size, get_probability(Probability), seed.get_0());
        }

        common::device_ptr<DataType> d_input_0(input_0);
        common::device_ptr<DataType> d_input_1;
        common::device_ptr<DataType> d_input_2;
        if(is_tuning)
        {
            d_input_1.store(input_1);
            d_input_2.store(input_2);
        }

        common::device_ptr<DataType> d_output(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](DataType* d_input)
            {
                HIP_CHECK(rocprim::unique<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input,
                                                  d_output.get(),
                                                  d_selected_count_output.get(),
                                                  size,
                                                  rocprim::equal_to<DataType>(),
                                                  stream));
            };

            dispatch_flags(d_input_0.get());
            if(is_tuning)
            {
                dispatch_flags(d_input_1.get());
                dispatch_flags(d_input_2.get());
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<typename KeyType,
         typename ValueType,
         typename Config                = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_unique_by_key_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:unique_by_key,key_type:"
                                         + std::string(Traits<KeyType>::name())
                                         + ",value_type:" + std::string(Traits<ValueType>::name())
                                         + ",probability:" + get_probability_name(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(KeyType);

        std::vector<KeyType> input_keys_0;
        std::vector<KeyType> input_keys_1;
        std::vector<KeyType> input_keys_2;

        if(is_tuning)
        {
            input_keys_0 = get_unique_input<KeyType>(size, 0.0f, seed.get_0());
            input_keys_1 = get_unique_input<KeyType>(size, 0.5f, seed.get_0());
            input_keys_2 = get_unique_input<KeyType>(size, 1.0f, seed.get_0());
        }
        else
        {
            input_keys_0
                = get_unique_input<KeyType>(size, get_probability(Probability), seed.get_0());
        }

        const auto random_range = limit_random_range<ValueType>(-1000, 1000);

        const auto input_values = get_random_data<ValueType>(size,
                                                             random_range.first,
                                                             random_range.second,
                                                             seed.get_1());

        common::device_ptr<KeyType> d_keys_input_0(input_keys_0);
        common::device_ptr<KeyType> d_keys_input_1;
        common::device_ptr<KeyType> d_keys_input_2;
        if(is_tuning)
        {
            d_keys_input_1.store(input_keys_1);
            d_keys_input_2.store(input_keys_2);
        }

        common::device_ptr<ValueType> d_values_input(input_values);

        common::device_ptr<KeyType> d_keys_output(size);

        common::device_ptr<ValueType> d_values_output(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](KeyType* d_keys_input)
            {
                HIP_CHECK(rocprim::unique_by_key<Config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_keys_input,
                                                         d_values_input.get(),
                                                         d_keys_output.get(),
                                                         d_values_output.get(),
                                                         d_selected_count_output.get(),
                                                         size,
                                                         rocprim::equal_to<KeyType>(),
                                                         stream));
            };

            dispatch_flags(d_keys_input_0.get());
            if(is_tuning)
            {
                dispatch_flags(d_keys_input_1.get());
                dispatch_flags(d_keys_input_2.get());
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(KeyType) + sizeof(ValueType));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename Config, typename KeyType, typename ValueType>
struct create_benchmark
{
    static constexpr unsigned int block_size           = Config().kernel_config.block_size;
    static constexpr unsigned int items_per_thread     = Config().kernel_config.items_per_thread;
    static constexpr unsigned int max_shared_memory    = TUNING_SHARED_MEMORY_MAX;
    static constexpr unsigned int max_size_per_element = sizeof(KeyType) + sizeof(ValueType);
    static constexpr unsigned int max_items_per_thread
        = max_shared_memory / (block_size * max_size_per_element);

    void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        storage.emplace_back(
            std::make_unique<device_select_unique_by_key_benchmark<KeyType, ValueType, Config>>());

        if(items_per_thread <= max_items_per_thread)
        {
            storage.emplace_back(
                std::make_unique<
                    device_select_predicated_flag_benchmark<KeyType, ValueType, Config>>());
        }
    }
};

template<typename Config, typename KeyType>
struct create_benchmark<Config, KeyType, rocprim::empty_type>
{
    void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        storage.emplace_back(std::make_unique<device_select_flag_benchmark<KeyType, Config>>());
        storage.emplace_back(
            std::make_unique<device_select_predicate_benchmark<KeyType, Config>>());
        storage.emplace_back(std::make_unique<device_select_unique_benchmark<KeyType, Config>>());
    }
};

template<typename KeyType, typename ValueType, int BlockSize>
struct device_select_benchmark_generator
{
    template<int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            using config = rocprim::select_config<BlockSize, ItemsPerThread>;
            create_benchmark<config, KeyType, ValueType>{}(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        static constexpr int max_items_per_thread
            = std::min(64 / std::max(sizeof(KeyType), sizeof(ValueType)), size_t{32});
        static_for_each<make_index_range<int, 4, max_items_per_thread>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_
