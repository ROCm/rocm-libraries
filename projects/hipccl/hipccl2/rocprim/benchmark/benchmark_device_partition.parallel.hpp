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

#ifndef ROCPRIM_BENCHMARK_DEVICE_PARTITION_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_PARTITION_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_partition.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/device/detail/device_config_helper.hpp>
#endif // BENCHMARK_CONFIG_TUNING

#include <array>
#include <string>
#include <utility>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <algorithm>
    #include <memory>
#endif // BENCHMARK_CONFIG_TUNING

enum class partition_probability
{
    p005,
    p025,
    p050,
    p075,
    tuning
};

inline float get_probability(partition_probability probability)
{
    switch(probability)
    {
        case partition_probability::p005: return 0.05f;
        case partition_probability::p025: return 0.25f;
        case partition_probability::p050: return 0.50f;
        case partition_probability::p075: return 0.75f;
        case partition_probability::tuning: return 0.0f; // not used
    }
    return 0.0f;
}

inline const char* get_probability_name(partition_probability probability)
{
    switch(probability)
    {
        case partition_probability::p005: return "0.05";
        case partition_probability::p025: return "0.25";
        case partition_probability::p050: return "0.50";
        case partition_probability::p075: return "0.75";
        case partition_probability::tuning: return "tuning";
    }
    return "invalid";
}

enum class partition_three_way_probability
{
    p005_p025,
    p025_p050,
    p050_p075,
    p075_p100,
    tuning
};

inline std::pair<float, float> get_probability(partition_three_way_probability probability)
{
    switch(probability)
    {
        case partition_three_way_probability::p005_p025: return std::make_pair(0.05f, 0.25f);
        case partition_three_way_probability::p025_p050: return std::make_pair(0.25f, 0.50f);
        case partition_three_way_probability::p050_p075: return std::make_pair(0.50f, 0.75f);
        case partition_three_way_probability::p075_p100: return std::make_pair(0.75f, 1.00f);
        case partition_three_way_probability::tuning:
            return std::make_pair(0.00f, 0.00f); // not used
    }
    return std::make_pair(0.00f, 0.00f);
}

inline const char* get_probability_name(partition_three_way_probability probability)
{
    switch(probability)
    {
        case partition_three_way_probability::p005_p025: return "0.05:0.25";
        case partition_three_way_probability::p025_p050: return "0.25:0.50";
        case partition_three_way_probability::p050_p075: return "0.50:0.75";
        case partition_three_way_probability::p075_p100: return "0.75:1.00";
        case partition_three_way_probability::tuning: return "tuning";
    }
    return "invalid";
}

template<typename DataType,
         typename Config                   = rocprim::default_config,
         typename FlagType                 = char,
         partition_probability Probability = partition_probability::tuning>
struct device_partition_flag_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:partition,subalgo:flag,data_type:"
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
                HIP_CHECK(rocprim::partition<Config>(d_temp_storage,
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

    static constexpr bool is_tuning = Probability == partition_probability::tuning;
};

template<typename DataType,
         typename Config                   = rocprim::default_config,
         partition_probability Probability = partition_probability::tuning>
struct device_partition_predicate_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:partition,subalgo:predicate,data_type:"
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
                HIP_CHECK(rocprim::partition<Config>(d_temp_storage,
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

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == partition_probability::tuning;
};

template<typename DataType,
         typename Config                   = rocprim::default_config,
         typename FlagType                 = char,
         partition_probability Probability = partition_probability::tuning>
struct device_partition_two_way_flag_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:partition_two_way,subalgo:flag,data_type:"
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

        common::device_ptr<DataType> d_output_selected(size);

        common::device_ptr<DataType> d_output_rejected(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](FlagType* d_flags)
            {
                HIP_CHECK(rocprim::partition_two_way<Config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             d_input.get(),
                                                             d_flags,
                                                             d_output_selected.get(),
                                                             d_output_rejected.get(),
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
        size_t temp_storage_size_bytes = 0;
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == partition_probability::tuning;
};

template<typename DataType,
         typename Config                   = rocprim::default_config,
         partition_probability Probability = partition_probability::tuning>
struct device_partition_two_way_predicate_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:partition_two_way,subalgo:predicate,data_type:"
            + std::string(Traits<DataType>::name()) + ",probability:"
            + get_probability_name(Probability) + ",cfg:" + partition_config_name<Config>() + "}");
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

        common::device_ptr<DataType> d_output_selected(size);

        common::device_ptr<DataType> d_output_rejected(size);

        common::device_ptr<unsigned int> d_selected_count_output(1);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_predicate = [&](float probability)
            {
                auto predicate = [probability](const DataType& value) -> bool
                { return value < static_cast<DataType>(127 * probability); };
                HIP_CHECK(rocprim::partition_two_way<Config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             d_input.get(),
                                                             d_output_selected.get(),
                                                             d_output_rejected.get(),
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

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == partition_probability::tuning;
};

template<typename DataType,
         typename Config                             = rocprim::default_config,
         partition_three_way_probability Probability = partition_three_way_probability::tuning>
struct device_partition_three_way_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:partition_three_way,data_type:"
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

        common::device_ptr<DataType> d_output_first(size);

        common::device_ptr<DataType> d_output_second(size);

        common::device_ptr<DataType> d_output_unselected(size);

        common::device_ptr<unsigned int> d_selected_count_output(2);

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_predicate = [&](std::pair<float, float> probability)
            {
                const float probability_one = probability.first;
                auto        predicate_one   = [probability_one](const DataType& value)
                { return value < DataType(127 * probability_one); };
                const float probability_two = probability.second;
                auto        predicate_two   = [probability_two](const DataType& value)
                { return value < DataType(127 * probability_two); };

                HIP_CHECK(rocprim::partition_three_way<Config>(d_temp_storage,
                                                               temp_storage_size_bytes,
                                                               d_input.get(),
                                                               d_output_first.get(),
                                                               d_output_second.get(),
                                                               d_output_unselected.get(),
                                                               d_selected_count_output.get(),
                                                               size,
                                                               predicate_one,
                                                               predicate_two,
                                                               stream));
            };

            if(is_tuning)
            {
                // clang-format off
                std::array<std::pair<float, float>, 7> probabilities = {{
                    {0.33f, 0.66f},   // 1st, 2nd, and 3rd bin
                    {0.50f, 1.00f},   // 1st and 2nd bin
                    {0.00f, 0.50f},   // 2nd and 3rd bin
                    {0.50f, 0.50f},   // 1st and 3rd bin
                    {1.00f, 1.00f},   // 1st bin
                    {0.00f, 1.00f},   // 2nd bin
                    {0.00f, 0.00f}}}; // 3rd bin
                // clang-format on

                for(const std::pair<float, float>& probability : probabilities)
                {
                    dispatch_predicate(probability);
                }
            }
            else
            {
                dispatch_predicate(get_probability(Probability));
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.run([&] { dispatch(d_temp_storage.get(), temp_storage_size_bytes); });

        state.set_throughput(size, sizeof(DataType));
    }

    static constexpr bool is_tuning = Probability == partition_three_way_probability::tuning;
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename DataType, int BlockSize>
struct device_partition_benchmark_generator
{
    template<int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            using config = rocprim::select_config<BlockSize, ItemsPerThread>;
            storage.emplace_back(
                std::make_unique<device_partition_flag_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_predicate_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_two_way_flag_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_two_way_predicate_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_three_way_benchmark<DataType, config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        static constexpr int max_items_per_thread = std::min(64 / sizeof(DataType), size_t{32});
        static_for_each<make_index_range<int, 4, max_items_per_thread>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_PARTITION_PARALLEL_HPP_
