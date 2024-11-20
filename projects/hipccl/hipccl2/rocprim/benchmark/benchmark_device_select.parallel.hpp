// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <rocprim/rocprim.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

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

constexpr int warmup_iter = 5;
constexpr int batch_size  = 10;

template<class DataType,
         class Config                   = rocprim::default_config,
         class FlagType                 = char,
         select_probability Probability = select_probability::tuning>
struct device_select_flag_benchmark : public config_autotune_interface
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

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        // Calculate the number of elements 
        size_t size = bytes / sizeof(DataType);

        std::vector<DataType> input = get_random_data<DataType>(size,
                                                                generate_limits<DataType>::min(),
                                                                generate_limits<DataType>::max(),
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

        DataType* d_input{};
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(*d_input), hipMemcpyHostToDevice));

        FlagType* d_flags_0{};
        FlagType* d_flags_1{};
        FlagType* d_flags_2{};
        HIP_CHECK(hipMalloc(&d_flags_0, size * sizeof(*d_flags_0)));
        HIP_CHECK(
            hipMemcpy(d_flags_0, flags_0.data(), size * sizeof(*d_flags_0), hipMemcpyHostToDevice));
        if(is_tuning)
        {
            HIP_CHECK(hipMalloc(&d_flags_1, size * sizeof(*d_flags_1)));
            HIP_CHECK(hipMemcpy(d_flags_1,
                                flags_1.data(),
                                size * sizeof(*d_flags_1),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMalloc(&d_flags_2, size * sizeof(*d_flags_2)));
            HIP_CHECK(hipMemcpy(d_flags_2,
                                flags_2.data(),
                                size * sizeof(*d_flags_2),
                                hipMemcpyHostToDevice));
        }

        DataType* d_output{};
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(*d_output)));

        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](FlagType* d_flags)
            {
                HIP_CHECK(rocprim::select<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input,
                                                  d_flags,
                                                  d_output,
                                                  d_selected_count_output,
                                                  size,
                                                  stream));
            };

            dispatch_flags(d_flags_0);
            if(is_tuning)
            {
                dispatch_flags(d_flags_1);
                dispatch_flags(d_flags_2);
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; i++)
        {
            dispatch(d_temp_storage, temp_storage_size_bytes);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                dispatch(d_temp_storage, temp_storage_size_bytes);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        if(is_tuning)
        {
            HIP_CHECK(hipFree(d_flags_2));
            HIP_CHECK(hipFree(d_flags_1));
        }
        HIP_CHECK(hipFree(d_flags_0));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<class DataType,
         class Config                   = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_predicate_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:predicate,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + get_probability_name(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        // Calculate the number of elements 
        size_t size = bytes / sizeof(DataType);

        // all data types can represent [0, 127], -1 so a predicate can select all
        std::vector<DataType> input = get_random_data<DataType>(size,
                                                                static_cast<DataType>(0),
                                                                static_cast<DataType>(126),
                                                                seed.get_0());

        DataType* d_input;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(*d_input), hipMemcpyHostToDevice));

        DataType* d_output;
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(*d_output)));

        unsigned int* d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_predicate = [&](float probability)
            {
                auto predicate = [probability](const DataType& value) -> bool
                { return value < static_cast<DataType>(127 * probability); };
                HIP_CHECK(rocprim::select<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input,
                                                  d_output,
                                                  d_selected_count_output,
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
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            dispatch(d_temp_storage, temp_storage_size_bytes);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                dispatch(d_temp_storage, temp_storage_size_bytes);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<class DataType,
         class FlagType                 = int,
         class Config                   = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_predicated_flag_benchmark : public config_autotune_interface
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

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        // Calculate the number of elements
        size_t size = bytes / sizeof(DataType);

        std::vector<DataType> input = get_random_data<DataType>(size,
                                                                generate_limits<DataType>::min(),
                                                                generate_limits<DataType>::max(),
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

        DataType* d_input{};
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(*d_input), hipMemcpyHostToDevice));

        FlagType* d_flags_0{};
        FlagType* d_flags_1{};
        FlagType* d_flags_2{};
        HIP_CHECK(hipMalloc(&d_flags_0, size * sizeof(*d_flags_0)));
        HIP_CHECK(
            hipMemcpy(d_flags_0, flags_0.data(), size * sizeof(*d_flags_0), hipMemcpyHostToDevice));
        if(is_tuning)
        {
            HIP_CHECK(hipMalloc(&d_flags_1, size * sizeof(*d_flags_1)));
            HIP_CHECK(hipMemcpy(d_flags_1,
                                flags_1.data(),
                                size * sizeof(*d_flags_1),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMalloc(&d_flags_2, size * sizeof(*d_flags_2)));
            HIP_CHECK(hipMemcpy(d_flags_2,
                                flags_2.data(),
                                size * sizeof(*d_flags_2),
                                hipMemcpyHostToDevice));
        }

        DataType* d_output{};
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(*d_output)));

        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_predicated_flags = [&](FlagType* d_flags)
            {
                auto predicate = [](const FlagType& value) -> bool { return value; };
                HIP_CHECK(rocprim::select<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input,
                                                  d_flags,
                                                  d_output,
                                                  d_selected_count_output,
                                                  size,
                                                  predicate,
                                                  stream));
            };

            dispatch_predicated_flags(d_flags_0);
            if(is_tuning)
            {
                dispatch_predicated_flags(d_flags_1);
                dispatch_predicated_flags(d_flags_2);
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; i++)
        {
            dispatch(d_temp_storage, temp_storage_size_bytes);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                dispatch(d_temp_storage, temp_storage_size_bytes);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        if(is_tuning)
        {
            HIP_CHECK(hipFree(d_flags_2));
            HIP_CHECK(hipFree(d_flags_1));
        }
        HIP_CHECK(hipFree(d_flags_0));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
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
    for(size_t i = 1; i < input01.size(); i++)
    {
        input[i] = op(acc, input01[i]);
    }

    return input;
}

template<class DataType,
         class Config                   = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_unique_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:unique,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + get_probability_name(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
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

        DataType* d_input_0{};
        DataType* d_input_1{};
        DataType* d_input_2{};
        HIP_CHECK(hipMalloc(&d_input_0, size * sizeof(*d_input_0)));
        HIP_CHECK(
            hipMemcpy(d_input_0, input_0.data(), size * sizeof(*d_input_0), hipMemcpyHostToDevice));
        if(is_tuning)
        {
            HIP_CHECK(hipMalloc(&d_input_1, size * sizeof(*d_input_1)));
            HIP_CHECK(hipMemcpy(d_input_1,
                                input_1.data(),
                                size * sizeof(*d_input_1),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMalloc(&d_input_2, size * sizeof(*d_input_2)));
            HIP_CHECK(hipMemcpy(d_input_2,
                                input_2.data(),
                                size * sizeof(*d_input_2),
                                hipMemcpyHostToDevice));
        }

        DataType* d_output{};
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(*d_output)));

        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](DataType* d_input)
            {
                HIP_CHECK(rocprim::unique<Config>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_input,
                                                  d_output,
                                                  d_selected_count_output,
                                                  size,
                                                  rocprim::equal_to<DataType>(),
                                                  stream));
            };

            dispatch_flags(d_input_0);
            if(is_tuning)
            {
                dispatch_flags(d_input_1);
                dispatch_flags(d_input_2);
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            dispatch(d_temp_storage, temp_storage_size_bytes);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                dispatch(d_temp_storage, temp_storage_size_bytes);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        if(is_tuning)
        {
            HIP_CHECK(hipFree(d_input_2));
            HIP_CHECK(hipFree(d_input_1));
        }
        HIP_CHECK(hipFree(d_input_0));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    static constexpr bool is_tuning = Probability == select_probability::tuning;
};

template<class KeyType,
         class ValueType,
         class Config                   = rocprim::default_config,
         select_probability Probability = select_probability::tuning>
struct device_select_unique_by_key_benchmark : public config_autotune_interface
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

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
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

        KeyType* d_keys_input_0{};
        KeyType* d_keys_input_1{};
        KeyType* d_keys_input_2{};
        HIP_CHECK(hipMalloc(&d_keys_input_0, size * sizeof(*d_keys_input_0)));
        HIP_CHECK(hipMemcpy(d_keys_input_0,
                            input_keys_0.data(),
                            size * sizeof(*d_keys_input_0),
                            hipMemcpyHostToDevice));
        if(is_tuning)
        {
            HIP_CHECK(hipMalloc(&d_keys_input_1, size * sizeof(*d_keys_input_1)));
            HIP_CHECK(hipMemcpy(d_keys_input_1,
                                input_keys_1.data(),
                                size * sizeof(*d_keys_input_1),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMalloc(&d_keys_input_2, size * sizeof(*d_keys_input_2)));
            HIP_CHECK(hipMemcpy(d_keys_input_2,
                                input_keys_2.data(),
                                size * sizeof(*d_keys_input_2),
                                hipMemcpyHostToDevice));
        }

        ValueType* d_values_input{};
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(*d_values_input)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            input_values.data(),
                            size * sizeof(*d_values_input),
                            hipMemcpyHostToDevice));

        KeyType* d_keys_output{};
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(*d_keys_output)));

        ValueType* d_values_output{};
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(*d_values_output)));

        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));

        const auto dispatch = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
        {
            const auto dispatch_flags = [&](KeyType* d_keys_input)
            {
                HIP_CHECK(rocprim::unique_by_key<Config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_keys_input,
                                                         d_values_input,
                                                         d_keys_output,
                                                         d_values_output,
                                                         d_selected_count_output,
                                                         size,
                                                         rocprim::equal_to<KeyType>(),
                                                         stream));
            };

            dispatch_flags(d_keys_input_0);
            if(is_tuning)
            {
                dispatch_flags(d_keys_input_1);
                dispatch_flags(d_keys_input_2);
            }
        };

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        dispatch(nullptr, temp_storage_size_bytes);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            dispatch(d_temp_storage, temp_storage_size_bytes);
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                dispatch(d_temp_storage, temp_storage_size_bytes);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(KeyType) + sizeof(ValueType)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        if(is_tuning)
        {
            HIP_CHECK(hipFree(d_keys_input_2));
            HIP_CHECK(hipFree(d_keys_input_1));
        }
        HIP_CHECK(hipFree(d_keys_input_0));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
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

    void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
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
    void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
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
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            using config = rocprim::select_config<BlockSize, ItemsPerThread>;
            create_benchmark<config, KeyType, ValueType>{}(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr int max_items_per_thread
            = std::min(64 / std::max(sizeof(KeyType), sizeof(ValueType)), size_t{32});
        static_for_each<make_index_range<int, 4, max_items_per_thread>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_
