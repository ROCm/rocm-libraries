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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_search_n.hpp>
#include <rocprim/functional.hpp>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/types.hpp>
#endif

#include <cstddef>
#include <string>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <memory>
#else
    #include <functional>
    #include <stdint.h>
    #include <type_traits>
#endif

template<typename First, typename... Types>
struct type_arr
{
    using type = First;
    using next = type_arr<Types...>;
};

template<typename First>
struct type_arr<First>
{
    using type = First;
};

template<typename...>
using void_type = void;

template<typename T, typename = void>
constexpr bool is_type_arr_end = true;

template<typename T>
constexpr bool is_type_arr_end<T, void_type<typename T::next>> = false;

template<size_t Value>
struct count_equal_to
{
    std::string name() const
    {
        return "count_equal_to<" + std::to_string(Value) + ">";
    }
    constexpr size_t resolve(size_t) const
    {
        return Value;
    }
};

template<size_t Value>
struct count_is_percent_of_size
{
    std::string name() const
    {
        return "count_is_percent_of_size<" + std::to_string(Value) + ">";
    }
    constexpr size_t resolve(size_t items) const
    {
        return items * Value / 100;
    }
};

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
            .add("ipt", config.kernel_config.items_per_thread)
            .add("threshold", config.threshold);
    }
}

template<class InputType,
         class OutputType,
         class CountCalculator,
         class Config = rocprim::default_config>
struct device_search_n_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_search_n")
            .add("data_type", primbench::name<InputType>())
            .add("count_calculator", CountCalculator{}.name())
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        const auto& stream    = state.stream;
        const auto& size_byte = state.size;

        InputType                      h_noise{0};
        InputType                      h_value{1};
        common::device_ptr<void>       d_temp_storage;
        size_t                         temp_storage_size = 0;
        size_t                         count;
        std::vector<InputType>         input{};
        common::device_ptr<InputType>  d_input;
        common::device_ptr<OutputType> d_output(1);
        common::device_ptr<InputType>  d_value(std::vector<decltype(h_value)>{h_value}, stream);

        size_t items = size_byte / sizeof(InputType);

        count            = CountCalculator{}.resolve(items);
        size_t cur_tile  = 0;
        size_t last_tile = items / count - 1;
        input            = std::vector<InputType>(items, h_value);
        while(cur_tile != last_tile)
        {
            input[cur_tile * count + count - 1] = h_noise;
            ++cur_tile;
        }

        d_input.store_async(input, stream);

        auto launch_search_n = [&]()
        {
            HIP_CHECK(::rocprim::search_n<Config>(d_temp_storage.get(),
                                                  temp_storage_size,
                                                  d_input.get(),
                                                  d_output.get(),
                                                  items,
                                                  count,
                                                  d_value.get(),
                                                  rocprim::equal_to<InputType>{},
                                                  stream,
                                                  false));
        };

        // allocate temp memory
        launch_search_n();
        d_temp_storage.resize_async(temp_storage_size, stream);

        state.set_items(items);
        state.add_reads<InputType>(items);

        state.run([&] { launch_search_n(); });
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, size_t Threshold>
struct device_search_n_benchmark_generator
{
    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        using config = rocprim::search_n_config<BlockSize, ItemsPerThread, Threshold>;
        storage.emplace_back(
            std::make_unique<device_search_n_benchmark<T, size_t, count_equal_to<1>, config>>());
        storage.emplace_back(
            std::make_unique<device_search_n_benchmark<T, size_t, count_equal_to<6>, config>>());
        storage.emplace_back(
            std::make_unique<device_search_n_benchmark<T, size_t, count_equal_to<10>, config>>());
        storage.emplace_back(
            std::make_unique<device_search_n_benchmark<T, size_t, count_equal_to<14>, config>>());
        storage.emplace_back(
            std::make_unique<device_search_n_benchmark<T, size_t, count_equal_to<25>, config>>());
        storage.emplace_back(
            std::make_unique<
                device_search_n_benchmark<T, size_t, count_is_percent_of_size<50>, config>>());
        storage.emplace_back(
            std::make_unique<
                device_search_n_benchmark<T, size_t, count_is_percent_of_size<100>, config>>());
    }
};

#endif // BENCHMARK_CONFIG_TUNING
